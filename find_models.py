import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CrystalRNNNet, get_sqw


COUNT_STEPS = 2000
DELTA = 10000
SEQUENCE_LENGTH = 3
STEP = 10
DT = 0.02
HIDDEN_SIZE = 100
DATA_LEN = 0.2
BATCH_SIZE = 200
NUM_LAYERS = 3
COUNT_RUN = 3
SAVE_THRESHOLD = 2.5
LATTICE_PARAMETER = 3.615
NCELLS = 5
KCOUNT = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Train and select RNN crystal predictors.")
    parser.add_argument("count_models", type=int, help="Number of models to train.")
    parser.add_argument("rnn_type", help="Recurrent block type: RNN, GRU, or LSTM.")
    parser.add_argument(
        "--coords-path",
        default="coords_lmp.raw",
        help="Path to the coordinate trajectory file.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where selected models will be saved.",
    )
    return parser.parse_args()


def load_coordinates(path):
    print("IMPORT DATA")
    with open(path) as f:
        lines = f.readlines()
    coords = np.array([[float(value) for value in line.split()] for line in lines], dtype=np.float32)
    print(coords.shape)
    return coords


def build_default_k_vectors():
    kmin = 2 * np.pi / (NCELLS * LATTICE_PARAMETER)
    kmax = NCELLS * kmin
    kmas = np.zeros((KCOUNT, 3), dtype=np.float32)
    kmas[:, 0] = np.linspace(kmin, kmax, KCOUNT)
    return kmas


def get_sqw_default(coords, dt, step):
    return get_sqw(coords, dt=dt, step=step, kmas=build_default_k_vectors())


def sample_train_data(dcoords_global, delta, sequence_length):
    if dcoords_global.shape[0] <= delta:
        start = 0
    else:
        start = np.random.randint(low=0, high=dcoords_global.shape[0] - delta)

    dcoords = dcoords_global[start : start + delta]
    x_coords = np.array(
        [dcoords[i : i + sequence_length].copy() for i in range(dcoords.shape[0] - sequence_length)]
    )
    y_coords = np.array([dcoords[i + sequence_length].copy() for i in range(dcoords.shape[0] - sequence_length)])
    return dcoords, x_coords, y_coords


def sample_initial_sequence(dcoords_global, sequence_length, count_steps):
    if dcoords_global.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = np.random.randint(low=sequence_length, high=dcoords_global.shape[0] - count_steps)
    return np.array([dcoords_global[begin_index - sequence_length : begin_index]], dtype=np.float32)


def evaluate_model(
    model,
    coords_reference,
    dcoords_global,
    dc,
    count_steps,
    count_run,
    dt,
    step,
    sequence_length,
):
    xi_ref, yi_ref, jlp_ref = get_sqw_default(coords_reference, dt, step)
    jlp_mean = np.zeros_like(jlp_ref)
    norm = 0.0
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_sequence(dcoords_global, sequence_length, count_steps)
        predicted_coords = model.run(count_steps, init)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords + dc, dt, step)
        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)

    norm /= count_run
    jlp_mean /= count_run
    return norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def plot_sqw(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp):
    plt.pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues")
    plt.show()
    plt.pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues")
    plt.show()


def save_model(model, models_dir, norm, rnn_type, data_len, count_steps):
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    filepath = models_path / f"mean_norm_{norm}_{rnn_type.lower()}_dcoords_{int(data_len * count_steps)}.pth"
    torch.save(model, filepath)
    print(f"==============> SAVE MODEL TO FILE - {filepath}")


def main():
    args = parse_args()
    coords_list = load_coordinates(args.coords_path)
    dc = coords_list.mean(axis=0)
    dcoords_global = coords_list - dc

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        coords, x_coords, y_coords = sample_train_data(
            dcoords_global=dcoords_global,
            delta=DELTA,
            sequence_length=SEQUENCE_LENGTH,
        )

        predictor = CrystalRNNNet(
            in_features=x_coords.shape[-1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            type=args.rnn_type.upper(),
        )
        predictor.batch_size = BATCH_SIZE
        predictor.train(x_coords, y_coords, data_len=DATA_LEN)

        norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=predictor,
            coords_reference=coords + dc,
            dcoords_global=dcoords_global,
            dc=dc,
            count_steps=COUNT_STEPS,
            count_run=COUNT_RUN,
            dt=DT,
            step=STEP,
            sequence_length=SEQUENCE_LENGTH,
        )
        print("CURRENT_NORM =", norm)

        if norm < SAVE_THRESHOLD:
            plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
            save_model(predictor, args.models_dir, norm, args.rnn_type, DATA_LEN, COUNT_STEPS)

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
