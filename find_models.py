import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CrystalRNNNet, DEFAULT_FLATTEN_ORDER, get_sqw


COUNT_STEPS = 2000
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
    parser = argparse.ArgumentParser(description="Train and select crystal-aware RNN predictors.")
    parser.add_argument("count_models", type=int, help="Number of models to train.")
    parser.add_argument("rnn_type", help="Recurrent block type: RNN, GRU, or LSTM.")
    parser.add_argument(
        "--data-path",
        default="crystal_training_data.npz",
        help="Path to an .npz file from prepare_crystal_data.py.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where selected models will be saved.",
    )
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic wrapping during crystal inference.",
    )
    return parser.parse_args()


def load_training_data(path):
    print("IMPORT DATA")
    data = np.load(path)
    required = [
        "X_blocks",
        "y_blocks",
        "displacements",
        "atom_order",
        "reference_positions",
        "crystal_shape",
        "train_supercell_shape",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")

    loaded = {key: data[key] for key in data.files}
    print("X_blocks", loaded["X_blocks"].shape)
    print("y_blocks", loaded["y_blocks"].shape)
    print("displacements", loaded["displacements"].shape)
    return loaded


def build_default_k_vectors():
    kmin = 2 * np.pi / (NCELLS * LATTICE_PARAMETER)
    kmax = NCELLS * kmin
    kmas = np.zeros((KCOUNT, 3), dtype=np.float32)
    kmas[:, 0] = np.linspace(kmin, kmax, KCOUNT)
    return kmas


def get_sqw_default(coords, dt, step):
    return get_sqw(coords, dt=dt, step=step, kmas=build_default_k_vectors())


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def sample_initial_crystal_sequence(displacements, sequence_length, count_steps):
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = np.random.randint(low=sequence_length, high=displacements.shape[0] - count_steps)
    return displacements[begin_index - sequence_length : begin_index].astype(np.float32)


def sample_reference_window(displacements, sequence_length, count_steps):
    if displacements.shape[0] <= count_steps + sequence_length:
        start = sequence_length
    else:
        start = np.random.randint(low=sequence_length, high=displacements.shape[0] - count_steps)
    return displacements[start : start + count_steps].astype(np.float32)


def evaluate_model(model, data, count_steps, count_run, dt, step, periodic):
    displacements = data["displacements"]
    atom_order = data["atom_order"]
    reference_positions = data["reference_positions"]
    reference_window = sample_reference_window(displacements, model_sequence_length(data), count_steps)
    reference_coords = crystal_frames_to_flat_positions(reference_window, reference_positions, atom_order)
    xi_ref, yi_ref, jlp_ref = get_sqw_default(reference_coords, dt, step)
    jlp_mean = np.zeros_like(jlp_ref)
    norm = 0.0
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_crystal_sequence(displacements, model_sequence_length(data), count_steps)
        predicted_displacements = model.run_crystal(count_steps, init, periodic=periodic)
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, dt, step)
        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)

    norm /= count_run
    jlp_mean /= count_run
    return norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def model_sequence_length(data):
    return int(data["X_blocks"].shape[1])


def plot_sqw(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp):
    plt.pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues")
    plt.show()
    plt.pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues")
    plt.show()


def save_model(model, models_dir, norm, rnn_type, data_len, count_steps):
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    filepath = models_path / f"mean_norm_{norm}_{rnn_type.lower()}_crystal_{int(data_len * count_steps)}.pth"
    torch.save(model, filepath)
    print(f"==============> SAVE MODEL TO FILE - {filepath}")


def main():
    args = parse_args()
    data = load_training_data(args.data_path)
    unit_cell_atoms = int(data["X_blocks"].shape[5])
    train_supercell_shape = tuple(int(value) for value in data["train_supercell_shape"])

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        predictor = CrystalRNNNet(
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            type=args.rnn_type.upper(),
            train_supercell_shape=train_supercell_shape,
            unit_cell_atoms=unit_cell_atoms,
            flatten_order=DEFAULT_FLATTEN_ORDER,
        )
        predictor.batch_size = BATCH_SIZE
        predictor.train_crystal_blocks(data["X_blocks"], data["y_blocks"], data_len=DATA_LEN)

        norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=predictor,
            data=data,
            count_steps=args.count_steps,
            count_run=args.count_run,
            dt=DT,
            step=STEP,
            periodic=args.periodic,
        )
        print("CURRENT_NORM =", norm)

        if norm < args.save_threshold:
            plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
            save_model(predictor, args.models_dir, norm, args.rnn_type, DATA_LEN, args.count_steps)

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
