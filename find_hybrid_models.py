"""Train and select hybrid flat-RNN + ConvRNN crystal predictors by S(q,w)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CrystalHybridRNNNet, get_sqw


COUNT_STEPS = 2000
DELTA = 10000
STEP = 10
DT = 0.02
DATA_LEN = 0.2
COUNT_RUN = 3
SAVE_THRESHOLD = 2.5
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3


def parse_args():
    """Parse command-line options for hybrid model search."""
    parser = argparse.ArgumentParser(description="Train and select hybrid crystal predictors.")
    parser.add_argument("count_models", type=int, help="Number of models to train.")
    parser.add_argument("--data-path", default="crystal_training_data.npz")
    parser.add_argument("--models-dir", default="models_hybrid")
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument("--flat-type", default="RNN", choices=["RNN", "GRU", "LSTM"])
    parser.add_argument("--flat-hidden-size", type=int, default=100)
    parser.add_argument("--flat-num-layers", type=int, default=3)
    parser.add_argument("--conv-type", default="ConvGRU", choices=["ConvRNN", "ConvGRU", "ConvLSTM", "RNN", "GRU", "LSTM"])
    parser.add_argument("--conv-hidden-channels", type=int, default=32)
    parser.add_argument("--conv-num-layers", type=int, default=1)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument(
        "--conv-periodic-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use circular padding inside the ConvRNN branch.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--target-mode", choices=["absolute", "delta", "absolute_delta"], default="absolute_delta")
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-epsilon", type=float, default=1e-6)
    parser.add_argument("--residual-output", action="store_true")
    parser.add_argument("--initial-alpha", type=float, default=0.5, help="Initial flat-branch mixture weight.")
    parser.add_argument("--pretrained-flat-model", default=None, help="Optional saved CrystalRNNNet for the flat branch.")
    parser.add_argument("--freeze-flat", action="store_true", help="Do not update pretrained flat-branch weights.")
    parser.add_argument("--merge-mode", default="owner", choices=["mean", "weighted", "center", "owner", "soft_center"])
    parser.add_argument("--periodic", action="store_true", help="Use periodic block wrapping in the flat branch.")
    return parser.parse_args()


def load_training_data(path):
    """Load crystal training arrays from an ``.npz`` dataset."""
    print("IMPORT DATA")
    data = np.load(path)
    required = ["X_blocks", "y_blocks", "displacements", "atom_order", "reference_positions", "train_supercell_shape"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    loaded = {key: data[key] for key in data.files}
    print("X_blocks", loaded["X_blocks"].shape)
    print("y_blocks", loaded["y_blocks"].shape)
    print("displacements", loaded["displacements"].shape)
    return loaded


def sample_train_data(data, delta, sequence_length):
    """Select a random consecutive frame window and its block samples."""
    displacements = data["displacements"]
    X_blocks = data["X_blocks"]
    y_blocks = data["y_blocks"]
    frame_count = displacements.shape[0]
    total_time_windows = frame_count - sequence_length
    if total_time_windows <= 0:
        raise ValueError("Need more displacement frames than sequence_length")
    if X_blocks.shape[0] % total_time_windows != 0:
        raise ValueError("X_blocks count is inconsistent with displacements and sequence_length")

    blocks_per_time_window = X_blocks.shape[0] // total_time_windows
    if frame_count <= delta:
        start_frame = 0
        window_frame_count = frame_count
    else:
        start_frame = np.random.randint(low=0, high=frame_count - delta + 1)
        window_frame_count = delta

    start_sample = start_frame * blocks_per_time_window
    sample_count = (window_frame_count - sequence_length) * blocks_per_time_window
    stop_sample = start_sample + sample_count
    print(f"TRAIN FRAMES {start_frame}:{start_frame + window_frame_count}")
    print(f"TRAIN BLOCKS {start_sample}:{stop_sample}")
    return (
        displacements[start_frame : start_frame + window_frame_count],
        X_blocks[start_sample:stop_sample],
        y_blocks[start_sample:stop_sample],
    )


def build_default_k_vectors():
    """Build the default reciprocal-space vectors used for S(q,w) scoring."""
    kmin = 2 * np.pi / (NCELLS * LATTICE_PARAMETER)
    kmax = NCELLS * kmin
    kmas = np.zeros((KCOUNT, 3), dtype=np.float32)
    kmas[:, 0] = np.linspace(kmin, kmax, KCOUNT)
    return kmas


def get_sqw_default(coords, dt, step):
    """Calculate S(q,w) with the repository's default k-vector grid."""
    return get_sqw(coords, dt=dt, step=step, kmas=build_default_k_vectors())


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacement frames to flat absolute positions."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def model_sequence_length(data):
    """Return the recurrent input history length."""
    return int(data["X_blocks"].shape[1])


def sample_initial_crystal_sequence(displacements, sequence_length, count_steps):
    """Sample an initial history for autoregressive evaluation."""
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = np.random.randint(low=sequence_length, high=displacements.shape[0] - count_steps)
    return displacements[begin_index - sequence_length : begin_index].astype(np.float32)


def evaluate_model(model, data, reference_displacements, count_steps, count_run, dt, step, periodic, merge_mode):
    """Run inference several times and compare S(q,w) to the training window."""
    displacements = data["displacements"]
    atom_order = data["atom_order"]
    reference_positions = data["reference_positions"]
    sequence_length = model_sequence_length(data)
    reference_coords = crystal_frames_to_flat_positions(reference_displacements, reference_positions, atom_order)
    xi_ref, yi_ref, jlp_ref = get_sqw_default(reference_coords, dt, step)
    jlp_mean = np.zeros_like(jlp_ref)
    norm = 0.0
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_crystal_sequence(displacements, sequence_length, count_steps)
        predicted_displacements = model.run_crystal(
            count_steps,
            init,
            periodic=periodic,
            merge_mode=merge_mode,
        )
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, dt, step)
        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)

    norm /= count_run
    jlp_mean /= count_run
    return norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def plot_sqw(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp):
    """Show reference and predicted S(q,w) maps."""
    plt.pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues")
    plt.show()
    plt.pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues")
    plt.show()


def save_model(model, models_dir, norm, count_steps):
    """Persist a selected hybrid model."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    filepath = (
        models_path
        / (
            f"mean_norm_{norm}_hybrid_{model.target_mode}"
            f"{'_residual' if model.residual_output else ''}"
            f"_alpha{model.alpha:.3f}_{count_steps}.pth"
        )
    )
    torch.save(model, filepath)
    print(f"==============> SAVE MODEL TO FILE - {filepath}")


def main():
    """Train candidate hybrid models, evaluate them, and save selected models."""
    args = parse_args()
    data = load_training_data(args.data_path)
    unit_cell_atoms = int(data["X_blocks"].shape[5])
    train_supercell_shape = tuple(int(value) for value in data["train_supercell_shape"])

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        predictor = CrystalHybridRNNNet(
            flat_hidden_size=args.flat_hidden_size,
            flat_num_layers=args.flat_num_layers,
            conv_hidden_channels=args.conv_hidden_channels,
            conv_num_layers=args.conv_num_layers,
            train_supercell_shape=train_supercell_shape,
            unit_cell_atoms=unit_cell_atoms,
            flat_type=args.flat_type,
            conv_type=args.conv_type,
            conv_kernel_size=args.conv_kernel_size,
            conv_periodic_padding=args.conv_periodic_padding,
            target_mode=args.target_mode,
            delta_loss_weight=args.delta_loss_weight,
            delta_loss_epsilon=args.delta_loss_epsilon,
            residual_output=args.residual_output,
            initial_alpha=args.initial_alpha,
            freeze_flat=args.freeze_flat,
        )
        if args.pretrained_flat_model is not None:
            flat_model = torch.load(args.pretrained_flat_model, map_location="cpu", weights_only=False)
            predictor.set_flat_model(flat_model, freeze_flat=args.freeze_flat)
        predictor.batch_size = args.batch_size
        predictor.epochs = args.epochs
        predictor.lr = args.learning_rate
        train_displacements, X_train_blocks, y_train_blocks = sample_train_data(
            data=data,
            delta=DELTA,
            sequence_length=model_sequence_length(data),
        )
        predictor.train_crystal_blocks(X_train_blocks, y_train_blocks, data_len=DATA_LEN)
        print("ALPHA =", predictor.alpha)

        norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=predictor,
            data=data,
            reference_displacements=train_displacements,
            count_steps=args.count_steps,
            count_run=args.count_run,
            dt=DT,
            step=STEP,
            periodic=args.periodic,
            merge_mode=args.merge_mode,
        )
        print("CURRENT_NORM =", norm)

        if norm < args.save_threshold:
            plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
            save_model(predictor, args.models_dir, norm, int(DATA_LEN * args.count_steps))

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
