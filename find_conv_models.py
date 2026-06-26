"""Train and select convolutional recurrent crystal predictors by S(q,w)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CrystalConvRNNNet, get_sqw


COUNT_STEPS = 2000
DELTA = 10000
STEP = 10
DT = 0.02
HIDDEN_CHANNELS = 32
DATA_LEN = 0.2
BATCH_SIZE = 32
NUM_LAYERS = 2
COUNT_RUN = 3
SAVE_THRESHOLD = 3.5
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3


def parse_args():
    """Parse command-line options for ConvRNN model search."""
    parser = argparse.ArgumentParser(description="Train and select ConvRNN crystal predictors.")
    parser.add_argument("count_models", type=int, help="Number of models to train.")
    parser.add_argument("rnn_type", choices=["ConvRNN", "ConvGRU", "ConvLSTM", "RNN", "GRU", "LSTM"])
    parser.add_argument("--data-path", default="crystal_training_data.npz")
    parser.add_argument("--models-dir", default="models_conv")
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument("--hidden-channels", type=int, default=HIDDEN_CHANNELS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--periodic-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use circular padding inside Conv3D recurrent cells.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "delta", "absolute_delta"],
        default="absolute_delta",
    )
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--residual-output",
        action="store_true",
        help="Predict a correction to the last input frame instead of a direct output.",
    )
    return parser.parse_args()


def load_training_data(path):
    """Load crystal training arrays from an ``.npz`` dataset."""
    print("IMPORT DATA")
    data = np.load(path)
    required = [
        "X_blocks",
        "y_blocks",
        "displacements",
        "atom_order",
        "reference_positions",
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


def sample_train_data(data, delta, sequence_length):
    """Select a random consecutive frame window and its block samples."""
    displacements = data["displacements"]
    X_blocks = data["X_blocks"]
    y_blocks = data["y_blocks"]
    frame_count = displacements.shape[0]
    total_time_windows = frame_count - sequence_length
    if total_time_windows <= 0:
        raise ValueError("Need more displacement frames than sequence_length")
    if X_blocks.shape[0] != y_blocks.shape[0]:
        raise ValueError("X_blocks and y_blocks must contain the same number of samples")
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
    if sample_count <= 0:
        raise ValueError("Selected training window is too short for sequence_length")

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


def evaluate_model(model, data, reference_displacements, count_steps, count_run, dt, step):
    """Run inference several times and compare S(q,w) to the training window."""
    if count_run <= 0:
        raise ValueError("count_run must be positive")

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
        predicted_displacements = model.run_crystal(count_steps, init)
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


def save_model(model, models_dir, norm, rnn_type, count_steps):
    """Persist a selected ConvRNN model."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    filepath = (
        models_path
        / (
            f"mean_norm_{norm}_{rnn_type.lower()}_conv_crystal_{model.target_mode}"
            f"{'_residual' if getattr(model, 'residual_output', False) else ''}"
            f"_hc{model.hidden_channels}_{count_steps}.pth"
        )
    )
    torch.save(model, filepath)
    print(f"==============> SAVE MODEL TO FILE - {filepath}")


def main():
    """Train candidate ConvRNN models, evaluate them, and save selected models."""
    args = parse_args()
    data = load_training_data(args.data_path)
    unit_cell_atoms = int(data["X_blocks"].shape[5])

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        predictor = CrystalConvRNNNet(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            unit_cell_atoms=unit_cell_atoms,
            type=args.rnn_type,
            kernel_size=args.kernel_size,
            periodic_padding=args.periodic_padding,
            target_mode=args.target_mode,
            delta_loss_weight=args.delta_loss_weight,
            delta_loss_epsilon=args.delta_loss_epsilon,
            residual_output=args.residual_output,
        )
        predictor.batch_size = args.batch_size
        predictor.epochs = args.epochs
        predictor.lr = args.learning_rate
        train_displacements, X_train_blocks, y_train_blocks = sample_train_data(
            data=data,
            delta=DELTA,
            sequence_length=model_sequence_length(data),
        )
        predictor.train_crystal_blocks(X_train_blocks, y_train_blocks, data_len=DATA_LEN)

        norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=predictor,
            data=data,
            reference_displacements=train_displacements,
            count_steps=args.count_steps,
            count_run=args.count_run,
            dt=DT,
            step=STEP,
        )
        print("CURRENT_NORM =", norm)

        if norm < args.save_threshold:
            plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
            save_model(predictor, args.models_dir, norm, args.rnn_type, int(DATA_LEN * args.count_steps))

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
