"""Train several crystal-aware RNN models and keep the best candidates.

The script expects training data prepared by ``prepare_crystal_data.py``.  The
data contains crystal-shaped displacement tensors for training and the original
flat atom order required by the legacy S(q,w) evaluation utility.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CrystalRNNNet, DEFAULT_FLATTEN_ORDER, get_sqw


COUNT_STEPS = 2000
DELTA = 10000
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
    """Parse command-line options for model search and evaluation."""
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
    """Load and validate crystal training arrays from an ``.npz`` dataset."""
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


def sample_train_data(data, delta, sequence_length):
    """Select a random consecutive frame window and its training blocks."""
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
    train_displacements = displacements[start_frame : start_frame + window_frame_count]
    return train_displacements, X_blocks[start_sample:stop_sample], y_blocks[start_sample:stop_sample]


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
    """Convert crystal-shaped displacement frames to legacy flat coordinates.

    Parameters
    ----------
    displacements:
        Displacement tensor with shape ``(frames, nx, ny, nz, atoms_per_cell, 3)``.
    reference_positions:
        Equilibrium atom positions in the original flat atom order.
    atom_order:
        Integer tensor that maps each crystal slot to its original flat atom
        index.

    Returns
    -------
    numpy.ndarray
        Absolute positions flattened as ``(frames, atoms * 3)`` for ``get_sqw``.
    """
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    # Restore the original atom order because S(q,w) works with flat trajectories.
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def sample_initial_crystal_sequence(displacements, sequence_length, count_steps):
    """Sample an initial crystal displacement history for autoregression."""
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = np.random.randint(low=sequence_length, high=displacements.shape[0] - count_steps)

    return displacements[begin_index - sequence_length : begin_index].astype(np.float32)


def evaluate_model(model, data, reference_displacements, count_steps, count_run, dt, step, periodic):
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
        predicted_displacements = model.run_crystal(count_steps, init, periodic=periodic)
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, dt, step)

        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)

    norm /= count_run
    jlp_mean /= count_run
    return norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def model_sequence_length(data):
    """Return the recurrent input history length stored in the training blocks."""
    return int(data["X_blocks"].shape[1])


def plot_sqw(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp):
    """Show reference and predicted S(q,w) maps using the same color scheme."""
    plt.pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues")
    plt.show()
    plt.pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues")
    plt.show()


def save_model(model, models_dir, norm, rnn_type, data_len, count_steps):
    """Persist a selected model with its score and basic training metadata."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    filepath = models_path / f"mean_norm_{norm}_{rnn_type.lower()}_crystal_{int(data_len * count_steps)}.pth"
    torch.save(model, filepath)
    print(f"==============> SAVE MODEL TO FILE - {filepath}")


def main():
    """Train candidate models, evaluate them, and save models below threshold."""
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
            periodic=args.periodic,
        )
        print("CURRENT_NORM =", norm)

        if norm < args.save_threshold:
            plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
            save_model(predictor, args.models_dir, norm, args.rnn_type, DATA_LEN, args.count_steps)

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
