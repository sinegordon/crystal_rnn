"""Train recurrent-convolutional crystal field predictors and select by S(q,w)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes import CrystalFieldRNNNet, get_sqw


COUNT_STEPS = 2000
DELTA = 10000
STEP = 10
DT = 0.02
ENCODER_CHANNELS = 32
RNN_HIDDEN_SIZE = 64
RNN_LAYERS = 1
CONV_LAYERS = 1
DATA_LEN = 0.2
BATCH_SIZE = 32
COUNT_RUN = 3
SAVE_THRESHOLD = 3.5
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3
VELOCITY_WINDOW_FRAMES = 10
VELOCITY_HIST_BINS = 80


def parse_args():
    """Parse command-line options for field-RNN model search."""
    parser = argparse.ArgumentParser(description="Train and select Conv-encoder + temporal-RNN crystal predictors.")
    parser.add_argument("count_models", type=int, help="Number of models to train.")
    parser.add_argument("rnn_type", choices=["RNN", "GRU", "LSTM"], help="Temporal recurrent block type.")
    parser.add_argument("--data-path", default="data333.npz")
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="Optional prepared .npz dataset used for model selection rollouts. Defaults to --data-path.",
    )
    parser.add_argument("--models-dir", default="models333_field_rnn")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument(
        "--velocity-score-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the velocity-histogram mismatch in the model-selection score. "
            "The default keeps the historical pure S(q,w) criterion."
        ),
    )
    parser.add_argument(
        "--velocity-window-frames",
        type=int,
        default=VELOCITY_WINDOW_FRAMES,
        help="Number of velocity intervals from the beginning and end of each rollout window.",
    )
    parser.add_argument(
        "--velocity-hist-bins",
        type=int,
        default=VELOCITY_HIST_BINS,
        help="Histogram bins used by the velocity-distribution selection penalty.",
    )
    parser.add_argument(
        "--velocity-max-end-speed-ratio",
        type=float,
        default=float("inf"),
        help=(
            "Reject saving a model when mean |v| in the final predicted window divided by "
            "the final reference window exceeds this value."
        ),
    )
    parser.add_argument("--delta-frames", type=int, default=DELTA)
    parser.add_argument("--encoder-channels", type=int, default=ENCODER_CHANNELS)
    parser.add_argument("--rnn-hidden-size", type=int, default=RNN_HIDDEN_SIZE)
    parser.add_argument("--rnn-layers", type=int, default=RNN_LAYERS)
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use a bidirectional temporal RNN/GRU/LSTM before the decoder.",
    )
    parser.add_argument("--conv-layers", type=int, default=CONV_LAYERS)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-len", type=float, default=DATA_LEN)
    parser.add_argument(
        "--periodic-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use circular padding inside Conv3D encoder/decoder layers.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "delta", "absolute_delta", "acceleration", "verlet"],
        default="absolute_delta",
    )
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--acceleration-normalization",
        choices=["none", "global", "channel"],
        default="none",
        help="Normalize acceleration targets when --target-mode acceleration.",
    )
    parser.add_argument(
        "--acceleration-normalization-epsilon",
        type=float,
        default=1e-12,
        help="Small stabilizer for acceleration target normalization.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for training/inference: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--loss-region",
        choices=["all", "center_cell"],
        default="all",
        help="Spatial region used in the supervised loss. center_cell trains only the central unit cell.",
    )
    parser.add_argument(
        "--cyclic-shift-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly roll crystal blocks over unit-cell axes during training.",
    )
    parser.add_argument(
        "--input-transform",
        choices=[
            "absolute",
            "center_mean_relative",
            "same_atom_relative",
            "absolute_plus_same_atom_relative",
        ],
        default="absolute",
        help="Transform displacement histories before the neural network.",
    )
    parser.add_argument(
        "--input-relative-scale",
        type=float,
        default=1.0,
        help="Scale factor for relative channels in absolute_plus_same_atom_relative input.",
    )
    parser.add_argument(
        "--force-balance-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight that penalizes mean acceleration over atoms in the supervised region.",
    )
    parser.add_argument(
        "--acceleration-rms-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for matching predicted/reference physical acceleration RMS.",
    )
    parser.add_argument(
        "--velocity-rms-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for matching predicted/reference next-step velocity RMS.",
    )
    parser.add_argument(
        "--rms-loss-epsilon",
        type=float,
        default=1e-12,
        help="Small stabilizer for log-RMS ratio auxiliary losses.",
    )
    parser.add_argument(
        "--low-q-stiffness-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss weight for matching low-q acceleration response per displacement amplitude. "
            "This is a long-wavelength effective-stiffness penalty."
        ),
    )
    parser.add_argument(
        "--low-q-stiffness-max-shell",
        type=int,
        default=1,
        help="Largest integer Fourier shell n_x^2+n_y^2+n_z^2 included in the low-q stiffness loss.",
    )
    parser.add_argument(
        "--low-q-stiffness-epsilon",
        type=float,
        default=1e-8,
        help="Small stabilizer for the low-q stiffness auxiliary loss.",
    )
    parser.add_argument("--activation", choices=["elu", "relu", "gelu"], default="elu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-all", action="store_true")
    parser.add_argument("--plot-output-dir", default=None)
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show each saved S(q,w) comparison image interactively.",
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


def model_sequence_length(data):
    """Return the recurrent input history length."""
    return int(data["X_blocks"].shape[1])


def sample_train_data(data, delta, sequence_length, rng, label="TRAIN"):
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
        start_frame = int(rng.integers(0, frame_count - delta + 1))
        window_frame_count = int(delta)

    start_sample = start_frame * blocks_per_time_window
    sample_count = (window_frame_count - sequence_length) * blocks_per_time_window
    if sample_count <= 0:
        raise ValueError("Selected training window is too short for sequence_length")

    stop_sample = start_sample + sample_count
    print(f"{label} FRAMES {start_frame}:{start_frame + window_frame_count}")
    print(f"{label} BLOCKS {start_sample}:{stop_sample}")
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


def sample_initial_crystal_sequence(displacements, sequence_length, count_steps, rng):
    """Sample an initial history for autoregressive evaluation."""
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = int(rng.integers(sequence_length, displacements.shape[0] - count_steps))
    return displacements[begin_index - sequence_length : begin_index].astype(np.float32)


def velocity_windows(displacements, dt, window_frames):
    """Return beginning and ending velocity windows from displacement frames."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if window_frames <= 0:
        raise ValueError("velocity-window-frames must be positive")
    if displacements.shape[0] <= window_frames:
        raise ValueError("trajectory is too short for the requested velocity window")

    velocities = np.diff(displacements, axis=0) / dt
    return velocities[:window_frames], velocities[-window_frames:]


def flatten_velocity_components(velocities):
    """Return all velocity components as one vector."""
    return np.asarray(velocities, dtype=np.float64).reshape(-1)


def flatten_velocity_speeds(velocities):
    """Return per-atom velocity magnitudes as one vector."""
    velocities = np.asarray(velocities, dtype=np.float64)
    return np.linalg.norm(velocities.reshape(-1, 3), axis=1)


def displacement_velocity(displacements):
    """Return one-step displacement increments."""
    displacements = np.asarray(displacements, dtype=np.float64)
    if displacements.shape[0] < 2:
        raise ValueError("At least two frames are required for velocity increments")
    return np.diff(displacements, axis=0)


def displacement_acceleration(displacements):
    """Return discrete displacement accelerations."""
    displacements = np.asarray(displacements, dtype=np.float64)
    if displacements.shape[0] < 3:
        raise ValueError("At least three frames are required for accelerations")
    return displacements[2:] - 2 * displacements[1:-1] + displacements[:-2]


def rms(values):
    """Return a scalar root-mean-square."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def safe_ratio(numerator, denominator):
    """Return a finite-friendly ratio for positive scale diagnostics."""
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator > 0:
        return numerator / denominator
    return 1.0 if numerator == 0 else np.inf


def validate_training_eval_compatibility(training_data, eval_data):
    """Validate that training and evaluation displacement conventions match."""
    if int(training_data["X_blocks"].shape[1]) != int(eval_data["X_blocks"].shape[1]):
        raise ValueError("training and evaluation sequence lengths differ")
    if int(training_data["X_blocks"].shape[5]) != int(eval_data["X_blocks"].shape[5]):
        raise ValueError("training and evaluation unit_cell_atoms differ")
    if not np.array_equal(training_data["atom_order"], eval_data["atom_order"]):
        raise ValueError("training and evaluation atom_order arrays differ")
    if not np.allclose(training_data["reference_positions"], eval_data["reference_positions"], atol=1e-4, rtol=1e-6):
        raise ValueError(
            "training and evaluation reference_positions differ. "
            "Rebase one dataset first, for example with rebase_crystal_reference.py."
        )


def probability_histogram(values, bins, hist_range):
    """Return a normalized histogram with zero-safe handling."""
    counts, _ = np.histogram(values, bins=bins, range=hist_range)
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros(bins, dtype=np.float64)
    return counts.astype(np.float64) / total


def total_variation_histogram_distance(predicted, reference, bins, hist_range):
    """Return total-variation distance between two one-dimensional histograms."""
    predicted_hist = probability_histogram(predicted, bins, hist_range)
    reference_hist = probability_histogram(reference, bins, hist_range)
    return 0.5 * float(np.sum(np.abs(predicted_hist - reference_hist)))


def finite_histogram_range(*arrays, lower_bound=None):
    """Build a non-degenerate histogram range that covers all finite values."""
    values = np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("velocity histogram inputs contain no finite values")

    low = float(np.min(values))
    high = float(np.max(values))
    if lower_bound is not None:
        low = float(lower_bound)
        high = max(high, low)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("velocity histogram range is not finite")
    if low == high:
        if lower_bound is not None:
            return low, low + 1.0
        pad = 1.0 if low == 0.0 else abs(low) * 0.05
        low -= pad
        high += pad
    return low, high


def velocity_distribution_metrics(predicted_displacements, reference_displacements, dt, window_frames, bins):
    """Compare predicted and reference velocity distributions in first/last windows."""
    pred_begin, pred_end = velocity_windows(predicted_displacements, dt, window_frames)
    ref_begin, ref_end = velocity_windows(reference_displacements, dt, window_frames)

    component_scores = []
    speed_scores = []
    for predicted_window, reference_window in [(pred_begin, ref_begin), (pred_end, ref_end)]:
        predicted_components = flatten_velocity_components(predicted_window)
        reference_components = flatten_velocity_components(reference_window)
        component_low, component_high = finite_histogram_range(predicted_components, reference_components)
        component_limit = max(abs(component_low), abs(component_high))
        component_range = (-component_limit, component_limit) if component_limit > 0 else (-1.0, 1.0)
        component_scores.append(
            total_variation_histogram_distance(
                predicted_components,
                reference_components,
                bins=bins,
                hist_range=component_range,
            )
        )

        predicted_speeds = flatten_velocity_speeds(predicted_window)
        reference_speeds = flatten_velocity_speeds(reference_window)
        speed_range = finite_histogram_range(predicted_speeds, reference_speeds, lower_bound=0.0)
        speed_scores.append(
            total_variation_histogram_distance(
                predicted_speeds,
                reference_speeds,
                bins=bins,
                hist_range=speed_range,
            )
        )

    predicted_end_speed_mean = float(np.mean(flatten_velocity_speeds(pred_end)))
    reference_end_speed_mean = float(np.mean(flatten_velocity_speeds(ref_end)))
    if reference_end_speed_mean > 0:
        end_speed_ratio = predicted_end_speed_mean / reference_end_speed_mean
    else:
        end_speed_ratio = 1.0 if predicted_end_speed_mean == 0 else np.inf
    component_score = float(np.mean(component_scores))
    speed_score = float(np.mean(speed_scores))
    return {
        "velocity_score": component_score + speed_score,
        "velocity_component_score": component_score,
        "velocity_speed_score": speed_score,
        "velocity_end_speed_ratio": float(end_speed_ratio),
        "velocity_pred_end_speed_mean": predicted_end_speed_mean,
        "velocity_ref_end_speed_mean": reference_end_speed_mean,
    }


def dynamics_scale_metrics(predicted_displacements, reference_displacements, window_frames):
    """Return RMS scale diagnostics for velocity increments and accelerations."""
    predicted_velocity = displacement_velocity(predicted_displacements)
    reference_velocity = displacement_velocity(reference_displacements)
    predicted_acceleration = displacement_acceleration(predicted_displacements)
    reference_acceleration = displacement_acceleration(reference_displacements)
    if predicted_velocity.shape[0] < window_frames or reference_velocity.shape[0] < window_frames:
        raise ValueError("trajectory is too short for velocity RMS diagnostics")
    if predicted_acceleration.shape[0] < window_frames or reference_acceleration.shape[0] < window_frames:
        raise ValueError("trajectory is too short for acceleration RMS diagnostics")

    predicted_velocity_rms = rms(predicted_velocity)
    reference_velocity_rms = rms(reference_velocity)
    predicted_velocity_end_rms = rms(predicted_velocity[-window_frames:])
    reference_velocity_end_rms = rms(reference_velocity[-window_frames:])
    predicted_acceleration_rms = rms(predicted_acceleration)
    reference_acceleration_rms = rms(reference_acceleration)
    predicted_acceleration_end_rms = rms(predicted_acceleration[-window_frames:])
    reference_acceleration_end_rms = rms(reference_acceleration[-window_frames:])
    return {
        "velocity_rms_ratio": safe_ratio(predicted_velocity_rms, reference_velocity_rms),
        "velocity_end_rms_ratio": safe_ratio(predicted_velocity_end_rms, reference_velocity_end_rms),
        "acceleration_rms_ratio": safe_ratio(predicted_acceleration_rms, reference_acceleration_rms),
        "acceleration_end_rms_ratio": safe_ratio(predicted_acceleration_end_rms, reference_acceleration_end_rms),
        "velocity_pred_rms": predicted_velocity_rms,
        "velocity_ref_rms": reference_velocity_rms,
        "velocity_pred_end_rms": predicted_velocity_end_rms,
        "velocity_ref_end_rms": reference_velocity_end_rms,
        "acceleration_pred_rms": predicted_acceleration_rms,
        "acceleration_ref_rms": reference_acceleration_rms,
        "acceleration_pred_end_rms": predicted_acceleration_end_rms,
        "acceleration_ref_end_rms": reference_acceleration_end_rms,
    }


def evaluate_model(model, data, reference_displacements, count_steps, count_run, dt, step, rng, velocity_window_frames, velocity_hist_bins):
    """Run inference several times and compare S(q,w) to the training window."""
    if count_run <= 0:
        raise ValueError("count-run must be positive")
    displacements = data["displacements"]
    atom_order = data["atom_order"]
    reference_positions = data["reference_positions"]
    sequence_length = model_sequence_length(data)
    reference_coords = crystal_frames_to_flat_positions(reference_displacements, reference_positions, atom_order)
    xi_ref, yi_ref, jlp_ref = get_sqw_default(reference_coords, dt, step)
    jlp_mean = np.zeros_like(jlp_ref)
    norm = 0.0
    velocity_metrics_sum = None
    scale_metrics_sum = None
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_crystal_sequence(displacements, sequence_length, count_steps, rng)
        predicted_displacements = model.run_crystal(count_steps, init)
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, dt, step)
        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)
        velocity_metrics = velocity_distribution_metrics(
            predicted_displacements=predicted_displacements,
            reference_displacements=reference_displacements,
            dt=dt,
            window_frames=velocity_window_frames,
            bins=velocity_hist_bins,
        )
        scale_metrics = dynamics_scale_metrics(
            predicted_displacements=predicted_displacements,
            reference_displacements=reference_displacements,
            window_frames=velocity_window_frames,
        )
        if velocity_metrics_sum is None:
            velocity_metrics_sum = {key: 0.0 for key in velocity_metrics}
        for key, value in velocity_metrics.items():
            velocity_metrics_sum[key] += float(value)
        if scale_metrics_sum is None:
            scale_metrics_sum = {key: 0.0 for key in scale_metrics}
        for key, value in scale_metrics.items():
            scale_metrics_sum[key] += float(value)

    norm /= count_run
    jlp_mean /= count_run
    velocity_metrics_mean = {key: value / count_run for key, value in velocity_metrics_sum.items()}
    scale_metrics_mean = {key: value / count_run for key, value in scale_metrics_sum.items()}
    velocity_metrics_mean.update(scale_metrics_mean)
    return norm, velocity_metrics_mean, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def save_sqw_plot(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp, output_path, title, show=False):
    """Save a side-by-side S(q,w) comparison image."""
    import matplotlib.pyplot as plt

    vmin = float(np.nanmin([np.nanmin(reference_jlp), np.nanmin(predicted_jlp)]))
    vmax = float(np.nanmax([np.nanmax(reference_jlp), np.nanmax(predicted_jlp)]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    mesh = axes[0].pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[0].set_title("Reference S(q,w)")
    axes[0].set_xlabel("|q|")
    axes[0].set_ylabel("Energy")
    axes[1].pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted S(q,w)")
    axes[1].set_xlabel("|q|")
    axes[1].set_ylabel("Energy")
    fig.colorbar(mesh, ax=axes, label="Normalized intensity")
    fig.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved S(q,w) plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_model(model, models_dir, norm, args):
    """Persist a selected field-RNN model."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    low_q_suffix = ""
    if getattr(model, "low_q_stiffness_loss_weight", 0.0) != 0.0:
        low_q_suffix = (
            f"_lqst{format(model.low_q_stiffness_loss_weight, 'g')}"
            f"s{model.low_q_stiffness_max_shell}"
        )
    filename = (
        f"mean_norm_{norm}_{args.rnn_type.lower()}_field_rnn_{model.target_mode}"
        f"_ec{model.encoder_channels}_rh{model.rnn_hidden_size}_rl{model.rnn_layers}"
        f"{'_bidir' if getattr(model, 'bidirectional', False) else ''}"
        f"{'_accnorm' + model.acceleration_normalization if model.acceleration_normalization != 'none' else ''}"
        f"{'_loss' + model.loss_region if getattr(model, 'loss_region', 'all') != 'all' else ''}"
        f"{'_cyshift' if getattr(model, 'cyclic_shift_augmentation', False) else ''}"
        f"{'_input' + model.input_transform if getattr(model, 'input_transform', 'absolute') != 'absolute' else ''}"
        f"{'_relscale' + format(model.input_relative_scale, 'g') if getattr(model, 'input_relative_scale', 1.0) != 1.0 else ''}"
        f"{'_fb' + format(model.force_balance_loss_weight, 'g') if getattr(model, 'force_balance_loss_weight', 0.0) != 0.0 else ''}"
        f"{'_arms' + format(model.acceleration_rms_loss_weight, 'g') if getattr(model, 'acceleration_rms_loss_weight', 0.0) != 0.0 else ''}"
        f"{'_vrms' + format(model.velocity_rms_loss_weight, 'g') if getattr(model, 'velocity_rms_loss_weight', 0.0) != 0.0 else ''}"
        f"{low_q_suffix}"
        f"_cl{model.conv_layers}_k{model.kernel_size}.pth"
    )
    path = models_path / filename
    torch.save(model, path)
    print(f"==============> SAVE MODEL TO FILE - {path}")
    return path


def write_metrics(path, rows):
    """Write model-search metrics to TSV."""
    fields = [
        "iteration",
        "model_path",
        "data_path",
        "eval_data_path",
        "sqw_norm",
        "selection_score",
        "velocity_score",
        "velocity_component_score",
        "velocity_speed_score",
        "velocity_end_speed_ratio",
        "velocity_pred_end_speed_mean",
        "velocity_ref_end_speed_mean",
        "velocity_rms_ratio",
        "velocity_end_rms_ratio",
        "acceleration_rms_ratio",
        "acceleration_end_rms_ratio",
        "velocity_pred_rms",
        "velocity_ref_rms",
        "velocity_pred_end_rms",
        "velocity_ref_end_rms",
        "acceleration_pred_rms",
        "acceleration_ref_rms",
        "acceleration_pred_end_rms",
        "acceleration_ref_end_rms",
        "velocity_score_weight",
        "velocity_window_frames",
        "velocity_hist_bins",
        "velocity_max_end_speed_ratio",
        "velocity_passed",
        "rnn_type",
        "encoder_channels",
        "rnn_hidden_size",
        "rnn_layers",
        "bidirectional",
        "target_mode",
        "acceleration_normalization",
        "loss_region",
        "cyclic_shift_augmentation",
        "input_transform",
        "input_relative_scale",
        "force_balance_loss_weight",
        "acceleration_rms_loss_weight",
        "velocity_rms_loss_weight",
        "rms_loss_epsilon",
        "low_q_stiffness_loss_weight",
        "low_q_stiffness_max_shell",
        "low_q_stiffness_epsilon",
        "conv_layers",
        "kernel_size",
        "delta_frames",
        "data_len",
        "count_steps",
        "count_run",
        "epochs",
        "batch_size",
        "learning_rate",
        "activation",
        "periodic_padding",
        "device",
        "final_train_loss",
        "best_train_loss",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[field]) if isinstance(row[field], str) else f"{row[field]:.10g}"
                for field in fields
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Train candidate field RNNs, evaluate them, and save selected models."""
    args = parse_args()
    if args.count_models <= 0:
        raise ValueError("count_models must be positive")
    if args.count_steps <= 0:
        raise ValueError("count-steps must be positive")
    if args.delta_frames <= 0:
        raise ValueError("delta-frames must be positive")
    if args.velocity_score_weight < 0:
        raise ValueError("velocity-score-weight must be non-negative")
    if args.velocity_window_frames <= 0:
        raise ValueError("velocity-window-frames must be positive")
    if args.velocity_hist_bins <= 0:
        raise ValueError("velocity-hist-bins must be positive")
    if args.velocity_max_end_speed_ratio <= 0:
        raise ValueError("velocity-max-end-speed-ratio must be positive")
    if args.acceleration_rms_loss_weight < 0:
        raise ValueError("acceleration-rms-loss-weight must be non-negative")
    if args.velocity_rms_loss_weight < 0:
        raise ValueError("velocity-rms-loss-weight must be non-negative")
    if args.rms_loss_epsilon <= 0:
        raise ValueError("rms-loss-epsilon must be positive")
    if args.low_q_stiffness_loss_weight < 0:
        raise ValueError("low-q-stiffness-loss-weight must be non-negative")
    if args.low_q_stiffness_max_shell <= 0:
        raise ValueError("low-q-stiffness-max-shell must be positive")
    if args.low_q_stiffness_epsilon <= 0:
        raise ValueError("low-q-stiffness-epsilon must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not 0 < args.data_len <= 1:
        raise ValueError("data-len must be in (0, 1]")

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data = load_training_data(args.data_path)
    eval_data = data if args.eval_data_path is None else load_training_data(args.eval_data_path)
    validate_training_eval_compatibility(data, eval_data)
    unit_cell_atoms = int(data["X_blocks"].shape[5])
    rows = []

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        predictor = CrystalFieldRNNNet(
            encoder_channels=args.encoder_channels,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_layers=args.rnn_layers,
            unit_cell_atoms=unit_cell_atoms,
            type=args.rnn_type,
            bidirectional=args.bidirectional,
            conv_layers=args.conv_layers,
            kernel_size=args.kernel_size,
            periodic_padding=args.periodic_padding,
            activation=args.activation,
            target_mode=args.target_mode,
            delta_loss_weight=args.delta_loss_weight,
            delta_loss_epsilon=args.delta_loss_epsilon,
            acceleration_normalization=args.acceleration_normalization,
            acceleration_normalization_epsilon=args.acceleration_normalization_epsilon,
            loss_region=args.loss_region,
            cyclic_shift_augmentation=args.cyclic_shift_augmentation,
            input_transform=args.input_transform,
            input_relative_scale=args.input_relative_scale,
            force_balance_loss_weight=args.force_balance_loss_weight,
            acceleration_rms_loss_weight=args.acceleration_rms_loss_weight,
            velocity_rms_loss_weight=args.velocity_rms_loss_weight,
            rms_loss_epsilon=args.rms_loss_epsilon,
            low_q_stiffness_loss_weight=args.low_q_stiffness_loss_weight,
            low_q_stiffness_max_shell=args.low_q_stiffness_max_shell,
            low_q_stiffness_epsilon=args.low_q_stiffness_epsilon,
            device=args.device,
        )
        print("DEVICE =", predictor.torch_device)
        predictor.batch_size = args.batch_size
        predictor.epochs = args.epochs
        predictor.lr = args.learning_rate
        train_displacements, X_train_blocks, y_train_blocks = sample_train_data(
            data=data,
            delta=args.delta_frames,
            sequence_length=model_sequence_length(data),
            rng=rng,
            label="TRAIN",
        )
        losses = predictor.train_crystal_blocks(X_train_blocks, y_train_blocks, data_len=args.data_len)
        if eval_data is data:
            eval_reference_displacements = train_displacements
        else:
            eval_reference_displacements, _, _ = sample_train_data(
                data=eval_data,
                delta=args.delta_frames,
                sequence_length=model_sequence_length(eval_data),
                rng=rng,
                label="EVAL",
            )

        norm, velocity_metrics, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=predictor,
            data=eval_data,
            reference_displacements=eval_reference_displacements,
            count_steps=args.count_steps,
            count_run=args.count_run,
            dt=DT,
            step=STEP,
            rng=rng,
            velocity_window_frames=args.velocity_window_frames,
            velocity_hist_bins=args.velocity_hist_bins,
        )
        selection_score = norm + args.velocity_score_weight * velocity_metrics["velocity_score"]
        velocity_passed = velocity_metrics["velocity_end_speed_ratio"] <= args.velocity_max_end_speed_ratio
        print("CURRENT_NORM =", norm)
        print("VELOCITY_SCORE =", velocity_metrics["velocity_score"])
        print("VELOCITY_END_SPEED_RATIO =", velocity_metrics["velocity_end_speed_ratio"])
        print("ACCELERATION_RMS_RATIO =", velocity_metrics["acceleration_rms_ratio"])
        print("ACCELERATION_END_RMS_RATIO =", velocity_metrics["acceleration_end_rms_ratio"])
        print("VELOCITY_RMS_RATIO =", velocity_metrics["velocity_rms_ratio"])
        print("VELOCITY_END_RMS_RATIO =", velocity_metrics["velocity_end_rms_ratio"])
        print("SELECTION_SCORE =", selection_score)
        print("VELOCITY_PASSED =", velocity_passed)

        if args.plot_all:
            output_dir = Path(args.plot_output_dir or args.models_dir)
            save_sqw_plot(
                xi_ref,
                yi_ref,
                jlp_ref,
                xi_pred,
                yi_pred,
                jlp_mean,
                output_dir / f"field_rnn_iter_{iteration:03d}_sqw.png",
                title=f"Field RNN iter {iteration}, S(q,w) norm = {norm:.6g}",
                show=args.show_plots,
            )

        model_path = ""
        if args.save_all or (selection_score < args.save_threshold and velocity_passed):
            model_path = str(save_model(predictor, args.models_dir, norm, args))

        rows.append(
            {
                "iteration": iteration,
                "model_path": model_path,
                "data_path": args.data_path,
                "eval_data_path": args.eval_data_path or args.data_path,
                "sqw_norm": norm,
                "selection_score": selection_score,
                "velocity_score": velocity_metrics["velocity_score"],
                "velocity_component_score": velocity_metrics["velocity_component_score"],
                "velocity_speed_score": velocity_metrics["velocity_speed_score"],
                "velocity_end_speed_ratio": velocity_metrics["velocity_end_speed_ratio"],
                "velocity_pred_end_speed_mean": velocity_metrics["velocity_pred_end_speed_mean"],
                "velocity_ref_end_speed_mean": velocity_metrics["velocity_ref_end_speed_mean"],
                "velocity_rms_ratio": velocity_metrics["velocity_rms_ratio"],
                "velocity_end_rms_ratio": velocity_metrics["velocity_end_rms_ratio"],
                "acceleration_rms_ratio": velocity_metrics["acceleration_rms_ratio"],
                "acceleration_end_rms_ratio": velocity_metrics["acceleration_end_rms_ratio"],
                "velocity_pred_rms": velocity_metrics["velocity_pred_rms"],
                "velocity_ref_rms": velocity_metrics["velocity_ref_rms"],
                "velocity_pred_end_rms": velocity_metrics["velocity_pred_end_rms"],
                "velocity_ref_end_rms": velocity_metrics["velocity_ref_end_rms"],
                "acceleration_pred_rms": velocity_metrics["acceleration_pred_rms"],
                "acceleration_ref_rms": velocity_metrics["acceleration_ref_rms"],
                "acceleration_pred_end_rms": velocity_metrics["acceleration_pred_end_rms"],
                "acceleration_ref_end_rms": velocity_metrics["acceleration_ref_end_rms"],
                "velocity_score_weight": args.velocity_score_weight,
                "velocity_window_frames": args.velocity_window_frames,
                "velocity_hist_bins": args.velocity_hist_bins,
                "velocity_max_end_speed_ratio": args.velocity_max_end_speed_ratio,
                "velocity_passed": bool(velocity_passed),
                "rnn_type": args.rnn_type,
                "encoder_channels": args.encoder_channels,
                "rnn_hidden_size": args.rnn_hidden_size,
                "rnn_layers": args.rnn_layers,
                "bidirectional": bool(args.bidirectional),
                "target_mode": args.target_mode,
                "acceleration_normalization": args.acceleration_normalization,
                "loss_region": args.loss_region,
                "cyclic_shift_augmentation": bool(args.cyclic_shift_augmentation),
                "input_transform": args.input_transform,
                "input_relative_scale": args.input_relative_scale,
                "force_balance_loss_weight": args.force_balance_loss_weight,
                "acceleration_rms_loss_weight": args.acceleration_rms_loss_weight,
                "velocity_rms_loss_weight": args.velocity_rms_loss_weight,
                "rms_loss_epsilon": args.rms_loss_epsilon,
                "low_q_stiffness_loss_weight": args.low_q_stiffness_loss_weight,
                "low_q_stiffness_max_shell": args.low_q_stiffness_max_shell,
                "low_q_stiffness_epsilon": args.low_q_stiffness_epsilon,
                "conv_layers": args.conv_layers,
                "kernel_size": args.kernel_size,
                "delta_frames": args.delta_frames,
                "data_len": args.data_len,
                "count_steps": args.count_steps,
                "count_run": args.count_run,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "activation": args.activation,
                "periodic_padding": args.periodic_padding,
                "device": str(predictor.torch_device),
                "final_train_loss": float(losses[-1]) if losses else np.nan,
                "best_train_loss": float(np.min(losses)) if losses else np.nan,
            }
        )

    if args.metrics_path is not None:
        write_metrics(args.metrics_path, rows)
    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
