"""Train shared local crystal RNN models and select them by S(q,w).

This script is the translation-equivariant counterpart of ``find_models.py``.
Instead of training a full block-to-block map, it samples local crystal patches
around random unit cells and trains one shared RNN to predict only the central
unit cell.  The same local operator can then be applied to crystals of any
compatible shape without overlap stitching.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes import CrystalLocalRNNNet, DEFAULT_FLATTEN_ORDER, get_sqw, make_local_patch_samples


COUNT_STEPS = 2000
COUNT_RUN = 3
DELTA = 10000
STEP = 10
DT = 0.02
HIDDEN_SIZE = 100
BATCH_SIZE = 200
NUM_LAYERS = 3
EPOCHS = 50
LR = 1e-3
SAVE_THRESHOLD = 4.5
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3
TRAIN_SAMPLES = 20000
VALIDATION_SAMPLES = 5000
VALIDATION_WINDOWS = 2000


def parse_args():
    """Parse command-line options for local-RNN model search."""
    parser = argparse.ArgumentParser(description="Train and select shared local crystal RNN predictors.")
    parser.add_argument("count_models", type=int, help="Number of candidate models to train.")
    parser.add_argument("rnn_type", choices=["RNN", "GRU", "LSTM"], help="Recurrent block type.")
    parser.add_argument("--data-path", default="data333.npz", help="Crystal training .npz file.")
    parser.add_argument("--models-dir", default="models333_local_rnn", help="Directory for selected models.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics output path.")
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(3, 3, 3), metavar=("PX", "PY", "PZ"))
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES)
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument("--delta-frames", type=int, default=DELTA)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic wrapping for local patches.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "delta", "absolute_delta", "acceleration", "verlet"],
        default="absolute_delta",
        help="Training target convention for the central unit cell.",
    )
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-epsilon", type=float, default=1e-6)
    parser.add_argument("--acceleration-loss-weight", type=float, default=0.0)
    parser.add_argument("--acceleration-loss-epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-sqw",
        action="store_true",
        help="Only run one-step validation; useful for quick smoke tests.",
    )
    parser.add_argument(
        "--save-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save every trained model, not only models below --save-threshold.",
    )
    parser.add_argument("--plot-selected", action="store_true", help="Show S(q,w) plots for selected models.")
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Save an S(q,w) comparison image for every evaluated model.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        help="Directory for per-iteration S(q,w) comparison images.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show each saved S(q,w) comparison image interactively. This blocks until the plot window is closed.",
    )
    return parser.parse_args()


def load_data(path):
    """Load arrays required for local-RNN training and S(q,w) evaluation."""
    print("IMPORT DATA")
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    loaded = {key: data[key] for key in data.files}
    print("X_blocks", loaded["X_blocks"].shape)
    print("displacements", loaded["displacements"].shape)
    return loaded


def model_sequence_length(data):
    """Return the recurrent history length stored in prepared training blocks."""
    return int(data["X_blocks"].shape[1])


def choose_train_window(displacements, sequence_length, delta_frames, rng):
    """Select a random consecutive frame window, matching the old DELTA logic."""
    frame_count = int(displacements.shape[0])
    if frame_count <= sequence_length:
        raise ValueError("Need more displacement frames than sequence_length")
    delta_frames = int(delta_frames)
    if delta_frames <= sequence_length:
        raise ValueError("delta-frames must be greater than sequence_length")
    if frame_count <= delta_frames:
        start_frame = 0
        window_frame_count = frame_count
    else:
        start_frame = int(rng.integers(0, frame_count - delta_frames + 1))
        window_frame_count = delta_frames
    print(f"TRAIN FRAMES {start_frame}:{start_frame + window_frame_count}")
    return start_frame, displacements[start_frame : start_frame + window_frame_count]


def build_model(args, unit_cell_atoms):
    """Create one shared local RNN candidate."""
    model = CrystalLocalRNNNet(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        type=args.rnn_type,
        patch_shape=tuple(args.patch_shape),
        unit_cell_atoms=unit_cell_atoms,
        flatten_order=DEFAULT_FLATTEN_ORDER,
        target_mode=args.target_mode,
        delta_loss_weight=args.delta_loss_weight,
        delta_loss_epsilon=args.delta_loss_epsilon,
        acceleration_loss_weight=args.acceleration_loss_weight,
        acceleration_loss_epsilon=args.acceleration_loss_epsilon,
    )
    model.batch_size = int(args.batch_size)
    model.epochs = int(args.epochs)
    model.lr = float(args.learning_rate)
    return model


def sample_training_data(args, train_displacements, sequence_length, rng):
    """Sample local patch training arrays from one consecutive frame window."""
    X_train, y_train = make_local_patch_samples(
        train_displacements,
        sequence_length=sequence_length,
        patch_shape=tuple(args.patch_shape),
        sample_count=args.train_samples,
        rng=rng,
        periodic=args.periodic,
    )
    print("X_local_train", X_train.shape)
    print("y_local_train", y_train.shape)
    return X_train, y_train


def sample_validation_data(args, displacements, sequence_length, rng):
    """Sample local patch validation arrays from the final time windows."""
    total_windows = displacements.shape[0] - sequence_length
    validation_windows = min(int(args.validation_windows), max(1, total_windows // 2))
    validation_start = total_windows - validation_windows
    X_val, y_val = make_local_patch_samples(
        displacements,
        sequence_length=sequence_length,
        patch_shape=tuple(args.patch_shape),
        sample_count=args.validation_samples,
        rng=rng,
        start_window=validation_start,
        stop_window=total_windows,
        periodic=args.periodic,
    )
    print(f"VALIDATION WINDOWS {validation_start}:{total_windows}")
    return X_val, y_val


def relative_l2(predicted, reference):
    """Return normalized L2 error."""
    denominator = np.linalg.norm(reference.reshape(-1))
    if denominator == 0:
        return np.nan
    return float(np.linalg.norm((predicted - reference).reshape(-1)) / denominator)


def correlation(first, second):
    """Return Pearson correlation between two arrays."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 2:
        return np.nan

    first = first[finite] - np.mean(first[finite])
    second = second[finite] - np.mean(second[finite])
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return np.nan
    return float(np.dot(first, second) / denominator)


def evaluate_one_step(model, X_val, y_val, batch_size):
    """Evaluate one-step central-cell prediction quality."""
    predicted = model.predict_local_patches(X_val, batch_size=batch_size)
    last = X_val[(slice(None), -1, *model.center_index, slice(None), slice(None))]
    pred_delta = predicted - last
    true_delta = y_val - last
    pred_std = float(np.std(pred_delta))
    true_std = float(np.std(true_delta))
    return {
        "one_step_mse": float(np.mean((predicted - y_val) ** 2)),
        "one_step_rel_l2": relative_l2(predicted, y_val),
        "delta_corr": correlation(pred_delta, true_delta),
        "delta_rel_l2": relative_l2(pred_delta, true_delta),
        "delta_std_ratio": float(pred_std / true_std) if true_std > 0 else np.nan,
    }


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
    """Convert crystal-shaped displacements to legacy flat coordinates."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def sample_initial_crystal_sequence(displacements, sequence_length, count_steps, rng):
    """Sample an initial crystal displacement history for autoregression."""
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = int(rng.integers(sequence_length, displacements.shape[0] - count_steps))
    return displacements[begin_index - sequence_length : begin_index].astype(np.float32)


def evaluate_sqw(model, data, reference_displacements, count_steps, count_run, dt, step, periodic, rng):
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
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_crystal_sequence(displacements, sequence_length, count_steps, rng)
        predicted_displacements = model.run_crystal(count_steps, init, periodic=periodic)
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, dt, step)
        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)

    norm /= count_run
    jlp_mean /= count_run
    return norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def plot_sqw(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp):
    """Show reference and predicted S(q,w) maps using the same color scheme."""
    import matplotlib.pyplot as plt

    plt.pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues")
    plt.show()
    plt.pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues")
    plt.show()


def save_sqw_plot(
    reference_xi,
    reference_yi,
    reference_jlp,
    predicted_xi,
    predicted_yi,
    predicted_jlp,
    output_path,
    title,
    show=False,
):
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


def save_model(model, models_dir, metrics, args, sqw_norm):
    """Save one local model with its main metrics in the filename."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    patch = "x".join(str(dim) for dim in model.patch_shape)
    target = getattr(model, "target_mode", "absolute_delta")
    sqw_text = "nan" if not np.isfinite(sqw_norm) else f"{sqw_norm:.12g}"
    filename = (
        f"mean_norm_{sqw_text}_{args.rnn_type.lower()}_local_patch{patch}_{target}"
        f"_dcorr{metrics['delta_corr']:.4g}_drel{metrics['delta_rel_l2']:.4g}.pth"
    )
    path = models_path / filename
    torch.save(model, path)
    print(f"==============> SAVE MODEL TO FILE - {path}")
    return path


def write_metrics(path, rows):
    """Write model-search metrics to a TSV file."""
    fields = [
        "iteration",
        "model_path",
        "sqw_norm",
        "one_step_mse",
        "one_step_rel_l2",
        "delta_corr",
        "delta_rel_l2",
        "delta_std_ratio",
        "train_samples",
        "validation_samples",
        "epochs",
        "batch_size",
        "learning_rate",
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
    """Train candidate local RNNs, evaluate them, and save selected models."""
    args = parse_args()
    if args.count_models <= 0:
        raise ValueError("count_models must be positive")
    if args.train_samples <= 0:
        raise ValueError("train-samples must be positive")
    if args.validation_samples <= 0:
        raise ValueError("validation-samples must be positive")

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    data = load_data(args.data_path)
    sequence_length = model_sequence_length(data)
    unit_cell_atoms = int(data["displacements"].shape[4])
    rows = []

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        model = build_model(args, unit_cell_atoms)
        _, train_displacements = choose_train_window(data["displacements"], sequence_length, args.delta_frames, rng)
        X_train, y_train = sample_training_data(args, train_displacements, sequence_length, rng)
        X_val, y_val = sample_validation_data(args, data["displacements"], sequence_length, rng)
        model.train_local_patches(X_train, y_train, data_len=1.0)

        metrics = evaluate_one_step(model, X_val, y_val, args.batch_size)
        print("ONE_STEP_METRICS", metrics)
        sqw_norm = np.nan
        sqw_payload = None
        if not args.skip_sqw:
            sqw_payload = evaluate_sqw(
                model=model,
                data=data,
                reference_displacements=train_displacements,
                count_steps=args.count_steps,
                count_run=args.count_run,
                dt=DT,
                step=STEP,
                periodic=args.periodic,
                rng=rng,
            )
            sqw_norm = sqw_payload[0]
            print("CURRENT_NORM =", sqw_norm)

        model_path = ""
        if args.save_all or (np.isfinite(sqw_norm) and sqw_norm < args.save_threshold):
            model_path = str(save_model(model, args.models_dir, metrics, args, sqw_norm))
            if args.plot_selected and sqw_payload is not None:
                _, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = sqw_payload
                plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
        if args.plot_all and sqw_payload is not None:
            _, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = sqw_payload
            output_dir = Path(args.plot_output_dir or args.models_dir)
            save_sqw_plot(
                xi_ref,
                yi_ref,
                jlp_ref,
                xi_pred,
                yi_pred,
                jlp_mean,
                output_dir / f"local_rnn_iter_{iteration:03d}_sqw.png",
                title=f"Local RNN iter {iteration}, S(q,w) norm = {sqw_norm:.6g}",
                show=args.show_plots,
            )

        rows.append(
            {
                "iteration": iteration,
                "model_path": model_path,
                "sqw_norm": sqw_norm,
                **metrics,
                "train_samples": args.train_samples,
                "validation_samples": args.validation_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            }
        )

    if args.metrics_path is not None:
        write_metrics(args.metrics_path, rows)
    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
