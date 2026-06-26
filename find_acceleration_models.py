"""Train flat RNN acceleration operators and select them by one-step acceleration metrics."""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes import CrystalRNNNet, DEFAULT_FLATTEN_ORDER
from base_classes.crystal_predictor import (
    _flatten_crystal_block_samples,
    _flatten_crystal_block_targets,
)
from plot_sqw_comparison import correlation


HIDDEN_SIZE = 100
NUM_LAYERS = 3
BATCH_SIZE = 200
EPOCHS = 50
LR = 1e-3
DATA_LEN = 1.0
VALIDATION_WINDOWS = 2000


def parse_args():
    """Parse command-line options for acceleration-model search."""
    parser = argparse.ArgumentParser(
        description="Train target_mode=acceleration RNN models and score one-step acceleration quality."
    )
    parser.add_argument("count_models", type=int, help="Number of candidate models to train.")
    parser.add_argument("rnn_type", choices=["RNN", "GRU", "LSTM"], help="Recurrent block type.")
    parser.add_argument("--data-path", default="data333.npz", help="Crystal training .npz file.")
    parser.add_argument("--models-dir", default="models333_acceleration_search")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics output path.")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--data-len", type=float, default=DATA_LEN)
    parser.add_argument(
        "--train-window-frames",
        type=int,
        default=0,
        help="Consecutive time windows per sampled train segment. 0 means all non-validation windows.",
    )
    parser.add_argument(
        "--train-window-count",
        type=int,
        default=1,
        help="Number of random train segments when --train-window-frames is positive.",
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        default=VALIDATION_WINDOWS,
        help="Number of final time windows reserved for one-step validation.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save every trained model with acceleration metrics in the filename.",
    )
    return parser.parse_args()


def load_data(path):
    """Load arrays required for acceleration training."""
    data = np.load(path)
    required = ["X_blocks", "y_blocks", "train_supercell_shape"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return {key: data[key] for key in data.files}


def time_window_sample_indices(windows, blocks_per_window):
    """Expand time-window indices into flattened block sample indices."""
    starts = np.asarray(windows, dtype=np.int64) * blocks_per_window
    offsets = np.arange(blocks_per_window, dtype=np.int64)
    return (starts[:, None] + offsets[None, :]).reshape(-1)


def choose_train_validation_indices(data, validation_windows, train_window_frames, train_window_count, rng):
    """Return train/validation sample indices based on time-window splits."""
    sequence_length = int(data["X_blocks"].shape[1])
    if "displacements" in data:
        total_time_windows = int(data["displacements"].shape[0] - sequence_length)
    else:
        total_time_windows = int(data["X_blocks"].shape[0])
    if total_time_windows <= 0:
        raise ValueError("Dataset is too short for the stored sequence_length")
    if data["X_blocks"].shape[0] % total_time_windows != 0:
        raise ValueError("X_blocks count is inconsistent with total time windows")

    blocks_per_window = data["X_blocks"].shape[0] // total_time_windows
    validation_windows = min(int(validation_windows), total_time_windows // 2)
    validation_start = total_time_windows - validation_windows
    validation_time_windows = np.arange(validation_start, total_time_windows, dtype=np.int64)

    if train_window_frames <= 0:
        train_time_windows = np.arange(0, validation_start, dtype=np.int64)
    else:
        train_window_frames = int(train_window_frames)
        train_window_count = int(train_window_count)
        if train_window_frames <= 0 or train_window_count <= 0:
            raise ValueError("train-window options must be positive")
        if validation_start < train_window_frames:
            raise ValueError("Not enough non-validation windows for requested train-window-frames")
        windows = []
        for _ in range(train_window_count):
            start = int(rng.integers(0, validation_start - train_window_frames + 1))
            windows.append(np.arange(start, start + train_window_frames, dtype=np.int64))
        train_time_windows = np.unique(np.concatenate(windows))

    train_indices = time_window_sample_indices(train_time_windows, blocks_per_window)
    validation_indices = time_window_sample_indices(validation_time_windows, blocks_per_window)
    return train_indices, validation_indices, blocks_per_window


def build_model(args, data):
    """Create a crystal-aware acceleration RNN candidate."""
    train_supercell_shape = tuple(int(value) for value in data["train_supercell_shape"])
    unit_cell_atoms = int(data["X_blocks"].shape[5])
    model = CrystalRNNNet(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        type=args.rnn_type,
        train_supercell_shape=train_supercell_shape,
        unit_cell_atoms=unit_cell_atoms,
        flatten_order=DEFAULT_FLATTEN_ORDER,
        target_mode="acceleration",
    )
    model.batch_size = int(args.batch_size)
    model.epochs = int(args.epochs)
    model.lr = float(args.learning_rate)
    return model


def acceleration_targets(X_blocks, y_blocks):
    """Return crystal-shaped discrete acceleration targets."""
    if X_blocks.shape[1] < 2:
        raise ValueError("Acceleration target requires at least two input history frames")
    return y_blocks - 2 * X_blocks[:, -1] + X_blocks[:, -2]


def evaluate_acceleration(model, X_blocks, y_blocks, batch_size):
    """Evaluate one-step acceleration metrics on validation blocks."""
    true_blocks = acceleration_targets(X_blocks, y_blocks)
    true_flat = _flatten_crystal_block_targets(true_blocks, model.flatten_order)
    pred_parts = []
    model.model.eval()
    with torch.no_grad():
        for start in range(0, X_blocks.shape[0], batch_size):
            stop = min(start + batch_size, X_blocks.shape[0])
            X_flat = _flatten_crystal_block_samples(X_blocks[start:stop], model.flatten_order)
            pred = model.model(torch.as_tensor(X_flat, dtype=torch.float32)).detach().cpu().numpy()
            pred_parts.append(pred)
    pred_flat = np.concatenate(pred_parts, axis=0)

    pred = pred_flat.reshape(-1).astype(np.float64)
    true = true_flat.reshape(-1).astype(np.float64)
    rel_l2 = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
    pred_std = float(np.std(pred))
    true_std = float(np.std(true))
    std_ratio = float(pred_std / true_std) if true_std > 0 else np.nan
    return {
        "acc_corr": correlation(pred, true),
        "acc_rel_l2": rel_l2,
        "acc_pred_std": pred_std,
        "acc_ref_std": true_std,
        "acc_std_ratio": std_ratio,
        "acc_mean_bias": float(np.mean(pred - true)),
    }


def metric_score(metrics):
    """Rank acceleration models by relative error and scale mismatch."""
    std_ratio = max(metrics["acc_std_ratio"], 1e-30)
    return float(metrics["acc_rel_l2"] + abs(np.log(std_ratio)))


def write_metrics(path, rows):
    """Write metrics rows to TSV."""
    fields = [
        "iteration",
        "model_path",
        "score",
        "acc_corr",
        "acc_rel_l2",
        "acc_std_ratio",
        "acc_pred_std",
        "acc_ref_std",
        "acc_mean_bias",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_model(model, models_dir, iteration, metrics, args):
    """Save a model with acceleration metrics in the filename."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"acc_score_{metric_score(metrics):.6g}"
        f"_corr_{metrics['acc_corr']:.4g}"
        f"_rel_{metrics['acc_rel_l2']:.4g}"
        f"_ratio_{metrics['acc_std_ratio']:.4g}"
        f"_{args.rnn_type.lower()}_iter_{iteration}.pth"
    )
    path = models_dir / filename
    torch.save(model, path)
    return path


def main():
    """Train acceleration candidates and save metrics."""
    args = parse_args()
    if args.count_models <= 0:
        raise ValueError("count_models must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0 < args.data_len <= 1:
        raise ValueError("data-len must be in (0, 1]")

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    data = load_data(args.data_path)
    train_indices, validation_indices, blocks_per_window = choose_train_validation_indices(
        data,
        args.validation_windows,
        args.train_window_frames,
        args.train_window_count,
        rng,
    )
    X_train = data["X_blocks"][train_indices].astype(np.float32)
    y_train = data["y_blocks"][train_indices].astype(np.float32)
    X_validation = data["X_blocks"][validation_indices].astype(np.float32)
    y_validation = data["y_blocks"][validation_indices].astype(np.float32)

    print("ACCELERATION MODEL SEARCH")
    print("data", args.data_path)
    print("X_blocks", data["X_blocks"].shape)
    print("blocks_per_window", blocks_per_window)
    print("train_samples", X_train.shape[0])
    print("validation_samples", X_validation.shape[0])
    print("epochs", args.epochs, "batch_size", args.batch_size, "lr", args.learning_rate)

    metrics_path = (
        Path(args.metrics_path)
        if args.metrics_path is not None
        else Path(args.models_dir) / "acceleration_metrics.tsv"
    )
    rows = []
    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        model = build_model(args, data)
        model.train_crystal_blocks(X_train, y_train, data_len=args.data_len, target_mode="acceleration")
        metrics = evaluate_acceleration(model, X_validation, y_validation, args.batch_size)
        score = metric_score(metrics)
        print(
            "ACC_METRICS",
            f"score={score:.6g}",
            f"corr={metrics['acc_corr']:.6g}",
            f"rel_l2={metrics['acc_rel_l2']:.6g}",
            f"std_ratio={metrics['acc_std_ratio']:.6g}",
        )

        model_path = ""
        if args.save_all:
            model_path = str(save_model(model, args.models_dir, iteration, metrics, args))
            print(f"SAVED {model_path}")

        row = {
            "iteration": iteration,
            "model_path": model_path,
            "score": score,
            **metrics,
            "train_samples": X_train.shape[0],
            "validation_samples": X_validation.shape[0],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
        rows.append(row)
        write_metrics(metrics_path, rows)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
