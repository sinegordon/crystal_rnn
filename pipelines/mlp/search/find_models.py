#!/usr/bin/env python3
"""Train and rank standalone one-frame pair-energy MLP models."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from base_classes import CrystalPairEnergyMLPNet  # noqa: E402
from find_field_rnn_models import (  # noqa: E402
    DT,
    STEP,
    crystal_frames_to_flat_positions,
    dynamics_scale_metrics,
    get_sqw_default,
    save_sqw_plot,
    velocity_distribution_metrics,
)


def parse_args():
    """Parse model-search options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count_models", type=int)
    parser.add_argument("--data-path", default="data333_mlp_force.npz")
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--models-dir", default="models333_pair_energy_mlp")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--neighbor-shells", type=int, default=2)
    parser.add_argument("--cutoff-scale", type=float, default=1.05)
    parser.add_argument("--delta-frames", type=int, default=30000)
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "consecutive"],
        default="random",
        help="Choose independent random trajectory frames or one consecutive window.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--count-steps", type=int, default=2000)
    parser.add_argument("--count-run", type=int, default=3)
    parser.add_argument("--velocity-window-frames", type=int, default=10)
    parser.add_argument("--velocity-hist-bins", type=int, default=80)
    parser.add_argument("--velocity-score-weight", type=float, default=0.0)
    parser.add_argument("--reference-pressure-loss-weight", type=float, default=0.0)
    parser.add_argument("--reference-pressure-target", type=float, default=0.0)
    parser.add_argument("--reference-pressure-loss-scale", type=float, default=1.0)
    parser.add_argument("--save-threshold", type=float, default=3.5)
    parser.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-output-dir", default=None)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_data(path):
    """Load and validate a standalone MLP dataset."""
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    if "input_blocks" not in data and "X_blocks" in data:
        if data["X_blocks"].ndim != 7:
            raise ValueError("Legacy X_blocks must have a temporal axis")
        data["input_blocks"] = data["X_blocks"][:, -1]
        print("Using the latest frame from legacy X_blocks as one-frame MLP input")
    required = {
        "input_blocks",
        "force_acceleration_blocks",
        "displacements",
        "atom_order",
        "reference_positions",
        "box_lengths",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing MLP arrays in {path}: {missing}")
    if data["input_blocks"].ndim != 6:
        raise ValueError("input_blocks must not contain a temporal axis")
    return data


def select_training_samples(data, delta_frames, rng, sampling_mode):
    """Select aligned one-frame inputs and force targets by trajectory index."""
    frame_count = int(data["displacements"].shape[0])
    sample_windows = frame_count - 1
    sample_count = int(data["input_blocks"].shape[0])
    if sample_windows <= 0 or sample_count % sample_windows:
        raise ValueError("input_blocks count is inconsistent with displacement frames")
    blocks_per_frame = sample_count // sample_windows
    if sampling_mode == "random":
        selected_count = min(int(delta_frames), sample_windows)
        time_indices = np.sort(rng.choice(sample_windows, size=selected_count, replace=False))
        sample_indices = (
            time_indices[:, None] * blocks_per_frame + np.arange(blocks_per_frame)[None, :]
        ).reshape(-1)
        print(f"TRAIN RANDOM FRAMES count={selected_count} range={time_indices[0]}:{time_indices[-1] + 1}")
        print(f"TRAIN RANDOM BLOCKS count={len(sample_indices)}")
    elif sampling_mode == "consecutive":
        selected_frames = min(int(delta_frames), frame_count)
        start = 0 if selected_frames == frame_count else int(rng.integers(0, frame_count - selected_frames + 1))
        usable_frames = selected_frames - 1
        sample_start = start * blocks_per_frame
        sample_stop = sample_start + usable_frames * blocks_per_frame
        sample_indices = np.arange(sample_start, sample_stop)
        print(f"TRAIN CONSECUTIVE FRAMES {start}:{start + selected_frames}")
        print(f"TRAIN CONSECUTIVE BLOCKS {sample_start}:{sample_stop}")
    else:  # pragma: no cover - guarded by argparse.
        raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")
    return (
        data["input_blocks"][sample_indices],
        data["force_acceleration_blocks"][sample_indices],
    )


def evaluate_model(model, data, count_steps, count_run, rng, velocity_window_frames, velocity_hist_bins):
    """Evaluate rollout S(q,w), velocity, and acceleration scales."""
    displacements = data["displacements"]
    if len(displacements) < count_steps + 2:
        raise ValueError("Evaluation trajectory is too short for count_steps")
    reference_positions = data["reference_positions"]
    atom_order = data["atom_order"]
    sqw_norm = 0.0
    reference_sqw_sum = predicted_sqw_sum = None
    velocity_sum = scale_sum = None
    xi_ref = yi_ref = xi_pred = yi_pred = None

    for _ in range(int(count_run)):
        start = int(rng.integers(2, len(displacements) - count_steps + 1))
        initial = displacements[start - 2 : start]
        reference = displacements[start : start + count_steps]
        predicted = model.rollout(count_steps, initial, periodic=True)
        reference_coords = crystal_frames_to_flat_positions(reference, reference_positions, atom_order)
        predicted_coords = crystal_frames_to_flat_positions(predicted, reference_positions, atom_order)
        xi_ref, yi_ref, reference_sqw = get_sqw_default(reference_coords, DT, STEP)
        xi_pred, yi_pred, predicted_sqw = get_sqw_default(predicted_coords, DT, STEP)
        sqw_norm += float(np.linalg.norm(predicted_sqw - reference_sqw))
        reference_sqw_sum = reference_sqw.copy() if reference_sqw_sum is None else reference_sqw_sum + reference_sqw
        predicted_sqw_sum = predicted_sqw.copy() if predicted_sqw_sum is None else predicted_sqw_sum + predicted_sqw
        velocity = velocity_distribution_metrics(
            predicted,
            reference,
            dt=1.0,
            window_frames=velocity_window_frames,
            bins=velocity_hist_bins,
        )
        scale = dynamics_scale_metrics(predicted, reference, window_frames=velocity_window_frames)
        velocity_sum = {key: 0.0 for key in velocity} if velocity_sum is None else velocity_sum
        scale_sum = {key: 0.0 for key in scale} if scale_sum is None else scale_sum
        for key, value in velocity.items():
            velocity_sum[key] += float(value)
        for key, value in scale.items():
            scale_sum[key] += float(value)

    divisor = float(count_run)
    metrics = {key: value / divisor for key, value in velocity_sum.items()}
    metrics.update({key: value / divisor for key, value in scale_sum.items()})
    return (
        sqw_norm / divisor,
        metrics,
        xi_ref,
        yi_ref,
        reference_sqw_sum / divisor,
        xi_pred,
        yi_pred,
        predicted_sqw_sum / divisor,
    )


def save_metrics(path, rows):
    """Write candidate metrics as TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Train, evaluate, plot, and save independent MLP candidates."""
    args = parse_args()
    if args.count_models <= 0 or args.delta_frames <= 1 or args.count_steps <= 2:
        raise ValueError("count_models, delta_frames, and count_steps must be positive and usable")
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    data = load_data(args.data_path)
    eval_data = data if args.eval_data_path is None else load_data(args.eval_data_path)
    models_dir = Path(args.models_dir)
    plot_dir = Path(args.plot_output_dir or models_dir)
    metrics_path = Path(args.metrics_path or models_dir / "metrics.tsv")
    rows = []

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}", flush=True)
        model = CrystalPairEnergyMLPNet(
            reference_positions=data["reference_positions"],
            atom_order=data["atom_order"],
            box_lengths=data["box_lengths"],
            size=args.size,
            neighbor_shells=args.neighbor_shells,
            cutoff_scale=args.cutoff_scale,
            device=args.device,
        )
        model.reference_pressure_loss_weight = args.reference_pressure_loss_weight
        model.reference_pressure_target = args.reference_pressure_target
        model.reference_pressure_loss_scale = args.reference_pressure_loss_scale
        inputs, targets = select_training_samples(
            data,
            args.delta_frames,
            rng,
            args.sampling_mode,
        )
        losses = model.fit(
            inputs,
            targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        result = evaluate_model(
            model,
            eval_data,
            args.count_steps,
            args.count_run,
            rng,
            args.velocity_window_frames,
            args.velocity_hist_bins,
        )
        sqw_norm, metrics, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_pred = result
        selection_score = sqw_norm + args.velocity_score_weight * metrics["velocity_score"]
        stem = f"mean_norm_{sqw_norm:.6f}_pair_energy_mlp_s{args.size}_iter{iteration:03d}"
        model_path = ""
        if args.save_all or selection_score < args.save_threshold:
            model_path = str(model.save(models_dir / f"{stem}.pth"))
        if args.plot_all:
            save_sqw_plot(
                xi_ref,
                yi_ref,
                jlp_ref,
                xi_pred,
                yi_pred,
                jlp_pred,
                plot_dir / f"{stem}_sqw.png",
                title=f"Pair-energy MLP iter {iteration}, S(q,w) norm = {sqw_norm:.6g}",
                show=args.show_plots,
            )
        row = {
            "iteration": iteration,
            "model_path": model_path,
            "sqw_norm": sqw_norm,
            "selection_score": selection_score,
            "velocity_score": metrics["velocity_score"],
            "velocity_rms_ratio": metrics["velocity_rms_ratio"],
            "velocity_end_rms_ratio": metrics["velocity_end_rms_ratio"],
            "acceleration_rms_ratio": metrics["acceleration_rms_ratio"],
            "acceleration_end_rms_ratio": metrics["acceleration_end_rms_ratio"],
            "final_train_loss": losses[-1],
            "best_train_loss": min(losses),
            "size": args.size,
            "parameter_count": model.parameter_count,
            "delta_frames": args.delta_frames,
            "sampling_mode": args.sampling_mode,
            "seed": "" if args.seed is None else args.seed,
        }
        rows.append(row)
        save_metrics(metrics_path, rows)
        print(f"CURRENT_NORM = {sqw_norm}")
        print(f"SELECTION_SCORE = {selection_score}")
        print(f"MODEL_PATH = {model_path}", flush=True)

    rows.sort(key=lambda row: float(row["selection_score"]))
    save_metrics(metrics_path, rows)
    print("TOP MODELS")
    for row in rows[:10]:
        print(f"{float(row['selection_score']):.6f}\t{row['model_path']}")


if __name__ == "__main__":
    main()
