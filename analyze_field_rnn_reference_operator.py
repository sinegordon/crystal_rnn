"""Compare a FieldRNN acceleration operator on reference trajectory frames.

This diagnostic does not run an autoregressive rollout.  For each selected
reference target frame it feeds the preceding history frames to the model and
compares the predicted local acceleration field with the finite-difference
reference acceleration from the same trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from infer_field_rnn_centered_acceleration import (
    build_centers,
    predict_center_accelerations,
    resolve_device,
    shape3,
)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Saved FieldRNN acceleration model.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal .npz dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic plots and metrics.")
    parser.add_argument("--output-npz", default=None, help="Optional output npz with sampled acceleration arrays.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help=(
            "First target frame index.  The model history is taken from frames "
            "start-frame-sequence_length ... start-frame-1.  Defaults to sequence_length."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of target frames to evaluate.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride between evaluated target frames.")
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap local patches around crystal boundaries.",
    )
    parser.add_argument("--patch-batch-size", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--q-decimals", type=int, default=8)
    parser.add_argument("--hist-bins", type=int, default=120)
    parser.add_argument("--hist-percentile", type=float, default=99.5)
    parser.add_argument("--title", default="FieldRNN operator on reference trajectory")
    return parser.parse_args()


def load_data(path):
    """Load arrays needed for operator diagnostics."""
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return data


def cell_matrix(data):
    """Return an orthorhombic cell matrix from the prepared dataset."""
    if "cell" in data.files:
        cell = np.asarray(data["cell"], dtype=np.float64)
        if cell.ndim == 3:
            cell = cell[0]
        if cell.shape == (3,):
            return np.diag(cell)
        if cell.shape == (3, 3):
            return cell
    if "box_lengths" in data.files:
        box_lengths = np.asarray(data["box_lengths"], dtype=np.float64)
        if box_lengths.ndim == 2:
            box_lengths = box_lengths[0]
        return np.diag(box_lengths)
    raise ValueError("Dataset must contain either cell or box_lengths")


def target_frame_indices(frame_count, sequence_length, start_frame, max_frames, frame_stride):
    """Return target frame indices to evaluate."""
    if frame_stride <= 0:
        raise ValueError("frame-stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max-frames must be positive when provided")

    first = sequence_length if start_frame is None else int(start_frame)
    if first < sequence_length:
        raise ValueError("start-frame must be at least sequence_length")
    if first >= frame_count:
        raise ValueError("start-frame is outside the trajectory")

    indices = np.arange(first, frame_count, int(frame_stride), dtype=np.int64)
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    if indices.size == 0:
        raise ValueError("No target frames selected")
    return indices


def reference_acceleration(displacements, target_frame):
    """Return finite-difference acceleration for a target frame."""
    return (
        displacements[target_frame]
        - 2.0 * displacements[target_frame - 1]
        + displacements[target_frame - 2]
    ).astype(np.float32)


def evaluate_operator(model, data, args):
    """Evaluate model accelerations on selected reference histories."""
    displacements = np.asarray(data["displacements"], dtype=np.float32)
    sequence_length = int(data["X_blocks"].shape[1])
    target_frames = target_frame_indices(
        frame_count=int(displacements.shape[0]),
        sequence_length=sequence_length,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )

    patch_shape = shape3("patch_shape", args.patch_shape)
    crystal_shape = tuple(int(dim) for dim in displacements.shape[1:4])
    centers = build_centers(crystal_shape, patch_shape, bool(args.periodic))

    predicted = []
    reference = []
    input_displacements = []
    for output_index, target_frame in enumerate(target_frames):
        history = displacements[target_frame - sequence_length : target_frame]
        predicted.append(
            predict_center_accelerations(
                model=model,
                history=history,
                centers=centers,
                patch_shape=patch_shape,
                periodic=bool(args.periodic),
                patch_batch_size=args.patch_batch_size,
                device=args.device,
            )
        )
        reference.append(reference_acceleration(displacements, int(target_frame)))
        input_displacements.append(displacements[target_frame - 1])
        if output_index == 0 or (output_index + 1) % 100 == 0 or output_index + 1 == len(target_frames):
            print(
                f"FRAME {output_index + 1}/{len(target_frames)} target={int(target_frame)}",
                flush=True,
            )

    return (
        np.asarray(predicted, dtype=np.float32),
        np.asarray(reference, dtype=np.float32),
        np.asarray(input_displacements, dtype=np.float32),
        target_frames,
        sequence_length,
    )


def q_grid(shape, cell):
    """Return |q| values in FFT index order for a rectangular supercell."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=np.float64), axis=1)
    q_axes = [2.0 * np.pi * np.fft.fftfreq(int(n), d=float(length) / int(n)) for n, length in zip(shape, lengths)]
    qx, qy, qz = np.meshgrid(q_axes[0], q_axes[1], q_axes[2], indexing="ij")
    return np.sqrt(qx**2 + qy**2 + qz**2)


def fft_rms_by_q(values):
    """Return per-grid-point RMS amplitude after spatial FFT over crystal cells."""
    values = np.asarray(values, dtype=np.float64)
    cell_count = int(np.prod(values.shape[1:4]))
    spectrum = np.fft.fftn(values, axes=(1, 2, 3)) / np.sqrt(cell_count)
    return np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=(0, 4, 5)))


def shell_average(q_abs, rms_values, decimals):
    """Average grid-point RMS amplitudes over equal-|q| shells."""
    q_flat = np.asarray(q_abs, dtype=np.float64).reshape(-1)
    rms_flat = np.asarray(rms_values, dtype=np.float64).reshape(-1)
    rounded = np.round(q_flat, decimals=int(decimals))
    rows = []
    for value in np.unique(rounded):
        mask = rounded == value
        rows.append(
            {
                "q": float(np.mean(q_flat[mask])),
                "rms": float(np.mean(rms_flat[mask])),
                "std": float(np.std(rms_flat[mask])),
                "count": int(np.count_nonzero(mask)),
            }
        )
    rows.sort(key=lambda row: row["q"])
    return rows


def rows_to_arrays(*row_groups):
    """Return common q values and RMS arrays from shell rows."""
    q = np.asarray([row["q"] for row in row_groups[0]], dtype=np.float64)
    arrays = []
    for rows in row_groups:
        q_current = np.asarray([row["q"] for row in rows], dtype=np.float64)
        if not np.allclose(q, q_current):
            raise ValueError("q shell grids differ")
        arrays.append(np.asarray([row["rms"] for row in rows], dtype=np.float64))
    return q, arrays


def safe_ratio(numerator, denominator):
    """Return a safe scalar ratio."""
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator > 0:
        return numerator / denominator
    return np.nan


def correlation(a, b):
    """Return Pearson correlation for flattened arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("correlation arrays must have the same size")
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def rms(values):
    """Return root-mean-square value."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def plot_q_resolved(output_path, title, q, input_u, predicted_a, reference_a):
    """Save q-resolved displacement and acceleration operator diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(q, input_u, "o-", color="tab:purple", label="reference input")
    axes[0, 0].set_title("q-resolved input displacement RMS")
    axes[0, 0].set_ylabel("RMS |u_q|, A")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)

    axes[1, 0].plot(q, input_u / np.nanmax(input_u), "o-", color="tab:purple")
    axes[1, 0].set_xlabel("|q|, 1/A")
    axes[1, 0].set_ylabel("normalized RMS")
    axes[1, 0].grid(alpha=0.25)

    axes[0, 1].plot(q, reference_a, "o-", color="tab:blue", label="reference")
    axes[0, 1].plot(q, predicted_a, "o-", color="tab:orange", label="model operator")
    axes[0, 1].set_title("q-resolved one-step acceleration RMS")
    axes[0, 1].set_ylabel("RMS |a_q|, A/step^2")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=9)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = predicted_a / reference_a
    axes[1, 1].axhline(1.0, color="black", linewidth=1, alpha=0.5)
    axes[1, 1].plot(q, ratio, "o-", color="tab:green")
    axes[1, 1].set_xlabel("|q|, 1/A")
    axes[1, 1].set_ylabel("model/reference")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def robust_symmetric_range(*arrays, percentile):
    """Return a symmetric histogram range around zero."""
    values = np.concatenate([np.abs(np.asarray(array, dtype=np.float64).reshape(-1)) for array in arrays])
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def robust_positive_range(*arrays, percentile):
    """Return a positive histogram range."""
    values = np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return 0.0, limit


def acceleration_magnitudes(values):
    """Return per-atom acceleration magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def plot_acceleration_histograms(output_path, title, predicted, reference, bins, percentile):
    """Save acceleration component and magnitude histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    component_range = robust_symmetric_range(predicted, reference, percentile=percentile)
    axes[0].hist(
        reference.reshape(-1),
        bins=bins,
        range=component_range,
        density=True,
        alpha=0.48,
        label="reference",
        color="tab:blue",
    )
    axes[0].hist(
        predicted.reshape(-1),
        bins=bins,
        range=component_range,
        density=True,
        alpha=0.48,
        label="model operator",
        color="tab:orange",
    )
    axes[0].set_title("discrete acceleration components")
    axes[0].set_xlabel("A/step^2")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=9)

    predicted_magnitude = acceleration_magnitudes(predicted)
    reference_magnitude = acceleration_magnitudes(reference)
    magnitude_range = robust_positive_range(predicted_magnitude, reference_magnitude, percentile=percentile)
    axes[1].hist(
        reference_magnitude,
        bins=bins,
        range=magnitude_range,
        density=True,
        alpha=0.48,
        label="reference",
        color="tab:blue",
    )
    axes[1].hist(
        predicted_magnitude,
        bins=bins,
        range=magnitude_range,
        density=True,
        alpha=0.48,
        label="model operator",
        color="tab:orange",
    )
    axes[1].set_title("discrete acceleration magnitudes")
    axes[1].set_xlabel("A/step^2")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=9)

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def metric_lines(input_rows, predicted_rows, reference_rows):
    """Return TSV metric lines for q-resolved diagnostics."""
    lines = [
        "quantity\tq_abs\tq_shell_count\tinput_displacement_rms\tmodel_acceleration_rms\treference_acceleration_rms\tacceleration_ratio\tinput_shell_std\tmodel_shell_std\treference_shell_std"
    ]
    for input_row, predicted_row, reference_row in zip(input_rows, predicted_rows, reference_rows):
        lines.append(
            f"operator\t{input_row['q']:.10g}\t{input_row['count']}\t"
            f"{input_row['rms']:.10g}\t{predicted_row['rms']:.10g}\t{reference_row['rms']:.10g}\t"
            f"{safe_ratio(predicted_row['rms'], reference_row['rms']):.10g}\t"
            f"{input_row['std']:.10g}\t{predicted_row['std']:.10g}\t{reference_row['std']:.10g}"
        )
    return lines


def write_metrics(path, lines):
    """Write text metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Run the reference-operator diagnostic."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.patch_batch_size <= 0:
        raise ValueError("patch-batch-size must be positive")

    data = load_data(args.data_path)
    device = resolve_device(args.device)
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if getattr(model, "target_mode", None) != "acceleration":
        raise ValueError("This diagnostic expects a target_mode='acceleration' model")
    if hasattr(model, "to"):
        model.to(device)
    args.device = device

    predicted_a, reference_a, input_u, target_frames, sequence_length = evaluate_operator(model, data, args)
    cell = cell_matrix(data)
    q_abs = q_grid(input_u.shape[1:4], cell)
    input_rows = shell_average(q_abs, fft_rms_by_q(input_u), args.q_decimals)
    predicted_rows = shell_average(q_abs, fft_rms_by_q(predicted_a), args.q_decimals)
    reference_rows = shell_average(q_abs, fft_rms_by_q(reference_a), args.q_decimals)
    q, arrays = rows_to_arrays(input_rows, predicted_rows, reference_rows)
    input_u_q, predicted_a_q, reference_a_q = arrays

    q_plot_path = output_dir / "field_rnn_reference_operator_q_resolved.png"
    hist_plot_path = output_dir / "field_rnn_reference_operator_acceleration_histograms.png"
    q_metrics_path = output_dir / "field_rnn_reference_operator_q_resolved.txt"
    hist_metrics_path = output_dir / "field_rnn_reference_operator_acceleration_histograms.txt"
    plot_q_resolved(q_plot_path, args.title, q, input_u_q, predicted_a_q, reference_a_q)
    plot_acceleration_histograms(
        hist_plot_path,
        args.title,
        predicted_a,
        reference_a,
        bins=args.hist_bins,
        percentile=args.hist_percentile,
    )

    global_metrics = [
        f"model_path = {args.model_path}",
        f"data_path = {args.data_path}",
        f"device = {device}",
        f"sequence_length = {sequence_length}",
        f"target_frame_start = {int(target_frames[0])}",
        f"target_frame_stop = {int(target_frames[-1]) + 1}",
        f"target_frame_count = {len(target_frames)}",
        f"frame_stride = {args.frame_stride}",
        f"patch_shape = {tuple(args.patch_shape)}",
        f"periodic = {bool(args.periodic)}",
        f"model_acceleration_component_rms = {rms(predicted_a):.10g}",
        f"reference_acceleration_component_rms = {rms(reference_a):.10g}",
        f"acceleration_component_rms_ratio = {safe_ratio(rms(predicted_a), rms(reference_a)):.10g}",
        f"acceleration_component_correlation = {correlation(predicted_a, reference_a):.10g}",
        "",
    ]
    q_lines = [
        f"Saved {q_plot_path}",
        *global_metrics,
        *metric_lines(input_rows, predicted_rows, reference_rows),
    ]
    hist_lines = [
        f"Saved {hist_plot_path}",
        *global_metrics,
    ]
    write_metrics(q_metrics_path, q_lines)
    write_metrics(hist_metrics_path, hist_lines)
    print("\n".join(q_lines))

    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            predicted_accelerations=predicted_a,
            reference_accelerations=reference_a,
            input_displacements=input_u,
            target_frames=target_frames,
            sequence_length=np.asarray(sequence_length, dtype=np.int64),
            model_path=np.asarray(str(args.model_path)),
            data_path=np.asarray(str(args.data_path)),
            device=np.asarray(str(device)),
        )
        print(f"Saved {output_npz}")


if __name__ == "__main__":
    main()
