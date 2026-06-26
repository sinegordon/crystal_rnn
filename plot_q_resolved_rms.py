"""Plot q-resolved RMS amplitudes for crystal displacements and accelerations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE trajectory npz with flat positions.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference npz.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics output.")
    parser.add_argument("--q-decimals", type=int, default=8, help="Decimal rounding used to group |q| shells.")
    parser.add_argument(
        "--exclude-q-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude q=0 from plots and summary metrics.",
    )
    parser.add_argument(
        "--title",
        default="q-resolved RMS displacement and acceleration",
        help="Figure title.",
    )
    return parser.parse_args()


def as_cell_matrix(value):
    """Return a 3x3 cell matrix."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.ndim == 2 and cell.shape[1] == 3 and cell.shape[0] != 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,) or (3, 3)")
    return cell


def cell_matrix(data, ase):
    """Return the simulation cell matrix from ASE output or source dataset."""
    if "cell" in ase.files:
        return as_cell_matrix(ase["cell"])
    if "cell" in data.files:
        return as_cell_matrix(data["cell"])
    if "box_lengths" in data.files:
        return as_cell_matrix(data["box_lengths"])
    raise ValueError("No cell or box_lengths found")


def minimum_image(delta, cell):
    """Wrap flat position differences into the nearest periodic image."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(delta, dtype=np.float64) @ inverse_cell
    fractional -= np.round(fractional)
    return fractional @ cell


def flat_to_crystal(flat_values, atom_order):
    """Convert flat ASE atom order values into crystal layout."""
    flat_values = np.asarray(flat_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    return flat_values[:, atom_order.reshape(-1), :].reshape(flat_values.shape[0], *atom_order.shape, 3)


def load_overlapping_displacements(ase_path, data_path):
    """Load overlapping ASE/reference displacement fields in crystal layout."""
    ase = np.load(ase_path)
    data = np.load(data_path)
    if "predicted_displacements" in ase.files and "reference_displacements" in ase.files:
        predicted = np.asarray(ase["predicted_displacements"], dtype=np.float64)
        reference = np.asarray(ase["reference_displacements"], dtype=np.float64)
        frames = min(int(predicted.shape[0]), int(reference.shape[0]))
        if frames < 3:
            raise ValueError(f"Need at least three overlapping direct-inference frames, got {frames}")
        if frames < predicted.shape[0] or frames < reference.shape[0]:
            print(
                f"WARNING: trimming direct-inference arrays to {frames} frames: "
                f"predicted={predicted.shape[0]}, reference={reference.shape[0]}."
            )
        start = int(ase["prediction_start_frame"]) if "prediction_start_frame" in ase.files else 0
        return predicted[:frames], reference[:frames], cell_matrix(data, ase), start

    positions = np.asarray(ase["positions"], dtype=np.float64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0
    available_reference_frames = max(0, int(data["displacements"].shape[0]) - start)
    frames = min(int(positions.shape[0]), available_reference_frames)
    if frames < 3:
        raise ValueError(
            f"Need at least three overlapping frames, got {frames}: "
            f"positions={positions.shape[0]}, reference_available={available_reference_frames}, start={start}"
        )
    if frames < positions.shape[0]:
        print(
            f"WARNING: trimming trajectory from {positions.shape[0]} to {frames} frames "
            f"because reference data ends at frame {int(data['displacements'].shape[0]) - 1}."
        )

    cell = cell_matrix(data, ase)
    predicted_flat = minimum_image(positions[:frames] - reference_positions[None, :, :], cell)
    predicted = flat_to_crystal(predicted_flat, atom_order)
    reference = np.asarray(data["displacements"][start : start + frames], dtype=np.float64)
    return predicted, reference, cell, start


def discrete_acceleration(displacements):
    """Return model-scale discrete accelerations."""
    return displacements[2:] - 2.0 * displacements[1:-1] + displacements[:-2]


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


def shell_average(q_abs, rms_values, decimals, exclude_q_zero):
    """Average grid-point RMS amplitudes over equal-|q| shells."""
    q_flat = np.asarray(q_abs, dtype=np.float64).reshape(-1)
    rms_flat = np.asarray(rms_values, dtype=np.float64).reshape(-1)
    if exclude_q_zero:
        mask = q_flat > 0
        q_flat = q_flat[mask]
        rms_flat = rms_flat[mask]

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


def ratio(numerator, denominator):
    """Return a safe ratio."""
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator > 0:
        return numerator / denominator
    return np.nan


def rows_to_arrays(predicted_rows, reference_rows):
    """Return common q and predicted/reference shell RMS arrays."""
    if len(predicted_rows) != len(reference_rows):
        raise ValueError("Predicted/reference q shell counts differ")
    q = np.asarray([row["q"] for row in predicted_rows], dtype=np.float64)
    pred = np.asarray([row["rms"] for row in predicted_rows], dtype=np.float64)
    ref = np.asarray([row["rms"] for row in reference_rows], dtype=np.float64)
    if not np.allclose(q, [row["q"] for row in reference_rows]):
        raise ValueError("Predicted/reference q shell grids differ")
    return q, pred, ref


def plot_quantity(ax_value, ax_ratio, q, predicted, reference, ylabel, title):
    """Plot shell RMS values and ASE/reference ratio."""
    ax_value.plot(q, reference, "o-", label="reference", color="tab:blue")
    ax_value.plot(q, predicted, "o-", label="inference", color="tab:orange")
    ax_value.set_title(title)
    ax_value.set_ylabel(ylabel)
    ax_value.grid(alpha=0.25)
    ax_value.legend(fontsize=9)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = predicted / reference
    ax_ratio.axhline(1.0, color="black", linewidth=1, alpha=0.5)
    ax_ratio.plot(q, ratio_values, "o-", color="tab:green")
    ax_ratio.set_xlabel("|q|, 1/A")
    ax_ratio.set_ylabel("inference/reference")
    ax_ratio.grid(alpha=0.25)


def metric_lines(name, predicted_rows, reference_rows):
    """Return TSV metric lines for one quantity."""
    lines = []
    for pred, ref in zip(predicted_rows, reference_rows):
        lines.append(
            f"{name}\t{pred['q']:.10g}\t{pred['count']}\t"
            f"{pred['rms']:.10g}\t{ref['rms']:.10g}\t{ratio(pred['rms'], ref['rms']):.10g}\t"
            f"{pred['std']:.10g}\t{ref['std']:.10g}"
        )
    return lines


def main():
    """Compute q-resolved RMS amplitudes and save plots/metrics."""
    args = parse_args()
    predicted_u, reference_u, cell, start = load_overlapping_displacements(args.ase_path, args.data_path)
    predicted_a = discrete_acceleration(predicted_u)
    reference_a = discrete_acceleration(reference_u)
    q_abs = q_grid(predicted_u.shape[1:4], cell)

    predicted_u_rows = shell_average(
        q_abs,
        fft_rms_by_q(predicted_u),
        decimals=args.q_decimals,
        exclude_q_zero=args.exclude_q_zero,
    )
    reference_u_rows = shell_average(
        q_abs,
        fft_rms_by_q(reference_u),
        decimals=args.q_decimals,
        exclude_q_zero=args.exclude_q_zero,
    )
    predicted_a_rows = shell_average(
        q_abs,
        fft_rms_by_q(predicted_a),
        decimals=args.q_decimals,
        exclude_q_zero=args.exclude_q_zero,
    )
    reference_a_rows = shell_average(
        q_abs,
        fft_rms_by_q(reference_a),
        decimals=args.q_decimals,
        exclude_q_zero=args.exclude_q_zero,
    )

    q_u, pred_u, ref_u = rows_to_arrays(predicted_u_rows, reference_u_rows)
    q_a, pred_a, ref_a = rows_to_arrays(predicted_a_rows, reference_a_rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", constrained_layout=True)
    plot_quantity(
        axes[0, 0],
        axes[1, 0],
        q_u,
        pred_u,
        ref_u,
        "RMS |u_q|, A",
        "q-resolved displacement RMS",
    )
    plot_quantity(
        axes[0, 1],
        axes[1, 1],
        q_a,
        pred_a,
        ref_a,
        "RMS |a_q|, A/step^2",
        "q-resolved discrete acceleration RMS",
    )
    fig.suptitle(args.title, fontsize=15)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Saved {output_path}",
        f"reference_frame_start = {start}",
        f"comparison_frames = {predicted_u.shape[0]}",
        f"acceleration_frames = {predicted_a.shape[0]}",
        "quantity\tq_abs\tq_shell_count\tinference_rms\treference_rms\tratio\tinference_shell_std\treference_shell_std",
        *metric_lines("displacement", predicted_u_rows, reference_u_rows),
        *metric_lines("discrete_acceleration", predicted_a_rows, reference_a_rows),
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
