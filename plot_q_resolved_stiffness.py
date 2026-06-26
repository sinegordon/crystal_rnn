"""Compare q-resolved displacement, acceleration, and effective stiffness."""

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
    parser.add_argument("--chunk-frames", type=int, default=256, help="Frames processed per FFT chunk.")
    parser.add_argument("--q-decimals", type=int, default=8, help="Decimal rounding used to group |q| shells.")
    parser.add_argument(
        "--max-inference-frames",
        type=int,
        default=None,
        help="Optional cap for ASE inference frames. By default all frames are used.",
    )
    parser.add_argument(
        "--max-reference-frames",
        type=int,
        default=None,
        help="Optional cap for reference frames. By default all available frames are used.",
    )
    parser.add_argument(
        "--exclude-q-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude q=0 from plots and shell metrics.",
    )
    parser.add_argument(
        "--title",
        default="q-resolved RMS amplitudes and effective stiffness",
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
        raise ValueError("cell must have shape (3,), (3, 3), or frame-stacked cell matrices")
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
    """Convert flat atom-order values into crystal layout."""
    flat_values = np.asarray(flat_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    return flat_values[:, atom_order.reshape(-1), :].reshape(flat_values.shape[0], *atom_order.shape, 3)


def q_grid(shape, cell):
    """Return |q| values in FFT index order for a rectangular supercell."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=np.float64), axis=1)
    q_axes = [2.0 * np.pi * np.fft.fftfreq(int(n), d=float(length) / int(n)) for n, length in zip(shape, lengths)]
    qx, qy, qz = np.meshgrid(q_axes[0], q_axes[1], q_axes[2], indexing="ij")
    return np.sqrt(qx**2 + qy**2 + qz**2)


def empty_accumulators(shape):
    """Create spectral accumulators for one trajectory."""
    return {
        "u_power": np.zeros(shape, dtype=np.float64),
        "a_power": np.zeros(shape, dtype=np.float64),
        "stiffness_num": np.zeros(shape, dtype=np.float64),
        "stiffness_den": np.zeros(shape, dtype=np.float64),
        "u_frames": 0,
        "a_frames": 0,
    }


def accumulate_power(accumulators, crystal_values):
    """Accumulate q-resolved displacement power."""
    cell_count = int(np.prod(crystal_values.shape[1:4]))
    spectrum = np.fft.fftn(crystal_values, axes=(1, 2, 3)) / np.sqrt(cell_count)
    accumulators["u_power"] += np.sum(np.abs(spectrum) ** 2, axis=(0, 4, 5))
    accumulators["u_frames"] += int(crystal_values.shape[0])


def accumulate_acceleration_and_stiffness(accumulators, crystal_values):
    """Accumulate q-resolved acceleration power and effective stiffness terms."""
    cell_count = int(np.prod(crystal_values.shape[1:4]))
    centered_u = crystal_values[1:-1]
    acceleration = crystal_values[2:] - 2.0 * centered_u + crystal_values[:-2]
    u_spectrum = np.fft.fftn(centered_u, axes=(1, 2, 3)) / np.sqrt(cell_count)
    a_spectrum = np.fft.fftn(acceleration, axes=(1, 2, 3)) / np.sqrt(cell_count)
    accumulators["a_power"] += np.sum(np.abs(a_spectrum) ** 2, axis=(0, 4, 5))
    accumulators["stiffness_num"] += np.sum(-np.real(a_spectrum * np.conj(u_spectrum)), axis=(0, 4, 5))
    accumulators["stiffness_den"] += np.sum(np.abs(u_spectrum) ** 2, axis=(0, 4, 5))
    accumulators["a_frames"] += int(acceleration.shape[0])


def finalize_accumulators(accumulators, atoms_per_cell, epsilon=1e-30):
    """Convert raw accumulators to RMS amplitudes and effective stiffness."""
    component_count = int(atoms_per_cell) * 3
    if accumulators["u_frames"] <= 0:
        raise ValueError("No displacement frames were accumulated")
    if accumulators["a_frames"] <= 0:
        raise ValueError("Need at least three frames to accumulate accelerations")
    u_samples = int(accumulators["u_frames"]) * component_count
    a_samples = int(accumulators["a_frames"]) * component_count
    return {
        "u_rms": np.sqrt(accumulators["u_power"] / float(u_samples)),
        "a_rms": np.sqrt(accumulators["a_power"] / float(a_samples)),
        "stiffness": accumulators["stiffness_num"] / (accumulators["stiffness_den"] + float(epsilon)),
        "u_frames": int(accumulators["u_frames"]),
        "a_frames": int(accumulators["a_frames"]),
    }


def compute_q_diagnostics(frame_count, get_crystal_chunk, crystal_shape, chunk_frames):
    """Compute q-resolved diagnostics by streaming crystal displacement chunks."""
    if frame_count < 3:
        raise ValueError(f"Need at least three frames, got {frame_count}")
    if chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    accumulators = empty_accumulators(tuple(crystal_shape[:3]))

    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        accumulate_power(accumulators, get_crystal_chunk(start, stop))

    acceleration_count = frame_count - 2
    for start in range(0, acceleration_count, chunk_frames):
        stop = min(acceleration_count, start + chunk_frames)
        accumulate_acceleration_and_stiffness(accumulators, get_crystal_chunk(start, stop + 2))

    return finalize_accumulators(accumulators, atoms_per_cell=int(crystal_shape[3]))


def shell_average(q_abs, values, decimals, exclude_q_zero):
    """Average grid-point values over equal-|q| shells."""
    q_flat = np.asarray(q_abs, dtype=np.float64).reshape(-1)
    value_flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(value_flat)
    if exclude_q_zero:
        finite &= q_flat > 0
    q_flat = q_flat[finite]
    value_flat = value_flat[finite]

    rounded = np.round(q_flat, decimals=int(decimals))
    rows = []
    for value in np.unique(rounded):
        mask = rounded == value
        rows.append(
            {
                "q": float(np.mean(q_flat[mask])),
                "value": float(np.mean(value_flat[mask])),
                "std": float(np.std(value_flat[mask])),
                "count": int(np.count_nonzero(mask)),
            }
        )
    rows.sort(key=lambda row: row["q"])
    return rows


def rows_to_arrays(inference_rows, reference_rows):
    """Return common q and inference/reference arrays."""
    if len(inference_rows) != len(reference_rows):
        raise ValueError("Inference/reference q shell counts differ")
    q = np.asarray([row["q"] for row in inference_rows], dtype=np.float64)
    reference_q = np.asarray([row["q"] for row in reference_rows], dtype=np.float64)
    if not np.allclose(q, reference_q):
        raise ValueError("Inference/reference q shell grids differ")
    inference = np.asarray([row["value"] for row in inference_rows], dtype=np.float64)
    reference = np.asarray([row["value"] for row in reference_rows], dtype=np.float64)
    return q, inference, reference


def safe_ratio(numerator, denominator):
    """Return a finite ratio where possible."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[~np.isfinite(result)] = np.nan
    return result


def plot_quantity(ax_value, ax_ratio, q, inference, reference, ylabel, title):
    """Plot shell values and inference/reference ratios."""
    ax_value.plot(q, reference, "o-", label="reference", color="tab:blue", markersize=3)
    ax_value.plot(q, inference, "o-", label="inference", color="tab:orange", markersize=3)
    ax_value.set_title(title)
    ax_value.set_ylabel(ylabel)
    ax_value.grid(alpha=0.25)
    ax_value.legend(fontsize=8)

    ax_ratio.axhline(1.0, color="black", linewidth=1, alpha=0.5)
    ax_ratio.plot(q, safe_ratio(inference, reference), "o-", color="tab:green", markersize=3)
    ax_ratio.set_xlabel("|q|, 1/A")
    ax_ratio.set_ylabel("inference/reference")
    ax_ratio.grid(alpha=0.25)


def metric_lines(name, inference_rows, reference_rows):
    """Return TSV metric lines for one q-resolved quantity."""
    lines = []
    for inference, reference in zip(inference_rows, reference_rows):
        ratio = np.nan
        if reference["value"] != 0:
            ratio = inference["value"] / reference["value"]
        lines.append(
            f"{name}\t{inference['q']:.10g}\t{inference['count']}\t"
            f"{inference['value']:.10g}\t{reference['value']:.10g}\t{ratio:.10g}\t"
            f"{inference['std']:.10g}\t{reference['std']:.10g}"
        )
    return lines


def make_position_chunker(positions, reference_positions, atom_order, cell):
    """Create a chunk loader converting flat ASE positions to crystal displacements."""
    def get_chunk(start, stop):
        flat = minimum_image(positions[start:stop] - reference_positions[None, :, :], cell)
        return flat_to_crystal(flat, atom_order)

    return get_chunk


def make_reference_chunker(reference_displacements):
    """Create a chunk loader for already prepared crystal displacements."""
    def get_chunk(start, stop):
        return np.asarray(reference_displacements[start:stop], dtype=np.float64)

    return get_chunk


def main():
    """Compute q-resolved diagnostics and save plots/metrics."""
    args = parse_args()
    ase_path = Path(args.ase_path)
    data_path = Path(args.data_path)
    ase = np.load(ase_path)
    data = np.load(data_path)

    positions = np.asarray(ase["positions"], dtype=np.float64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    crystal_shape = tuple(atom_order.shape)
    if len(crystal_shape) != 4:
        raise ValueError(f"atom_order must have shape (nx, ny, nz, atoms), got {crystal_shape}")

    cell = cell_matrix(data, ase)
    inference_frames = int(positions.shape[0])
    if args.max_inference_frames is not None:
        inference_frames = min(inference_frames, int(args.max_inference_frames))

    reference_start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0
    reference_available = max(0, int(data["displacements"].shape[0]) - reference_start)
    reference_frames = reference_available
    if args.max_reference_frames is not None:
        reference_frames = min(reference_frames, int(args.max_reference_frames))
    if reference_frames < 3:
        raise ValueError(f"Need at least three reference frames, got {reference_frames}")

    q_abs = q_grid(crystal_shape[:3], cell)
    inference = compute_q_diagnostics(
        inference_frames,
        make_position_chunker(positions, reference_positions, atom_order, cell),
        crystal_shape,
        args.chunk_frames,
    )
    reference_displacements = np.asarray(
        data["displacements"][reference_start : reference_start + reference_frames],
        dtype=np.float64,
    )
    reference = compute_q_diagnostics(
        reference_frames,
        make_reference_chunker(reference_displacements),
        crystal_shape,
        args.chunk_frames,
    )

    quantities = [
        ("displacement", "u_rms", "RMS |u_q|, A", "q-resolved displacement RMS"),
        ("discrete_acceleration", "a_rms", "RMS |a_q|, A/step^2", "q-resolved discrete acceleration RMS"),
        ("effective_stiffness", "stiffness", "k_eff, 1/step^2", "effective stiffness: -Re<a_q u_q*>/<|u_q|^2>"),
    ]
    shell_rows = {}
    arrays = {}
    for label, key, _ylabel, _title in quantities:
        inference_rows = shell_average(q_abs, inference[key], args.q_decimals, args.exclude_q_zero)
        reference_rows = shell_average(q_abs, reference[key], args.q_decimals, args.exclude_q_zero)
        shell_rows[label] = (inference_rows, reference_rows)
        arrays[label] = rows_to_arrays(inference_rows, reference_rows)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col", constrained_layout=True)
    for row, (label, _key, ylabel, title) in enumerate(quantities):
        q, inference_values, reference_values = arrays[label]
        plot_quantity(axes[row, 0], axes[row, 1], q, inference_values, reference_values, ylabel, title)
    fig.suptitle(args.title, fontsize=15)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Saved {output_path}",
        f"ase_path = {ase_path}",
        f"data_path = {data_path}",
        f"reference_frame_start = {reference_start}",
        f"inference_frames = {inference['u_frames']}",
        f"inference_acceleration_frames = {inference['a_frames']}",
        f"reference_frames = {reference['u_frames']}",
        f"reference_acceleration_frames = {reference['a_frames']}",
        f"exclude_q_zero = {args.exclude_q_zero}",
        "quantity\tq_abs\tq_shell_count\tinference\treference\tratio\tinference_shell_std\treference_shell_std",
    ]
    for label, _key, _ylabel, _title in quantities:
        inference_rows, reference_rows = shell_rows[label]
        lines.extend(metric_lines(label, inference_rows, reference_rows))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
