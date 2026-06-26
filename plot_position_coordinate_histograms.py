"""Plot wrapped absolute-coordinate histograms for ASE inference and reference."""

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
    parser.add_argument("--bins", type=int, default=180, help="Histogram bin count per coordinate.")
    parser.add_argument("--chunk-frames", type=int, default=512, help="Frames processed per chunk.")
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
        "--title",
        default="Wrapped absolute-coordinate histograms",
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


def crystal_to_flat_values(crystal_values, atom_order):
    """Convert crystal-shaped values to flat atom order."""
    crystal_values = np.asarray(crystal_values)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    frames = crystal_values.shape[0]
    atoms = int(np.prod(atom_order.shape))
    flat = np.empty((frames, atoms, 3), dtype=crystal_values.dtype)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = crystal_values[(slice(None), *crystal_index, slice(None))]
    return flat


def crystal_displacements_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacements to flat absolute positions."""
    flat_displacements = crystal_to_flat_values(displacements, atom_order)
    return reference_positions[None, :, :] + flat_displacements


def wrap_positions(positions, cell):
    """Wrap Cartesian positions into the periodic cell."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(positions, dtype=np.float64) @ inverse_cell
    fractional -= np.floor(fractional)
    return fractional @ cell


def histogram_edges(cell, bins):
    """Return component-wise coordinate histogram edges."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=np.float64), axis=1)
    return [np.linspace(0.0, float(length), int(bins) + 1) for length in lengths]


def empty_stats():
    """Create online summary statistics storage."""
    return {
        "count": np.zeros(3, dtype=np.int64),
        "sum": np.zeros(3, dtype=np.float64),
        "sum2": np.zeros(3, dtype=np.float64),
        "min": np.full(3, np.inf, dtype=np.float64),
        "max": np.full(3, -np.inf, dtype=np.float64),
    }


def update_stats(stats, wrapped):
    """Update coordinate summary statistics."""
    values = wrapped.reshape(-1, 3)
    stats["count"] += values.shape[0]
    stats["sum"] += np.sum(values, axis=0)
    stats["sum2"] += np.sum(values * values, axis=0)
    stats["min"] = np.minimum(stats["min"], np.min(values, axis=0))
    stats["max"] = np.maximum(stats["max"], np.max(values, axis=0))


def finalize_stats(stats):
    """Return mean and standard deviation for online stats."""
    count = np.maximum(stats["count"].astype(np.float64), 1.0)
    mean = stats["sum"] / count
    variance = np.maximum(stats["sum2"] / count - mean * mean, 0.0)
    return {
        "count": stats["count"],
        "mean": mean,
        "std": np.sqrt(variance),
        "min": stats["min"],
        "max": stats["max"],
    }


def accumulate_histograms(frame_count, get_position_chunk, cell, edges, chunk_frames):
    """Accumulate coordinate histograms and summary statistics in chunks."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    histograms = [np.zeros(len(edge) - 1, dtype=np.float64) for edge in edges]
    stats = empty_stats()
    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        wrapped = wrap_positions(get_position_chunk(start, stop), cell)
        update_stats(stats, wrapped)
        for component in range(3):
            counts, _ = np.histogram(wrapped[..., component].reshape(-1), bins=edges[component])
            histograms[component] += counts.astype(np.float64)
    return histograms, finalize_stats(stats)


def make_reference_chunker(displacements, reference_positions, atom_order):
    """Create a chunk loader for reference absolute positions."""
    def get_chunk(start, stop):
        flat_displacements = crystal_to_flat_values(displacements[start:stop], atom_order)
        return reference_positions[None, :, :] + flat_displacements

    return get_chunk


def reshape_flat_positions(values, atom_count):
    """Return position arrays as (frames, atoms, 3)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3 and values.shape[1:] == (atom_count, 3):
        return values
    if values.ndim == 2 and values.shape[1] == atom_count * 3:
        return values.reshape(values.shape[0], atom_count, 3)
    raise ValueError(f"Position array must have shape (frames, {atom_count}, 3) or (frames, {atom_count * 3})")


def load_position_sources(prediction, data):
    """Load inference/reference absolute positions from ASE or direct-inference output."""
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    atom_count = int(atom_order.size)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)

    if "positions" in prediction.files:
        reference_start = int(prediction["initial_frames"][-1]) if "initial_frames" in prediction.files else 0
        reference_available = max(0, int(data["displacements"].shape[0]) - reference_start)
        return {
            "mode": "ase",
            "inference_positions": np.asarray(prediction["positions"], dtype=np.float64),
            "reference_positions": None,
            "reference_start": reference_start,
            "reference_available": reference_available,
        }

    if "predicted_positions" in prediction.files:
        inference_positions = reshape_flat_positions(prediction["predicted_positions"], atom_count)
    elif "predicted_displacements" in prediction.files:
        inference_positions = crystal_displacements_to_flat_positions(
            prediction["predicted_displacements"],
            reference_positions,
            atom_order,
        )
    else:
        raise ValueError("Prediction file must contain positions, predicted_positions, or predicted_displacements")

    if "reference_positions_output" in prediction.files:
        reference_output = reshape_flat_positions(prediction["reference_positions_output"], atom_count)
    elif "reference_displacements" in prediction.files:
        reference_output = crystal_displacements_to_flat_positions(
            prediction["reference_displacements"],
            reference_positions,
            atom_order,
        )
    else:
        reference_output = None

    return {
        "mode": "direct",
        "inference_positions": inference_positions,
        "reference_positions": reference_output,
        "reference_start": int(prediction["prediction_start_frame"]) if "prediction_start_frame" in prediction.files else 0,
        "reference_available": 0 if reference_output is None else int(reference_output.shape[0]),
    }


def normalize_histogram(counts, edges):
    """Return density-normalized histogram values."""
    total = float(np.sum(counts))
    widths = np.diff(edges)
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / (total * widths)


def total_variation(counts_a, counts_b):
    """Return total-variation distance between two discrete histograms."""
    total_a = float(np.sum(counts_a))
    total_b = float(np.sum(counts_b))
    if total_a <= 0 or total_b <= 0:
        return np.nan
    prob_a = counts_a / total_a
    prob_b = counts_b / total_b
    return 0.5 * float(np.sum(np.abs(prob_a - prob_b)))


def plot_histograms(output_path, title, edges, inference_hist, reference_hist):
    """Save overlaid coordinate histograms."""
    labels = ["x", "y", "z"]
    colors = {"reference": "tab:blue", "inference": "tab:orange"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for component, ax in enumerate(axes):
        centers = 0.5 * (edges[component][1:] + edges[component][:-1])
        ref_density = normalize_histogram(reference_hist[component], edges[component])
        inf_density = normalize_histogram(inference_hist[component], edges[component])
        ax.plot(centers, ref_density, color=colors["reference"], linewidth=1.4, label="reference")
        ax.plot(centers, inf_density, color=colors["inference"], linewidth=1.2, label="inference")
        ax.fill_between(centers, ref_density, color=colors["reference"], alpha=0.18)
        ax.fill_between(centers, inf_density, color=colors["inference"], alpha=0.18)
        ax.set_title(f"{labels[component]} coordinate")
        ax.set_xlabel(f"{labels[component]}, A")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def stats_lines(name, stats):
    """Return TSV summary lines for one dataset."""
    labels = ["x", "y", "z"]
    lines = []
    for component, label in enumerate(labels):
        lines.append(
            f"{name}\t{label}\t{int(stats['count'][component])}\t"
            f"{stats['mean'][component]:.10g}\t{stats['std'][component]:.10g}\t"
            f"{stats['min'][component]:.10g}\t{stats['max'][component]:.10g}"
        )
    return lines


def main():
    """Build wrapped coordinate histograms and save metrics."""
    args = parse_args()
    ase_path = Path(args.ase_path)
    data_path = Path(args.data_path)
    prediction = np.load(ase_path)
    data = np.load(data_path)
    cell = cell_matrix(data, prediction)
    edges = histogram_edges(cell, args.bins)
    sources = load_position_sources(prediction, data)

    positions = sources["inference_positions"]
    inference_frames = int(positions.shape[0])
    if args.max_inference_frames is not None:
        inference_frames = min(inference_frames, int(args.max_inference_frames))
    inference_hist, inference_stats = accumulate_histograms(
        inference_frames,
        lambda start, stop: positions[start:stop],
        cell,
        edges,
        args.chunk_frames,
    )

    reference_start = sources["reference_start"]
    reference_frames = int(sources["reference_available"])
    if args.max_reference_frames is not None:
        reference_frames = min(reference_frames, int(args.max_reference_frames))
    if reference_frames <= 0:
        raise ValueError("No reference frames are available")
    if sources["reference_positions"] is None:
        reference_displacements = np.asarray(
            data["displacements"][reference_start : reference_start + reference_frames],
            dtype=np.float64,
        )
        reference_chunker = make_reference_chunker(
            reference_displacements,
            np.asarray(data["reference_positions"], dtype=np.float64),
            np.asarray(data["atom_order"], dtype=np.int64),
        )
    else:
        reference_positions_output = sources["reference_positions"]
        reference_chunker = lambda start, stop: reference_positions_output[start:stop]
    reference_hist, reference_stats = accumulate_histograms(
        reference_frames,
        reference_chunker,
        cell,
        edges,
        args.chunk_frames,
    )

    output_path = Path(args.output_path)
    plot_histograms(output_path, args.title, edges, inference_hist, reference_hist)

    labels = ["x", "y", "z"]
    lines = [
        f"Saved {output_path}",
        f"ase_path = {ase_path}",
        f"data_path = {data_path}",
        f"input_mode = {sources['mode']}",
        f"inference_frames = {inference_frames}",
        f"reference_frame_start = {reference_start}",
        f"reference_frames = {reference_frames}",
        "dataset\tcomponent\tcount\tmean_A\tstd_A\tmin_A\tmax_A",
        *stats_lines("inference", inference_stats),
        *stats_lines("reference", reference_stats),
        "component\ttotal_variation_distance",
    ]
    for component, label in enumerate(labels):
        lines.append(f"{label}\t{total_variation(inference_hist[component], reference_hist[component]):.10g}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
