"""Plot displacement component histograms and layer-resolved displacement stats."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXIS_LABELS = ("x", "y", "z")


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE trajectory npz with flat positions.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference npz.")
    parser.add_argument("--hist-output-path", required=True, help="Output PNG path for component histograms.")
    parser.add_argument("--layer-output-path", required=True, help="Output PNG path for layer diagnostics.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics output.")
    parser.add_argument("--bins", type=int, default=180, help="Histogram bin count per component.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Robust percentile for histogram ranges.")
    parser.add_argument("--chunk-frames", type=int, default=512, help="Frames processed per chunk.")
    parser.add_argument(
        "--sample-frame-stride",
        type=int,
        default=20,
        help="Frame stride used only to estimate robust histogram ranges.",
    )
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
        "--subtract-frame-mean",
        action="store_true",
        help="Subtract the per-frame mean displacement from inference and reference before diagnostics.",
    )
    parser.add_argument(
        "--title",
        default="ASE displacement diagnostics",
        help="Figure title prefix.",
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


def crystal_to_flat_values(crystal_values, atom_order):
    """Convert crystal-shaped values to flat atom order."""
    crystal_values = np.asarray(crystal_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    frames = int(crystal_values.shape[0])
    atoms = int(atom_order.size)
    flat = np.empty((frames, atoms, 3), dtype=np.float64)
    flat[:, atom_order.reshape(-1), :] = crystal_values.reshape(frames, atoms, 3)
    return flat


def make_inference_flat_chunker(positions, reference_positions, cell):
    """Create a chunk loader for ASE minimum-image displacements."""
    def get_chunk(start, stop):
        return minimum_image(positions[start:stop] - reference_positions[None, :, :], cell)

    return get_chunk


def make_reference_flat_chunker(reference_displacements, atom_order):
    """Create a chunk loader for reference flat displacements."""
    def get_chunk(start, stop):
        return crystal_to_flat_values(reference_displacements[start:stop], atom_order)

    return get_chunk


def make_crystal_chunker(flat_chunker, atom_order):
    """Wrap a flat displacement chunker with crystal layout conversion."""
    def get_chunk(start, stop):
        return flat_to_crystal(flat_chunker(start, stop), atom_order)

    return get_chunk


def make_reference_crystal_chunker(reference_displacements):
    """Create a chunk loader for already crystal-shaped reference displacements."""
    def get_chunk(start, stop):
        return np.asarray(reference_displacements[start:stop], dtype=np.float64)

    return get_chunk


def subtract_flat_frame_mean(flat):
    """Remove the equal-mass center-of-mass displacement from flat frames."""
    flat = np.asarray(flat, dtype=np.float64)
    if flat.shape[0] == 0:
        return flat
    return flat - np.mean(flat, axis=1, keepdims=True)


def subtract_crystal_frame_mean(crystal):
    """Remove the equal-mass center-of-mass displacement from crystal-shaped frames."""
    crystal = np.asarray(crystal, dtype=np.float64)
    if crystal.shape[0] == 0:
        return crystal
    spatial_axes = tuple(range(1, crystal.ndim - 1))
    return crystal - np.mean(crystal, axis=spatial_axes, keepdims=True)


def wrap_frame_mean_subtraction(get_chunk, *, crystal):
    """Wrap a chunk loader so every frame has zero mean displacement."""
    subtract = subtract_crystal_frame_mean if crystal else subtract_flat_frame_mean

    def wrapped(start, stop):
        return subtract(get_chunk(start, stop))

    return wrapped


def load_displacement_sources(prediction, data):
    """Load inference/reference displacements from ASE or direct-inference output."""
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    cell = cell_matrix(data, prediction)

    if "positions" in prediction.files:
        positions = np.asarray(prediction["positions"], dtype=np.float64)
        reference_start = int(prediction["initial_frames"][-1]) if "initial_frames" in prediction.files else 0
        reference_available = max(0, int(data["displacements"].shape[0]) - reference_start)
        inference_flat_chunker = make_inference_flat_chunker(positions, reference_positions, cell)
        reference_displacements = np.asarray(data["displacements"][reference_start:], dtype=np.float64)
        return {
            "mode": "ase",
            "cell": cell,
            "atom_order": atom_order,
            "inference_frames": int(positions.shape[0]),
            "reference_frames": reference_available,
            "reference_start": reference_start,
            "inference_flat_chunker": inference_flat_chunker,
            "reference_flat_chunker": make_reference_flat_chunker(reference_displacements, atom_order),
            "inference_crystal_chunker": make_crystal_chunker(inference_flat_chunker, atom_order),
            "reference_crystal_chunker": make_reference_crystal_chunker(reference_displacements),
        }

    if "predicted_displacements" not in prediction.files:
        raise ValueError("Prediction file must contain positions or predicted_displacements")
    predicted_displacements = np.asarray(prediction["predicted_displacements"], dtype=np.float64)
    if "reference_displacements" in prediction.files:
        reference_displacements = np.asarray(prediction["reference_displacements"], dtype=np.float64)
        reference_start = int(prediction["prediction_start_frame"]) if "prediction_start_frame" in prediction.files else 0
    else:
        reference_start = int(prediction["prediction_start_frame"]) if "prediction_start_frame" in prediction.files else 0
        reference_displacements = np.asarray(data["displacements"][reference_start:], dtype=np.float64)

    return {
        "mode": "direct",
        "cell": cell,
        "atom_order": atom_order,
        "inference_frames": int(predicted_displacements.shape[0]),
        "reference_frames": int(reference_displacements.shape[0]),
        "reference_start": reference_start,
        "inference_flat_chunker": make_reference_flat_chunker(predicted_displacements, atom_order),
        "reference_flat_chunker": make_reference_flat_chunker(reference_displacements, atom_order),
        "inference_crystal_chunker": make_reference_crystal_chunker(predicted_displacements),
        "reference_crystal_chunker": make_reference_crystal_chunker(reference_displacements),
    }


def collect_component_samples(frame_count, get_flat_chunk, chunk_frames, sample_frame_stride):
    """Collect a decimated component sample for robust histogram ranges."""
    if sample_frame_stride <= 0:
        raise ValueError("--sample-frame-stride must be positive")
    samples = [[] for _ in range(3)]
    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        chunk = get_flat_chunk(start, stop)[::sample_frame_stride]
        if chunk.shape[0] == 0:
            continue
        for component in range(3):
            samples[component].append(chunk[..., component].reshape(-1))
    return [np.concatenate(parts) if parts else np.array([], dtype=np.float64) for parts in samples]


def component_ranges(inference_samples, reference_samples, percentile):
    """Return robust symmetric histogram ranges for every vector component."""
    ranges = []
    for component in range(3):
        values = np.concatenate(
            [
                np.abs(np.asarray(inference_samples[component], dtype=np.float64).reshape(-1)),
                np.abs(np.asarray(reference_samples[component], dtype=np.float64).reshape(-1)),
            ]
        )
        values = values[np.isfinite(values)]
        limit = float(np.percentile(values, percentile)) if values.size else 1.0
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        ranges.append((-limit, limit))
    return ranges


def empty_component_stats():
    """Create online component summary statistics."""
    return {
        "count": np.zeros(3, dtype=np.int64),
        "sum": np.zeros(3, dtype=np.float64),
        "sum2": np.zeros(3, dtype=np.float64),
        "min": np.full(3, np.inf, dtype=np.float64),
        "max": np.full(3, -np.inf, dtype=np.float64),
    }


def update_component_stats(stats, flat):
    """Update component summary statistics."""
    values = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    stats["count"] += values.shape[0]
    stats["sum"] += np.sum(values, axis=0)
    stats["sum2"] += np.sum(values * values, axis=0)
    stats["min"] = np.minimum(stats["min"], np.min(values, axis=0))
    stats["max"] = np.maximum(stats["max"], np.max(values, axis=0))


def finalize_component_stats(stats):
    """Return mean, standard deviation, and RMS for components."""
    count = np.maximum(stats["count"].astype(np.float64), 1.0)
    mean = stats["sum"] / count
    mean2 = stats["sum2"] / count
    variance = np.maximum(mean2 - mean * mean, 0.0)
    return {
        "count": stats["count"],
        "mean": mean,
        "std": np.sqrt(variance),
        "rms": np.sqrt(mean2),
        "min": stats["min"],
        "max": stats["max"],
    }


def accumulate_component_histograms(frame_count, get_flat_chunk, ranges, bins, chunk_frames):
    """Accumulate component histograms and statistics."""
    histograms = [np.zeros(int(bins), dtype=np.float64) for _ in range(3)]
    stats = empty_component_stats()
    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        chunk = get_flat_chunk(start, stop)
        update_component_stats(stats, chunk)
        for component in range(3):
            counts, _ = np.histogram(chunk[..., component].reshape(-1), bins=bins, range=ranges[component])
            histograms[component] += counts.astype(np.float64)
    return histograms, finalize_component_stats(stats)


def normalize_histogram(counts, hist_range):
    """Return density-normalized histogram values."""
    total = float(np.sum(counts))
    width = (float(hist_range[1]) - float(hist_range[0])) / float(len(counts))
    if total <= 0 or width <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / (total * width)


def total_variation(counts_a, counts_b):
    """Return total-variation distance between two histograms."""
    total_a = float(np.sum(counts_a))
    total_b = float(np.sum(counts_b))
    if total_a <= 0 or total_b <= 0:
        return np.nan
    return 0.5 * float(np.sum(np.abs(counts_a / total_a - counts_b / total_b)))


def empty_layer_stats(crystal_shape):
    """Create online layer-resolved displacement statistics."""
    return [
        {
            "count": np.zeros(int(crystal_shape[axis]), dtype=np.int64),
            "sum": np.zeros((int(crystal_shape[axis]), 3), dtype=np.float64),
            "sum2": np.zeros((int(crystal_shape[axis]), 3), dtype=np.float64),
            "mag2": np.zeros(int(crystal_shape[axis]), dtype=np.float64),
        }
        for axis in range(3)
    ]


def update_layer_stats(stats, crystal):
    """Update layer-resolved statistics from a crystal displacement chunk."""
    crystal = np.asarray(crystal, dtype=np.float64)
    for axis in range(3):
        moved = np.moveaxis(crystal, 1 + axis, 1)
        layer_values = moved.reshape(moved.shape[0], moved.shape[1], -1, 3)
        stats[axis]["count"] += layer_values.shape[0] * layer_values.shape[2]
        stats[axis]["sum"] += np.sum(layer_values, axis=(0, 2))
        stats[axis]["sum2"] += np.sum(layer_values * layer_values, axis=(0, 2))
        stats[axis]["mag2"] += np.sum(np.sum(layer_values * layer_values, axis=-1), axis=(0, 2))


def accumulate_layer_stats(frame_count, get_crystal_chunk, crystal_shape, chunk_frames):
    """Accumulate layer-resolved mean and RMS displacement values."""
    stats = empty_layer_stats(crystal_shape)
    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        update_layer_stats(stats, get_crystal_chunk(start, stop))
    return finalize_layer_stats(stats)


def finalize_layer_stats(stats):
    """Convert raw layer statistics to means and RMS values."""
    finalized = []
    for axis_stats in stats:
        count = np.maximum(axis_stats["count"].astype(np.float64), 1.0)
        mean = axis_stats["sum"] / count[:, None]
        mean2 = axis_stats["sum2"] / count[:, None]
        finalized.append(
            {
                "count": axis_stats["count"],
                "mean": mean,
                "std": np.sqrt(np.maximum(mean2 - mean * mean, 0.0)),
                "rms": np.sqrt(mean2),
                "rms_magnitude": np.sqrt(axis_stats["mag2"] / count),
            }
        )
    return finalized


def plot_histograms(output_path, title, ranges, inference_hist, reference_hist):
    """Save component displacement histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for component, ax in enumerate(axes):
        low, high = ranges[component]
        centers = np.linspace(low, high, len(inference_hist[component]), endpoint=False)
        centers += 0.5 * (high - low) / len(inference_hist[component])
        reference_density = normalize_histogram(reference_hist[component], ranges[component])
        inference_density = normalize_histogram(inference_hist[component], ranges[component])
        ax.plot(centers, reference_density, color="tab:blue", label="reference", linewidth=1.5)
        ax.plot(centers, inference_density, color="tab:orange", label="inference", linewidth=1.3)
        ax.fill_between(centers, reference_density, color="tab:blue", alpha=0.18)
        ax.fill_between(centers, inference_density, color="tab:orange", alpha=0.18)
        ax.set_title(f"u_{AXIS_LABELS[component]}")
        ax.set_xlabel("minimum-image displacement, A")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"{title}: displacement component histograms", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layer_stats(output_path, title, inference_layers, reference_layers):
    """Save layer-resolved displacement mean and RMS diagnostics."""
    component_colors = ("tab:red", "tab:green", "tab:purple")
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for axis in range(3):
        layer_index = np.arange(inference_layers[axis]["mean"].shape[0])
        ax_mean = axes[axis, 0]
        ax_rms = axes[axis, 1]
        for component in range(3):
            label = f"u_{AXIS_LABELS[component]}"
            ax_mean.plot(
                layer_index,
                reference_layers[axis]["mean"][:, component],
                "--",
                color=component_colors[component],
                linewidth=1.2,
                label=f"ref {label}",
            )
            ax_mean.plot(
                layer_index,
                inference_layers[axis]["mean"][:, component],
                "-",
                color=component_colors[component],
                linewidth=1.5,
                label=f"inf {label}",
            )
        ax_mean.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax_mean.set_title(f"Mean displacement by {AXIS_LABELS[axis]} layer")
        ax_mean.set_xlabel(f"{AXIS_LABELS[axis]} layer index")
        ax_mean.set_ylabel("mean u, A")
        ax_mean.grid(alpha=0.25)
        ax_mean.legend(fontsize=7, ncol=2)

        ax_rms.plot(
            layer_index,
            reference_layers[axis]["rms_magnitude"],
            "o--",
            color="tab:blue",
            linewidth=1.3,
            markersize=3,
            label="reference |u| RMS",
        )
        ax_rms.plot(
            layer_index,
            inference_layers[axis]["rms_magnitude"],
            "o-",
            color="tab:orange",
            linewidth=1.5,
            markersize=3,
            label="inference |u| RMS",
        )
        ax_rms.set_title(f"Displacement RMS magnitude by {AXIS_LABELS[axis]} layer")
        ax_rms.set_xlabel(f"{AXIS_LABELS[axis]} layer index")
        ax_rms.set_ylabel("|u| RMS, A")
        ax_rms.grid(alpha=0.25)
        ax_rms.legend(fontsize=8)
    fig.suptitle(f"{title}: layer-resolved displacements", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def component_summary_lines(source, stats):
    """Return TSV component summary lines."""
    lines = []
    for component, label in enumerate(AXIS_LABELS):
        lines.append(
            f"component_summary\t{source}\t{label}\t{int(stats['count'][component])}\t"
            f"{stats['mean'][component]:.10g}\t{stats['std'][component]:.10g}\t"
            f"{stats['rms'][component]:.10g}\t{stats['min'][component]:.10g}\t{stats['max'][component]:.10g}"
        )
    return lines


def layer_summary_lines(source, layers):
    """Return TSV layer summary lines."""
    lines = []
    for axis, axis_label in enumerate(AXIS_LABELS):
        for layer in range(layers[axis]["mean"].shape[0]):
            mean = layers[axis]["mean"][layer]
            rms = layers[axis]["rms"][layer]
            lines.append(
                f"layer_summary\t{source}\t{axis_label}\t{layer}\t{int(layers[axis]['count'][layer])}\t"
                f"{mean[0]:.10g}\t{mean[1]:.10g}\t{mean[2]:.10g}\t"
                f"{rms[0]:.10g}\t{rms[1]:.10g}\t{rms[2]:.10g}\t"
                f"{layers[axis]['rms_magnitude'][layer]:.10g}"
            )
    return lines


def main():
    """Build displacement histograms and layer diagnostics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("--percentile must be in (0, 100]")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")

    ase_path = Path(args.ase_path)
    data_path = Path(args.data_path)
    prediction = np.load(ase_path)
    data = np.load(data_path)
    sources = load_displacement_sources(prediction, data)
    atom_order = sources["atom_order"]
    crystal_shape = tuple(atom_order.shape)
    if len(crystal_shape) != 4:
        raise ValueError(f"atom_order must have shape (nx, ny, nz, atoms), got {crystal_shape}")

    inference_frames = int(sources["inference_frames"])
    if args.max_inference_frames is not None:
        inference_frames = min(inference_frames, int(args.max_inference_frames))

    reference_start = int(sources["reference_start"])
    reference_frames = int(sources["reference_frames"])
    if args.max_reference_frames is not None:
        reference_frames = min(reference_frames, int(args.max_reference_frames))
    if inference_frames < 1 or reference_frames < 1:
        raise ValueError("Need at least one inference and one reference frame")
    inference_flat_chunker = sources["inference_flat_chunker"]
    reference_flat_chunker = sources["reference_flat_chunker"]
    inference_crystal_chunker = sources["inference_crystal_chunker"]
    reference_crystal_chunker = sources["reference_crystal_chunker"]
    if args.subtract_frame_mean:
        inference_flat_chunker = wrap_frame_mean_subtraction(inference_flat_chunker, crystal=False)
        reference_flat_chunker = wrap_frame_mean_subtraction(reference_flat_chunker, crystal=False)
        inference_crystal_chunker = wrap_frame_mean_subtraction(inference_crystal_chunker, crystal=True)
        reference_crystal_chunker = wrap_frame_mean_subtraction(reference_crystal_chunker, crystal=True)

    inference_samples = collect_component_samples(
        inference_frames,
        inference_flat_chunker,
        args.chunk_frames,
        args.sample_frame_stride,
    )
    reference_samples = collect_component_samples(
        reference_frames,
        reference_flat_chunker,
        args.chunk_frames,
        max(1, min(args.sample_frame_stride, reference_frames)),
    )
    ranges = component_ranges(inference_samples, reference_samples, args.percentile)

    inference_hist, inference_stats = accumulate_component_histograms(
        inference_frames,
        inference_flat_chunker,
        ranges,
        args.bins,
        args.chunk_frames,
    )
    reference_hist, reference_stats = accumulate_component_histograms(
        reference_frames,
        reference_flat_chunker,
        ranges,
        args.bins,
        args.chunk_frames,
    )
    inference_layers = accumulate_layer_stats(
        inference_frames,
        inference_crystal_chunker,
        crystal_shape,
        args.chunk_frames,
    )
    reference_layers = accumulate_layer_stats(
        reference_frames,
        reference_crystal_chunker,
        crystal_shape,
        args.chunk_frames,
    )

    hist_output_path = Path(args.hist_output_path)
    layer_output_path = Path(args.layer_output_path)
    plot_histograms(hist_output_path, args.title, ranges, inference_hist, reference_hist)
    plot_layer_stats(layer_output_path, args.title, inference_layers, reference_layers)

    lines = [
        f"Saved {hist_output_path}",
        f"Saved {layer_output_path}",
        f"ase_path = {ase_path}",
        f"data_path = {data_path}",
        f"input_mode = {sources['mode']}",
        f"subtract_frame_mean = {int(bool(args.subtract_frame_mean))}",
        f"inference_frames = {inference_frames}",
        f"reference_frame_start = {reference_start}",
        f"reference_frames = {reference_frames}",
        "component_tv\tcomponent\ttotal_variation_distance\trange_low_A\trange_high_A",
    ]
    for component, label in enumerate(AXIS_LABELS):
        lines.append(
            f"component_tv\t{label}\t{total_variation(inference_hist[component], reference_hist[component]):.10g}\t"
            f"{ranges[component][0]:.10g}\t{ranges[component][1]:.10g}"
        )
    lines.extend(
        [
            "component_summary\tsource\tcomponent\tcount\tmean_A\tstd_A\trms_A\tmin_A\tmax_A",
            *component_summary_lines("inference", inference_stats),
            *component_summary_lines("reference", reference_stats),
            "layer_summary\tsource\tlayer_axis\tlayer\tcount\tmean_ux_A\tmean_uy_A\tmean_uz_A\t"
            "rms_ux_A\trms_uy_A\trms_uz_A\trms_magnitude_A",
            *layer_summary_lines("inference", inference_layers),
            *layer_summary_lines("reference", reference_layers),
        ]
    )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
