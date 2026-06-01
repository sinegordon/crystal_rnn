"""Plot full ASE inference histograms overlaid with available reference data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE trajectory .npz.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics output.")
    parser.add_argument("--bins", type=int, default=140)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--title", default="Full ASE inference vs available reference histograms")
    return parser.parse_args()


def as_cell_matrix(value):
    """Return a 3x3 cell matrix."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,) or (3, 3)")
    return cell


def cell_matrix(data, ase):
    """Return the simulation cell matrix."""
    if "cell" in ase.files:
        return as_cell_matrix(ase["cell"])
    if "cell" in data.files:
        return as_cell_matrix(data["cell"])
    if "box_lengths" in data.files:
        return as_cell_matrix(data["box_lengths"])
    raise ValueError("No cell or box_lengths found")


def minimum_image(delta, cell):
    """Wrap position differences into the nearest periodic image."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(delta, dtype=np.float64) @ inverse_cell
    fractional -= np.round(fractional)
    return fractional @ cell


def crystal_to_flat_values(crystal_values, atom_order):
    """Convert crystal-shaped frame values into flat ASE atom order."""
    crystal_values = np.asarray(crystal_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    frames = int(crystal_values.shape[0])
    atom_count = int(atom_order.size)
    flat = np.empty((frames, atom_count, 3), dtype=np.float64)
    flat[:, atom_order.reshape(-1), :] = crystal_values.reshape(frames, atom_count, 3)
    return flat


def components(values):
    """Return flattened vector components."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def magnitudes(values):
    """Return flattened vector magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def discrete_acceleration(displacements):
    """Return discrete accelerations from displacement frames."""
    if displacements.shape[0] < 3:
        raise ValueError("At least three displacement frames are required")
    return displacements[2:] - 2.0 * displacements[1:-1] + displacements[:-2]


def robust_symmetric_range(*arrays, percentile):
    """Return a symmetric robust histogram range for several arrays."""
    values = np.concatenate([np.abs(components(array)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def robust_positive_range(*arrays, percentile):
    """Return a positive robust histogram range for several arrays."""
    values = np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(values, percentile))
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    return 0.0, high


def metric_line(quantity, source, values):
    """Return a tab-separated metric line for a vector quantity."""
    component = components(values)
    magnitude = magnitudes(values)
    rms = float(np.sqrt(np.mean(component**2)))
    return (
        f"{quantity}\t{source}\t{component.mean():.8g}\t{component.std():.8g}\t"
        f"{rms:.8g}\t{magnitude.mean():.8g}\t{magnitude.std():.8g}\t"
        f"{np.percentile(magnitude, 95):.8g}"
    )


def plot_pair(ax, inference, reference, title, xlabel, bins, percentile, magnitude=False):
    """Plot one overlaid inference/reference histogram pair."""
    transform = magnitudes if magnitude else components
    inference_values = transform(inference)
    reference_values = transform(reference)
    hist_range = (
        robust_positive_range(inference_values, reference_values, percentile=percentile)
        if magnitude
        else robust_symmetric_range(inference_values, reference_values, percentile=percentile)
    )
    ax.hist(
        reference_values,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=0.48,
        label="reference",
        color="tab:blue",
    )
    ax.hist(
        inference_values,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=0.48,
        label="full inference",
        color="tab:orange",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def main():
    """Build overlaid histograms for full inference and available reference."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    cell = cell_matrix(data, ase)
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0

    inference_positions = np.asarray(ase["positions"], dtype=np.float64)
    inference_displacements = minimum_image(inference_positions - reference_positions[None, :, :], cell)
    inference_velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    inference_acceleration = discrete_acceleration(inference_displacements)

    reference_crystal = np.asarray(data["displacements"][start:], dtype=np.float64)
    reference_displacements = crystal_to_flat_values(reference_crystal, atom_order)
    reference_flat_positions = reference_positions[None, :, :] + reference_displacements
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    reference_velocities = np.diff(reference_flat_positions, axis=0) / dt_ps
    reference_acceleration = discrete_acceleration(reference_displacements)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    plot_pair(
        axes[0, 0],
        inference_displacements,
        reference_displacements,
        "displacement components",
        "u component, A",
        args.bins,
        args.percentile,
    )
    plot_pair(
        axes[0, 1],
        inference_displacements,
        reference_displacements,
        "displacement magnitudes",
        "|u|, A",
        args.bins,
        args.percentile,
        magnitude=True,
    )
    plot_pair(
        axes[1, 0],
        inference_velocities,
        reference_velocities,
        "velocity components",
        "v component, A/ps",
        args.bins,
        args.percentile,
    )
    plot_pair(
        axes[1, 1],
        inference_velocities,
        reference_velocities,
        "velocity magnitudes",
        "|v|, A/ps",
        args.bins,
        args.percentile,
        magnitude=True,
    )
    plot_pair(
        axes[2, 0],
        inference_acceleration,
        reference_acceleration,
        "discrete acceleration components",
        "a component, A/step^2",
        args.bins,
        args.percentile,
    )
    plot_pair(
        axes[2, 1],
        inference_acceleration,
        reference_acceleration,
        "discrete acceleration magnitudes",
        "|a|, A/step^2",
        args.bins,
        args.percentile,
        magnitude=True,
    )
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(args.title, fontsize=15)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Saved {output_path}",
        f"reference_frame_start = {start}",
        f"inference_frames = {inference_positions.shape[0]}",
        f"reference_frames = {reference_displacements.shape[0]}",
        f"inference_acceleration_frames = {inference_acceleration.shape[0]}",
        f"reference_acceleration_frames = {reference_acceleration.shape[0]}",
        "quantity\tsource\tcomponent_mean\tcomponent_std\tcomponent_rms\tmagnitude_mean\tmagnitude_std\tmagnitude_p95",
        metric_line("displacement", "full_inference", inference_displacements),
        metric_line("displacement", "reference", reference_displacements),
        metric_line("velocity", "full_inference", inference_velocities),
        metric_line("velocity", "reference", reference_velocities),
        metric_line("discrete_acceleration", "full_inference", inference_acceleration),
        metric_line("discrete_acceleration", "reference", reference_acceleration),
    ]
    if "temperature_k" in ase.files:
        temperature = np.asarray(ase["temperature_k"], dtype=np.float64)
        lines.append(
            "temperature_k\tfull_inference\t"
            f"{temperature[0]:.8g}\t{temperature[-1]:.8g}\t{temperature.mean():.8g}\t"
            f"{temperature.min():.8g}\t{temperature.max():.8g}"
        )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
