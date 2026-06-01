"""Plot inference-only histograms for a full ASE RNN trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE trajectory .npz.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal .npz with reference geometry.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics output.")
    parser.add_argument("--bins", type=int, default=140)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--title", default="Full ASE inference histograms")
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


def robust_symmetric_range(values, percentile):
    """Return a symmetric robust histogram range."""
    finite = np.abs(components(values))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(finite, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def robust_positive_range(values, percentile):
    """Return a positive robust histogram range."""
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(finite, percentile))
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    return 0.0, high


def metric_line(quantity, values):
    """Return a tab-separated metric line for a vector quantity."""
    component = components(values)
    magnitude = magnitudes(values)
    rms = float(np.sqrt(np.mean(component**2)))
    return (
        f"{quantity}\t{component.mean():.8g}\t{component.std():.8g}\t"
        f"{rms:.8g}\t{magnitude.mean():.8g}\t{magnitude.std():.8g}\t"
        f"{np.percentile(magnitude, 95):.8g}"
    )


def plot_vector_hist(ax, values, title, xlabel, bins, percentile, magnitude=False):
    """Plot one vector-component or vector-magnitude histogram."""
    histogram_values = magnitudes(values) if magnitude else components(values)
    hist_range = (
        robust_positive_range(histogram_values, percentile)
        if magnitude
        else robust_symmetric_range(histogram_values, percentile)
    )
    ax.hist(histogram_values, bins=bins, range=hist_range, density=True, alpha=0.72, color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def main():
    """Build full-trajectory inference histograms and summary metrics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    positions = np.asarray(ase["positions"], dtype=np.float64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    cell = cell_matrix(data, ase)

    displacements = minimum_image(positions - reference_positions[None, :, :], cell)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    acceleration = discrete_acceleration(displacements)
    temperature = np.asarray(ase["temperature_k"], dtype=np.float64) if "temperature_k" in ase.files else None

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    plot_vector_hist(axes[0, 0], displacements, "displacement components", "u component, A", args.bins, args.percentile)
    plot_vector_hist(
        axes[0, 1], displacements, "displacement magnitudes", "|u|, A", args.bins, args.percentile, magnitude=True
    )
    plot_vector_hist(axes[1, 0], velocities, "velocity components", "v component, A/ps", args.bins, args.percentile)
    plot_vector_hist(
        axes[1, 1], velocities, "velocity magnitudes", "|v|, A/ps", args.bins, args.percentile, magnitude=True
    )
    plot_vector_hist(
        axes[2, 0], acceleration, "discrete acceleration components", "a component, A/step^2", args.bins, args.percentile
    )
    plot_vector_hist(
        axes[2, 1],
        acceleration,
        "discrete acceleration magnitudes",
        "|a|, A/step^2",
        args.bins,
        args.percentile,
        magnitude=True,
    )
    fig.suptitle(args.title, fontsize=15)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Saved {output_path}",
        f"frames = {positions.shape[0]}",
        f"velocity_frames = {velocities.shape[0]}",
        f"acceleration_frames = {acceleration.shape[0]}",
        "quantity\tcomponent_mean\tcomponent_std\tcomponent_rms\tmagnitude_mean\tmagnitude_std\tmagnitude_p95",
        metric_line("displacement", displacements),
        metric_line("velocity", velocities),
        metric_line("discrete_acceleration", acceleration),
    ]
    if temperature is not None:
        lines.append(
            "temperature_k\t"
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
