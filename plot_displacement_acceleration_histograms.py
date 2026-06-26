"""Plot displacement and discrete-acceleration histograms against reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="Inference npz with positions or displacements.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference npz.")
    parser.add_argument("--output-path", required=True, help="Output histogram PNG.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics output.")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile used to set robust histogram ranges.",
    )
    parser.add_argument(
        "--title",
        default="ASE displacement and acceleration histograms vs reference",
        help="Figure title.",
    )
    return parser.parse_args()


def cell_matrix(data, ase):
    """Return the simulation cell matrix."""
    if "cell" in ase.files:
        cell = np.asarray(ase["cell"], dtype=np.float64)
    elif "cell" in data.files:
        cell = np.asarray(data["cell"], dtype=np.float64)
    elif "box_lengths" in data.files:
        cell = np.asarray(data["box_lengths"], dtype=np.float64)
    else:
        raise ValueError("No cell or box_lengths found")
    if cell.ndim == 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must be shape (3,) or (3, 3)")
    return cell


def minimum_image(delta, cell):
    """Wrap flat or batched position differences into the nearest image."""
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


def robust_symmetric_range(*arrays, percentile):
    """Return a symmetric histogram range around zero."""
    values = np.concatenate([np.abs(np.asarray(array, dtype=np.float64).reshape(-1)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def robust_positive_range(*arrays, percentile):
    """Return a positive histogram range starting at zero."""
    values = np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(values, percentile))
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    return 0.0, high


def components(values):
    """Flatten vector components."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def magnitudes(values):
    """Flatten vector magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def discrete_acceleration(displacements):
    """Return model-scale discrete accelerations from displacement frames."""
    displacements = np.asarray(displacements, dtype=np.float64)
    if displacements.shape[0] < 3:
        raise ValueError("At least three frames are required for acceleration histograms")
    return displacements[2:] - 2.0 * displacements[1:-1] + displacements[:-2]


def metric_line(quantity, source, values):
    """Return a tab-separated summary metric line."""
    component = components(values)
    magnitude = magnitudes(values)
    rms = float(np.sqrt(np.mean(component**2)))
    return (
        f"{quantity}\t{source}\t{component.mean():.8g}\t{component.std():.8g}\t"
        f"{rms:.8g}\t{magnitude.mean():.8g}\t{magnitude.std():.8g}\t"
        f"{np.percentile(magnitude, 95):.8g}"
    )


def plot_pair(ax, predicted, reference, label, xlabel, bins, percentile, magnitude=False):
    """Plot one predicted/reference histogram pair."""
    transform = magnitudes if magnitude else components
    predicted_values = transform(predicted)
    reference_values = transform(reference)
    hist_range = (
        robust_positive_range(predicted_values, reference_values, percentile=percentile)
        if magnitude
        else robust_symmetric_range(predicted_values, reference_values, percentile=percentile)
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
        predicted_values,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=0.48,
        label="inference",
        color="tab:orange",
    )
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def main():
    """Build displacement/acceleration histograms and metric summaries."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    if "predicted_displacements" in ase.files and "reference_displacements" in ase.files:
        predicted_crystal = np.asarray(ase["predicted_displacements"], dtype=np.float64)
        reference_crystal = np.asarray(ase["reference_displacements"], dtype=np.float64)
        frames = min(int(predicted_crystal.shape[0]), int(reference_crystal.shape[0]))
        if frames < 3:
            raise ValueError(f"Need at least three overlapping direct-inference frames, got {frames}")
        if frames < predicted_crystal.shape[0] or frames < reference_crystal.shape[0]:
            print(
                f"WARNING: trimming direct-inference arrays to {frames} frames: "
                f"predicted={predicted_crystal.shape[0]}, reference={reference_crystal.shape[0]}."
            )
        start = int(ase["prediction_start_frame"]) if "prediction_start_frame" in ase.files else 0
        predicted_displacements = crystal_to_flat_values(predicted_crystal[:frames], atom_order)
        reference_displacements = crystal_to_flat_values(reference_crystal[:frames], atom_order)
    else:
        positions = np.asarray(ase["positions"], dtype=np.float64)
        reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
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
        predicted_displacements = minimum_image(positions[:frames] - reference_positions[None, :, :], cell)
        reference_displacements = crystal_to_flat_values(data["displacements"][start : start + frames], atom_order)
    predicted_acceleration = discrete_acceleration(predicted_displacements)
    reference_acceleration = discrete_acceleration(reference_displacements)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_pair(
        axes[0, 0],
        predicted_displacements,
        reference_displacements,
        "displacement components",
        "u component, A",
        args.bins,
        args.percentile,
    )
    plot_pair(
        axes[0, 1],
        predicted_displacements,
        reference_displacements,
        "displacement magnitudes",
        "|u|, A",
        args.bins,
        args.percentile,
        magnitude=True,
    )
    plot_pair(
        axes[1, 0],
        predicted_acceleration,
        reference_acceleration,
        "discrete acceleration components",
        "a component, A/step^2",
        args.bins,
        args.percentile,
    )
    plot_pair(
        axes[1, 1],
        predicted_acceleration,
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
        f"comparison_frames = {frames}",
        f"acceleration_frames = {frames - 2}",
        "quantity\tsource\tcomponent_mean\tcomponent_std\tcomponent_rms\tmagnitude_mean\tmagnitude_std\tmagnitude_p95",
        metric_line("displacement", "inference", predicted_displacements),
        metric_line("displacement", "reference", reference_displacements),
        metric_line("discrete_acceleration", "inference", predicted_acceleration),
        metric_line("discrete_acceleration", "reference", reference_acceleration),
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
