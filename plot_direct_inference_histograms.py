"""Plot full direct-inference position and velocity histograms against reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--input-path", action="append", required=True, help="Direct inference output .npz.")
    parser.add_argument("--label", action="append", default=None, help="Label for the matching --input-path.")
    parser.add_argument("--position-output", required=True, help="Output PNG for wrapped position histograms.")
    parser.add_argument("--velocity-output", required=True, help="Output PNG for velocity histograms.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics path.")
    parser.add_argument("--dt", type=float, default=0.02, help="Frame spacing in ps.")
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--position-percentile", type=float, default=100.0)
    parser.add_argument("--velocity-percentile", type=float, default=99.5)
    parser.add_argument("--speed-percentile", type=float, default=99.5)
    parser.add_argument("--title", default="Direct inference histograms")
    return parser.parse_args()


def crystal_to_flat_values(crystal_values, atom_order):
    """Convert crystal-shaped values to flat atom order."""
    crystal_values = np.asarray(crystal_values)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    frames = crystal_values.shape[0]
    atoms = int(atom_order.size)
    flat = np.empty((frames, atoms, 3), dtype=crystal_values.dtype)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = crystal_values[(slice(None), *crystal_index, slice(None))]
    return flat


def crystal_displacements_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacements to flat absolute positions."""
    flat_displacements = crystal_to_flat_values(displacements, atom_order)
    return np.asarray(reference_positions, dtype=np.float64)[None, :, :] + flat_displacements


def cell_matrix(data):
    """Return a 3x3 orthorhombic cell matrix from reference data."""
    if "cell" in data.files:
        cell = np.asarray(data["cell"], dtype=np.float64)
        if cell.ndim == 3:
            cell = cell[0]
    elif "box_lengths" in data.files:
        cell = np.asarray(data["box_lengths"], dtype=np.float64)
        if cell.ndim == 2:
            cell = cell[0]
    else:
        raise ValueError("data file must contain cell or box_lengths")
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,) or (3, 3)")
    return cell


def wrap_positions(positions, cell):
    """Wrap Cartesian positions into a periodic cell."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(positions, dtype=np.float64) @ inverse_cell
    fractional -= np.floor(fractional)
    return fractional @ cell


def load_reference_displacements(prediction, data):
    """Return the reference displacement continuation matching one prediction file."""
    if "reference_displacements" in prediction.files:
        return np.asarray(prediction["reference_displacements"], dtype=np.float64)
    start = int(prediction["prediction_start_frame"])
    count_steps = int(prediction["count_steps"])
    stop = start + count_steps
    return np.asarray(data["displacements"][start:stop], dtype=np.float64)


def load_sources(input_paths, labels, data, reference_label):
    """Load model predictions and one matching reference trajectory."""
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    sources = []
    reference_displacements = None
    reference_meta = None

    for path, label in zip(input_paths, labels):
        prediction = np.load(path)
        predicted_displacements = np.asarray(prediction["predicted_displacements"], dtype=np.float64)
        sources.append(
            {
                "label": label,
                "positions": crystal_displacements_to_flat_positions(
                    predicted_displacements,
                    reference_positions,
                    atom_order,
                ),
            }
        )
        if reference_displacements is None:
            reference_displacements = load_reference_displacements(prediction, data)
            reference_meta = {
                "prediction_start_frame": int(prediction["prediction_start_frame"]),
                "count_steps": int(prediction["count_steps"]),
            }

    reference_source = {
        "label": reference_label,
        "positions": crystal_displacements_to_flat_positions(reference_displacements, reference_positions, atom_order),
    }
    return reference_source, sources, reference_meta


def velocity_from_positions(positions, dt):
    """Return finite-difference velocities in A/ps."""
    return np.diff(np.asarray(positions, dtype=np.float64), axis=0) / float(dt)


def components(values):
    """Return flattened vector components."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def magnitudes(values):
    """Return flattened vector magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def finite_percentile(values, percentile):
    """Return a robust finite percentile."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    result = float(np.percentile(values, percentile))
    return result if np.isfinite(result) and result > 0 else 1.0


def position_edges(wrapped_sources, cell, bins, percentile):
    """Return component-wise position histogram edges."""
    lengths = np.linalg.norm(cell, axis=1)
    if percentile >= 100:
        return [np.linspace(0.0, float(length), bins + 1) for length in lengths]
    edges = []
    for component in range(3):
        values = np.concatenate([source["positions"][..., component].reshape(-1) for source in wrapped_sources])
        upper = finite_percentile(values, percentile)
        edges.append(np.linspace(0.0, min(float(lengths[component]), upper), bins + 1))
    return edges


def density_hist(values, edges):
    """Return density histogram values and centers."""
    hist, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def plot_positions(reference, predictions, cell, output_path, bins, percentile, title):
    """Save wrapped absolute-coordinate histograms."""
    wrapped = [{"label": reference["label"], "positions": wrap_positions(reference["positions"], cell)}]
    wrapped.extend({"label": item["label"], "positions": wrap_positions(item["positions"], cell)} for item in predictions)
    edges = position_edges(wrapped, cell, bins, percentile)
    colors = ["black", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for component, ax in enumerate(axes):
        for source, color in zip(wrapped, colors):
            centers, hist = density_hist(source["positions"][..., component].reshape(-1), edges[component])
            ax.plot(centers, hist, label=source["label"], color=color, linewidth=1.4)
        ax.set_title(f"{labels[component]} wrapped position")
        ax.set_xlabel(f"{labels[component]}, A")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_velocities(reference, predictions, output_path, dt, bins, velocity_percentile, speed_percentile, title):
    """Save full-trajectory velocity component and speed histograms."""
    sources = [{"label": reference["label"], "velocity": velocity_from_positions(reference["positions"], dt)}]
    sources.extend({"label": item["label"], "velocity": velocity_from_positions(item["positions"], dt)} for item in predictions)
    component_limit = finite_percentile(
        np.concatenate([np.abs(components(source["velocity"])) for source in sources]),
        velocity_percentile,
    )
    speed_limit = finite_percentile(
        np.concatenate([magnitudes(source["velocity"]) for source in sources]),
        speed_percentile,
    )
    colors = ["black", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for source, color in zip(sources, colors):
        axes[0].hist(
            components(source["velocity"]),
            bins=bins,
            range=(-component_limit, component_limit),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=source["label"],
            color=color,
        )
        axes[1].hist(
            magnitudes(source["velocity"]),
            bins=bins,
            range=(0.0, speed_limit),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=source["label"],
            color=color,
        )
    axes[0].set_title("velocity components")
    axes[0].set_xlabel("v component, A/ps")
    axes[0].set_ylabel("density")
    axes[1].set_title("velocity magnitudes")
    axes[1].set_xlabel("|v|, A/ps")
    axes[1].set_ylabel("density")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return sources


def summary_line(quantity, label, values):
    """Return one compact metric line."""
    values = np.asarray(values, dtype=np.float64)
    component_std = float(np.std(components(values)))
    magnitude = magnitudes(values)
    return (
        f"{quantity}\t{label}\t{values.shape[0]}\t"
        f"{float(np.mean(magnitude)):.8g}\t{float(np.std(magnitude)):.8g}\t"
        f"{float(np.percentile(magnitude, 95)):.8g}\t{component_std:.8g}"
    )


def main():
    """Load predictions, plot histograms, and write metrics."""
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("dt must be positive")
    labels = args.label or [Path(path).stem for path in args.input_path]
    if len(labels) != len(args.input_path):
        raise ValueError("--label must be repeated exactly once per --input-path")

    data = np.load(args.data_path)
    reference, predictions, reference_meta = load_sources(
        args.input_path,
        labels,
        data,
        f"reference {Path(args.data_path).stem}",
    )
    cell = cell_matrix(data)
    plot_positions(
        reference,
        predictions,
        cell,
        Path(args.position_output),
        args.bins,
        args.position_percentile,
        args.title + ": positions",
    )
    velocity_sources = plot_velocities(
        reference,
        predictions,
        Path(args.velocity_output),
        args.dt,
        args.bins,
        args.velocity_percentile,
        args.speed_percentile,
        args.title + ": velocities",
    )

    lines = [
        f"position_output = {args.position_output}",
        f"velocity_output = {args.velocity_output}",
        f"prediction_start_frame = {reference_meta['prediction_start_frame']}",
        f"count_steps = {reference_meta['count_steps']}",
        "quantity\tlabel\tframes\tmagnitude_mean\tmagnitude_std\tmagnitude_p95\tcomponent_std",
    ]
    for source in velocity_sources:
        lines.append(summary_line("velocity", source["label"], source["velocity"]))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
