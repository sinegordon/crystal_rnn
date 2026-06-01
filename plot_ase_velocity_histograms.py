"""Plot ASE velocity histograms over the full inference trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sqw_comparison import crystal_frames_to_flat_positions


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE trajectory .npz.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics output.")
    parser.add_argument("--window-frames", type=int, default=10, help="Frames in first/last diagnostic windows.")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Robust component range percentile.")
    parser.add_argument("--speed-percentile", type=float, default=99.5, help="Robust speed range percentile.")
    parser.add_argument("--title", default="ASE NVT velocity histograms")
    return parser.parse_args()


def components(values):
    """Return flattened velocity components."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def speeds(values):
    """Return flattened velocity magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def robust_component_limit(*arrays, percentile):
    """Return a symmetric robust component histogram limit."""
    values = np.concatenate([np.abs(components(array)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(values, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def robust_speed_limit(*arrays, percentile):
    """Return a robust positive speed histogram limit."""
    values = np.concatenate([speeds(array) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(values, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def load_reference_velocities(data, start, dt_ps):
    """Return all available reference velocities from the prepared dataset."""
    reference_displacements = np.asarray(data["displacements"][start:], dtype=np.float32)
    if reference_displacements.shape[0] < 2:
        raise ValueError(
            f"Need at least two reference frames for velocities, got {reference_displacements.shape[0]}"
        )
    reference_flat = crystal_frames_to_flat_positions(
        reference_displacements,
        data["reference_positions"],
        data["atom_order"],
    )
    reference_positions = reference_flat.reshape(reference_displacements.shape[0], -1, 3).astype(np.float64)
    return np.diff(reference_positions, axis=0) / float(dt_ps)


def window_sets(predicted_velocities, reference_velocities, window):
    """Return full, first-window, and last-window velocity sets."""
    if window <= 0:
        raise ValueError("window-frames must be positive")
    if predicted_velocities.shape[0] < window or reference_velocities.shape[0] < window:
        raise ValueError(
            "Not enough velocity frames for requested window: "
            f"inference={predicted_velocities.shape[0]}, reference={reference_velocities.shape[0]}, window={window}"
        )
    return [
        ("all available", predicted_velocities, reference_velocities),
        (f"first {window}", predicted_velocities[:window], reference_velocities[:window]),
        (f"last {window}", predicted_velocities[-window:], reference_velocities[-window:]),
    ]


def summarize(label, source, values):
    """Return one TSV summary line."""
    speed = speeds(values)
    component = components(values)
    return (
        f"{label}\t{source}\t{values.shape[0]}\t{speed.mean():.6g}\t{speed.std():.6g}\t"
        f"{np.percentile(speed, 95):.6g}\t{np.percentile(speed, 99):.6g}\t"
        f"{component.std():.6g}"
    )


def main():
    """Build full-inference velocity histograms and metrics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if not 0 < args.speed_percentile <= 100:
        raise ValueError("speed-percentile must be in (0, 100]")

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0

    predicted_velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    reference_velocities = load_reference_velocities(data, start=start, dt_ps=dt_ps)
    sets = window_sets(predicted_velocities, reference_velocities, args.window_frames)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for column, (label, predicted, reference) in enumerate(sets):
        limit = robust_component_limit(predicted, reference, percentile=args.percentile)
        ax = axes[0, column]
        ax.hist(
            components(reference),
            bins=args.bins,
            range=(-limit, limit),
            density=True,
            alpha=0.48,
            label="reference available",
            color="tab:blue",
        )
        ax.hist(
            components(predicted),
            bins=args.bins,
            range=(-limit, limit),
            density=True,
            alpha=0.48,
            label="ASE inference",
            color="tab:orange",
        )
        ax.set_title(f"v components, {label}")
        ax.set_xlabel("v component, A/ps")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=9)

        limit = robust_speed_limit(predicted, reference, percentile=args.speed_percentile)
        ax = axes[1, column]
        ax.hist(
            speeds(reference),
            bins=args.bins,
            range=(0, limit),
            density=True,
            alpha=0.48,
            label="reference available",
            color="tab:blue",
        )
        ax.hist(
            speeds(predicted),
            bins=args.bins,
            range=(0, limit),
            density=True,
            alpha=0.48,
            label="ASE inference",
            color="tab:orange",
        )
        ax.set_title(f"|v|, {label}")
        ax.set_xlabel("|v|, A/ps")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)

    fig.suptitle(args.title, fontsize=15)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Saved {output_path}",
        f"inference_velocity_frames = {predicted_velocities.shape[0]}",
        f"reference_velocity_intervals = {reference_velocities.shape[0]}",
        f"reference_frame_start = {start}",
        "window\tsource\tframes\tspeed_mean\tspeed_std\tspeed_p95\tspeed_p99\tcomponent_std",
    ]
    for label, predicted, reference in sets:
        lines.append(summarize(label, "ASE", predicted))
        lines.append(summarize(label, "reference", reference))

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
