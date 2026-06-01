"""Plot phase-resolved power and acceleration histograms for an ASE run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from base_classes import CU_MASS_AMU, forces_to_discrete_accelerations


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--data-path", default=None, help="Optional prepared .npz used to read atom mass.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--window-frames", type=int, default=1000, help="Frames per begin/middle/end phase.")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--percentile", type=float, default=99.0, help="Robust range percentile.")
    parser.add_argument("--title", default="ASE phase power and acceleration diagnostics")
    return parser.parse_args()


def load_atom_mass(data_path: str | None) -> float:
    """Return atom mass from the reference dataset, falling back to copper."""
    if data_path is None:
        return CU_MASS_AMU
    data = np.load(data_path)
    if "atom_mass_amu" not in data.files:
        return CU_MASS_AMU
    return float(np.asarray(data["atom_mass_amu"], dtype=np.float64))


def phase_slices(frame_count: int, window_frames: int) -> dict[str, slice]:
    """Return beginning, middle, and end frame slices."""
    if window_frames <= 0:
        raise ValueError("window-frames must be positive")
    if frame_count < 3:
        raise ValueError("ASE trajectory needs at least three frames")
    window = min(int(window_frames), frame_count)
    middle_start = max(0, frame_count // 2 - window // 2)
    middle_start = min(middle_start, frame_count - window)
    return {
        "begin": slice(0, window),
        "middle": slice(middle_start, middle_start + window),
        "end": slice(frame_count - window, frame_count),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    """Return compact scalar summary statistics."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "std", "rms", "p05", "p50", "p95", "p99")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def robust_range(series: list[np.ndarray], percentile: float, symmetric: bool) -> tuple[float, float]:
    """Return a robust plotting range shared by several histograms."""
    values = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        limit = float(np.percentile(np.abs(values), percentile))
        return (-limit, limit) if limit > 0 else (-1.0, 1.0)
    low = float(np.percentile(values, 100.0 - percentile))
    high = float(np.percentile(values, percentile))
    return (low, high) if high > low else (0.0, 1.0)


def collect_phase_values(ase_path: Path, data_path: str | None, window_frames: int):
    """Load ASE arrays and collect phase-resolved power/acceleration values."""
    ase = np.load(ase_path)
    required = ["forces_ev_per_ang", "velocities_ang_per_ps", "dt_ps"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"{ase_path} is missing required arrays: {missing}")

    mass = load_atom_mass(data_path)
    forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    acceleration = forces_to_discrete_accelerations(
        forces,
        dt_ps=float(np.asarray(ase["dt_ps"], dtype=np.float64)),
        atom_mass_amu=mass,
    ).astype(np.float64)
    if acceleration.shape != velocities.shape:
        raise ValueError(f"acceleration shape {acceleration.shape} does not match velocities {velocities.shape}")

    phases = phase_slices(acceleration.shape[0], window_frames)
    values = {}
    rows = []
    for phase, slc in phases.items():
        a = acceleration[slc]
        v = velocities[slc]
        power = np.sum(a * v, axis=-1).reshape(-1)
        component_power = (a * v).reshape(-1)
        acceleration_component = a.reshape(-1)
        acceleration_norm = np.linalg.norm(a, axis=-1).reshape(-1)
        numerator = float(np.mean(a * v))
        denominator = max(float(np.sqrt(np.mean(a**2)) * np.sqrt(np.mean(v**2))), 1e-30)
        normalized_power = numerator / denominator
        values[phase] = {
            "atom_power": power,
            "component_power": component_power,
            "acceleration_component": acceleration_component,
            "acceleration_norm": acceleration_norm,
        }
        for quantity, array in values[phase].items():
            rows.append(
                {
                    "phase": phase,
                    "quantity": quantity,
                    "count": int(array.size),
                    "normalized_power": normalized_power if quantity == "atom_power" else "",
                    **summarize(array),
                }
            )
    return values, rows


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write phase metrics as TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phase", "quantity", "count", "mean", "std", "rms", "p05", "p50", "p95", "p99", "normalized_power"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_phase_histograms(
    output_path: Path,
    values: dict[str, dict[str, np.ndarray]],
    bins: int,
    percentile: float,
    title: str,
) -> None:
    """Save phase histograms for power and acceleration values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phase_names = ["begin", "middle", "end"]
    panels = [
        ("atom a·v", "atom_power", True),
        ("component a·v", "component_power", True),
        ("acceleration component", "acceleration_component", True),
        ("|acceleration|", "acceleration_norm", False),
    ]
    colors = {"begin": "tab:blue", "middle": "tab:orange", "end": "tab:green"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    fig.suptitle(title)
    for axis, (panel_title, key, symmetric) in zip(axes.ravel(), panels):
        arrays = [values[phase][key] for phase in phase_names]
        hist_range = robust_range(arrays, percentile, symmetric=symmetric)
        for phase, array in zip(phase_names, arrays):
            axis.hist(
                array,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=colors[phase],
                label=phase,
            )
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.set_title(panel_title)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Run phase power/acceleration diagnostics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    values, rows = collect_phase_values(Path(args.ase_path), args.data_path, args.window_frames)
    plot_phase_histograms(Path(args.output_path), values, args.bins, args.percentile, args.title)
    print(f"Saved {args.output_path}")
    if args.metrics_path:
        write_metrics(Path(args.metrics_path), rows)
        print(f"Saved {args.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
