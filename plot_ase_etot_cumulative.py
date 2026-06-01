"""Plot cumulative direct ASE E_tot diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--series-path", default=None, help="Optional sampled TSV time series path.")
    parser.add_argument("--summary-path", default=None, help="Optional TSV summary path.")
    parser.add_argument("--title", default="ASE cumulative direct E_tot diagnostics")
    parser.add_argument("--smooth-window", type=int, default=501, help="Odd moving-average window in frames.")
    parser.add_argument("--series-stride", type=int, default=10, help="Frame stride for sampled TSV output.")
    return parser.parse_args()


def load_etot(path: Path) -> dict[str, np.ndarray]:
    """Load direct E_tot arrays from ASE output."""
    ase = np.load(path)
    required = ["kinetic_energy_ev", "potential_energy_ev", "dt_ps"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"Missing arrays in ASE output: {missing}")

    kinetic = np.asarray(ase["kinetic_energy_ev"], dtype=np.float64)
    potential = np.asarray(ase["potential_energy_ev"], dtype=np.float64)
    if "steps" in ase.files:
        steps = np.asarray(ase["steps"], dtype=np.float64)
    else:
        steps = np.arange(kinetic.size, dtype=np.float64)

    frame_count = min(kinetic.size, potential.size, steps.size)
    kinetic = kinetic[:frame_count]
    potential = potential[:frame_count]
    steps = steps[:frame_count]
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64))
    time_ps = steps * dt_ps
    total = kinetic + potential
    return {
        "time_ps": time_ps,
        "total": total,
        "kinetic": kinetic,
        "potential": potential,
    }


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    """Return a centered moving average and the effective odd window."""
    if window <= 1:
        return values, 1
    effective = min(int(window), max(1, values.size))
    if effective % 2 == 0:
        effective -= 1
    effective = max(1, effective)
    if effective <= 1:
        return values, 1
    kernel = np.ones(effective, dtype=np.float64) / effective
    return np.convolve(values, kernel, mode="same"), effective


def build_cumulative(arrays: dict[str, np.ndarray], smooth_window: int) -> dict[str, np.ndarray | float | int]:
    """Build centered and cumulative E_tot diagnostics."""
    time_ps = arrays["time_ps"]
    total = arrays["total"]
    mean_total = float(np.mean(total))
    initial_total = float(total[0])
    centered = total - mean_total
    running_mean = np.cumsum(total) / np.arange(1, total.size + 1, dtype=np.float64)
    running_mean_minus_mean = running_mean - mean_total
    running_mean_minus_initial = running_mean - initial_total
    dt = np.diff(time_ps, prepend=time_ps[0])
    cumulative_centered_integral = np.cumsum(centered * dt)
    smoothed_centered, effective_window = moving_average(centered, smooth_window)
    return {
        "time_ps": time_ps,
        "total": total,
        "centered": centered,
        "running_mean": running_mean,
        "running_mean_minus_mean": running_mean_minus_mean,
        "running_mean_minus_initial": running_mean_minus_initial,
        "cumulative_centered_integral": cumulative_centered_integral,
        "smoothed_centered": smoothed_centered,
        "mean_total": mean_total,
        "initial_total": initial_total,
        "effective_window": effective_window,
    }


def plot_cumulative(output_path: Path, data: dict[str, np.ndarray | float | int], title: str) -> None:
    """Save cumulative E_tot diagnostics."""
    time_ps = np.asarray(data["time_ps"], dtype=np.float64)
    centered = np.asarray(data["centered"], dtype=np.float64)
    smoothed = np.asarray(data["smoothed_centered"], dtype=np.float64)
    running_mean_minus_mean = np.asarray(data["running_mean_minus_mean"], dtype=np.float64)
    running_mean_minus_initial = np.asarray(data["running_mean_minus_initial"], dtype=np.float64)
    cumulative_integral = np.asarray(data["cumulative_centered_integral"], dtype=np.float64)
    effective_window = int(data["effective_window"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(time_ps, centered, lw=0.35, alpha=0.35, label=r"$E_{tot} - \langle E_{tot} \rangle$")
    ax.plot(time_ps, smoothed, lw=1.0, color="tab:blue", label=f"smoothed, window={effective_window}")
    ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
    ax.set_ylabel(r"$\Delta E_{tot}$, eV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(time_ps, running_mean_minus_mean, lw=1.2, color="tab:green", label="running mean minus full mean")
    ax.plot(time_ps, running_mean_minus_initial, lw=0.9, color="tab:olive", alpha=0.7, label="running mean minus initial")
    ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
    ax.set_ylabel("cumulative mean, eV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(time_ps, cumulative_integral, lw=1.1, color="tab:purple")
    ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
    ax.set_xlabel("time, ps")
    ax.set_ylabel(r"$\int (E_{tot}-\langle E_{tot}\rangle) dt$, eV ps")
    ax.grid(alpha=0.25)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_series(path: Path, data: dict[str, np.ndarray | float | int], stride: int) -> None:
    """Write a sampled cumulative time series."""
    if stride <= 0:
        raise ValueError("series-stride must be positive")
    fields = [
        "time_ps",
        "etot_ev",
        "etot_centered_ev",
        "etot_cumulative_mean_ev",
        "etot_cumulative_mean_minus_full_mean_ev",
        "etot_cumulative_mean_minus_initial_ev",
        "etot_cumulative_centered_integral_ev_ps",
    ]
    arrays = {key: np.asarray(data[key], dtype=np.float64) for key in [
        "time_ps",
        "total",
        "centered",
        "running_mean",
        "running_mean_minus_mean",
        "running_mean_minus_initial",
        "cumulative_centered_integral",
    ]}
    frame_count = arrays["time_ps"].size
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        for index in range(frame_count):
            if index % stride != 0 and index != frame_count - 1:
                continue
            writer.writerow([
                f"{arrays['time_ps'][index]:.10g}",
                f"{arrays['total'][index]:.12g}",
                f"{arrays['centered'][index]:.12g}",
                f"{arrays['running_mean'][index]:.12g}",
                f"{arrays['running_mean_minus_mean'][index]:.12g}",
                f"{arrays['running_mean_minus_initial'][index]:.12g}",
                f"{arrays['cumulative_centered_integral'][index]:.12g}",
            ])


def write_summary(path: Path, data: dict[str, np.ndarray | float | int]) -> None:
    """Write compact cumulative diagnostics."""
    rows = {
        "frame_count": int(np.asarray(data["time_ps"]).size),
        "time_final_ps": float(np.asarray(data["time_ps"])[-1]),
        "etot_initial_ev": float(data["initial_total"]),
        "etot_mean_ev": float(data["mean_total"]),
        "final_running_mean_minus_full_mean_ev": float(np.asarray(data["running_mean_minus_mean"])[-1]),
        "final_running_mean_minus_initial_ev": float(np.asarray(data["running_mean_minus_initial"])[-1]),
        "final_cumulative_centered_integral_ev_ps": float(np.asarray(data["cumulative_centered_integral"])[-1]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["quantity", "value"])
        for key, value in rows.items():
            writer.writerow([key, value])


def main() -> int:
    """Run cumulative direct E_tot postprocessing."""
    args = parse_args()
    arrays = load_etot(Path(args.ase_path))
    data = build_cumulative(arrays, args.smooth_window)
    plot_cumulative(Path(args.output_path), data, args.title)
    print(f"Saved {args.output_path}")
    if args.series_path:
        write_series(Path(args.series_path), data, args.series_stride)
        print(f"Saved {args.series_path}")
    if args.summary_path:
        write_summary(Path(args.summary_path), data)
        print(f"Saved {args.summary_path}")
    print(f"final running mean minus full mean = {float(np.asarray(data['running_mean_minus_mean'])[-1]):.6g} eV")
    print(f"final running mean minus initial = {float(np.asarray(data['running_mean_minus_initial'])[-1]):.6g} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
