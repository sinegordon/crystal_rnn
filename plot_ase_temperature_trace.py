"""Plot the temperature trace saved by an ASE NVT inference run."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--metrics-path", default=None, help="Optional text metrics path.")
    parser.add_argument("--target-temperature-k", type=float, default=None, help="Optional target temperature line.")
    parser.add_argument("--title", default="ASE NVT temperature trace", help="Plot title.")
    return parser.parse_args()


def load_temperature(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return time in ps, temperature in K, and the saved target temperature if present."""
    data = np.load(path)
    if "temperature_k" not in data.files:
        raise ValueError(f"{path} does not contain 'temperature_k'")
    temperature = np.asarray(data["temperature_k"], dtype=np.float64)
    if temperature.ndim != 1 or temperature.size == 0:
        raise ValueError("'temperature_k' must be a non-empty 1D array")

    if "steps" in data.files and "dt_ps" in data.files:
        time_ps = np.asarray(data["steps"], dtype=np.float64) * float(np.asarray(data["dt_ps"]))
    elif "dt_ps" in data.files:
        time_ps = np.arange(temperature.size, dtype=np.float64) * float(np.asarray(data["dt_ps"]))
    else:
        time_ps = np.arange(temperature.size, dtype=np.float64)

    target = None
    if "temperature_target_k" in data.files:
        target = float(np.asarray(data["temperature_target_k"]))
    return time_ps, temperature, target


def write_metrics(path: Path, time_ps: np.ndarray, temperature: np.ndarray) -> None:
    """Save compact temperature statistics and the full trace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"temperature first/last/mean/min/max = "
        f"{temperature[0]:.8g} {temperature[-1]:.8g} "
        f"{temperature.mean():.8g} {temperature.min():.8g} {temperature.max():.8g}\n"
        "columns: time_ps temperature_k"
    )
    np.savetxt(path, np.column_stack([time_ps, temperature]), header=header)


def plot_temperature(
    output_path: Path,
    time_ps: np.ndarray,
    temperature: np.ndarray,
    target_temperature_k: float | None,
    title: str,
) -> None:
    """Save a temperature trace figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.plot(time_ps, temperature, lw=1.2, color="#1f77b4", label="ASE inference")
    if target_temperature_k is not None:
        ax.axhline(
            target_temperature_k,
            color="black",
            lw=1.0,
            ls="--",
            alpha=0.65,
            label=f"target {target_temperature_k:g} K",
        )
    ax.set_xlabel("time, ps")
    ax.set_ylabel("temperature, K")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    """Run the temperature post-processing step."""
    args = parse_args()
    time_ps, temperature, saved_target = load_temperature(Path(args.ase_path))
    target_temperature_k = args.target_temperature_k if args.target_temperature_k is not None else saved_target
    plot_temperature(Path(args.output_path), time_ps, temperature, target_temperature_k, args.title)
    print(f"Saved {args.output_path}")
    print(
        "temperature first/last/mean/min/max = "
        f"{temperature[0]:.8g} {temperature[-1]:.8g} "
        f"{temperature.mean():.8g} {temperature.min():.8g} {temperature.max():.8g}"
    )
    if args.metrics_path:
        write_metrics(Path(args.metrics_path), time_ps, temperature)
        print(f"Saved {args.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
