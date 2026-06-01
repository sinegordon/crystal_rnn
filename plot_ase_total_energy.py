"""Plot direct ASE total energy saved by an energy-aware calculator."""

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
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--title", default="ASE direct E_tot diagnostics")
    return parser.parse_args()


def load_direct_energy(path: Path) -> dict[str, np.ndarray]:
    """Load time, direct energies, and temperature from an ASE output file."""
    ase = np.load(path)
    required = ["kinetic_energy_ev", "potential_energy_ev", "temperature_k"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"Missing arrays in ASE output: {missing}")

    kinetic = np.asarray(ase["kinetic_energy_ev"], dtype=np.float64)
    potential = np.asarray(ase["potential_energy_ev"], dtype=np.float64)
    temperature = np.asarray(ase["temperature_k"], dtype=np.float64)
    frame_count = min(kinetic.size, potential.size, temperature.size)
    kinetic = kinetic[:frame_count]
    potential = potential[:frame_count]
    temperature = temperature[:frame_count]

    if "steps" in ase.files and "dt_ps" in ase.files:
        time_ps = np.asarray(ase["steps"], dtype=np.float64)[:frame_count] * float(np.asarray(ase["dt_ps"]))
    elif "dt_ps" in ase.files:
        time_ps = np.arange(frame_count, dtype=np.float64) * float(np.asarray(ase["dt_ps"]))
    else:
        time_ps = np.arange(frame_count, dtype=np.float64)

    atom_count = int(ase["positions"].shape[1]) if "positions" in ase.files else 0

    return {
        "time_ps": time_ps,
        "kinetic": kinetic,
        "potential": potential,
        "total": kinetic + potential,
        "temperature": temperature,
        "atom_count": np.asarray(atom_count, dtype=np.int64),
    }


def energy_metrics(time_ps: np.ndarray, values: np.ndarray, prefix: str, atom_count: int) -> dict[str, object]:
    """Return compact drift metrics for one energy trace."""
    slope, intercept = np.polyfit(time_ps, values, 1)
    fit = slope * time_ps + intercept
    residual = values - fit
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    slope_per_atom = float(slope / atom_count) if atom_count > 0 else float("nan")
    return {
        "quantity": prefix,
        "atom_count": atom_count,
        "initial_ev": float(values[0]),
        "final_ev": float(values[-1]),
        "delta_ev": float(values[-1] - values[0]),
        "mean_ev": float(np.mean(values)),
        "std_ev": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "min_ev": float(np.min(values)),
        "max_ev": float(np.max(values)),
        "slope_ev_per_ps": float(slope),
        "slope_ev_per_atom_per_ps": slope_per_atom,
        "r2": r2,
    }


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write energy metrics as TSV."""
    fields = [
        "quantity",
        "atom_count",
        "initial_ev",
        "final_ev",
        "delta_ev",
        "mean_ev",
        "std_ev",
        "min_ev",
        "max_ev",
        "slope_ev_per_ps",
        "slope_ev_per_atom_per_ps",
        "r2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_energy(output_path: Path, arrays: dict[str, np.ndarray], title: str) -> None:
    """Save direct energy and temperature diagnostics."""
    time_ps = arrays["time_ps"]
    total = arrays["total"]
    kinetic = arrays["kinetic"]
    potential = arrays["potential"]
    temperature = arrays["temperature"]
    slope, intercept = np.polyfit(time_ps, total, 1)
    fit = slope * time_ps + intercept
    total_centered = total - np.mean(total)
    fit_centered = fit - np.mean(total)
    kinetic_centered = kinetic - np.mean(kinetic)
    potential_centered = potential - np.mean(potential)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(time_ps, total_centered, lw=1.0, label=r"$E_{tot} - \langle E_{tot} \rangle$")
    ax.plot(time_ps, fit_centered, "--", lw=1.0, color="black", label=f"slope {slope:.4g} eV/ps")
    ax.axhline(0.0, color="0.4", lw=0.8, alpha=0.5)
    ax.set_ylabel(r"$\Delta E_{tot}$, eV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(time_ps, potential_centered, lw=0.8, label=r"$E_{pot} - \langle E_{pot} \rangle$")
    ax.plot(time_ps, kinetic_centered, lw=0.8, label=r"$E_{kin} - \langle E_{kin} \rangle$")
    ax.axhline(0.0, color="0.4", lw=0.8, alpha=0.5)
    ax.set_ylabel(r"$\Delta E$, eV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(time_ps, temperature, lw=1.0, color="tab:red")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("temperature, K")
    ax.grid(alpha=0.25)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Run direct total-energy postprocessing."""
    args = parse_args()
    arrays = load_direct_energy(Path(args.ase_path))
    output_path = Path(args.output_path)
    plot_energy(output_path, arrays, args.title)
    print(f"Saved {output_path}")

    atom_count = int(np.asarray(arrays["atom_count"]))
    rows = [
        energy_metrics(arrays["time_ps"], arrays["total"], "total", atom_count),
        energy_metrics(arrays["time_ps"], arrays["potential"], "potential", atom_count),
        energy_metrics(arrays["time_ps"], arrays["kinetic"], "kinetic", atom_count),
    ]
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        write_metrics(metrics_path, rows)
        print(f"Saved {metrics_path}")

    total = rows[0]
    print(
        "direct E_tot delta/slope = "
        f"{total['delta_ev']:.6g} eV / {total['slope_ev_per_ps']:.6g} eV/ps"
    )
    print(f"direct E_tot slope per atom = {total['slope_ev_per_atom_per_ps']:.6g} eV/atom/ps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
