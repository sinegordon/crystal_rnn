"""Reconstruct path-dependent energy diagnostics from ASE forces and motion.

The model does not provide an explicit potential energy.  Along a given
trajectory we can still define a path potential through the accumulated work

    U_path(t + dt) = U_path(t) - integral F . dr

up to an arbitrary additive constant.  This diagnostic plots total-system and
OX-slab kinetic, path-potential, and total path energies.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AMU_ANG2_PER_PS2_TO_EV = 1.0364269656262175e-4
CU_MASS_AMU = 63.546


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x", help="Axis used for slab diagnostics.")
    parser.add_argument("--slabs", type=int, default=10, help="Number of slabs along the selected axis.")
    parser.add_argument("--mass-amu", type=float, default=CU_MASS_AMU, help="Atomic mass used for per-atom kinetic energy.")
    parser.add_argument(
        "--fit-fraction-start",
        type=float,
        default=0.1,
        help="Start fraction of the trajectory used for energy-drift linear fits.",
    )
    parser.add_argument(
        "--fit-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of the trajectory used for energy-drift linear fits.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=1,
        help="Plot every Nth frame. Metrics always use all frames.",
    )
    parser.add_argument("--title", default="ASE path-energy diagnostics")
    return parser.parse_args()


def axis_index(axis: str) -> int:
    """Return the integer index for an axis name."""
    return {"x": 0, "y": 1, "z": 2}[axis]


def as_cell_matrix(value: np.ndarray) -> np.ndarray:
    """Return a 3x3 cell matrix."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.ndim == 2 and cell.shape[1] == 3 and cell.shape[0] != 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,), (frames, 3), or (3, 3)")
    return cell


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap flat position differences into the nearest periodic image."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(delta, dtype=np.float64) @ inverse_cell
    fractional -= np.round(fractional)
    return fractional @ cell


def wrapped_fractional(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return wrapped fractional coordinates in [0, 1)."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(positions, dtype=np.float64) @ inverse_cell
    return fractional - np.floor(fractional)


def frame_window(frame_count: int, start_fraction: float, end_fraction: float) -> slice:
    """Return a fractional frame window for fitting."""
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("fit fractions must satisfy 0 <= start < end <= 1")
    first = int(np.floor(frame_count * start_fraction))
    last = int(np.ceil(frame_count * end_fraction))
    first = min(max(first, 0), frame_count - 2)
    last = min(max(last, first + 2), frame_count)
    return slice(first, last)


def per_atom_kinetic_energy(velocities: np.ndarray, mass_amu: float) -> np.ndarray:
    """Return per-atom kinetic energies in eV."""
    if mass_amu <= 0:
        raise ValueError("mass-amu must be positive")
    velocity_squared = np.sum(np.asarray(velocities, dtype=np.float64) ** 2, axis=2)
    return 0.5 * mass_amu * AMU_ANG2_PER_PS2_TO_EV * velocity_squared


def full_path_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    kinetic_energy_total: np.ndarray | None,
    cell: np.ndarray,
    mass_amu: float,
) -> dict[str, np.ndarray]:
    """Return total-system kinetic, path-potential, and total path energy."""
    dr = minimum_image(positions[1:] - positions[:-1], cell)
    work_increment = np.sum(0.5 * (forces[:-1] + forces[1:]) * dr, axis=(1, 2))
    potential = np.empty(positions.shape[0], dtype=np.float64)
    potential[0] = 0.0
    potential[1:] = -np.cumsum(work_increment)
    if kinetic_energy_total is None:
        kinetic = np.sum(per_atom_kinetic_energy(velocities, mass_amu), axis=1)
    else:
        kinetic = np.asarray(kinetic_energy_total, dtype=np.float64)
    return {
        "kinetic": kinetic,
        "potential": potential,
        "total": kinetic + potential,
        "work_increment": work_increment,
    }


def slab_indices(positions: np.ndarray, cell: np.ndarray, axis: int, slabs: int) -> np.ndarray:
    """Return slab indices for one frame of positions."""
    if slabs <= 0:
        raise ValueError("slabs must be positive")
    fractional_axis = wrapped_fractional(positions, cell)[:, axis]
    return np.minimum((fractional_axis * slabs).astype(np.int64), slabs - 1)


def slab_path_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    cell: np.ndarray,
    axis: int,
    slabs: int,
    mass_amu: float,
) -> dict[str, np.ndarray]:
    """Return slab-resolved kinetic, path-potential, and total path energy."""
    frame_count = int(positions.shape[0])
    kinetic_atom = per_atom_kinetic_energy(velocities, mass_amu)
    kinetic = np.zeros((frame_count, slabs), dtype=np.float64)
    potential = np.zeros((frame_count, slabs), dtype=np.float64)

    for frame in range(frame_count):
        indices = slab_indices(positions[frame], cell, axis, slabs)
        kinetic[frame] = np.bincount(indices, weights=kinetic_atom[frame], minlength=slabs)

    dr = minimum_image(positions[1:] - positions[:-1], cell)
    midpoint = positions[:-1] + 0.5 * dr
    work_atom = np.sum(0.5 * (forces[:-1] + forces[1:]) * dr, axis=2)
    for frame in range(frame_count - 1):
        indices = slab_indices(midpoint[frame], cell, axis, slabs)
        work_by_slab = np.bincount(indices, weights=work_atom[frame], minlength=slabs)
        potential[frame + 1] = potential[frame] - work_by_slab

    return {
        "kinetic": kinetic,
        "potential": potential,
        "total": kinetic + potential,
    }


def linear_drift(time_ps: np.ndarray, values: np.ndarray, fit_slice: slice) -> dict[str, float]:
    """Return linear drift statistics for one time series."""
    x = np.asarray(time_ps[fit_slice], dtype=np.float64)
    y = np.asarray(values[fit_slice], dtype=np.float64)
    if x.size < 2:
        return {"slope_ev_per_ps": float("nan"), "intercept_ev": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    centered = y - np.mean(y)
    r2 = 1.0 - float(np.sum(residual**2) / max(np.sum(centered**2), 1.0e-300))
    return {"slope_ev_per_ps": float(slope), "intercept_ev": float(intercept), "r2": float(r2)}


def summary_row(quantity: str, source: str, time_ps: np.ndarray, values: np.ndarray, fit_slice: slice) -> dict[str, float | str]:
    """Return one scalar metric row."""
    values = np.asarray(values, dtype=np.float64)
    drift = linear_drift(time_ps, values, fit_slice)
    return {
        "quantity": quantity,
        "source": source,
        "initial_ev": float(values[0]),
        "final_ev": float(values[-1]),
        "delta_ev": float(values[-1] - values[0]),
        "mean_ev": float(np.mean(values)),
        "std_ev": float(np.std(values)),
        "min_ev": float(np.min(values)),
        "max_ev": float(np.max(values)),
        "slope_ev_per_ps": drift["slope_ev_per_ps"],
        "slope_ev_per_atom_per_ps": "",
        "r2": drift["r2"],
    }


def build_metrics(
    time_ps: np.ndarray,
    full: dict[str, np.ndarray],
    slabs: dict[str, np.ndarray],
    fit_slice: slice,
    atom_count: int,
) -> list[dict[str, object]]:
    """Build metrics rows for full-system and slab-resolved energies."""
    rows: list[dict[str, object]] = []
    for key in ("kinetic", "potential", "total"):
        row = summary_row(key, "full", time_ps, full[key], fit_slice)
        row["slope_ev_per_atom_per_ps"] = row["slope_ev_per_ps"] / atom_count
        rows.append(row)

    for slab_index in range(slabs["total"].shape[1]):
        for key in ("kinetic", "potential", "total"):
            rows.append(summary_row(key, f"slab_{slab_index}", time_ps, slabs[key][:, slab_index], fit_slice))
    return rows


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write metric rows as TSV."""
    fields = [
        "quantity",
        "source",
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


def plot_energy_diagnostics(
    output_path: Path,
    time_ps: np.ndarray,
    full: dict[str, np.ndarray],
    slabs: dict[str, np.ndarray],
    fit_slice: slice,
    atom_count: int,
    args: argparse.Namespace,
) -> None:
    """Save the path-energy diagnostic figure."""
    stride = max(1, int(args.plot_stride))
    plot_slice = slice(None, None, stride)
    time_plot = time_ps[plot_slice]
    total_centered = full["total"] - full["total"][0]
    potential_centered = full["potential"] - full["potential"][0]
    kinetic_centered = full["kinetic"] - full["kinetic"][0]
    slab_total_centered = slabs["total"] - slabs["total"][0:1]
    slab_kinetic_centered = slabs["kinetic"] - slabs["kinetic"][0:1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time_plot, kinetic_centered[plot_slice], label="K - K0", lw=0.9)
    ax.plot(time_plot, potential_centered[plot_slice], label="U_path - U0", lw=0.9)
    ax.plot(time_plot, total_centered[plot_slice], label="E_path - E0", lw=1.2)
    ax.axvspan(time_ps[fit_slice.start], time_ps[fit_slice.stop - 1], color="black", alpha=0.06, label="fit window")
    ax.set_title("full-system path energy")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("energy change, eV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(time_plot, total_centered[plot_slice] / atom_count, lw=1.1, color="tab:green")
    ax.set_title("full-system path energy per atom")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("(E_path - E0) / atom, eV")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    image = ax.imshow(
        slab_total_centered[plot_slice].T,
        aspect="auto",
        origin="lower",
        extent=[time_plot[0], time_plot[-1], -0.5, args.slabs - 0.5],
        cmap="coolwarm",
    )
    ax.set_title(f"slab E_path - E_path(0), O{args.axis.upper()}")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("slab index")
    plt.colorbar(image, ax=ax, label="eV")

    ax = axes[1, 1]
    image = ax.imshow(
        slab_kinetic_centered[plot_slice].T,
        aspect="auto",
        origin="lower",
        extent=[time_plot[0], time_plot[-1], -0.5, args.slabs - 0.5],
        cmap="viridis",
    )
    ax.set_title(f"slab K - K(0), O{args.axis.upper()}")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("slab index")
    plt.colorbar(image, ax=ax, label="eV")

    drift = linear_drift(time_ps, full["total"], fit_slice)
    fig.suptitle(
        f"{args.title}; full E drift = {drift['slope_ev_per_ps']:.4g} eV/ps",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Run path-energy postprocessing."""
    args = parse_args()
    if args.slabs <= 0:
        raise ValueError("slabs must be positive")
    if args.plot_stride <= 0:
        raise ValueError("plot-stride must be positive")

    ase = np.load(args.ase_path)
    required = ["positions", "velocities_ang_per_ps", "forces_ev_per_ang", "cell", "dt_ps"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"Missing arrays in ASE output: {missing}")

    positions = np.asarray(ase["positions"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
    frame_count = min(int(positions.shape[0]), int(velocities.shape[0]), int(forces.shape[0]))
    positions = positions[:frame_count]
    velocities = velocities[:frame_count]
    forces = forces[:frame_count]
    cell = as_cell_matrix(ase["cell"])
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64))
    steps = np.asarray(ase["steps"], dtype=np.float64)[:frame_count] if "steps" in ase.files else np.arange(frame_count)
    time_ps = steps * dt_ps
    kinetic_total = np.asarray(ase["kinetic_energy_ev"], dtype=np.float64)[:frame_count] if "kinetic_energy_ev" in ase.files else None

    full = full_path_energy(positions, velocities, forces, kinetic_total, cell, args.mass_amu)
    slabs = slab_path_energy(positions, velocities, forces, cell, axis_index(args.axis), args.slabs, args.mass_amu)
    fit_slice = frame_window(frame_count, args.fit_fraction_start, args.fit_fraction_end)
    rows = build_metrics(time_ps, full, slabs, fit_slice, atom_count=int(positions.shape[1]))

    output_path = Path(args.output_path)
    plot_energy_diagnostics(output_path, time_ps, full, slabs, fit_slice, int(positions.shape[1]), args)
    print(f"Saved {output_path}")

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        write_metrics(metrics_path, rows)
        print(f"Saved {metrics_path}")

    total_row = next(row for row in rows if row["quantity"] == "total" and row["source"] == "full")
    print(
        "full path-energy delta/slope = "
        f"{total_row['delta_ev']:.6g} eV / {total_row['slope_ev_per_ps']:.6g} eV/ps"
    )
    print(f"full path-energy slope per atom = {total_row['slope_ev_per_atom_per_ps']:.6g} eV/atom/ps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
