"""Estimate heat-capacity diagnostics from ASE path-energy fluctuations.

For a canonical trajectory with a true total energy, the fluctuation formula is

    C_V = Var(E) / (k_B T^2).

This script intentionally uses the trajectory-dependent path energy reconstructed
from model forces.  Energy-aware models can additionally be analyzed with
``plot_ase_etot_heat_capacity.py``, which uses the direct ASE
``E_tot = E_pot_model + E_kin`` trace.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ase_path_energy import CU_MASS_AMU, as_cell_matrix, full_path_energy


KB_EV_PER_K = 8.617333262145e-5
EV_TO_J = 1.602176634e-19
AVOGADRO = 6.02214076e23
EV_PER_ATOM_K_TO_J_PER_MOL_K = EV_TO_J * AVOGADRO


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--mass-amu", type=float, default=CU_MASS_AMU, help="Atomic mass used if K is missing.")
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=None,
        help="Temperature used in C_V. Defaults to mean ASE temperature in the selected window.",
    )
    parser.add_argument(
        "--frame-fraction-start",
        type=float,
        default=0.0,
        help="Start fraction of the trajectory used for heat-capacity statistics.",
    )
    parser.add_argument(
        "--frame-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of the trajectory used for heat-capacity statistics.",
    )
    parser.add_argument(
        "--detrend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove a linear drift before the main fluctuation estimate.",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="*",
        default=[250, 500, 1000, 2500, 5000],
        help="Frame block sizes used for blockwise C_V estimates.",
    )
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--title", default="ASE energy heat-capacity diagnostic")
    return parser.parse_args()


def validate_fraction_window(start: float, end: float) -> None:
    """Validate a fractional frame window."""
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Frame fractions must satisfy 0 <= start < end <= 1")


def fraction_slice(frame_count: int, start: float, end: float) -> slice:
    """Convert fractional bounds into a frame slice."""
    validate_fraction_window(start, end)
    first = int(np.floor(frame_count * start))
    last = int(np.ceil(frame_count * end))
    first = min(max(first, 0), frame_count - 4)
    last = min(max(last, first + 4), frame_count)
    return slice(first, last)


def load_energy_arrays(
    path: Path,
    frame_fraction_start: float,
    frame_fraction_end: float,
    mass_amu: float,
):
    """Load ASE arrays and reconstruct full-system path energies."""
    ase = np.load(path)
    required = ["positions", "velocities_ang_per_ps", "forces_ev_per_ang", "cell", "dt_ps"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"Missing arrays in ASE output: {missing}")

    positions = np.asarray(ase["positions"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
    frame_count = min(int(positions.shape[0]), int(velocities.shape[0]), int(forces.shape[0]))
    frame_slice = fraction_slice(frame_count, frame_fraction_start, frame_fraction_end)
    positions = positions[:frame_count][frame_slice]
    velocities = velocities[:frame_count][frame_slice]
    forces = forces[:frame_count][frame_slice]
    kinetic_total = (
        np.asarray(ase["kinetic_energy_ev"], dtype=np.float64)[:frame_count][frame_slice]
        if "kinetic_energy_ev" in ase.files
        else None
    )
    temperature = (
        np.asarray(ase["temperature_k"], dtype=np.float64)[:frame_count][frame_slice]
        if "temperature_k" in ase.files
        else None
    )
    steps = np.asarray(ase["steps"], dtype=np.float64)[:frame_count] if "steps" in ase.files else np.arange(frame_count)
    steps = steps[frame_slice]
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64))
    cell = as_cell_matrix(ase["cell"])
    energy = full_path_energy(positions, velocities, forces, kinetic_total, cell, mass_amu)
    return {
        "time_ps": steps * dt_ps,
        "temperature": temperature,
        "kinetic": energy["kinetic"],
        "potential": energy["potential"],
        "total": energy["total"],
        "atom_count": int(positions.shape[1]),
        "dt_ps": dt_ps,
        "energy_source": "path",
    }


def linear_detrend(time_ps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return values with a fitted linear trend removed, plus slope and intercept."""
    time_ps = np.asarray(time_ps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    slope, intercept = np.polyfit(time_ps, values, 1)
    trend = slope * time_ps + intercept
    detrended = values - trend + np.mean(trend)
    return detrended, float(slope), float(intercept)


def sample_variance(values: np.ndarray) -> float:
    """Return sample variance with ddof=1 when possible."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    return float(np.var(values, ddof=1))


def cv_from_energy(values: np.ndarray, temperature_k: float) -> dict[str, float]:
    """Return heat-capacity estimates from energy fluctuations."""
    if temperature_k <= 0:
        raise ValueError("temperature must be positive")
    variance = sample_variance(values)
    cv_e_v_per_k = variance / (KB_EV_PER_K * temperature_k**2)
    return {
        "variance_ev2": variance,
        "cv_e_v_per_k": cv_e_v_per_k,
    }


def convert_cv(cv_e_v_per_k: float, atom_count: int) -> dict[str, float]:
    """Return C_V in per-atom and molar units."""
    per_atom_e_v_per_k = cv_e_v_per_k / atom_count
    return {
        "cv_e_v_per_k": cv_e_v_per_k,
        "cv_e_v_per_atom_k": per_atom_e_v_per_k,
        "cv_kb_per_atom": per_atom_e_v_per_k / KB_EV_PER_K,
        "cv_j_per_mol_k": per_atom_e_v_per_k * EV_PER_ATOM_K_TO_J_PER_MOL_K,
    }


def block_cv_estimates(
    values: np.ndarray,
    time_ps: np.ndarray,
    temperature_k: float,
    atom_count: int,
    block_sizes: list[int],
    per_block_detrend: bool,
):
    """Return blockwise heat-capacity estimates for requested block sizes."""
    rows = []
    values = np.asarray(values, dtype=np.float64)
    time_ps = np.asarray(time_ps, dtype=np.float64)
    for block_size in block_sizes:
        if block_size < 4 or block_size > values.size:
            continue
        block_count = values.size // block_size
        if block_count < 1:
            continue
        block_values = values[: block_count * block_size].reshape(block_count, block_size)
        block_times = time_ps[: block_count * block_size].reshape(block_count, block_size)
        cv_values = []
        for block, block_time in zip(block_values, block_times):
            if per_block_detrend:
                block, _slope, _intercept = linear_detrend(block_time, block)
            cv_values.append(convert_cv(cv_from_energy(block, temperature_k)["cv_e_v_per_k"], atom_count))
        rows.append(
            {
                "block_size": int(block_size),
                "block_count": int(block_count),
                "per_block_detrend": bool(per_block_detrend),
                "mean_kb_per_atom": float(np.mean([row["cv_kb_per_atom"] for row in cv_values])),
                "std_kb_per_atom": float(np.std([row["cv_kb_per_atom"] for row in cv_values])),
            }
        )
    return rows


def metric_row(quantity: str, source: str, value: float, unit: str, note: str = "") -> dict[str, object]:
    """Return one TSV metric row."""
    return {"quantity": quantity, "source": source, "value": value, "unit": unit, "note": note}


def build_metrics(
    arrays: dict[str, np.ndarray | float | int],
    temperature_k: float,
    raw_total: dict[str, float],
    detrended_total: dict[str, float],
    kinetic_cv: dict[str, float],
    block_rows: list[dict[str, float]],
    total_slope_ev_per_ps: float,
    use_detrended: bool,
) -> list[dict[str, object]]:
    """Build TSV metric rows."""
    atom_count = int(arrays["atom_count"])
    rows: list[dict[str, object]] = [
        metric_row("temperature", "metadata", temperature_k, "K"),
        metric_row("frames", "metadata", int(np.asarray(arrays["total"]).size), "count"),
        metric_row("atom_count", "metadata", atom_count, "count"),
        metric_row("classical_dulong_petit", "reference", 3.0, "kB/atom"),
        metric_row("total_energy_slope", str(arrays.get("energy_source", "energy")), total_slope_ev_per_ps, "eV/ps"),
        metric_row("detrend_enabled", "metadata", int(use_detrended), "bool"),
    ]
    for source, cv in [("raw_total", raw_total), ("detrended_total", detrended_total), ("kinetic", kinetic_cv)]:
        converted = convert_cv(cv["cv_e_v_per_k"], atom_count)
        rows.extend(
            [
                metric_row("energy_variance", source, cv["variance_ev2"], "eV^2"),
                metric_row("cv_total", source, converted["cv_e_v_per_k"], "eV/K"),
                metric_row("cv_per_atom", source, converted["cv_kb_per_atom"], "kB/atom"),
                metric_row("cv_molar", source, converted["cv_j_per_mol_k"], "J/mol/K"),
            ]
        )
    for block in block_rows:
        rows.append(
            metric_row(
                "block_cv_mean",
                f"block_{block['block_size']}",
                block["mean_kb_per_atom"],
                "kB/atom",
                f"block_count={block['block_count']}; per_block_detrend={int(block['per_block_detrend'])}",
            )
        )
        rows.append(
            metric_row(
                "block_cv_std",
                f"block_{block['block_size']}",
                block["std_kb_per_atom"],
                "kB/atom",
                f"block_count={block['block_count']}; per_block_detrend={int(block['per_block_detrend'])}",
            )
        )
    return rows


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write metric rows as TSV."""
    fields = ["quantity", "source", "value", "unit", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_heat_capacity(
    output_path: Path,
    arrays: dict[str, np.ndarray | float | int],
    total_for_estimate: np.ndarray,
    raw_total_cv: dict[str, float],
    detrended_total_cv: dict[str, float],
    kinetic_cv: dict[str, float],
    block_rows: list[dict[str, float]],
    temperature_k: float,
    title: str,
    bins: int,
) -> None:
    """Save heat-capacity diagnostic plots."""
    time_ps = np.asarray(arrays["time_ps"], dtype=np.float64)
    total = np.asarray(arrays["total"], dtype=np.float64)
    kinetic = np.asarray(arrays["kinetic"], dtype=np.float64)
    potential = np.asarray(arrays["potential"], dtype=np.float64)
    atom_count = int(arrays["atom_count"])
    energy_source = str(arrays.get("energy_source", "path"))
    total_label = "E_tot" if energy_source == "direct" else "E_path"
    potential_label = "E_pot" if energy_source == "direct" else "U_path"

    raw_converted = convert_cv(raw_total_cv["cv_e_v_per_k"], atom_count)
    detrended_converted = convert_cv(detrended_total_cv["cv_e_v_per_k"], atom_count)
    kinetic_converted = convert_cv(kinetic_cv["cv_e_v_per_k"], atom_count)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time_ps, kinetic - kinetic[0], lw=0.8, label="K - K0")
    ax.plot(time_ps, potential - potential[0], lw=0.8, label=f"{potential_label} - {potential_label}0")
    ax.plot(time_ps, total - total[0], lw=1.1, label=f"{total_label} - {total_label}0")
    ax.set_title(f"{energy_source} energy trace")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("energy change, eV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(total - np.mean(total), bins=bins, density=True, alpha=0.45, label=f"raw {total_label}")
    ax.hist(total_for_estimate - np.mean(total_for_estimate), bins=bins, density=True, alpha=0.45, label="estimate signal")
    ax.set_title("energy fluctuation histogram")
    ax.set_xlabel("E - <E>, eV")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["raw E", "detrended E", "kinetic", "3NkB ref"]
    values = [
        raw_converted["cv_kb_per_atom"],
        detrended_converted["cv_kb_per_atom"],
        kinetic_converted["cv_kb_per_atom"],
        3.0,
    ]
    colors = ["tab:orange", "tab:green", "tab:blue", "black"]
    ax.bar(labels, values, color=colors, alpha=0.72)
    ax.set_ylabel("C, kB/atom")
    ax.set_title(f"fluctuation heat capacity diagnostic at T={temperature_k:.3g} K")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    if block_rows:
        for per_block_detrend, label, marker in [
            (False, "global detrend blocks", "o-"),
            (True, "per-block detrend", "s--"),
        ]:
            selected = [row for row in block_rows if bool(row["per_block_detrend"]) == per_block_detrend]
            if not selected:
                continue
            block_sizes = np.asarray([row["block_size"] for row in selected], dtype=np.float64)
            means = np.asarray([row["mean_kb_per_atom"] for row in selected], dtype=np.float64)
            stds = np.asarray([row["std_kb_per_atom"] for row in selected], dtype=np.float64)
            ax.errorbar(block_sizes, means, yerr=stds, fmt=marker, capsize=3, label=label)
        ax.axhline(detrended_converted["cv_kb_per_atom"], color="tab:green", lw=1.0, ls="--", label="global detrended")
        ax.axhline(3.0, color="black", lw=1.0, ls=":", label="3NkB")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    ax.set_title("blockwise C_V stability")
    ax.set_xlabel("block size, frames")
    ax.set_ylabel("C, kB/atom")
    ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Run heat-capacity postprocessing."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    arrays = load_energy_arrays(
        Path(args.ase_path),
        args.frame_fraction_start,
        args.frame_fraction_end,
        args.mass_amu,
    )
    temperature_series = arrays["temperature"]
    temperature_k = float(args.temperature_k) if args.temperature_k is not None else float(np.mean(temperature_series))
    total = np.asarray(arrays["total"], dtype=np.float64)
    kinetic = np.asarray(arrays["kinetic"], dtype=np.float64)
    detrended_total, total_slope, _ = linear_detrend(np.asarray(arrays["time_ps"], dtype=np.float64), total)
    estimate_signal = detrended_total if args.detrend else total

    raw_total_cv = cv_from_energy(total, temperature_k)
    detrended_total_cv = cv_from_energy(detrended_total, temperature_k)
    kinetic_cv = cv_from_energy(kinetic, temperature_k)
    block_rows = [
        *block_cv_estimates(
            estimate_signal,
            np.asarray(arrays["time_ps"], dtype=np.float64),
            temperature_k,
            int(arrays["atom_count"]),
            args.block_sizes,
            per_block_detrend=False,
        ),
        *block_cv_estimates(
            estimate_signal,
            np.asarray(arrays["time_ps"], dtype=np.float64),
            temperature_k,
            int(arrays["atom_count"]),
            args.block_sizes,
            per_block_detrend=True,
        ),
    ]
    rows = build_metrics(
        arrays,
        temperature_k,
        raw_total_cv,
        detrended_total_cv,
        kinetic_cv,
        block_rows,
        total_slope,
        args.detrend,
    )

    output_path = Path(args.output_path)
    plot_heat_capacity(
        output_path,
        arrays,
        estimate_signal,
        raw_total_cv,
        detrended_total_cv,
        kinetic_cv,
        block_rows,
        temperature_k,
        args.title,
        args.bins,
    )
    print(f"Saved {output_path}")

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        write_metrics(metrics_path, rows)
        print(f"Saved {metrics_path}")

    converted = convert_cv((detrended_total_cv if args.detrend else raw_total_cv)["cv_e_v_per_k"], int(arrays["atom_count"]))
    print(
        "heat capacity (path) = "
        f"{converted['cv_kb_per_atom']:.6g} kB/atom = {converted['cv_j_per_mol_k']:.6g} J/mol/K"
    )
    print("classical Dulong-Petit reference = 3 kB/atom = 24.9434 J/mol/K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
