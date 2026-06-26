"""Estimate Green-Kubo-like thermal conductivity from path energies.

This is an exploratory lattice-thermal-conductivity diagnostic for RNN/ASE
runs where the model provides forces but no explicit potential energy.  It
reconstructs a per-atom path energy from

    U_i(t + dt) = U_i(t) - 0.5 * (F_i(t) + F_i(t + dt)) . dr_i

and uses that path energy to build proxy heat-flux time series.  The estimate
is useful for comparing runs, but it is not a rigorous Green-Kubo conductivity
unless the force field is conservative and the heat current is completed with
the proper interaction/virial contribution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ase_path_energy import CU_MASS_AMU, as_cell_matrix, minimum_image, per_atom_kinetic_energy


EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1.0e-10
PS_TO_S = 1.0e-12
KB_J_PER_K = 1.380649e-23
HEAT_FLUX_EV_A2_PS_TO_W_M2 = EV_TO_J / (ANGSTROM_TO_M**2 * PS_TO_S)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--mass-amu", type=float, default=CU_MASS_AMU, help="Atomic mass used for kinetic energy.")
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=None,
        help="Temperature used in the Green-Kubo prefactor. Defaults to mean ASE temperature.",
    )
    parser.add_argument(
        "--frame-fraction-start",
        type=float,
        default=0.0,
        help="Start fraction of the ASE trajectory used for conductivity.",
    )
    parser.add_argument(
        "--frame-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of the ASE trajectory used for conductivity.",
    )
    parser.add_argument(
        "--max-correlation-time-ps",
        type=float,
        default=20.0,
        help="Maximum heat-current autocorrelation lag in ps.",
    )
    parser.add_argument(
        "--plateau-fraction",
        type=float,
        default=0.25,
        help="Last fraction of the integration window averaged as a rough plateau estimate.",
    )
    parser.add_argument(
        "--plot-flux-time-ps",
        type=float,
        default=10.0,
        help="Initial heat-flux time span shown in the plot.",
    )
    parser.add_argument(
        "--flux-mode",
        choices=["energy-moment", "convective", "both"],
        default="both",
        help="Heat-flux proxy to plot and report.",
    )
    parser.add_argument("--title", default="Path-energy Green-Kubo conductivity proxy")
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


def load_ase_arrays(path: Path, start_fraction: float, end_fraction: float) -> dict[str, np.ndarray | float]:
    """Load and trim ASE arrays needed for path-energy heat-flux estimates."""
    ase = np.load(path)
    required = ["positions", "velocities_ang_per_ps", "forces_ev_per_ang", "cell", "dt_ps"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"Missing arrays in ASE output: {missing}")

    positions = np.asarray(ase["positions"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
    frame_count = min(int(positions.shape[0]), int(velocities.shape[0]), int(forces.shape[0]))
    frame_slice = fraction_slice(frame_count, start_fraction, end_fraction)
    positions = positions[:frame_count][frame_slice]
    velocities = velocities[:frame_count][frame_slice]
    forces = forces[:frame_count][frame_slice]
    if positions.shape[0] < 4:
        raise ValueError("Need at least four selected frames")

    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64))
    steps = np.asarray(ase["steps"], dtype=np.float64)[:frame_count] if "steps" in ase.files else np.arange(frame_count)
    steps = steps[frame_slice]
    temperature = (
        np.asarray(ase["temperature_k"], dtype=np.float64)[:frame_count][frame_slice]
        if "temperature_k" in ase.files
        else None
    )
    return {
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "cell": as_cell_matrix(ase["cell"]),
        "dt_ps": dt_ps,
        "steps": steps,
        "time_ps": steps * dt_ps,
        "temperature": temperature,
    }


def unwrapped_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Reconstruct unwrapped positions from minimum-image frame displacements."""
    dr = minimum_image(positions[1:] - positions[:-1], cell)
    unwrapped = np.empty_like(positions, dtype=np.float64)
    unwrapped[0] = positions[0]
    unwrapped[1:] = positions[0] + np.cumsum(dr, axis=0)
    return unwrapped


def per_atom_path_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    cell: np.ndarray,
    mass_amu: float,
) -> dict[str, np.ndarray]:
    """Return per-atom kinetic, path-potential, and path-total energies."""
    kinetic = per_atom_kinetic_energy(velocities, mass_amu)
    dr = minimum_image(positions[1:] - positions[:-1], cell)
    work_atom = np.sum(0.5 * (forces[:-1] + forces[1:]) * dr, axis=2)
    potential = np.empty_like(kinetic, dtype=np.float64)
    potential[0] = 0.0
    potential[1:] = -np.cumsum(work_atom, axis=0)
    return {
        "kinetic": kinetic,
        "potential": potential,
        "total": kinetic + potential,
        "work_atom": work_atom,
    }


def centered_atom_energy(total_energy: np.ndarray) -> np.ndarray:
    """Remove the arbitrary per-frame energy constant from per-atom energies."""
    energy = np.asarray(total_energy, dtype=np.float64)
    return energy - np.mean(energy, axis=1, keepdims=True)


def cell_volume_ang3(cell: np.ndarray) -> float:
    """Return the positive cell volume in A^3."""
    return float(abs(np.linalg.det(np.asarray(cell, dtype=np.float64))))


def energy_moment_flux(
    positions_unwrapped: np.ndarray,
    centered_energy: np.ndarray,
    volume_ang3: float,
    dt_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return heat-flux density from d/dt sum_i r_i e_i divided by volume."""
    moment = np.einsum("tni,tn->ti", positions_unwrapped, centered_energy)
    flux = (moment[2:] - moment[:-2]) / (2.0 * dt_ps * volume_ang3)
    frame_indices = np.arange(1, moment.shape[0] - 1, dtype=np.int64)
    return flux, frame_indices


def convective_flux(velocities: np.ndarray, centered_energy: np.ndarray, volume_ang3: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the simple convective heat-flux density sum_i e_i v_i / volume."""
    flux = np.einsum("tni,tn->ti", velocities, centered_energy) / volume_ang3
    frame_indices = np.arange(flux.shape[0], dtype=np.int64)
    return flux, frame_indices


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    """Return unbiased autocorrelation for each vector component using FFT."""
    series = np.asarray(values, dtype=np.float64)
    series = series - np.mean(series, axis=0, keepdims=True)
    frame_count = int(series.shape[0])
    nfft = 1 << (2 * frame_count - 1).bit_length()
    spectrum = np.fft.rfft(series, n=nfft, axis=0)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft, axis=0)[:frame_count]
    normalization = np.arange(frame_count, 0, -1, dtype=np.float64)[:, None]
    return acf / normalization


def cumulative_trapezoid(values: np.ndarray, dx: float) -> np.ndarray:
    """Return cumulative trapezoidal integral with a leading zero."""
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    if values.shape[0] > 1:
        result[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * dx, axis=0)
    return result


def conductivity_from_flux(
    flux_ev_a2_ps: np.ndarray,
    dt_ps: float,
    volume_ang3: float,
    temperature_k: float,
    max_correlation_time_ps: float,
) -> dict[str, np.ndarray]:
    """Return HCACF and cumulative Green-Kubo conductivity curves."""
    if temperature_k <= 0:
        raise ValueError("temperature must be positive")
    if max_correlation_time_ps <= 0:
        raise ValueError("max-correlation-time-ps must be positive")
    max_lag = min(int(np.floor(max_correlation_time_ps / dt_ps)) + 1, int(flux_ev_a2_ps.shape[0]))
    if max_lag < 2:
        raise ValueError("Need at least two heat-flux lags")

    acf = autocorrelation_fft(flux_ev_a2_ps)[:max_lag]
    lag_time_ps = np.arange(max_lag, dtype=np.float64) * dt_ps
    acf_si = acf * HEAT_FLUX_EV_A2_PS_TO_W_M2**2
    integral_si = cumulative_trapezoid(acf_si, dt_ps * PS_TO_S)
    volume_m3 = volume_ang3 * ANGSTROM_TO_M**3
    prefactor = volume_m3 / (KB_J_PER_K * temperature_k**2)
    kappa = prefactor * integral_si
    return {
        "lag_time_ps": lag_time_ps,
        "acf_ev": acf,
        "acf_si": acf_si,
        "kappa_w_mk": kappa,
    }


def plateau_values(curve: np.ndarray, plateau_fraction: float) -> np.ndarray:
    """Return mean values over the last fraction of a conductivity curve."""
    if not 0 < plateau_fraction <= 1:
        raise ValueError("plateau-fraction must be in (0, 1]")
    curve = np.asarray(curve, dtype=np.float64)
    start = int(np.floor(curve.shape[0] * (1.0 - plateau_fraction)))
    start = min(max(start, 0), curve.shape[0] - 1)
    return np.mean(curve[start:], axis=0)


def component_mean(values: np.ndarray) -> np.ndarray:
    """Append an isotropic mean column to component values."""
    return np.column_stack([values, np.mean(values, axis=1)])


def mode_label(mode: str) -> str:
    """Return a readable plot label."""
    return {"energy-moment": "energy moment", "convective": "convective"}[mode]


def selected_modes(flux_mode: str) -> list[str]:
    """Return flux modes requested by the CLI."""
    if flux_mode == "both":
        return ["energy-moment", "convective"]
    return [flux_mode]


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write metrics rows as TSV."""
    fields = ["quantity", "mode", "component", "value", "unit", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_metric_rows(
    analyses: dict[str, dict[str, np.ndarray]],
    temperature_k: float,
    volume_ang3: float,
    dt_ps: float,
    frame_count: int,
    plateau_fraction: float,
) -> list[dict[str, object]]:
    """Build compact TSV metrics."""
    rows: list[dict[str, object]] = [
        {"quantity": "temperature", "mode": "metadata", "component": "scalar", "value": temperature_k, "unit": "K"},
        {"quantity": "volume", "mode": "metadata", "component": "scalar", "value": volume_ang3, "unit": "A^3"},
        {"quantity": "dt", "mode": "metadata", "component": "scalar", "value": dt_ps, "unit": "ps"},
        {"quantity": "frames", "mode": "metadata", "component": "scalar", "value": frame_count, "unit": "count"},
    ]
    components = ["x", "y", "z", "mean"]
    for mode, analysis in analyses.items():
        kappa = component_mean(analysis["kappa_w_mk"])
        final = kappa[-1]
        plateau = np.append(
            plateau_values(analysis["kappa_w_mk"], plateau_fraction),
            float(np.mean(plateau_values(analysis["kappa_w_mk"], plateau_fraction))),
        )
        for component, value in zip(components, final):
            rows.append(
                {
                    "quantity": "kappa_final",
                    "mode": mode,
                    "component": component,
                    "value": float(value),
                    "unit": "W/m/K",
                    "note": "cumulative integral at max lag",
                }
            )
        for component, value in zip(components, plateau):
            rows.append(
                {
                    "quantity": "kappa_plateau",
                    "mode": mode,
                    "component": component,
                    "value": float(value),
                    "unit": "W/m/K",
                    "note": f"mean over last {plateau_fraction:g} of lag window",
                }
            )
    return rows


def plot_results(
    output_path: Path,
    fluxes: dict[str, tuple[np.ndarray, np.ndarray]],
    analyses: dict[str, dict[str, np.ndarray]],
    time_ps: np.ndarray,
    temperature_k: float,
    title: str,
    plot_flux_time_ps: float,
) -> None:
    """Save heat-flux, HCACF, and cumulative conductivity plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue", "mean": "black"}
    components = ["x", "y", "z"]

    main_mode = next(iter(analyses))
    main_flux, main_indices = fluxes[main_mode]
    flux_time = time_ps[main_indices] - time_ps[main_indices][0]
    flux_mask = flux_time <= plot_flux_time_ps
    ax = axes[0, 0]
    for index, component in enumerate(components):
        ax.plot(flux_time[flux_mask], main_flux[flux_mask, index], lw=0.8, color=colors[component], label=component)
    ax.set_title(f"heat-flux proxy, {mode_label(main_mode)}")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("j, eV / A^2 / ps")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for mode, analysis in analyses.items():
        acf = analysis["acf_ev"]
        normalized = acf / np.maximum(np.abs(acf[0:1]), 1.0e-300)
        mean_norm = np.mean(normalized, axis=1)
        ax.plot(analysis["lag_time_ps"], mean_norm, lw=1.0, label=mode_label(mode))
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    ax.set_title("normalized HCACF mean")
    ax.set_xlabel("lag, ps")
    ax.set_ylabel("C(t) / |C(0)|")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for mode, analysis in analyses.items():
        kappa = component_mean(analysis["kappa_w_mk"])
        for index, component in enumerate(["x", "y", "z", "mean"]):
            alpha = 0.45 if component != "mean" else 1.0
            lw = 0.85 if component != "mean" else 1.8
            ax.plot(
                analysis["lag_time_ps"],
                kappa[:, index],
                lw=lw,
                alpha=alpha,
                color=colors[component],
                ls="-" if mode == main_mode else "--",
                label=f"{mode_label(mode)} {component}",
            )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    ax.set_title("cumulative Green-Kubo conductivity proxy")
    ax.set_xlabel("correlation time, ps")
    ax.set_ylabel("kappa, W / m / K")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "Caveat:",
        "path energy is trajectory-defined,",
        "not a strict state potential for",
        "non-conservative model forces.",
        "",
        f"T used = {temperature_k:.3g} K",
    ]
    for mode, analysis in analyses.items():
        kappa = component_mean(analysis["kappa_w_mk"])
        lines.extend(
            [
                "",
                f"{mode_label(mode)}:",
                f"  final x/y/z = {kappa[-1,0]:.3g}, {kappa[-1,1]:.3g}, {kappa[-1,2]:.3g}",
                f"  final mean = {kappa[-1,3]:.3g} W/m/K",
            ]
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    ax.set_title("summary")

    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Run Green-Kubo-like path-energy conductivity postprocessing."""
    args = parse_args()
    if args.max_correlation_time_ps <= 0:
        raise ValueError("max-correlation-time-ps must be positive")
    arrays = load_ase_arrays(Path(args.ase_path), args.frame_fraction_start, args.frame_fraction_end)
    positions = arrays["positions"]
    velocities = arrays["velocities"]
    forces = arrays["forces"]
    cell = arrays["cell"]
    dt_ps = float(arrays["dt_ps"])
    time_ps = arrays["time_ps"]
    temperature_series = arrays["temperature"]
    temperature_k = float(args.temperature_k) if args.temperature_k is not None else float(np.mean(temperature_series))
    volume_ang3 = cell_volume_ang3(cell)

    energies = per_atom_path_energy(positions, velocities, forces, cell, args.mass_amu)
    energy = centered_atom_energy(energies["total"])
    positions_unwrapped = unwrapped_positions(positions, cell)

    fluxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.flux_mode in {"energy-moment", "both"}:
        fluxes["energy-moment"] = energy_moment_flux(positions_unwrapped, energy, volume_ang3, dt_ps)
    if args.flux_mode in {"convective", "both"}:
        fluxes["convective"] = convective_flux(velocities, energy, volume_ang3)

    analyses = {
        mode: conductivity_from_flux(flux, dt_ps, volume_ang3, temperature_k, args.max_correlation_time_ps)
        for mode, (flux, _indices) in fluxes.items()
    }

    output_path = Path(args.output_path)
    plot_results(output_path, fluxes, analyses, time_ps, temperature_k, args.title, args.plot_flux_time_ps)
    print(f"Saved {output_path}")

    rows = build_metric_rows(
        analyses,
        temperature_k=temperature_k,
        volume_ang3=volume_ang3,
        dt_ps=dt_ps,
        frame_count=int(positions.shape[0]),
        plateau_fraction=args.plateau_fraction,
    )
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        write_metrics(metrics_path, rows)
        print(f"Saved {metrics_path}")

    for mode in selected_modes(args.flux_mode):
        kappa = component_mean(analyses[mode]["kappa_w_mk"])
        print(
            f"{mode} kappa final x/y/z/mean = "
            f"{kappa[-1,0]:.6g} {kappa[-1,1]:.6g} {kappa[-1,2]:.6g} {kappa[-1,3]:.6g} W/m/K"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
