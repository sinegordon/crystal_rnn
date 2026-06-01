"""Estimate heat-capacity diagnostics from direct ASE E_tot fluctuations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_ase_heat_capacity import (
    block_cv_estimates,
    build_metrics,
    convert_cv,
    cv_from_energy,
    fraction_slice,
    linear_detrend,
    plot_heat_capacity,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
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
    parser.add_argument("--title", default="ASE direct E_tot heat-capacity diagnostic")
    return parser.parse_args()


def load_etot_arrays(path: Path, frame_fraction_start: float, frame_fraction_end: float):
    """Load direct ASE energies in the requested frame window."""
    ase = np.load(path)
    required = ["kinetic_energy_ev", "potential_energy_ev", "temperature_k", "positions", "dt_ps"]
    missing = [key for key in required if key not in ase.files]
    if missing:
        raise ValueError(f"Missing arrays in ASE output: {missing}")

    kinetic = np.asarray(ase["kinetic_energy_ev"], dtype=np.float64)
    potential = np.asarray(ase["potential_energy_ev"], dtype=np.float64)
    temperature = np.asarray(ase["temperature_k"], dtype=np.float64)
    positions = np.asarray(ase["positions"])
    frame_count = min(kinetic.size, potential.size, temperature.size, int(positions.shape[0]))
    frame_slice = fraction_slice(frame_count, frame_fraction_start, frame_fraction_end)

    kinetic = kinetic[:frame_count][frame_slice]
    potential = potential[:frame_count][frame_slice]
    temperature = temperature[:frame_count][frame_slice]
    steps = np.asarray(ase["steps"], dtype=np.float64)[:frame_count] if "steps" in ase.files else np.arange(frame_count)
    steps = steps[frame_slice]
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64))

    return {
        "time_ps": steps * dt_ps,
        "temperature": temperature,
        "kinetic": kinetic,
        "potential": potential,
        "total": kinetic + potential,
        "atom_count": int(positions.shape[1]),
        "dt_ps": dt_ps,
        "energy_source": "direct",
    }


def main() -> int:
    """Run direct E_tot heat-capacity postprocessing."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    arrays = load_etot_arrays(Path(args.ase_path), args.frame_fraction_start, args.frame_fraction_end)
    temperature_series = arrays["temperature"]
    temperature_k = float(args.temperature_k) if args.temperature_k is not None else float(np.mean(temperature_series))
    total = np.asarray(arrays["total"], dtype=np.float64)
    kinetic = np.asarray(arrays["kinetic"], dtype=np.float64)
    time_ps = np.asarray(arrays["time_ps"], dtype=np.float64)
    detrended_total, total_slope, _ = linear_detrend(time_ps, total)
    estimate_signal = detrended_total if args.detrend else total

    raw_total_cv = cv_from_energy(total, temperature_k)
    detrended_total_cv = cv_from_energy(detrended_total, temperature_k)
    kinetic_cv = cv_from_energy(kinetic, temperature_k)
    block_rows = [
        *block_cv_estimates(
            estimate_signal,
            time_ps,
            temperature_k,
            int(arrays["atom_count"]),
            args.block_sizes,
            per_block_detrend=False,
        ),
        *block_cv_estimates(
            estimate_signal,
            time_ps,
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
        "heat capacity (direct E_tot) = "
        f"{converted['cv_kb_per_atom']:.6g} kB/atom = {converted['cv_j_per_mol_k']:.6g} J/mol/K"
    )
    print("classical Dulong-Petit reference = 3 kB/atom = 24.9434 J/mol/K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
