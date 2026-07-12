#!/usr/bin/env python3
"""Build the standard article diagnostics for one MLP ASE trajectory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    """Parse postprocessing paths and common plot settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ncells", type=int, default=10)
    parser.add_argument("--kcount", type=int, default=10)
    parser.add_argument("--sqw-step", type=int, default=10)
    parser.add_argument("--dt-ps", type=float, default=None, help="Frame timestep; defaults to dt_ps saved in ASE NPZ.")
    parser.add_argument("--velocity-window-frames", type=int, default=10)
    parser.add_argument("--phase-window-frames", type=int, default=1000)
    parser.add_argument("--strict", action="store_true", help="Stop when any optional diagnostic fails.")
    return parser.parse_args()


def main():
    """Run all compatible root-level postprocessing scripts."""
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ase_path = str(Path(args.ase_path))
    data_path = str(Path(args.data_path))
    python = sys.executable
    if args.dt_ps is None:
        with np.load(ase_path) as trajectory:
            if "dt_ps" not in trajectory.files:
                raise ValueError("ASE trajectory has no dt_ps; pass --dt-ps explicitly")
            dt_ps = float(np.asarray(trajectory["dt_ps"]))
    else:
        dt_ps = float(args.dt_ps)
    if dt_ps <= 0:
        raise ValueError("dt-ps must be positive")

    def command(script, *options):
        return [python, str(ROOT / script), *map(str, options)]

    commands = [
        command(
            "plot_sqw_comparison.py",
            "--input-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "sqw_comparison.png",
            "--ncells", args.ncells,
            "--kcount", args.kcount,
            "--dt", dt_ps,
            "--step", args.sqw_step,
        ),
        command(
            "plot_ase_velocity_histograms.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "velocity_histograms.png",
            "--metrics-path", output / "velocity_metrics.tsv",
            "--window-frames", args.velocity_window_frames,
        ),
        command(
            "plot_displacement_component_layers.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--hist-output-path", output / "displacement_components.png",
            "--layer-output-path", output / "displacement_layers.png",
            "--metrics-path", output / "displacement_metrics.tsv",
        ),
        command(
            "plot_displacement_component_layers.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--hist-output-path", output / "displacement_components_com_subtracted.png",
            "--layer-output-path", output / "displacement_layers_com_subtracted.png",
            "--metrics-path", output / "displacement_com_subtracted_metrics.tsv",
            "--subtract-frame-mean",
        ),
        command(
            "plot_ase_vacf.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "vacf.png",
            "--metrics-path", output / "vacf_metrics.tsv",
            "--remove-frame-com",
        ),
        command(
            "plot_ase_rdf.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "rdf.png",
            "--metrics-path", output / "rdf_metrics.tsv",
        ),
        command(
            "plot_ase_temperature_trace.py",
            "--ase-path", ase_path,
            "--output-path", output / "temperature.png",
            "--metrics-path", output / "temperature_metrics.tsv",
        ),
        command(
            "plot_ase_com_motion.py",
            "--ase-npz", ase_path,
            "--output-prefix", output / "com_motion",
        ),
        command(
            "plot_ase_phase_histograms.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "phase_histograms.png",
            "--metrics-path", output / "phase_metrics.tsv",
            "--window-frames", args.phase_window_frames,
        ),
        command(
            "plot_ase_canonical_checks.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "canonical_checks.png",
            "--metrics-path", output / "canonical_metrics.tsv",
        ),
        command(
            "plot_ase_sound_speed_ox.py",
            "--ase-path", ase_path,
            "--data-path", data_path,
            "--output-path", output / "sound_speed_ox.png",
            "--metrics-path", output / "sound_speed_ox_metrics.tsv",
        ),
        command(
            "plot_ase_path_energy.py",
            "--ase-path", ase_path,
            "--output-path", output / "path_energy.png",
            "--metrics-path", output / "path_energy_metrics.tsv",
        ),
        command(
            "plot_ase_heat_capacity.py",
            "--ase-path", ase_path,
            "--output-path", output / "path_energy_heat_capacity.png",
            "--metrics-path", output / "path_energy_heat_capacity_metrics.tsv",
        ),
        command(
            "plot_ase_total_energy.py",
            "--ase-path", ase_path,
            "--output-path", output / "total_energy.png",
            "--metrics-path", output / "total_energy_metrics.tsv",
        ),
        command(
            "plot_ase_etot_heat_capacity.py",
            "--ase-path", ase_path,
            "--output-path", output / "total_energy_heat_capacity.png",
            "--metrics-path", output / "total_energy_heat_capacity_metrics.tsv",
        ),
        command(
            "plot_ase_etot_cumulative.py",
            "--ase-path", ase_path,
            "--output-path", output / "total_energy_cumulative.png",
            "--series-path", output / "total_energy_cumulative.tsv",
            "--summary-path", output / "total_energy_cumulative_summary.tsv",
        ),
    ]
    failures = []
    for item in commands:
        print("$ " + " ".join(map(str, item)), flush=True)
        result = subprocess.run(item, check=False)
        if result.returncode:
            failures.append((Path(item[1]).name, result.returncode))
            if args.strict:
                return result.returncode
    if failures:
        print("Optional diagnostics that failed:")
        for script, returncode in failures:
            print(f"  {script}: exit {returncode}")
    print(f"Postprocessing output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
