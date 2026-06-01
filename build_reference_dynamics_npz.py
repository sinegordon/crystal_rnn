"""Build an ASE-like dynamics file from a prepared crystal reference dataset.

The prepared ``data*.npz`` files contain reference displacements but not forces.
For diagnostic comparison with model runs, this script reconstructs positions,
estimates velocities and accelerations by finite differences, and writes
``forces_ev_per_ang = mass * acceleration``.  The resulting file is intended
for the same path-energy/heat-capacity/Green-Kubo proxy postprocessing used for
ASE inference outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_ase_path_energy import AMU_ANG2_PER_PS2_TO_EV, CU_MASS_AMU, as_cell_matrix
from plot_sqw_comparison import crystal_frames_to_flat_positions


KB_EV_PER_K = 8.617333262145e-5


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Prepared crystal .npz file.")
    parser.add_argument("--output-path", required=True, help="Output ASE-like .npz file.")
    parser.add_argument("--dt-ps", type=float, default=0.002, help="Frame time step in ps.")
    parser.add_argument("--mass-amu", type=float, default=CU_MASS_AMU, help="Atomic mass.")
    parser.add_argument("--start-frame", type=int, default=0, help="First reference frame to use.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum frame count.")
    parser.add_argument(
        "--temperature-dof-mode",
        choices=["3n", "3n-3"],
        default="3n",
        help="Degrees of freedom used for kinetic temperature.",
    )
    return parser.parse_args()


def load_reference_positions(data_path: Path, start_frame: int, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return flat absolute reference positions and a representative cell."""
    data = np.load(data_path)
    required = ["displacements", "reference_positions", "atom_order"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing arrays in {data_path}: {missing}")
    displacements = np.asarray(data["displacements"], dtype=np.float32)
    if start_frame < 0 or start_frame >= displacements.shape[0]:
        raise ValueError("start-frame is outside the displacement trajectory")
    end_frame = displacements.shape[0] if max_frames is None else min(displacements.shape[0], start_frame + max_frames)
    displacements = displacements[start_frame:end_frame]
    if displacements.shape[0] < 5:
        raise ValueError("Need at least five frames for finite-difference reference dynamics")
    flat = crystal_frames_to_flat_positions(displacements, data["reference_positions"], data["atom_order"]).reshape(
        displacements.shape[0],
        -1,
        3,
    )

    if "cell" in data.files:
        cell = as_cell_matrix(data["cell"])
    elif "box_lengths" in data.files:
        box_lengths = np.asarray(data["box_lengths"], dtype=np.float64)
        if box_lengths.ndim == 2:
            cell = np.diag(np.mean(box_lengths[start_frame:end_frame], axis=0))
        else:
            cell = as_cell_matrix(box_lengths)
    else:
        raise ValueError("No cell or box_lengths found in reference dataset")
    return flat.astype(np.float64), cell.astype(np.float64)


def finite_difference_velocity(positions: np.ndarray, dt_ps: float) -> np.ndarray:
    """Return finite-difference velocities in A/ps."""
    return np.gradient(np.asarray(positions, dtype=np.float64), dt_ps, axis=0, edge_order=2)


def finite_difference_acceleration(positions: np.ndarray, dt_ps: float) -> np.ndarray:
    """Return finite-difference accelerations in A/ps^2."""
    velocities = finite_difference_velocity(positions, dt_ps)
    return np.gradient(velocities, dt_ps, axis=0, edge_order=2)


def kinetic_energy_ev(velocities: np.ndarray, mass_amu: float) -> np.ndarray:
    """Return total kinetic energy per frame in eV."""
    return 0.5 * mass_amu * AMU_ANG2_PER_PS2_TO_EV * np.sum(velocities**2, axis=(1, 2))


def kinetic_temperature_k(kinetic_ev: np.ndarray, atom_count: int, dof_mode: str) -> np.ndarray:
    """Return kinetic temperature from total kinetic energy."""
    dof = 3 * int(atom_count)
    if dof_mode == "3n-3":
        dof = max(1, dof - 3)
    return 2.0 * np.asarray(kinetic_ev, dtype=np.float64) / (dof * KB_EV_PER_K)


def main() -> int:
    """Build and save an ASE-like reference dynamics file."""
    args = parse_args()
    if args.dt_ps <= 0:
        raise ValueError("dt-ps must be positive")
    if args.mass_amu <= 0:
        raise ValueError("mass-amu must be positive")
    positions, cell = load_reference_positions(Path(args.data_path), args.start_frame, args.max_frames)
    velocities = finite_difference_velocity(positions, args.dt_ps)
    accelerations = finite_difference_acceleration(positions, args.dt_ps)
    forces = args.mass_amu * AMU_ANG2_PER_PS2_TO_EV * accelerations
    kinetic = kinetic_energy_ev(velocities, args.mass_amu)
    temperature = kinetic_temperature_k(kinetic, positions.shape[1], args.temperature_dof_mode)
    steps = np.arange(positions.shape[0], dtype=np.int64) + int(args.start_frame)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=steps,
        positions=positions.astype(np.float32),
        velocities_ang_per_ps=velocities.astype(np.float32),
        forces_ev_per_ang=forces.astype(np.float32),
        temperature_k=temperature.astype(np.float32),
        kinetic_energy_ev=kinetic.astype(np.float32),
        cell=cell.astype(np.float32),
        dt_ps=np.asarray(args.dt_ps, dtype=np.float32),
        temperature_target_k=np.asarray(float(np.mean(temperature)), dtype=np.float32),
        initial_frames=np.asarray([args.start_frame, args.start_frame + 1, args.start_frame + 2], dtype=np.int64),
        reference_source=np.asarray(str(args.data_path)),
        force_source=np.asarray("finite_difference_mass_acceleration"),
        mass_amu=np.asarray(args.mass_amu, dtype=np.float32),
    )
    print(f"Saved {output_path}")
    print(
        "temperature first/last/mean/min/max = "
        f"{temperature[0]:.8g} {temperature[-1]:.8g} {temperature.mean():.8g} "
        f"{temperature.min():.8g} {temperature.max():.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
