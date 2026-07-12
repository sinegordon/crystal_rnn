#!/usr/bin/env python3
"""Run stateless pair-energy MLP dynamics in ASE with a Bussi thermostat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.bussi import Bussi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.mlp.ase.calculator import CrystalPairEnergyMLPCalculator  # noqa: E402


def parse_args():
    """Parse ASE inference options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--initial-frames", nargs=2, type=int, required=True, metavar=("PREVIOUS", "CURRENT"))
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--taut-fs", type=float, default=200.0)
    parser.add_argument("--dt-ps", type=float, default=0.02)
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument("--block-batch-size", type=int, default=250)
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-initial-com-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rescale-initial-temperature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trajectory-path", default=None)
    parser.add_argument("--log-path", default="-")
    return parser.parse_args()


def load_data(path):
    """Load arrays required to initialize ASE dynamics."""
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    for key in ("displacements", "reference_positions", "atom_order", "box_lengths"):
        if key not in data:
            raise ValueError(f"Dataset is missing {key!r}")
    return data


def flat_vectors(crystal_values, atom_order):
    """Convert crystal vectors to flat atom order."""
    flat = np.empty((atom_order.size, 3), dtype=np.float64)
    flat[atom_order.reshape(-1)] = np.asarray(crystal_values).reshape(atom_order.size, 3)
    return flat


def cell_matrix(data):
    """Return the dataset cell matrix."""
    lengths = np.asarray(data["box_lengths"], dtype=np.float64)
    if lengths.ndim == 2:
        lengths = lengths[0]
    return np.diag(lengths)


def minimum_image(delta, cell):
    """Apply minimum-image wrapping to flat vectors."""
    fractional = np.asarray(delta) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def initial_state(data, frame_indices, dt_ps, remove_com, rescale_temperature, target_temperature):
    """Return current positions and velocity reconstructed from two frames."""
    reference = np.asarray(data["reference_positions"], dtype=np.float64)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    cell = cell_matrix(data)
    positions = []
    for frame in frame_indices:
        positions.append(reference + flat_vectors(data["displacements"][frame], atom_order))
    positions = np.asarray(positions)
    velocity = minimum_image(positions[1] - positions[0], cell) / (dt_ps * 1000.0 * units.fs)
    masses = np.full(len(reference), 63.546, dtype=np.float64)
    if remove_com:
        velocity -= np.average(velocity, axis=0, weights=masses)
    if rescale_temperature:
        kinetic = 0.5 * np.sum(masses[:, None] * velocity**2)
        current_temperature = 2.0 * kinetic / (3.0 * len(masses) * units.kB)
        if current_temperature <= 0:
            raise ValueError("Cannot rescale a zero initial velocity")
        velocity *= np.sqrt(float(target_temperature) / current_temperature)
    return positions, velocity, cell


def main():
    """Run NVT dynamics and save a postprocessing-compatible NPZ."""
    args = parse_args()
    if args.steps <= 0 or args.record_interval <= 0 or args.taut_fs <= 0 or args.dt_ps <= 0:
        raise ValueError("steps, record_interval, taut_fs, and dt_ps must be positive")
    data = load_data(args.data_path)
    frame_count = len(data["displacements"])
    if any(index < 0 or index >= frame_count for index in args.initial_frames):
        raise ValueError(f"initial-frames must be in [0, {frame_count})")
    initial_positions, velocity, cell = initial_state(
        data,
        args.initial_frames,
        args.dt_ps,
        args.remove_initial_com_velocity,
        args.rescale_initial_temperature,
        args.temperature_k,
    )
    atoms = Atoms(
        symbols=["Cu"] * len(data["reference_positions"]),
        positions=initial_positions[-1],
        cell=cell,
        pbc=True,
    )
    atoms.set_velocities(velocity)
    atoms.calc = CrystalPairEnergyMLPCalculator(
        model_path=args.model_path,
        data_path=args.data_path,
        dt_ps=args.dt_ps,
        device=args.device,
        periodic=args.periodic,
        block_batch_size=args.block_batch_size,
    )
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = None
    if args.trajectory_path:
        trajectory_path = Path(args.trajectory_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory = Trajectory(str(trajectory_path), "w", atoms)
    dynamics = Bussi(
        atoms,
        timestep=args.dt_ps * 1000.0 * units.fs,
        temperature_K=args.temperature_k,
        taut=args.taut_fs * units.fs,
        logfile=sys.stdout if args.log_path == "-" else args.log_path,
    )
    records = {
        "steps": [],
        "positions": [],
        "velocities": [],
        "forces": [],
        "temperature": [],
        "kinetic": [],
        "potential": [],
    }

    def record():
        step = int(dynamics.nsteps)
        if records["steps"] and records["steps"][-1] == step:
            return
        records["steps"].append(step)
        records["positions"].append(atoms.get_positions().copy())
        records["velocities"].append(atoms.get_velocities().copy() * (1000.0 * units.fs))
        records["forces"].append(atoms.get_forces().copy())
        records["temperature"].append(float(atoms.get_temperature()))
        records["kinetic"].append(float(atoms.get_kinetic_energy()))
        records["potential"].append(float(atoms.get_potential_energy()))
        if trajectory is not None:
            trajectory.write(atoms)

    record()
    dynamics.attach(record, interval=args.record_interval)
    dynamics.run(args.steps)
    record()
    if trajectory is not None:
        trajectory.close()
    np.savez_compressed(
        output_path,
        steps=np.asarray(records["steps"], dtype=np.int64),
        positions=np.asarray(records["positions"], dtype=np.float32),
        velocities_ang_per_ps=np.asarray(records["velocities"], dtype=np.float32),
        forces_ev_per_ang=np.asarray(records["forces"], dtype=np.float32),
        temperature_k=np.asarray(records["temperature"], dtype=np.float32),
        kinetic_energy_ev=np.asarray(records["kinetic"], dtype=np.float32),
        potential_energy_ev=np.asarray(records["potential"], dtype=np.float32),
        initial_frames=np.asarray(args.initial_frames, dtype=np.int64),
        initial_history_positions=initial_positions.astype(np.float32),
        cell=cell.astype(np.float32),
        dt_ps=np.asarray(args.dt_ps, dtype=np.float32),
        temperature_target_k=np.asarray(args.temperature_k, dtype=np.float32),
        taut_fs=np.asarray(args.taut_fs, dtype=np.float32),
        remove_initial_com_velocity=np.asarray(args.remove_initial_com_velocity),
        model_path=np.asarray(str(args.model_path)),
        data_path=np.asarray(str(args.data_path)),
        model_interface=np.asarray("crystal-pair-energy-mlp-v1"),
    )
    print(f"Saved {output_path}")
    print(f"temperature mean = {np.mean(records['temperature']):.8g} K")


if __name__ == "__main__":
    main()
