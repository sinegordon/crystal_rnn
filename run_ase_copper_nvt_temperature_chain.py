"""Run chained Berendsen NVT ASE dynamics with changing target temperatures."""

import argparse
import sys
from pathlib import Path

import numpy as np

from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.nvtberendsen import NVTBerendsen

from ase_copper_calculator import CopperFieldRNNCalculator
from run_ase_copper_nvt import (
    cell_from_dataset,
    frame_positions,
    initialize_history_and_velocities,
    load_crystal_dataset,
)


def parse_args():
    """Parse command-line options for temperature-chain ASE NVT inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Saved RNN acceleration model.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal .npz dataset.")
    parser.add_argument("--output-npz", required=True, help="Output trajectory .npz path.")
    parser.add_argument("--trajectory-path", default=None, help="Optional ASE .traj output path.")
    parser.add_argument("--log-path", default="-", help="ASE MD log path, or '-' for stdout.")
    parser.add_argument(
        "--steps-per-temperature",
        type=int,
        required=True,
        help="ASE MD steps to run at each target temperature.",
    )
    parser.add_argument("--start-temperature-k", type=float, required=True, help="First target temperature in K.")
    parser.add_argument("--end-temperature-k", type=float, required=True, help="Last target temperature in K.")
    parser.add_argument(
        "--temperature-segments",
        "--temperature-count",
        dest="temperature_segments",
        type=int,
        required=True,
        help="Number of temperature plateaus between start and end, inclusive.",
    )
    parser.add_argument("--taut-fs", type=float, default=10.0, help="Fixed Berendsen thermostat time constant in fs.")
    parser.add_argument(
        "--initial-frames",
        type=int,
        nargs=3,
        default=(0, 1, 2),
        metavar=("FRAME0", "FRAME1", "FRAME2"),
        help="Initial history frames from data-path.",
    )
    parser.add_argument("--dt-ps", type=float, default=0.002, help="Model/MD time step in ps.")
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--patch-batch-size", type=int, default=250)
    parser.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic centered patches in the FieldRNN calculator.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument(
        "--rescale-initial-temperature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale frame-derived initial velocities to start-temperature-k.",
    )
    parser.add_argument(
        "--fixcm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove center-of-mass drift inside ASE NVTBerendsen.",
    )
    return parser.parse_args()


def build_temperature_chain(start_temperature_k, end_temperature_k, temperature_segments):
    """Return target temperatures for all chain segments."""
    if temperature_segments <= 0:
        raise ValueError("temperature-segments must be positive")
    return np.linspace(
        float(start_temperature_k),
        float(end_temperature_k),
        int(temperature_segments),
        dtype=np.float64,
    )


def validate_args(args, frame_count):
    """Validate numeric options and initial frame indices."""
    if args.steps_per_temperature <= 0:
        raise ValueError("steps-per-temperature must be positive")
    if args.start_temperature_k <= 0:
        raise ValueError("start-temperature-k must be positive")
    if args.end_temperature_k <= 0:
        raise ValueError("end-temperature-k must be positive")
    if args.taut_fs <= 0:
        raise ValueError("taut-fs must be positive")
    if args.dt_ps <= 0:
        raise ValueError("dt-ps must be positive")
    if args.patch_batch_size <= 0:
        raise ValueError("patch-batch-size must be positive")
    if args.record_interval <= 0:
        raise ValueError("record-interval must be positive")
    if any(frame < 0 or frame >= frame_count for frame in args.initial_frames):
        raise ValueError(f"initial-frames must be in [0, {frame_count})")


def write_summary(args, atoms, temperature_chain, initial_velocity_scale):
    """Print a compact run summary."""
    print("ASE temperature-chain NVT Cu RNN run")
    print(f"model_path = {args.model_path}")
    print(f"data_path = {args.data_path}")
    print(f"steps_per_temperature = {args.steps_per_temperature}")
    print(f"temperature_chain_k = {tuple(float(value) for value in temperature_chain)}")
    print(f"temperature_segments = {len(temperature_chain)}")
    print(f"taut_fs = {args.taut_fs}")
    print(f"initial_frames = {tuple(args.initial_frames)}")
    print(f"dt_ps = {args.dt_ps}")
    print(f"initial_temperature_k = {atoms.get_temperature():.8g}")
    print(f"initial_velocity_scale = {initial_velocity_scale:.8g}")


def main():
    """Run chained ASE NVT temperature segments and save one trajectory."""
    args = parse_args()
    data = load_crystal_dataset(args.data_path)
    validate_args(args, int(data["displacements"].shape[0]))
    temperature_chain = build_temperature_chain(
        args.start_temperature_k,
        args.end_temperature_k,
        args.temperature_segments,
    )

    cell = cell_from_dataset(data)
    positions_history, velocities, initial_velocity_scale = initialize_history_and_velocities(
        positions_history=frame_positions(data, args.initial_frames),
        cell=cell,
        dt_ps=args.dt_ps,
        target_temperature_k=float(temperature_chain[0]),
        rescale_temperature=args.rescale_initial_temperature,
    )

    atoms = Atoms(
        symbols=["Cu"] * positions_history.shape[1],
        positions=positions_history[-1],
        cell=cell,
        pbc=True,
    )
    atoms.set_velocities(velocities)
    atoms.calc = CopperFieldRNNCalculator(
        model_path=args.model_path,
        data_path=args.data_path,
        dt_ps=args.dt_ps,
        history_positions=positions_history,
        patch_shape=tuple(args.patch_shape),
        periodic=args.periodic,
        patch_batch_size=args.patch_batch_size,
        device=args.device,
    )

    write_summary(args, atoms, temperature_chain, initial_velocity_scale)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_path:
        trajectory_path = Path(args.trajectory_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory = Trajectory(str(trajectory_path), "w", atoms)
    else:
        trajectory = None

    recorded_steps = []
    recorded_segment = []
    recorded_target_temperature = []
    recorded_positions = []
    recorded_velocities = []
    recorded_forces = []
    recorded_temperature = []
    recorded_kinetic_energy = []

    def record(global_step, segment_index, target_temperature_k):
        if recorded_steps and recorded_steps[-1] == global_step:
            return
        recorded_steps.append(global_step)
        recorded_segment.append(segment_index)
        recorded_target_temperature.append(target_temperature_k)
        recorded_positions.append(atoms.get_positions().copy())
        recorded_velocities.append(atoms.get_velocities().copy() * (1000.0 * units.fs))
        recorded_forces.append(atoms.get_forces().copy())
        recorded_temperature.append(float(atoms.get_temperature()))
        recorded_kinetic_energy.append(float(atoms.get_kinetic_energy()))
        if trajectory is not None:
            trajectory.write(atoms)

    global_step_start = 0
    record(0, 0, float(temperature_chain[0]))
    timestep = args.dt_ps * 1000.0 * units.fs
    log_handle = sys.stdout if args.log_path == "-" else open(args.log_path, "w", encoding="utf-8")
    try:
        for segment_index, target_temperature_k in enumerate(temperature_chain):
            target_temperature_k = float(target_temperature_k)
            print(
                f"BEGIN_SEGMENT index={segment_index} target_temperature_k={target_temperature_k:g} "
                f"taut_fs={args.taut_fs:g} global_step_start={global_step_start}",
                flush=True,
            )
            dynamics = NVTBerendsen(
                atoms,
                timestep=timestep,
                temperature_K=target_temperature_k,
                taut=float(args.taut_fs) * units.fs,
                fixcm=args.fixcm,
                logfile=log_handle,
            )

            def attached_record(
                segment_index=segment_index,
                target_temperature_k=target_temperature_k,
                dynamics=dynamics,
            ):
                record(global_step_start + dynamics.nsteps, segment_index, target_temperature_k)

            dynamics.attach(attached_record, interval=args.record_interval)
            dynamics.run(args.steps_per_temperature)
            global_step_start += args.steps_per_temperature
            print(
                f"END_SEGMENT index={segment_index} target_temperature_k={target_temperature_k:g} "
                f"global_step={global_step_start} temperature_k={atoms.get_temperature():.8g}",
                flush=True,
            )
    finally:
        if log_handle is not sys.stdout:
            log_handle.close()
        if trajectory is not None:
            trajectory.close()

    np.savez_compressed(
        output_npz,
        steps=np.asarray(recorded_steps, dtype=np.int64),
        segment_index=np.asarray(recorded_segment, dtype=np.int64),
        recorded_target_temperature_k=np.asarray(recorded_target_temperature, dtype=np.float32),
        positions=np.asarray(recorded_positions, dtype=np.float32),
        velocities_ang_per_ps=np.asarray(recorded_velocities, dtype=np.float32),
        forces_ev_per_ang=np.asarray(recorded_forces, dtype=np.float32),
        temperature_k=np.asarray(recorded_temperature, dtype=np.float32),
        kinetic_energy_ev=np.asarray(recorded_kinetic_energy, dtype=np.float32),
        initial_frames=np.asarray(args.initial_frames, dtype=np.int64),
        initial_history_positions=np.asarray(positions_history, dtype=np.float32),
        initial_velocity_scale=np.asarray(initial_velocity_scale, dtype=np.float32),
        cell=np.asarray(cell, dtype=np.float32),
        dt_ps=np.asarray(args.dt_ps, dtype=np.float32),
        taut_fs=np.asarray(args.taut_fs, dtype=np.float32),
        temperature_chain_k=np.asarray(temperature_chain, dtype=np.float32),
        start_temperature_k=np.asarray(args.start_temperature_k, dtype=np.float32),
        end_temperature_k=np.asarray(args.end_temperature_k, dtype=np.float32),
        temperature_segments=np.asarray(args.temperature_segments, dtype=np.int64),
        steps_per_temperature=np.asarray(args.steps_per_temperature, dtype=np.int64),
        model_path=np.asarray(str(args.model_path)),
        data_path=np.asarray(str(args.data_path)),
    )
    print(f"Saved {output_npz}")
    if args.trajectory_path:
        print(f"Saved {args.trajectory_path}")


if __name__ == "__main__":
    main()
