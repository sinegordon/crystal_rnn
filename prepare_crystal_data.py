import argparse
from pathlib import Path

import numpy as np

from base_classes import (
    CU_MASS_AMU,
    FCC_CONVENTIONAL_BASIS,
    build_crystal_atom_order,
    flat_vectors_to_crystal_values,
    forces_to_discrete_accelerations,
    make_crystal_block_samples,
    positions_to_crystal_displacements,
    read_lammps_dump_arrays,
    read_raw_positions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare crystal-shaped RNN training data.")
    parser.add_argument("input_path", help="Path to a LAMMPS dump or raw coordinate file.")
    parser.add_argument("output_path", help="Path to the output .npz file.")
    parser.add_argument("--input-format", choices=["dump", "raw"], default="dump")
    parser.add_argument("--crystal-shape", nargs=3, type=int, required=True)
    parser.add_argument("--train-supercell-shape", nargs=3, type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--unit-cell-atoms", type=int, default=None)
    parser.add_argument("--basis", choices=["fcc", "none"], default="fcc")
    parser.add_argument("--stride-shape", nargs=3, type=int, default=None)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--include-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read fx fy fz from LAMMPS dump when available and store force-derived acceleration targets.",
    )
    parser.add_argument(
        "--require-forces",
        action="store_true",
        help="Fail if --include-forces is enabled but the dump does not contain fx fy fz.",
    )
    parser.add_argument("--dt-ps", type=float, default=0.002, help="Trajectory timestep in picoseconds.")
    parser.add_argument("--atom-mass-amu", type=float, default=CU_MASS_AMU, help="Atomic mass used for force conversion.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_format == "dump":
        dump_arrays = read_lammps_dump_arrays(
            path=args.input_path,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            read_forces=args.include_forces,
            require_forces=args.require_forces,
        )
        positions = dump_arrays["positions"]
        box_lengths = dump_arrays["box_lengths"]
        forces_ev_per_ang = dump_arrays.get("forces_ev_per_ang")
    else:
        positions = read_raw_positions(args.input_path, max_frames=args.max_frames, start_frame=args.start_frame)
        box_lengths = None
        forces_ev_per_ang = None

    if box_lengths is None:
        raise ValueError("Raw input currently requires box lengths; use dump input or add box metadata.")

    basis = FCC_CONVENTIONAL_BASIS if args.basis == "fcc" else None
    reference_positions = positions.mean(axis=0)
    atom_order = build_crystal_atom_order(
        reference_positions=reference_positions,
        crystal_shape=tuple(args.crystal_shape),
        box_lengths=box_lengths[0],
        unit_cell_atoms=args.unit_cell_atoms,
        basis_fractional=basis,
    )
    displacements = positions_to_crystal_displacements(
        positions=positions,
        reference_positions=reference_positions,
        atom_order=atom_order,
        box_lengths=box_lengths,
    )
    X_blocks, y_blocks = make_crystal_block_samples(
        displacements=displacements,
        train_supercell_shape=tuple(args.train_supercell_shape),
        sequence_length=args.sequence_length,
        stride_shape=None if args.stride_shape is None else tuple(args.stride_shape),
        periodic=args.periodic,
    )
    optional_arrays = {}
    if forces_ev_per_ang is not None:
        crystal_forces = flat_vectors_to_crystal_values(forces_ev_per_ang, atom_order)
        force_discrete_accelerations = forces_to_discrete_accelerations(
            crystal_forces,
            dt_ps=args.dt_ps,
            atom_mass_amu=args.atom_mass_amu,
        )
        _, force_acceleration_blocks = make_crystal_block_samples(
            displacements=displacements,
            train_supercell_shape=tuple(args.train_supercell_shape),
            sequence_length=args.sequence_length,
            stride_shape=None if args.stride_shape is None else tuple(args.stride_shape),
            periodic=args.periodic,
            target_fields=force_discrete_accelerations,
            target_frame_offset=args.sequence_length - 1,
        )
        optional_arrays.update(
            {
                "forces_ev_per_ang": forces_ev_per_ang,
                "crystal_forces_ev_per_ang": crystal_forces,
                "force_discrete_accelerations": force_discrete_accelerations,
                "force_acceleration_blocks": force_acceleration_blocks,
                "dt_ps": np.asarray(args.dt_ps, dtype=np.float32),
                "atom_mass_amu": np.asarray(args.atom_mass_amu, dtype=np.float32),
                "force_target_frame_offset": np.asarray(args.sequence_length - 1, dtype=np.int64),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_blocks=X_blocks,
        y_blocks=y_blocks,
        displacements=displacements,
        atom_order=atom_order,
        reference_positions=reference_positions,
        box_lengths=box_lengths,
        crystal_shape=np.asarray(args.crystal_shape, dtype=np.int64),
        train_supercell_shape=np.asarray(args.train_supercell_shape, dtype=np.int64),
        start_frame=np.asarray(args.start_frame, dtype=np.int64),
        reference_mode=np.asarray("mean"),
        **optional_arrays,
    )
    print(f"Saved {output_path}")
    print(f"X_blocks shape: {X_blocks.shape}")
    print(f"y_blocks shape: {y_blocks.shape}")
    if "force_acceleration_blocks" in optional_arrays:
        print(f"force_acceleration_blocks shape: {optional_arrays['force_acceleration_blocks'].shape}")


if __name__ == "__main__":
    main()
