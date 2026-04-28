import argparse
from pathlib import Path

import numpy as np

from base_classes import (
    FCC_CONVENTIONAL_BASIS,
    build_crystal_atom_order,
    make_crystal_block_samples,
    positions_to_crystal_displacements,
    read_lammps_dump_positions,
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_format == "dump":
        positions, box_lengths = read_lammps_dump_positions(
            args.input_path,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
        )
    else:
        positions = read_raw_positions(args.input_path, max_frames=args.max_frames, start_frame=args.start_frame)
        box_lengths = None

    if box_lengths is None:
        raise ValueError("Raw input currently requires box lengths; use dump input or add box metadata.")

    basis = FCC_CONVENTIONAL_BASIS if args.basis == "fcc" else None
    atom_order = build_crystal_atom_order(
        reference_positions=positions[0],
        crystal_shape=tuple(args.crystal_shape),
        box_lengths=box_lengths[0],
        unit_cell_atoms=args.unit_cell_atoms,
        basis_fractional=basis,
    )
    displacements = positions_to_crystal_displacements(
        positions=positions,
        reference_positions=positions[0],
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

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_blocks=X_blocks,
        y_blocks=y_blocks,
        displacements=displacements,
        atom_order=atom_order,
        reference_positions=positions[0],
        box_lengths=box_lengths,
        crystal_shape=np.asarray(args.crystal_shape, dtype=np.int64),
        train_supercell_shape=np.asarray(args.train_supercell_shape, dtype=np.int64),
        start_frame=np.asarray(args.start_frame, dtype=np.int64),
    )
    print(f"Saved {output_path}")
    print(f"X_blocks shape: {X_blocks.shape}")
    print(f"y_blocks shape: {y_blocks.shape}")


if __name__ == "__main__":
    main()
