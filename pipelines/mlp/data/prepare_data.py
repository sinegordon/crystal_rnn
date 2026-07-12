#!/usr/bin/env python3
"""Prepare one-frame force-training data for CrystalPairEnergyMLPNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from base_classes import (  # noqa: E402
    CU_MASS_AMU,
    FCC_CONVENTIONAL_BASIS,
    build_crystal_atom_order,
    flat_vectors_to_crystal_values,
    forces_to_discrete_accelerations,
    make_crystal_block_samples,
    positions_to_crystal_displacements,
    read_lammps_dump_arrays,
)


def parse_args():
    """Parse data-preparation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="LAMMPS dump containing id, x/y/z, and fx/fy/fz columns.")
    parser.add_argument("output_path", help="Output MLP .npz path.")
    parser.add_argument("--crystal-shape", nargs=3, type=int, required=True)
    parser.add_argument("--train-supercell-shape", nargs=3, type=int, default=(3, 3, 3))
    parser.add_argument("--stride-shape", nargs=3, type=int, default=None)
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--unit-cell-atoms", type=int, default=4)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--atom-mass-amu", type=float, default=CU_MASS_AMU)
    return parser.parse_args()


def main():
    """Convert a LAMMPS trajectory to the standalone MLP format."""
    args = parse_args()
    arrays = read_lammps_dump_arrays(
        args.input_path,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        read_forces=True,
        require_forces=True,
    )
    positions = arrays["positions"]
    box_lengths = arrays["box_lengths"]
    forces = arrays["forces_ev_per_ang"]
    reference_positions = positions.mean(axis=0)
    atom_order = build_crystal_atom_order(
        reference_positions,
        tuple(args.crystal_shape),
        box_lengths[0],
        unit_cell_atoms=args.unit_cell_atoms,
        basis_fractional=FCC_CONVENTIONAL_BASIS,
    )
    displacements = positions_to_crystal_displacements(
        positions,
        reference_positions,
        atom_order,
        box_lengths,
    )
    crystal_forces = flat_vectors_to_crystal_values(forces, atom_order)
    force_accelerations = forces_to_discrete_accelerations(
        crystal_forces,
        dt_ps=args.dt_ps,
        atom_mass_amu=args.atom_mass_amu,
    )
    block_args = {
        "train_supercell_shape": tuple(args.train_supercell_shape),
        "sequence_length": 1,
        "stride_shape": None if args.stride_shape is None else tuple(args.stride_shape),
        "periodic": args.periodic,
    }
    histories, _ = make_crystal_block_samples(displacements, **block_args)
    _, force_blocks = make_crystal_block_samples(
        displacements,
        target_fields=force_accelerations,
        target_frame_offset=0,
        **block_args,
    )
    input_blocks = histories[:, 0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format=np.asarray("crystal-pair-energy-mlp-v1"),
        input_blocks=input_blocks,
        force_acceleration_blocks=force_blocks,
        displacements=displacements,
        force_discrete_accelerations=force_accelerations,
        forces_ev_per_ang=forces,
        crystal_forces_ev_per_ang=crystal_forces,
        atom_order=atom_order,
        reference_positions=reference_positions,
        box_lengths=box_lengths,
        crystal_shape=np.asarray(args.crystal_shape, dtype=np.int64),
        train_supercell_shape=np.asarray(args.train_supercell_shape, dtype=np.int64),
        dt_ps=np.asarray(args.dt_ps, dtype=np.float32),
        atom_mass_amu=np.asarray(args.atom_mass_amu, dtype=np.float32),
        start_frame=np.asarray(args.start_frame, dtype=np.int64),
        reference_mode=np.asarray("mean"),
    )
    print(f"Saved {output_path}")
    print(f"input_blocks shape: {input_blocks.shape}")
    print(f"force_acceleration_blocks shape: {force_blocks.shape}")
    print(f"displacements shape: {displacements.shape}")


if __name__ == "__main__":
    main()
