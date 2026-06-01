"""Rebuild a prepared crystal dataset with reference positions from another dataset."""

import argparse
from pathlib import Path

import numpy as np

from base_classes import make_crystal_block_samples


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True, help="Prepared source .npz dataset to rebase.")
    parser.add_argument("--reference-path", required=True, help="Prepared .npz dataset that provides reference metadata.")
    parser.add_argument("--output-path", required=True, help="Output rebased .npz path.")
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic block extraction when rebuilding X_blocks/y_blocks.",
    )
    return parser.parse_args()


def first_box_lengths(box_lengths):
    """Return one orthorhombic box-length vector."""
    box_lengths = np.asarray(box_lengths, dtype=np.float32)
    if box_lengths.ndim == 2:
        return box_lengths[0]
    if box_lengths.shape != (3,):
        raise ValueError("box_lengths must have shape (3,) or (frames, 3)")
    return box_lengths


def rebase_displacements(source, reference):
    """Return source displacements expressed relative to the reference dataset."""
    source_atom_order = np.asarray(source["atom_order"], dtype=np.int64)
    reference_atom_order = np.asarray(reference["atom_order"], dtype=np.int64)
    if not np.array_equal(source_atom_order, reference_atom_order):
        raise ValueError("source and reference atom_order arrays differ; rebasing is ambiguous")

    source_reference = np.asarray(source["reference_positions"], dtype=np.float32)
    target_reference = np.asarray(reference["reference_positions"], dtype=np.float32)
    reference_delta = source_reference - target_reference
    crystal_reference_delta = reference_delta[reference_atom_order]

    box_lengths = first_box_lengths(source["box_lengths"])
    displacements = np.asarray(source["displacements"], dtype=np.float32) + crystal_reference_delta[None, ...]
    displacements -= box_lengths * np.round(displacements / box_lengths)
    return displacements.astype(np.float32)


def main():
    """Rebase the source dataset and rebuild its training samples."""
    args = parse_args()
    source = np.load(args.source_path)
    reference = np.load(args.reference_path)
    required_source = ["displacements", "X_blocks", "atom_order", "reference_positions", "box_lengths"]
    required_reference = ["atom_order", "reference_positions"]
    missing_source = [key for key in required_source if key not in source.files]
    missing_reference = [key for key in required_reference if key not in reference.files]
    if missing_source:
        raise ValueError(f"Missing source arrays: {missing_source}")
    if missing_reference:
        raise ValueError(f"Missing reference arrays: {missing_reference}")

    sequence_length = int(source["X_blocks"].shape[1])
    train_supercell_shape = tuple(int(value) for value in source["train_supercell_shape"])
    displacements = rebase_displacements(source, reference)
    X_blocks, y_blocks = make_crystal_block_samples(
        displacements=displacements,
        train_supercell_shape=train_supercell_shape,
        sequence_length=sequence_length,
        periodic=bool(args.periodic),
    )
    optional_arrays = {}
    if "force_discrete_accelerations" in source.files:
        force_discrete_accelerations = np.asarray(source["force_discrete_accelerations"], dtype=np.float32)
        _, force_acceleration_blocks = make_crystal_block_samples(
            displacements=displacements,
            train_supercell_shape=train_supercell_shape,
            sequence_length=sequence_length,
            periodic=bool(args.periodic),
            target_fields=force_discrete_accelerations,
            target_frame_offset=sequence_length - 1,
        )
        optional_arrays["force_discrete_accelerations"] = force_discrete_accelerations
        optional_arrays["force_acceleration_blocks"] = force_acceleration_blocks
        for key in [
            "forces_ev_per_ang",
            "crystal_forces_ev_per_ang",
            "dt_ps",
            "atom_mass_amu",
            "force_target_frame_offset",
        ]:
            if key in source.files:
                optional_arrays[key] = source[key]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_blocks=X_blocks,
        y_blocks=y_blocks,
        displacements=displacements,
        atom_order=reference["atom_order"],
        reference_positions=reference["reference_positions"],
        box_lengths=source["box_lengths"],
        crystal_shape=source["crystal_shape"],
        train_supercell_shape=source["train_supercell_shape"],
        start_frame=source["start_frame"] if "start_frame" in source.files else np.asarray(0, dtype=np.int64),
        reference_mode=np.asarray("rebased"),
        reference_source_path=np.asarray(str(args.reference_path)),
        source_path=np.asarray(str(args.source_path)),
        **optional_arrays,
    )
    print(f"Saved {output_path}")
    print(f"X_blocks shape: {X_blocks.shape}")
    print(f"y_blocks shape: {y_blocks.shape}")
    if "force_acceleration_blocks" in optional_arrays:
        print(f"force_acceleration_blocks shape: {optional_arrays['force_acceleration_blocks'].shape}")


if __name__ == "__main__":
    main()
