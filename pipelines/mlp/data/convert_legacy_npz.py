#!/usr/bin/env python3
"""Convert a force-enabled legacy crystal NPZ to the one-frame MLP format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    """Parse source and destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Legacy history frame used as the one-frame input; defaults to the latest.",
    )
    return parser.parse_args()


def main():
    """Validate legacy arrays and write only arrays required by the MLP pipeline."""
    args = parse_args()
    source_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path)
    with np.load(source_path) as source:
        required = {
            "X_blocks",
            "force_acceleration_blocks",
            "displacements",
            "atom_order",
            "reference_positions",
            "box_lengths",
            "crystal_shape",
            "train_supercell_shape",
            "dt_ps",
            "atom_mass_amu",
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"Legacy NPZ is missing arrays: {missing}")
        histories = source["X_blocks"]
        if histories.ndim != 7:
            raise ValueError("X_blocks must have shape (samples, history, 3, 3, 3, atoms, 3)")
        frame_index = int(args.frame_index)
        if not -histories.shape[1] <= frame_index < histories.shape[1]:
            raise ValueError("frame-index is outside the legacy history axis")
        input_blocks = histories[:, frame_index].astype(np.float32, copy=False)
        force_blocks = source["force_acceleration_blocks"]
        if force_blocks.shape != input_blocks.shape:
            raise ValueError("force_acceleration_blocks and selected input frame have different shapes")
        arrays = {
            "format": np.asarray("crystal-pair-energy-mlp-v1"),
            "input_blocks": input_blocks,
            "force_acceleration_blocks": force_blocks,
            "displacements": source["displacements"],
            "atom_order": source["atom_order"],
            "reference_positions": source["reference_positions"],
            "box_lengths": source["box_lengths"],
            "crystal_shape": source["crystal_shape"],
            "train_supercell_shape": source["train_supercell_shape"],
            "dt_ps": source["dt_ps"],
            "atom_mass_amu": source["atom_mass_amu"],
            "reference_mode": source["reference_mode"] if "reference_mode" in source.files else np.asarray("unknown"),
            "start_frame": source["start_frame"] if "start_frame" in source.files else np.asarray(0, dtype=np.int64),
            "source_npz": np.asarray(str(source_path)),
            "source_history_length": np.asarray(histories.shape[1], dtype=np.int64),
            "source_history_frame_index": np.asarray(frame_index, dtype=np.int64),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)

    with np.load(output_path) as converted:
        print(f"Saved {output_path.resolve()}")
        print(f"format = {converted['format'].item()}")
        print(f"input_blocks = {converted['input_blocks'].shape}")
        print(f"force_acceleration_blocks = {converted['force_acceleration_blocks'].shape}")
        print(f"displacements = {converted['displacements'].shape}")
        print(f"dt_ps = {float(converted['dt_ps']):.9g}")


if __name__ == "__main__":
    main()
