"""Run no-overlap tiled inference while cyclically shifting tile origins."""

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_tiled_boundary import (
    _origin3,
    _shape3,
    crop_slices,
    crystal_frames_to_flat_positions,
    remap_atom_order,
)


def parse_args():
    """Parse command-line options for cyclic-shift tiled inference."""
    parser = argparse.ArgumentParser(description="Run no-overlap inference with moving periodic tile boundaries.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--count-steps", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--crop-origin", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--crop-shape", type=int, nargs=3, required=True)
    parser.add_argument("--tile-shape", type=int, nargs=3, default=None)
    parser.add_argument(
        "--shift-schedule",
        choices=["fixed", "x", "diagonal", "cycle"],
        default="cycle",
        help="How tile origins are shifted between autoregressive steps.",
    )
    parser.add_argument("--reference-output", action="store_true")
    parser.add_argument("--save-positions", action="store_true")
    return parser.parse_args()


def validate_tiling(crop_shape, tile_shape):
    """Ensure a periodic no-overlap tiling exists for the crop."""
    if any(crop_dim % tile_dim != 0 for crop_dim, tile_dim in zip(crop_shape, tile_shape)):
        raise ValueError("crop_shape must be divisible by tile_shape")


def periodic_tile_index(origin, tile_shape, crop_shape):
    """Return np.ix_ indices for one periodic tile."""
    axes = [
        (np.arange(start, start + size, dtype=np.int64) % crop_dim)
        for start, size, crop_dim in zip(origin, tile_shape, crop_shape)
    ]
    return np.ix_(axes[0], axes[1], axes[2])


def shifted_tile_origins(crop_shape, tile_shape, offset):
    """Return all tile origins for one shifted periodic no-overlap tiling."""
    validate_tiling(crop_shape, tile_shape)
    offset = tuple(int(value) % tile_dim for value, tile_dim in zip(offset, tile_shape))
    return [
        ((offset[0] + ix) % crop_shape[0], (offset[1] + iy) % crop_shape[1], (offset[2] + iz) % crop_shape[2])
        for ix in range(0, crop_shape[0], tile_shape[0])
        for iy in range(0, crop_shape[1], tile_shape[1])
        for iz in range(0, crop_shape[2], tile_shape[2])
    ]


def cycle_offsets(tile_shape):
    """Return the full offset cycle, with x varying fastest."""
    return [
        (ox, oy, oz)
        for oz in range(tile_shape[2])
        for oy in range(tile_shape[1])
        for ox in range(tile_shape[0])
    ]


def step_offset(step_index, tile_shape, schedule):
    """Return the tile offset for one autoregressive step."""
    if schedule == "fixed":
        return (0, 0, 0)
    if schedule == "x":
        return (step_index % tile_shape[0], 0, 0)
    if schedule == "diagonal":
        return tuple(step_index % dim for dim in tile_shape)
    if schedule == "cycle":
        offsets = cycle_offsets(tile_shape)
        return offsets[step_index % len(offsets)]
    raise ValueError(f"Unsupported shift schedule: {schedule}")


def run_cyclic_shift(model, count_steps, init_displacements, tile_shape, shift_schedule):
    """Run tiled autoregression with moving periodic no-overlap boundaries."""
    crop_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
    validate_tiling(crop_shape, tile_shape)
    x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
    predictions = []

    for step_index in range(count_steps):
        y = torch.zeros_like(x[-1])
        offset = step_offset(step_index, tile_shape, shift_schedule)
        for origin in shifted_tile_origins(crop_shape, tile_shape, offset):
            index = periodic_tile_index(origin, tile_shape, crop_shape)
            tile_history = x[(slice(None), *index, slice(None), slice(None))].detach().cpu().numpy()
            tile_prediction = model.run_crystal(
                count_steps=1,
                init_displacements=tile_history,
                stride_shape=tile_shape,
                merge_mode="owner",
            )[0]
            y[(*index, slice(None), slice(None))] = torch.as_tensor(tile_prediction, dtype=x.dtype)

        predictions.append(y.detach().cpu().numpy())
        x[:-1] = x[1:].clone()
        x[-1] = y

    return np.asarray(predictions, dtype=np.float32)


def main():
    """Run cyclic-shift inference and save a plot-compatible ``.npz`` file."""
    args = parse_args()
    if args.count_steps <= 0:
        raise ValueError("count_steps must be positive")
    crop_origin = _origin3("crop_origin", args.crop_origin)
    crop_shape = _shape3("crop_shape", args.crop_shape)
    data = np.load(args.data_path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {args.data_path}: {missing}")

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    tile_shape = tuple(model.train_supercell_shape) if args.tile_shape is None else _shape3("tile_shape", args.tile_shape)
    validate_tiling(crop_shape, tile_shape)

    sequence_length = int(data["X_blocks"].shape[1])
    slices = crop_slices(crop_origin, crop_shape)
    prediction_start = args.start_frame + sequence_length
    prediction_stop = prediction_start + args.count_steps
    if prediction_stop > data["displacements"].shape[0]:
        raise ValueError("Not enough frames for requested count_steps")

    init_displacements = data["displacements"][args.start_frame : prediction_start, *slices].astype(np.float32)
    predicted_displacements = run_cyclic_shift(
        model=model,
        count_steps=args.count_steps,
        init_displacements=init_displacements,
        tile_shape=tile_shape,
        shift_schedule=args.shift_schedule,
    )

    cropped_atom_order_original = data["atom_order"][slices]
    atom_order, used_atoms = remap_atom_order(cropped_atom_order_original)
    reference_positions = data["reference_positions"][used_atoms]

    output = {
        "predicted_displacements": predicted_displacements,
        "init_displacements": init_displacements,
        "reference_positions": reference_positions,
        "atom_order": atom_order,
        "start_frame": np.asarray(args.start_frame, dtype=np.int64),
        "sequence_length": np.asarray(sequence_length, dtype=np.int64),
        "prediction_start_frame": np.asarray(prediction_start, dtype=np.int64),
        "count_steps": np.asarray(args.count_steps, dtype=np.int64),
        "crop_origin": np.asarray(crop_origin, dtype=np.int64),
        "crop_shape": np.asarray(crop_shape, dtype=np.int64),
        "tile_shape": np.asarray(tile_shape, dtype=np.int64),
        "shift_schedule": np.asarray(args.shift_schedule),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
    }

    if args.reference_output:
        output["reference_displacements"] = data["displacements"][prediction_start:prediction_stop, *slices].astype(
            np.float32
        )
    if args.save_positions:
        output["predicted_positions"] = crystal_frames_to_flat_positions(
            predicted_displacements,
            reference_positions,
            atom_order,
        )
        output["init_positions"] = crystal_frames_to_flat_positions(init_displacements, reference_positions, atom_order)
        if "reference_displacements" in output:
            output["reference_positions_output"] = crystal_frames_to_flat_positions(
                output["reference_displacements"],
                reference_positions,
                atom_order,
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"Saved {output_path}")
    print("predicted_displacements", predicted_displacements.shape)


if __name__ == "__main__":
    main()
