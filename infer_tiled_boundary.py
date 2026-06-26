"""Run tiled no-overlap crystal inference with optional boundary correction."""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    """Parse command-line options for tiled boundary-corrected inference."""
    parser = argparse.ArgumentParser(description="Run no-overlap tiled inference with face smoothing.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--count-steps", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--crop-origin", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--crop-shape", type=int, nargs=3, required=True)
    parser.add_argument("--tile-shape", type=int, nargs=3, default=None)
    parser.add_argument("--boundary-beta", type=float, default=0.0)
    parser.add_argument("--boundary-axes", nargs="*", choices=["x", "y", "z"], default=["x", "y", "z"])
    parser.add_argument("--reference-output", action="store_true")
    parser.add_argument("--save-positions", action="store_true")
    return parser.parse_args()


def _shape3(name, value):
    """Validate a positive 3D integer shape."""
    if len(value) != 3:
        raise ValueError(f"{name} must contain three dimensions")
    value = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in value):
        raise ValueError(f"{name} dimensions must be positive")
    return value


def _origin3(name, value):
    """Validate a non-negative 3D integer origin."""
    if len(value) != 3:
        raise ValueError(f"{name} must contain three dimensions")
    value = tuple(int(dim) for dim in value)
    if any(dim < 0 for dim in value):
        raise ValueError(f"{name} dimensions must be non-negative")
    return value


def load_crystal_data(path):
    """Load arrays needed for tiled crystal inference."""
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return {key: data[key] for key in data.files}


def crop_slices(crop_origin, crop_shape):
    """Build x/y/z crop slices."""
    return tuple(slice(start, start + size) for start, size in zip(crop_origin, crop_shape))


def build_tiles(crop_shape, tile_shape):
    """Return all no-overlap tile origins for a crop."""
    if any(crop_dim % tile_dim != 0 for crop_dim, tile_dim in zip(crop_shape, tile_shape)):
        raise ValueError("crop_shape must be divisible by tile_shape for no-overlap tiled inference")
    return [
        (ix, iy, iz)
        for ix in range(0, crop_shape[0], tile_shape[0])
        for iy in range(0, crop_shape[1], tile_shape[1])
        for iz in range(0, crop_shape[2], tile_shape[2])
    ]


def tile_index(origin, tile_shape):
    """Return a tuple of slices for one tile."""
    return tuple(slice(start, start + size) for start, size in zip(origin, tile_shape))


def apply_face_smoothing(frame, tile_shape, beta, axes):
    """Softly equalize opposite faces of neighboring no-overlap tiles.

    The correction is conservative per face pair: one side moves by ``-beta/2``
    of the face mismatch and the other by ``+beta/2``.
    """
    if beta == 0:
        return frame
    if not 0 <= beta <= 1:
        raise ValueError("boundary_beta must be between 0 and 1")

    corrected = frame.clone()
    axis_map = {"x": 0, "y": 1, "z": 2}
    for axis_name in axes:
        axis = axis_map[axis_name]
        tile_dim = tile_shape[axis]
        for boundary in range(tile_dim, frame.shape[axis], tile_dim):
            left_index = [slice(None)] * frame.ndim
            right_index = [slice(None)] * frame.ndim
            left_index[axis] = boundary - 1
            right_index[axis] = boundary
            left_index = tuple(left_index)
            right_index = tuple(right_index)
            mismatch = corrected[left_index] - corrected[right_index]
            corrected[left_index] = corrected[left_index] - 0.5 * beta * mismatch
            corrected[right_index] = corrected[right_index] + 0.5 * beta * mismatch
    return corrected


def run_tiled(model, count_steps, init_displacements, tile_shape, boundary_beta, boundary_axes):
    """Run no-overlap tiled autoregressive inference."""
    crop_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
    tiles = build_tiles(crop_shape, tile_shape)
    x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
    predictions = []

    for _ in range(count_steps):
        y = torch.zeros_like(x[-1])
        for origin in tiles:
            index = tile_index(origin, tile_shape)
            tile_history = x[(slice(None), *index, slice(None), slice(None))].detach().cpu().numpy()
            tile_prediction = model.run_crystal(
                count_steps=1,
                init_displacements=tile_history,
                stride_shape=tile_shape,
                merge_mode="owner",
            )[0]
            y[(*index, slice(None), slice(None))] = torch.as_tensor(tile_prediction, dtype=x.dtype)

        y = apply_face_smoothing(y, tile_shape, boundary_beta, boundary_axes)
        predictions.append(y.detach().cpu().numpy())
        x[:-1] = x[1:].clone()
        x[-1] = y

    return np.asarray(predictions, dtype=np.float32)


def remap_atom_order(atom_order):
    """Remap cropped original atom ids to compact local ids for plotting."""
    used = atom_order.reshape(-1)
    remap = {int(old): index for index, old in enumerate(used)}
    return np.vectorize(lambda value: remap[int(value)])(atom_order).astype(np.int64), used


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacement frames to flat absolute positions."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def main():
    """Run tiled inference and save a plot-compatible ``.npz`` file."""
    args = parse_args()
    if args.count_steps <= 0:
        raise ValueError("count_steps must be positive")
    crop_origin = _origin3("crop_origin", args.crop_origin)
    crop_shape = _shape3("crop_shape", args.crop_shape)
    data = load_crystal_data(args.data_path)
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    tile_shape = tuple(model.train_supercell_shape) if args.tile_shape is None else _shape3("tile_shape", args.tile_shape)

    sequence_length = int(data["X_blocks"].shape[1])
    slices = crop_slices(crop_origin, crop_shape)
    init_start = args.start_frame
    init_stop = init_start + sequence_length
    prediction_start = init_stop
    prediction_stop = prediction_start + args.count_steps
    if prediction_stop > data["displacements"].shape[0]:
        raise ValueError("Not enough frames for requested count_steps")

    init_displacements = data["displacements"][init_start:init_stop, *slices].astype(np.float32)
    predicted_displacements = run_tiled(
        model=model,
        count_steps=args.count_steps,
        init_displacements=init_displacements,
        tile_shape=tile_shape,
        boundary_beta=float(args.boundary_beta),
        boundary_axes=tuple(args.boundary_axes),
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
        "boundary_beta": np.asarray(args.boundary_beta, dtype=np.float32),
        "boundary_axes": np.asarray(args.boundary_axes),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
    }

    if args.reference_output:
        reference_displacements = data["displacements"][prediction_start:prediction_stop, *slices].astype(np.float32)
        output["reference_displacements"] = reference_displacements

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
