"""Run block/tile inference through discrete acceleration and Verlet update."""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    """Parse command-line options for Verlet-acceleration inference."""
    parser = argparse.ArgumentParser(description="Run model predictions as discrete accelerations.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--count-steps", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--crop-origin", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--crop-shape", type=int, nargs=3, required=True)
    parser.add_argument("--tile-shape", type=int, nargs=3, default=None)
    parser.add_argument(
        "--merge-mode",
        choices=["tile", "mean", "weighted", "owner", "center", "central_core"],
        default="tile",
    )
    parser.add_argument(
        "--core-shape",
        type=int,
        nargs=3,
        default=(1, 1, 1),
        help="Central block region used by central_core mode.",
    )
    parser.add_argument(
        "--periodic-blocks",
        action="store_true",
        help="Allow block extraction/merge to wrap around crop boundaries.",
    )
    parser.add_argument("--acceleration-beta", type=float, default=0.0)
    parser.add_argument("--acceleration-axes", nargs="*", choices=["x", "y", "z"], default=["x", "y", "z"])
    parser.add_argument("--zero-mean-acceleration", action="store_true")
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


def crop_slices(crop_origin, crop_shape):
    """Build x/y/z crop slices."""
    return tuple(slice(start, start + size) for start, size in zip(crop_origin, crop_shape))


def build_tile_origins(crop_shape, tile_shape):
    """Return all no-overlap tile origins for a crop."""
    if any(crop_dim % tile_dim != 0 for crop_dim, tile_dim in zip(crop_shape, tile_shape)):
        raise ValueError("crop_shape must be divisible by tile_shape")
    return [
        (ix, iy, iz)
        for ix in range(0, crop_shape[0], tile_shape[0])
        for iy in range(0, crop_shape[1], tile_shape[1])
        for iz in range(0, crop_shape[2], tile_shape[2])
    ]


def tile_index(origin, tile_shape):
    """Return a tuple of slices for one tile."""
    return tuple(slice(start, start + size) for start, size in zip(origin, tile_shape))


def periodic_block_index(origin, block_shape, crop_shape):
    """Return np.ix_ indices for a possibly wrapped block."""
    axes = [
        (np.arange(start, start + size, dtype=np.int64) % crop_dim)
        for start, size, crop_dim in zip(origin, block_shape, crop_shape)
    ]
    return np.ix_(axes[0], axes[1], axes[2])


def build_periodic_origins(crop_shape, stride_shape):
    """Return block origins that cover a periodic crop."""
    return [
        (ix, iy, iz)
        for ix in range(0, crop_shape[0], stride_shape[0])
        for iy in range(0, crop_shape[1], stride_shape[1])
        for iz in range(0, crop_shape[2], stride_shape[2])
    ]


def core_local_index(block_shape, core_shape):
    """Return local slices for the central core of one training block."""
    slices = []
    for block_dim, core_dim in zip(block_shape, core_shape):
        if core_dim > block_dim:
            raise ValueError("core_shape cannot be larger than tile_shape")
        start = (block_dim - core_dim) // 2
        slices.append(slice(start, start + core_dim))
    return tuple(slices)


def core_global_index(origin, block_shape, core_shape, crop_shape):
    """Return global indices covered by the central core of a block."""
    local = core_local_index(block_shape, core_shape)
    axes = []
    for axis, local_slice in enumerate(local):
        start = origin[axis] + local_slice.start
        stop = origin[axis] + local_slice.stop
        axes.append(np.arange(start, stop, dtype=np.int64) % crop_shape[axis])
    return np.ix_(axes[0], axes[1], axes[2])


def smooth_faces(field, tile_shape, beta, axes):
    """Softly smooth a field across no-overlap tile faces."""
    if beta == 0:
        return field
    if not 0 <= beta <= 1:
        raise ValueError("acceleration_beta must be between 0 and 1")
    corrected = field.clone()
    axis_map = {"x": 0, "y": 1, "z": 2}
    for axis_name in axes:
        axis = axis_map[axis_name]
        tile_dim = tile_shape[axis]
        for boundary in range(tile_dim, field.shape[axis], tile_dim):
            left_index = [slice(None)] * field.ndim
            right_index = [slice(None)] * field.ndim
            left_index[axis] = boundary - 1
            right_index[axis] = boundary
            left_index = tuple(left_index)
            right_index = tuple(right_index)
            mismatch = corrected[left_index] - corrected[right_index]
            corrected[left_index] = corrected[left_index] - 0.5 * beta * mismatch
            corrected[right_index] = corrected[right_index] + 0.5 * beta * mismatch
    return corrected


def merge_acceleration(acceleration_sum, acceleration_weight, index, acceleration, weights):
    """Accumulate one acceleration block into the global acceleration field."""
    acceleration_sum[index] += acceleration * weights
    acceleration_weight[index] += weights


def run_verlet(
    model,
    count_steps,
    init_displacements,
    tile_shape,
    merge_mode,
    acceleration_beta,
    acceleration_axes,
    zero_mean,
    core_shape,
    periodic_blocks,
):
    """Run autoregressive Verlet update from model-derived discrete accelerations."""
    crop_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
    if merge_mode == "tile":
        tile_origins = build_tile_origins(crop_shape, tile_shape)
        model_merge_mode = "owner"
        model_stride = tile_shape
    elif merge_mode == "central_core":
        tile_origins = build_periodic_origins(crop_shape, core_shape)
        model_merge_mode = "owner"
        model_stride = tile_shape
    else:
        tile_origins = [(0, 0, 0)]
        model_merge_mode = merge_mode
        model_stride = (1, 1, 1)

    x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
    predictions = []

    for _ in range(count_steps):
        acceleration_sum = torch.zeros_like(x[-1])
        acceleration_weight = torch.zeros((*crop_shape, x.shape[4], 1), dtype=x.dtype)

        for origin in tile_origins:
            if merge_mode == "tile":
                index = tile_index(origin, tile_shape)
                output_index = index
                local_output_index = tuple(slice(None) for _ in tile_shape)
            elif merge_mode == "central_core":
                index = periodic_block_index(origin, tile_shape, crop_shape)
                output_index = core_global_index(origin, tile_shape, core_shape, crop_shape)
                local_output_index = core_local_index(tile_shape, core_shape)
            else:
                index = (
                    slice(0, crop_shape[0]),
                    slice(0, crop_shape[1]),
                    slice(0, crop_shape[2]),
                )
                output_index = index
                local_output_index = tuple(slice(None) for _ in crop_shape)
            history = x[(slice(None), *index, slice(None), slice(None))]
            predicted_next = model.run_crystal(
                count_steps=1,
                init_displacements=history.detach().cpu().numpy(),
                stride_shape=model_stride,
                merge_mode=model_merge_mode,
                periodic=periodic_blocks,
            )[0]
            predicted_next = torch.as_tensor(predicted_next, dtype=x.dtype)
            acceleration = predicted_next - 2 * history[-1] + history[-2]
            acceleration = acceleration[(*local_output_index, slice(None), slice(None))]
            weights = torch.ones((*acceleration.shape[:3], 1, 1), dtype=x.dtype)
            merge_acceleration(
                acceleration_sum,
                acceleration_weight,
                (*output_index, slice(None), slice(None)),
                acceleration,
                weights,
            )

        if torch.any(acceleration_weight == 0):
            raise ValueError("Some cells were not covered by any acceleration block")
        acceleration_global = acceleration_sum / acceleration_weight
        if zero_mean:
            acceleration_global = acceleration_global - acceleration_global.mean(dim=(0, 1, 2, 3), keepdim=True)
        acceleration_global = smooth_faces(acceleration_global, tile_shape, acceleration_beta, acceleration_axes)
        y = 2 * x[-1] - x[-2] + acceleration_global
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
    """Run Verlet-acceleration inference and save a plot-compatible ``.npz`` file."""
    args = parse_args()
    if args.count_steps <= 0:
        raise ValueError("count_steps must be positive")
    crop_origin = _origin3("crop_origin", args.crop_origin)
    crop_shape = _shape3("crop_shape", args.crop_shape)
    data = np.load(args.data_path)
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    tile_shape = tuple(model.train_supercell_shape) if args.tile_shape is None else _shape3("tile_shape", args.tile_shape)
    sequence_length = int(data["X_blocks"].shape[1])
    if sequence_length < 2:
        raise ValueError("Verlet acceleration mode requires at least two history frames")
    slices = crop_slices(crop_origin, crop_shape)
    prediction_start = args.start_frame + sequence_length
    prediction_stop = prediction_start + args.count_steps
    if prediction_stop > data["displacements"].shape[0]:
        raise ValueError("Not enough frames for requested count_steps")

    init_displacements = data["displacements"][args.start_frame : prediction_start, *slices].astype(np.float32)
    predicted_displacements = run_verlet(
        model=model,
        count_steps=args.count_steps,
        init_displacements=init_displacements,
        tile_shape=tile_shape,
        merge_mode=args.merge_mode,
        acceleration_beta=float(args.acceleration_beta),
        acceleration_axes=tuple(args.acceleration_axes),
        zero_mean=args.zero_mean_acceleration,
        core_shape=_shape3("core_shape", args.core_shape),
        periodic_blocks=args.periodic_blocks,
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
        "merge_mode": np.asarray(args.merge_mode),
        "core_shape": np.asarray(args.core_shape, dtype=np.int64),
        "periodic_blocks": np.asarray(args.periodic_blocks),
        "acceleration_beta": np.asarray(args.acceleration_beta, dtype=np.float32),
        "acceleration_axes": np.asarray(args.acceleration_axes),
        "zero_mean_acceleration": np.asarray(args.zero_mean_acceleration),
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
