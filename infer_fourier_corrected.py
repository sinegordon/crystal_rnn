"""Run tiled inference with low-q Fourier correction of model-derived acceleration."""

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_tiled_boundary import (
    _origin3,
    _shape3,
    build_tiles,
    crop_slices,
    crystal_frames_to_flat_positions,
    remap_atom_order,
    tile_index,
)


def parse_args():
    """Parse command-line options for Fourier-corrected tiled inference."""
    parser = argparse.ArgumentParser(description="Correct low-q acceleration modes after no-overlap RNN inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--count-steps", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--crop-origin", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--crop-shape", type=int, nargs=3, required=True)
    parser.add_argument("--tile-shape", type=int, nargs=3, default=None)
    parser.add_argument(
        "--low-q-radius",
        type=float,
        default=1.0,
        help="Radius in integer FFT-mode units for corrected low-q modes.",
    )
    parser.add_argument(
        "--low-q-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to selected low-q acceleration Fourier modes.",
    )
    parser.add_argument(
        "--low-q-axis-scales",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help=(
            "Optional separate multipliers for pure ±qx, ±qy, ±qz low-q modes. "
            "Other selected low-q modes keep --low-q-scale."
        ),
    )
    parser.add_argument(
        "--include-zero-mode",
        action="store_true",
        help="Also scale the q=0 acceleration mode. Disabled by default to avoid center-of-mass drift.",
    )
    parser.add_argument(
        "--zero-mean-acceleration",
        action="store_true",
        help="Remove the real-space mean acceleration before Fourier correction.",
    )
    parser.add_argument("--reference-output", action="store_true")
    parser.add_argument("--save-positions", action="store_true")
    return parser.parse_args()


def build_fft_mode_axes(shape):
    """Return integer FFT-mode coordinate grids for a crystal shape."""
    return np.meshgrid(
        *[np.fft.fftfreq(size) * size for size in shape],
        indexing="ij",
    )


def build_fft_mode_mask(shape, low_q_radius, include_zero_mode):
    """Build a mask of low-q FFT modes in integer mode units."""
    if low_q_radius < 0:
        raise ValueError("low_q_radius must be non-negative")
    axes = build_fft_mode_axes(shape)
    radius = np.sqrt(sum(axis**2 for axis in axes))
    mask = radius <= low_q_radius
    if not include_zero_mode:
        mask &= radius > 0
    return mask


def build_fft_mode_scales(shape, low_q_radius, low_q_scale, axis_scales, include_zero_mode):
    """Build per-mode Fourier multipliers for low-q acceleration correction."""
    if axis_scales is not None and len(axis_scales) != 3:
        raise ValueError("axis_scales must contain three values")
    axes = build_fft_mode_axes(shape)
    radius = np.sqrt(sum(axis**2 for axis in axes))
    mask = build_fft_mode_mask(shape, low_q_radius, include_zero_mode)
    scales = np.ones(shape, dtype=np.float32)
    scales[mask] = low_q_scale

    if axis_scales is not None:
        for axis_index, axis_scale in enumerate(axis_scales):
            pure_axis_mask = mask & (np.abs(axes[axis_index]) > 0)
            for other_axis in range(3):
                if other_axis != axis_index:
                    pure_axis_mask &= np.isclose(axes[other_axis], 0)
            scales[pure_axis_mask] = float(axis_scale)
    return scales


def correct_low_q_acceleration(acceleration, mode_scales):
    """Scale selected spatial Fourier modes of an acceleration field."""
    if np.allclose(mode_scales, 1.0):
        return acceleration
    spectrum = np.fft.fftn(acceleration, axes=(0, 1, 2))
    spectrum *= mode_scales[..., None, None]
    corrected = np.fft.ifftn(spectrum, axes=(0, 1, 2)).real
    return corrected.astype(np.float32)


def run_fourier_corrected(
    model,
    count_steps,
    init_displacements,
    tile_shape,
    low_q_radius,
    low_q_scale,
    low_q_axis_scales,
    include_zero_mode,
    zero_mean_acceleration,
):
    """Run no-overlap local inference and correct low-q acceleration modes."""
    crop_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
    tiles = build_tiles(crop_shape, tile_shape)
    mode_scales = build_fft_mode_scales(crop_shape, low_q_radius, low_q_scale, low_q_axis_scales, include_zero_mode)
    x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
    predictions = []

    for _ in range(count_steps):
        local_next = torch.zeros_like(x[-1])
        for origin in tiles:
            index = tile_index(origin, tile_shape)
            tile_history = x[(slice(None), *index, slice(None), slice(None))].detach().cpu().numpy()
            tile_prediction = model.run_crystal(
                count_steps=1,
                init_displacements=tile_history,
                stride_shape=tile_shape,
                merge_mode="owner",
            )[0]
            local_next[(*index, slice(None), slice(None))] = torch.as_tensor(tile_prediction, dtype=x.dtype)

        acceleration = (local_next - 2 * x[-1] + x[-2]).detach().cpu().numpy()
        if zero_mean_acceleration:
            acceleration = acceleration - acceleration.mean(axis=(0, 1, 2, 3), keepdims=True)
        acceleration = correct_low_q_acceleration(acceleration, mode_scales)
        y = 2 * x[-1] - x[-2] + torch.as_tensor(acceleration, dtype=x.dtype)
        predictions.append(y.detach().cpu().numpy())
        x[:-1] = x[1:].clone()
        x[-1] = y

    return np.asarray(predictions, dtype=np.float32)


def main():
    """Run Fourier-corrected inference and save a plot-compatible ``.npz`` file."""
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

    sequence_length = int(data["X_blocks"].shape[1])
    slices = crop_slices(crop_origin, crop_shape)
    prediction_start = args.start_frame + sequence_length
    prediction_stop = prediction_start + args.count_steps
    if prediction_stop > data["displacements"].shape[0]:
        raise ValueError("Not enough frames for requested count_steps")

    init_displacements = data["displacements"][args.start_frame : prediction_start, *slices].astype(np.float32)
    predicted_displacements = run_fourier_corrected(
        model=model,
        count_steps=args.count_steps,
        init_displacements=init_displacements,
        tile_shape=tile_shape,
        low_q_radius=float(args.low_q_radius),
        low_q_scale=float(args.low_q_scale),
        low_q_axis_scales=None if args.low_q_axis_scales is None else tuple(args.low_q_axis_scales),
        include_zero_mode=args.include_zero_mode,
        zero_mean_acceleration=args.zero_mean_acceleration,
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
        "low_q_radius": np.asarray(args.low_q_radius, dtype=np.float32),
        "low_q_scale": np.asarray(args.low_q_scale, dtype=np.float32),
        "low_q_axis_scales": np.asarray(
            [] if args.low_q_axis_scales is None else args.low_q_axis_scales,
            dtype=np.float32,
        ),
        "include_zero_mode": np.asarray(args.include_zero_mode),
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
