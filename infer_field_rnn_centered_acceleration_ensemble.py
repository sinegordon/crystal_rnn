"""Apply several FieldRNN acceleration models as one averaged local operator."""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes.conv_crystal_predictor import _channels_to_crystal, _crystal_to_channels
from infer_field_rnn_centered_acceleration import (
    build_centers,
    build_reference_output,
    crystal_frames_to_flat_positions,
    denormalize_acceleration,
    extract_patch_batch,
    apply_velocity_thermostat,
    frame_velocity_rms,
    load_crystal_data,
    resolve_device,
    shape3,
)


def parse_args():
    """Parse command-line options for centered acceleration ensemble inference."""
    parser = argparse.ArgumentParser(
        description="Run several FieldRNN acceleration models and average central-cell accelerations."
    )
    parser.add_argument("--model-path", action="append", required=True, help="Path to a saved acceleration model.")
    parser.add_argument(
        "--model-weight",
        action="append",
        type=float,
        default=None,
        help="Optional model weight. If omitted, all models are weighted equally.",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--count-steps", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap local patches around crystal boundaries.",
    )
    parser.add_argument("--patch-batch-size", type=int, default=250)
    parser.add_argument(
        "--velocity-thermostat-interval",
        type=int,
        default=0,
        help=(
            "Apply a Berendsen-like velocity RMS thermostat every N rollout steps. "
            "Use 0 to disable it."
        ),
    )
    parser.add_argument(
        "--velocity-thermostat-coupling",
        type=float,
        default=1.0,
        help=(
            "Berendsen coupling alpha in lambda=sqrt(1+alpha*(T0/T-1)). "
            "alpha=1 restores the initial RMS exactly; smaller values are softer."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-output", action="store_true")
    parser.add_argument("--save-positions", action="store_true")
    return parser.parse_args()


def normalize_weights(weights, count):
    """Return normalized ensemble weights with one entry per model."""
    if weights is None:
        weights = np.ones(count, dtype=np.float32)
    else:
        if len(weights) != count:
            raise ValueError("--model-weight must be passed once per --model-path")
        weights = np.asarray(weights, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("--model-weight values must be non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("At least one model weight must be positive")
    return weights / weight_sum


def load_models(paths, device):
    """Load acceleration models and move them to the requested device."""
    models = []
    for path in paths:
        model = torch.load(path, map_location="cpu", weights_only=False)
        if getattr(model, "target_mode", None) != "acceleration":
            raise ValueError(f"{path} is not an acceleration model")
        if hasattr(model, "to"):
            model.to(device)
        model.model.eval()
        models.append(model)
    return models


def predict_center_accelerations_ensemble(models, weights, history, centers, patch_shape, periodic, patch_batch_size, device):
    """Predict averaged central-cell accelerations for all global cells."""
    unit_cell_atoms = int(models[0].unit_cell_atoms)
    if any(int(model.unit_cell_atoms) != unit_cell_atoms for model in models):
        raise ValueError("All ensemble models must use the same unit_cell_atoms")

    acceleration = np.zeros_like(history[-1], dtype=np.float32)
    center_index = tuple(dim // 2 for dim in patch_shape)

    with torch.no_grad():
        for start in range(0, len(centers), patch_batch_size):
            batch_centers = centers[start : start + patch_batch_size]
            patch_batch = extract_patch_batch(history, batch_centers, patch_shape, periodic)
            x = _crystal_to_channels(patch_batch).to(device)
            center_acceleration_sum = None

            for model, weight in zip(models, weights):
                x_model = model._transform_input_channels(x) if hasattr(model, "_transform_input_channels") else x
                raw = model.model(x_model)
                raw = denormalize_acceleration(model, raw)
                raw_crystal = _channels_to_crystal(raw, unit_cell_atoms)
                center_acceleration = raw_crystal[(slice(None), *center_index, slice(None), slice(None))]
                weighted = center_acceleration * float(weight)
                center_acceleration_sum = weighted if center_acceleration_sum is None else center_acceleration_sum + weighted

            center_acceleration_np = center_acceleration_sum.detach().cpu().numpy()
            for local_index, global_center in enumerate(batch_centers):
                acceleration[(*global_center, slice(None), slice(None))] = center_acceleration_np[local_index]

    return acceleration


def run_centered_acceleration_ensemble(
    models,
    weights,
    init_displacements,
    count_steps,
    patch_shape,
    periodic,
    patch_batch_size,
    device,
    velocity_thermostat_interval=0,
    velocity_thermostat_coupling=1.0,
):
    """Run Verlet rollout using ensemble-averaged local accelerations."""
    if count_steps <= 0:
        raise ValueError("count_steps must be positive")
    if init_displacements.shape[0] < 2:
        raise ValueError("At least two history frames are required for acceleration inference")
    if velocity_thermostat_interval < 0:
        raise ValueError("velocity_thermostat_interval must be non-negative")
    if not 0 < velocity_thermostat_coupling <= 1:
        raise ValueError("velocity_thermostat_coupling must be in (0, 1]")

    crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
    centers = build_centers(crystal_shape, patch_shape, periodic)
    history = np.asarray(init_displacements, dtype=np.float32).copy()
    predictions = []
    target_velocity_rms = frame_velocity_rms(history[-1] - history[-2])
    thermostat_scales = []
    thermostat_velocity_rms = []

    for step in range(count_steps):
        acceleration = predict_center_accelerations_ensemble(
            models=models,
            weights=weights,
            history=history,
            centers=centers,
            patch_shape=patch_shape,
            periodic=periodic,
            patch_batch_size=patch_batch_size,
            device=device,
        )
        next_frame = 2 * history[-1] - history[-2] + acceleration
        if velocity_thermostat_interval and (step + 1) % velocity_thermostat_interval == 0:
            next_frame, scale, current_rms = apply_velocity_thermostat(
                previous_frame=history[-2],
                current_frame=history[-1],
                next_frame=next_frame,
                target_rms=target_velocity_rms,
                coupling=velocity_thermostat_coupling,
            )
            thermostat_scales.append(scale)
            thermostat_velocity_rms.append(current_rms)
            print(
                f"THERMOSTAT STEP {step + 1}: velocity_rms={current_rms:.6g}, "
                f"target={target_velocity_rms:.6g}, scale={scale:.6g}",
                flush=True,
            )
        predictions.append(next_frame.astype(np.float32))
        history[:-1] = history[1:]
        history[-1] = next_frame
        if (step + 1) % 100 == 0 or step == 0:
            print(f"STEP {step + 1}/{count_steps}", flush=True)

    metadata = {
        "velocity_thermostat_target_rms": target_velocity_rms,
        "velocity_thermostat_scales": np.asarray(thermostat_scales, dtype=np.float32),
        "velocity_thermostat_velocity_rms": np.asarray(thermostat_velocity_rms, dtype=np.float32),
    }
    return np.asarray(predictions, dtype=np.float32), metadata


def main():
    """Run centered local-acceleration ensemble inference and save an npz."""
    args = parse_args()
    patch_shape = shape3("patch_shape", args.patch_shape)
    if args.patch_batch_size <= 0:
        raise ValueError("patch_batch_size must be positive")
    if args.velocity_thermostat_interval < 0:
        raise ValueError("velocity-thermostat-interval must be non-negative")
    if not 0 < args.velocity_thermostat_coupling <= 1:
        raise ValueError("velocity-thermostat-coupling must be in (0, 1]")

    data = load_crystal_data(args.data_path)
    sequence_length = int(data["X_blocks"].shape[1])
    init_stop = args.start_frame + sequence_length
    if args.start_frame < 0 or init_stop > data["displacements"].shape[0]:
        raise ValueError("Invalid start_frame for available displacement frames")
    init_displacements = data["displacements"][args.start_frame:init_stop].astype(np.float32)

    device = resolve_device(args.device)
    weights = normalize_weights(args.model_weight, len(args.model_path))
    models = load_models(args.model_path, device)

    predicted_displacements, thermostat_metadata = run_centered_acceleration_ensemble(
        models=models,
        weights=weights,
        init_displacements=init_displacements,
        count_steps=args.count_steps,
        patch_shape=patch_shape,
        periodic=bool(args.periodic),
        patch_batch_size=args.patch_batch_size,
        device=device,
        velocity_thermostat_interval=args.velocity_thermostat_interval,
        velocity_thermostat_coupling=args.velocity_thermostat_coupling,
    )

    output = {
        "predicted_displacements": predicted_displacements,
        "init_displacements": init_displacements,
        "reference_positions": data["reference_positions"],
        "atom_order": data["atom_order"],
        "start_frame": np.asarray(args.start_frame, dtype=np.int64),
        "sequence_length": np.asarray(sequence_length, dtype=np.int64),
        "prediction_start_frame": np.asarray(args.start_frame + sequence_length, dtype=np.int64),
        "count_steps": np.asarray(args.count_steps, dtype=np.int64),
        "patch_shape": np.asarray(patch_shape, dtype=np.int64),
        "periodic": np.asarray(bool(args.periodic)),
        "patch_batch_size": np.asarray(args.patch_batch_size, dtype=np.int64),
        "velocity_thermostat_interval": np.asarray(args.velocity_thermostat_interval, dtype=np.int64),
        "velocity_thermostat_coupling": np.asarray(args.velocity_thermostat_coupling, dtype=np.float32),
        "velocity_thermostat_target_rms": np.asarray(
            thermostat_metadata["velocity_thermostat_target_rms"],
            dtype=np.float32,
        ),
        "velocity_thermostat_scales": thermostat_metadata["velocity_thermostat_scales"],
        "velocity_thermostat_velocity_rms": thermostat_metadata["velocity_thermostat_velocity_rms"],
        "inference_mode": np.asarray("centered_acceleration_ensemble"),
        "device": np.asarray(str(device)),
        "model_paths": np.asarray([str(path) for path in args.model_path]),
        "model_weights": np.asarray(weights, dtype=np.float32),
        "data_path": np.asarray(str(args.data_path)),
    }

    if args.reference_output:
        output["reference_displacements"] = build_reference_output(
            data["displacements"],
            sequence_length,
            args.start_frame,
            args.count_steps,
        )

    if args.save_positions:
        output["predicted_positions"] = crystal_frames_to_flat_positions(
            predicted_displacements,
            data["reference_positions"],
            data["atom_order"],
        )
        output["init_positions"] = crystal_frames_to_flat_positions(
            init_displacements,
            data["reference_positions"],
            data["atom_order"],
        )
        if "reference_displacements" in output:
            output["reference_positions_output"] = crystal_frames_to_flat_positions(
                output["reference_displacements"],
                data["reference_positions"],
                data["atom_order"],
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"Saved {output_path}")
    print("predicted_displacements", predicted_displacements.shape)


if __name__ == "__main__":
    main()
