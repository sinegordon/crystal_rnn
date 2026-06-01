"""Apply a FieldRNN acceleration model as a centered local operator."""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes.conv_crystal_predictor import _channels_to_crystal, _crystal_to_channels


def parse_args():
    """Parse command-line options for centered FieldRNN acceleration inference."""
    parser = argparse.ArgumentParser(
        description="Run a FieldRNN acceleration model on local patches and keep only central-cell accelerations."
    )
    parser.add_argument("--model-path", required=True)
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
    parser.add_argument(
        "--thermostat-log-interval",
        type=int,
        default=1,
        help="Print thermostat diagnostics every N thermostat applications. Use 0 to silence them.",
    )
    parser.add_argument(
        "--integrator",
        choices=["verlet", "predictor_corrector"],
        default="verlet",
        help=(
            "Rollout integrator. 'verlet' uses one model acceleration per step; "
            "'predictor_corrector' recomputes acceleration on the predicted next frame "
            "and blends the two accelerations."
        ),
    )
    parser.add_argument(
        "--corrector-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of the corrected acceleration for predictor_corrector. "
            "0.5 means arithmetic averaging of predictor and corrector accelerations."
        ),
    )
    parser.add_argument(
        "--q-zero-mode",
        choices=["none", "initial", "constant_velocity", "zero"],
        default="none",
        help=(
            "Control the spatial q=0 displacement mode after each rollout step. "
            "'initial' keeps each atom/channel spatial mean equal to the last input frame; "
            "'constant_velocity' continues the initial q=0 velocity; "
            "'zero' removes the q=0 displacement mode."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-output", action="store_true")
    parser.add_argument("--save-positions", action="store_true")
    return parser.parse_args()


def resolve_device(device):
    """Resolve a requested torch device."""
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {device!r}, but CUDA is not available")
    return torch.device(device)


def shape3(name, value):
    """Validate a positive odd 3D shape."""
    if len(value) != 3:
        raise ValueError(f"{name} must contain three dimensions")
    value = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in value):
        raise ValueError(f"{name} dimensions must be positive")
    if any(dim % 2 == 0 for dim in value):
        raise ValueError(f"{name} dimensions must be odd for a unique center cell")
    return value


def load_crystal_data(path):
    """Load arrays needed for crystal inference without forcing X_blocks into memory."""
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return data


def build_centers(crystal_shape, patch_shape, periodic):
    """Return global cells for which centered patches can be extracted."""
    if periodic:
        return [
            (ix, iy, iz)
            for ix in range(crystal_shape[0])
            for iy in range(crystal_shape[1])
            for iz in range(crystal_shape[2])
        ]

    radius = tuple(dim // 2 for dim in patch_shape)
    centers = [
        (ix, iy, iz)
        for ix in range(radius[0], crystal_shape[0] - radius[0])
        for iy in range(radius[1], crystal_shape[1] - radius[1])
        for iz in range(radius[2], crystal_shape[2] - radius[2])
    ]
    if len(centers) != int(np.prod(crystal_shape)):
        raise ValueError("periodic=False does not cover boundary cells for full-crystal inference")
    return centers


def centered_patch_index(center, patch_shape, crystal_shape, periodic):
    """Build np.ix_ indices for a patch centered at one global cell."""
    axes = []
    for center_axis, patch_dim, crystal_dim in zip(center, patch_shape, crystal_shape):
        radius = patch_dim // 2
        values = np.arange(center_axis - radius, center_axis + radius + 1, dtype=np.int64)
        if periodic:
            values = values % crystal_dim
        elif np.any((values < 0) | (values >= crystal_dim)):
            raise ValueError("Patch crosses crystal boundary with periodic=False")
        axes.append(values)
    return np.ix_(axes[0], axes[1], axes[2])


def extract_patch_batch(history, centers, patch_shape, periodic):
    """Extract a batch of centered local history patches."""
    crystal_shape = tuple(int(dim) for dim in history.shape[1:4])
    patches = []
    for center in centers:
        index = centered_patch_index(center, patch_shape, crystal_shape, periodic)
        patches.append(history[(slice(None), *index, slice(None), slice(None))])
    return np.asarray(patches, dtype=np.float32)


def denormalize_acceleration(model, raw_prediction):
    """Return acceleration prediction in physical displacement units."""
    if getattr(model, "target_mode", None) != "acceleration":
        raise ValueError("Centered acceleration inference requires target_mode='acceleration'")
    if hasattr(model, "_denormalize_acceleration_prediction"):
        return model._denormalize_acceleration_prediction(raw_prediction)
    return raw_prediction


def predict_center_accelerations(model, history, centers, patch_shape, periodic, patch_batch_size, device):
    """Predict one acceleration vector for the central cell of every patch."""
    unit_cell_atoms = int(model.unit_cell_atoms)
    acceleration = np.zeros_like(history[-1], dtype=np.float32)
    center_index = tuple(dim // 2 for dim in patch_shape)
    model.model.eval()

    with torch.no_grad():
        for start in range(0, len(centers), patch_batch_size):
            batch_centers = centers[start : start + patch_batch_size]
            patch_batch = extract_patch_batch(history, batch_centers, patch_shape, periodic)
            x = _crystal_to_channels(patch_batch).to(device)
            if hasattr(model, "_transform_input_channels"):
                x_model = model._transform_input_channels(x)
            else:
                x_model = x
            raw = model.model(x_model)
            raw = denormalize_acceleration(model, raw)
            raw_crystal = _channels_to_crystal(raw, unit_cell_atoms)
            center_acceleration = raw_crystal[
                (slice(None), *center_index, slice(None), slice(None))
            ].detach().cpu().numpy()
            for local_index, global_center in enumerate(batch_centers):
                acceleration[(*global_center, slice(None), slice(None))] = center_acceleration[local_index]

    return acceleration


def frame_velocity_rms(frame_delta):
    """Return global RMS of the frame-to-frame displacement increment."""
    return float(np.sqrt(np.mean(np.asarray(frame_delta, dtype=np.float64) ** 2)))


def apply_velocity_thermostat(previous_frame, current_frame, next_frame, target_rms, coupling, epsilon=1e-12):
    """Apply one Berendsen-like scaling step to the Verlet velocity increment."""
    del previous_frame
    velocity_delta = next_frame - current_frame
    current_rms = frame_velocity_rms(velocity_delta)
    if current_rms <= epsilon:
        return next_frame, 1.0, current_rms

    temperature_ratio = (target_rms / current_rms) ** 2
    scale_squared = 1.0 + coupling * (temperature_ratio - 1.0)
    scale = float(np.sqrt(max(scale_squared, epsilon)))
    corrected_next_frame = current_frame + velocity_delta * scale
    return corrected_next_frame.astype(np.float32), scale, current_rms


def shift_history(history, next_frame):
    """Return a new history window with ``next_frame`` appended."""
    shifted = np.empty_like(history, dtype=np.float32)
    shifted[:-1] = history[1:]
    shifted[-1] = next_frame
    return shifted


def spatial_q_zero(frame):
    """Return the per-atom, per-coordinate spatial q=0 mode."""
    return np.mean(np.asarray(frame, dtype=np.float32), axis=(0, 1, 2), keepdims=True)


def q_zero_target(step, mode, initial_q_zero, initial_q_zero_velocity):
    """Return the requested q=0 target for a predicted frame."""
    if mode == "initial":
        return initial_q_zero
    if mode == "constant_velocity":
        return initial_q_zero + float(step + 1) * initial_q_zero_velocity
    if mode == "zero":
        return np.zeros_like(initial_q_zero)
    raise ValueError(f"Unsupported q_zero_mode={mode!r}")


def apply_q_zero_control(next_frame, step, mode, initial_q_zero, initial_q_zero_velocity):
    """Replace the spatial q=0 displacement mode without touching finite-q modes."""
    if mode == "none":
        return next_frame
    target = q_zero_target(step, mode, initial_q_zero, initial_q_zero_velocity)
    corrected = next_frame - spatial_q_zero(next_frame) + target
    return corrected.astype(np.float32)


def run_centered_acceleration(
    model,
    init_displacements,
    count_steps,
    patch_shape,
    periodic,
    patch_batch_size,
    device,
    velocity_thermostat_interval=0,
    velocity_thermostat_coupling=1.0,
    thermostat_log_interval=1,
    integrator="verlet",
    corrector_weight=0.5,
    q_zero_mode="none",
):
    """Run global Verlet rollout with centered local acceleration predictions."""
    if count_steps <= 0:
        raise ValueError("count_steps must be positive")
    if init_displacements.shape[0] < 2:
        raise ValueError("At least two history frames are required for acceleration inference")
    if velocity_thermostat_interval < 0:
        raise ValueError("velocity_thermostat_interval must be non-negative")
    if not 0 < velocity_thermostat_coupling <= 1:
        raise ValueError("velocity_thermostat_coupling must be in (0, 1]")
    if thermostat_log_interval < 0:
        raise ValueError("thermostat_log_interval must be non-negative")
    if integrator not in {"verlet", "predictor_corrector"}:
        raise ValueError("integrator must be 'verlet' or 'predictor_corrector'")
    if not 0 <= corrector_weight <= 1:
        raise ValueError("corrector_weight must be in [0, 1]")
    if q_zero_mode not in {"none", "initial", "constant_velocity", "zero"}:
        raise ValueError("q_zero_mode must be 'none', 'initial', 'constant_velocity', or 'zero'")

    crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
    centers = build_centers(crystal_shape, patch_shape, periodic)
    history = np.asarray(init_displacements, dtype=np.float32).copy()
    predictions = []
    target_velocity_rms = frame_velocity_rms(history[-1] - history[-2])
    initial_q_zero = spatial_q_zero(history[-1])
    initial_q_zero_velocity = spatial_q_zero(history[-1]) - spatial_q_zero(history[-2])
    thermostat_scales = []
    thermostat_velocity_rms = []

    for step in range(count_steps):
        predictor_acceleration = predict_center_accelerations(
            model=model,
            history=history,
            centers=centers,
            patch_shape=patch_shape,
            periodic=periodic,
            patch_batch_size=patch_batch_size,
            device=device,
        )
        base_next = 2 * history[-1] - history[-2]
        next_frame = base_next + predictor_acceleration
        if integrator == "predictor_corrector":
            predicted_history = shift_history(history, next_frame)
            corrected_acceleration = predict_center_accelerations(
                model=model,
                history=predicted_history,
                centers=centers,
                patch_shape=patch_shape,
                periodic=periodic,
                patch_batch_size=patch_batch_size,
                device=device,
            )
            acceleration = (1.0 - corrector_weight) * predictor_acceleration + corrector_weight * corrected_acceleration
            next_frame = base_next + acceleration
        next_frame = apply_q_zero_control(
            next_frame,
            step,
            q_zero_mode,
            initial_q_zero,
            initial_q_zero_velocity,
        )
        if velocity_thermostat_interval and (step + 1) % velocity_thermostat_interval == 0:
            next_frame, scale, current_rms = apply_velocity_thermostat(
                previous_frame=history[-2],
                current_frame=history[-1],
                next_frame=next_frame,
                target_rms=target_velocity_rms,
                coupling=velocity_thermostat_coupling,
            )
            next_frame = apply_q_zero_control(
                next_frame,
                step,
                q_zero_mode,
                initial_q_zero,
                initial_q_zero_velocity,
            )
            thermostat_scales.append(scale)
            thermostat_velocity_rms.append(current_rms)
            thermostat_application = len(thermostat_scales)
            if thermostat_log_interval and thermostat_application % thermostat_log_interval == 0:
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
        "initial_q_zero": initial_q_zero.astype(np.float32),
        "initial_q_zero_velocity": initial_q_zero_velocity.astype(np.float32),
    }
    return np.asarray(predictions, dtype=np.float32), metadata


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacement frames to flat absolute positions."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def build_reference_output(displacements, sequence_length, start_frame, count_steps):
    """Return the real continuation matching predicted frames."""
    prediction_start = start_frame + sequence_length
    prediction_stop = prediction_start + count_steps
    if prediction_stop > displacements.shape[0]:
        raise ValueError("Not enough frames to save the requested reference output")
    return displacements[prediction_start:prediction_stop].astype(np.float32)


def main():
    """Run centered local-acceleration inference and save a plot-compatible npz."""
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
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "to"):
        model.to(device)

    predicted_displacements, thermostat_metadata = run_centered_acceleration(
        model=model,
        init_displacements=init_displacements,
        count_steps=args.count_steps,
        patch_shape=patch_shape,
        periodic=bool(args.periodic),
        patch_batch_size=args.patch_batch_size,
        device=device,
        velocity_thermostat_interval=args.velocity_thermostat_interval,
        velocity_thermostat_coupling=args.velocity_thermostat_coupling,
        thermostat_log_interval=args.thermostat_log_interval,
        integrator=args.integrator,
        corrector_weight=args.corrector_weight,
        q_zero_mode=args.q_zero_mode,
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
        "thermostat_log_interval": np.asarray(args.thermostat_log_interval, dtype=np.int64),
        "integrator": np.asarray(args.integrator),
        "corrector_weight": np.asarray(args.corrector_weight, dtype=np.float32),
        "q_zero_mode": np.asarray(args.q_zero_mode),
        "initial_q_zero": thermostat_metadata["initial_q_zero"],
        "initial_q_zero_velocity": thermostat_metadata["initial_q_zero_velocity"],
        "velocity_thermostat_target_rms": np.asarray(
            thermostat_metadata["velocity_thermostat_target_rms"],
            dtype=np.float32,
        ),
        "velocity_thermostat_scales": thermostat_metadata["velocity_thermostat_scales"],
        "velocity_thermostat_velocity_rms": thermostat_metadata["velocity_thermostat_velocity_rms"],
        "inference_mode": np.asarray("centered_acceleration"),
        "device": np.asarray(str(device)),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
    }
    for attr in (
        "unit_cell_atoms",
        "target_mode",
        "acceleration_normalization",
        "loss_region",
        "encoder_channels",
        "rnn_hidden_size",
        "rnn_layers",
        "bidirectional",
        "conv_layers",
        "kernel_size",
        "cyclic_shift_augmentation",
        "input_transform",
        "input_relative_scale",
        "force_balance_loss_weight",
    ):
        if hasattr(model, attr):
            output[attr] = np.asarray(getattr(model, attr))
    if hasattr(model, "acceleration_mean"):
        output["acceleration_mean"] = np.asarray(model.acceleration_mean, dtype=np.float32)
    if hasattr(model, "acceleration_std"):
        output["acceleration_std"] = np.asarray(model.acceleration_std, dtype=np.float32)

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
