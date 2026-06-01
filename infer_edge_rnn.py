"""Run full-crystal inference with a saved CrystalEdgeRNNNet model."""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes.edge_rnn_predictor import _extract_patch_batch, build_centers


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--count-steps", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-batch-size", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-output", action="store_true")
    parser.add_argument("--save-positions", action="store_true")
    parser.add_argument(
        "--q-zero-mode",
        choices=["none", "initial", "constant_velocity", "zero"],
        default="none",
        help="Control the spatial q=0 displacement mode after each rollout step.",
    )
    parser.add_argument(
        "--velocity-thermostat-interval",
        type=int,
        default=0,
        help="Apply a velocity-RMS thermostat every N rollout steps. Use 0 to disable it.",
    )
    parser.add_argument(
        "--velocity-thermostat-coupling",
        type=float,
        default=1.0,
        help="Berendsen-like coupling alpha for velocity-RMS correction.",
    )
    return parser.parse_args()


def resolve_device(device):
    """Resolve a requested torch device."""
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {device!r}, but CUDA is not available")
    if device.startswith("mps"):
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError(f"Requested device {device!r}, but MPS is not available")
    return torch.device(device)


def load_crystal_data(path):
    """Load arrays needed for edge-RNN inference."""
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return data


def spatial_q_zero(frame):
    """Return the per-atom, per-coordinate spatial q=0 mode."""
    return np.mean(np.asarray(frame, dtype=np.float32), axis=(0, 1, 2), keepdims=True)


def q_zero_target(step, mode, initial_q_zero, initial_q_zero_velocity):
    """Return the q=0 target for a predicted frame."""
    if mode == "initial":
        return initial_q_zero
    if mode == "constant_velocity":
        return initial_q_zero + float(step + 1) * initial_q_zero_velocity
    if mode == "zero":
        return np.zeros_like(initial_q_zero)
    raise ValueError(f"Unsupported q_zero_mode={mode!r}")


def apply_q_zero_control(next_frame, step, mode, initial_q_zero, initial_q_zero_velocity):
    """Replace the spatial q=0 mode without changing finite-q modes."""
    if mode == "none":
        return next_frame
    target = q_zero_target(step, mode, initial_q_zero, initial_q_zero_velocity)
    return (next_frame - spatial_q_zero(next_frame) + target).astype(np.float32)


def frame_velocity_rms(frame_delta):
    """Return global RMS of the frame-to-frame displacement increment."""
    return float(np.sqrt(np.mean(np.asarray(frame_delta, dtype=np.float64) ** 2)))


def apply_velocity_thermostat(current_frame, next_frame, target_rms, coupling, epsilon=1e-12):
    """Apply one Berendsen-like velocity-RMS correction."""
    velocity_delta = next_frame - current_frame
    current_rms = frame_velocity_rms(velocity_delta)
    if current_rms <= epsilon:
        return next_frame, 1.0, current_rms
    ratio = (target_rms / current_rms) ** 2
    scale = float(np.sqrt(max(1.0 + float(coupling) * (ratio - 1.0), epsilon)))
    return (current_frame + velocity_delta * scale).astype(np.float32), scale, current_rms


def predict_full_acceleration(model, history, centers, periodic, patch_batch_size):
    """Predict central-cell accelerations for every crystal cell."""
    acceleration = np.zeros_like(history[-1], dtype=np.float32)
    for start in range(0, len(centers), patch_batch_size):
        batch_centers = centers[start : start + patch_batch_size]
        patch_batch = _extract_patch_batch(history, batch_centers, model.patch_shape, periodic)
        center_acceleration = model.predict_center_accelerations(patch_batch)
        for local_index, center in enumerate(batch_centers):
            acceleration[(*center, slice(None), slice(None))] = center_acceleration[local_index]
    return acceleration


def run_edge_rollout(
    model,
    init_displacements,
    count_steps,
    periodic,
    patch_batch_size,
    q_zero_mode="none",
    velocity_thermostat_interval=0,
    velocity_thermostat_coupling=1.0,
):
    """Run global Verlet rollout driven by edge-RNN local accelerations."""
    if count_steps <= 0:
        raise ValueError("count_steps must be positive")
    if init_displacements.shape[0] < 2:
        raise ValueError("At least two history frames are required")
    if patch_batch_size <= 0:
        raise ValueError("patch_batch_size must be positive")
    if velocity_thermostat_interval < 0:
        raise ValueError("velocity_thermostat_interval must be non-negative")
    if not 0 < velocity_thermostat_coupling <= 1:
        raise ValueError("velocity_thermostat_coupling must be in (0, 1]")

    history = np.asarray(init_displacements, dtype=np.float32).copy()
    centers = build_centers(history.shape[1:4], periodic=periodic)
    target_velocity_rms = frame_velocity_rms(history[-1] - history[-2])
    initial_q_zero = spatial_q_zero(history[-1])
    initial_q_zero_velocity = spatial_q_zero(history[-1]) - spatial_q_zero(history[-2])
    predictions = []
    thermostat_scales = []
    thermostat_velocity_rms = []

    for step in range(count_steps):
        acceleration = predict_full_acceleration(model, history, centers, periodic, patch_batch_size)
        next_frame = 2.0 * history[-1] - history[-2] + acceleration
        next_frame = apply_q_zero_control(next_frame, step, q_zero_mode, initial_q_zero, initial_q_zero_velocity)
        if velocity_thermostat_interval and (step + 1) % velocity_thermostat_interval == 0:
            next_frame, scale, current_rms = apply_velocity_thermostat(
                current_frame=history[-1],
                next_frame=next_frame,
                target_rms=target_velocity_rms,
                coupling=velocity_thermostat_coupling,
            )
            next_frame = apply_q_zero_control(next_frame, step, q_zero_mode, initial_q_zero, initial_q_zero_velocity)
            thermostat_scales.append(scale)
            thermostat_velocity_rms.append(current_rms)
        predictions.append(next_frame.astype(np.float32))
        history[:-1] = history[1:]
        history[-1] = next_frame
        if step == 0 or (step + 1) % 100 == 0:
            print(f"STEP {step + 1}/{count_steps}", flush=True)

    metadata = {
        "velocity_thermostat_target_rms": np.asarray(target_velocity_rms, dtype=np.float32),
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
    start = start_frame + sequence_length
    stop = start + count_steps
    if stop > displacements.shape[0]:
        raise ValueError("Not enough frames to save the requested reference output")
    return displacements[start:stop].astype(np.float32)


def main():
    """Run edge-RNN inference and save a plot-compatible npz."""
    args = parse_args()
    data = load_crystal_data(args.data_path)
    sequence_length = int(data["X_blocks"].shape[1])
    init_stop = args.start_frame + sequence_length
    if args.start_frame < 0 or init_stop > data["displacements"].shape[0]:
        raise ValueError("Invalid start_frame for available displacement frames")

    device = resolve_device(args.device)
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "to"):
        model.to(device)
    init_displacements = data["displacements"][args.start_frame:init_stop].astype(np.float32)
    predicted_displacements, metadata = run_edge_rollout(
        model=model,
        init_displacements=init_displacements,
        count_steps=args.count_steps,
        periodic=bool(args.periodic),
        patch_batch_size=args.patch_batch_size,
        q_zero_mode=args.q_zero_mode,
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
        "periodic": np.asarray(bool(args.periodic)),
        "patch_batch_size": np.asarray(args.patch_batch_size, dtype=np.int64),
        "q_zero_mode": np.asarray(args.q_zero_mode),
        "velocity_thermostat_interval": np.asarray(args.velocity_thermostat_interval, dtype=np.int64),
        "velocity_thermostat_coupling": np.asarray(args.velocity_thermostat_coupling, dtype=np.float32),
        "velocity_thermostat_target_rms": metadata["velocity_thermostat_target_rms"],
        "velocity_thermostat_scales": metadata["velocity_thermostat_scales"],
        "velocity_thermostat_velocity_rms": metadata["velocity_thermostat_velocity_rms"],
        "initial_q_zero": metadata["initial_q_zero"],
        "initial_q_zero_velocity": metadata["initial_q_zero_velocity"],
        "inference_mode": np.asarray("edge_rnn_acceleration"),
        "device": np.asarray(str(device)),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
    }
    for attr in (
        "unit_cell_atoms",
        "hidden_size",
        "rnn_layers",
        "rnn_type",
        "bidirectional",
        "neighbor_shells",
        "neighbor_count",
        "cutoff_scale",
        "lattice_parameter",
        "acceleration_normalization",
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
