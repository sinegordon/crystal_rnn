"""Save the rollout trajectory used by FieldRNN S(q,w) model selection."""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the train-window reference and autoregressive prediction "
            "used by find_field_rnn_models.py during S(q,w) model selection."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", default="data333.npz")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--delta-frames", type=int, default=30000)
    parser.add_argument("--count-steps", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device):
    """Resolve a requested torch device."""
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {device!r}, but CUDA is not available")
    return torch.device(device)


def load_training_data(path):
    """Load arrays needed to reproduce the model-selection rollout."""
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return data


def select_train_window(displacements, delta_frames, sequence_length, rng):
    """Repeat find_field_rnn_models.sample_train_data window selection."""
    frame_count = int(displacements.shape[0])
    if frame_count <= sequence_length:
        raise ValueError("Need more displacement frames than sequence_length")
    if delta_frames <= sequence_length:
        raise ValueError("delta-frames must be greater than sequence_length")

    if frame_count <= delta_frames:
        start_frame = 0
        window_frame_count = frame_count
    else:
        start_frame = int(rng.integers(0, frame_count - delta_frames + 1))
        window_frame_count = int(delta_frames)
    stop_frame = start_frame + window_frame_count
    return start_frame, stop_frame


def sample_initial_sequence(displacements, sequence_length, count_steps, rng):
    """Repeat find_field_rnn_models.sample_initial_crystal_sequence with the begin index."""
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = int(rng.integers(sequence_length, displacements.shape[0] - count_steps))
    init = displacements[begin_index - sequence_length : begin_index].astype(np.float32)
    return begin_index, init


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacements to flat absolute coordinates."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def main():
    """Recreate and save a model-selection rollout."""
    args = parse_args()
    if args.count_steps <= 0:
        raise ValueError("count-steps must be positive")
    if args.delta_frames <= 0:
        raise ValueError("delta-frames must be positive")

    data = load_training_data(args.data_path)
    displacements = data["displacements"].astype(np.float32)
    sequence_length = int(data["X_blocks"].shape[1])
    rng = np.random.default_rng(args.seed)

    train_start, train_stop = select_train_window(
        displacements=displacements,
        delta_frames=args.delta_frames,
        sequence_length=sequence_length,
        rng=rng,
    )
    prediction_begin, init_displacements = sample_initial_sequence(
        displacements=displacements,
        sequence_length=sequence_length,
        count_steps=args.count_steps,
        rng=rng,
    )

    device = resolve_device(args.device)
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "to"):
        model.to(device)

    predicted_displacements = model.run_crystal(args.count_steps, init_displacements)
    reference_displacements = displacements[train_start:train_stop].astype(np.float32)

    output = {
        "predicted_displacements": predicted_displacements,
        "reference_displacements": reference_displacements,
        "init_displacements": init_displacements,
        "reference_positions": data["reference_positions"],
        "atom_order": data["atom_order"],
        "seed": np.asarray(args.seed, dtype=np.int64),
        "delta_frames": np.asarray(args.delta_frames, dtype=np.int64),
        "count_steps": np.asarray(args.count_steps, dtype=np.int64),
        "sequence_length": np.asarray(sequence_length, dtype=np.int64),
        "train_start_frame": np.asarray(train_start, dtype=np.int64),
        "train_stop_frame": np.asarray(train_stop, dtype=np.int64),
        "prediction_begin_frame": np.asarray(prediction_begin, dtype=np.int64),
        "prediction_start_frame": np.asarray(prediction_begin, dtype=np.int64),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
        "rollout_mode": np.asarray("field_rnn_model_selection"),
        "device": np.asarray(str(device)),
    }
    output["predicted_positions"] = crystal_frames_to_flat_positions(
        predicted_displacements,
        data["reference_positions"],
        data["atom_order"],
    )
    output["reference_positions_output"] = crystal_frames_to_flat_positions(
        reference_displacements,
        data["reference_positions"],
        data["atom_order"],
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"Saved {output_path}")
    print(f"TRAIN FRAMES {train_start}:{train_stop}")
    print(f"PREDICTION BEGIN {prediction_begin}")
    print(f"predicted_displacements {predicted_displacements.shape}")
    print(f"reference_displacements {reference_displacements.shape}")


if __name__ == "__main__":
    main()
