"""Run crystal inference with a saved `CrystalRNNNet` model."""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    """Parse command-line options for saved-model inference."""
    parser = argparse.ArgumentParser(description="Run inference with a saved crystal-aware RNN model.")
    parser.add_argument("--model-path", required=True, help="Path to a saved .pth model.")
    parser.add_argument("--data-path", required=True, help="Path to a crystal .npz dataset.")
    parser.add_argument("--output-path", required=True, help="Path to the output .npz file.")
    parser.add_argument("--count-steps", type=int, required=True, help="Number of autoregressive steps to predict.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame of the initial history inside data['displacements'].",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic wrapping when applying the training supercell.",
    )
    parser.add_argument(
        "--reference-output",
        action="store_true",
        help="Save the real continuation from the dataset for comparison.",
    )
    parser.add_argument(
        "--save-positions",
        action="store_true",
        help="Also save absolute positions in the original flat atom order.",
    )
    return parser.parse_args()


def load_crystal_data(path):
    """Load and validate the arrays needed for crystal inference."""
    data = np.load(path)
    required = ["X_blocks", "displacements", "atom_order", "reference_positions"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return {key: data[key] for key in data.files}


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacement frames to flat absolute positions."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def build_initial_history(displacements, sequence_length, start_frame):
    """Return the initial displacement history for autoregressive inference."""
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    stop_frame = start_frame + sequence_length
    if stop_frame > displacements.shape[0]:
        raise ValueError("Not enough frames for the requested start_frame and sequence_length")
    return displacements[start_frame:stop_frame].astype(np.float32)


def build_reference_output(displacements, sequence_length, start_frame, count_steps):
    """Return the real continuation that corresponds to the predicted frames."""
    prediction_start = start_frame + sequence_length
    prediction_stop = prediction_start + count_steps
    if prediction_stop > displacements.shape[0]:
        raise ValueError("Not enough frames to save the requested reference_output")
    return displacements[prediction_start:prediction_stop].astype(np.float32)


def main():
    """Load a saved model, run crystal inference, and save the result."""
    args = parse_args()
    if args.count_steps <= 0:
        raise ValueError("count_steps must be positive")

    data = load_crystal_data(args.data_path)
    sequence_length = int(data["X_blocks"].shape[1])
    init_displacements = build_initial_history(data["displacements"], sequence_length, args.start_frame)

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    predicted_displacements = model.run_crystal(
        count_steps=args.count_steps,
        init_displacements=init_displacements,
        periodic=args.periodic,
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
        "periodic": np.asarray(args.periodic),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
    }

    if hasattr(model, "train_supercell_shape") and model.train_supercell_shape is not None:
        output["train_supercell_shape"] = np.asarray(model.train_supercell_shape, dtype=np.int64)
    if hasattr(model, "unit_cell_atoms") and model.unit_cell_atoms is not None:
        output["unit_cell_atoms"] = np.asarray(model.unit_cell_atoms, dtype=np.int64)
    if hasattr(model, "flatten_order"):
        output["flatten_order"] = np.asarray(model.flatten_order)

    if args.reference_output:
        reference_displacements = build_reference_output(
            data["displacements"],
            sequence_length,
            args.start_frame,
            args.count_steps,
        )
        output["reference_displacements"] = reference_displacements

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
