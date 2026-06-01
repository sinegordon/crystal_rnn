"""Run crystal inference with a saved crystal predictor model."""

import argparse
import inspect
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
        "--merge-mode",
        choices=[
            "mean",
            "weighted",
            "center",
            "owner",
            "fixed_random_owner",
            "random_owner",
            "soft_center",
            "robust_center",
            "robust_mean",
            "delta_mean",
            "delta_weighted",
            "delta_center",
            "delta_owner",
            "delta_fixed_random_owner",
            "delta_random_owner",
            "delta_soft_center",
            "delta_robust_center",
            "delta_robust_mean",
        ],
        default="mean",
        help="How to stitch overlapping block predictions.",
    )
    parser.add_argument(
        "--merge-top-k",
        type=int,
        default=None,
        help="Number of candidate block predictions to keep for soft_center or robust_mean.",
    )
    parser.add_argument(
        "--merge-alpha",
        type=float,
        default=1.0,
        help="Distance weight for soft_center or centrality penalty for robust merge modes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for fixed_random_owner or random_owner merge modes.",
    )
    parser.add_argument(
        "--owner-order",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        default="xyz",
        help="Axis priority for owner-style merge modes.",
    )
    parser.add_argument(
        "--owner-reverse",
        nargs="*",
        choices=["x", "y", "z"],
        default=[],
        help="Axes traversed in descending order for owner-style merge modes.",
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for models that support device placement: auto, cpu, cuda, cuda:0, etc.",
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


def run_saved_model(model, count_steps, init_displacements, args):
    """Run either blockwise RNN or direct ConvRNN crystal inference."""
    signature = inspect.signature(model.run_crystal)
    actual_periodic = args.periodic or bool(getattr(model, "default_periodic", False))
    kwargs = {
        "count_steps": count_steps,
        "init_displacements": init_displacements,
    }
    if "periodic" in signature.parameters:
        kwargs["periodic"] = actual_periodic
    if "merge_mode" in signature.parameters:
        kwargs["merge_mode"] = args.merge_mode
    if "merge_top_k" in signature.parameters:
        kwargs["merge_top_k"] = args.seed if args.merge_mode in {"fixed_random_owner", "random_owner"} else args.merge_top_k
    if "merge_alpha" in signature.parameters:
        kwargs["merge_alpha"] = args.merge_alpha
    if "owner_order" in signature.parameters:
        kwargs["owner_order"] = tuple(args.owner_order)
    if "owner_reverse" in signature.parameters:
        kwargs["owner_reverse"] = tuple(args.owner_reverse)
    return model.run_crystal(**kwargs)


def main():
    """Load a saved model, run crystal inference, and save the result."""
    args = parse_args()
    if args.count_steps <= 0:
        raise ValueError("count_steps must be positive")

    data = load_crystal_data(args.data_path)
    sequence_length = int(data["X_blocks"].shape[1])
    init_displacements = build_initial_history(data["displacements"], sequence_length, args.start_frame)

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "to"):
        model.to(args.device)
    actual_periodic = args.periodic or bool(getattr(model, "default_periodic", False))
    predicted_displacements = run_saved_model(model, args.count_steps, init_displacements, args)

    output = {
        "predicted_displacements": predicted_displacements,
        "init_displacements": init_displacements,
        "reference_positions": data["reference_positions"],
        "atom_order": data["atom_order"],
        "start_frame": np.asarray(args.start_frame, dtype=np.int64),
        "sequence_length": np.asarray(sequence_length, dtype=np.int64),
        "prediction_start_frame": np.asarray(args.start_frame + sequence_length, dtype=np.int64),
        "count_steps": np.asarray(args.count_steps, dtype=np.int64),
        "periodic": np.asarray(actual_periodic),
        "merge_mode": np.asarray(args.merge_mode),
        "merge_top_k": np.asarray(-1 if args.merge_top_k is None else args.merge_top_k, dtype=np.int64),
        "merge_alpha": np.asarray(args.merge_alpha, dtype=np.float32),
        "device": np.asarray(args.device),
        "model_path": np.asarray(str(args.model_path)),
        "data_path": np.asarray(str(args.data_path)),
    }

    if hasattr(model, "train_supercell_shape") and model.train_supercell_shape is not None:
        output["train_supercell_shape"] = np.asarray(model.train_supercell_shape, dtype=np.int64)
    if hasattr(model, "unit_cell_atoms") and model.unit_cell_atoms is not None:
        output["unit_cell_atoms"] = np.asarray(model.unit_cell_atoms, dtype=np.int64)
    if hasattr(model, "flatten_order"):
        output["flatten_order"] = np.asarray(model.flatten_order)
    if hasattr(model, "target_mode"):
        output["target_mode"] = np.asarray(model.target_mode)
    if hasattr(model, "delta_loss_weight"):
        output["delta_loss_weight"] = np.asarray(model.delta_loss_weight, dtype=np.float32)
    if hasattr(model, "delta_loss_epsilon"):
        output["delta_loss_epsilon"] = np.asarray(model.delta_loss_epsilon, dtype=np.float32)
    if hasattr(model, "loss_weight_mode"):
        output["loss_weight_mode"] = np.asarray(model.loss_weight_mode)
    if hasattr(model, "center_loss_weight"):
        output["center_loss_weight"] = np.asarray(model.center_loss_weight, dtype=np.float32)
    if hasattr(model, "center_loss_alpha"):
        output["center_loss_alpha"] = np.asarray(model.center_loss_alpha, dtype=np.float32)
    if hasattr(model, "encoder_channels"):
        output["encoder_channels"] = np.asarray(model.encoder_channels, dtype=np.int64)
    if hasattr(model, "rnn_hidden_size"):
        output["rnn_hidden_size"] = np.asarray(model.rnn_hidden_size, dtype=np.int64)
    if hasattr(model, "rnn_layers"):
        output["rnn_layers"] = np.asarray(model.rnn_layers, dtype=np.int64)
    if hasattr(model, "bidirectional"):
        output["bidirectional"] = np.asarray(model.bidirectional)
    if hasattr(model, "conv_layers"):
        output["conv_layers"] = np.asarray(model.conv_layers, dtype=np.int64)
    if hasattr(model, "kernel_size"):
        output["kernel_size"] = np.asarray(model.kernel_size, dtype=np.int64)
    if hasattr(model, "acceleration_normalization"):
        output["acceleration_normalization"] = np.asarray(model.acceleration_normalization)
    if hasattr(model, "loss_region"):
        output["loss_region"] = np.asarray(model.loss_region)
    if hasattr(model, "cyclic_shift_augmentation"):
        output["cyclic_shift_augmentation"] = np.asarray(model.cyclic_shift_augmentation)
    if hasattr(model, "input_transform"):
        output["input_transform"] = np.asarray(model.input_transform)
    if hasattr(model, "input_relative_scale"):
        output["input_relative_scale"] = np.asarray(model.input_relative_scale, dtype=np.float32)
    if hasattr(model, "force_balance_loss_weight"):
        output["force_balance_loss_weight"] = np.asarray(model.force_balance_loss_weight, dtype=np.float32)
    if hasattr(model, "acceleration_normalization_epsilon"):
        output["acceleration_normalization_epsilon"] = np.asarray(
            model.acceleration_normalization_epsilon,
            dtype=np.float32,
        )
    if hasattr(model, "acceleration_mean"):
        output["acceleration_mean"] = np.asarray(model.acceleration_mean, dtype=np.float32)
    if hasattr(model, "acceleration_std"):
        output["acceleration_std"] = np.asarray(model.acceleration_std, dtype=np.float32)
    if hasattr(model, "low_q_stiffness_loss_weight"):
        output["low_q_stiffness_loss_weight"] = np.asarray(model.low_q_stiffness_loss_weight, dtype=np.float32)
    if hasattr(model, "low_q_stiffness_max_shell"):
        output["low_q_stiffness_max_shell"] = np.asarray(model.low_q_stiffness_max_shell, dtype=np.int64)
    if hasattr(model, "low_q_stiffness_epsilon"):
        output["low_q_stiffness_epsilon"] = np.asarray(model.low_q_stiffness_epsilon, dtype=np.float32)

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
