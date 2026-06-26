"""Fine-tune a crystal model through differentiable blockwise rollout.

This script is intentionally separate from ``infer_model.py``.  It loads a
model trained on small crystal blocks, applies the same blockwise stitching used
for large-crystal inference, and optimizes the stitched prediction against a
larger reference trajectory.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from base_classes.crystal_predictor import (
    DEFAULT_FLATTEN_ORDER,
    ORDER_AXIS_TO_DIM,
    _as_shape3,
    _build_merge_blocks,
    _build_supercell_origins,
    _normalize_delta_loss_epsilon,
    _normalize_delta_loss_weight,
    _normalize_flatten_order,
    _normalize_merge_mode,
    _normalize_target_mode,
)


def parse_args():
    """Parse fine-tuning command-line options."""
    parser = argparse.ArgumentParser(description="Fine-tune a saved model through blockwise crystal rollout.")
    parser.add_argument("--model-path", required=True, help="Path to a saved .pth model.")
    parser.add_argument("--data-path", required=True, help="Path to a crystal .npz dataset.")
    parser.add_argument("--output-model-path", required=True, help="Where to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Random windows per epoch.")
    parser.add_argument("--rollout-steps", type=int, default=1, help="Differentiable autoregressive steps per sample.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Fine-tuning learning rate.")
    parser.add_argument(
        "--merge-mode",
        choices=["mean", "weighted", "center", "owner", "soft_center"],
        default="owner",
        help="Block stitching mode used during fine-tuning.",
    )
    parser.add_argument("--merge-top-k", type=int, default=None, help="Top-k candidates for soft_center.")
    parser.add_argument("--merge-alpha", type=float, default=1.0, help="Distance weight for soft_center.")
    parser.add_argument("--periodic", action="store_true", help="Use periodic block wrapping.")
    parser.add_argument(
        "--stride-shape",
        type=int,
        nargs=3,
        default=None,
        help="Block origin stride in unit-cell coordinates. Defaults to (1, 1, 1).",
    )
    parser.add_argument(
        "--delta-loss-weight",
        type=float,
        default=0.0,
        help="Optional full-crystal normalized one-step delta loss weight.",
    )
    parser.add_argument(
        "--delta-loss-epsilon",
        type=float,
        default=1e-6,
        help="Small stabilizer for the optional normalized delta loss.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args()


def torch_flatten_supercell(block, flatten_order):
    """Pack a torch supercell tensor into the model's flat feature order."""
    axes = [ORDER_AXIS_TO_DIM[name] for name in flatten_order]
    return block.permute(*axes).reshape(-1)


def torch_unflatten_supercell(flat_features, train_supercell_shape, unit_cell_atoms, flatten_order):
    """Restore one flat torch vector to canonical crystal block shape."""
    ordered_shape = []
    for name in flatten_order:
        if name == "x":
            ordered_shape.append(train_supercell_shape[0])
        elif name == "y":
            ordered_shape.append(train_supercell_shape[1])
        elif name == "z":
            ordered_shape.append(train_supercell_shape[2])
        elif name == "atom":
            ordered_shape.append(unit_cell_atoms)
        elif name == "coord":
            ordered_shape.append(3)

    ordered = flat_features.reshape(*ordered_shape)
    inverse_axes = np.argsort([ORDER_AXIS_TO_DIM[name] for name in flatten_order]).tolist()
    return ordered.permute(*inverse_axes).reshape(*train_supercell_shape, unit_cell_atoms, 3)


def build_torch_merge_blocks(crystal_shape, train_supercell_shape, stride_shape, periodic, merge_mode, merge_top_k, merge_alpha, device):
    """Build block indices and weights as torch tensors for differentiable rollout."""
    _, base_merge_mode = _normalize_merge_mode(merge_mode)
    origins = _build_supercell_origins(crystal_shape, train_supercell_shape, stride_shape, periodic)
    if not origins:
        raise ValueError("No supercell origins were generated")

    blocks = _build_merge_blocks(
        crystal_shape,
        train_supercell_shape,
        origins,
        periodic,
        base_merge_mode,
        merge_top_k,
        merge_alpha,
    )
    torch_blocks = []
    for _, index, local_weights in blocks:
        torch_index = tuple(torch.as_tensor(axis, dtype=torch.long, device=device) for axis in index)
        torch_weights = torch.as_tensor(local_weights, dtype=torch.float32, device=device)
        torch_blocks.append((torch_index, torch_weights))
    return torch_blocks


def differentiable_blockwise_step(
    torch_model,
    history,
    merge_blocks,
    train_supercell_shape,
    unit_cell_atoms,
    flatten_order,
    merge_mode,
    target_mode,
):
    """Run one differentiable blockwise prediction step."""
    crystal_shape = tuple(history.shape[1:4])
    merge_deltas, _ = _normalize_merge_mode(merge_mode)
    if target_mode == "delta" and not merge_mode.startswith("delta_"):
        merge_deltas = True

    prediction_sum = torch.zeros_like(history[-1])
    prediction_weight = torch.zeros((*crystal_shape, unit_cell_atoms, 1), dtype=history.dtype, device=history.device)

    for index, local_weights in merge_blocks:
        block_x = history[(slice(None), *index, slice(None), slice(None))]
        block_x_flat = torch.stack([torch_flatten_supercell(frame, flatten_order) for frame in block_x], dim=0)
        block_y_flat = torch_model(block_x_flat.reshape(1, history.shape[0], -1)).squeeze(0)
        block_y = torch_unflatten_supercell(block_y_flat, train_supercell_shape, unit_cell_atoms, flatten_order)
        if target_mode != "delta" and merge_deltas:
            block_y = block_y - block_x[-1]

        prediction_sum[(index[0], index[1], index[2], slice(None), slice(None))] += block_y * local_weights
        prediction_weight[(index[0], index[1], index[2], slice(None), slice(None))] += local_weights

    if torch.any(prediction_weight == 0):
        raise ValueError("Some crystal cells were not covered by any inference block")

    y = prediction_sum / prediction_weight
    if merge_deltas:
        y = history[-1] + y
    return y


def normalized_delta_loss(predicted, target, last_input, epsilon):
    """Relative one-step delta loss over a full crystal frame."""
    pred_delta = predicted - last_input
    true_delta = target - last_input
    scale = torch.sqrt(torch.mean(true_delta**2)).clamp_min(epsilon)
    return torch.mean((pred_delta / scale - true_delta / scale) ** 2)


def load_training_displacements(path):
    """Load the full crystal displacement trajectory from an ``.npz`` file."""
    data = np.load(path)
    if "displacements" not in data.files or "X_blocks" not in data.files:
        raise ValueError("Dataset must contain 'displacements' and 'X_blocks'")
    return data["displacements"].astype(np.float32), int(data["X_blocks"].shape[1])


def main():
    """Fine-tune the model and save the updated checkpoint."""
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    displacements, sequence_length = load_training_displacements(args.data_path)
    max_start = displacements.shape[0] - sequence_length - args.rollout_steps
    if max_start < 0:
        raise ValueError("Dataset is too short for sequence_length + rollout_steps")

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.model.train()

    train_supercell_shape = _as_shape3("train_supercell_shape", model.train_supercell_shape)
    unit_cell_atoms = int(model.unit_cell_atoms)
    flatten_order = _normalize_flatten_order(getattr(model, "flatten_order", DEFAULT_FLATTEN_ORDER))
    target_mode = _normalize_target_mode(getattr(model, "target_mode", "absolute"))
    stride_shape = (1, 1, 1) if args.stride_shape is None else _as_shape3("stride_shape", args.stride_shape)
    crystal_shape = tuple(int(dim) for dim in displacements.shape[1:4])

    device = torch.device("cpu")
    merge_blocks = build_torch_merge_blocks(
        crystal_shape,
        train_supercell_shape,
        stride_shape,
        args.periodic,
        args.merge_mode,
        args.merge_top_k,
        args.merge_alpha,
        device,
    )

    tensor_displacements = torch.as_tensor(displacements, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.model.parameters(), lr=args.learning_rate)
    mse = nn.MSELoss()
    delta_loss_weight = _normalize_delta_loss_weight(args.delta_loss_weight)
    delta_loss_epsilon = _normalize_delta_loss_epsilon(args.delta_loss_epsilon)

    print("FINETUNE BLOCKWISE")
    print("data", args.data_path, "frames", displacements.shape[0])
    print("model", args.model_path)
    print("merge_mode", args.merge_mode, "rollout_steps", args.rollout_steps)
    print("blocks", len(merge_blocks), "crystal_shape", crystal_shape)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            start = np.random.randint(0, max_start + 1)
            history = tensor_displacements[start : start + sequence_length].clone()
            loss = torch.zeros((), dtype=torch.float32, device=device)

            for step in range(args.rollout_steps):
                target = tensor_displacements[start + sequence_length + step]
                predicted = differentiable_blockwise_step(
                    model.model,
                    history,
                    merge_blocks,
                    train_supercell_shape,
                    unit_cell_atoms,
                    flatten_order,
                    args.merge_mode,
                    target_mode,
                )
                loss = loss + mse(predicted, target)
                if delta_loss_weight > 0:
                    loss = loss + delta_loss_weight * normalized_delta_loss(
                        predicted,
                        target,
                        history[-1],
                        delta_loss_epsilon,
                    )
                history = torch.cat([history[1:], predicted.unsqueeze(0)], dim=0)

            loss = loss / args.rollout_steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        epoch_loss /= args.steps_per_epoch
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.8f}")

    model.blockwise_finetune = {
        "data_path": str(args.data_path),
        "merge_mode": args.merge_mode,
        "rollout_steps": args.rollout_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "delta_loss_weight": args.delta_loss_weight,
        "delta_loss_epsilon": args.delta_loss_epsilon,
    }
    output_path = Path(args.output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
