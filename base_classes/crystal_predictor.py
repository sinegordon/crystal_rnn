import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from .datasets import RNNCustomDataset
from .models import FrameLayerRNNNet, RNNNet


DEFAULT_FLATTEN_ORDER = ("x", "y", "z", "atom", "coord")
"""Default order for packing a supercell tensor into the flat RNN input."""
ORDER_AXIS_TO_DIM = {
    "x": 0,
    "y": 1,
    "z": 2,
    "atom": 3,
    "coord": 4,
}


def _as_shape3(name, value):
    if len(value) != 3:
        raise ValueError(f"{name} must contain three dimensions")
    shape = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} dimensions must be positive")
    return shape


def _normalize_flatten_order(flatten_order):
    flatten_order = tuple(flatten_order)
    if set(flatten_order) != set(DEFAULT_FLATTEN_ORDER) or len(flatten_order) != len(DEFAULT_FLATTEN_ORDER):
        raise ValueError("flatten_order must be a permutation of ('x', 'y', 'z', 'atom', 'coord')")
    return flatten_order


def _normalize_target_mode(target_mode):
    """Validate the model target convention."""
    target_mode = str(target_mode)
    if target_mode not in {"absolute", "delta", "absolute_delta", "acceleration", "verlet"}:
        raise ValueError("target_mode must be 'absolute', 'delta', 'absolute_delta', 'acceleration', or 'verlet'")
    return target_mode


def _normalize_temporal_architecture(temporal_architecture):
    """Validate the flat temporal model architecture."""
    temporal_architecture = str(temporal_architecture).lower().replace("_", "-")
    aliases = {
        "stacked": "stacked",
        "standard": "stacked",
        "pytorch": "stacked",
        "frame-layered": "frame-layered",
        "frame": "frame-layered",
        "time-layered": "frame-layered",
    }
    if temporal_architecture not in aliases:
        raise ValueError("temporal_architecture must be 'stacked' or 'frame-layered'")
    return aliases[temporal_architecture]


def _normalize_delta_loss_weight(delta_loss_weight):
    """Validate the auxiliary delta-loss weight."""
    delta_loss_weight = float(delta_loss_weight)
    if delta_loss_weight < 0:
        raise ValueError("delta_loss_weight must be non-negative")
    return delta_loss_weight


def _normalize_delta_loss_epsilon(delta_loss_epsilon):
    """Validate the small stabilizer used by the normalized delta loss."""
    delta_loss_epsilon = float(delta_loss_epsilon)
    if delta_loss_epsilon <= 0:
        raise ValueError("delta_loss_epsilon must be positive")
    return delta_loss_epsilon


def _normalize_acceleration_loss_weight(acceleration_loss_weight):
    """Validate the auxiliary acceleration-loss weight."""
    acceleration_loss_weight = float(acceleration_loss_weight)
    if acceleration_loss_weight < 0:
        raise ValueError("acceleration_loss_weight must be non-negative")
    return acceleration_loss_weight


def _normalize_acceleration_loss_epsilon(acceleration_loss_epsilon):
    """Validate the small stabilizer used by the normalized acceleration loss."""
    acceleration_loss_epsilon = float(acceleration_loss_epsilon)
    if acceleration_loss_epsilon <= 0:
        raise ValueError("acceleration_loss_epsilon must be positive")
    return acceleration_loss_epsilon


def _normalize_loss_weight_mode(loss_weight_mode):
    """Validate the feature weighting mode used during block training."""
    loss_weight_mode = str(loss_weight_mode)
    if loss_weight_mode not in {"uniform", "center", "soft_center"}:
        raise ValueError("loss_weight_mode must be 'uniform', 'center', or 'soft_center'")
    return loss_weight_mode


def _normalize_center_loss_weight(center_loss_weight):
    """Validate the multiplier applied to central block cells."""
    center_loss_weight = float(center_loss_weight)
    if center_loss_weight <= 0:
        raise ValueError("center_loss_weight must be positive")
    return center_loss_weight


def _normalize_center_loss_alpha(center_loss_alpha):
    """Validate the soft-center distance decay."""
    center_loss_alpha = float(center_loss_alpha)
    if center_loss_alpha <= 0:
        raise ValueError("center_loss_alpha must be positive")
    return center_loss_alpha


def _flatten_supercell(block, flatten_order):
    """Pack a `(bx, by, bz, unit_cell_atoms, 3)` supercell into a flat vector."""
    axes = [ORDER_AXIS_TO_DIM[name] for name in flatten_order]
    return np.transpose(block, axes=axes).reshape(-1)


def _unflatten_supercell(flat_features, train_supercell_shape, unit_cell_atoms, flatten_order):
    """Restore a flat model output to `(bx, by, bz, unit_cell_atoms, 3)`."""
    ordered_shape = []
    canonical_shape = (*train_supercell_shape, unit_cell_atoms, 3)
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
    inverse_axes = np.argsort([ORDER_AXIS_TO_DIM[name] for name in flatten_order])
    return np.transpose(ordered, axes=inverse_axes).reshape(canonical_shape)


def _build_supercell_origins(crystal_shape, train_supercell_shape, stride_shape, periodic):
    if periodic:
        return [
            (ix, iy, iz)
            for ix in range(0, crystal_shape[0], stride_shape[0])
            for iy in range(0, crystal_shape[1], stride_shape[1])
            for iz in range(0, crystal_shape[2], stride_shape[2])
        ]

    for crystal_dim, block_dim in zip(crystal_shape, train_supercell_shape):
        if crystal_dim < block_dim:
            raise ValueError("train_supercell_shape must fit inside crystal_shape when periodic=False")

    return [
        (ix, iy, iz)
        for ix in range(0, crystal_shape[0] - train_supercell_shape[0] + 1, stride_shape[0])
        for iy in range(0, crystal_shape[1] - train_supercell_shape[1] + 1, stride_shape[1])
        for iz in range(0, crystal_shape[2] - train_supercell_shape[2] + 1, stride_shape[2])
    ]


def _order_owner_origins(origins, owner_order=None, owner_reverse=None):
    """Order block origins for deterministic owner-style merge modes."""
    if owner_order is None:
        return origins
    axis_map = {"x": 0, "y": 1, "z": 2}
    owner_order = tuple(str(axis) for axis in owner_order)
    if set(owner_order) != {"x", "y", "z"} or len(owner_order) != 3:
        raise ValueError("owner_order must be a permutation of x, y, z")
    owner_reverse = set() if owner_reverse is None else set(owner_reverse)
    if not owner_reverse <= {"x", "y", "z"}:
        raise ValueError("owner_reverse can only contain x, y, z")

    def sort_key(origin):
        key = []
        for axis_name in owner_order:
            value = origin[axis_map[axis_name]]
            key.append(-value if axis_name in owner_reverse else value)
        return tuple(key)

    return sorted(origins, key=sort_key)


def _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic):
    axes = []
    for start, block_dim, crystal_dim in zip(origin, train_supercell_shape, crystal_shape):
        indices = np.arange(start, start + block_dim)
        if periodic:
            indices = indices % crystal_dim
        axes.append(indices)
    return np.ix_(axes[0], axes[1], axes[2])


def _local_merge_weights(train_supercell_shape, merge_mode):
    """Build local block weights for merge modes that average overlaps."""
    if merge_mode == "mean":
        return np.ones((*train_supercell_shape, 1, 1), dtype=np.float32)
    if merge_mode != "weighted":
        raise ValueError(f"Unsupported local merge mode: {merge_mode}")

    weights = np.ones(train_supercell_shape, dtype=np.float32)
    for axis, size in enumerate(train_supercell_shape):
        center = (size - 1) / 2
        axis_weights = 1 / (1 + np.abs(np.arange(size, dtype=np.float32) - center))
        shape = [1, 1, 1]
        shape[axis] = size
        weights *= axis_weights.reshape(shape)
    return weights.reshape(*train_supercell_shape, 1, 1)


def _normalize_merge_mode(merge_mode):
    """Return whether to merge deltas and the base merge mode."""
    if merge_mode.startswith("delta_"):
        return True, merge_mode.removeprefix("delta_")
    return False, merge_mode


def _random_owner_mode(merge_mode):
    """Return whether merge mode randomly assigns block owners."""
    if merge_mode.startswith("fixed_random_owner"):
        return "fixed"
    if merge_mode.startswith("random_owner"):
        return "step"
    return None


def _crystal_index_from_ix(index, local_index):
    """Return the physical crystal cell covered by a local block index."""
    return tuple(
        index[axis][local_index[axis], 0, 0]
        if axis == 0
        else index[axis][0, local_index[axis], 0]
        if axis == 1
        else index[axis][0, 0, local_index[axis]]
        for axis in range(3)
    )


def _build_merge_blocks(crystal_shape, train_supercell_shape, origins, periodic, merge_mode, merge_top_k=None, merge_alpha=1.0):
    """Precompute block indices and per-cell merge weights."""
    _, merge_mode = _normalize_merge_mode(merge_mode)
    if merge_mode not in {"mean", "weighted", "center", "owner", "soft_center", "fixed_random_owner", "random_owner"}:
        raise ValueError(
            "merge_mode must be one of: mean, weighted, center, owner, fixed_random_owner, "
            "random_owner, soft_center, robust_center, robust_mean, delta_mean, "
            "delta_weighted, delta_center, delta_owner, delta_soft_center"
        )
    if merge_top_k is not None and merge_top_k <= 0:
        raise ValueError("merge_top_k must be positive")
    if merge_alpha <= 0:
        raise ValueError("merge_alpha must be positive")

    blocks = []
    if merge_mode in {"mean", "weighted"}:
        local_weights = _local_merge_weights(train_supercell_shape, merge_mode)
        for origin in origins:
            blocks.append((origin, _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic), local_weights))
        return blocks

    candidates = {}
    best_score = {}
    random_owner = _random_owner_mode(merge_mode)
    center = (np.asarray(train_supercell_shape, dtype=np.float32) - 1) / 2
    for origin_index, origin in enumerate(origins):
        index = _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic)
        for local_index in np.ndindex(train_supercell_shape):
            crystal_index = _crystal_index_from_ix(index, local_index)
            distance = float(np.sum(np.abs(np.asarray(local_index, dtype=np.float32) - center)))
            candidates.setdefault(crystal_index, []).append((distance, origin_index, local_index))
            if merge_mode == "soft_center" or random_owner is not None:
                continue
            if merge_mode == "owner":
                score = origin_index
            else:
                score = distance
            if crystal_index not in best_score or score < best_score[crystal_index]:
                best_score[crystal_index] = score

    local_weights_by_origin = [np.zeros((*train_supercell_shape, 1, 1), dtype=np.float32) for _ in origins]
    if merge_mode == "soft_center":
        for cell_candidates in candidates.values():
            cell_candidates = sorted(cell_candidates, key=lambda item: item[0])
            selected = cell_candidates if merge_top_k is None else cell_candidates[:merge_top_k]
            for distance, origin_index, local_index in selected:
                local_weights_by_origin[origin_index][(*local_index, 0, 0)] = np.exp(-merge_alpha * distance)
    elif random_owner is not None:
        rng = np.random.default_rng(None if merge_top_k is None else int(merge_top_k))
        for cell_candidates in candidates.values():
            _, origin_index, local_index = cell_candidates[int(rng.integers(0, len(cell_candidates)))]
            local_weights_by_origin[origin_index][(*local_index, 0, 0)] = 1.0
    else:
        for crystal_index, score in best_score.items():
            for distance, origin_index, local_index in candidates[crystal_index]:
                candidate_score = origin_index if merge_mode == "owner" else distance
                if candidate_score == score:
                    local_weights_by_origin[origin_index][(*local_index, 0, 0)] = 1.0
                    break

    for origin_index, origin in enumerate(origins):
        blocks.append(
            (
                origin,
                _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic),
                local_weights_by_origin[origin_index],
            )
        )
    return blocks


def _merge_robust_candidates(candidates, crystal_shape, unit_cell_atoms, dtype, device, merge_mode, merge_top_k, merge_alpha):
    """Merge per-cell block candidates using median distance and centrality."""
    y = torch.zeros((*crystal_shape, unit_cell_atoms, 3), dtype=dtype, device=device)
    if len(candidates) != int(np.prod(crystal_shape)):
        raise ValueError("Some crystal cells were not covered by any inference block")

    for crystal_index, cell_candidates in candidates.items():
        distances = torch.as_tensor(
            [candidate[0] for candidate in cell_candidates],
            dtype=dtype,
            device=device,
        )
        values = torch.stack([candidate[1] for candidate in cell_candidates], dim=0)
        median = torch.median(values, dim=0).values
        disagreement = torch.sqrt(torch.mean((values - median.unsqueeze(0)) ** 2, dim=(1, 2)))
        scores = disagreement + merge_alpha * distances

        if merge_mode == "robust_center":
            selected = torch.argmin(scores).reshape(1)
        else:
            top_k = merge_top_k if merge_top_k is not None else min(3, values.shape[0])
            top_k = min(int(top_k), values.shape[0])
            selected = torch.topk(scores, k=top_k, largest=False).indices
        y[crystal_index] = values[selected].mean(dim=0)

    return y


def _validate_crystal_input(init_displacements, crystal_shape, unit_cell_atoms):
    expected_ndim = 6
    if init_displacements.ndim != expected_ndim:
        raise ValueError(
            "init_displacements must have shape "
            "(sequence_length, nx, ny, nz, unit_cell_atoms, 3)"
        )
    if tuple(init_displacements.shape[1:4]) != crystal_shape:
        raise ValueError("init_displacements shape does not match crystal_shape")
    if init_displacements.shape[4] != unit_cell_atoms:
        raise ValueError("init_displacements shape does not match unit_cell_atoms")
    if init_displacements.shape[5] != 3:
        raise ValueError("The last init_displacements dimension must be 3")


def _run_model_on_crystal(
    model,
    count_steps,
    init_displacements,
    train_supercell_shape,
    unit_cell_atoms,
    stride_shape,
    periodic,
    flatten_order,
    merge_mode="mean",
    merge_top_k=None,
    merge_alpha=1.0,
    target_mode="absolute",
    owner_order=None,
    owner_reverse=None,
):
    """Run one PyTorch model over a full crystal displacement field."""
    crystal_shape = tuple(init_displacements.shape[1:4])
    target_mode = _normalize_target_mode(target_mode)
    if target_mode == "acceleration" and init_displacements.shape[0] < 2:
        raise ValueError("target_mode='acceleration' requires at least two input history frames")
    merge_deltas, base_merge_mode = _normalize_merge_mode(merge_mode)
    if target_mode == "delta" and not merge_mode.startswith("delta_"):
        merge_deltas = True
    robust_merge = base_merge_mode in {"robust_center", "robust_mean"}
    random_owner = _random_owner_mode(base_merge_mode)
    if robust_merge and merge_top_k is not None and merge_top_k <= 0:
        raise ValueError("merge_top_k must be positive")
    if robust_merge and merge_alpha < 0:
        raise ValueError("merge_alpha must be non-negative for robust merge modes")
    origins = _build_supercell_origins(crystal_shape, train_supercell_shape, stride_shape, periodic)
    if base_merge_mode in {"owner", "fixed_random_owner", "random_owner"}:
        origins = _order_owner_origins(origins, owner_order=owner_order, owner_reverse=owner_reverse)
    if not origins:
        raise ValueError("No supercell origins were generated")
    if robust_merge:
        center = (np.asarray(train_supercell_shape, dtype=np.float32) - 1) / 2
        merge_blocks = [
            (origin, _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic))
            for origin in origins
        ]
    elif random_owner == "step":
        merge_blocks = None
    else:
        merge_blocks = _build_merge_blocks(
            crystal_shape,
            train_supercell_shape,
            origins,
            periodic,
            base_merge_mode,
            merge_top_k,
            merge_alpha,
        )

    model.eval()
    x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
    predictions = []

    with torch.no_grad():
        for step_index in range(count_steps):
            if random_owner == "step":
                step_seed = None if merge_top_k is None else int(merge_top_k) + step_index
                merge_blocks = _build_merge_blocks(
                    crystal_shape,
                    train_supercell_shape,
                    origins,
                    periodic,
                    base_merge_mode,
                    step_seed,
                    merge_alpha,
                )
            if robust_merge:
                candidates = {}
            else:
                prediction_sum = torch.zeros_like(x[-1])
                prediction_weight = torch.zeros((*crystal_shape, unit_cell_atoms, 1), dtype=x.dtype, device=x.device)

            for block_item in merge_blocks:
                if robust_merge:
                    _, index = block_item
                else:
                    _, index, local_weights = block_item
                block_x = x[(slice(None), *index, slice(None), slice(None))]
                # The RNN still sees the same flat input format it was trained on.
                block_x_flat = torch.stack(
                    [
                        torch.as_tensor(_flatten_supercell(frame.detach().cpu().numpy(), flatten_order), dtype=x.dtype)
                        for frame in block_x
                    ],
                    dim=0,
                ).to(x.device)
                block_y_flat = model(block_x_flat.reshape(1, x.shape[0], -1)).squeeze(0)
                block_y = torch.as_tensor(
                    _unflatten_supercell(
                        block_y_flat.detach().cpu().numpy(),
                        train_supercell_shape,
                        unit_cell_atoms,
                        flatten_order,
                    ),
                    dtype=x.dtype,
                    device=x.device,
                )
                if target_mode in {"acceleration", "verlet"}:
                    block_y = 2 * block_x[-1] - block_x[-2] + block_y
                if target_mode != "delta" and merge_deltas:
                    block_y = block_y - block_x[-1]

                if robust_merge:
                    for local_index in np.ndindex(train_supercell_shape):
                        crystal_index = _crystal_index_from_ix(index, local_index)
                        distance = float(np.sum(np.abs(np.asarray(local_index, dtype=np.float32) - center)))
                        candidates.setdefault(crystal_index, []).append((distance, block_y[local_index]))
                else:
                    weights = torch.as_tensor(local_weights, dtype=x.dtype, device=x.device)
                    prediction_sum[index] += block_y * weights
                    prediction_weight[index] += weights

            if robust_merge:
                y = _merge_robust_candidates(
                    candidates,
                    crystal_shape,
                    unit_cell_atoms,
                    x.dtype,
                    x.device,
                    base_merge_mode,
                    merge_top_k,
                    merge_alpha,
                )
            else:
                if torch.any(prediction_weight == 0):
                    raise ValueError("Some crystal cells were not covered by any inference block")

                # Merge overlapping supercell predictions per physical cell and atom.
                y = prediction_sum / prediction_weight
            if merge_deltas:
                y = x[-1] + y
            predictions.append(y.detach().cpu().numpy())
            x[:-1] = x[1:].clone()
            x[-1] = y

    return np.array(predictions)


def _flatten_crystal_block_samples(blocks, flatten_order):
    """Flatten crystal block histories for crystal-aware training."""
    return np.array(
        [
            [_flatten_supercell(frame, flatten_order) for frame in sample]
            for sample in blocks
        ],
        dtype=np.float32,
    )


def _flatten_crystal_block_targets(blocks, flatten_order):
    """Flatten one-step crystal block targets for crystal-aware training."""
    return np.array([_flatten_supercell(block, flatten_order) for block in blocks], dtype=np.float32)


def _build_supercell_loss_weights(
    train_supercell_shape,
    unit_cell_atoms,
    flatten_order,
    loss_weight_mode,
    center_loss_weight,
    center_loss_alpha,
):
    """Build per-feature loss weights for a crystal supercell output."""
    loss_weight_mode = _normalize_loss_weight_mode(loss_weight_mode)
    center_loss_weight = _normalize_center_loss_weight(center_loss_weight)
    center_loss_alpha = _normalize_center_loss_alpha(center_loss_alpha)
    weights = np.ones((*train_supercell_shape, unit_cell_atoms, 3), dtype=np.float32)
    if loss_weight_mode == "uniform" or center_loss_weight == 1.0:
        return weights

    center = (np.asarray(train_supercell_shape, dtype=np.float32) - 1) / 2
    distance = np.zeros(train_supercell_shape, dtype=np.float32)
    for local_index in np.ndindex(train_supercell_shape):
        distance[local_index] = float(np.sum(np.abs(np.asarray(local_index, dtype=np.float32) - center)))

    if loss_weight_mode == "center":
        weights[distance == np.min(distance)] = center_loss_weight
    else:
        cell_weights = 1 + (center_loss_weight - 1) * np.exp(-center_loss_alpha * distance)
        weights *= cell_weights.reshape(*train_supercell_shape, 1, 1)
    return weights


def _weighted_mse_loss(predicted, target, weights=None):
    """Return MSE with optional per-feature weights and stable normalization."""
    squared_error = (predicted - target) ** 2
    if weights is None:
        return torch.mean(squared_error)

    weights = weights.to(device=predicted.device, dtype=predicted.dtype)
    while weights.ndim < squared_error.ndim:
        weights = weights.unsqueeze(0)
    repeat_factor = squared_error.numel() / weights.numel()
    return torch.sum(squared_error * weights) / (torch.sum(weights) * repeat_factor)


class CrystalRNNNet:
    """Train and run an RNN predictor for flat or crystal-structured displacement data.

    A flat model is created by passing `in_features`.
    A crystal-aware model is created by passing `train_supercell_shape` and
    `unit_cell_atoms`; in that case `in_features` is derived from geometry.
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        in_features=None,
        type="RNN",
        train_supercell_shape=None,
        unit_cell_atoms=None,
        flatten_order=DEFAULT_FLATTEN_ORDER,
        target_mode="absolute",
        delta_loss_weight=1.0,
        delta_loss_epsilon=1e-6,
        acceleration_loss_weight=0.0,
        acceleration_loss_epsilon=1e-8,
        loss_weight_mode="uniform",
        center_loss_weight=1.0,
        center_loss_alpha=1.0,
        temporal_architecture="stacked",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = type.upper()
        self.temporal_architecture = _normalize_temporal_architecture(temporal_architecture)
        self.train_supercell_shape = None if train_supercell_shape is None else _as_shape3(
            "train_supercell_shape",
            train_supercell_shape,
        )
        self.unit_cell_atoms = None if unit_cell_atoms is None else int(unit_cell_atoms)
        self.flatten_order = _normalize_flatten_order(flatten_order)
        self.target_mode = _normalize_target_mode(target_mode)
        self.delta_loss_weight = _normalize_delta_loss_weight(delta_loss_weight)
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(delta_loss_epsilon)
        self.acceleration_loss_weight = _normalize_acceleration_loss_weight(acceleration_loss_weight)
        self.acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(acceleration_loss_epsilon)
        self.loss_weight_mode = _normalize_loss_weight_mode(loss_weight_mode)
        self.center_loss_weight = _normalize_center_loss_weight(center_loss_weight)
        self.center_loss_alpha = _normalize_center_loss_alpha(center_loss_alpha)
        self.in_features = self._resolve_in_features(in_features)
        self._validate_crystal_metadata()
        self.model = self._build_model()
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 200
        self.train_count = 200

    def reset(self):
        self.target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))
        self.delta_loss_weight = _normalize_delta_loss_weight(getattr(self, "delta_loss_weight", 1.0))
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(getattr(self, "delta_loss_epsilon", 1e-6))
        self.acceleration_loss_weight = _normalize_acceleration_loss_weight(
            getattr(self, "acceleration_loss_weight", 0.0)
        )
        self.acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(
            getattr(self, "acceleration_loss_epsilon", 1e-8)
        )
        self.loss_weight_mode = _normalize_loss_weight_mode(getattr(self, "loss_weight_mode", "uniform"))
        self.center_loss_weight = _normalize_center_loss_weight(getattr(self, "center_loss_weight", 1.0))
        self.center_loss_alpha = _normalize_center_loss_alpha(getattr(self, "center_loss_alpha", 1.0))
        self.temporal_architecture = _normalize_temporal_architecture(
            getattr(self, "temporal_architecture", "stacked")
        )
        self.model = self._build_model()

    def _build_model(self):
        """Create the underlying flat temporal network."""
        architecture = _normalize_temporal_architecture(getattr(self, "temporal_architecture", "stacked"))
        if architecture == "frame-layered":
            return FrameLayerRNNNet(self.in_features, self.hidden_size, self.num_layers, type=self.rnn_type)
        return RNNNet(self.in_features, self.hidden_size, self.num_layers, type=self.rnn_type)

    def _resolve_in_features(self, in_features):
        """Resolve the flat model input size from explicit features or crystal metadata."""
        if self.train_supercell_shape is None and self.unit_cell_atoms is None:
            if in_features is None:
                raise ValueError("in_features is required for models without crystal metadata")
            return int(in_features)
        if self.train_supercell_shape is None or self.unit_cell_atoms is None:
            raise ValueError("train_supercell_shape and unit_cell_atoms must be provided together")

        resolved_features = int(np.prod(self.train_supercell_shape) * self.unit_cell_atoms * 3)
        if in_features is not None and int(in_features) != resolved_features:
            raise ValueError("in_features does not match train_supercell_shape and unit_cell_atoms")
        return resolved_features

    def _validate_crystal_metadata(self):
        if self.train_supercell_shape is None and self.unit_cell_atoms is None:
            return
        if self.train_supercell_shape is None or self.unit_cell_atoms is None:
            raise ValueError("train_supercell_shape and unit_cell_atoms must be provided together")
        if self.unit_cell_atoms <= 0:
            raise ValueError("unit_cell_atoms must be positive")

        expected_features = int(np.prod(self.train_supercell_shape) * self.unit_cell_atoms * 3)
        if self.in_features != expected_features:
            raise ValueError("Model input size does not match crystal metadata")

    @property
    def is_crystal_aware(self):
        """Whether this predictor has enough metadata for crystal-shaped IO."""
        return self.train_supercell_shape is not None and self.unit_cell_atoms is not None

    def loss_weights_supercell(self):
        """Return crystal-shaped output weights for weighted training losses."""
        if not self.is_crystal_aware:
            return None
        return _build_supercell_loss_weights(
            self.train_supercell_shape,
            self.unit_cell_atoms,
            self.flatten_order,
            getattr(self, "loss_weight_mode", "uniform"),
            getattr(self, "center_loss_weight", 1.0),
            getattr(self, "center_loss_alpha", 1.0),
        )

    def loss_weights_flat(self):
        """Return flattened output weights matching the RNN output order."""
        weights = self.loss_weights_supercell()
        if weights is None:
            return None
        return _flatten_supercell(weights, self.flatten_order)

    def flatten_supercell(self, supercell_displacements):
        """Pack one training supercell according to this model's `flatten_order`.

        Args:
            supercell_displacements: Array with shape
                `(bx, by, bz, unit_cell_atoms, 3)`.

        Returns:
            Flat vector with length `self.in_features`.
        """
        if not self.is_crystal_aware:
            raise ValueError("Crystal metadata is required to flatten supercell displacements")
        return _flatten_supercell(
            np.asarray(supercell_displacements, dtype=np.float32),
            self.flatten_order,
        )

    def unflatten_supercell(self, flat_features):
        """Restore one flat model vector to this model's supercell shape.

        Args:
            flat_features: Flat vector with length `self.in_features`.

        Returns:
            Array with shape `(bx, by, bz, unit_cell_atoms, 3)`.
        """
        if not self.is_crystal_aware:
            raise ValueError("Crystal metadata is required to unflatten supercell displacements")
        return _unflatten_supercell(
            np.asarray(flat_features, dtype=np.float32),
            self.train_supercell_shape,
            self.unit_cell_atoms,
            self.flatten_order,
        )

    def train(self, X_coords, y_coords, data_len=0.5):
        """Train the underlying RNN from already flattened samples."""
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))
        delta_loss_weight = _normalize_delta_loss_weight(getattr(self, "delta_loss_weight", 1.0))
        delta_loss_epsilon = _normalize_delta_loss_epsilon(getattr(self, "delta_loss_epsilon", 1e-6))
        acceleration_loss_weight = _normalize_acceleration_loss_weight(
            getattr(self, "acceleration_loss_weight", 0.0)
        )
        acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(
            getattr(self, "acceleration_loss_epsilon", 1e-8)
        )
        if acceleration_loss_weight > 0 and X_coords.shape[1] < 2:
            raise ValueError("acceleration loss requires at least two input history frames")
        self.train_count = int(data_len * X_coords.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_coords.shape[0]:
            ind = 0
        else:
            ind = np.random.randint(low=0, high=X_coords.shape[0] - self.train_count)

        X_train = X_coords[ind : ind + self.train_count]
        y_train = y_coords[ind : ind + self.train_count]
        train_dataset = RNNCustomDataset(X_train, y_train)
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        loss_weights = None
        if self.is_crystal_aware:
            loss_weights = torch.as_tensor(self.loss_weights_flat(), dtype=torch.float32)
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0
            lm_count = 0

            for x_train, y_train in train_data:
                predict = self.model(x_train)
                if target_mode == "verlet":
                    previous_input = x_train[:, -2, :]
                    last_input = x_train[:, -1, :]
                    train_prediction = 2 * last_input - previous_input + predict
                else:
                    train_prediction = predict
                loss = _weighted_mse_loss(train_prediction, y_train, loss_weights)
                if target_mode == "absolute_delta" and delta_loss_weight > 0:
                    last_input = x_train[:, -1, :]
                    true_delta = y_train - last_input
                    pred_delta = train_prediction - last_input
                    delta_scale = torch.sqrt(torch.mean(true_delta**2, dim=1, keepdim=True)).clamp_min(delta_loss_epsilon)
                    loss = loss + delta_loss_weight * _weighted_mse_loss(
                        pred_delta / delta_scale,
                        true_delta / delta_scale,
                        loss_weights,
                    )
                if target_mode == "verlet" and delta_loss_weight > 0:
                    last_input = x_train[:, -1, :]
                    true_delta = y_train - last_input
                    pred_delta = train_prediction - last_input
                    delta_scale = torch.sqrt(torch.mean(true_delta**2, dim=1, keepdim=True)).clamp_min(delta_loss_epsilon)
                    loss = loss + delta_loss_weight * _weighted_mse_loss(
                        pred_delta / delta_scale,
                        true_delta / delta_scale,
                        loss_weights,
                    )
                if acceleration_loss_weight > 0:
                    previous_input = x_train[:, -2, :]
                    last_input = x_train[:, -1, :]
                    if target_mode in {"acceleration", "verlet"}:
                        pred_acceleration = predict
                        true_acceleration = y_train if target_mode == "acceleration" else y_train - 2 * last_input + previous_input
                    elif target_mode == "delta":
                        pred_next = last_input + predict
                        true_next = last_input + y_train
                        pred_acceleration = pred_next - 2 * last_input + previous_input
                        true_acceleration = true_next - 2 * last_input + previous_input
                    else:
                        pred_next = predict
                        true_next = y_train
                        pred_acceleration = pred_next - 2 * last_input + previous_input
                        true_acceleration = true_next - 2 * last_input + previous_input
                    acceleration_scale = torch.sqrt(
                        torch.mean(true_acceleration**2, dim=1, keepdim=True)
                    ).clamp_min(acceleration_loss_epsilon)
                    loss = loss + acceleration_loss_weight * _weighted_mse_loss(
                        pred_acceleration / acceleration_scale,
                        true_acceleration / acceleration_scale,
                        loss_weights,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

            losses.append(loss_mean)

        return losses

    def train_crystal_blocks(self, X_blocks, y_blocks, data_len=0.5, target_mode=None):
        """Train from crystal-shaped supercell blocks.

        Args:
            X_blocks: Training histories with shape
                `(n_samples, sequence_length, bx, by, bz, unit_cell_atoms, 3)`.
            y_blocks: One-step targets with shape
                `(n_samples, bx, by, bz, unit_cell_atoms, 3)`.
            data_len: Fraction of samples used for one randomized training slice.
            target_mode: Optional override for this model's target convention.
                `absolute` learns next displacement directly; `delta` learns
                `next - last_history_frame`; `absolute_delta` learns absolute
                displacement with an auxiliary normalized delta loss;
                `acceleration` learns `next - 2 * last + previous` and uses a
                Verlet-style reconstruction during inference; `verlet` makes
                the network output acceleration but trains the reconstructed
                next displacement with the usual position/delta losses.

        Returns:
            List of mean training losses, one value per epoch.
        """
        if not self.is_crystal_aware:
            raise ValueError("Crystal metadata is required to train from crystal blocks")

        X_blocks = np.asarray(X_blocks, dtype=np.float32)
        y_blocks = np.asarray(y_blocks, dtype=np.float32)
        if X_blocks.ndim != 7:
            raise ValueError(
                "X_blocks must have shape "
                "(n_samples, sequence_length, bx, by, bz, unit_cell_atoms, 3)"
            )
        if y_blocks.ndim != 6:
            raise ValueError("y_blocks must have shape (n_samples, bx, by, bz, unit_cell_atoms, 3)")
        if tuple(X_blocks.shape[2:5]) != self.train_supercell_shape:
            raise ValueError("X_blocks supercell shape does not match train_supercell_shape")
        if tuple(y_blocks.shape[1:4]) != self.train_supercell_shape:
            raise ValueError("y_blocks supercell shape does not match train_supercell_shape")
        if tuple(X_blocks.shape[5:7]) != (self.unit_cell_atoms, 3):
            raise ValueError("X_blocks atom/coordinate dimensions do not match crystal metadata")
        if tuple(y_blocks.shape[4:6]) != (self.unit_cell_atoms, 3):
            raise ValueError("y_blocks atom/coordinate dimensions do not match crystal metadata")
        if X_blocks.shape[0] != y_blocks.shape[0]:
            raise ValueError("X_blocks and y_blocks must contain the same number of samples")

        if target_mode is not None:
            self.target_mode = _normalize_target_mode(target_mode)
        self.target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))
        if self.target_mode == "delta":
            y_blocks = y_blocks - X_blocks[:, -1]
        elif self.target_mode == "acceleration":
            if X_blocks.shape[1] < 2:
                raise ValueError("target_mode='acceleration' requires at least two input history frames")
            y_blocks = y_blocks - 2 * X_blocks[:, -1] + X_blocks[:, -2]
        elif self.target_mode == "verlet" and X_blocks.shape[1] < 2:
            raise ValueError("target_mode='verlet' requires at least two input history frames")

        X_coords = _flatten_crystal_block_samples(X_blocks, self.flatten_order)
        y_coords = _flatten_crystal_block_targets(y_blocks, self.flatten_order)
        return self.train(X_coords, y_coords, data_len=data_len)

    def run(self, count_steps, init_features):
        """Autoregressively roll out predictions from one flat input history."""
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))
        self.model.eval()
        mas = []
        x = torch.as_tensor(init_features, dtype=torch.float32).clone()

        with torch.no_grad():
            for _ in range(count_steps):
                y = self.model(x).squeeze(0)
                if target_mode == "delta":
                    y = x[0, -1] + y
                elif target_mode in {"acceleration", "verlet"}:
                    y = 2 * x[0, -1] - x[0, -2] + y
                mas.append(y.detach().cpu().numpy())
                x[0] = torch.vstack([x[0, 1:], y])

        mas = np.array(mas)
        return mas

    def run_crystal(
        self,
        count_steps,
        init_displacements,
        train_supercell_shape=None,
        unit_cell_atoms=None,
        stride_shape=None,
        periodic=False,
        merge_mode="mean",
        merge_top_k=None,
        merge_alpha=1.0,
        owner_order=None,
        owner_reverse=None,
    ):
        """Autoregressively roll out predictions for a full crystal.

        Args:
            count_steps: Number of predicted time steps.
            init_displacements: Initial history with shape
                `(sequence_length, nx, ny, nz, unit_cell_atoms, 3)`.
            train_supercell_shape: Optional override for the training supercell shape.
            unit_cell_atoms: Optional override for the number of atoms per unit cell.
            stride_shape: Supercell origin stride in unit-cell coordinates. Defaults
                to `(1, 1, 1)`, giving maximal overlap.
            periodic: Whether supercells may wrap around crystal boundaries.
            merge_mode: How overlapping block predictions are stitched:
                `mean`, `weighted`, `center`, `owner`, `soft_center`,
                `robust_center`, `robust_mean`, or their `delta_*` variants.
            merge_top_k: Number of best-centered block predictions to keep for
                `soft_center` or `robust_mean`.
            merge_alpha: Distance weight for `soft_center` or centrality
                penalty for robust merge modes.
            owner_order: Optional axis priority for owner-style modes.
            owner_reverse: Optional axis names traversed in descending order.

        Returns:
            Array with shape `(count_steps, nx, ny, nz, unit_cell_atoms, 3)`.
        """
        if train_supercell_shape is None:
            if self.train_supercell_shape is None:
                raise ValueError("train_supercell_shape is required for models without crystal metadata")
            train_supercell_shape = self.train_supercell_shape
        else:
            train_supercell_shape = _as_shape3("train_supercell_shape", train_supercell_shape)

        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if unit_cell_atoms is None:
            unit_cell_atoms = self.unit_cell_atoms if self.unit_cell_atoms is not None else int(init_displacements.shape[4])
        crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
        stride_shape = (1, 1, 1) if stride_shape is None else _as_shape3("stride_shape", stride_shape)
        _validate_crystal_input(init_displacements, crystal_shape, unit_cell_atoms)

        expected_features = int(np.prod(train_supercell_shape) * unit_cell_atoms * 3)
        if self.in_features != expected_features:
            raise ValueError("Model input size does not match train_supercell_shape and unit_cell_atoms")
        self.target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))

        return _run_model_on_crystal(
            self.model,
            count_steps,
            init_displacements,
            train_supercell_shape,
            unit_cell_atoms,
            stride_shape,
            periodic,
            self.flatten_order,
            merge_mode=merge_mode,
            merge_top_k=merge_top_k,
            merge_alpha=merge_alpha,
            target_mode=self.target_mode,
            owner_order=owner_order,
            owner_reverse=owner_reverse,
        )


class CrystalRNNNetBagging:
    """Average predictions from several `CrystalRNNNet` predictors."""

    def __init__(self, models, in_features=None):
        if not models and in_features is None:
            raise ValueError("in_features is required when models is empty")
        self.models = [model.model for model in models]
        first_model = models[0] if models else None
        self.in_features = int(in_features) if in_features is not None else first_model.in_features
        self.train_supercell_shape = getattr(first_model, "train_supercell_shape", None)
        self.unit_cell_atoms = getattr(first_model, "unit_cell_atoms", None)
        self.flatten_order = getattr(first_model, "flatten_order", DEFAULT_FLATTEN_ORDER)
        self.target_mode = _normalize_target_mode(getattr(first_model, "target_mode", "absolute"))
        self.delta_loss_weight = _normalize_delta_loss_weight(getattr(first_model, "delta_loss_weight", 1.0))
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(getattr(first_model, "delta_loss_epsilon", 1e-6))
        self.acceleration_loss_weight = _normalize_acceleration_loss_weight(
            getattr(first_model, "acceleration_loss_weight", 0.0)
        )
        self.acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(
            getattr(first_model, "acceleration_loss_epsilon", 1e-8)
        )
        self.loss_weight_mode = _normalize_loss_weight_mode(getattr(first_model, "loss_weight_mode", "uniform"))
        self.center_loss_weight = _normalize_center_loss_weight(getattr(first_model, "center_loss_weight", 1.0))
        self.center_loss_alpha = _normalize_center_loss_alpha(getattr(first_model, "center_loss_alpha", 1.0))

    def run(self, count_steps, init_features, separate=False):
        """Run an ensemble rollout from one flat input history."""
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))
        for model in self.models:
            model.eval()

        mas = []
        if separate:
            for model in self.models:
                x = torch.as_tensor(init_features, dtype=torch.float32).clone()
                mas1 = []
                with torch.no_grad():
                    for _ in range(count_steps):
                        y = model(x).squeeze(0)
                        if target_mode == "delta":
                            y = x[0, -1] + y
                        elif target_mode in {"acceleration", "verlet"}:
                            y = 2 * x[0, -1] - x[0, -2] + y
                        mas1.append(y.detach().cpu().numpy())
                        x[0] = torch.vstack([x[0, 1:], y])
                mas.append(mas1)
            mas = np.array(mas)
            mas = mas.mean(axis=0)
        else:
            x = torch.as_tensor(init_features, dtype=torch.float32).clone()
            with torch.no_grad():
                for _ in range(count_steps):
                    y = torch.zeros(len(self.models), self.in_features, dtype=x.dtype)
                    for i, model in enumerate(self.models):
                        y[i, :] = model(x).squeeze(0)
                    z = torch.mean(y, dim=0)
                    if target_mode == "delta":
                        z = x[0, -1] + z
                    elif target_mode in {"acceleration", "verlet"}:
                        z = 2 * x[0, -1] - x[0, -2] + z
                    mas.append(z.detach().cpu().numpy())
                    x[0] = torch.vstack([x[0, 1:], z])

        mas = np.array(mas)
        return mas

    def run_crystal(
        self,
        count_steps,
        init_displacements,
        train_supercell_shape=None,
        unit_cell_atoms=None,
        stride_shape=None,
        periodic=False,
        separate=False,
        flatten_order=None,
        merge_mode="mean",
        merge_top_k=None,
        merge_alpha=1.0,
    ):
        """Run ensemble inference over a full crystal displacement field."""
        if train_supercell_shape is None:
            if self.train_supercell_shape is None:
                raise ValueError("train_supercell_shape is required for models without crystal metadata")
            train_supercell_shape = self.train_supercell_shape
        if flatten_order is None:
            flatten_order = self.flatten_order
        flatten_order = _normalize_flatten_order(flatten_order)
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute"))
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if unit_cell_atoms is None:
            unit_cell_atoms = self.unit_cell_atoms if self.unit_cell_atoms is not None else int(init_displacements.shape[4])

        expected_features = int(np.prod(_as_shape3("train_supercell_shape", train_supercell_shape)) * unit_cell_atoms * 3)
        if self.in_features != expected_features:
            raise ValueError("Model input size does not match train_supercell_shape and unit_cell_atoms")

        if separate:
            predictions = [
                _run_model_on_crystal(
                    model,
                    count_steps,
                    init_displacements,
                    _as_shape3("train_supercell_shape", train_supercell_shape),
                    unit_cell_atoms,
                    (1, 1, 1) if stride_shape is None else _as_shape3("stride_shape", stride_shape),
                    periodic,
                    flatten_order,
                    merge_mode=merge_mode,
                    merge_top_k=merge_top_k,
                    merge_alpha=merge_alpha,
                    target_mode=target_mode,
                )
                for model in self.models
            ]
            return np.array(predictions).mean(axis=0)

        train_supercell_shape = _as_shape3("train_supercell_shape", train_supercell_shape)
        crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
        stride_shape = (1, 1, 1) if stride_shape is None else _as_shape3("stride_shape", stride_shape)
        _validate_crystal_input(init_displacements, crystal_shape, unit_cell_atoms)

        crystal_shape = tuple(init_displacements.shape[1:4])
        origins = _build_supercell_origins(crystal_shape, train_supercell_shape, stride_shape, periodic)
        if not origins:
            raise ValueError("No supercell origins were generated")
        merge_deltas, base_merge_mode = _normalize_merge_mode(merge_mode)
        if target_mode == "delta" and not merge_mode.startswith("delta_"):
            merge_deltas = True
        merge_blocks = _build_merge_blocks(
            crystal_shape,
            train_supercell_shape,
            origins,
            periodic,
            base_merge_mode,
            merge_top_k,
            merge_alpha,
        )

        for model in self.models:
            model.eval()

        x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
        predictions = []

        with torch.no_grad():
            for _ in range(count_steps):
                model_step_predictions = []
                for model in self.models:
                    prediction_sum = torch.zeros_like(x[-1])
                    prediction_weight = torch.zeros((*crystal_shape, unit_cell_atoms, 1), dtype=x.dtype, device=x.device)

                    for _, index, local_weights in merge_blocks:
                        block_x = x[(slice(None), *index, slice(None), slice(None))]
                        # Match the flat ordering used by the crystal-aware training path.
                        block_x_flat = torch.stack(
                            [
                                torch.as_tensor(
                                    _flatten_supercell(frame.detach().cpu().numpy(), flatten_order),
                                    dtype=x.dtype,
                                )
                                for frame in block_x
                            ],
                            dim=0,
                        ).to(x.device)
                        block_y_flat = model(block_x_flat.reshape(1, x.shape[0], -1)).squeeze(0)
                        block_y = torch.as_tensor(
                            _unflatten_supercell(
                                block_y_flat.detach().cpu().numpy(),
                                train_supercell_shape,
                                unit_cell_atoms,
                                flatten_order,
                            ),
                            dtype=x.dtype,
                            device=x.device,
                        )
                        if target_mode != "delta" and merge_deltas:
                            block_y = block_y - block_x[-1]
                        if target_mode in {"acceleration", "verlet"}:
                            block_y = 2 * block_x[-1] - block_x[-2] + block_y
                        weights = torch.as_tensor(local_weights, dtype=x.dtype, device=x.device)
                        prediction_sum[index] += block_y * weights
                        prediction_weight[index] += weights

                    if torch.any(prediction_weight == 0):
                        raise ValueError("Some crystal cells were not covered by any inference block")

                    # Merge overlaps for this model before ensemble averaging.
                    y_model = prediction_sum / prediction_weight
                    if merge_deltas:
                        y_model = x[-1] + y_model
                    model_step_predictions.append(y_model)

                y = torch.stack(model_step_predictions, dim=0).mean(dim=0)
                predictions.append(y.detach().cpu().numpy())
                x[:-1] = x[1:].clone()
                x[-1] = y

        return np.array(predictions)
