"""Local translation-equivariant RNN predictor for crystal displacements.

The regular ``CrystalRNNNet`` learns a full supercell-to-supercell map.  That
gives every position inside the training block its own output channel, so the
model can easily learn position-specific biases.  ``CrystalLocalRNNNet`` uses a
different convention: one shared RNN sees a local crystal patch around each
unit cell and predicts only the central unit-cell displacement.  Applying the
same predictor at every cell makes the rollout translation-equivariant up to
the fixed orientation of the local patch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from .crystal_predictor import (
    DEFAULT_FLATTEN_ORDER,
    ORDER_AXIS_TO_DIM,
    _as_shape3,
    _flatten_supercell,
    _normalize_acceleration_loss_epsilon,
    _normalize_acceleration_loss_weight,
    _normalize_delta_loss_epsilon,
    _normalize_delta_loss_weight,
    _normalize_flatten_order,
    _normalize_target_mode,
    _validate_crystal_input,
)
from .datasets import RNNCustomDataset
from .models import RNNNet


CELL_AXIS_TO_DIM = {
    "atom": 0,
    "coord": 1,
}


def _normalize_patch_shape(patch_shape):
    """Validate the local patch shape and require a unique central cell."""
    patch_shape = _as_shape3("patch_shape", patch_shape)
    if any(dim % 2 == 0 for dim in patch_shape):
        raise ValueError("patch_shape dimensions must be odd so that the central cell is unique")
    return patch_shape


def _cell_order(flatten_order):
    """Return the atom/coord part of a full crystal flatten order."""
    return tuple(name for name in flatten_order if name in CELL_AXIS_TO_DIM)


def _flatten_unit_cell(cell_displacements, flatten_order):
    """Pack one unit cell according to the atom/coord part of flatten_order."""
    cell_displacements = np.asarray(cell_displacements)
    axes = [CELL_AXIS_TO_DIM[name] for name in _cell_order(flatten_order)]
    return np.transpose(cell_displacements, axes=axes).reshape(-1)


def _unflatten_unit_cell(flat_features, unit_cell_atoms, flatten_order):
    """Restore a flat central-cell output to ``(unit_cell_atoms, 3)``."""
    ordered_names = _cell_order(flatten_order)
    ordered_shape = [unit_cell_atoms if name == "atom" else 3 for name in ordered_names]
    ordered = np.asarray(flat_features, dtype=np.float32).reshape(*ordered_shape)
    inverse_axes = np.argsort([CELL_AXIS_TO_DIM[name] for name in ordered_names])
    return np.transpose(ordered, axes=inverse_axes).reshape(unit_cell_atoms, 3)


def _patch_axis_indices(center, patch_dim, crystal_dim, periodic):
    """Return crystal indices covered by one local patch axis."""
    radius = patch_dim // 2
    indices = np.arange(center - radius, center + radius + 1)
    if periodic:
        return indices % crystal_dim
    if indices[0] < 0 or indices[-1] >= crystal_dim:
        raise ValueError("Patch crosses a non-periodic crystal boundary")
    return indices


def _extract_patch(frame, center, patch_shape, periodic):
    """Extract one local patch around ``center`` from a crystal frame."""
    crystal_shape = frame.shape[:3]
    indices = [
        _patch_axis_indices(center[axis], patch_shape[axis], crystal_shape[axis], periodic)
        for axis in range(3)
    ]
    return frame[np.ix_(indices[0], indices[1], indices[2])]


def _valid_patch_centers(crystal_shape, patch_shape, periodic):
    """List centers whose local patch can be extracted."""
    if periodic:
        return list(np.ndindex(crystal_shape))

    radii = tuple(dim // 2 for dim in patch_shape)
    ranges = [
        range(radii[axis], crystal_shape[axis] - radii[axis])
        for axis in range(3)
    ]
    return [(ix, iy, iz) for ix in ranges[0] for iy in ranges[1] for iz in ranges[2]]


def _flatten_local_patch_samples(patches, flatten_order):
    """Flatten local patch histories into RNN input tensors."""
    return np.array(
        [
            [_flatten_supercell(frame, flatten_order) for frame in sample]
            for sample in patches
        ],
        dtype=np.float32,
    )


def _flatten_local_cell_targets(cells, flatten_order):
    """Flatten central-cell targets into RNN output tensors."""
    return np.array([_flatten_unit_cell(cell, flatten_order) for cell in cells], dtype=np.float32)


def make_local_patch_samples(
    displacements,
    sequence_length,
    patch_shape,
    sample_count,
    rng=None,
    start_window=0,
    stop_window=None,
    periodic=True,
    return_metadata=False,
):
    """Randomly sample local patch histories and central-cell targets.

    Parameters
    ----------
    displacements:
        Crystal displacement trajectory with shape
        ``(frames, nx, ny, nz, unit_cell_atoms, 3)``.
    sequence_length:
        Number of consecutive history frames supplied to the RNN.
    patch_shape:
        Odd local neighborhood size in unit cells, for example ``(3, 3, 3)``.
    sample_count:
        Number of random ``(time, center_cell)`` samples to generate.
    rng:
        Optional ``numpy.random.Generator`` or seed.
    start_window, stop_window:
        Time-window range.  A window index ``t`` uses frames
        ``t:t+sequence_length`` as input and frame ``t+sequence_length`` as the
        target.
    periodic:
        Whether patches wrap around crystal boundaries.
    return_metadata:
        If true, also return sampled time-window indices and center cells.
    """
    displacements = np.asarray(displacements, dtype=np.float32)
    patch_shape = _normalize_patch_shape(patch_shape)
    if displacements.ndim != 6:
        raise ValueError("displacements must have shape (frames, nx, ny, nz, unit_cell_atoms, 3)")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if displacements.shape[0] <= sequence_length:
        raise ValueError("Need more frames than sequence_length")

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    total_windows = displacements.shape[0] - sequence_length
    start_window = int(start_window)
    stop_window = total_windows if stop_window is None else int(stop_window)
    if start_window < 0 or stop_window > total_windows or start_window >= stop_window:
        raise ValueError("Invalid local patch time-window range")

    crystal_shape = tuple(int(dim) for dim in displacements.shape[1:4])
    centers = _valid_patch_centers(crystal_shape, patch_shape, periodic)
    if not centers:
        raise ValueError("No valid local patch centers")

    time_indices = rng.integers(start_window, stop_window, size=sample_count)
    center_indices = rng.integers(0, len(centers), size=sample_count)
    sampled_centers = [centers[index] for index in center_indices]

    patches = np.empty(
        (sample_count, sequence_length, *patch_shape, displacements.shape[4], 3),
        dtype=np.float32,
    )
    targets = np.empty((sample_count, displacements.shape[4], 3), dtype=np.float32)
    for sample_index, (time_index, center) in enumerate(zip(time_indices, sampled_centers)):
        for history_index in range(sequence_length):
            patches[sample_index, history_index] = _extract_patch(
                displacements[time_index + history_index],
                center,
                patch_shape,
                periodic,
            )
        targets[sample_index] = displacements[time_index + sequence_length][center]

    if return_metadata:
        return patches, targets, {"time_indices": time_indices, "centers": np.asarray(sampled_centers)}
    return patches, targets


class CrystalLocalRNNNet:
    """Shared local RNN operator for crystal displacement rollouts.

    The model input is a history of local patches with shape
    ``(sequence_length, px, py, pz, unit_cell_atoms, 3)``.  The model output is
    only the next displacement of the central unit cell, with shape
    ``(unit_cell_atoms, 3)``.  During ``run_crystal`` the same network is applied
    to every unit cell of the crystal, using periodic patches by default.
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        patch_shape=(3, 3, 3),
        unit_cell_atoms=None,
        type="RNN",
        flatten_order=DEFAULT_FLATTEN_ORDER,
        target_mode="absolute_delta",
        delta_loss_weight=1.0,
        delta_loss_epsilon=1e-6,
        acceleration_loss_weight=0.0,
        acceleration_loss_epsilon=1e-8,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.rnn_type = type.upper()
        self.patch_shape = _normalize_patch_shape(patch_shape)
        self.train_supercell_shape = self.patch_shape
        if unit_cell_atoms is None:
            raise ValueError("unit_cell_atoms is required")
        self.unit_cell_atoms = int(unit_cell_atoms)
        if self.unit_cell_atoms <= 0:
            raise ValueError("unit_cell_atoms must be positive")
        self.flatten_order = _normalize_flatten_order(flatten_order)
        self.target_mode = _normalize_target_mode(target_mode)
        self.delta_loss_weight = _normalize_delta_loss_weight(delta_loss_weight)
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(delta_loss_epsilon)
        self.acceleration_loss_weight = _normalize_acceleration_loss_weight(acceleration_loss_weight)
        self.acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(acceleration_loss_epsilon)
        self.in_features = int(np.prod(self.patch_shape) * self.unit_cell_atoms * 3)
        self.out_features = int(self.unit_cell_atoms * 3)
        self.model = RNNNet(
            self.in_features,
            self.hidden_size,
            self.num_layers,
            type=self.rnn_type,
            out_features=self.out_features,
        )
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 200
        self.train_count = 200
        self.default_periodic = True
        self._center_feature_indices = self._build_center_feature_indices()

    @property
    def center_index(self):
        """Return the central patch index."""
        return tuple(dim // 2 for dim in self.patch_shape)

    def _build_center_feature_indices(self):
        """Map flattened patch features to the flattened central-cell order."""
        dummy = np.arange(self.in_features, dtype=np.int64).reshape(
            *self.patch_shape,
            self.unit_cell_atoms,
            3,
        )
        flat_patch = _flatten_supercell(dummy, self.flatten_order)
        flat_cell = _flatten_unit_cell(dummy[self.center_index], self.flatten_order)
        return np.array([int(np.where(flat_patch == value)[0][0]) for value in flat_cell], dtype=np.int64)

    def reset(self):
        """Reinitialize the underlying RNN while preserving geometry metadata."""
        self.target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute_delta"))
        self.delta_loss_weight = _normalize_delta_loss_weight(getattr(self, "delta_loss_weight", 1.0))
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(getattr(self, "delta_loss_epsilon", 1e-6))
        self.acceleration_loss_weight = _normalize_acceleration_loss_weight(
            getattr(self, "acceleration_loss_weight", 0.0)
        )
        self.acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(
            getattr(self, "acceleration_loss_epsilon", 1e-8)
        )
        self.model = RNNNet(
            self.in_features,
            self.hidden_size,
            self.num_layers,
            type=self.rnn_type,
            out_features=self.out_features,
        )

    def flatten_patch(self, patch_displacements):
        """Pack one local patch according to this model's ``flatten_order``."""
        patch_displacements = np.asarray(patch_displacements, dtype=np.float32)
        if tuple(patch_displacements.shape) != (*self.patch_shape, self.unit_cell_atoms, 3):
            raise ValueError("patch_displacements shape does not match model geometry")
        return _flatten_supercell(patch_displacements, self.flatten_order)

    def flatten_cell(self, cell_displacements):
        """Pack one central-cell displacement vector."""
        cell_displacements = np.asarray(cell_displacements, dtype=np.float32)
        if tuple(cell_displacements.shape) != (self.unit_cell_atoms, 3):
            raise ValueError("cell_displacements shape does not match unit_cell_atoms")
        return _flatten_unit_cell(cell_displacements, self.flatten_order)

    def unflatten_cell(self, flat_features):
        """Restore a flat central-cell output to ``(unit_cell_atoms, 3)``."""
        return _unflatten_unit_cell(flat_features, self.unit_cell_atoms, self.flatten_order)

    def _center_from_flat_history(self, x_train, history_index):
        """Select central-cell features from a flattened patch history batch."""
        indices = torch.as_tensor(self._center_feature_indices, dtype=torch.long, device=x_train.device)
        return x_train[:, history_index, :].index_select(dim=1, index=indices)

    def _prediction_loss(self, x_train, y_train, raw_prediction, loss_func):
        """Build the training loss according to the configured target mode."""
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute_delta"))
        delta_loss_weight = _normalize_delta_loss_weight(getattr(self, "delta_loss_weight", 1.0))
        delta_loss_epsilon = _normalize_delta_loss_epsilon(getattr(self, "delta_loss_epsilon", 1e-6))
        acceleration_loss_weight = _normalize_acceleration_loss_weight(
            getattr(self, "acceleration_loss_weight", 0.0)
        )
        acceleration_loss_epsilon = _normalize_acceleration_loss_epsilon(
            getattr(self, "acceleration_loss_epsilon", 1e-8)
        )

        last_input = self._center_from_flat_history(x_train, -1)
        if target_mode == "verlet":
            previous_input = self._center_from_flat_history(x_train, -2)
            train_prediction = 2 * last_input - previous_input + raw_prediction
        else:
            train_prediction = raw_prediction

        loss = loss_func(train_prediction, y_train)
        if target_mode in {"absolute_delta", "verlet"} and delta_loss_weight > 0:
            true_delta = y_train - last_input
            pred_delta = train_prediction - last_input
            delta_scale = torch.sqrt(torch.mean(true_delta**2, dim=1, keepdim=True)).clamp_min(delta_loss_epsilon)
            loss = loss + delta_loss_weight * loss_func(pred_delta / delta_scale, true_delta / delta_scale)

        if acceleration_loss_weight > 0:
            previous_input = self._center_from_flat_history(x_train, -2)
            if target_mode in {"acceleration", "verlet"}:
                pred_acceleration = raw_prediction
                true_acceleration = y_train if target_mode == "acceleration" else y_train - 2 * last_input + previous_input
            elif target_mode == "delta":
                pred_next = last_input + raw_prediction
                true_next = last_input + y_train
                pred_acceleration = pred_next - 2 * last_input + previous_input
                true_acceleration = true_next - 2 * last_input + previous_input
            else:
                pred_next = raw_prediction
                true_next = y_train
                pred_acceleration = pred_next - 2 * last_input + previous_input
                true_acceleration = true_next - 2 * last_input + previous_input
            acceleration_scale = torch.sqrt(
                torch.mean(true_acceleration**2, dim=1, keepdim=True)
            ).clamp_min(acceleration_loss_epsilon)
            loss = loss + acceleration_loss_weight * loss_func(
                pred_acceleration / acceleration_scale,
                true_acceleration / acceleration_scale,
            )

        return loss

    def train(self, X_coords, y_coords, data_len=1.0):
        """Train from flattened local patch histories and central-cell targets."""
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute_delta"))
        if target_mode in {"acceleration", "verlet"} and X_coords.shape[1] < 2:
            raise ValueError(f"target_mode={target_mode!r} requires at least two history frames")
        if self.acceleration_loss_weight > 0 and X_coords.shape[1] < 2:
            raise ValueError("acceleration loss requires at least two input history frames")

        self.train_count = int(data_len * X_coords.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_coords.shape[0]:
            ind = 0
        else:
            ind = np.random.randint(low=0, high=X_coords.shape[0] - self.train_count)

        train_dataset = RNNCustomDataset(
            X_coords[ind : ind + self.train_count],
            y_coords[ind : ind + self.train_count],
        )
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0.0
            lm_count = 0
            for x_train, y_train in train_data:
                raw_prediction = self.model(x_train)
                loss = self._prediction_loss(x_train, y_train, raw_prediction, loss_func)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            losses.append(loss_mean)

        return losses

    def train_local_patches(self, X_patches, y_cells, data_len=1.0, target_mode=None):
        """Train from crystal-shaped local patches.

        ``X_patches`` has shape
        ``(n_samples, sequence_length, px, py, pz, unit_cell_atoms, 3)`` and
        ``y_cells`` has shape ``(n_samples, unit_cell_atoms, 3)``.
        """
        X_patches = np.asarray(X_patches, dtype=np.float32)
        y_cells = np.asarray(y_cells, dtype=np.float32)
        if X_patches.ndim != 7:
            raise ValueError(
                "X_patches must have shape "
                "(n_samples, sequence_length, px, py, pz, unit_cell_atoms, 3)"
            )
        if y_cells.ndim != 3:
            raise ValueError("y_cells must have shape (n_samples, unit_cell_atoms, 3)")
        if tuple(X_patches.shape[2:5]) != self.patch_shape:
            raise ValueError("X_patches patch shape does not match model patch_shape")
        if tuple(X_patches.shape[5:7]) != (self.unit_cell_atoms, 3):
            raise ValueError("X_patches atom/coordinate dimensions do not match model metadata")
        if tuple(y_cells.shape[1:3]) != (self.unit_cell_atoms, 3):
            raise ValueError("y_cells atom/coordinate dimensions do not match model metadata")
        if X_patches.shape[0] != y_cells.shape[0]:
            raise ValueError("X_patches and y_cells must contain the same number of samples")

        if target_mode is not None:
            self.target_mode = _normalize_target_mode(target_mode)
        self.target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute_delta"))
        center_history = X_patches[(slice(None), slice(None), *self.center_index, slice(None), slice(None))]
        if self.target_mode == "delta":
            y_cells = y_cells - center_history[:, -1]
        elif self.target_mode == "acceleration":
            if X_patches.shape[1] < 2:
                raise ValueError("target_mode='acceleration' requires at least two input history frames")
            y_cells = y_cells - 2 * center_history[:, -1] + center_history[:, -2]
        elif self.target_mode == "verlet" and X_patches.shape[1] < 2:
            raise ValueError("target_mode='verlet' requires at least two input history frames")

        X_coords = _flatten_local_patch_samples(X_patches, self.flatten_order)
        y_coords = _flatten_local_cell_targets(y_cells, self.flatten_order)
        return self.train(X_coords, y_coords, data_len=data_len)

    def _decode_raw_cells(self, raw_prediction, center_history):
        """Convert raw network outputs to absolute next-cell displacements."""
        raw_cells = np.array(
            [self.unflatten_cell(flat_cell) for flat_cell in np.asarray(raw_prediction, dtype=np.float32)],
            dtype=np.float32,
        )
        target_mode = _normalize_target_mode(getattr(self, "target_mode", "absolute_delta"))
        if target_mode == "delta":
            return center_history[:, -1] + raw_cells
        if target_mode in {"acceleration", "verlet"}:
            return 2 * center_history[:, -1] - center_history[:, -2] + raw_cells
        return raw_cells

    def predict_local_patches(self, X_patches, batch_size=None):
        """Predict next central-cell displacements for local patch samples."""
        X_patches = np.asarray(X_patches, dtype=np.float32)
        if X_patches.ndim != 7:
            raise ValueError(
                "X_patches must have shape "
                "(n_samples, sequence_length, px, py, pz, unit_cell_atoms, 3)"
            )
        if tuple(X_patches.shape[2:5]) != self.patch_shape:
            raise ValueError("X_patches patch shape does not match model patch_shape")
        center_history = X_patches[(slice(None), slice(None), *self.center_index, slice(None), slice(None))]
        batch_size = X_patches.shape[0] if batch_size is None else int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model.eval()
        predictions = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for start in range(0, X_patches.shape[0], batch_size):
                stop = min(start + batch_size, X_patches.shape[0])
                X_flat = _flatten_local_patch_samples(X_patches[start:stop], self.flatten_order)
                raw = self.model(torch.as_tensor(X_flat, dtype=torch.float32, device=device))
                predictions.append(raw.detach().cpu().numpy())
        raw_prediction = np.concatenate(predictions, axis=0)
        return self._decode_raw_cells(raw_prediction, center_history)

    def run(self, count_steps, init_features):
        """Local models need full-crystal context for autoregressive rollout."""
        raise NotImplementedError("CrystalLocalRNNNet.rollout requires run_crystal(), not flat run().")

    def run_crystal(self, count_steps, init_displacements, periodic=True):
        """Autoregressively roll out a full crystal with the shared local RNN."""
        if count_steps <= 0:
            raise ValueError("count_steps must be positive")
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
        _validate_crystal_input(init_displacements, crystal_shape, self.unit_cell_atoms)
        if self.target_mode in {"acceleration", "verlet"} and init_displacements.shape[0] < 2:
            raise ValueError(f"target_mode={self.target_mode!r} requires at least two input history frames")

        centers = _valid_patch_centers(crystal_shape, self.patch_shape, periodic)
        if len(centers) != int(np.prod(crystal_shape)):
            raise ValueError("periodic=True is required unless every crystal cell has a full local patch")

        self.model.eval()
        x = init_displacements.copy()
        predictions = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for _ in range(count_steps):
                patch_batch = np.empty(
                    (len(centers), x.shape[0], *self.patch_shape, self.unit_cell_atoms, 3),
                    dtype=np.float32,
                )
                for center_index, center in enumerate(centers):
                    for history_index in range(x.shape[0]):
                        patch_batch[center_index, history_index] = _extract_patch(
                            x[history_index],
                            center,
                            self.patch_shape,
                            periodic,
                        )

                X_flat = _flatten_local_patch_samples(patch_batch, self.flatten_order)
                raw = self.model(torch.as_tensor(X_flat, dtype=torch.float32, device=device))
                raw = raw.detach().cpu().numpy()
                center_history = patch_batch[(slice(None), slice(None), *self.center_index, slice(None), slice(None))]
                next_cells = self._decode_raw_cells(raw, center_history)

                y = np.empty_like(x[-1])
                for center, cell in zip(centers, next_cells):
                    y[center] = cell

                predictions.append(y)
                x[:-1] = x[1:].copy()
                x[-1] = y

        return np.asarray(predictions, dtype=np.float32)
