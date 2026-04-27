import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from .datasets import RNNCustomDataset
from .models import RNNNet


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


def _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic):
    axes = []
    for start, block_dim, crystal_dim in zip(origin, train_supercell_shape, crystal_shape):
        indices = np.arange(start, start + block_dim)
        if periodic:
            indices = indices % crystal_dim
        axes.append(indices)
    return np.ix_(axes[0], axes[1], axes[2])


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
):
    """Run one PyTorch model over a full crystal displacement field."""
    crystal_shape = tuple(init_displacements.shape[1:4])
    origins = _build_supercell_origins(crystal_shape, train_supercell_shape, stride_shape, periodic)
    if not origins:
        raise ValueError("No supercell origins were generated")

    model.eval()
    x = torch.as_tensor(init_displacements, dtype=torch.float32)
    predictions = []

    with torch.no_grad():
        for _ in range(count_steps):
            prediction_sum = torch.zeros_like(x[-1])
            prediction_count = torch.zeros((*crystal_shape, unit_cell_atoms, 1), dtype=x.dtype, device=x.device)

            for origin in origins:
                index = _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic)
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
                prediction_sum[index] += block_y
                prediction_count[index] += 1

            if torch.any(prediction_count == 0):
                raise ValueError("Some crystal cells were not covered by any inference block")

            # Overlapping supercell predictions are averaged per physical cell and atom.
            y = prediction_sum / prediction_count
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
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = type.upper()
        self.train_supercell_shape = None if train_supercell_shape is None else _as_shape3(
            "train_supercell_shape",
            train_supercell_shape,
        )
        self.unit_cell_atoms = None if unit_cell_atoms is None else int(unit_cell_atoms)
        self.flatten_order = _normalize_flatten_order(flatten_order)
        self.in_features = self._resolve_in_features(in_features)
        self._validate_crystal_metadata()
        self.model = RNNNet(self.in_features, self.hidden_size, num_layers, type=self.rnn_type)
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 200
        self.train_count = 200

    def reset(self):
        self.model = RNNNet(self.in_features, self.hidden_size, self.num_layers, type=self.rnn_type)

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
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0
            lm_count = 0

            for x_train, y_train in train_data:
                predict = self.model(x_train)
                loss = loss_func(predict, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

            losses.append(loss_mean)

        return losses

    def train_crystal_blocks(self, X_blocks, y_blocks, data_len=0.5):
        """Train from crystal-shaped supercell blocks.

        Args:
            X_blocks: Training histories with shape
                `(n_samples, sequence_length, bx, by, bz, unit_cell_atoms, 3)`.
            y_blocks: One-step targets with shape
                `(n_samples, bx, by, bz, unit_cell_atoms, 3)`.
            data_len: Fraction of samples used for one randomized training slice.

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

        X_coords = _flatten_crystal_block_samples(X_blocks, self.flatten_order)
        y_coords = _flatten_crystal_block_targets(y_blocks, self.flatten_order)
        return self.train(X_coords, y_coords, data_len=data_len)

    def run(self, count_steps, init_features):
        """Autoregressively roll out predictions from one flat input history."""
        self.model.eval()
        mas = []
        x = torch.as_tensor(init_features, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(count_steps):
                y = self.model(x).squeeze(0)
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

        return _run_model_on_crystal(
            self.model,
            count_steps,
            init_displacements,
            train_supercell_shape,
            unit_cell_atoms,
            stride_shape,
            periodic,
            self.flatten_order,
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

    def run(self, count_steps, init_features, separate=False):
        """Run an ensemble rollout from one flat input history."""
        for model in self.models:
            model.eval()

        mas = []
        if separate:
            for model in self.models:
                x = torch.as_tensor(init_features, dtype=torch.float32)
                mas1 = []
                with torch.no_grad():
                    for _ in range(count_steps):
                        y = model(x).squeeze(0)
                        mas1.append(y.detach().cpu().numpy())
                        x[0] = torch.vstack([x[0, 1:], y])
                mas.append(mas1)
            mas = np.array(mas)
            mas = mas.mean(axis=0)
        else:
            x = torch.as_tensor(init_features, dtype=torch.float32)
            with torch.no_grad():
                for _ in range(count_steps):
                    y = torch.zeros(len(self.models), self.in_features, dtype=x.dtype)
                    for i, model in enumerate(self.models):
                        y[i, :] = model(x).squeeze(0)
                    z = torch.mean(y, dim=0)
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
    ):
        """Run ensemble inference over a full crystal displacement field."""
        if train_supercell_shape is None:
            if self.train_supercell_shape is None:
                raise ValueError("train_supercell_shape is required for models without crystal metadata")
            train_supercell_shape = self.train_supercell_shape
        if flatten_order is None:
            flatten_order = self.flatten_order
        flatten_order = _normalize_flatten_order(flatten_order)
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

        for model in self.models:
            model.eval()

        x = torch.as_tensor(init_displacements, dtype=torch.float32)
        predictions = []

        with torch.no_grad():
            for _ in range(count_steps):
                model_step_predictions = []
                for model in self.models:
                    prediction_sum = torch.zeros_like(x[-1])
                    prediction_count = torch.zeros((*crystal_shape, unit_cell_atoms, 1), dtype=x.dtype, device=x.device)

                    for origin in origins:
                        index = _supercell_indices(origin, train_supercell_shape, crystal_shape, periodic)
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
                        prediction_sum[index] += block_y
                        prediction_count[index] += 1

                    if torch.any(prediction_count == 0):
                        raise ValueError("Some crystal cells were not covered by any inference block")

                    # Average overlaps for this model before ensemble averaging.
                    model_step_predictions.append(prediction_sum / prediction_count)

                y = torch.stack(model_step_predictions, dim=0).mean(dim=0)
                predictions.append(y.detach().cpu().numpy())
                x[:-1] = x[1:].clone()
                x[-1] = y

        return np.array(predictions)
