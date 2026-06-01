"""Hybrid flat-RNN and ConvRNN crystal predictor."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from .conv_crystal_predictor import (
    CrystalConvRNNNet,
    _channels_to_crystal,
    _crystal_to_channels,
    _normalize_delta_loss_epsilon,
    _normalize_delta_loss_weight,
)
from .crystal_predictor import (
    DEFAULT_FLATTEN_ORDER,
    ORDER_AXIS_TO_DIM,
    CrystalRNNNet,
    _as_shape3,
    _build_merge_blocks,
    _build_supercell_origins,
    _normalize_flatten_order,
    _normalize_merge_mode,
    _normalize_target_mode,
)


def _torch_flatten_supercell_batch(blocks, flatten_order):
    """Flatten torch crystal blocks with shape ``(..., bx, by, bz, atoms, 3)``."""
    leading_dims = blocks.shape[:-5]
    axes = [ORDER_AXIS_TO_DIM[name] for name in flatten_order]
    permute_axes = list(range(len(leading_dims))) + [len(leading_dims) + axis for axis in axes]
    return blocks.permute(*permute_axes).reshape(*leading_dims, -1)


def _torch_unflatten_supercell_batch(flat_features, train_supercell_shape, unit_cell_atoms, flatten_order):
    """Unflatten torch vectors to ``(..., bx, by, bz, atoms, 3)`` blocks."""
    leading_dims = flat_features.shape[:-1]
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

    ordered = flat_features.reshape(*leading_dims, *ordered_shape)
    inverse_axes = np.argsort([ORDER_AXIS_TO_DIM[name] for name in flatten_order]).tolist()
    permute_axes = list(range(len(leading_dims))) + [len(leading_dims) + axis for axis in inverse_axes]
    return ordered.permute(*permute_axes).reshape(*leading_dims, *train_supercell_shape, unit_cell_atoms, 3)


class CrystalHybridRNNNet:
    """Blend flat block RNN predictions with full-field ConvRNN predictions.

    The flat branch keeps the legacy high-capacity block predictor.  The
    convolutional branch predicts a spatially consistent full-crystal field.
    Their outputs are mixed as

    ``alpha * flat_prediction + (1 - alpha) * conv_prediction``

    where ``alpha = sigmoid(logit_alpha)`` is trained together with both
    branches.
    """

    def __init__(
        self,
        flat_hidden_size,
        flat_num_layers,
        conv_hidden_channels,
        conv_num_layers,
        train_supercell_shape,
        unit_cell_atoms,
        flat_type="RNN",
        conv_type="ConvGRU",
        conv_kernel_size=3,
        conv_periodic_padding=True,
        flatten_order=DEFAULT_FLATTEN_ORDER,
        target_mode="absolute_delta",
        delta_loss_weight=1.0,
        delta_loss_epsilon=1e-6,
        residual_output=False,
        initial_alpha=0.5,
        freeze_flat=False,
    ):
        self.train_supercell_shape = _as_shape3("train_supercell_shape", train_supercell_shape)
        self.unit_cell_atoms = int(unit_cell_atoms)
        self.flatten_order = _normalize_flatten_order(flatten_order)
        self.target_mode = _normalize_target_mode(target_mode)
        self.delta_loss_weight = _normalize_delta_loss_weight(delta_loss_weight)
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(delta_loss_epsilon)
        self.residual_output = bool(residual_output)
        self.flat_residual_output = bool(residual_output)
        self.conv_residual_output = bool(residual_output)
        self.freeze_flat = bool(freeze_flat)
        initial_alpha = float(initial_alpha)
        if not 0 < initial_alpha < 1:
            raise ValueError("initial_alpha must be between 0 and 1")

        self.flat_model = CrystalRNNNet(
            hidden_size=flat_hidden_size,
            num_layers=flat_num_layers,
            type=flat_type,
            train_supercell_shape=self.train_supercell_shape,
            unit_cell_atoms=self.unit_cell_atoms,
            flatten_order=self.flatten_order,
            target_mode="absolute",
        )
        self.conv_model = CrystalConvRNNNet(
            hidden_channels=conv_hidden_channels,
            num_layers=conv_num_layers,
            unit_cell_atoms=self.unit_cell_atoms,
            type=conv_type,
            kernel_size=conv_kernel_size,
            periodic_padding=conv_periodic_padding,
            target_mode="absolute",
            residual_output=False,
        )
        self.logit_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha / (1 - initial_alpha)), dtype=torch.float32))
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 64
        self.train_count = 200

    @property
    def alpha(self):
        """Current flat-branch mixture weight as a Python float."""
        return float(torch.sigmoid(self.logit_alpha).detach().cpu())

    def _parameters(self):
        parameters = list(self.conv_model.model.parameters()) + [self.logit_alpha]
        if not self.freeze_flat:
            parameters = list(self.flat_model.model.parameters()) + parameters
        return parameters

    def set_flat_model(self, flat_model, freeze_flat=None):
        """Replace the flat branch with a pretrained `CrystalRNNNet`."""
        if flat_model.train_supercell_shape != self.train_supercell_shape:
            raise ValueError("Pretrained flat model train_supercell_shape does not match hybrid geometry")
        if flat_model.unit_cell_atoms != self.unit_cell_atoms:
            raise ValueError("Pretrained flat model unit_cell_atoms does not match hybrid geometry")
        if tuple(flat_model.flatten_order) != tuple(self.flatten_order):
            raise ValueError("Pretrained flat model flatten_order does not match hybrid flatten_order")
        self.flat_model = flat_model
        self.flat_residual_output = getattr(flat_model, "target_mode", "absolute") == "delta"
        if freeze_flat is not None:
            self.freeze_flat = bool(freeze_flat)
        for parameter in self.flat_model.model.parameters():
            parameter.requires_grad = not self.freeze_flat

    def _branch_predictions(self, x_blocks):
        """Predict one block from both branches and return mixed block output."""
        flat_input = _torch_flatten_supercell_batch(x_blocks, self.flatten_order)
        flat_prediction = self.flat_model.model(flat_input)
        flat_prediction = _torch_unflatten_supercell_batch(
            flat_prediction,
            self.train_supercell_shape,
            self.unit_cell_atoms,
            self.flatten_order,
        )

        conv_input = _crystal_to_channels(x_blocks)
        conv_prediction = self.conv_model.model(conv_input)
        conv_prediction = _channels_to_crystal(conv_prediction, self.unit_cell_atoms)

        if self.flat_residual_output and self.target_mode != "delta":
            flat_prediction = x_blocks[:, -1] + flat_prediction
        if self.conv_residual_output and self.target_mode != "delta":
            conv_prediction = x_blocks[:, -1] + conv_prediction

        alpha = torch.sigmoid(self.logit_alpha)
        return alpha * flat_prediction + (1 - alpha) * conv_prediction

    def train_crystal_blocks(self, X_blocks, y_blocks, data_len=0.5):
        """Train both branches and the mixture weight from crystal block samples."""
        X_blocks = np.asarray(X_blocks, dtype=np.float32)
        y_blocks = np.asarray(y_blocks, dtype=np.float32)
        if X_blocks.ndim != 7:
            raise ValueError("X_blocks must have shape (n, sequence, bx, by, bz, atoms, 3)")
        if y_blocks.ndim != 6:
            raise ValueError("y_blocks must have shape (n, bx, by, bz, atoms, 3)")
        if tuple(X_blocks.shape[2:5]) != self.train_supercell_shape:
            raise ValueError("X_blocks supercell shape does not match train_supercell_shape")
        if X_blocks.shape[5] != self.unit_cell_atoms:
            raise ValueError("X_blocks unit_cell_atoms does not match model metadata")
        if X_blocks.shape[0] != y_blocks.shape[0]:
            raise ValueError("X_blocks and y_blocks must contain the same number of samples")

        self.train_count = int(data_len * X_blocks.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_blocks.shape[0]:
            start_index = 0
        else:
            start_index = np.random.randint(low=0, high=X_blocks.shape[0] - self.train_count)

        X_train = X_blocks[start_index : start_index + self.train_count]
        y_train = y_blocks[start_index : start_index + self.train_count]
        if self.target_mode == "delta":
            y_train = y_train - X_train[:, -1]

        dataset = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        if self.freeze_flat:
            self.flat_model.model.eval()
        else:
            self.flat_model.model.train()
        self.conv_model.model.train()
        optimizer = optim.Adam(self._parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0.0
            batch_count = 0
            for x_train, y_train in loader:
                predicted = self._branch_predictions(x_train)
                loss = loss_func(predicted, y_train)
                if self.target_mode == "absolute_delta" and self.delta_loss_weight > 0:
                    last_input = x_train[:, -1]
                    true_delta = y_train - last_input
                    pred_delta = predicted - last_input
                    delta_scale = torch.sqrt(torch.mean(true_delta**2, dim=(1, 2, 3, 4, 5), keepdim=True)).clamp_min(
                        self.delta_loss_epsilon
                    )
                    loss = loss + self.delta_loss_weight * loss_func(pred_delta / delta_scale, true_delta / delta_scale)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1
                loss_mean = loss.item() / batch_count + (1 - 1 / batch_count) * loss_mean
            losses.append(loss_mean)

        return losses

    def run_crystal(
        self,
        count_steps,
        init_displacements,
        stride_shape=None,
        periodic=False,
        merge_mode="owner",
        merge_top_k=None,
        merge_alpha=1.0,
    ):
        """Autoregressively run the hybrid predictor on a full crystal."""
        if count_steps <= 0:
            raise ValueError("count_steps must be positive")
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if init_displacements.ndim != 6:
            raise ValueError("init_displacements must have shape (sequence, nx, ny, nz, atoms, 3)")
        if init_displacements.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match init_displacements")

        crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
        stride_shape = (1, 1, 1) if stride_shape is None else _as_shape3("stride_shape", stride_shape)
        merge_deltas, base_merge_mode = _normalize_merge_mode(merge_mode)
        if self.target_mode == "delta" and not merge_mode.startswith("delta_"):
            merge_deltas = True
        origins = _build_supercell_origins(crystal_shape, self.train_supercell_shape, stride_shape, periodic)
        merge_blocks = _build_merge_blocks(
            crystal_shape,
            self.train_supercell_shape,
            origins,
            periodic,
            base_merge_mode,
            merge_top_k,
            merge_alpha,
        )

        self.flat_model.model.eval()
        self.conv_model.model.eval()
        x = torch.as_tensor(init_displacements, dtype=torch.float32).clone()
        predictions = []

        with torch.no_grad():
            for _ in range(count_steps):
                conv_input = _crystal_to_channels(x).unsqueeze(0)
                conv_full = self.conv_model.model(conv_input).squeeze(0)
                conv_full = _channels_to_crystal(conv_full, self.unit_cell_atoms)
                if self.conv_residual_output and self.target_mode != "delta":
                    conv_full = x[-1] + conv_full

                prediction_sum = torch.zeros_like(x[-1])
                prediction_weight = torch.zeros((*crystal_shape, self.unit_cell_atoms, 1), dtype=x.dtype)
                alpha = torch.sigmoid(self.logit_alpha)

                for _, index, local_weights in merge_blocks:
                    block_x = x[(slice(None), *index, slice(None), slice(None))]
                    flat_input = _torch_flatten_supercell_batch(block_x.unsqueeze(0), self.flatten_order)
                    flat_prediction = self.flat_model.model(flat_input).squeeze(0)
                    flat_prediction = _torch_unflatten_supercell_batch(
                        flat_prediction,
                        self.train_supercell_shape,
                        self.unit_cell_atoms,
                        self.flatten_order,
                    )
                    if self.flat_residual_output and self.target_mode != "delta":
                        flat_prediction = block_x[-1] + flat_prediction

                    block_prediction = alpha * flat_prediction + (1 - alpha) * conv_full[index]
                    if self.target_mode != "delta" and merge_deltas:
                        block_prediction = block_prediction - block_x[-1]

                    weights = torch.as_tensor(local_weights, dtype=x.dtype)
                    prediction_sum[index] += block_prediction * weights
                    prediction_weight[index] += weights

                if torch.any(prediction_weight == 0):
                    raise ValueError("Some crystal cells were not covered by any inference block")

                y = prediction_sum / prediction_weight
                if merge_deltas:
                    y = x[-1] + y
                predictions.append(y.detach().cpu().numpy())
                x[:-1] = x[1:].clone()
                x[-1] = y

        return np.asarray(predictions, dtype=np.float32)
