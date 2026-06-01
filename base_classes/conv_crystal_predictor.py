"""Convolutional recurrent predictors for crystal displacement fields."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset


def _normalize_conv_rnn_type(rnn_type):
    """Validate the convolutional recurrent cell type."""
    rnn_type = str(rnn_type).upper()
    if rnn_type not in {"CONVRNN", "CONVGRU", "CONVLSTM", "RNN", "GRU", "LSTM"}:
        raise ValueError("type must be ConvRNN, ConvGRU, or ConvLSTM")
    if not rnn_type.startswith("CONV"):
        rnn_type = f"CONV{rnn_type}"
    return rnn_type


def _normalize_target_mode(target_mode):
    """Validate the model target convention."""
    target_mode = str(target_mode)
    if target_mode not in {"absolute", "delta", "absolute_delta"}:
        raise ValueError("target_mode must be 'absolute', 'delta', or 'absolute_delta'")
    return target_mode


def _normalize_positive_int(name, value):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_delta_loss_weight(delta_loss_weight):
    delta_loss_weight = float(delta_loss_weight)
    if delta_loss_weight < 0:
        raise ValueError("delta_loss_weight must be non-negative")
    return delta_loss_weight


def _normalize_delta_loss_epsilon(delta_loss_epsilon):
    delta_loss_epsilon = float(delta_loss_epsilon)
    if delta_loss_epsilon <= 0:
        raise ValueError("delta_loss_epsilon must be positive")
    return delta_loss_epsilon


def _crystal_to_channels(displacements):
    """Convert crystal layout to Conv3D channels.

    Input shape:
        ``(..., nx, ny, nz, unit_cell_atoms, 3)``

    Output shape:
        ``(..., unit_cell_atoms * 3, nx, ny, nz)``
    """
    tensor = torch.as_tensor(displacements, dtype=torch.float32)
    leading_dims = tensor.shape[:-5]
    nx, ny, nz, unit_cell_atoms, coords = tensor.shape[-5:]
    if coords != 3:
        raise ValueError("Last displacement dimension must contain x, y, z coordinates")
    tensor = tensor.reshape(*leading_dims, nx, ny, nz, unit_cell_atoms * coords)
    return tensor.permute(*range(len(leading_dims)), len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 1, len(leading_dims) + 2)


def _channels_to_crystal(channels, unit_cell_atoms):
    """Convert Conv3D channel layout back to crystal displacement layout."""
    tensor = torch.as_tensor(channels, dtype=torch.float32)
    leading_dims = tensor.shape[:-4]
    channels_count, nx, ny, nz = tensor.shape[-4:]
    expected_channels = unit_cell_atoms * 3
    if channels_count != expected_channels:
        raise ValueError("Channel count does not match unit_cell_atoms * 3")
    tensor = tensor.permute(*range(len(leading_dims)), len(leading_dims) + 1, len(leading_dims) + 2, len(leading_dims) + 3, len(leading_dims))
    return tensor.reshape(*leading_dims, nx, ny, nz, unit_cell_atoms, 3)


class _ConvRNNCell3D(nn.Module):
    """Simple Elman-style ConvRNN cell for 3D fields."""

    def __init__(self, input_channels, hidden_channels, kernel_size, padding_mode):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x, h):
        return torch.tanh(self.conv(torch.cat([x, h], dim=1)))


class _ConvGRUCell3D(nn.Module):
    """GRU-style recurrent cell with 3D convolutions."""

    def __init__(self, input_channels, hidden_channels, kernel_size, padding_mode):
        super().__init__()
        padding = kernel_size // 2
        gate_channels = input_channels + hidden_channels
        self.gates = nn.Conv3d(
            gate_channels,
            hidden_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.candidate = nn.Conv3d(
            gate_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x, h):
        z, r = torch.chunk(torch.sigmoid(self.gates(torch.cat([x, h], dim=1))), 2, dim=1)
        h_tilde = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class _ConvLSTMCell3D(nn.Module):
    """LSTM-style recurrent cell with 3D convolutions."""

    def __init__(self, input_channels, hidden_channels, kernel_size, padding_mode):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv3d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class _ConvRecurrentNet3D(nn.Module):
    """Stacked ConvRNN/ConvGRU/ConvLSTM followed by a 1x1x1 output convolution."""

    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size, rnn_type, padding_mode):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.rnn_type = _normalize_conv_rnn_type(rnn_type)
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode

        cells = []
        for layer_index in range(num_layers):
            layer_input_channels = input_channels if layer_index == 0 else hidden_channels
            if self.rnn_type == "CONVGRU":
                cell = _ConvGRUCell3D(layer_input_channels, hidden_channels, kernel_size, padding_mode)
            elif self.rnn_type == "CONVLSTM":
                cell = _ConvLSTMCell3D(layer_input_channels, hidden_channels, kernel_size, padding_mode)
            else:
                cell = _ConvRNNCell3D(layer_input_channels, hidden_channels, kernel_size, padding_mode)
            cells.append(cell)
        self.cells = nn.ModuleList(cells)
        self.out = nn.Conv3d(hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        """Predict one next frame from ``(batch, sequence, channels, nx, ny, nz)``."""
        batch_size, sequence_length, _, nx, ny, nz = x.shape
        states = []
        for _ in range(self.num_layers):
            h = x.new_zeros((batch_size, self.hidden_channels, nx, ny, nz))
            if self.rnn_type == "CONVLSTM":
                c = x.new_zeros((batch_size, self.hidden_channels, nx, ny, nz))
                states.append((h, c))
            else:
                states.append(h)

        for time_index in range(sequence_length):
            layer_input = x[:, time_index]
            next_states = []
            for layer_index, cell in enumerate(self.cells):
                if self.rnn_type == "CONVLSTM":
                    h, c = cell(layer_input, states[layer_index])
                    next_states.append((h, c))
                    layer_input = h
                else:
                    h = cell(layer_input, states[layer_index])
                    next_states.append(h)
                    layer_input = h
            states = next_states

        last_hidden = states[-1][0] if self.rnn_type == "CONVLSTM" else states[-1]
        return self.out(last_hidden)


class CrystalConvRNNNet:
    """Train and run convolutional recurrent predictors on crystal grids.

    The model treats ``unit_cell_atoms * 3`` as channels and keeps the unit-cell
    lattice as a 3D grid.  Because all recurrent operations are convolutional,
    a model trained on a small rectangular supercell can be applied directly to
    a larger crystal with the same number of atoms per unit cell.
    """

    def __init__(
        self,
        hidden_channels,
        num_layers,
        unit_cell_atoms,
        type="ConvGRU",
        kernel_size=3,
        periodic_padding=True,
        target_mode="absolute",
        delta_loss_weight=1.0,
        delta_loss_epsilon=1e-6,
        residual_output=False,
    ):
        self.hidden_channels = _normalize_positive_int("hidden_channels", hidden_channels)
        self.num_layers = _normalize_positive_int("num_layers", num_layers)
        self.unit_cell_atoms = _normalize_positive_int("unit_cell_atoms", unit_cell_atoms)
        self.rnn_type = _normalize_conv_rnn_type(type)
        self.kernel_size = _normalize_positive_int("kernel_size", kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve crystal shape")
        self.periodic_padding = bool(periodic_padding)
        self.target_mode = _normalize_target_mode(target_mode)
        self.delta_loss_weight = _normalize_delta_loss_weight(delta_loss_weight)
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(delta_loss_epsilon)
        self.residual_output = bool(residual_output)
        self.channels = self.unit_cell_atoms * 3
        self.model = _ConvRecurrentNet3D(
            input_channels=self.channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            rnn_type=self.rnn_type,
            padding_mode="circular" if self.periodic_padding else "zeros",
        )
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 32
        self.train_count = 200

    def reset(self):
        """Reinitialize the underlying neural network."""
        self.model = _ConvRecurrentNet3D(
            input_channels=self.channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            rnn_type=self.rnn_type,
            padding_mode="circular" if self.periodic_padding else "zeros",
        )

    def _prepare_targets(self, X_blocks, y_blocks):
        """Apply the selected target convention before channel conversion."""
        if self.target_mode == "delta":
            return y_blocks - X_blocks[:, -1]
        return y_blocks

    def train_crystal_blocks(self, X_blocks, y_blocks, data_len=0.5):
        """Train from crystal-shaped block samples.

        Args:
            X_blocks: Shape
                ``(n_samples, sequence_length, nx, ny, nz, unit_cell_atoms, 3)``.
            y_blocks: Shape ``(n_samples, nx, ny, nz, unit_cell_atoms, 3)``.
            data_len: Fraction of a random consecutive sample window used for
                this training run, matching the legacy RNN workflow.
        """
        X_blocks = np.asarray(X_blocks, dtype=np.float32)
        y_blocks = np.asarray(y_blocks, dtype=np.float32)
        if X_blocks.ndim != 7:
            raise ValueError("X_blocks must have shape (n, sequence, nx, ny, nz, unit_cell_atoms, 3)")
        if y_blocks.ndim != 6:
            raise ValueError("y_blocks must have shape (n, nx, ny, nz, unit_cell_atoms, 3)")
        if X_blocks.shape[0] != y_blocks.shape[0]:
            raise ValueError("X_blocks and y_blocks must contain the same number of samples")
        if X_blocks.shape[5] != self.unit_cell_atoms or y_blocks.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match the training data")

        self.train_count = int(data_len * X_blocks.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_blocks.shape[0]:
            start_index = 0
        else:
            start_index = np.random.randint(low=0, high=X_blocks.shape[0] - self.train_count)

        X_train = X_blocks[start_index : start_index + self.train_count]
        y_train = self._prepare_targets(X_train, y_blocks[start_index : start_index + self.train_count])
        X_channels = _crystal_to_channels(X_train)
        y_channels = _crystal_to_channels(y_train)
        dataset = TensorDataset(X_channels, y_channels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0.0
            batch_count = 0
            for x_train, y_train in loader:
                predicted = self.model(x_train)
                if self.residual_output:
                    predicted = x_train[:, -1] + predicted
                loss = loss_func(predicted, y_train)
                if self.target_mode == "absolute_delta" and self.delta_loss_weight > 0:
                    last_input = x_train[:, -1]
                    true_delta = y_train - last_input
                    pred_delta = predicted - last_input
                    delta_scale = torch.sqrt(torch.mean(true_delta**2, dim=(1, 2, 3, 4), keepdim=True)).clamp_min(
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

    def run_crystal(self, count_steps, init_displacements):
        """Autoregressively predict a full crystal without blockwise stitching."""
        if count_steps <= 0:
            raise ValueError("count_steps must be positive")
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if init_displacements.ndim != 6:
            raise ValueError("init_displacements must have shape (sequence, nx, ny, nz, unit_cell_atoms, 3)")
        if init_displacements.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match init_displacements")

        self.model.eval()
        x = _crystal_to_channels(init_displacements).unsqueeze(0).clone()
        predictions = []

        with torch.no_grad():
            for _ in range(count_steps):
                y = self.model(x).squeeze(0)
                if self.residual_output or self.target_mode == "delta":
                    y = x[0, -1] + y
                predictions.append(_channels_to_crystal(y, self.unit_cell_atoms).detach().cpu().numpy())
                x = torch.cat([x[:, 1:], y.reshape(1, 1, *y.shape)], dim=1)

        return np.asarray(predictions, dtype=np.float32)
