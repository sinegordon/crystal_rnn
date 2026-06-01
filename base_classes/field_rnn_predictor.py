"""Convolutional encoder plus shared temporal RNN for crystal fields.

This architecture keeps recurrent processing over time, but removes dense
position-specific channels over the crystal block.  Each frame is encoded by
periodic 3D convolutions, then the same temporal RNN is applied independently
to every unit-cell location, and a convolutional decoder returns the next
crystal displacement field.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from .conv_crystal_predictor import _channels_to_crystal, _crystal_to_channels
from .crystal_predictor import (
    _normalize_delta_loss_epsilon,
    _normalize_delta_loss_weight,
    _normalize_target_mode,
)


def _normalize_positive_int(name, value):
    """Validate a positive integer model parameter."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_field_rnn_type(rnn_type):
    """Validate the temporal recurrent block type."""
    rnn_type = str(rnn_type).upper()
    if rnn_type not in {"RNN", "GRU", "LSTM"}:
        raise ValueError("type must be RNN, GRU, or LSTM")
    return rnn_type


def _make_activation(name):
    """Create a small nonlinearity module by name."""
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError("activation must be 'elu', 'relu', or 'gelu'")


def _normalize_acceleration_normalization(mode):
    """Validate acceleration target normalization mode."""
    mode = str(mode).lower()
    if mode not in {"none", "global", "channel"}:
        raise ValueError("acceleration_normalization must be 'none', 'global', or 'channel'")
    return mode


def _normalize_loss_region(region):
    """Validate which spatial part of a block contributes to training loss."""
    region = str(region).lower()
    if region not in {"all", "center_cell"}:
        raise ValueError("loss_region must be 'all' or 'center_cell'")
    return region


def _normalize_input_transform(transform):
    """Validate the displacement convention used before the neural network."""
    transform = str(transform).lower()
    if transform not in {
        "absolute",
        "center_mean_relative",
        "same_atom_relative",
        "absolute_plus_same_atom_relative",
    }:
        raise ValueError(
            "input_transform must be 'absolute', 'center_mean_relative', "
            "'same_atom_relative', or 'absolute_plus_same_atom_relative'"
        )
    return transform


def _normalize_positive_float(name, value):
    """Validate a positive floating-point model parameter."""
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_nonnegative_float(name, value):
    """Validate a non-negative floating-point model parameter."""
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _resolve_torch_device(device):
    """Return the torch device used for training and inference."""
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {device!r}, but CUDA is not available")
    if device.startswith("mps"):
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError(f"Requested device {device!r}, but MPS is not available")
    return torch.device(device)


def _target_to_next(raw_prediction, history, target_mode):
    """Convert raw network output channels into absolute next-frame channels."""
    if target_mode == "delta":
        return history[:, -1] + raw_prediction
    if target_mode in {"acceleration", "verlet"}:
        return 2 * history[:, -1] - history[:, -2] + raw_prediction
    return raw_prediction


class _FieldRNN3D(nn.Module):
    """Encode crystal frames, run shared temporal RNN per cell, then decode."""

    def __init__(
        self,
        input_channels,
        encoder_channels,
        rnn_hidden_size,
        rnn_layers,
        conv_layers,
        kernel_size,
        rnn_type,
        bidirectional,
        padding_mode,
        activation,
        output_channels=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels if output_channels is None else output_channels
        self.encoder_channels = encoder_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.rnn_type = _normalize_field_rnn_type(rnn_type)
        self.bidirectional = bool(bidirectional)
        self.padding_mode = padding_mode

        padding = kernel_size // 2
        encoder = []
        current_channels = input_channels
        for _ in range(conv_layers):
            encoder.append(
                nn.Conv3d(
                    current_channels,
                    encoder_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            )
            encoder.append(_make_activation(activation))
            current_channels = encoder_channels
        self.encoder = nn.Sequential(*encoder)

        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=encoder_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        decoder_input_channels = rnn_hidden_size * (2 if self.bidirectional else 1)

        self.decoder = nn.Sequential(
            nn.Conv3d(
                decoder_input_channels,
                rnn_hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            _make_activation(activation),
            nn.Conv3d(rnn_hidden_size, self.output_channels, kernel_size=1),
        )

    def forward(self, x):
        """Predict one field from ``(batch, sequence, channels, nx, ny, nz)``."""
        batch_size, sequence_length, channels, nx, ny, nz = x.shape
        encoded = self.encoder(x.reshape(batch_size * sequence_length, channels, nx, ny, nz))
        encoded = encoded.reshape(batch_size, sequence_length, self.encoder_channels, nx, ny, nz)
        encoded = encoded.permute(0, 3, 4, 5, 1, 2).reshape(
            batch_size * nx * ny * nz,
            sequence_length,
            self.encoder_channels,
        )

        sequence_output, _ = self.rnn(encoded)
        last_output = sequence_output[:, -1]
        decoder_input_channels = self.rnn_hidden_size * (2 if self.bidirectional else 1)
        hidden_field = last_output.reshape(batch_size, nx, ny, nz, decoder_input_channels).permute(0, 4, 1, 2, 3)
        return self.decoder(hidden_field)


class CrystalFieldRNNNet:
    """Train and run a recurrent-convolutional operator on crystal fields.

    The network uses shared temporal recurrent weights for every unit-cell
    location.  Spatial coupling is handled by periodic 3D convolutional encoder
    and decoder layers, so the model can be applied to larger rectangular
    crystals without block stitching.
    """

    def __init__(
        self,
        encoder_channels,
        rnn_hidden_size,
        rnn_layers,
        unit_cell_atoms,
        type="GRU",
        conv_layers=1,
        kernel_size=3,
        bidirectional=False,
        periodic_padding=True,
        activation="elu",
        target_mode="absolute_delta",
        delta_loss_weight=1.0,
        delta_loss_epsilon=1e-6,
        acceleration_normalization="none",
        acceleration_normalization_epsilon=1e-12,
        loss_region="all",
        cyclic_shift_augmentation=False,
        input_transform="absolute",
        input_relative_scale=1.0,
        force_balance_loss_weight=0.0,
        acceleration_rms_loss_weight=0.0,
        velocity_rms_loss_weight=0.0,
        rms_loss_epsilon=1e-12,
        low_q_stiffness_loss_weight=0.0,
        low_q_stiffness_max_shell=1,
        low_q_stiffness_epsilon=1e-8,
        device="auto",
    ):
        self.encoder_channels = _normalize_positive_int("encoder_channels", encoder_channels)
        self.rnn_hidden_size = _normalize_positive_int("rnn_hidden_size", rnn_hidden_size)
        self.rnn_layers = _normalize_positive_int("rnn_layers", rnn_layers)
        self.conv_layers = _normalize_positive_int("conv_layers", conv_layers)
        self.kernel_size = _normalize_positive_int("kernel_size", kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve crystal shape")
        self.unit_cell_atoms = _normalize_positive_int("unit_cell_atoms", unit_cell_atoms)
        self.rnn_type = _normalize_field_rnn_type(type)
        self.bidirectional = bool(bidirectional)
        self.periodic_padding = bool(periodic_padding)
        self.activation = str(activation).lower()
        self.target_mode = _normalize_target_mode(target_mode)
        self.delta_loss_weight = _normalize_delta_loss_weight(delta_loss_weight)
        self.delta_loss_epsilon = _normalize_delta_loss_epsilon(delta_loss_epsilon)
        self.acceleration_normalization = _normalize_acceleration_normalization(acceleration_normalization)
        self.acceleration_normalization_epsilon = _normalize_positive_float(
            "acceleration_normalization_epsilon",
            acceleration_normalization_epsilon,
        )
        self.loss_region = _normalize_loss_region(loss_region)
        self.cyclic_shift_augmentation = bool(cyclic_shift_augmentation)
        self.input_transform = _normalize_input_transform(input_transform)
        self.input_relative_scale = _normalize_nonnegative_float("input_relative_scale", input_relative_scale)
        self.force_balance_loss_weight = _normalize_nonnegative_float(
            "force_balance_loss_weight",
            force_balance_loss_weight,
        )
        self.acceleration_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_rms_loss_weight",
            acceleration_rms_loss_weight,
        )
        self.velocity_rms_loss_weight = _normalize_nonnegative_float(
            "velocity_rms_loss_weight",
            velocity_rms_loss_weight,
        )
        self.rms_loss_epsilon = _normalize_positive_float("rms_loss_epsilon", rms_loss_epsilon)
        self.low_q_stiffness_loss_weight = _normalize_nonnegative_float(
            "low_q_stiffness_loss_weight",
            low_q_stiffness_loss_weight,
        )
        self.low_q_stiffness_max_shell = _normalize_positive_int(
            "low_q_stiffness_max_shell",
            low_q_stiffness_max_shell,
        )
        self.low_q_stiffness_epsilon = _normalize_positive_float(
            "low_q_stiffness_epsilon",
            low_q_stiffness_epsilon,
        )
        if self.acceleration_normalization != "none" and self.target_mode != "acceleration":
            raise ValueError("acceleration_normalization is only supported with target_mode='acceleration'")
        self.acceleration_mean = np.asarray(0.0, dtype=np.float32)
        self.acceleration_std = np.asarray(1.0, dtype=np.float32)
        self.device = str(_resolve_torch_device(device))
        self.channels = self.unit_cell_atoms * 3
        self.input_channels = self._transformed_channel_count(self.channels)
        self.model = _FieldRNN3D(
            input_channels=self.input_channels,
            encoder_channels=self.encoder_channels,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_layers=self.rnn_layers,
            conv_layers=self.conv_layers,
            kernel_size=self.kernel_size,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            padding_mode="circular" if self.periodic_padding else "zeros",
            activation=self.activation,
            output_channels=self.channels,
        ).to(self.torch_device)
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 32
        self.train_count = 200
        self.default_periodic = True

    @property
    def torch_device(self):
        """Return the current torch device, defaulting old saved models to CPU."""
        return _resolve_torch_device(getattr(self, "device", "cpu"))

    def to(self, device):
        """Move the underlying PyTorch model to a new device."""
        self.device = str(_resolve_torch_device(device))
        self.model.to(self.torch_device)
        return self

    def _set_acceleration_normalization_stats(self, acceleration_channels):
        """Compute and store normalization statistics for acceleration targets."""
        if not (self.cyclic_shift_augmentation and self.loss_region == "center_cell"):
            acceleration_channels = self._select_loss_region(acceleration_channels)
        if self.acceleration_normalization == "none":
            self.acceleration_mean = np.asarray(0.0, dtype=np.float32)
            self.acceleration_std = np.asarray(1.0, dtype=np.float32)
            return

        if self.acceleration_normalization == "global":
            mean = torch.mean(acceleration_channels)
            std = torch.sqrt(torch.mean((acceleration_channels - mean) ** 2)).clamp_min(
                self.acceleration_normalization_epsilon
            )
        else:
            mean = torch.mean(acceleration_channels, dim=(0, 2, 3, 4))
            std = torch.sqrt(
                torch.mean((acceleration_channels - mean.reshape(1, -1, 1, 1, 1)) ** 2, dim=(0, 2, 3, 4))
            ).clamp_min(self.acceleration_normalization_epsilon)

        self.acceleration_mean = mean.detach().cpu().numpy().astype(np.float32)
        self.acceleration_std = std.detach().cpu().numpy().astype(np.float32)

    def _acceleration_stats_tensors(self, reference):
        """Return normalization stats broadcastable to ``(batch, channels, nx, ny, nz)``."""
        mean = torch.as_tensor(self.acceleration_mean, dtype=reference.dtype, device=reference.device)
        std = torch.as_tensor(self.acceleration_std, dtype=reference.dtype, device=reference.device).clamp_min(
            self.acceleration_normalization_epsilon
        )
        if mean.ndim == 1:
            mean = mean.reshape(1, -1, 1, 1, 1)
            std = std.reshape(1, -1, 1, 1, 1)
        return mean, std

    def _normalize_acceleration_target(self, acceleration_channels):
        """Normalize acceleration targets using stored training statistics."""
        if self.acceleration_normalization == "none":
            return acceleration_channels
        mean, std = self._acceleration_stats_tensors(acceleration_channels)
        return (acceleration_channels - mean) / std

    def _denormalize_acceleration_prediction(self, raw_prediction):
        """Convert normalized acceleration predictions back to physical scale."""
        if self.acceleration_normalization == "none":
            return raw_prediction
        mean, std = self._acceleration_stats_tensors(raw_prediction)
        return raw_prediction * std + mean

    def _center_acceleration_from_patch_tensor(self, patches):
        """Return physical central-cell accelerations from differentiable patches.

        ``patches`` must use crystal layout
        ``(batch, sequence, nx, ny, nz, unit_cell_atoms, 3)``.  The method is
        intentionally torch-only so inference-time Jacobian corrections can
        differentiate the central acceleration with respect to the local
        displacement history.
        """
        if patches.ndim != 7:
            raise ValueError("patches must have shape (batch, sequence, nx, ny, nz, atoms, 3)")
        if patches.shape[-2] != self.unit_cell_atoms or patches.shape[-1] != 3:
            raise ValueError("Patch atom/coordinate dimensions do not match model metadata")
        if patches.shape[1] < 2:
            raise ValueError("Acceleration extraction requires at least two history frames")

        history_channels = _crystal_to_channels(patches)
        raw_prediction = self.model(self._transform_input_channels(history_channels))
        acceleration_channels = self._prediction_to_physical_acceleration(raw_prediction, history_channels)
        acceleration = _channels_to_crystal(acceleration_channels, self.unit_cell_atoms)
        cx, cy, cz = (int(dim) // 2 for dim in patches.shape[2:5])
        return acceleration[:, cx, cy, cz]

    def reset(self):
        """Reinitialize the underlying neural network."""
        self.input_channels = self._transformed_channel_count(self.channels)
        self.model = _FieldRNN3D(
            input_channels=self.input_channels,
            encoder_channels=self.encoder_channels,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_layers=self.rnn_layers,
            conv_layers=self.conv_layers,
            kernel_size=self.kernel_size,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            padding_mode="circular" if self.periodic_padding else "zeros",
            activation=self.activation,
            output_channels=self.channels,
        ).to(self.torch_device)

    def _transformed_channel_count(self, base_channels):
        """Return network input channels after the configured transform."""
        input_transform = _normalize_input_transform(getattr(self, "input_transform", "absolute"))
        if input_transform == "absolute_plus_same_atom_relative":
            return int(base_channels) * 2
        return int(base_channels)

    def _select_loss_region(self, tensor):
        """Select the spatial region that contributes to supervised loss."""
        if getattr(self, "loss_region", "all") == "all":
            return tensor
        nx, ny, nz = tensor.shape[-3:]
        if nx % 2 == 0 or ny % 2 == 0 or nz % 2 == 0:
            raise ValueError("loss_region='center_cell' requires odd spatial block sizes")
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        return tensor[..., cx : cx + 1, cy : cy + 1, cz : cz + 1]

    def _transform_input_channels(self, history_channels):
        """Apply the configured input displacement convention."""
        input_transform = _normalize_input_transform(getattr(self, "input_transform", "absolute"))
        if input_transform == "absolute":
            return history_channels

        channels, nx, ny, nz = history_channels.shape[-4:]
        if channels != self.channels:
            raise ValueError("Input channel count does not match model metadata")
        if nx % 2 == 0 or ny % 2 == 0 or nz % 2 == 0:
            raise ValueError(f"{input_transform} requires odd spatial block sizes")

        cx, cy, cz = nx // 2, ny // 2, nz // 2
        leading_shape = history_channels.shape[:-4]
        crystal = history_channels.reshape(*leading_shape, self.unit_cell_atoms, 3, nx, ny, nz)

        if input_transform == "center_mean_relative":
            center_mean = crystal[..., cx, cy, cz].mean(dim=-2)
            center_mean = center_mean.reshape(*leading_shape, 1, 3, 1, 1, 1)
            return (crystal - center_mean).reshape_as(history_channels)

        if input_transform == "same_atom_relative":
            center_atom_displacement = crystal[..., cx, cy, cz].reshape(*leading_shape, self.unit_cell_atoms, 3, 1, 1, 1)
            return (crystal - center_atom_displacement).reshape_as(history_channels)

        if input_transform == "absolute_plus_same_atom_relative":
            center_atom_displacement = crystal[..., cx, cy, cz].reshape(*leading_shape, self.unit_cell_atoms, 3, 1, 1, 1)
            relative = (crystal - center_atom_displacement).reshape_as(history_channels)
            relative = relative * float(getattr(self, "input_relative_scale", 1.0))
            return torch.cat([history_channels, relative], dim=-4)

        raise ValueError(f"Unsupported input_transform={input_transform!r}")

    def _augment_cyclic_shift_batch(self, history_channels, target_channels):
        """Randomly roll a training batch over crystal axes before center loss."""
        if not getattr(self, "cyclic_shift_augmentation", False):
            return history_channels, target_channels

        nx, ny, nz = history_channels.shape[-3:]
        shifts = (
            int(torch.randint(0, nx, (1,), device=history_channels.device).item()),
            int(torch.randint(0, ny, (1,), device=history_channels.device).item()),
            int(torch.randint(0, nz, (1,), device=history_channels.device).item()),
        )
        if shifts == (0, 0, 0):
            return history_channels, target_channels

        history_channels = torch.roll(history_channels, shifts=shifts, dims=(-3, -2, -1))
        target_channels = torch.roll(target_channels, shifts=shifts, dims=(-3, -2, -1))
        return history_channels, target_channels

    def _prepare_targets(self, X_channels, y_channels):
        """Apply the selected target convention in channel layout."""
        if self.target_mode == "delta":
            return y_channels - X_channels[:, -1]
        if self.target_mode == "acceleration":
            if X_channels.shape[1] < 2:
                raise ValueError("target_mode='acceleration' requires at least two input history frames")
            return y_channels - 2 * X_channels[:, -1] + X_channels[:, -2]
        if self.target_mode == "verlet" and X_channels.shape[1] < 2:
            raise ValueError("target_mode='verlet' requires at least two input history frames")
        return y_channels

    def _loss(self, raw_prediction, target_channels, history_channels, loss_func):
        """Return position and optional normalized-delta losses."""
        if self.target_mode == "verlet":
            train_prediction = _target_to_next(raw_prediction, history_channels, self.target_mode)
        else:
            train_prediction = raw_prediction

        train_prediction_loss = self._select_loss_region(train_prediction)
        target_channels_loss = self._select_loss_region(target_channels)
        history_channels_loss = self._select_loss_region(history_channels)
        loss = loss_func(train_prediction_loss, target_channels_loss)
        if self.target_mode in {"absolute_delta", "verlet"} and self.delta_loss_weight > 0:
            last_input = history_channels_loss[:, -1]
            true_delta = target_channels_loss - last_input
            pred_delta = train_prediction_loss - last_input
            delta_scale = torch.sqrt(torch.mean(true_delta**2, dim=(1, 2, 3, 4), keepdim=True)).clamp_min(
                self.delta_loss_epsilon
            )
            loss = loss + self.delta_loss_weight * loss_func(pred_delta / delta_scale, true_delta / delta_scale)
        if getattr(self, "force_balance_loss_weight", 0.0) > 0:
            pred_acceleration = self._prediction_to_physical_acceleration(raw_prediction, history_channels)
            loss = loss + self.force_balance_loss_weight * self._force_balance_loss(pred_acceleration, loss_func)
        if getattr(self, "acceleration_rms_loss_weight", 0.0) > 0 or getattr(self, "velocity_rms_loss_weight", 0.0) > 0:
            pred_acceleration = self._prediction_to_physical_acceleration(raw_prediction, history_channels)
            true_acceleration = self._target_to_physical_acceleration(target_channels, history_channels)
            if getattr(self, "acceleration_rms_loss_weight", 0.0) > 0:
                loss = loss + self.acceleration_rms_loss_weight * self._acceleration_rms_loss(
                    pred_acceleration,
                    true_acceleration,
                )
            if getattr(self, "velocity_rms_loss_weight", 0.0) > 0:
                loss = loss + self.velocity_rms_loss_weight * self._velocity_rms_loss(
                    pred_acceleration,
                    true_acceleration,
                    history_channels,
                )
        if getattr(self, "low_q_stiffness_loss_weight", 0.0) > 0:
            pred_acceleration = self._prediction_to_physical_acceleration(raw_prediction, history_channels)
            true_acceleration = self._target_to_physical_acceleration(target_channels, history_channels)
            loss = loss + self.low_q_stiffness_loss_weight * self._low_q_stiffness_loss(
                pred_acceleration,
                true_acceleration,
                history_channels,
            )
        return loss

    def _prediction_to_physical_acceleration(self, raw_prediction, history_channels):
        """Convert the current prediction into physical acceleration channels."""
        if history_channels.shape[1] < 2:
            raise ValueError("Acceleration-based losses require at least two input history frames")
        if self.target_mode == "acceleration":
            return self._denormalize_acceleration_prediction(raw_prediction)
        if self.target_mode == "verlet":
            return raw_prediction
        pred_next = _target_to_next(raw_prediction, history_channels, self.target_mode)
        return pred_next - 2 * history_channels[:, -1] + history_channels[:, -2]

    def _target_to_physical_acceleration(self, target_channels, history_channels):
        """Convert the supervised target into physical acceleration channels."""
        if history_channels.shape[1] < 2:
            raise ValueError("Acceleration-based losses require at least two input history frames")
        if self.target_mode == "acceleration":
            if self.acceleration_normalization != "none":
                return self._denormalize_acceleration_prediction(target_channels)
            return target_channels
        if self.target_mode == "verlet":
            return target_channels - 2 * history_channels[:, -1] + history_channels[:, -2]
        if self.target_mode == "delta":
            target_next = history_channels[:, -1] + target_channels
        else:
            target_next = target_channels
        return target_next - 2 * history_channels[:, -1] + history_channels[:, -2]

    def _region_rms(self, values):
        """Return per-sample RMS over the supervised region in physical units."""
        selected = self._select_loss_region(values)
        return torch.sqrt(torch.mean(selected**2, dim=(1, 2, 3, 4)).clamp_min(0.0))

    def _log_rms_ratio_loss(self, predicted, reference):
        """Penalize mismatch of RMS scale without directly damping the target."""
        epsilon = float(getattr(self, "rms_loss_epsilon", 1e-12))
        predicted_rms = self._region_rms(predicted).clamp_min(epsilon)
        reference_rms = self._region_rms(reference).clamp_min(epsilon)
        log_ratio = torch.log(predicted_rms / reference_rms)
        return torch.mean(log_ratio**2)

    def _acceleration_rms_loss(self, pred_acceleration, true_acceleration):
        """Penalize the predicted/reference physical acceleration RMS ratio."""
        return self._log_rms_ratio_loss(pred_acceleration, true_acceleration)

    def _velocity_rms_loss(self, pred_acceleration, true_acceleration, history_channels):
        """Penalize the RMS ratio of the next-step velocity increment."""
        previous_delta = history_channels[:, -1] - history_channels[:, -2]
        pred_next_delta = previous_delta + pred_acceleration
        true_next_delta = previous_delta + true_acceleration
        return self._log_rms_ratio_loss(pred_next_delta, true_next_delta)

    def _low_q_mode_mask(self, spatial_shape, device):
        """Return a mask for non-zero Fourier modes in the low-q shell."""
        nx, ny, nz = (int(value) for value in spatial_shape)
        kx = torch.fft.fftfreq(nx, d=1.0, device=device) * nx
        ky = torch.fft.fftfreq(ny, d=1.0, device=device) * ny
        kz = torch.fft.fftfreq(nz, d=1.0, device=device) * nz
        q2 = (
            kx.reshape(nx, 1, 1) ** 2
            + ky.reshape(1, ny, 1) ** 2
            + kz.reshape(1, 1, nz) ** 2
        )
        max_shell = float(getattr(self, "low_q_stiffness_max_shell", 1))
        return (q2 > 0) & (q2 <= max_shell)

    def _low_q_stiffness_loss(self, pred_acceleration, true_acceleration, history_channels):
        """Match low-q acceleration response per displacement amplitude.

        For each selected Fourier mode, the acceleration spectra are divided by
        the RMS displacement amplitude of the same mode.  This compares the
        effective long-wavelength stiffness without directly suppressing the
        displacement amplitude itself.
        """
        displacement = history_channels[:, -1]
        batch_size, channels, nx, ny, nz = displacement.shape
        mode_mask = self._low_q_mode_mask((nx, ny, nz), displacement.device).reshape(-1)
        if not torch.any(mode_mask):
            return pred_acceleration.new_zeros(())

        fft_dims = (-3, -2, -1)
        displacement_modes = torch.fft.fftn(displacement, dim=fft_dims, norm="ortho").reshape(
            batch_size,
            channels,
            -1,
        )[:, :, mode_mask]
        pred_modes = torch.fft.fftn(pred_acceleration, dim=fft_dims, norm="ortho").reshape(
            batch_size,
            channels,
            -1,
        )[:, :, mode_mask]
        true_modes = torch.fft.fftn(true_acceleration, dim=fft_dims, norm="ortho").reshape(
            batch_size,
            channels,
            -1,
        )[:, :, mode_mask]

        epsilon = float(getattr(self, "low_q_stiffness_epsilon", 1e-8))
        displacement_scale = torch.sqrt(torch.mean(torch.abs(displacement_modes) ** 2, dim=1, keepdim=True)).clamp_min(
            epsilon
        )
        pred_response = pred_modes / displacement_scale
        true_response = true_modes / displacement_scale
        response_scale = torch.sqrt(torch.mean(torch.abs(true_response) ** 2, dim=(1, 2), keepdim=True)).clamp_min(
            epsilon
        )
        normalized_error = (pred_response - true_response) / response_scale
        return torch.mean(torch.abs(normalized_error) ** 2)

    def _force_balance_loss(self, acceleration_channels, loss_func):
        """Penalize systematic center-of-mass acceleration in the supervised region."""
        selected = self._select_loss_region(acceleration_channels)
        batch_size, channels, nx, ny, nz = selected.shape
        if channels != self.channels:
            raise ValueError("Acceleration channel count does not match model metadata")
        acceleration = selected.reshape(batch_size, self.unit_cell_atoms, 3, nx, ny, nz)
        mean_acceleration = acceleration.mean(dim=1)
        scale = self._force_balance_scale(acceleration_channels)
        return loss_func(mean_acceleration / scale, torch.zeros_like(mean_acceleration))

    def _force_balance_scale(self, reference):
        """Return a scalar acceleration scale for force-balance regularization."""
        if self.target_mode == "acceleration" and self.acceleration_normalization != "none":
            _, std = self._acceleration_stats_tensors(reference)
            return torch.sqrt(torch.mean(std**2)).clamp_min(self.acceleration_normalization_epsilon)
        return torch.as_tensor(1.0, dtype=reference.dtype, device=reference.device)

    def train_crystal_blocks(self, X_blocks, y_blocks, data_len=0.5):
        """Train from crystal-shaped block samples."""
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
        y_train = y_blocks[start_index : start_index + self.train_count]
        X_channels = _crystal_to_channels(X_train)
        y_channels = _crystal_to_channels(y_train)
        X_channels = X_channels.to(self.torch_device)
        y_channels = y_channels.to(self.torch_device)
        target_channels = self._prepare_targets(X_channels, y_channels)
        if self.target_mode == "acceleration":
            self._set_acceleration_normalization_stats(target_channels)
            target_channels = self._normalize_acceleration_target(target_channels)
        dataset = TensorDataset(X_channels, target_channels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0.0
            batch_count = 0
            for x_train, y_train in loader:
                x_train, y_train = self._augment_cyclic_shift_batch(x_train, y_train)
                raw_prediction = self.model(self._transform_input_channels(x_train))
                loss = self._loss(raw_prediction, y_train, x_train, loss_func)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1
                loss_mean = loss.item() / batch_count + (1 - 1 / batch_count) * loss_mean
            losses.append(loss_mean)

        return losses

    def run_crystal(self, count_steps, init_displacements):
        """Autoregressively predict a full crystal without block stitching."""
        if count_steps <= 0:
            raise ValueError("count_steps must be positive")
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if init_displacements.ndim != 6:
            raise ValueError("init_displacements must have shape (sequence, nx, ny, nz, unit_cell_atoms, 3)")
        if init_displacements.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match init_displacements")
        if self.target_mode in {"acceleration", "verlet"} and init_displacements.shape[0] < 2:
            raise ValueError(f"target_mode={self.target_mode!r} requires at least two input history frames")

        self.model.eval()
        self.model.to(self.torch_device)
        x = _crystal_to_channels(init_displacements).unsqueeze(0).clone().to(self.torch_device)
        predictions = []

        with torch.no_grad():
            for _ in range(count_steps):
                raw = self.model(self._transform_input_channels(x))
                if self.target_mode == "acceleration":
                    raw = self._denormalize_acceleration_prediction(raw)
                y = _target_to_next(raw, x, self.target_mode).squeeze(0)
                predictions.append(_channels_to_crystal(y, self.unit_cell_atoms).detach().cpu().numpy())
                x = torch.cat([x[:, 1:], y.reshape(1, 1, *y.shape)], dim=1)

        return np.asarray(predictions, dtype=np.float32)
