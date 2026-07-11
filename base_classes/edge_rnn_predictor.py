"""Temporal RNN predictor that uses local pair vectors instead of absolute coordinates."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from .field_rnn_predictor import _normalize_field_rnn_type, _resolve_torch_device


def _normalize_positive_int(name, value):
    """Validate a positive integer parameter."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_acceleration_normalization(mode):
    """Validate acceleration normalization mode."""
    mode = str(mode).lower()
    if mode not in {"none", "global", "channel"}:
        raise ValueError("acceleration_normalization must be 'none', 'global', or 'channel'")
    return mode


def _normalize_training_target(mode):
    """Validate edge-RNN acceleration target source."""
    mode = str(mode).lower()
    if mode not in {"displacement", "force"}:
        raise ValueError("training_target must be 'displacement' or 'force'")
    return mode


def _normalize_q_power_loss_mode(mode):
    """Validate how q-resolved acceleration power is penalized."""
    mode = str(mode).lower().replace("_", "-")
    aliases = {
        "match": "match",
        "positive": "positive-excess",
        "positive-excess": "positive-excess",
        "excess": "positive-excess",
    }
    if mode not in aliases:
        raise ValueError("q_power_loss_mode must be 'match' or 'positive-excess'")
    return aliases[mode]


def _normalize_rnn_readout_mode(mode):
    """Validate how the recurrent sequence is reduced to one feature vector."""
    mode = str(mode).lower().replace("_", "-")
    aliases = {
        "last": "last-output",
        "output": "last-output",
        "last-output": "last-output",
        "final": "final-hidden",
        "hidden": "final-hidden",
        "final-hidden": "final-hidden",
        "flat": "final-hidden",
        "old-flat": "final-hidden",
    }
    if mode not in aliases:
        raise ValueError("rnn_readout_mode must be 'last-output' or 'final-hidden'")
    return aliases[mode]


def _normalize_temporal_architecture(mode):
    """Validate how history frames are routed through the temporal core."""
    mode = str(mode).lower().replace("_", "-")
    aliases = {
        "stacked": "stacked",
        "standard": "stacked",
        "pytorch": "stacked",
        "frame": "frame-layered",
        "framelayered": "frame-layered",
        "frame-layered": "frame-layered",
        "frame-layer": "frame-layered",
        "per-frame": "frame-layered",
        "mlp": "mlp",
        "ffn": "mlp",
        "feed-forward": "mlp",
        "feedforward": "mlp",
        "dense": "mlp",
    }
    if mode not in aliases:
        raise ValueError("temporal_architecture must be 'stacked', 'frame-layered', or 'mlp'")
    return aliases[mode]


def _normalize_temporal_input_mode(mode):
    """Validate how raw history frames are converted before recurrent encoding."""
    mode = str(mode).lower().replace("_", "-")
    aliases = {
        "absolute": "absolute-pair",
        "absolute-pair": "absolute-pair",
        "pair": "absolute-pair",
        "default": "absolute-pair",
        "relative": "relative-to-first",
        "relative-first": "relative-to-first",
        "relative-to-first": "relative-to-first",
        "frame-relative": "relative-to-first",
        "ref-plus-delta": "ref-plus-delta",
        "reference-plus-delta": "ref-plus-delta",
        "ref-delta": "ref-plus-delta",
        "split-reference": "ref-plus-delta",
    }
    if mode not in aliases:
        raise ValueError("temporal_input_mode must be 'absolute-pair', 'relative-to-first', or 'ref-plus-delta'")
    return aliases[mode]


def _feature_channels_for_temporal_input_mode(mode):
    """Return the per-pair channel count for a temporal input mode."""
    return 6 if _normalize_temporal_input_mode(mode) == "ref-plus-delta" else 3


def _dynamic_channel_slice_for_temporal_input_mode(mode):
    """Return channels differentiated to produce pair forces."""
    return slice(3, 6) if _normalize_temporal_input_mode(mode) == "ref-plus-delta" else slice(0, 3)


def _select_rnn_readout(output, hidden, bidirectional, readout_mode):
    """Return either the last sequence output or the final hidden state readout."""
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    if readout_mode == "last-output":
        return output[:, -1]
    if readout_mode != "final-hidden":
        raise ValueError(f"Unsupported rnn_readout_mode: {readout_mode}")
    if bidirectional:
        return torch.cat((hidden[-2], hidden[-1]), dim=1)
    return hidden[-1]


class _FrameLayerTemporalEncoder(nn.Module):
    """Temporal encoder with one recurrent cell assigned to each history frame.

    Unlike ``nn.RNN(num_layers=N)``, which applies every layer to every time
    step, this encoder uses exactly one cell per frame:
    ``h_i = cell_i(x_i, h_{i-1})``.  Therefore ``sequence_length`` must match
    ``frame_layers``.  The optional backward branch mirrors the same rule from
    the last frame to the first frame.
    """

    def __init__(self, input_size, hidden_size, frame_layers, rnn_type, bidirectional):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.frame_layers = int(frame_layers)
        if self.frame_layers <= 0:
            raise ValueError("frame_layers must be positive")
        self.rnn_type = _normalize_field_rnn_type(rnn_type)
        self.bidirectional = bool(bidirectional)

        if self.rnn_type == "GRU":
            cell_cls = nn.GRUCell
        elif self.rnn_type == "LSTM":
            cell_cls = nn.LSTMCell
        else:
            cell_cls = nn.RNNCell

        self.forward_cells = nn.ModuleList(
            cell_cls(self.input_size, self.hidden_size) for _ in range(self.frame_layers)
        )
        if self.bidirectional:
            self.backward_cells = nn.ModuleList(
                cell_cls(self.input_size, self.hidden_size) for _ in range(self.frame_layers)
            )
            self.output_size = self.hidden_size * 2
        else:
            self.backward_cells = None
            self.output_size = self.hidden_size

    def _initial_state(self, batch_size, device, dtype):
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.rnn_type == "LSTM":
            cell = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            return hidden, cell
        return hidden

    def _run_cells(self, x, cells, reverse=False):
        state = self._initial_state(x.shape[0], x.device, x.dtype)
        indices = range(self.frame_layers - 1, -1, -1) if reverse else range(self.frame_layers)
        for frame_index in indices:
            state = cells[frame_index](x[:, frame_index, :], state)
        if self.rnn_type == "LSTM":
            return state[0]
        return state

    def forward(self, x):
        """Return the final frame-layered hidden readout."""
        if x.ndim != 3:
            raise ValueError("Frame-layered temporal encoder expects input shape (batch, sequence, features)")
        if x.shape[1] != self.frame_layers:
            raise ValueError("Frame-layered temporal encoder sequence length must equal rnn_layers")
        forward_hidden = self._run_cells(x, self.forward_cells, reverse=False)
        if not self.bidirectional:
            return forward_hidden
        backward_hidden = self._run_cells(x, self.backward_cells, reverse=True)
        return torch.cat((forward_hidden, backward_hidden), dim=1)


def _normalize_positive_float(name, value):
    """Validate a positive floating-point parameter."""
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_nonnegative_float(name, value):
    """Validate a non-negative floating-point parameter."""
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _minimum_image(delta, box_lengths):
    """Return minimum-image vectors for an orthorhombic cell."""
    box_lengths = np.asarray(box_lengths, dtype=np.float32)
    return delta - box_lengths * np.round(delta / box_lengths)


def _first_box_lengths(box_lengths):
    """Return one orthorhombic box-length vector."""
    box_lengths = np.asarray(box_lengths, dtype=np.float32)
    if box_lengths.ndim == 2:
        return box_lengths[0]
    if box_lengths.shape != (3,):
        raise ValueError("box_lengths must have shape (3,) or (frames, 3)")
    return box_lengths


def lattice_parameter_from_box(box_lengths, crystal_shape):
    """Estimate the cubic lattice parameter from box lengths and crystal shape."""
    box_lengths = _first_box_lengths(box_lengths)
    crystal_shape = np.asarray(crystal_shape, dtype=np.float32)
    values = box_lengths / crystal_shape
    return float(np.mean(values))


def build_edge_stencil(
    reference_positions,
    atom_order,
    box_lengths,
    neighbor_shells=2,
    cutoff_scale=1.05,
    shell_tolerance=0.08,
):
    """Build a fixed FCC neighbor stencil for central-cell atoms.

    The first two FCC coordination shells contain 12 and 6 neighbors.  The
    stencil is constructed from equilibrium positions but stores only relative
    vectors, so inference never needs absolute atom coordinates as features.
    """
    reference_positions = np.asarray(reference_positions, dtype=np.float32)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    box_lengths = _first_box_lengths(box_lengths)
    if atom_order.ndim != 4:
        raise ValueError("atom_order must have shape (nx, ny, nz, unit_cell_atoms)")
    if tuple(atom_order.shape[:3]) != (3, 3, 3):
        raise ValueError("edge stencil currently expects 3x3x3 local blocks")
    if int(neighbor_shells) != 2:
        raise ValueError("the first implementation supports exactly two coordination shells")

    crystal_shape = np.asarray(atom_order.shape[:3], dtype=np.float32)
    lattice_parameter = lattice_parameter_from_box(box_lengths, crystal_shape)
    cutoff = float(cutoff_scale) * lattice_parameter
    center_cell = (1, 1, 1)
    unit_cell_atoms = int(atom_order.shape[3])
    neighbor_indices = []
    reference_vectors = []
    shell_ids = []

    for center_atom in range(unit_cell_atoms):
        center_flat = atom_order[center_cell + (center_atom,)]
        center_position = reference_positions[center_flat]
        rows = []
        for cell_index in np.ndindex(atom_order.shape[:3]):
            for neighbor_atom in range(unit_cell_atoms):
                if cell_index == center_cell and neighbor_atom == center_atom:
                    continue
                neighbor_flat = atom_order[cell_index + (neighbor_atom,)]
                vector = _minimum_image(reference_positions[neighbor_flat] - center_position, box_lengths)
                distance = float(np.linalg.norm(vector))
                if distance <= cutoff and distance > 1e-6:
                    shell = 1 if distance < (1.0 - shell_tolerance) * lattice_parameter else 2
                    rows.append((shell, distance, vector[0], vector[1], vector[2], cell_index, neighbor_atom, vector))
        rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        expected = 18
        if len(rows) != expected:
            raise ValueError(
                f"Expected {expected} neighbors for center atom {center_atom}, got {len(rows)}. "
                f"Try adjusting cutoff_scale."
            )
        neighbor_indices.append([(*row[5], row[6]) for row in rows])
        reference_vectors.append([row[7] for row in rows])
        shell_ids.append([row[0] for row in rows])

    return {
        "neighbor_indices": np.asarray(neighbor_indices, dtype=np.int64),
        "reference_vectors": np.asarray(reference_vectors, dtype=np.float32),
        "shell_ids": np.asarray(shell_ids, dtype=np.int64),
        "lattice_parameter": np.asarray(lattice_parameter, dtype=np.float32),
        "cutoff": np.asarray(cutoff, dtype=np.float32),
    }


def _extract_patch_batch(history, centers, patch_shape, periodic):
    """Extract local history patches centered at the requested crystal cells."""
    crystal_shape = tuple(int(dim) for dim in history.shape[1:4])
    radius = tuple(int(dim) // 2 for dim in patch_shape)
    patches = []
    for center in centers:
        axes = []
        for value, rad, size in zip(center, radius, crystal_shape):
            values = np.arange(value - rad, value + rad + 1, dtype=np.int64)
            if periodic:
                values %= size
            elif np.any((values < 0) | (values >= size)):
                raise ValueError("Patch crosses crystal boundary with periodic=False")
            axes.append(values)
        patches.append(history[(slice(None), *np.ix_(axes[0], axes[1], axes[2]), slice(None), slice(None))])
    return np.asarray(patches, dtype=np.float32)


def build_centers(crystal_shape, periodic=True):
    """Return all full-crystal cell centers for inference."""
    crystal_shape = tuple(int(value) for value in crystal_shape)
    if not periodic and any(value < 3 for value in crystal_shape):
        raise ValueError("non-periodic edge inference needs at least 3 cells per axis")
    if periodic:
        return [(ix, iy, iz) for ix in range(crystal_shape[0]) for iy in range(crystal_shape[1]) for iz in range(crystal_shape[2])]
    return [
        (ix, iy, iz)
        for ix in range(1, crystal_shape[0] - 1)
        for iy in range(1, crystal_shape[1] - 1)
        for iz in range(1, crystal_shape[2] - 1)
    ]


class _EdgeRNN(nn.Module):
    """Small temporal RNN that maps edge-vector histories to central accelerations."""

    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_layers,
        output_size,
        rnn_type,
        bidirectional,
        dropout=0.0,
        readout_mode="last-output",
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.rnn_layers = int(rnn_layers)
        self.output_size = int(output_size)
        self.rnn_type = _normalize_field_rnn_type(rnn_type)
        self.bidirectional = bool(bidirectional)
        self.readout_mode = _normalize_rnn_readout_mode(readout_mode)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if self.rnn_layers > 1 else 0.0,
        )
        recurrent_width = self.hidden_size * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(recurrent_width, recurrent_width),
            nn.ELU(inplace=True),
            nn.Linear(recurrent_width, self.output_size),
        )

    def forward(self, x):
        """Predict flattened central-cell accelerations from edge-vector histories."""
        output, hidden = self.rnn(x)
        readout_mode = getattr(self, "readout_mode", "last-output")
        readout = _select_rnn_readout(output, hidden, self.bidirectional, readout_mode)
        return self.head(readout)


class _OddPairForceRNN(nn.Module):
    """Shared edge RNN that predicts odd pair-acceleration contributions.

    The same recurrent network is evaluated on an oriented pair-vector history
    and on the reversed history.  The antisymmetric part is kept:

        f(edge) = 0.5 * (g(edge) - g(-edge))

    so the same pair observed in the opposite direction gives the opposite
    contribution by construction.
    """

    def __init__(self, input_size, hidden_size, rnn_layers, rnn_type, bidirectional, dropout=0.0, readout_mode="last-output"):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.rnn_layers = int(rnn_layers)
        self.rnn_type = _normalize_field_rnn_type(rnn_type)
        self.bidirectional = bool(bidirectional)
        self.readout_mode = _normalize_rnn_readout_mode(readout_mode)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if self.rnn_layers > 1 else 0.0,
        )
        recurrent_width = self.hidden_size * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(recurrent_width, recurrent_width),
            nn.ELU(inplace=True),
            nn.Linear(recurrent_width, 3),
        )

    def raw_forward(self, x):
        """Return unconstrained pair-vector outputs."""
        output, hidden = self.rnn(x)
        readout_mode = getattr(self, "readout_mode", "last-output")
        readout = _select_rnn_readout(output, hidden, self.bidirectional, readout_mode)
        return self.head(readout)

    def forward(self, x):
        """Return antisymmetric pair-vector outputs."""
        return 0.5 * (self.raw_forward(x) - self.raw_forward(-x))


class _EvenPairEnergyRNN(nn.Module):
    """Shared temporal encoder that predicts an even scalar pair potential.

    Forces are later obtained by differentiating this scalar with respect to
    the current pair vector.  The explicit even symmetrization keeps the energy
    invariant under reversing the oriented pair.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_layers,
        rnn_type,
        bidirectional,
        dropout=0.0,
        readout_mode="last-output",
        temporal_architecture="stacked",
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.rnn_layers = int(rnn_layers)
        self.rnn_type = _normalize_field_rnn_type(rnn_type)
        self.bidirectional = bool(bidirectional)
        self.readout_mode = _normalize_rnn_readout_mode(readout_mode)
        self.temporal_architecture = _normalize_temporal_architecture(temporal_architecture)
        if self.temporal_architecture == "mlp":
            self.rnn = None
            self.temporal_encoder = None
            self.mlp_sequence_length = self.rnn_layers
            encoder_input_width = self.input_size * self.mlp_sequence_length
            encoder_output_width = self.hidden_size * (2 if self.bidirectional else 1)
            self.mlp_encoder = nn.Sequential(
                nn.Linear(encoder_input_width, encoder_output_width),
                nn.ELU(inplace=True),
            )
        elif self.temporal_architecture == "frame-layered":
            self.rnn = None
            self.mlp_encoder = None
            self.temporal_encoder = _FrameLayerTemporalEncoder(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                frame_layers=self.rnn_layers,
                rnn_type=self.rnn_type,
                bidirectional=self.bidirectional,
            )
        else:
            rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[self.rnn_type]
            self.rnn = rnn_cls(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.rnn_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=float(dropout) if self.rnn_layers > 1 else 0.0,
            )
            self.mlp_encoder = None
            self.temporal_encoder = None
        recurrent_width = self.hidden_size * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(recurrent_width, recurrent_width),
            nn.ELU(inplace=True),
            nn.Linear(recurrent_width, 1),
        )

    def raw_forward(self, x):
        """Return unconstrained scalar pair energies."""
        temporal_architecture = getattr(self, "temporal_architecture", "stacked")
        if temporal_architecture == "mlp":
            if x.ndim != 3:
                raise ValueError("MLP temporal encoder expects input shape (batch, sequence, features)")
            if x.shape[1] != getattr(self, "mlp_sequence_length", self.rnn_layers):
                raise ValueError("MLP temporal encoder sequence length must equal rnn_layers")
            readout = self.mlp_encoder(x.reshape(x.shape[0], -1))
        elif temporal_architecture == "frame-layered":
            readout = self.temporal_encoder(x)
        else:
            output, hidden = self.rnn(x)
            readout_mode = getattr(self, "readout_mode", "last-output")
            readout = _select_rnn_readout(output, hidden, self.bidirectional, readout_mode)
        return self.head(readout).squeeze(-1)

    def forward(self, x):
        """Return orientation-even scalar pair energies."""
        return 0.5 * (self.raw_forward(x) + self.raw_forward(-x))


class CrystalEdgeRNNNet:
    """RNN acceleration model whose inputs are local interatomic pair vectors."""

    def __init__(
        self,
        reference_positions,
        atom_order,
        box_lengths,
        hidden_size,
        rnn_layers,
        type="GRU",
        bidirectional=True,
        neighbor_shells=2,
        cutoff_scale=1.05,
        acceleration_normalization="channel",
        rnn_readout_mode="last-output",
        temporal_input_mode="absolute-pair",
        device="auto",
    ):
        self.reference_positions = np.asarray(reference_positions, dtype=np.float32)
        self.atom_order = np.asarray(atom_order, dtype=np.int64)
        self.box_lengths = _first_box_lengths(box_lengths).astype(np.float32)
        self.hidden_size = _normalize_positive_int("hidden_size", hidden_size)
        self.rnn_layers = _normalize_positive_int("rnn_layers", rnn_layers)
        self.rnn_type = _normalize_field_rnn_type(type)
        self.bidirectional = bool(bidirectional)
        self.rnn_readout_mode = _normalize_rnn_readout_mode(rnn_readout_mode)
        self.neighbor_shells = _normalize_positive_int("neighbor_shells", neighbor_shells)
        self.cutoff_scale = _normalize_positive_float("cutoff_scale", cutoff_scale)
        self.acceleration_normalization = _normalize_acceleration_normalization(acceleration_normalization)
        self.torch_device = _resolve_torch_device(device)
        self.temporal_input_mode = _normalize_temporal_input_mode(temporal_input_mode)
        self.feature_channels = _feature_channels_for_temporal_input_mode(self.temporal_input_mode)
        self.unit_cell_atoms = int(self.atom_order.shape[3])
        self.patch_shape = (3, 3, 3)
        self.edge_stencil = build_edge_stencil(
            reference_positions=self.reference_positions,
            atom_order=self.atom_order,
            box_lengths=self.box_lengths,
            neighbor_shells=self.neighbor_shells,
            cutoff_scale=self.cutoff_scale,
        )
        self.neighbor_indices = self.edge_stencil["neighbor_indices"]
        self.reference_vectors = self.edge_stencil["reference_vectors"]
        self.shell_ids = self.edge_stencil["shell_ids"]
        self.lattice_parameter = float(self.edge_stencil["lattice_parameter"])
        self.neighbor_count = int(self.neighbor_indices.shape[1])
        self.input_size = self.unit_cell_atoms * self.neighbor_count * self.feature_channels
        self.output_size = self.unit_cell_atoms * 3
        self.model = _EdgeRNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            output_size=self.output_size,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            readout_mode=self.rnn_readout_mode,
        ).to(self.torch_device)
        self.acceleration_mean = np.asarray(0.0, dtype=np.float32)
        self.acceleration_std = np.asarray(1.0, dtype=np.float32)
        self.training_target = "displacement"
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 32
        self.train_count = 200
        self.displacement_moment_loss_weight = 0.0
        self.displacement_moment_mean_weight = 1.0
        self.displacement_moment_std_weight = 1.0
        self.displacement_moment_rms_weight = 0.0
        self.displacement_moment_component_weights = np.ones(3, dtype=np.float32)
        self.displacement_moment_loss_epsilon = 1e-12
        self.power_mean_loss_weight = 0.0
        self.power_mean_loss_epsilon = 1e-12
        self.acceleration_rms_loss_weight = 0.0
        self.acceleration_batch_rms_loss_weight = 0.0
        self.acceleration_tail_loss_weight = 0.0
        self.acceleration_over_rms_loss_weight = 0.0
        self.acceleration_under_rms_loss_weight = 0.0
        self.acceleration_over_rms_loss_power = 2.0
        self.acceleration_under_rms_loss_power = 2.0
        self.rms_loss_epsilon = 1e-12
        self.curl_loss_weight = 0.0
        self.curl_loss_sample_count = 4
        self.curl_loss_interval = 1
        self.curl_loss_epsilon = 1e-12
        self.reference_pressure_loss_weight = 0.0
        self.reference_pressure_target = 0.0
        self.reference_pressure_loss_scale = 1.0

    def _temporal_feature_patches(self, patches):
        """Return the history frames that should be encoded by the RNN.

        ``absolute-pair`` and ``ref-plus-delta`` keep the raw sequence length.
        The experimental ``relative-to-first`` mode uses exactly three raw
        frames but exposes only two recurrent steps: frame 1 and frame 2
        measured relative to frame 0.  The neighbor stencil is still chosen
        from the equilibrium geometry.
        """
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        if mode in {"absolute-pair", "ref-plus-delta"}:
            return patches
        if patches.shape[1] != 3:
            raise ValueError("temporal_input_mode='relative-to-first' expects exactly three history frames")
        return patches[:, 1:] - patches[:, :1]

    def _temporal_feature_patches_torch(self, patches):
        """Torch equivalent of ``_temporal_feature_patches``."""
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        if mode in {"absolute-pair", "ref-plus-delta"}:
            return patches
        if patches.shape[1] != 3:
            raise ValueError("temporal_input_mode='relative-to-first' expects exactly three history frames")
        return patches[:, 1:] - patches[:, :1]

    def to(self, device):
        """Move the underlying torch module to a device."""
        self.torch_device = _resolve_torch_device(device)
        self.model.to(self.torch_device)
        return self

    def edge_features_from_patches(self, patches):
        """Return normalized pair-vector features from local displacement patches."""
        patches = np.asarray(patches, dtype=np.float32)
        if patches.ndim != 7:
            raise ValueError("patches must have shape (batch, sequence, 3, 3, 3, atoms, 3)")
        patches = self._temporal_feature_patches(patches)
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        batch_size, sequence_length = patches.shape[:2]
        feature_channels = _feature_channels_for_temporal_input_mode(mode)
        features = np.empty(
            (batch_size, sequence_length, self.unit_cell_atoms, self.neighbor_count, feature_channels),
            dtype=np.float32,
        )
        center_displacements = patches[:, :, 1, 1, 1]
        for atom_index in range(self.unit_cell_atoms):
            center = center_displacements[:, :, atom_index, :]
            for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                neighbor = patches[(slice(None), slice(None), *tuple(local_index), slice(None))]
                vector = neighbor - center
                if mode == "absolute-pair":
                    vector = vector + self.reference_vectors[atom_index, neighbor_index]
                    features[:, :, atom_index, neighbor_index, :] = vector / self.lattice_parameter
                elif mode == "ref-plus-delta":
                    reference = np.broadcast_to(self.reference_vectors[atom_index, neighbor_index], vector.shape)
                    features[:, :, atom_index, neighbor_index, :3] = reference / self.lattice_parameter
                    features[:, :, atom_index, neighbor_index, 3:] = vector / self.lattice_parameter
                else:
                    features[:, :, atom_index, neighbor_index, :] = vector / self.lattice_parameter
        return features.reshape(batch_size, sequence_length, self.input_size)

    def _edge_features_from_patches_torch(self, patches):
        """Return differentiable edge features from torch displacement patches."""
        if patches.ndim != 7:
            raise ValueError("patches must have shape (batch, sequence, 3, 3, 3, atoms, 3)")
        patches = self._temporal_feature_patches_torch(patches)
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        features = []
        center_displacements = patches[:, :, 1, 1, 1]
        reference_vectors = torch.as_tensor(
            self.reference_vectors,
            dtype=patches.dtype,
            device=patches.device,
        )
        scale = torch.as_tensor(self.lattice_parameter, dtype=patches.dtype, device=patches.device)
        for atom_index in range(self.unit_cell_atoms):
            center = center_displacements[:, :, atom_index, :]
            for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                lx, ly, lz, neighbor_atom = (int(value) for value in local_index)
                neighbor = patches[:, :, lx, ly, lz, neighbor_atom, :]
                vector = neighbor - center
                if mode == "absolute-pair":
                    vector = vector + reference_vectors[atom_index, neighbor_index]
                    features.append(vector / scale)
                elif mode == "ref-plus-delta":
                    reference = reference_vectors[atom_index, neighbor_index].reshape(1, 1, 3).expand_as(vector)
                    features.append(torch.cat((reference / scale, vector / scale), dim=-1))
                else:
                    features.append(vector / scale)
        return torch.stack(features, dim=2).reshape(patches.shape[0], patches.shape[1], self.input_size)

    def _center_acceleration_from_patch_tensor(self, patches):
        """Return physical central-cell accelerations from differentiable patches."""
        features = self._edge_features_from_patches_torch(patches)
        raw = self.model(features)
        return self._denormalize_acceleration(raw).reshape(-1, self.unit_cell_atoms, 3)

    def _curl_loss(self, patch_batch):
        """Penalize the antisymmetric part of d a_center / d u_center.

        A force field derived from a potential has a symmetric local Jacobian.
        This regularizer is evaluated on a small mini-batch subset because it
        requires higher-order autograd through the recurrent model.
        """
        sample_count = min(int(self.curl_loss_sample_count), int(patch_batch.shape[0]))
        if sample_count <= 0:
            raise ValueError("curl_loss_sample_count must be positive")

        patches = patch_batch[:sample_count].detach().clone().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            acceleration = self._center_acceleration_from_patch_tensor(patches).reshape(sample_count, self.output_size)
            jacobian_rows = []
            for output_index in range(self.output_size):
                gradient = torch.autograd.grad(
                    acceleration[:, output_index].sum(),
                    patches,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                jacobian_rows.append(gradient[:, -1, 1, 1, 1].reshape(sample_count, self.output_size))

        jacobian = torch.stack(jacobian_rows, dim=1)
        antisymmetric = 0.5 * (jacobian - jacobian.transpose(1, 2))
        numerator = torch.sum(antisymmetric**2, dim=(1, 2))
        denominator = torch.sum(jacobian**2, dim=(1, 2)).clamp_min(float(self.curl_loss_epsilon))
        return torch.mean(numerator / denominator)

    def _target_acceleration(self, X_blocks, y_blocks):
        """Return central-cell acceleration targets from displacement blocks."""
        center = (1, 1, 1)
        return y_blocks[(slice(None), *center, slice(None), slice(None))] - 2.0 * X_blocks[
            (slice(None), -1, *center, slice(None), slice(None))
        ] + X_blocks[(slice(None), -2, *center, slice(None), slice(None))]

    def _target_force_acceleration(self, target_blocks):
        """Return central-cell force-derived discrete acceleration targets."""
        center = (1, 1, 1)
        return target_blocks[(slice(None), *center, slice(None), slice(None))]

    def _set_acceleration_normalization_stats(self, targets):
        """Store target normalization statistics."""
        targets = np.asarray(targets, dtype=np.float32)
        if self.acceleration_normalization == "none":
            self.acceleration_mean = np.asarray(0.0, dtype=np.float32)
            self.acceleration_std = np.asarray(1.0, dtype=np.float32)
        elif self.acceleration_normalization == "global":
            self.acceleration_mean = np.asarray(float(np.mean(targets)), dtype=np.float32)
            self.acceleration_std = np.asarray(float(np.std(targets)), dtype=np.float32)
        else:
            flat = targets.reshape(targets.shape[0], self.output_size)
            self.acceleration_mean = np.mean(flat, axis=0).astype(np.float32)
            self.acceleration_std = np.std(flat, axis=0).astype(np.float32)
        self.acceleration_std = np.maximum(self.acceleration_std, np.asarray(1e-12, dtype=np.float32))

    def _normalize_acceleration(self, targets):
        """Normalize acceleration targets."""
        flat = np.asarray(targets, dtype=np.float32).reshape(targets.shape[0], self.output_size)
        return ((flat - self.acceleration_mean) / self.acceleration_std).astype(np.float32)

    def _denormalize_acceleration(self, raw):
        """Convert raw torch predictions to physical accelerations."""
        mean = torch.as_tensor(self.acceleration_mean, dtype=raw.dtype, device=raw.device)
        std = torch.as_tensor(self.acceleration_std, dtype=raw.dtype, device=raw.device)
        return raw * std + mean

    def _center_displacements(self, blocks):
        """Return central-cell displacement vectors from 3x3x3 blocks."""
        center = (1, 1, 1)
        return np.asarray(blocks, dtype=np.float32)[(slice(None), *center, slice(None), slice(None))]

    def _displacement_moment_loss(self, predicted_next, true_next):
        """Match per-axis displacement moments for a mini-batch.

        The loss compares the distribution of predicted and reference central
        cell displacements across all samples and unit-cell atoms in the
        mini-batch.  It is intentionally low-dimensional: this acts like a
        differentiable histogram check for component mean and width without
        making training depend on fragile bin edges.
        """
        epsilon = float(self.displacement_moment_loss_epsilon)
        mean_weight = float(self.displacement_moment_mean_weight)
        std_weight = float(self.displacement_moment_std_weight)
        rms_weight = float(self.displacement_moment_rms_weight)
        component_weights = torch.as_tensor(
            self.displacement_moment_component_weights,
            dtype=predicted_next.dtype,
            device=predicted_next.device,
        )
        component_weights = component_weights / torch.mean(component_weights).clamp_min(epsilon)

        true_mean = torch.mean(true_next, dim=(0, 1))
        predicted_mean = torch.mean(predicted_next, dim=(0, 1))
        true_centered = true_next - true_mean.reshape(1, 1, 3)
        predicted_centered = predicted_next - predicted_mean.reshape(1, 1, 3)

        true_std = torch.sqrt(torch.mean(true_centered**2, dim=(0, 1)).clamp_min(epsilon))
        predicted_std = torch.sqrt(torch.mean(predicted_centered**2, dim=(0, 1)).clamp_min(epsilon))
        true_rms = torch.sqrt(torch.mean(true_next**2, dim=(0, 1)).clamp_min(epsilon))
        predicted_rms = torch.sqrt(torch.mean(predicted_next**2, dim=(0, 1)).clamp_min(epsilon))

        loss = predicted_next.new_zeros(())
        if mean_weight > 0:
            loss = loss + mean_weight * torch.mean(component_weights * ((predicted_mean - true_mean) / true_std) ** 2)
        if std_weight > 0:
            loss = loss + std_weight * torch.mean(component_weights * torch.log(predicted_std / true_std) ** 2)
        if rms_weight > 0:
            loss = loss + rms_weight * torch.mean(component_weights * torch.log(predicted_rms / true_rms) ** 2)
        return loss

    def _power_mean_loss(self, predicted_acceleration, true_acceleration, reference_velocity):
        """Match the mini-batch mean acceleration power against reference.

        The instantaneous sign of ``a dot v`` is not constrained: atoms may
        physically exchange kinetic and potential energy.  Only the average
        bias over the mini-batch is penalized, because a persistent positive
        mean is the drift mechanism observed in long ASE rollouts.
        """
        epsilon = float(self.power_mean_loss_epsilon)
        predicted_power = torch.sum(predicted_acceleration * reference_velocity, dim=-1)
        true_power = torch.sum(true_acceleration * reference_velocity, dim=-1)
        scale = torch.sqrt(torch.mean(true_power**2).clamp_min(epsilon))
        return ((torch.mean(predicted_power) - torch.mean(true_power)) / scale) ** 2

    def _predict_periodic_acceleration_field_from_blocks(self, X_blocks):
        """Predict one full periodic block by cyclically moving each cell to the center.

        Pair-force and pair-energy models supervise only the central unit cell of
        a 3x3x3 patch.  A q-resolved loss, however, needs a spatial acceleration
        field.  For the training 333 supercell we obtain that field by rolling
        the periodic block so every cell is evaluated once as the center.
        """
        if X_blocks.ndim != 7:
            raise ValueError("X_blocks must have shape (batch, sequence, nx, ny, nz, atoms, 3)")
        nx, ny, nz = (int(value) for value in X_blocks.shape[2:5])
        predicted = X_blocks.new_zeros((X_blocks.shape[0], nx, ny, nz, self.unit_cell_atoms, 3))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    rolled = torch.roll(X_blocks, shifts=(1 - ix, 1 - iy, 1 - iz), dims=(2, 3, 4))
                    predicted[:, ix, iy, iz] = self._center_acceleration_from_patch_tensor(rolled)
        return predicted

    def _q_power_shell_loss(self, predicted_acceleration, true_acceleration, reference_velocity):
        """Compare acceleration power in Fourier shells.

        This is a spatially resolved version of ``_power_mean_loss``.  It
        prevents positive power in one q-shell from being hidden by negative
        power in another shell when the total mini-batch average is formed.
        """
        if predicted_acceleration.ndim != 6:
            raise ValueError("predicted_acceleration must have shape (batch, nx, ny, nz, atoms, 3)")
        epsilon = float(self.q_power_loss_epsilon)
        margin = float(self.q_power_loss_margin)
        mode = _normalize_q_power_loss_mode(getattr(self, "q_power_loss_mode", "positive-excess"))
        nx, ny, nz = (int(value) for value in predicted_acceleration.shape[1:4])
        # MPS supports FFT only on the trailing dimensions for rank-6 tensors,
        # so keep atom/component channels before the spatial axes.
        pred_field = predicted_acceleration.permute(0, 4, 5, 1, 2, 3)
        true_field = true_acceleration.permute(0, 4, 5, 1, 2, 3)
        velocity_field = reference_velocity.permute(0, 4, 5, 1, 2, 3)
        fft_dims = (-3, -2, -1)

        pred_modes = torch.fft.fftn(pred_field, dim=fft_dims, norm="ortho")
        true_modes = torch.fft.fftn(true_field, dim=fft_dims, norm="ortho")
        velocity_modes = torch.fft.fftn(velocity_field, dim=fft_dims, norm="ortho")
        pred_power = torch.real(torch.sum(pred_modes * torch.conj(velocity_modes), dim=(1, 2)))
        true_power = torch.real(torch.sum(true_modes * torch.conj(velocity_modes), dim=(1, 2)))

        qx = torch.fft.fftfreq(nx, d=1.0, device=predicted_acceleration.device) * nx
        qy = torch.fft.fftfreq(ny, d=1.0, device=predicted_acceleration.device) * ny
        qz = torch.fft.fftfreq(nz, d=1.0, device=predicted_acceleration.device) * nz
        qx_grid, qy_grid, qz_grid = torch.meshgrid(qx, qy, qz, indexing="ij")
        shell_ids = torch.round(torch.sqrt(qx_grid**2 + qy_grid**2 + qz_grid**2) * 1000).to(torch.int64)

        loss = predicted_acceleration.new_zeros(())
        shell_count = 0
        for shell_id in torch.unique(shell_ids):
            if bool(getattr(self, "q_power_loss_exclude_q_zero", True)) and int(shell_id.item()) == 0:
                continue
            mask = shell_ids == shell_id
            pred_shell = pred_power[:, mask].sum(dim=1)
            true_shell = true_power[:, mask].sum(dim=1)
            pred_mean = torch.mean(pred_shell)
            true_mean = torch.mean(true_shell)
            scale = torch.sqrt(torch.mean(true_shell**2).clamp_min(epsilon))
            normalized_delta = (pred_mean - true_mean) / scale
            if mode == "match":
                loss = loss + normalized_delta**2
            else:
                loss = loss + torch.relu(normalized_delta - margin) ** 2
            shell_count += 1
        if shell_count == 0:
            return loss
        return loss / shell_count

    def _q_power_loss(self, X_blocks, y_blocks, displacement_y_blocks, training_target):
        """Return q-shell power loss for a small set of full periodic blocks."""
        sample_count = min(int(self.q_power_loss_sample_count), int(X_blocks.shape[0]))
        if sample_count <= 0:
            raise ValueError("q_power_loss_sample_count must be positive")
        X_sample = X_blocks[:sample_count]
        y_sample = y_blocks[:sample_count]
        displacement_y_sample = displacement_y_blocks[:sample_count]

        predicted_acceleration = self._predict_periodic_acceleration_field_from_blocks(X_sample)
        if training_target == "displacement":
            true_acceleration = y_sample - 2.0 * X_sample[:, -1] + X_sample[:, -2]
        else:
            true_acceleration = y_sample
        reference_velocity = 0.5 * (displacement_y_sample - X_sample[:, -2])
        return self._q_power_shell_loss(predicted_acceleration, true_acceleration, reference_velocity)

    def _acceleration_rms_loss(self, predicted_acceleration, true_acceleration):
        """Penalize mismatch of predicted/reference acceleration RMS scale."""
        epsilon = float(self.rms_loss_epsilon)
        predicted_rms = torch.sqrt(torch.mean(predicted_acceleration**2, dim=(1, 2)).clamp_min(epsilon))
        true_rms = torch.sqrt(torch.mean(true_acceleration**2, dim=(1, 2)).clamp_min(epsilon))
        return torch.mean(torch.log(predicted_rms / true_rms) ** 2)

    def _acceleration_batch_rms_loss(self, predicted_acceleration, true_acceleration):
        """Match per-component acceleration RMS over a whole mini-batch."""
        epsilon = float(self.rms_loss_epsilon)
        predicted_rms = torch.sqrt(torch.mean(predicted_acceleration**2, dim=(0, 1)).clamp_min(epsilon))
        true_rms = torch.sqrt(torch.mean(true_acceleration**2, dim=(0, 1)).clamp_min(epsilon))
        return torch.mean(torch.log(predicted_rms / true_rms) ** 2)

    def _acceleration_tail_loss(self, predicted_acceleration, true_acceleration):
        """Match normalized fourth moments of acceleration components."""
        epsilon = float(self.rms_loss_epsilon)
        true_rms = torch.sqrt(torch.mean(true_acceleration**2, dim=(0, 1), keepdim=True).clamp_min(epsilon))
        predicted_tail = torch.mean((predicted_acceleration / true_rms) ** 4, dim=(0, 1)).clamp_min(epsilon)
        true_tail = torch.mean((true_acceleration / true_rms) ** 4, dim=(0, 1)).clamp_min(epsilon)
        return torch.mean(torch.log(predicted_tail / true_tail) ** 2)

    def _acceleration_asymmetric_rms_loss(self, predicted_acceleration, true_acceleration):
        """Penalize overestimated acceleration RMS more directly than log-ratio loss."""
        epsilon = float(self.rms_loss_epsilon)
        over_weight = float(self.acceleration_over_rms_loss_weight)
        under_weight = float(self.acceleration_under_rms_loss_weight)
        over_power = float(getattr(self, "acceleration_over_rms_loss_power", 2.0))
        under_power = float(getattr(self, "acceleration_under_rms_loss_power", 2.0))
        predicted_rms = torch.sqrt(torch.mean(predicted_acceleration**2, dim=(1, 2)).clamp_min(epsilon))
        true_rms = torch.sqrt(torch.mean(true_acceleration**2, dim=(1, 2)).clamp_min(epsilon))
        ratio = predicted_rms / true_rms
        over_loss = torch.relu(ratio - 1.0) ** over_power
        under_loss = torch.relu(1.0 - ratio) ** under_power
        return over_weight * torch.mean(over_loss) + under_weight * torch.mean(under_loss)

    def train_crystal_blocks(
        self,
        X_blocks,
        y_blocks,
        data_len=0.5,
        training_target="displacement",
        displacement_y_blocks=None,
    ):
        """Train on 3x3x3 displacement blocks and central-cell acceleration targets.

        Args:
            X_blocks: History blocks with shape ``(samples, sequence, 3, 3, 3, atoms, 3)``.
            y_blocks: Either next-displacement blocks for ``training_target='displacement'``
                or force-derived discrete acceleration blocks for ``training_target='force'``.
            data_len: Fraction of the selected block window used for optimization.
            training_target: Source of supervised acceleration targets.
            displacement_y_blocks: Optional next-displacement blocks used by
                the displacement moment loss when ``training_target='force'``.
        """
        training_target = _normalize_training_target(training_target)
        X_blocks = np.asarray(X_blocks, dtype=np.float32)
        y_blocks = np.asarray(y_blocks, dtype=np.float32)
        moment_loss_weight = _normalize_nonnegative_float(
            "displacement_moment_loss_weight",
            getattr(self, "displacement_moment_loss_weight", 0.0),
        )
        self.displacement_moment_loss_weight = moment_loss_weight
        self.displacement_moment_mean_weight = _normalize_nonnegative_float(
            "displacement_moment_mean_weight",
            getattr(self, "displacement_moment_mean_weight", 1.0),
        )
        self.displacement_moment_std_weight = _normalize_nonnegative_float(
            "displacement_moment_std_weight",
            getattr(self, "displacement_moment_std_weight", 1.0),
        )
        self.displacement_moment_rms_weight = _normalize_nonnegative_float(
            "displacement_moment_rms_weight",
            getattr(self, "displacement_moment_rms_weight", 0.0),
        )
        self.displacement_moment_component_weights = np.asarray(
            getattr(self, "displacement_moment_component_weights", np.ones(3, dtype=np.float32)),
            dtype=np.float32,
        )
        if self.displacement_moment_component_weights.shape != (3,):
            raise ValueError("displacement_moment_component_weights must have shape (3,)")
        if np.any(self.displacement_moment_component_weights < 0):
            raise ValueError("displacement_moment_component_weights must be non-negative")
        if not np.any(self.displacement_moment_component_weights > 0):
            raise ValueError("At least one displacement moment component weight must be positive")
        self.displacement_moment_loss_epsilon = _normalize_positive_float(
            "displacement_moment_loss_epsilon",
            getattr(self, "displacement_moment_loss_epsilon", 1e-12),
        )
        power_mean_loss_weight = _normalize_nonnegative_float(
            "power_mean_loss_weight",
            getattr(self, "power_mean_loss_weight", 0.0),
        )
        self.power_mean_loss_weight = power_mean_loss_weight
        self.power_mean_loss_epsilon = _normalize_positive_float(
            "power_mean_loss_epsilon",
            getattr(self, "power_mean_loss_epsilon", 1e-12),
        )
        q_power_loss_weight = _normalize_nonnegative_float(
            "q_power_loss_weight",
            getattr(self, "q_power_loss_weight", 0.0),
        )
        self.q_power_loss_weight = q_power_loss_weight
        self.q_power_loss_mode = _normalize_q_power_loss_mode(
            getattr(self, "q_power_loss_mode", "positive-excess")
        )
        self.q_power_loss_sample_count = _normalize_positive_int(
            "q_power_loss_sample_count",
            getattr(self, "q_power_loss_sample_count", 2),
        )
        self.q_power_loss_interval = _normalize_positive_int(
            "q_power_loss_interval",
            getattr(self, "q_power_loss_interval", 10),
        )
        self.q_power_loss_margin = _normalize_nonnegative_float(
            "q_power_loss_margin",
            getattr(self, "q_power_loss_margin", 0.0),
        )
        self.q_power_loss_epsilon = _normalize_positive_float(
            "q_power_loss_epsilon",
            getattr(self, "q_power_loss_epsilon", 1e-12),
        )
        self.q_power_loss_exclude_q_zero = bool(getattr(self, "q_power_loss_exclude_q_zero", True))
        acceleration_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_rms_loss_weight",
            getattr(self, "acceleration_rms_loss_weight", 0.0),
        )
        self.acceleration_rms_loss_weight = acceleration_rms_loss_weight
        acceleration_batch_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_batch_rms_loss_weight",
            getattr(self, "acceleration_batch_rms_loss_weight", 0.0),
        )
        self.acceleration_batch_rms_loss_weight = acceleration_batch_rms_loss_weight
        acceleration_tail_loss_weight = _normalize_nonnegative_float(
            "acceleration_tail_loss_weight",
            getattr(self, "acceleration_tail_loss_weight", 0.0),
        )
        self.acceleration_tail_loss_weight = acceleration_tail_loss_weight
        acceleration_over_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_over_rms_loss_weight",
            getattr(self, "acceleration_over_rms_loss_weight", 0.0),
        )
        self.acceleration_over_rms_loss_weight = acceleration_over_rms_loss_weight
        acceleration_under_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_under_rms_loss_weight",
            getattr(self, "acceleration_under_rms_loss_weight", 0.0),
        )
        self.acceleration_under_rms_loss_weight = acceleration_under_rms_loss_weight
        self.acceleration_over_rms_loss_power = _normalize_positive_float(
            "acceleration_over_rms_loss_power",
            getattr(self, "acceleration_over_rms_loss_power", 2.0),
        )
        self.acceleration_under_rms_loss_power = _normalize_positive_float(
            "acceleration_under_rms_loss_power",
            getattr(self, "acceleration_under_rms_loss_power", 2.0),
        )
        self.rms_loss_epsilon = _normalize_positive_float(
            "rms_loss_epsilon",
            getattr(self, "rms_loss_epsilon", 1e-12),
        )
        curl_loss_weight = _normalize_nonnegative_float(
            "curl_loss_weight",
            getattr(self, "curl_loss_weight", 0.0),
        )
        self.curl_loss_weight = curl_loss_weight
        self.curl_loss_sample_count = _normalize_positive_int(
            "curl_loss_sample_count",
            getattr(self, "curl_loss_sample_count", 4),
        )
        self.curl_loss_interval = _normalize_positive_int(
            "curl_loss_interval",
            getattr(self, "curl_loss_interval", 1),
        )
        self.curl_loss_epsilon = _normalize_positive_float(
            "curl_loss_epsilon",
            getattr(self, "curl_loss_epsilon", 1e-12),
        )
        if X_blocks.ndim != 7:
            raise ValueError("X_blocks must have shape (n, sequence, 3, 3, 3, atoms, 3)")
        if y_blocks.ndim != 6:
            raise ValueError("y_blocks must have shape (n, 3, 3, 3, atoms, 3)")
        if tuple(X_blocks.shape[2:5]) != self.patch_shape or tuple(y_blocks.shape[1:4]) != self.patch_shape:
            raise ValueError("CrystalEdgeRNNNet expects 3x3x3 training blocks")
        if X_blocks.shape[0] != y_blocks.shape[0]:
            raise ValueError("X_blocks and y_blocks must contain the same number of samples")
        if X_blocks.shape[5] != self.unit_cell_atoms or y_blocks.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match model metadata")
        if X_blocks.shape[1] < 2:
            raise ValueError("At least two history frames are required for acceleration targets")
        if (
            moment_loss_weight > 0
            and self.displacement_moment_mean_weight == 0
            and self.displacement_moment_std_weight == 0
            and self.displacement_moment_rms_weight == 0
        ):
            raise ValueError("At least one displacement moment sub-weight must be positive")
        needs_q_power_loss = q_power_loss_weight > 0
        needs_displacement_y = moment_loss_weight > 0 or power_mean_loss_weight > 0 or needs_q_power_loss
        needs_target_acceleration = (
            power_mean_loss_weight > 0
            or acceleration_rms_loss_weight > 0
            or acceleration_batch_rms_loss_weight > 0
            or acceleration_tail_loss_weight > 0
            or acceleration_over_rms_loss_weight > 0
            or acceleration_under_rms_loss_weight > 0
        )
        needs_curl_loss = curl_loss_weight > 0
        if needs_displacement_y:
            if displacement_y_blocks is None:
                if training_target == "displacement":
                    displacement_y_blocks = y_blocks
                else:
                    raise ValueError(
                        "displacement_y_blocks is required for auxiliary displacement-based losses "
                        "when training_target='force'"
                    )
            displacement_y_blocks = np.asarray(displacement_y_blocks, dtype=np.float32)
            if displacement_y_blocks.shape != y_blocks.shape:
                raise ValueError("displacement_y_blocks must have the same block shape as y_blocks")

        self.train_count = int(data_len * X_blocks.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_blocks.shape[0]:
            start_index = 0
        else:
            start_index = np.random.randint(low=0, high=X_blocks.shape[0] - self.train_count)

        X_train = X_blocks[start_index : start_index + self.train_count]
        y_train = y_blocks[start_index : start_index + self.train_count]
        displacement_y_train = (
            None
            if displacement_y_blocks is None
            else displacement_y_blocks[start_index : start_index + self.train_count]
        )
        features = self.edge_features_from_patches(X_train)
        target_accelerations = (
            self._target_acceleration(X_train, y_train)
            if training_target == "displacement"
            else self._target_force_acceleration(y_train)
        )
        self.training_target = training_target
        self._set_acceleration_normalization_stats(target_accelerations)
        targets = self._normalize_acceleration(target_accelerations)

        dataset_tensors = [
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
        ]
        if needs_displacement_y:
            previous_center = X_train[(slice(None), -2, 1, 1, 1, slice(None), slice(None))]
            last_center = X_train[(slice(None), -1, 1, 1, 1, slice(None), slice(None))]
            true_next_center = self._center_displacements(displacement_y_train)
            dataset_tensors.extend(
                [
                    torch.as_tensor(previous_center, dtype=torch.float32),
                    torch.as_tensor(last_center, dtype=torch.float32),
                    torch.as_tensor(true_next_center, dtype=torch.float32),
                ]
            )
        if needs_target_acceleration:
            dataset_tensors.append(torch.as_tensor(target_accelerations, dtype=torch.float32))
        curl_batch_index = None
        if needs_curl_loss:
            curl_batch_index = len(dataset_tensors)
            dataset_tensors.append(torch.as_tensor(X_train, dtype=torch.float32))
        q_power_batch_index = None
        if needs_q_power_loss:
            q_power_batch_index = len(dataset_tensors)
            dataset_tensors.extend(
                [
                    torch.as_tensor(X_train, dtype=torch.float32),
                    torch.as_tensor(y_train, dtype=torch.float32),
                    torch.as_tensor(displacement_y_train, dtype=torch.float32),
                ]
            )
        dataset = TensorDataset(*dataset_tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        losses = []
        self.model.train()

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0.0
            batch_count = 0
            for batch in loader:
                x_train = batch[0]
                y_train = batch[1]
                x_train = x_train.to(self.torch_device)
                y_train = y_train.to(self.torch_device)
                prediction = self.model(x_train)
                loss = loss_func(prediction, y_train)
                predicted_acceleration = None
                if needs_displacement_y:
                    previous_center = batch[2].to(self.torch_device)
                    last_center = batch[3].to(self.torch_device)
                    true_next_center = batch[4].to(self.torch_device)
                if (
                    moment_loss_weight > 0
                    or power_mean_loss_weight > 0
                    or acceleration_rms_loss_weight > 0
                    or acceleration_batch_rms_loss_weight > 0
                    or acceleration_tail_loss_weight > 0
                    or acceleration_over_rms_loss_weight > 0
                    or acceleration_under_rms_loss_weight > 0
                ):
                    predicted_acceleration = self._denormalize_acceleration(prediction).reshape(
                        -1, self.unit_cell_atoms, 3
                    )
                if moment_loss_weight > 0:
                    predicted_next_center = 2.0 * last_center - previous_center + predicted_acceleration
                    loss = loss + moment_loss_weight * self._displacement_moment_loss(
                        predicted_next_center,
                        true_next_center,
                    )
                batch_index = 5 if needs_displacement_y else 2
                true_acceleration = None
                if needs_target_acceleration:
                    true_acceleration = batch[batch_index].to(self.torch_device)
                if acceleration_rms_loss_weight > 0:
                    loss = loss + acceleration_rms_loss_weight * self._acceleration_rms_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if acceleration_batch_rms_loss_weight > 0:
                    loss = loss + acceleration_batch_rms_loss_weight * self._acceleration_batch_rms_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if acceleration_tail_loss_weight > 0:
                    loss = loss + acceleration_tail_loss_weight * self._acceleration_tail_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if acceleration_over_rms_loss_weight > 0 or acceleration_under_rms_loss_weight > 0:
                    loss = loss + self._acceleration_asymmetric_rms_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if power_mean_loss_weight > 0:
                    reference_velocity = 0.5 * (true_next_center - previous_center)
                    loss = loss + power_mean_loss_weight * self._power_mean_loss(
                        predicted_acceleration,
                        true_acceleration,
                        reference_velocity,
                    )
                if needs_curl_loss and batch_count % self.curl_loss_interval == 0:
                    loss = loss + curl_loss_weight * self._curl_loss(batch[curl_batch_index].to(self.torch_device))
                if needs_q_power_loss and batch_count % self.q_power_loss_interval == 0:
                    loss = loss + q_power_loss_weight * self._q_power_loss(
                        batch[q_power_batch_index].to(self.torch_device),
                        batch[q_power_batch_index + 1].to(self.torch_device),
                        batch[q_power_batch_index + 2].to(self.torch_device),
                        training_target,
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_count += 1
                loss_mean = loss.item() / batch_count + (1.0 - 1.0 / batch_count) * loss_mean
            losses.append(loss_mean)
        return losses

    def predict_center_accelerations(self, patch_batch):
        """Predict central-cell accelerations for a batch of local history patches."""
        features = self.edge_features_from_patches(patch_batch)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(torch.as_tensor(features, dtype=torch.float32, device=self.torch_device))
            acceleration = self._denormalize_acceleration(raw)
        return acceleration.detach().cpu().numpy().reshape(len(patch_batch), self.unit_cell_atoms, 3).astype(np.float32)

    def run_crystal(self, count_steps, init_displacements, periodic=True, patch_batch_size=250):
        """Run centered local acceleration inference over a full crystal."""
        if count_steps <= 0:
            raise ValueError("count_steps must be positive")
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if init_displacements.ndim != 6:
            raise ValueError("init_displacements must have shape (sequence, nx, ny, nz, atoms, 3)")
        if init_displacements.shape[0] < 2:
            raise ValueError("At least two history frames are required")
        if init_displacements.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match init_displacements")
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive")

        crystal_shape = tuple(int(dim) for dim in init_displacements.shape[1:4])
        centers = build_centers(crystal_shape, periodic=periodic)
        history = init_displacements.copy()
        predictions = []

        for _ in range(count_steps):
            acceleration = np.zeros_like(history[-1], dtype=np.float32)
            for start in range(0, len(centers), patch_batch_size):
                batch_centers = centers[start : start + patch_batch_size]
                patch_batch = _extract_patch_batch(history, batch_centers, self.patch_shape, periodic)
                center_acceleration = self.predict_center_accelerations(patch_batch)
                for local_index, center in enumerate(batch_centers):
                    acceleration[(*center, slice(None), slice(None))] = center_acceleration[local_index]
            next_frame = 2.0 * history[-1] - history[-2] + acceleration
            predictions.append(next_frame.astype(np.float32))
            history[:-1] = history[1:]
            history[-1] = next_frame

        return np.asarray(predictions, dtype=np.float32)


class CrystalPairForceRNNNet(CrystalEdgeRNNNet):
    """RNN model that assembles accelerations from antisymmetric pair terms.

    This class keeps the same local-patch interface as ``CrystalEdgeRNNNet`` so
    the existing training, S(q,w) scoring, and ASE code paths can reuse it.  The
    important difference is the output parameterization: a shared RNN predicts
    one vector contribution for each oriented neighbor pair, the contribution is
    made odd under pair reversal, and atom accelerations are obtained by summing
    pair contributions.

    During full-crystal inference the default ``pair_scatter_inference=True``
    uses each periodic pair once and scatters ``+f`` and ``-f`` to the two atoms.
    This enforces zero total force at every model step, which is the main
    physical constraint this experimental architecture is meant to test.
    """

    def __init__(
        self,
        reference_positions,
        atom_order,
        box_lengths,
        hidden_size,
        rnn_layers,
        type="GRU",
        bidirectional=True,
        neighbor_shells=2,
        cutoff_scale=1.05,
        acceleration_normalization="global",
        rnn_readout_mode="last-output",
        temporal_input_mode="absolute-pair",
        device="auto",
        pair_scatter_inference=True,
    ):
        super().__init__(
            reference_positions=reference_positions,
            atom_order=atom_order,
            box_lengths=box_lengths,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            type=type,
            bidirectional=bidirectional,
            neighbor_shells=neighbor_shells,
            cutoff_scale=cutoff_scale,
            acceleration_normalization=acceleration_normalization,
            rnn_readout_mode=rnn_readout_mode,
            temporal_input_mode=temporal_input_mode,
            device=device,
        )
        self.model = _OddPairForceRNN(
            input_size=self.feature_channels,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            readout_mode=self.rnn_readout_mode,
        ).to(self.torch_device)
        self.architecture = "pair-force"
        self.pair_scatter_inference = bool(pair_scatter_inference)
        self.pair_output_scale = np.asarray(1.0, dtype=np.float32)

    def edge_sequence_features_from_patches(self, patches):
        """Return per-pair feature sequences with shape ``(batch, atoms, neighbors, sequence, 3)``."""
        patches = np.asarray(patches, dtype=np.float32)
        if patches.ndim != 7:
            raise ValueError("patches must have shape (batch, sequence, 3, 3, 3, atoms, 3)")
        patches = self._temporal_feature_patches(patches)
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        batch_size, sequence_length = patches.shape[:2]
        feature_channels = _feature_channels_for_temporal_input_mode(mode)
        features = np.empty(
            (batch_size, self.unit_cell_atoms, self.neighbor_count, sequence_length, feature_channels),
            dtype=np.float32,
        )
        center_displacements = patches[:, :, 1, 1, 1]
        for atom_index in range(self.unit_cell_atoms):
            center = center_displacements[:, :, atom_index, :]
            for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                neighbor = patches[(slice(None), slice(None), *tuple(local_index), slice(None))]
                vector = neighbor - center
                if mode == "absolute-pair":
                    vector = vector + self.reference_vectors[atom_index, neighbor_index]
                    features[:, atom_index, neighbor_index, :, :] = vector / self.lattice_parameter
                elif mode == "ref-plus-delta":
                    reference = np.broadcast_to(self.reference_vectors[atom_index, neighbor_index], vector.shape)
                    features[:, atom_index, neighbor_index, :, :3] = reference / self.lattice_parameter
                    features[:, atom_index, neighbor_index, :, 3:] = vector / self.lattice_parameter
                else:
                    features[:, atom_index, neighbor_index, :, :] = vector / self.lattice_parameter
        return features

    def _edge_sequence_features_from_patches_torch(self, patches):
        """Return differentiable per-pair feature sequences from torch patches."""
        if patches.ndim != 7:
            raise ValueError("patches must have shape (batch, sequence, 3, 3, 3, atoms, 3)")
        patches = self._temporal_feature_patches_torch(patches)
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        per_atom_features = []
        center_displacements = patches[:, :, 1, 1, 1]
        reference_vectors = torch.as_tensor(
            self.reference_vectors,
            dtype=patches.dtype,
            device=patches.device,
        )
        scale = torch.as_tensor(self.lattice_parameter, dtype=patches.dtype, device=patches.device)
        for atom_index in range(self.unit_cell_atoms):
            neighbor_features = []
            center = center_displacements[:, :, atom_index, :]
            for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                lx, ly, lz, neighbor_atom = (int(value) for value in local_index)
                neighbor = patches[:, :, lx, ly, lz, neighbor_atom, :]
                vector = neighbor - center
                if mode == "absolute-pair":
                    vector = vector + reference_vectors[atom_index, neighbor_index]
                    neighbor_features.append(vector / scale)
                elif mode == "ref-plus-delta":
                    reference = reference_vectors[atom_index, neighbor_index].reshape(1, 1, 3).expand_as(vector)
                    neighbor_features.append(torch.cat((reference / scale, vector / scale), dim=-1))
                else:
                    neighbor_features.append(vector / scale)
            per_atom_features.append(torch.stack(neighbor_features, dim=1))
        return torch.stack(per_atom_features, dim=1)

    def _center_acceleration_from_patch_tensor(self, patches):
        """Return physical central-cell accelerations from differentiable patches."""
        features = self._edge_sequence_features_from_patches_torch(patches)
        return self._center_acceleration_from_features(features)

    def _set_acceleration_normalization_stats(self, targets):
        """Store acceleration stats and a conservative pair-output scale."""
        super()._set_acceleration_normalization_stats(targets)
        std = np.asarray(self.acceleration_std, dtype=np.float32)
        # Neighbor contributions add vectorially; sqrt(N) gives a less
        # over-damped initial scale than assuming all pair terms add coherently.
        scale = float(np.mean(std)) / np.sqrt(max(1, self.neighbor_count))
        self.pair_output_scale = np.asarray(max(scale, 1e-12), dtype=np.float32)

    def _normalize_acceleration_torch(self, acceleration):
        """Normalize a torch acceleration tensor with stored target statistics."""
        flat = acceleration.reshape(acceleration.shape[0], self.output_size)
        mean = torch.as_tensor(self.acceleration_mean, dtype=flat.dtype, device=flat.device)
        std = torch.as_tensor(self.acceleration_std, dtype=flat.dtype, device=flat.device)
        return (flat - mean) / std

    def _pair_contributions_from_features(self, features):
        """Return physical pair contributions from per-pair feature sequences."""
        if features.ndim != 5:
            raise ValueError("features must have shape (batch, atoms, neighbors, sequence, channels)")
        batch_size = int(features.shape[0])
        flat_features = features.reshape(-1, features.shape[-2], features.shape[-1])
        raw = self.model(flat_features)
        scale = torch.as_tensor(self.pair_output_scale, dtype=raw.dtype, device=raw.device)
        return (raw * scale).reshape(batch_size, self.unit_cell_atoms, self.neighbor_count, 3)

    def _center_acceleration_from_features(self, features):
        """Return physical central-cell accelerations from per-pair features."""
        return torch.sum(self._pair_contributions_from_features(features), dim=2)

    def train_crystal_blocks(
        self,
        X_blocks,
        y_blocks,
        data_len=0.5,
        training_target="displacement",
        displacement_y_blocks=None,
    ):
        """Train the pair-force model on local blocks and central accelerations."""
        training_target = _normalize_training_target(training_target)
        X_blocks = np.asarray(X_blocks, dtype=np.float32)
        y_blocks = np.asarray(y_blocks, dtype=np.float32)
        moment_loss_weight = _normalize_nonnegative_float(
            "displacement_moment_loss_weight",
            getattr(self, "displacement_moment_loss_weight", 0.0),
        )
        self.displacement_moment_loss_weight = moment_loss_weight
        self.displacement_moment_mean_weight = _normalize_nonnegative_float(
            "displacement_moment_mean_weight",
            getattr(self, "displacement_moment_mean_weight", 1.0),
        )
        self.displacement_moment_std_weight = _normalize_nonnegative_float(
            "displacement_moment_std_weight",
            getattr(self, "displacement_moment_std_weight", 1.0),
        )
        self.displacement_moment_rms_weight = _normalize_nonnegative_float(
            "displacement_moment_rms_weight",
            getattr(self, "displacement_moment_rms_weight", 0.0),
        )
        self.displacement_moment_component_weights = np.asarray(
            getattr(self, "displacement_moment_component_weights", np.ones(3, dtype=np.float32)),
            dtype=np.float32,
        )
        if self.displacement_moment_component_weights.shape != (3,):
            raise ValueError("displacement_moment_component_weights must have shape (3,)")
        if np.any(self.displacement_moment_component_weights < 0):
            raise ValueError("displacement_moment_component_weights must be non-negative")
        if not np.any(self.displacement_moment_component_weights > 0):
            raise ValueError("At least one displacement moment component weight must be positive")
        self.displacement_moment_loss_epsilon = _normalize_positive_float(
            "displacement_moment_loss_epsilon",
            getattr(self, "displacement_moment_loss_epsilon", 1e-12),
        )
        power_mean_loss_weight = _normalize_nonnegative_float(
            "power_mean_loss_weight",
            getattr(self, "power_mean_loss_weight", 0.0),
        )
        self.power_mean_loss_weight = power_mean_loss_weight
        self.power_mean_loss_epsilon = _normalize_positive_float(
            "power_mean_loss_epsilon",
            getattr(self, "power_mean_loss_epsilon", 1e-12),
        )
        q_power_loss_weight = _normalize_nonnegative_float(
            "q_power_loss_weight",
            getattr(self, "q_power_loss_weight", 0.0),
        )
        self.q_power_loss_weight = q_power_loss_weight
        self.q_power_loss_mode = _normalize_q_power_loss_mode(
            getattr(self, "q_power_loss_mode", "positive-excess")
        )
        self.q_power_loss_sample_count = _normalize_positive_int(
            "q_power_loss_sample_count",
            getattr(self, "q_power_loss_sample_count", 2),
        )
        self.q_power_loss_interval = _normalize_positive_int(
            "q_power_loss_interval",
            getattr(self, "q_power_loss_interval", 10),
        )
        self.q_power_loss_margin = _normalize_nonnegative_float(
            "q_power_loss_margin",
            getattr(self, "q_power_loss_margin", 0.0),
        )
        self.q_power_loss_epsilon = _normalize_positive_float(
            "q_power_loss_epsilon",
            getattr(self, "q_power_loss_epsilon", 1e-12),
        )
        self.q_power_loss_exclude_q_zero = bool(getattr(self, "q_power_loss_exclude_q_zero", True))
        acceleration_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_rms_loss_weight",
            getattr(self, "acceleration_rms_loss_weight", 0.0),
        )
        self.acceleration_rms_loss_weight = acceleration_rms_loss_weight
        acceleration_batch_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_batch_rms_loss_weight",
            getattr(self, "acceleration_batch_rms_loss_weight", 0.0),
        )
        self.acceleration_batch_rms_loss_weight = acceleration_batch_rms_loss_weight
        acceleration_tail_loss_weight = _normalize_nonnegative_float(
            "acceleration_tail_loss_weight",
            getattr(self, "acceleration_tail_loss_weight", 0.0),
        )
        self.acceleration_tail_loss_weight = acceleration_tail_loss_weight
        acceleration_over_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_over_rms_loss_weight",
            getattr(self, "acceleration_over_rms_loss_weight", 0.0),
        )
        self.acceleration_over_rms_loss_weight = acceleration_over_rms_loss_weight
        acceleration_under_rms_loss_weight = _normalize_nonnegative_float(
            "acceleration_under_rms_loss_weight",
            getattr(self, "acceleration_under_rms_loss_weight", 0.0),
        )
        self.acceleration_under_rms_loss_weight = acceleration_under_rms_loss_weight
        self.acceleration_over_rms_loss_power = _normalize_positive_float(
            "acceleration_over_rms_loss_power",
            getattr(self, "acceleration_over_rms_loss_power", 2.0),
        )
        self.acceleration_under_rms_loss_power = _normalize_positive_float(
            "acceleration_under_rms_loss_power",
            getattr(self, "acceleration_under_rms_loss_power", 2.0),
        )
        self.rms_loss_epsilon = _normalize_positive_float(
            "rms_loss_epsilon",
            getattr(self, "rms_loss_epsilon", 1e-12),
        )
        curl_loss_weight = _normalize_nonnegative_float(
            "curl_loss_weight",
            getattr(self, "curl_loss_weight", 0.0),
        )
        self.curl_loss_weight = curl_loss_weight
        self.curl_loss_sample_count = _normalize_positive_int(
            "curl_loss_sample_count",
            getattr(self, "curl_loss_sample_count", 4),
        )
        self.curl_loss_interval = _normalize_positive_int(
            "curl_loss_interval",
            getattr(self, "curl_loss_interval", 1),
        )
        self.curl_loss_epsilon = _normalize_positive_float(
            "curl_loss_epsilon",
            getattr(self, "curl_loss_epsilon", 1e-12),
        )
        reference_pressure_loss_weight = _normalize_nonnegative_float(
            "reference_pressure_loss_weight",
            getattr(self, "reference_pressure_loss_weight", 0.0),
        )
        self.reference_pressure_loss_weight = reference_pressure_loss_weight
        self.reference_pressure_target = float(getattr(self, "reference_pressure_target", 0.0))
        self.reference_pressure_loss_scale = _normalize_positive_float(
            "reference_pressure_loss_scale",
            getattr(self, "reference_pressure_loss_scale", 1.0),
        )
        if X_blocks.ndim != 7:
            raise ValueError("X_blocks must have shape (n, sequence, 3, 3, 3, atoms, 3)")
        if y_blocks.ndim != 6:
            raise ValueError("y_blocks must have shape (n, 3, 3, 3, atoms, 3)")
        if tuple(X_blocks.shape[2:5]) != self.patch_shape or tuple(y_blocks.shape[1:4]) != self.patch_shape:
            raise ValueError("CrystalPairForceRNNNet expects 3x3x3 training blocks")
        if X_blocks.shape[0] != y_blocks.shape[0]:
            raise ValueError("X_blocks and y_blocks must contain the same number of samples")
        if X_blocks.shape[5] != self.unit_cell_atoms or y_blocks.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match model metadata")
        if X_blocks.shape[1] < 1:
            raise ValueError("At least one model input frame is required")
        if (
            moment_loss_weight > 0
            and self.displacement_moment_mean_weight == 0
            and self.displacement_moment_std_weight == 0
            and self.displacement_moment_rms_weight == 0
        ):
            raise ValueError("At least one displacement moment sub-weight must be positive")
        needs_q_power_loss = q_power_loss_weight > 0
        needs_displacement_y = moment_loss_weight > 0 or power_mean_loss_weight > 0 or needs_q_power_loss
        needs_target_acceleration = (
            power_mean_loss_weight > 0
            or acceleration_rms_loss_weight > 0
            or acceleration_batch_rms_loss_weight > 0
            or acceleration_tail_loss_weight > 0
            or acceleration_over_rms_loss_weight > 0
            or acceleration_under_rms_loss_weight > 0
        )
        needs_curl_loss = curl_loss_weight > 0
        needs_reference_pressure_loss = reference_pressure_loss_weight > 0
        if needs_reference_pressure_loss and getattr(self, "architecture", None) != "pair-energy":
            raise ValueError("reference pressure loss is only supported by pair-energy models")
        if X_blocks.shape[1] < 2 and (training_target == "displacement" or needs_displacement_y):
            raise ValueError(
                "At least two history frames are required for displacement-derived targets or losses"
            )
        if needs_displacement_y:
            if displacement_y_blocks is None:
                if training_target == "displacement":
                    displacement_y_blocks = y_blocks
                else:
                    raise ValueError(
                        "displacement_y_blocks is required for auxiliary displacement-based losses "
                        "when training_target='force'"
                    )
            displacement_y_blocks = np.asarray(displacement_y_blocks, dtype=np.float32)
            if displacement_y_blocks.shape != y_blocks.shape:
                raise ValueError("displacement_y_blocks must have the same block shape as y_blocks")

        self.train_count = int(data_len * X_blocks.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_blocks.shape[0]:
            start_index = 0
        else:
            start_index = np.random.randint(low=0, high=X_blocks.shape[0] - self.train_count)

        X_train = X_blocks[start_index : start_index + self.train_count]
        y_train = y_blocks[start_index : start_index + self.train_count]
        displacement_y_train = (
            None
            if displacement_y_blocks is None
            else displacement_y_blocks[start_index : start_index + self.train_count]
        )
        features = self.edge_sequence_features_from_patches(X_train)
        target_accelerations = (
            self._target_acceleration(X_train, y_train)
            if training_target == "displacement"
            else self._target_force_acceleration(y_train)
        )
        self.training_target = training_target
        self._set_acceleration_normalization_stats(target_accelerations)
        targets = self._normalize_acceleration(target_accelerations)

        dataset_tensors = [
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
        ]
        if needs_displacement_y:
            previous_center = X_train[(slice(None), -2, 1, 1, 1, slice(None), slice(None))]
            last_center = X_train[(slice(None), -1, 1, 1, 1, slice(None), slice(None))]
            true_next_center = self._center_displacements(displacement_y_train)
            dataset_tensors.extend(
                [
                    torch.as_tensor(previous_center, dtype=torch.float32),
                    torch.as_tensor(last_center, dtype=torch.float32),
                    torch.as_tensor(true_next_center, dtype=torch.float32),
                ]
            )
        if needs_target_acceleration:
            dataset_tensors.append(torch.as_tensor(target_accelerations, dtype=torch.float32))
        curl_batch_index = None
        if needs_curl_loss:
            curl_batch_index = len(dataset_tensors)
            dataset_tensors.append(torch.as_tensor(X_train, dtype=torch.float32))
        q_power_batch_index = None
        if needs_q_power_loss:
            q_power_batch_index = len(dataset_tensors)
            dataset_tensors.extend(
                [
                    torch.as_tensor(X_train, dtype=torch.float32),
                    torch.as_tensor(y_train, dtype=torch.float32),
                    torch.as_tensor(displacement_y_train, dtype=torch.float32),
                ]
            )
        dataset = TensorDataset(*dataset_tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        losses = []
        self.model.train()

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0.0
            batch_count = 0
            for batch in loader:
                x_train = batch[0].to(self.torch_device)
                y_train = batch[1].to(self.torch_device)
                predicted_acceleration = self._center_acceleration_from_features(x_train)
                prediction = self._normalize_acceleration_torch(predicted_acceleration)
                loss = loss_func(prediction, y_train)
                if needs_displacement_y:
                    previous_center = batch[2].to(self.torch_device)
                    last_center = batch[3].to(self.torch_device)
                    true_next_center = batch[4].to(self.torch_device)
                if moment_loss_weight > 0:
                    predicted_next_center = 2.0 * last_center - previous_center + predicted_acceleration
                    loss = loss + moment_loss_weight * self._displacement_moment_loss(
                        predicted_next_center,
                        true_next_center,
                    )
                batch_index = 5 if needs_displacement_y else 2
                true_acceleration = None
                if needs_target_acceleration:
                    true_acceleration = batch[batch_index].to(self.torch_device)
                if acceleration_rms_loss_weight > 0:
                    loss = loss + acceleration_rms_loss_weight * self._acceleration_rms_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if acceleration_batch_rms_loss_weight > 0:
                    loss = loss + acceleration_batch_rms_loss_weight * self._acceleration_batch_rms_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if acceleration_tail_loss_weight > 0:
                    loss = loss + acceleration_tail_loss_weight * self._acceleration_tail_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if acceleration_over_rms_loss_weight > 0 or acceleration_under_rms_loss_weight > 0:
                    loss = loss + self._acceleration_asymmetric_rms_loss(
                        predicted_acceleration,
                        true_acceleration,
                    )
                if power_mean_loss_weight > 0:
                    reference_velocity = 0.5 * (true_next_center - previous_center)
                    loss = loss + power_mean_loss_weight * self._power_mean_loss(
                        predicted_acceleration,
                        true_acceleration,
                        reference_velocity,
                    )
                if needs_curl_loss and batch_count % self.curl_loss_interval == 0:
                    loss = loss + curl_loss_weight * self._curl_loss(batch[curl_batch_index].to(self.torch_device))
                if needs_q_power_loss and batch_count % self.q_power_loss_interval == 0:
                    loss = loss + q_power_loss_weight * self._q_power_loss(
                        batch[q_power_batch_index].to(self.torch_device),
                        batch[q_power_batch_index + 1].to(self.torch_device),
                        batch[q_power_batch_index + 2].to(self.torch_device),
                        training_target,
                    )
                if needs_reference_pressure_loss:
                    loss = loss + reference_pressure_loss_weight * self._reference_pressure_loss(
                        features.shape[-2],
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_count += 1
                loss_mean = loss.item() / batch_count + (1.0 - 1.0 / batch_count) * loss_mean
            losses.append(loss_mean)
        return losses

    def predict_pair_contributions(self, patch_batch):
        """Predict physical pair contributions for local history patches."""
        features = self.edge_sequence_features_from_patches(patch_batch)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.torch_device)
            contributions = self._pair_contributions_from_features(tensor)
        return contributions.detach().cpu().numpy().astype(np.float32)

    def predict_center_accelerations(self, patch_batch):
        """Predict central-cell accelerations by summing pair contributions."""
        contributions = self.predict_pair_contributions(patch_batch)
        return np.sum(contributions, axis=2).astype(np.float32)

    def _crystal_atom_id(self, cell, atom_index, crystal_shape):
        """Return a stable integer id for a crystal atom tuple."""
        ix, iy, iz = (int(value) for value in cell)
        nx, ny, nz = (int(value) for value in crystal_shape)
        return (((ix * ny) + iy) * nz + iz) * self.unit_cell_atoms + int(atom_index)

    def _neighbor_global_cell(self, center, local_index, crystal_shape, periodic):
        """Map a local 3x3x3 neighbor index to a global crystal cell."""
        cell = np.asarray(center, dtype=np.int64) + np.asarray(local_index[:3], dtype=np.int64) - 1
        shape = np.asarray(crystal_shape, dtype=np.int64)
        if periodic:
            cell %= shape
        elif np.any((cell < 0) | (cell >= shape)):
            return None
        return tuple(int(value) for value in cell)

    def _full_pair_scatter_indices(self, crystal_shape, periodic):
        """Return cached unique-pair indices for vectorized full-crystal scatter."""
        crystal_shape = tuple(int(dim) for dim in crystal_shape)
        key = (crystal_shape, bool(periodic))
        cache = getattr(self, "_full_pair_scatter_index_cache", None)
        if cache is None:
            cache = {}
            self._full_pair_scatter_index_cache = cache
        if key in cache:
            return cache[key]

        centers = build_centers(crystal_shape, periodic=periodic)
        pair_center_indices = []
        pair_atom_indices = []
        pair_neighbor_indices = []
        pair_center_flat_indices = []
        pair_neighbor_flat_indices = []
        for center_index, center in enumerate(centers):
            for atom_index in range(self.unit_cell_atoms):
                center_id = self._crystal_atom_id(center, atom_index, crystal_shape)
                for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                    neighbor_cell = self._neighbor_global_cell(center, local_index, crystal_shape, periodic)
                    if neighbor_cell is None:
                        continue
                    neighbor_atom = int(local_index[3])
                    neighbor_id = self._crystal_atom_id(neighbor_cell, neighbor_atom, crystal_shape)
                    if center_id >= neighbor_id:
                        continue
                    pair_center_indices.append(center_index)
                    pair_atom_indices.append(atom_index)
                    pair_neighbor_indices.append(neighbor_index)
                    pair_center_flat_indices.append(center_id)
                    pair_neighbor_flat_indices.append(neighbor_id)

        indices = {
            "centers": centers,
            "center_index": np.asarray(pair_center_indices, dtype=np.int64),
            "atom": np.asarray(pair_atom_indices, dtype=np.int64),
            "neighbor": np.asarray(pair_neighbor_indices, dtype=np.int64),
            "center_flat": np.asarray(pair_center_flat_indices, dtype=np.int64),
            "neighbor_flat": np.asarray(pair_neighbor_flat_indices, dtype=np.int64),
        }
        cache[key] = indices
        return indices

    def _scatter_pair_contributions(
        self,
        acceleration_flat,
        pair_contributions,
        scatter_indices,
        start,
        stop,
        pair_energies=None,
    ):
        """Scatter one batch of unique pair contributions and optionally sum energies."""
        batch_mask = (scatter_indices["center_index"] >= start) & (scatter_indices["center_index"] < stop)
        if not np.any(batch_mask):
            return 0.0

        batch_indices = scatter_indices["center_index"][batch_mask] - int(start)
        atom_indices = scatter_indices["atom"][batch_mask]
        neighbor_indices = scatter_indices["neighbor"][batch_mask]
        center_flat = scatter_indices["center_flat"][batch_mask]
        neighbor_flat = scatter_indices["neighbor_flat"][batch_mask]

        if pair_contributions is not None:
            contributions = pair_contributions[batch_indices, atom_indices, neighbor_indices]
            np.add.at(acceleration_flat, center_flat, contributions)
            np.add.at(acceleration_flat, neighbor_flat, -contributions)

        if pair_energies is None:
            return 0.0
        return float(np.sum(pair_energies[batch_indices, atom_indices, neighbor_indices], dtype=np.float64))

    def _assign_center_accelerations(self, acceleration, batch_centers, pair_contributions):
        """Assign centered accelerations for the non-pair-scatter inference mode."""
        center_acceleration = np.sum(pair_contributions, axis=2)
        for local_index, center in enumerate(batch_centers):
            acceleration[(*center, slice(None), slice(None))] = center_acceleration[local_index]

    def _model_history_from_integration_history(self, history):
        """Return the suffix of physical history consumed by the model.

        Verlet integration needs at least two physical frames to define the
        current velocity, while an MLP force model can consume only the latest
        displacement frame. Frame-layered models similarly have a fixed input
        length independent of the integration history kept by the caller.
        """
        if getattr(self, "temporal_architecture", "stacked") not in {"frame-layered", "mlp"}:
            return history
        raw_length = int(self.rnn_layers)
        if _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair")) == "relative-to-first":
            raw_length += 1
        if history.shape[0] < raw_length:
            raise ValueError(
                f"Model needs {raw_length} raw history frames, but only {history.shape[0]} were provided"
            )
        return history[-raw_length:]

    def predict_full_accelerations(self, history, periodic=True, patch_batch_size=250, pair_scatter=None):
        """Predict one full-crystal acceleration frame.

        If ``pair_scatter`` is true, each unique periodic pair is used once and
        the opposite contribution is scattered to the neighbor atom.  If false,
        the method falls back to the centered assignment used by
        ``CrystalEdgeRNNNet``.
        """
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 6:
            raise ValueError("history must have shape (sequence, nx, ny, nz, atoms, 3)")
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive")
        pair_scatter = self.pair_scatter_inference if pair_scatter is None else bool(pair_scatter)
        crystal_shape = tuple(int(dim) for dim in history.shape[1:4])
        scatter_indices = self._full_pair_scatter_indices(crystal_shape, periodic) if pair_scatter else None
        centers = scatter_indices["centers"] if pair_scatter else build_centers(crystal_shape, periodic=periodic)
        model_history = self._model_history_from_integration_history(history)
        acceleration = np.zeros_like(history[-1], dtype=np.float32)
        acceleration_flat = acceleration.reshape(-1, 3)

        for start in range(0, len(centers), patch_batch_size):
            stop = min(start + patch_batch_size, len(centers))
            batch_centers = centers[start : start + patch_batch_size]
            patch_batch = _extract_patch_batch(model_history, batch_centers, self.patch_shape, periodic)
            pair_contributions = self.predict_pair_contributions(patch_batch)
            if not pair_scatter:
                self._assign_center_accelerations(acceleration, batch_centers, pair_contributions)
                continue

            self._scatter_pair_contributions(
                acceleration_flat,
                pair_contributions,
                scatter_indices,
                start,
                stop,
            )
        return acceleration

    def run_crystal(self, count_steps, init_displacements, periodic=True, patch_batch_size=250):
        """Run pair-force acceleration inference over a full crystal."""
        if count_steps <= 0:
            raise ValueError("count_steps must be positive")
        init_displacements = np.asarray(init_displacements, dtype=np.float32)
        if init_displacements.ndim != 6:
            raise ValueError("init_displacements must have shape (sequence, nx, ny, nz, atoms, 3)")
        if init_displacements.shape[0] < 2:
            raise ValueError("At least two history frames are required")
        if init_displacements.shape[4] != self.unit_cell_atoms:
            raise ValueError("unit_cell_atoms does not match init_displacements")

        history = init_displacements.copy()
        predictions = []
        for _ in range(count_steps):
            acceleration = self.predict_full_accelerations(
                history,
                periodic=periodic,
                patch_batch_size=patch_batch_size,
                pair_scatter=self.pair_scatter_inference,
            )
            next_frame = 2.0 * history[-1] - history[-2] + acceleration
            predictions.append(next_frame.astype(np.float32))
            history[:-1] = history[1:]
            history[-1] = next_frame
        return np.asarray(predictions, dtype=np.float32)


class CrystalPairEnergyRNNNet(CrystalPairForceRNNNet):
    """Pair-energy model trained from force/acceleration targets.

    The selected temporal encoder maps each local pair input to a scalar
    potential. The paper baseline uses a one-frame MLP; recurrent encoders are
    retained for compatibility and ablations. Central-cell accelerations are obtained as
    ``-grad U`` with respect to the current central displacements, which makes
    the learned local force field conservative by construction for a fixed
    history context.
    """

    def __init__(
        self,
        reference_positions,
        atom_order,
        box_lengths,
        hidden_size,
        rnn_layers,
        type="GRU",
        bidirectional=True,
        neighbor_shells=2,
        cutoff_scale=1.05,
        acceleration_normalization="global",
        rnn_readout_mode="last-output",
        temporal_architecture="stacked",
        temporal_input_mode="absolute-pair",
        device="auto",
        pair_scatter_inference=True,
    ):
        super().__init__(
            reference_positions=reference_positions,
            atom_order=atom_order,
            box_lengths=box_lengths,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            type=type,
            bidirectional=bidirectional,
            neighbor_shells=neighbor_shells,
            cutoff_scale=cutoff_scale,
            acceleration_normalization=acceleration_normalization,
            rnn_readout_mode=rnn_readout_mode,
            temporal_input_mode=temporal_input_mode,
            device=device,
            pair_scatter_inference=pair_scatter_inference,
        )
        self.temporal_architecture = _normalize_temporal_architecture(temporal_architecture)
        self.model = _EvenPairEnergyRNN(
            input_size=self.feature_channels,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            readout_mode=self.rnn_readout_mode,
            temporal_architecture=self.temporal_architecture,
        ).to(self.torch_device)
        self.architecture = "pair-energy"
        self.energy_output_scale = np.asarray(1.0, dtype=np.float32)

    def _set_acceleration_normalization_stats(self, targets):
        """Store acceleration stats and a pair-energy scale."""
        CrystalEdgeRNNNet._set_acceleration_normalization_stats(self, targets)
        std = np.asarray(self.acceleration_std, dtype=np.float32)
        scale = float(np.mean(std)) * float(self.lattice_parameter) / np.sqrt(max(1, self.neighbor_count))
        self.energy_output_scale = np.asarray(max(scale, 1e-12), dtype=np.float32)

    def _pair_energies_from_features(self, features):
        """Return model-unit scalar pair energies from per-pair feature sequences."""
        if features.ndim != 5:
            raise ValueError("features must have shape (batch, atoms, neighbors, sequence, channels)")
        batch_size = int(features.shape[0])
        flat_features = features.reshape(-1, features.shape[-2], features.shape[-1])
        raw = self.model(flat_features)
        scale = torch.as_tensor(self.energy_output_scale, dtype=raw.dtype, device=raw.device)
        return (raw * scale).reshape(batch_size, self.unit_cell_atoms, self.neighbor_count)

    def _pair_contributions_from_features(self, features, create_graph=None):
        """Return conservative pair acceleration contributions.

        For a pair feature ``x = (r_j - r_i) / a0`` and scalar local energy
        ``U(x)``, the contribution to the center atom is
        ``dU/dx / a0``.  The neighbor receives the opposite contribution during
        pair-scatter full-crystal inference.
        """
        if not features.requires_grad:
            features = features.detach().clone().requires_grad_(True)
        if create_graph is None:
            # Training a force from an energy gradient needs second-order
            # autograd so model parameters remain connected to the loss.
            create_graph = bool(torch.is_grad_enabled())
        with torch.enable_grad():
            with torch.backends.cudnn.flags(enabled=False):
                energies = self._pair_energies_from_features(features)
                gradient = torch.autograd.grad(
                    energies.sum(),
                    features,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    only_inputs=True,
                )[0]
        scale = torch.as_tensor(self.lattice_parameter, dtype=gradient.dtype, device=gradient.device)
        dynamic_slice = _dynamic_channel_slice_for_temporal_input_mode(
            getattr(self, "temporal_input_mode", "absolute-pair")
        )
        return gradient[..., -1, dynamic_slice] / scale

    def _reference_sequence_features(self, sequence_length):
        """Return pair features for the undeformed reference unit cell."""
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("reference pressure sequence_length must be positive")
        mode = _normalize_temporal_input_mode(getattr(self, "temporal_input_mode", "absolute-pair"))
        if mode == "relative-to-first":
            raise ValueError("reference pressure loss is not defined for relative-to-first input")

        reference_vectors = torch.as_tensor(
            self.reference_vectors,
            dtype=torch.float32,
            device=self.torch_device,
        )
        scale = torch.as_tensor(self.lattice_parameter, dtype=torch.float32, device=self.torch_device)
        normalized_reference = reference_vectors / scale
        if mode == "ref-plus-delta":
            base = torch.cat((normalized_reference, torch.zeros_like(normalized_reference)), dim=-1)
        else:
            base = normalized_reference
        return base.unsqueeze(0).unsqueeze(3).expand(
            1,
            self.unit_cell_atoms,
            self.neighbor_count,
            sequence_length,
            base.shape[-1],
        )

    def _unit_cell_volume(self):
        """Return the orthorhombic reference unit-cell volume in Angstrom^3."""
        crystal_shape = np.asarray(self.atom_order.shape[:3], dtype=np.float64)
        unit_lengths = np.asarray(self.box_lengths, dtype=np.float64) / crystal_shape
        return float(np.prod(unit_lengths))

    def _reference_pressure_loss(self, sequence_length):
        """Penalize configurational pressure at the undeformed reference cell."""
        features = self._reference_sequence_features(sequence_length)
        contributions = self._pair_contributions_from_features(features, create_graph=True)[0]
        reference_vectors = torch.as_tensor(
            self.reference_vectors,
            dtype=contributions.dtype,
            device=contributions.device,
        )
        # The stencil stores both pair orientations, hence the factor one half.
        virial = 0.5 * torch.einsum("ani,anj->ij", reference_vectors, contributions)
        volume = torch.as_tensor(
            self._unit_cell_volume(),
            dtype=contributions.dtype,
            device=contributions.device,
        )
        pressure = -torch.trace(virial) / (3.0 * volume)
        target = torch.as_tensor(
            float(self.reference_pressure_target),
            dtype=contributions.dtype,
            device=contributions.device,
        )
        pressure_scale = torch.as_tensor(
            float(self.reference_pressure_loss_scale),
            dtype=contributions.dtype,
            device=contributions.device,
        )
        return ((pressure - target) / pressure_scale) ** 2

    def _pair_contributions_and_energies_from_features(self, features, create_graph=None):
        """Return conservative pair contributions and the scalar pair energies."""
        if not features.requires_grad:
            features = features.detach().clone().requires_grad_(True)
        if create_graph is None:
            create_graph = bool(torch.is_grad_enabled())
        with torch.enable_grad():
            with torch.backends.cudnn.flags(enabled=False):
                energies = self._pair_energies_from_features(features)
                gradient = torch.autograd.grad(
                    energies.sum(),
                    features,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    only_inputs=True,
                )[0]
        scale = torch.as_tensor(self.lattice_parameter, dtype=gradient.dtype, device=gradient.device)
        dynamic_slice = _dynamic_channel_slice_for_temporal_input_mode(
            getattr(self, "temporal_input_mode", "absolute-pair")
        )
        contributions = gradient[..., -1, dynamic_slice] / scale
        return contributions, energies

    def _center_acceleration_from_features(self, features):
        """Return conservative central-cell accelerations from pair energies."""
        return torch.sum(self._pair_contributions_from_features(features), dim=2)

    def predict_pair_energies(self, patch_batch):
        """Predict model-unit scalar pair energies for local history patches."""
        features = self.edge_sequence_features_from_patches(patch_batch)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.torch_device)
            energies = self._pair_energies_from_features(tensor)
        return energies.detach().cpu().numpy().astype(np.float32)

    def predict_pair_contributions(self, patch_batch):
        """Predict conservative pair acceleration contributions."""
        features = self.edge_sequence_features_from_patches(patch_batch)
        self.model.eval()
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.torch_device)
        contributions = self._pair_contributions_from_features(tensor, create_graph=False)
        return contributions.detach().cpu().numpy().astype(np.float32)

    def predict_pair_contributions_and_energies(self, patch_batch):
        """Predict pair contributions and energies with one model/autograd pass."""
        features = self.edge_sequence_features_from_patches(patch_batch)
        self.model.eval()
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.torch_device)
        contributions, energies = self._pair_contributions_and_energies_from_features(
            tensor,
            create_graph=False,
        )
        return (
            contributions.detach().cpu().numpy().astype(np.float32),
            energies.detach().cpu().numpy().astype(np.float32),
        )

    def predict_center_accelerations(self, patch_batch):
        """Predict central-cell accelerations by differentiating local energies."""
        contributions = self.predict_pair_contributions(patch_batch)
        return np.sum(contributions, axis=2).astype(np.float32)

    def predict_full_accelerations_and_energy(self, history, periodic=True, patch_batch_size=250, pair_scatter=None):
        """Return full-crystal accelerations and unique-pair potential in one pass."""
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 6:
            raise ValueError("history must have shape (sequence, nx, ny, nz, atoms, 3)")
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive")
        pair_scatter = self.pair_scatter_inference if pair_scatter is None else bool(pair_scatter)
        crystal_shape = tuple(int(dim) for dim in history.shape[1:4])
        scatter_indices = self._full_pair_scatter_indices(crystal_shape, periodic) if pair_scatter else None
        centers = scatter_indices["centers"] if pair_scatter else build_centers(crystal_shape, periodic=periodic)
        model_history = self._model_history_from_integration_history(history)
        acceleration = np.zeros_like(history[-1], dtype=np.float32)
        acceleration_flat = acceleration.reshape(-1, 3)
        total_energy = 0.0

        for start in range(0, len(centers), patch_batch_size):
            stop = min(start + patch_batch_size, len(centers))
            batch_centers = centers[start:stop]
            patch_batch = _extract_patch_batch(model_history, batch_centers, self.patch_shape, periodic)
            pair_contributions, pair_energies = self.predict_pair_contributions_and_energies(patch_batch)
            if not pair_scatter:
                self._assign_center_accelerations(acceleration, batch_centers, pair_contributions)
                total_energy += float(np.sum(pair_energies, dtype=np.float64))
                continue
            total_energy += self._scatter_pair_contributions(
                acceleration_flat,
                pair_contributions,
                scatter_indices,
                start,
                stop,
                pair_energies=pair_energies,
            )
        return acceleration, float(total_energy)

    def predict_full_potential_energy(self, history, periodic=True, patch_batch_size=250):
        """Return the unique-pair full-crystal potential in model units."""
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 6:
            raise ValueError("history must have shape (sequence, nx, ny, nz, atoms, 3)")
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive")
        crystal_shape = tuple(int(dim) for dim in history.shape[1:4])
        scatter_indices = self._full_pair_scatter_indices(crystal_shape, periodic)
        centers = scatter_indices["centers"]
        model_history = self._model_history_from_integration_history(history)
        total = 0.0

        for start in range(0, len(centers), patch_batch_size):
            stop = min(start + patch_batch_size, len(centers))
            batch_centers = centers[start:stop]
            patch_batch = _extract_patch_batch(model_history, batch_centers, self.patch_shape, periodic)
            pair_energies = self.predict_pair_energies(patch_batch)
            total += self._scatter_pair_contributions(
                None,
                None,
                scatter_indices,
                start,
                stop,
                pair_energies=pair_energies,
            )
        return float(total)


class CrystalEdgeFinalHiddenRNNNet(CrystalEdgeRNNNet):
    """Edge-RNN variant that reads the recurrent state like the original flat model."""

    def __init__(self, *args, **kwargs):
        kwargs["rnn_readout_mode"] = "final-hidden"
        super().__init__(*args, **kwargs)


class CrystalPairForceFinalHiddenRNNNet(CrystalPairForceRNNNet):
    """Pair-force RNN variant that uses final hidden states instead of output[:, -1]."""

    def __init__(self, *args, **kwargs):
        kwargs["rnn_readout_mode"] = "final-hidden"
        super().__init__(*args, **kwargs)


class CrystalPairEnergyFinalHiddenRNNNet(CrystalPairEnergyRNNNet):
    """Pair-energy RNN variant that uses final hidden states."""

    def __init__(self, *args, **kwargs):
        kwargs["rnn_readout_mode"] = "final-hidden"
        super().__init__(*args, **kwargs)
