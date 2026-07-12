"""One-frame conservative pair-energy MLP for fixed-frame crystal dynamics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(device):
    """Return a concrete torch device."""
    if str(device).lower() != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _first_box_lengths(box_lengths):
    """Return one orthorhombic box-length vector."""
    values = np.asarray(box_lengths, dtype=np.float32)
    if values.ndim == 2:
        values = values[0]
    if values.shape != (3,):
        raise ValueError("box_lengths must have shape (3,) or (frames, 3)")
    return values


def _minimum_image(delta, box_lengths):
    """Return minimum-image vectors for an orthorhombic box."""
    return delta - box_lengths * np.round(delta / box_lengths)


def _build_fcc_stencil(
    reference_positions,
    atom_order,
    box_lengths,
    neighbor_shells=2,
    cutoff_scale=1.05,
    shell_tolerance=0.08,
):
    """Build the fixed 18-neighbor FCC stencil for central-cell atoms."""
    reference_positions = np.asarray(reference_positions, dtype=np.float32)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    box_lengths = _first_box_lengths(box_lengths)
    if atom_order.ndim != 4 or tuple(atom_order.shape[:3]) != (3, 3, 3):
        raise ValueError("atom_order must describe a 3x3x3 training block")
    if int(neighbor_shells) != 2:
        raise ValueError("CrystalPairEnergyMLPNet supports exactly two FCC neighbor shells")

    crystal_shape = np.asarray(atom_order.shape[:3], dtype=np.float32)
    lattice_parameter = float(np.mean(box_lengths / crystal_shape))
    cutoff = float(cutoff_scale) * lattice_parameter
    center_cell = (1, 1, 1)
    neighbor_indices = []
    reference_vectors = []
    shell_ids = []

    for center_atom in range(atom_order.shape[3]):
        center_flat = atom_order[center_cell + (center_atom,)]
        center_position = reference_positions[center_flat]
        rows = []
        for cell_index in np.ndindex(atom_order.shape[:3]):
            for neighbor_atom in range(atom_order.shape[3]):
                if cell_index == center_cell and neighbor_atom == center_atom:
                    continue
                neighbor_flat = atom_order[cell_index + (neighbor_atom,)]
                vector = _minimum_image(
                    reference_positions[neighbor_flat] - center_position,
                    box_lengths,
                )
                distance = float(np.linalg.norm(vector))
                if 1e-6 < distance <= cutoff:
                    shell = 1 if distance < (1.0 - shell_tolerance) * lattice_parameter else 2
                    rows.append(
                        (
                            shell,
                            distance,
                            vector[0],
                            vector[1],
                            vector[2],
                            cell_index,
                            neighbor_atom,
                            vector,
                        )
                    )
        rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        if len(rows) != 18:
            raise ValueError(
                f"Expected 18 FCC neighbors for center atom {center_atom}, got {len(rows)}"
            )
        neighbor_indices.append([(*row[5], row[6]) for row in rows])
        reference_vectors.append([row[7] for row in rows])
        shell_ids.append([row[0] for row in rows])

    return {
        "neighbor_indices": np.asarray(neighbor_indices, dtype=np.int64),
        "reference_vectors": np.asarray(reference_vectors, dtype=np.float32),
        "shell_ids": np.asarray(shell_ids, dtype=np.int64),
        "lattice_parameter": lattice_parameter,
    }


def _extract_block_batch(displacements, centers, periodic):
    """Extract 3x3x3 blocks centered on crystal cells."""
    shape = np.asarray(displacements.shape[:3], dtype=np.int64)
    blocks = []
    for center in centers:
        axes = []
        for axis, coordinate in enumerate(center):
            indices = np.arange(int(coordinate) - 1, int(coordinate) + 2)
            if periodic:
                indices %= shape[axis]
            elif np.any((indices < 0) | (indices >= shape[axis])):
                raise ValueError("Non-periodic inference center is too close to a boundary")
            axes.append(indices)
        blocks.append(displacements[np.ix_(axes[0], axes[1], axes[2])])
    return np.asarray(blocks, dtype=np.float32)


class _EvenPairEnergyMLP(nn.Module):
    """Map one six-channel pair feature to an exchange-even scalar energy."""

    def __init__(self, size):
        super().__init__()
        self.size = int(size)
        self.raw_model = nn.Sequential(
            nn.Linear(6, self.size),
            nn.ELU(inplace=True),
            nn.Linear(self.size, self.size),
            nn.ELU(inplace=True),
            nn.Linear(self.size, 1),
        )

    def raw_forward(self, features):
        """Return unconstrained scalar pair energies."""
        return self.raw_model(features).squeeze(-1)

    def forward(self, features):
        """Return pair energies invariant under exchanging pair orientation."""
        return 0.5 * (self.raw_forward(features) + self.raw_forward(-features))


class CrystalPairEnergyMLPNet:
    """Conservative one-frame crystal force operator used by the MLP article.

    The network consumes one current displacement field. Each central-cell atom
    is represented by 18 fixed FCC pair features
    ``[R_ref / a0, (u_neighbor - u_center) / a0]``. A shared MLP predicts an
    even scalar pair energy and forces are obtained through autograd.
    """

    architecture = "pair-energy-mlp"
    input_mode = "ref-plus-delta"
    patch_shape = (3, 3, 3)

    def __init__(
        self,
        reference_positions,
        atom_order,
        box_lengths,
        size=256,
        neighbor_shells=2,
        cutoff_scale=1.05,
        device="auto",
    ):
        self.reference_positions = np.asarray(reference_positions, dtype=np.float32)
        self.atom_order = np.asarray(atom_order, dtype=np.int64)
        self.box_lengths = _first_box_lengths(box_lengths).astype(np.float32)
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("size must be positive")
        self.neighbor_shells = int(neighbor_shells)
        self.cutoff_scale = float(cutoff_scale)
        self.torch_device = _resolve_device(device)
        self.unit_cell_atoms = int(self.atom_order.shape[3])
        self.stencil = _build_fcc_stencil(
            self.reference_positions,
            self.atom_order,
            self.box_lengths,
            neighbor_shells=self.neighbor_shells,
            cutoff_scale=self.cutoff_scale,
        )
        self.neighbor_indices = self.stencil["neighbor_indices"]
        self.reference_vectors = self.stencil["reference_vectors"]
        self.shell_ids = self.stencil["shell_ids"]
        self.lattice_parameter = float(self.stencil["lattice_parameter"])
        self.neighbor_count = int(self.neighbor_indices.shape[1])
        self.model = _EvenPairEnergyMLP(self.size).to(self.torch_device)
        self.acceleration_mean = np.asarray(0.0, dtype=np.float32)
        self.acceleration_std = np.asarray(1.0, dtype=np.float32)
        self.energy_output_scale = np.asarray(1.0, dtype=np.float32)
        self.reference_pressure_loss_weight = 0.0
        self.reference_pressure_target = 0.0
        self.reference_pressure_loss_scale = 1.0
        self._scatter_cache = {}

    @property
    def parameter_count(self):
        """Return the number of trainable neural-network parameters."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    def to(self, device):
        """Move the neural network to a torch device."""
        self.torch_device = _resolve_device(device)
        self.model.to(self.torch_device)
        return self

    def checkpoint(self):
        """Return a portable checkpoint without pickling the Python object."""
        return {
            "format": "crystal-pair-energy-mlp-v1",
            "config": {
                "reference_positions": self.reference_positions,
                "atom_order": self.atom_order,
                "box_lengths": self.box_lengths,
                "size": self.size,
                "neighbor_shells": self.neighbor_shells,
                "cutoff_scale": self.cutoff_scale,
            },
            "state_dict": self.model.state_dict(),
            "training": {
                "acceleration_mean": self.acceleration_mean,
                "acceleration_std": self.acceleration_std,
                "energy_output_scale": self.energy_output_scale,
                "reference_pressure_loss_weight": self.reference_pressure_loss_weight,
                "reference_pressure_target": self.reference_pressure_target,
                "reference_pressure_loss_scale": self.reference_pressure_loss_scale,
            },
        }

    def save(self, path):
        """Save a portable model checkpoint and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)
        return path

    @classmethod
    def load(cls, path, device="auto"):
        """Load a model saved by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "crystal-pair-energy-mlp-v1":
            raise ValueError("Not a CrystalPairEnergyMLPNet checkpoint")
        model = cls(**checkpoint["config"], device=device)
        model.model.load_state_dict(checkpoint["state_dict"])
        for name, value in checkpoint.get("training", {}).items():
            setattr(model, name, value)
        model.model.eval()
        return model

    def pair_features_from_blocks(self, blocks):
        """Return pair features from one-frame 3x3x3 displacement blocks."""
        blocks = np.asarray(blocks, dtype=np.float32)
        if blocks.ndim != 6 or tuple(blocks.shape[1:4]) != self.patch_shape:
            raise ValueError("blocks must have shape (batch, 3, 3, 3, atoms, 3)")
        if blocks.shape[4] != self.unit_cell_atoms or blocks.shape[5] != 3:
            raise ValueError("block atom dimensions do not match model metadata")
        features = np.empty(
            (len(blocks), self.unit_cell_atoms, self.neighbor_count, 6),
            dtype=np.float32,
        )
        center_displacements = blocks[:, 1, 1, 1]
        for atom_index in range(self.unit_cell_atoms):
            center = center_displacements[:, atom_index]
            for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                lx, ly, lz, neighbor_atom = (int(value) for value in local_index)
                delta = blocks[:, lx, ly, lz, neighbor_atom] - center
                features[:, atom_index, neighbor_index, :3] = (
                    self.reference_vectors[atom_index, neighbor_index] / self.lattice_parameter
                )
                features[:, atom_index, neighbor_index, 3:] = delta / self.lattice_parameter
        return features

    def _pair_energies(self, features):
        """Return scaled pair energies from differentiable features."""
        flat = features.reshape(-1, 6)
        raw = self.model(flat).reshape(features.shape[:-1])
        scale = torch.as_tensor(self.energy_output_scale, dtype=raw.dtype, device=raw.device)
        return raw * scale

    def _pair_contributions(self, features, create_graph=None):
        """Differentiate pair energies into central-atom acceleration terms."""
        if not features.requires_grad:
            features = features.detach().clone().requires_grad_(True)
        if create_graph is None:
            create_graph = bool(torch.is_grad_enabled())
        with torch.enable_grad():
            energies = self._pair_energies(features)
            gradient = torch.autograd.grad(
                energies.sum(),
                features,
                create_graph=create_graph,
                retain_graph=create_graph,
                only_inputs=True,
            )[0]
        return gradient[..., 3:6] / self.lattice_parameter

    def _pair_contributions_and_energies(self, features, create_graph=False):
        """Return pair accelerations and energies from one autograd pass."""
        if not features.requires_grad:
            features = features.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            energies = self._pair_energies(features)
            gradient = torch.autograd.grad(
                energies.sum(),
                features,
                create_graph=create_graph,
                retain_graph=create_graph,
                only_inputs=True,
            )[0]
        return gradient[..., 3:6] / self.lattice_parameter, energies

    def predict_center_accelerations(self, blocks):
        """Predict accelerations of all atoms in each central unit cell."""
        features = torch.as_tensor(
            self.pair_features_from_blocks(blocks),
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.model.eval()
        contributions = self._pair_contributions(features, create_graph=False)
        return contributions.sum(dim=2).detach().cpu().numpy().astype(np.float32)

    def _crystal_atom_id(self, cell, atom_index, crystal_shape):
        """Return a flat atom id for a crystal cell and basis atom."""
        ix, iy, iz = (int(value) for value in cell)
        _, ny, nz = (int(value) for value in crystal_shape)
        return (((ix * ny) + iy) * nz + iz) * self.unit_cell_atoms + int(atom_index)

    def _scatter_indices(self, crystal_shape, periodic):
        """Return cached unique periodic pair-scatter indices."""
        key = (tuple(int(value) for value in crystal_shape), bool(periodic))
        if key in self._scatter_cache:
            return self._scatter_cache[key]
        crystal_shape = key[0]
        if periodic:
            centers = list(np.ndindex(crystal_shape))
        else:
            centers = list(
                np.ndindex(tuple(max(0, value - 2) for value in crystal_shape))
            )
            centers = [tuple(value + 1 for value in center) for center in centers]
        center_rows = []
        atom_rows = []
        neighbor_rows = []
        center_flat_rows = []
        neighbor_flat_rows = []
        shape_array = np.asarray(crystal_shape, dtype=np.int64)
        for center_index, center in enumerate(centers):
            for atom_index in range(self.unit_cell_atoms):
                center_id = self._crystal_atom_id(center, atom_index, crystal_shape)
                for neighbor_index, local_index in enumerate(self.neighbor_indices[atom_index]):
                    neighbor_cell = np.asarray(center, dtype=np.int64) + np.asarray(local_index[:3]) - 1
                    if periodic:
                        neighbor_cell %= shape_array
                    elif np.any((neighbor_cell < 0) | (neighbor_cell >= shape_array)):
                        continue
                    neighbor_atom = int(local_index[3])
                    neighbor_cell = tuple(int(value) for value in neighbor_cell)
                    neighbor_id = self._crystal_atom_id(neighbor_cell, neighbor_atom, crystal_shape)
                    if center_id >= neighbor_id:
                        continue
                    center_rows.append(center_index)
                    atom_rows.append(atom_index)
                    neighbor_rows.append(neighbor_index)
                    center_flat_rows.append(center_id)
                    neighbor_flat_rows.append(neighbor_id)
        result = {
            "centers": centers,
            "center_index": np.asarray(center_rows, dtype=np.int64),
            "atom": np.asarray(atom_rows, dtype=np.int64),
            "neighbor": np.asarray(neighbor_rows, dtype=np.int64),
            "center_flat": np.asarray(center_flat_rows, dtype=np.int64),
            "neighbor_flat": np.asarray(neighbor_flat_rows, dtype=np.int64),
        }
        self._scatter_cache[key] = result
        return result

    def predict_full_accelerations_and_energy(
        self,
        displacements,
        periodic=True,
        block_batch_size=250,
    ):
        """Predict a full acceleration field and unique-pair model energy."""
        displacements = np.asarray(displacements, dtype=np.float32)
        if displacements.ndim != 5 or displacements.shape[-2:] != (self.unit_cell_atoms, 3):
            raise ValueError("displacements must have shape (nx, ny, nz, atoms, 3)")
        if block_batch_size <= 0:
            raise ValueError("block_batch_size must be positive")
        scatter = self._scatter_indices(displacements.shape[:3], periodic)
        acceleration = np.zeros_like(displacements, dtype=np.float32)
        acceleration_flat = acceleration.reshape(-1, 3)
        total_energy = 0.0
        self.model.eval()

        for start in range(0, len(scatter["centers"]), block_batch_size):
            stop = min(start + block_batch_size, len(scatter["centers"]))
            centers = scatter["centers"][start:stop]
            blocks = _extract_block_batch(displacements, centers, periodic)
            features = torch.as_tensor(
                self.pair_features_from_blocks(blocks),
                dtype=torch.float32,
                device=self.torch_device,
            )
            contributions, energies = self._pair_contributions_and_energies(features)
            contributions = contributions.detach().cpu().numpy()
            energies = energies.detach().cpu().numpy()
            mask = (scatter["center_index"] >= start) & (scatter["center_index"] < stop)
            batch_indices = scatter["center_index"][mask] - start
            atom_indices = scatter["atom"][mask]
            neighbor_indices = scatter["neighbor"][mask]
            center_flat = scatter["center_flat"][mask]
            neighbor_flat = scatter["neighbor_flat"][mask]
            selected = contributions[batch_indices, atom_indices, neighbor_indices]
            np.add.at(acceleration_flat, center_flat, selected)
            np.add.at(acceleration_flat, neighbor_flat, -selected)
            total_energy += float(np.sum(energies[batch_indices, atom_indices, neighbor_indices]))
        return acceleration, total_energy

    def predict_full_accelerations(self, displacements, periodic=True, block_batch_size=250):
        """Predict one full acceleration field from one current frame."""
        acceleration, _ = self.predict_full_accelerations_and_energy(
            displacements,
            periodic=periodic,
            block_batch_size=block_batch_size,
        )
        return acceleration

    def predict_full_potential_energy(self, displacements, periodic=True, block_batch_size=250):
        """Predict the unique-pair model energy of one current frame."""
        _, energy = self.predict_full_accelerations_and_energy(
            displacements,
            periodic=periodic,
            block_batch_size=block_batch_size,
        )
        return energy

    def rollout(self, count_steps, initial_displacements, periodic=True, block_batch_size=250):
        """Run discrete Verlet dynamics from previous and current frames."""
        initial_displacements = np.asarray(initial_displacements, dtype=np.float32)
        if initial_displacements.ndim != 6 or initial_displacements.shape[0] != 2:
            raise ValueError("initial_displacements must have shape (2, nx, ny, nz, atoms, 3)")
        previous = initial_displacements[0].copy()
        current = initial_displacements[1].copy()
        predictions = []
        for _ in range(int(count_steps)):
            acceleration = self.predict_full_accelerations(
                current,
                periodic=periodic,
                block_batch_size=block_batch_size,
            )
            next_frame = 2.0 * current - previous + acceleration
            predictions.append(next_frame.astype(np.float32))
            previous, current = current, next_frame
        return np.asarray(predictions, dtype=np.float32)

    def _set_acceleration_scale(self, targets):
        """Store global acceleration normalization and energy-output scale."""
        targets = np.asarray(targets, dtype=np.float32)
        self.acceleration_mean = np.asarray(float(np.mean(targets)), dtype=np.float32)
        std = max(float(np.std(targets)), 1e-12)
        self.acceleration_std = np.asarray(std, dtype=np.float32)
        energy_scale = std * self.lattice_parameter / np.sqrt(max(1, self.neighbor_count))
        self.energy_output_scale = np.asarray(max(energy_scale, 1e-12), dtype=np.float32)

    def _reference_features(self):
        """Return differentiable pair features at zero displacement."""
        reference = torch.as_tensor(
            self.reference_vectors / self.lattice_parameter,
            dtype=torch.float32,
            device=self.torch_device,
        )
        zeros = torch.zeros_like(reference)
        return torch.cat((reference, zeros), dim=-1).unsqueeze(0)

    def _unit_cell_volume(self):
        """Return the reference conventional-cell volume in Angstrom^3."""
        unit_lengths = self.box_lengths / np.asarray(self.atom_order.shape[:3], dtype=np.float32)
        return float(np.prod(unit_lengths))

    def reference_pressure_loss(self):
        """Return the optional zero-pressure anchor at the reference lattice."""
        contributions = self._pair_contributions(self._reference_features(), create_graph=True)[0]
        reference_vectors = torch.as_tensor(
            self.reference_vectors,
            dtype=contributions.dtype,
            device=contributions.device,
        )
        virial = 0.5 * torch.einsum("ani,anj->ij", reference_vectors, contributions)
        pressure = -torch.trace(virial) / (3.0 * self._unit_cell_volume())
        target = torch.as_tensor(
            self.reference_pressure_target,
            dtype=pressure.dtype,
            device=pressure.device,
        )
        return ((pressure - target) / self.reference_pressure_loss_scale) ** 2

    def fit(
        self,
        input_blocks,
        force_acceleration_blocks,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        shuffle=True,
    ):
        """Fit the one-frame MLP to central-cell force-derived accelerations."""
        input_blocks = np.asarray(input_blocks, dtype=np.float32)
        targets = np.asarray(force_acceleration_blocks, dtype=np.float32)
        if input_blocks.ndim != 6:
            raise ValueError("input_blocks must have shape (samples, 3, 3, 3, atoms, 3)")
        if targets.ndim != 6 or targets.shape[0] != input_blocks.shape[0]:
            raise ValueError("force_acceleration_blocks must match input_blocks samples")
        center_targets = targets[:, 1, 1, 1]
        self._set_acceleration_scale(center_targets)
        pair_features = self.pair_features_from_blocks(input_blocks)
        dataset = TensorDataset(
            torch.as_tensor(pair_features, dtype=torch.float32),
            torch.as_tensor(center_targets, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle))
        optimizer = optim.Adam(self.model.parameters(), lr=float(learning_rate))
        mean = torch.as_tensor(self.acceleration_mean, dtype=torch.float32, device=self.torch_device)
        std = torch.as_tensor(self.acceleration_std, dtype=torch.float32, device=self.torch_device)
        losses = []
        self.model.train()
        for _ in range(int(epochs)):
            loss_sum = 0.0
            batch_count = 0
            for features, reference in loader:
                features = features.to(self.torch_device)
                reference = reference.to(self.torch_device)
                prediction = self._pair_contributions(features).sum(dim=2)
                loss = nn.functional.mse_loss((prediction - mean) / std, (reference - mean) / std)
                if self.reference_pressure_loss_weight > 0:
                    loss = loss + self.reference_pressure_loss_weight * self.reference_pressure_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                batch_count += 1
            losses.append(loss_sum / max(1, batch_count))
        return losses
