"""Stateless ASE calculator for the standalone pair-energy MLP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from base_classes import (  # noqa: E402
    AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM,
    CrystalPairEnergyMLPNet,
)

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:  # pragma: no cover
    Calculator = object
    all_changes = ("positions", "numbers", "cell", "pbc")


def _cell_matrix(data):
    """Return the orthorhombic reference cell from an MLP dataset."""
    if "cell" in data:
        cell = np.asarray(data["cell"], dtype=np.float64)
        if cell.ndim == 3:
            cell = cell[0]
        return np.diag(cell) if cell.shape == (3,) else cell
    lengths = np.asarray(data["box_lengths"], dtype=np.float64)
    if lengths.ndim == 2:
        lengths = lengths[0]
    return np.diag(lengths)


class CrystalPairEnergyMLPCalculator(Calculator):
    """Convert one-frame model accelerations and energies to ASE units."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path,
        data_path,
        dt_ps=0.002,
        device="auto",
        periodic=True,
        block_batch_size=250,
        enforce_cell=True,
        **kwargs,
    ):
        if Calculator is object:
            raise ImportError("ASE is required to use CrystalPairEnergyMLPCalculator")
        super().__init__(**kwargs)
        if dt_ps <= 0:
            raise ValueError("dt_ps must be positive")
        if block_batch_size <= 0:
            raise ValueError("block_batch_size must be positive")
        with np.load(data_path) as source:
            data = {key: source[key] for key in source.files}
        for key in ("reference_positions", "atom_order", "box_lengths"):
            if key not in data:
                raise ValueError(f"Dataset is missing {key!r}")
        self.model = CrystalPairEnergyMLPNet.load(model_path, device=device)
        self.reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
        self.atom_order = np.asarray(data["atom_order"], dtype=np.int64)
        self.reference_cell = _cell_matrix(data)
        self.dt_ps = float(dt_ps)
        self.periodic = bool(periodic)
        self.block_batch_size = int(block_batch_size)
        self.enforce_cell = bool(enforce_cell)
        self.atom_count = int(self.atom_order.size)
        if self.reference_positions.shape != (self.atom_count, 3):
            raise ValueError("reference_positions and atom_order describe different atom counts")
        if self.model.unit_cell_atoms != self.atom_order.shape[3]:
            raise ValueError("Model and dataset use different unit-cell atom counts")

    def _crystal_to_flat(self, values):
        """Convert crystal-layout vectors to ASE atom order."""
        flat = np.empty((self.atom_count, 3), dtype=np.float64)
        flat[self.atom_order.reshape(-1)] = np.asarray(values).reshape(self.atom_count, 3)
        return flat

    def _displacements(self, atoms):
        """Convert current ASE positions to minimum-image crystal displacements."""
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        delta = positions - self.reference_positions
        inverse_cell = np.linalg.inv(self.reference_cell)
        fractional = delta @ inverse_cell
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        fractional[:, pbc] -= np.round(fractional[:, pbc])
        return (fractional @ self.reference_cell)[self.atom_order].astype(np.float32)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        """Evaluate model energy and forces for the current configuration."""
        super().calculate(atoms, properties, system_changes)
        if len(atoms) != self.atom_count:
            raise ValueError(f"Expected {self.atom_count} atoms, got {len(atoms)}")
        if self.enforce_cell and not np.allclose(
            np.asarray(atoms.get_cell()), self.reference_cell, atol=1e-5, rtol=1e-6
        ):
            raise ValueError("ASE cell differs from the dataset reference cell")
        accelerations, model_energy = self.model.predict_full_accelerations_and_energy(
            self._displacements(atoms),
            periodic=self.periodic,
            block_batch_size=self.block_batch_size,
        )
        acceleration_flat = self._crystal_to_flat(accelerations)
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        conversion = AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM / self.dt_ps**2
        forces = acceleration_flat * masses[:, None] * conversion
        if not np.allclose(masses, masses[0]):
            raise ValueError("The scalar pair energy conversion currently requires one atomic mass")
        energy = float(model_energy) * float(masses[0]) * conversion
        self.results = {"energy": energy, "forces": forces}
