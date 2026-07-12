import tempfile
import unittest
from pathlib import Path

import numpy as np

from base_classes import CrystalPairEnergyMLPNet
from pipelines.mlp.ase.calculator import CrystalPairEnergyMLPCalculator


def fcc_reference(shape=(3, 3, 3), lattice_parameter=3.6):
    """Return conventional-cell FCC geometry and atom ordering."""
    basis = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ],
        dtype=np.float32,
    )
    atom_order = np.empty((*shape, len(basis)), dtype=np.int64)
    positions = []
    atom_index = 0
    for cell in np.ndindex(shape):
        for basis_index, offset in enumerate(basis):
            atom_order[(*cell, basis_index)] = atom_index
            positions.append((np.asarray(cell, dtype=np.float32) + offset) * lattice_parameter)
            atom_index += 1
    return (
        np.asarray(positions, dtype=np.float32),
        atom_order,
        np.asarray(shape, dtype=np.float32) * lattice_parameter,
    )


class StandalonePairEnergyMLPTest(unittest.TestCase):
    def make_model(self, size=16):
        reference_positions, atom_order, box_lengths = fcc_reference()
        return CrystalPairEnergyMLPNet(
            reference_positions=reference_positions,
            atom_order=atom_order,
            box_lengths=box_lengths,
            size=size,
            device="cpu",
        )

    def test_article_parameter_count(self):
        self.assertEqual(self.make_model(size=256).parameter_count, 67841)

    def test_one_frame_features_and_pair_scatter(self):
        model = self.make_model()
        displacements = np.zeros((3, 3, 3, 4, 3), dtype=np.float32)
        displacements[1, 1, 1, 0, 0] = 0.01

        features = model.pair_features_from_blocks(displacements[None])
        acceleration, energy = model.predict_full_accelerations_and_energy(displacements)

        self.assertEqual(features.shape, (1, 4, 18, 6))
        self.assertEqual(acceleration.shape, displacements.shape)
        np.testing.assert_allclose(acceleration.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-6)
        self.assertTrue(np.isfinite(energy))

    def test_rollout_requires_two_physical_frames(self):
        model = self.make_model()
        frames = np.zeros((2, 3, 3, 3, 4, 3), dtype=np.float32)
        prediction = model.rollout(2, frames)
        self.assertEqual(prediction.shape, (2, 3, 3, 3, 4, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            model.rollout(1, frames[:1])

    def test_portable_checkpoint_round_trip(self):
        model = self.make_model()
        model.acceleration_std = np.asarray(0.125, dtype=np.float32)
        blocks = np.zeros((1, 3, 3, 3, 4, 3), dtype=np.float32)
        expected = model.predict_center_accelerations(blocks)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "model.pth")
            restored = CrystalPairEnergyMLPNet.load(path, device="cpu")
        actual = restored.predict_center_accelerations(blocks)

        self.assertEqual(restored.size, model.size)
        self.assertAlmostEqual(float(restored.acceleration_std), 0.125)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_stateless_ase_calculator(self):
        try:
            from ase import Atoms
        except ImportError:
            self.skipTest("ASE is not installed")
        model = self.make_model()
        reference_positions, atom_order, box_lengths = fcc_reference()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = model.save(directory / "model.pth")
            data_path = directory / "data.npz"
            np.savez(
                data_path,
                reference_positions=reference_positions,
                atom_order=atom_order,
                box_lengths=box_lengths,
            )
            atoms = Atoms(
                symbols=["Cu"] * len(reference_positions),
                positions=reference_positions,
                cell=np.diag(box_lengths),
                pbc=True,
            )
            atoms.calc = CrystalPairEnergyMLPCalculator(
                model_path=model_path,
                data_path=data_path,
                dt_ps=0.02,
                device="cpu",
            )
            forces = atoms.get_forces()

        self.assertEqual(forces.shape, reference_positions.shape)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
