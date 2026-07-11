import unittest

import numpy as np

from base_classes.edge_rnn_predictor import CrystalPairEnergyRNNNet


def fcc_reference(shape=(3, 3, 3), lattice_parameter=3.6):
    """Return reference geometry for a conventional-cell FCC supercell."""
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
        origin = np.asarray(cell, dtype=np.float32)
        for basis_index, offset in enumerate(basis):
            atom_order[(*cell, basis_index)] = atom_index
            positions.append((origin + offset) * lattice_parameter)
            atom_index += 1
    box_lengths = np.asarray(shape, dtype=np.float32) * lattice_parameter
    return np.asarray(positions, dtype=np.float32), atom_order, box_lengths


class PairEnergyMLPTest(unittest.TestCase):
    def make_model(self):
        reference_positions, atom_order, box_lengths = fcc_reference()
        return CrystalPairEnergyRNNNet(
            reference_positions=reference_positions,
            atom_order=atom_order,
            box_lengths=box_lengths,
            hidden_size=16,
            rnn_layers=1,
            type="RNN",
            bidirectional=True,
            neighbor_shells=2,
            cutoff_scale=1.05,
            acceleration_normalization="global",
            temporal_architecture="mlp",
            temporal_input_mode="ref-plus-delta",
            device="cpu",
        )

    def test_full_inference_uses_latest_integration_frame(self):
        model = self.make_model()
        history = np.zeros((3, 3, 3, 3, 4, 3), dtype=np.float32)
        history[-1, 1, 1, 1, 0, 0] = 0.01

        acceleration = model.predict_full_accelerations(
            history,
            periodic=True,
            patch_batch_size=27,
        )
        acceleration_with_energy, energy = model.predict_full_accelerations_and_energy(
            history,
            periodic=True,
            patch_batch_size=27,
        )

        self.assertEqual(acceleration.shape, history.shape[1:])
        np.testing.assert_allclose(acceleration, acceleration_with_energy, atol=1e-6)
        np.testing.assert_allclose(acceleration.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-5)
        self.assertTrue(np.isfinite(energy))

    def test_one_frame_force_training_with_pressure_anchor(self):
        model = self.make_model()
        patches = np.zeros((2, 1, 3, 3, 3, 4, 3), dtype=np.float32)
        targets = np.zeros((2, 3, 3, 3, 4, 3), dtype=np.float32)
        targets[:, 1, 1, 1] = np.linspace(-0.1, 0.1, 24, dtype=np.float32).reshape(2, 4, 3)

        model.batch_size = 1
        model.epochs = 1
        model.reference_pressure_loss_weight = 1.0
        losses = model.train_crystal_blocks(
            patches,
            targets,
            data_len=1.0,
            training_target="force",
        )

        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))
        self.assertTrue(np.isfinite(float(model._reference_pressure_loss(1).detach())))


if __name__ == "__main__":
    unittest.main()
