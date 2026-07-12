#!/usr/bin/env python3
"""Convert a legacy temporal-MLP pair-energy checkpoint to the standalone format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from base_classes import CrystalPairEnergyMLPNet  # noqa: E402


def parse_args():
    """Parse checkpoint paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_path")
    parser.add_argument("output_path")
    return parser.parse_args()


def main():
    """Map the equivalent legacy MLP layers and save a portable checkpoint."""
    args = parse_args()
    legacy = torch.load(args.legacy_path, map_location="cpu", weights_only=False)
    expected = {
        "architecture": "pair-energy",
        "temporal_architecture": "mlp",
        "temporal_input_mode": "ref-plus-delta",
        "rnn_layers": 1,
    }
    for name, value in expected.items():
        if getattr(legacy, name, None) != value:
            raise ValueError(f"Legacy checkpoint {name} must be {value!r}")
    if hasattr(legacy, "to"):
        legacy.to("cpu")
    legacy_state = legacy.model.state_dict()
    layer_map = {
        "raw_model.0.weight": "mlp_encoder.0.weight",
        "raw_model.0.bias": "mlp_encoder.0.bias",
        "raw_model.2.weight": "head.0.weight",
        "raw_model.2.bias": "head.0.bias",
        "raw_model.4.weight": "head.2.weight",
        "raw_model.4.bias": "head.2.bias",
    }
    size = int(legacy_state["mlp_encoder.0.weight"].shape[0])
    model = CrystalPairEnergyMLPNet(
        reference_positions=legacy.reference_positions,
        atom_order=legacy.atom_order,
        box_lengths=legacy.box_lengths,
        size=size,
        neighbor_shells=legacy.neighbor_shells,
        cutoff_scale=legacy.cutoff_scale,
        device="cpu",
    )
    standalone_state = model.model.state_dict()
    for destination, source in layer_map.items():
        if standalone_state[destination].shape != legacy_state[source].shape:
            raise ValueError(f"Layer shape mismatch: {source} -> {destination}")
        standalone_state[destination] = legacy_state[source]
    model.model.load_state_dict(standalone_state)
    for name in (
        "acceleration_mean",
        "acceleration_std",
        "energy_output_scale",
        "reference_pressure_loss_weight",
        "reference_pressure_target",
        "reference_pressure_loss_scale",
    ):
        if hasattr(legacy, name):
            setattr(model, name, getattr(legacy, name))
    rng = np.random.default_rng(20260712)
    patches = rng.normal(0.0, 0.01, size=(2, 1, 3, 3, 3, model.unit_cell_atoms, 3)).astype(np.float32)
    legacy_prediction = legacy.predict_center_accelerations(patches)
    standalone_prediction = model.predict_center_accelerations(patches[:, 0])
    maximum_error = float(np.max(np.abs(legacy_prediction - standalone_prediction)))
    if maximum_error > 1e-6:
        raise RuntimeError(f"Conversion validation failed: maximum acceleration error {maximum_error:g}")
    output = model.save(args.output_path)
    print(f"Saved {output}")
    print(f"size = {size}")
    print(f"parameter_count = {model.parameter_count}")
    print(f"maximum_validation_error = {maximum_error:.8g}")


if __name__ == "__main__":
    main()
