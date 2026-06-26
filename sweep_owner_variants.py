"""Sweep deterministic owner merge orders for a saved crystal model."""

import argparse
import csv
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import torch

from base_classes import get_sqw


STEP = 10
DT = 0.02
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3


def parse_args():
    """Parse command-line options for owner-order sweep."""
    parser = argparse.ArgumentParser(description="Sweep owner merge axis orders and directions.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--count-steps", type=int, default=400)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--step", type=int, default=STEP)
    return parser.parse_args()


def build_default_k_vectors():
    """Build the default reciprocal-space vectors used for S(q,w) scoring."""
    kmin = 2 * np.pi / (NCELLS * LATTICE_PARAMETER)
    kmax = NCELLS * kmin
    kmas = np.zeros((KCOUNT, 3), dtype=np.float32)
    kmas[:, 0] = np.linspace(kmin, kmax, KCOUNT)
    return kmas


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacement frames to flat absolute positions."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def correlation_from_displacements(predicted, reference, reference_positions, atom_order, step):
    """Calculate S(q,w) correlation for predicted and reference displacements."""
    predicted_coords = crystal_frames_to_flat_positions(predicted, reference_positions, atom_order)
    reference_coords = crystal_frames_to_flat_positions(reference, reference_positions, atom_order)
    _, _, predicted_sqw = get_sqw(predicted_coords, dt=DT, step=step, kmas=build_default_k_vectors())
    _, _, reference_sqw = get_sqw(reference_coords, dt=DT, step=step, kmas=build_default_k_vectors())
    predicted_flat = predicted_sqw.reshape(-1)
    reference_flat = reference_sqw.reshape(-1)
    mask = np.isfinite(predicted_flat) & np.isfinite(reference_flat)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(predicted_flat[mask], reference_flat[mask])[0, 1])


def reverse_sets():
    """Yield all axis subsets that should be traversed in descending order."""
    axes = ("x", "y", "z")
    yield ()
    for size in range(1, len(axes) + 1):
        yield from combinations(axes, size)


def main():
    """Run the owner-order sweep and save a CSV table."""
    args = parse_args()
    data = np.load(args.data_path)
    sequence_length = int(data["X_blocks"].shape[1])
    init = data["displacements"][args.start_frame : args.start_frame + sequence_length].astype(np.float32)
    prediction_start = args.start_frame + sequence_length
    prediction_stop = prediction_start + args.count_steps
    if prediction_stop > data["displacements"].shape[0]:
        raise ValueError("Not enough frames for requested count_steps")
    reference = data["displacements"][prediction_start:prediction_stop].astype(np.float32)
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)

    rows = []
    total = 0
    for order in permutations(("x", "y", "z")):
        for reversed_axes in reverse_sets():
            total += 1
            order_string = "".join(order)
            reverse_string = "".join(reversed_axes) if reversed_axes else "-"
            print(f"[{total:02d}/48] owner_order={order_string} owner_reverse={reverse_string}")
            predicted = model.run_crystal(
                args.count_steps,
                init,
                merge_mode="owner",
                owner_order=order,
                owner_reverse=reversed_axes,
            )
            corr = correlation_from_displacements(
                predicted,
                reference,
                data["reference_positions"],
                data["atom_order"],
                args.step,
            )
            row = {
                "owner_order": order_string,
                "owner_reverse": reverse_string,
                "correlation": corr,
                "pred_std": float(predicted.std()),
                "ref_std": float(reference.std()),
                "first10_pred_std": float(predicted[:10].std()),
                "first10_ref_std": float(reference[:10].std()),
            }
            print(row)
            rows.append(row)

    rows.sort(key=lambda row: row["correlation"], reverse=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_csv}")
    print("TOP 5")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
