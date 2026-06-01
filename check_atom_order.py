"""Compare crystal atom ordering between two prepared `.npz` datasets."""

import argparse

import numpy as np


def parse_args():
    """Parse command-line options for atom-order diagnostics."""
    parser = argparse.ArgumentParser(description="Compare atom_in_cell ordering between two crystal datasets.")
    parser.add_argument("reference_npz", help="Reference .npz dataset, usually the training dataset.")
    parser.add_argument("candidate_npz", help="Candidate .npz dataset to compare against the reference.")
    parser.add_argument(
        "--warn-distance",
        type=float,
        default=0.15,
        help="Warn when a matched mean basis distance is larger than this value.",
    )
    parser.add_argument(
        "--warn-std",
        type=float,
        default=0.25,
        help="Warn when local fractional coordinate std is larger than this value.",
    )
    return parser.parse_args()


def load_dataset(path):
    """Load arrays required to diagnose crystal atom ordering."""
    data = np.load(path)
    required = ["atom_order", "reference_positions", "box_lengths"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")

    atom_order = data["atom_order"]
    crystal_shape = tuple(int(value) for value in data["crystal_shape"]) if "crystal_shape" in data.files else atom_order.shape[:3]
    box_lengths = data["box_lengths"][0] if data["box_lengths"].ndim == 2 else data["box_lengths"]
    reference_positions = data["reference_positions"]
    return {
        "path": path,
        "atom_order": atom_order,
        "crystal_shape": crystal_shape,
        "unit_cell_atoms": atom_order.shape[3],
        "box_lengths": box_lengths.astype(np.float64),
        "reference_positions": reference_positions.astype(np.float64),
    }


def circular_mean(values):
    """Return a periodic mean for fractional coordinates in [0, 1)."""
    angles = 2 * np.pi * values
    mean_angle = np.angle(np.mean(np.exp(1j * angles), axis=0))
    return (mean_angle / (2 * np.pi)) % 1.0


def periodic_delta(first, second):
    """Return shortest signed fractional displacement from second to first."""
    return (first - second + 0.5) % 1.0 - 0.5


def slot_basis_statistics(dataset):
    """Compute local fractional basis statistics for each atom_in_cell slot."""
    atom_order = dataset["atom_order"]
    reference_positions = dataset["reference_positions"]
    box_lengths = dataset["box_lengths"]
    crystal_shape = np.asarray(dataset["crystal_shape"], dtype=np.float64)
    unit_cell_atoms = dataset["unit_cell_atoms"]

    scaled = (reference_positions % box_lengths) / box_lengths * crystal_shape
    local = scaled - np.floor(scaled)

    slot_values = [[] for _ in range(unit_cell_atoms)]
    for crystal_index in np.ndindex(atom_order.shape[:3]):
        for atom_index in range(unit_cell_atoms):
            flat_atom_index = atom_order[(*crystal_index, atom_index)]
            slot_values[atom_index].append(local[flat_atom_index])

    means = []
    stds = []
    max_deviations = []
    for values in slot_values:
        values = np.asarray(values, dtype=np.float64)
        mean = circular_mean(values)
        deltas = periodic_delta(values, mean)
        means.append(mean)
        stds.append(deltas.std(axis=0))
        max_deviations.append(np.max(np.abs(deltas), axis=0))

    return np.asarray(means), np.asarray(stds), np.asarray(max_deviations)


def distance_matrix(reference_means, candidate_means):
    """Compute periodic distances between basis slots."""
    deltas = periodic_delta(reference_means[:, None, :], candidate_means[None, :, :])
    return np.linalg.norm(deltas, axis=2)


def greedy_assignment(distances):
    """Match reference slots to candidate slots by greedy minimum distance."""
    remaining_reference = set(range(distances.shape[0]))
    remaining_candidate = set(range(distances.shape[1]))
    assignment = {}

    while remaining_reference and remaining_candidate:
        best = None
        for ref_slot in remaining_reference:
            for candidate_slot in remaining_candidate:
                value = distances[ref_slot, candidate_slot]
                if best is None or value < best[0]:
                    best = (value, ref_slot, candidate_slot)

        _, ref_slot, candidate_slot = best
        assignment[ref_slot] = candidate_slot
        remaining_reference.remove(ref_slot)
        remaining_candidate.remove(candidate_slot)

    return assignment


def print_dataset_summary(label, dataset, means, stds, max_deviations, warn_std):
    """Print per-slot basis diagnostics for one dataset."""
    print(f"\n{label}: {dataset['path']}")
    print(f"  crystal_shape = {dataset['crystal_shape']}")
    print(f"  atom_order.shape = {dataset['atom_order'].shape}")
    print(f"  unit_cell_atoms = {dataset['unit_cell_atoms']}")
    print("  first cell atom ids =", dataset["atom_order"][0, 0, 0].tolist())
    print("  slot basis statistics:")
    for slot_index, (mean, std, max_dev) in enumerate(zip(means, stds, max_deviations)):
        warning = "  <-- high periodic spread" if np.any(std > warn_std) else ""
        print(
            f"    slot {slot_index}: "
            f"mean={np.round(mean, 6).tolist()} "
            f"std={np.round(std, 6).tolist()} "
            f"max_dev={np.round(max_dev, 6).tolist()}"
            f"{warning}"
        )


def main():
    """Compare atom_in_cell slot ordering between two crystal datasets."""
    args = parse_args()
    reference = load_dataset(args.reference_npz)
    candidate = load_dataset(args.candidate_npz)

    if reference["unit_cell_atoms"] != candidate["unit_cell_atoms"]:
        raise ValueError("Datasets have different unit_cell_atoms")

    reference_means, reference_stds, reference_max_devs = slot_basis_statistics(reference)
    candidate_means, candidate_stds, candidate_max_devs = slot_basis_statistics(candidate)

    print_dataset_summary("REFERENCE", reference, reference_means, reference_stds, reference_max_devs, args.warn_std)
    print_dataset_summary("CANDIDATE", candidate, candidate_means, candidate_stds, candidate_max_devs, args.warn_std)

    distances = distance_matrix(reference_means, candidate_means)
    assignment = greedy_assignment(distances)

    print("\nPeriodic distance matrix: reference slots as rows, candidate slots as columns")
    print(np.array2string(distances, precision=6, suppress_small=True))

    print("\nSuggested slot mapping to make candidate match reference:")
    max_matched_distance = 0.0
    mapping = []
    for ref_slot in range(reference["unit_cell_atoms"]):
        candidate_slot = assignment[ref_slot]
        distance = float(distances[ref_slot, candidate_slot])
        max_matched_distance = max(max_matched_distance, distance)
        mapping.append(candidate_slot)
        warning = "  <-- large distance" if distance > args.warn_distance else ""
        print(f"  reference slot {ref_slot} <- candidate slot {candidate_slot}, distance={distance:.6f}{warning}")

    print(f"\nCandidate reorder index: {mapping}")
    if mapping == list(range(reference["unit_cell_atoms"])):
        print("Direct slot order looks compatible.")
    else:
        print("Direct slot order differs; candidate atom_in_cell axis should be reordered before model inference.")

    if max_matched_distance > args.warn_distance:
        print("WARNING: even the best slot mapping has large distances; check wrapping, basis, or reference positions.")
    if np.any(reference_stds > args.warn_std) or np.any(candidate_stds > args.warn_std):
        print("WARNING: high slot spread detected; periodic wrapping or mean reference positions may need attention.")


if __name__ == "__main__":
    main()
