"""Plot predicted and reference S(q,w) maps from an inference output file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base_classes import get_sqw


STEP = 10
DT = 0.02
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3


def parse_args():
    """Parse command-line options for S(q,w) comparison plotting."""
    parser = argparse.ArgumentParser(description="Plot predicted/reference S(q,w) maps side by side.")
    parser.add_argument("--input-path", required=True, help="Path to an inference output .npz file.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Prepared crystal .npz dataset. Required for ASE outputs that contain positions only.",
    )
    parser.add_argument(
        "--output-path",
        default="sqw_comparison.png",
        help="Path to the output image. Use an extension supported by Matplotlib.",
    )
    parser.add_argument("--dt", type=float, default=DT, help="Time step between trajectory frames.")
    parser.add_argument("--step", type=int, default=STEP, help="Frame stride used by get_sqw.")
    parser.add_argument("--lattice-parameter", type=float, default=LATTICE_PARAMETER)
    parser.add_argument("--ncells", type=int, default=NCELLS)
    parser.add_argument("--kcount", type=int, default=KCOUNT)
    parser.add_argument("--cmap", default="Blues", help="Matplotlib colormap name.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    return parser.parse_args()


def build_k_vectors(ncells, lattice_parameter, kcount):
    """Build the same one-dimensional k-vector grid used by find_models.py."""
    kmin = 2 * np.pi / (ncells * lattice_parameter)
    kmax = ncells * kmin
    kmas = np.zeros((kcount, 3), dtype=np.float32)
    kmas[:, 0] = np.linspace(kmin, kmax, kcount)
    return kmas


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacements to flat absolute coordinates."""
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    for crystal_index in np.ndindex(atom_order.shape):
        flat[:, atom_order[crystal_index], :] = positions[(slice(None), *crystal_index, slice(None))]
    return flat.reshape(frames, atoms * 3)


def load_positions(data, positions_key, displacements_key):
    """Load flat positions directly or reconstruct them from crystal displacements."""
    if positions_key in data.files:
        return data[positions_key]
    required = [displacements_key, "reference_positions", "atom_order"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Cannot reconstruct {positions_key}; missing arrays: {missing}")
    return crystal_frames_to_flat_positions(
        data[displacements_key],
        data["reference_positions"],
        data["atom_order"],
    )


def scalar_string(value):
    """Return a Python string from an npz scalar value."""
    return str(np.asarray(value).item())


def resolve_reference_data_path(args, inference_data):
    """Return the dataset path used to build reference positions for ASE outputs."""
    if args.data_path:
        return args.data_path
    if "data_path" in inference_data.files:
        return scalar_string(inference_data["data_path"])
    raise ValueError("--data-path is required when input contains ASE positions only")


def load_ase_positions_with_reference(args, inference_data):
    """Load ASE positions and matching reference positions from the source dataset."""
    if "positions" not in inference_data.files:
        return None

    data_path = resolve_reference_data_path(args, inference_data)
    reference_data = np.load(data_path)
    required = ["displacements", "reference_positions", "atom_order"]
    missing = [key for key in required if key not in reference_data.files]
    if missing:
        raise ValueError(f"Cannot build ASE reference positions; missing arrays in {data_path}: {missing}")

    positions = np.asarray(inference_data["positions"], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("ASE positions must have shape (frames, atoms, 3)")
    start = int(inference_data["initial_frames"][-1]) if "initial_frames" in inference_data.files else 0
    available = int(reference_data["displacements"].shape[0]) - start
    frames = min(int(positions.shape[0]), available)
    if frames < 3:
        raise ValueError(f"Need at least three overlapping frames, got {frames}")
    if frames < positions.shape[0]:
        print(
            f"WARNING: trimming ASE trajectory from {positions.shape[0]} to {frames} frames "
            f"because reference data ends at frame {int(reference_data['displacements'].shape[0]) - 1}."
        )

    predicted_positions = positions[:frames].reshape(frames, -1)
    reference_positions = crystal_frames_to_flat_positions(
        reference_data["displacements"][start : start + frames],
        reference_data["reference_positions"],
        reference_data["atom_order"],
    )
    return predicted_positions, reference_positions


def load_predicted_and_reference_positions(args, data):
    """Load predicted/reference flat positions from direct or ASE inference output."""
    ase_pair = load_ase_positions_with_reference(args, data)
    if ase_pair is not None:
        return ase_pair
    return (
        load_positions(data, "predicted_positions", "predicted_displacements"),
        load_positions(data, "reference_positions_output", "reference_displacements"),
    )


def correlation(first, second):
    """Return Pearson correlation between two S(q,w) intensity maps."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 2:
        return np.nan

    first = first[finite]
    second = second[finite]
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return np.nan
    return float(np.dot(first, second) / denominator)


def plot_maps(xi, yi, predicted_jlp, reference_jlp, corr, output_path, cmap):
    """Create a side-by-side S(q,w) comparison figure."""
    vmin = float(np.nanmin([np.nanmin(predicted_jlp), np.nanmin(reference_jlp)]))
    vmax = float(np.nanmax([np.nanmax(predicted_jlp), np.nanmax(reference_jlp)]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    predicted_mesh = axes[0].pcolormesh(xi, yi, predicted_jlp, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted S(q,w)")
    axes[0].set_xlabel("|q|")
    axes[0].set_ylabel("Energy")

    axes[1].pcolormesh(xi, yi, reference_jlp, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Reference S(q,w)")
    axes[1].set_xlabel("|q|")
    axes[1].set_ylabel("Energy")

    fig.colorbar(predicted_mesh, ax=axes, label="Normalized intensity")
    fig.suptitle(f"S(q,w) comparison, correlation = {corr:.4f}")
    fig.savefig(output_path, dpi=200)
    return fig


def main():
    """Load inference output, compute S(q,w), and save a comparison plot."""
    args = parse_args()
    data = np.load(args.input_path)
    predicted_positions, reference_positions = load_predicted_and_reference_positions(args, data)

    kmas = build_k_vectors(args.ncells, args.lattice_parameter, args.kcount)
    xi_pred, yi_pred, predicted_jlp = get_sqw(predicted_positions, dt=args.dt, step=args.step, kmas=kmas)
    xi_ref, yi_ref, reference_jlp = get_sqw(reference_positions, dt=args.dt, step=args.step, kmas=kmas)
    if not np.allclose(xi_pred, xi_ref) or not np.allclose(yi_pred, yi_ref):
        raise ValueError("Predicted and reference S(q,w) grids do not match")

    corr = correlation(predicted_jlp, reference_jlp)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_maps(xi_pred, yi_pred, predicted_jlp, reference_jlp, corr, output_path, args.cmap)
    print(f"Saved {output_path}")
    print(f"correlation = {corr:.6f}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
