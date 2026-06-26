"""Compare radial distribution functions for ASE inference and reference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_sqw_comparison import crystal_frames_to_flat_positions


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz file.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--bins", type=int, default=220, help="RDF bin count.")
    parser.add_argument("--r-max", type=float, default=None, help="Maximum radius in Angstrom.")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum uniformly sampled frames per trajectory.")
    parser.add_argument(
        "--frame-fraction-start",
        type=float,
        default=0.0,
        help="Start fraction of each trajectory used for RDF sampling.",
    )
    parser.add_argument(
        "--frame-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of each trajectory used for RDF sampling.",
    )
    parser.add_argument("--title", default="ASE NVT radial distribution function")
    return parser.parse_args()


def as_cell_matrix(value: np.ndarray) -> np.ndarray:
    """Return a 3x3 cell matrix."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.ndim == 2 and cell.shape[1] == 3 and cell.shape[0] != 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError(f"Unsupported cell shape: {cell.shape}")
    return cell


def cell_from_data(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return a representative reference cell matrix."""
    if "cell" in data.files:
        return as_cell_matrix(data["cell"])
    if "box_lengths" in data.files:
        return as_cell_matrix(data["box_lengths"])
    raise ValueError("Reference data must contain cell or box_lengths for RDF.")


def cell_from_ase(ase: np.lib.npyio.NpzFile, fallback: np.ndarray) -> np.ndarray:
    """Return the ASE cell matrix."""
    if "cell" in ase.files:
        return as_cell_matrix(ase["cell"])
    return fallback


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap Cartesian differences into the nearest periodic image."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(delta, dtype=np.float64) @ inverse_cell
    fractional -= np.round(fractional)
    return fractional @ cell


def uniform_indices(frame_count: int, max_frames: int, fraction_start: float, fraction_end: float) -> np.ndarray:
    """Return deterministic frame indices sampled over the requested frame fraction."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if max_frames <= 0:
        raise ValueError("max-frames must be positive")
    if not 0.0 <= fraction_start < fraction_end <= 1.0:
        raise ValueError("Frame fractions must satisfy 0 <= start < end <= 1")
    start = int(np.floor(frame_count * fraction_start))
    end = int(np.ceil(frame_count * fraction_end))
    end = min(max(end, start + 1), frame_count)
    count = min(max_frames, end - start)
    return np.unique(np.linspace(start, end - 1, count, dtype=np.int64))


def load_reference_positions(data: np.lib.npyio.NpzFile, start: int) -> np.ndarray:
    """Return reference positions in flat atom order."""
    reference_displacements = np.asarray(data["displacements"][start:], dtype=np.float32)
    return crystal_frames_to_flat_positions(
        reference_displacements,
        data["reference_positions"],
        data["atom_order"],
    ).reshape(reference_displacements.shape[0], -1, 3)


def rdf_for_positions(
    positions: np.ndarray,
    cell: np.ndarray,
    r_max: float,
    bins: int,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RDF centers, g(r), and cumulative coordination number."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (frames, atoms, 3), got {positions.shape}")
    atom_count = positions.shape[1]
    if atom_count < 2:
        raise ValueError("Need at least two atoms for RDF.")

    edges = np.linspace(0.0, float(r_max), int(bins) + 1)
    hist = np.zeros(int(bins), dtype=np.float64)
    upper_i, upper_j = np.triu_indices(atom_count, k=1)
    for frame_index in indices:
        delta = positions[frame_index, upper_j] - positions[frame_index, upper_i]
        delta = minimum_image(delta, cell)
        distances = np.linalg.norm(delta, axis=1)
        hist += np.histogram(distances, bins=edges)[0]

    volume = abs(float(np.linalg.det(cell)))
    density = atom_count / volume
    shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    frame_count = int(indices.shape[0])
    # Unique pairs are counted once, so multiply by two to get neighbors per atom.
    rdf = (2.0 * hist) / (frame_count * atom_count * density * shell_volumes)
    coordination = np.cumsum(2.0 * hist / (frame_count * atom_count))
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, rdf, coordination


def first_peak_and_minimum(radius: np.ndarray, rdf: np.ndarray):
    """Return first-peak and following-minimum coordinates."""
    if radius.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    search = radius > 1.5
    if not np.any(search):
        search = np.ones_like(radius, dtype=bool)
    offset = int(np.argmax(search))
    peak_index = offset + int(np.argmax(rdf[search]))
    if peak_index + 2 < radius.size:
        minimum_index = peak_index + 1 + int(np.argmin(rdf[peak_index + 1 :]))
    else:
        minimum_index = peak_index
    return (
        float(radius[peak_index]),
        float(rdf[peak_index]),
        float(radius[minimum_index]),
        float(rdf[minimum_index]),
    )


def write_curve(path: Path, radius: np.ndarray, ase_rdf: np.ndarray, ref_rdf: np.ndarray, ase_cn: np.ndarray, ref_cn: np.ndarray):
    """Write RDF curve data."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["r_ang", "g_ase", "g_reference", "coordination_ase", "coordination_reference"])
        for row in zip(radius, ase_rdf, ref_rdf, ase_cn, ref_cn):
            writer.writerow([f"{float(value):.10g}" for value in row])


def difference_metrics(radius: np.ndarray, predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None):
    """Return scale-aware RDF difference metrics."""
    if mask is None:
        mask = np.ones_like(radius, dtype=bool)
    predicted = np.asarray(predicted, dtype=np.float64)[mask]
    reference = np.asarray(reference, dtype=np.float64)[mask]
    radius = np.asarray(radius, dtype=np.float64)[mask]
    diff = predicted - reference
    if diff.size == 0:
        return {
            "corr": float("nan"),
            "rms": float("nan"),
            "rel_l2": float("nan"),
            "mae": float("nan"),
            "max_abs": float("nan"),
            "integral_abs": float("nan"),
        }
    dr = float(np.median(np.diff(radius))) if radius.size > 1 else 1.0
    reference_norm = float(np.linalg.norm(reference))
    return {
        "corr": float(np.corrcoef(predicted, reference)[0, 1]) if diff.size > 1 else float("nan"),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "rel_l2": float(np.linalg.norm(diff) / reference_norm) if reference_norm > 0 else float("nan"),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "integral_abs": float(np.sum(np.abs(diff)) * dr),
    }


def main() -> None:
    """Build an RDF comparison plot and metrics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0
    reference_cell = cell_from_data(data)
    ase_cell = cell_from_ase(ase, fallback=reference_cell)

    ase_positions = np.asarray(ase["positions"], dtype=np.float64)
    reference_positions = load_reference_positions(data, start=start).astype(np.float64)

    r_max = args.r_max
    if r_max is None:
        lengths = np.linalg.norm(ase_cell, axis=1)
        r_max = float(0.5 * np.min(lengths) * 0.999)
    ase_indices = uniform_indices(
        ase_positions.shape[0],
        max_frames=args.max_frames,
        fraction_start=args.frame_fraction_start,
        fraction_end=args.frame_fraction_end,
    )
    reference_indices = uniform_indices(
        reference_positions.shape[0],
        max_frames=args.max_frames,
        fraction_start=args.frame_fraction_start,
        fraction_end=args.frame_fraction_end,
    )

    radius, ase_rdf, ase_coordination = rdf_for_positions(
        ase_positions,
        cell=ase_cell,
        r_max=r_max,
        bins=args.bins,
        indices=ase_indices,
    )
    reference_radius, reference_rdf, reference_coordination = rdf_for_positions(
        reference_positions,
        cell=reference_cell,
        r_max=r_max,
        bins=args.bins,
        indices=reference_indices,
    )
    if not np.allclose(radius, reference_radius):
        raise ValueError("ASE/reference RDF radii differ")

    peak_a, peak_g_a, min_a, min_g_a = first_peak_and_minimum(radius, ase_rdf)
    peak_r, peak_g_r, min_r, min_g_r = first_peak_and_minimum(radius, reference_rdf)
    all_metrics = difference_metrics(radius, ase_rdf, reference_rdf)
    visible_peak_metrics = difference_metrics(radius, ase_rdf, reference_rdf, mask=reference_rdf > 0.5)
    corr = all_metrics["corr"]
    rms_diff = all_metrics["rms"]
    rdf_difference = ase_rdf - reference_rdf

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    axes[0].plot(radius, reference_rdf, color="tab:blue", label="reference")
    axes[0].plot(radius, ase_rdf, color="tab:orange", label="ASE inference")
    axes[0].set_xlabel("r, A")
    axes[0].set_ylabel("g(r)")
    axes[0].set_title(f"RDF, corr={corr:.3f}, RMS diff={rms_diff:.3f}")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(radius, reference_coordination, color="tab:blue", label="reference")
    axes[1].plot(radius, ase_coordination, color="tab:orange", label="ASE inference")
    axes[1].set_xlabel("r, A")
    axes[1].set_ylabel("coordination number")
    axes[1].set_title("Cumulative coordination")
    axes[1].grid(alpha=0.2)

    axes[2].plot(radius, rdf_difference, color="tab:red")
    axes[2].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[2].set_xlabel("r, A")
    axes[2].set_ylabel("g_ASE(r) - g_ref(r)")
    axes[2].set_title(f"Residual, rel L2={all_metrics['rel_l2']:.4f}")
    axes[2].grid(alpha=0.2)

    fig.suptitle(args.title)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved {output_path}")
    print(f"RDF correlation = {corr:.6g}")
    print(f"RDF RMS diff = {rms_diff:.6g}")
    print(f"RDF relative L2 diff = {all_metrics['rel_l2']:.6g}")
    print(f"RDF max abs diff = {all_metrics['max_abs']:.6g}")
    print(f"ASE first peak/min = {peak_a:.6g} A, {peak_g_a:.6g} / {min_a:.6g} A, {min_g_a:.6g}")
    print(f"reference first peak/min = {peak_r:.6g} A, {peak_g_r:.6g} / {min_r:.6g} A, {min_g_r:.6g}")

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_curve(metrics_path, radius, ase_rdf, reference_rdf, ase_coordination, reference_coordination)
        summary_path = metrics_path.with_name(metrics_path.stem + "_summary.tsv")
        rows = [
            {
                "source": "comparison",
                "frames": "",
                "r_max_ang": r_max,
                "first_peak_r_ang": "",
                "first_peak_g": "",
                "first_min_r_ang": "",
                "first_min_g": "",
                "rdf_correlation": corr,
                "rdf_rms_diff": rms_diff,
                "rdf_rel_l2_diff": all_metrics["rel_l2"],
                "rdf_mae": all_metrics["mae"],
                "rdf_max_abs_diff": all_metrics["max_abs"],
                "rdf_integral_abs_diff": all_metrics["integral_abs"],
                "peak_region_correlation": visible_peak_metrics["corr"],
                "peak_region_rms_diff": visible_peak_metrics["rms"],
                "peak_region_rel_l2_diff": visible_peak_metrics["rel_l2"],
            },
            {
                "source": "ASE",
                "frames": int(ase_indices.shape[0]),
                "r_max_ang": r_max,
                "first_peak_r_ang": peak_a,
                "first_peak_g": peak_g_a,
                "first_min_r_ang": min_a,
                "first_min_g": min_g_a,
                "rdf_correlation": "",
                "rdf_rms_diff": "",
                "rdf_rel_l2_diff": "",
                "rdf_mae": "",
                "rdf_max_abs_diff": "",
                "rdf_integral_abs_diff": "",
                "peak_region_correlation": "",
                "peak_region_rms_diff": "",
                "peak_region_rel_l2_diff": "",
            },
            {
                "source": "reference",
                "frames": int(reference_indices.shape[0]),
                "r_max_ang": r_max,
                "first_peak_r_ang": peak_r,
                "first_peak_g": peak_g_r,
                "first_min_r_ang": min_r,
                "first_min_g": min_g_r,
                "rdf_correlation": "",
                "rdf_rms_diff": "",
                "rdf_rel_l2_diff": "",
                "rdf_mae": "",
                "rdf_max_abs_diff": "",
                "rdf_integral_abs_diff": "",
                "peak_region_correlation": "",
                "peak_region_rms_diff": "",
                "peak_region_rel_l2_diff": "",
            },
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
