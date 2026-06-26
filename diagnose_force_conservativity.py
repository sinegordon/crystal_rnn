#!/usr/bin/env python3
"""Diagnose non-conservative force components with virtual loop work.

The script probes a trained local acceleration model around sampled 3x3x3
training blocks.  For each block it creates small closed rectangular loops in
the central-cell displacement space and measures the model work around the
loop.  A conservative force field should produce zero work around sufficiently
small closed loops.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CU_MASS_AMU, forces_to_discrete_accelerations


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["model", "eam-reference"],
        default="model",
        help="Acceleration source used for diagnostics.",
    )
    parser.add_argument("--model-path", default=None, help="Saved torch model path.")
    parser.add_argument("--eam-potential", default=None, help="EAM potential for --source eam-reference.")
    parser.add_argument(
        "--eam-backend",
        choices=["ase", "asap"],
        default="ase",
        help="EAM calculator backend for --source eam-reference.",
    )
    parser.add_argument("--data-path", required=True, help="Prepared crystal npz with X_blocks.")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and metrics.")
    parser.add_argument("--samples", type=int, default=200, help="Number of sampled blocks.")
    parser.add_argument("--loops-per-sample", type=int, default=50, help="Random loops per sampled block.")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Loop side length in Angstrom displacement units.")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="cpu", help="Torch device used for model inference.")
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument(
        "--jacobian-samples",
        type=int,
        default=0,
        help="Optional number of sampled blocks for finite-difference Jacobian antisymmetry diagnostics.",
    )
    parser.add_argument(
        "--jacobian-epsilon",
        type=float,
        default=1e-4,
        help="Central finite-difference step for Jacobian diagnostics.",
    )
    return parser.parse_args()


def resolve_device(device):
    """Return a usable torch device."""
    device = str(device)
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.startswith("mps") and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise ValueError("MPS requested but unavailable")
    return torch.device(device)


def load_model(path, device):
    """Load a saved torch model and move it to the requested device."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(model, "to"):
        model.to(device)
    elif hasattr(model, "model"):
        model.model.to(device)
        model.torch_device = device
    else:
        raise TypeError("Loaded object does not look like a supported model")
    return model


class EAMReferenceAccelerator:
    """Evaluate reference EAM forces and convert them to discrete accelerations."""

    def __init__(self, data, potential_path, backend="ase"):
        from ase import Atoms

        self.atom_order = np.asarray(data["atom_order"], dtype=np.int64)
        self.reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
        box_lengths = np.asarray(data["box_lengths"], dtype=np.float64)
        if box_lengths.ndim == 2:
            box_lengths = box_lengths[0]
        self.cell = np.diag(box_lengths)
        self.dt_ps = float(np.asarray(data["dt_ps"], dtype=np.float64)) if "dt_ps" in data.files else 0.002
        self.atom_mass_amu = (
            float(np.asarray(data["atom_mass_amu"], dtype=np.float64)) if "atom_mass_amu" in data.files else CU_MASS_AMU
        )
        symbols = ["Cu"] * int(self.reference_positions.shape[0])
        self.atoms = Atoms(symbols=symbols, positions=self.reference_positions.copy(), cell=self.cell, pbc=True)
        self.atoms.calc = self._make_eam_calculator(potential_path, backend)

    @staticmethod
    def _make_eam_calculator(potential_path, backend):
        """Create an EAM calculator using ASE or ASAP."""
        if backend == "ase":
            from ase.calculators.eam import EAM

            return EAM(potential=str(potential_path))
        try:
            from asap3.Internal.BuiltinPotentials import EAM
        except ImportError as exc:
            raise ImportError(
                "ASAP EAM backend requires the asap3 package. Install it or use --eam-backend ase."
            ) from exc
        return EAM(str(potential_path))

    def predict_center_accelerations(self, patches):
        """Return EAM-derived central-cell accelerations for patch vertices."""
        patches = np.asarray(patches, dtype=np.float64)
        accelerations = []
        for patch in patches:
            crystal_displacements = patch[-1]
            flat_positions = self._flat_positions(crystal_displacements)
            self.atoms.set_positions(flat_positions, apply_constraint=False)
            forces = self.atoms.get_forces()
            flat_acceleration = forces_to_discrete_accelerations(
                forces[None, :, :],
                dt_ps=self.dt_ps,
                atom_mass_amu=self.atom_mass_amu,
            )[0]
            crystal_acceleration = flat_acceleration[self.atom_order]
            accelerations.append(crystal_acceleration[1, 1, 1])
        return np.asarray(accelerations, dtype=np.float64)

    def _flat_positions(self, crystal_displacements):
        reference_crystal = self.reference_positions[self.atom_order]
        crystal_positions = reference_crystal + np.asarray(crystal_displacements, dtype=np.float64)
        flat_positions = np.empty_like(self.reference_positions)
        flat_positions[self.atom_order.reshape(-1)] = crystal_positions.reshape(-1, 3)
        return flat_positions


def normalize_direction(direction, epsilon):
    """Scale one central-cell direction to a fixed RMS side length."""
    direction = np.asarray(direction, dtype=np.float32)
    rms = float(np.sqrt(np.mean(direction**2)))
    if rms == 0.0:
        raise ValueError("zero random direction")
    return direction * (float(epsilon) / rms)


def central_patch_loop_points(patch, du, dv):
    """Return four loop vertices in a 3x3x3 patch."""
    points = []
    for shift in (
        np.zeros_like(du),
        du,
        du + dv,
        dv,
    ):
        candidate = np.array(patch, copy=True)
        candidate[-1, 1, 1, 1] = candidate[-1, 1, 1, 1] + shift
        points.append(candidate)
    return np.asarray(points, dtype=np.float32)


def model_accelerations(model, patches):
    """Predict central-cell accelerations for loop vertices."""
    if not hasattr(model, "predict_center_accelerations"):
        raise TypeError("Model must provide predict_center_accelerations(patch_batch)")
    return np.asarray(model.predict_center_accelerations(patches), dtype=np.float64)


def load_acceleration_source(args, data, device):
    """Load the selected acceleration source."""
    if args.source == "model":
        if args.model_path is None:
            raise ValueError("--model-path is required for --source model")
        return load_model(args.model_path, device)
    if args.eam_potential is None:
        raise ValueError("--eam-potential is required for --source eam-reference")
    return EAMReferenceAccelerator(data, args.eam_potential, backend=args.eam_backend)


def loop_work(accelerations, du, dv):
    """Approximate work around a four-segment closed rectangular loop."""
    segments = (
        np.asarray(du, dtype=np.float64),
        np.asarray(dv, dtype=np.float64),
        -np.asarray(du, dtype=np.float64),
        -np.asarray(dv, dtype=np.float64),
    )
    work = 0.0
    scale = 0.0
    for index, segment in enumerate(segments):
        # Trapezoidal segment work cancels conservative linear fields more cleanly
        # than using only the force at the segment start.
        acceleration = 0.5 * (accelerations[index] + accelerations[(index + 1) % len(accelerations)])
        work += float(np.sum(acceleration * segment))
        scale += float(np.sqrt(np.sum(acceleration**2)) * np.sqrt(np.sum(segment**2)))
    return work, scale


def summarize(values):
    """Return scalar summary statistics."""
    values = np.asarray(values, dtype=np.float64)
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean_abs": float(np.mean(abs_values)),
        "p50_abs": float(np.percentile(abs_values, 50)),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "p99_abs": float(np.percentile(abs_values, 99)),
    }


def write_metrics(path, raw_work_summary, normalized_work_summary, args):
    """Write summary metrics as a TSV file."""
    fields = [
        "quantity",
        "count",
        "mean",
        "std",
        "rms",
        "mean_abs",
        "p50_abs",
        "p95_abs",
        "p99_abs",
        "epsilon",
        "samples",
        "loops_per_sample",
        "model_path",
        "data_path",
    ]
    rows = []
    for quantity, summary in (
        ("loop_work", raw_work_summary),
        ("normalized_loop_work", normalized_work_summary),
    ):
        row = {"quantity": quantity, **summary}
        row.update(
            {
                "epsilon": args.epsilon,
                "samples": args.samples,
                "loops_per_sample": args.loops_per_sample,
                "model_path": args.model_path,
                "data_path": args.data_path,
            }
        )
        rows.append(row)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def robust_range(values, percentile):
    """Return a symmetric robust plotting range."""
    values = np.asarray(values, dtype=np.float64)
    limit = float(np.percentile(np.abs(values[np.isfinite(values)]), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    return -limit, limit


def plot_histograms(path, loop_works, normalized_loop_works, bins, percentile):
    """Plot loop-work histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(
        loop_works,
        bins=bins,
        range=robust_range(loop_works, percentile),
        density=True,
        color="tab:orange",
        alpha=0.72,
    )
    axes[0].axvline(0.0, color="black", linewidth=0.9)
    axes[0].set_title("Virtual closed-loop work")
    axes[0].set_xlabel("sum a · du, A^2/step^2")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.22)

    axes[1].hist(
        normalized_loop_works,
        bins=bins,
        range=robust_range(normalized_loop_works, percentile),
        density=True,
        color="tab:green",
        alpha=0.72,
    )
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[1].set_title("Normalized loop work")
    axes[1].set_xlabel("loop work / sum |a||du|")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.22)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def central_patch_with_flat_shift(patch, flat_index, shift):
    """Return a patch with one central-cell displacement component shifted."""
    candidate = np.array(patch, copy=True)
    central = candidate[-1, 1, 1, 1]
    central.reshape(-1)[flat_index] += shift
    return candidate


def finite_difference_jacobian(model, patch, epsilon):
    """Estimate d a_center / d u_center with central finite differences."""
    central_size = int(np.prod(patch[-1, 1, 1, 1].shape))
    rows = []
    for flat_index in range(central_size):
        plus = central_patch_with_flat_shift(patch, flat_index, epsilon)
        minus = central_patch_with_flat_shift(patch, flat_index, -epsilon)
        accelerations = model_accelerations(model, np.asarray([plus, minus], dtype=np.float32)).reshape(2, -1)
        rows.append((accelerations[0] - accelerations[1]) / (2.0 * epsilon))
    return np.asarray(rows, dtype=np.float64).T


def jacobian_antisymmetry(model, x_blocks, sample_indices, epsilon):
    """Compute antisymmetric Jacobian ratios for sampled blocks."""
    ratios = []
    antisym_norms = []
    jacobian_norms = []
    for sample_index in sample_indices:
        jacobian = finite_difference_jacobian(model, x_blocks[sample_index], epsilon)
        antisym = 0.5 * (jacobian - jacobian.T)
        jacobian_norm = float(np.linalg.norm(jacobian))
        antisym_norm = float(np.linalg.norm(antisym))
        ratios.append(antisym_norm / jacobian_norm if jacobian_norm > 0.0 else np.nan)
        antisym_norms.append(antisym_norm)
        jacobian_norms.append(jacobian_norm)
    return (
        np.asarray(ratios, dtype=np.float64),
        np.asarray(antisym_norms, dtype=np.float64),
        np.asarray(jacobian_norms, dtype=np.float64),
    )


def write_jacobian_metrics(path, ratio_summary, antisym_summary, norm_summary, args):
    """Write finite-difference Jacobian diagnostics."""
    fields = [
        "quantity",
        "count",
        "mean",
        "std",
        "rms",
        "mean_abs",
        "p50_abs",
        "p95_abs",
        "p99_abs",
        "jacobian_epsilon",
        "jacobian_samples",
        "model_path",
        "data_path",
    ]
    rows = []
    for quantity, summary in (
        ("antisymmetry_ratio", ratio_summary),
        ("antisymmetric_norm", antisym_summary),
        ("jacobian_norm", norm_summary),
    ):
        row = {"quantity": quantity, **summary}
        row.update(
            {
                "jacobian_epsilon": args.jacobian_epsilon,
                "jacobian_samples": args.jacobian_samples,
                "model_path": args.model_path,
                "data_path": args.data_path,
            }
        )
        rows.append(row)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_jacobian_histogram(path, ratios, bins, percentile):
    """Plot finite-difference Jacobian antisymmetry ratios."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    finite = ratios[np.isfinite(ratios)]
    upper = float(np.percentile(finite, percentile)) if finite.size else 1.0
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0
    ax.hist(finite, bins=bins, range=(0.0, upper), density=True, color="tab:red", alpha=0.72)
    ax.set_title("Jacobian antisymmetry")
    ax.set_xlabel("||0.5 * (J - J.T)|| / ||J||")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Run the virtual-loop diagnostic."""
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.loops_per_sample <= 0:
        raise ValueError("loops-per-sample must be positive")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if args.jacobian_samples < 0:
        raise ValueError("jacobian-samples must be non-negative")
    if args.jacobian_epsilon <= 0:
        raise ValueError("jacobian-epsilon must be positive")

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    data = np.load(args.data_path)
    model = load_acceleration_source(args, data, device)
    x_blocks = np.asarray(data["X_blocks"], dtype=np.float32)
    if x_blocks.ndim != 7:
        raise ValueError("X_blocks must have shape (samples, sequence, 3, 3, 3, atoms, 3)")

    sample_count = min(int(args.samples), int(x_blocks.shape[0]))
    sample_indices = rng.choice(x_blocks.shape[0], size=sample_count, replace=False)
    loop_works = []
    normalized_loop_works = []
    for sample_index in sample_indices:
        patch = x_blocks[sample_index]
        for _ in range(args.loops_per_sample):
            du = normalize_direction(rng.normal(size=patch[-1, 1, 1, 1].shape), args.epsilon)
            dv = normalize_direction(rng.normal(size=patch[-1, 1, 1, 1].shape), args.epsilon)
            patches = central_patch_loop_points(patch, du, dv)
            accelerations = model_accelerations(model, patches)
            work, scale = loop_work(accelerations, du, dv)
            loop_works.append(work)
            normalized_loop_works.append(work / scale if scale > 0.0 else np.nan)

    loop_works = np.asarray(loop_works, dtype=np.float64)
    normalized_loop_works = np.asarray(normalized_loop_works, dtype=np.float64)
    finite = np.isfinite(normalized_loop_works)
    loop_works = loop_works[finite]
    normalized_loop_works = normalized_loop_works[finite]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "virtual_loop_work_metrics.tsv"
    plot_path = output_dir / "virtual_loop_work_histograms.png"
    samples_path = output_dir / "virtual_loop_work_samples.npz"

    write_metrics(metrics_path, summarize(loop_works), summarize(normalized_loop_works), args)
    plot_histograms(plot_path, loop_works, normalized_loop_works, args.bins, args.percentile)
    np.savez_compressed(samples_path, loop_work=loop_works, normalized_loop_work=normalized_loop_works)
    print(f"Saved {metrics_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {samples_path}")

    if args.jacobian_samples:
        jacobian_count = min(int(args.jacobian_samples), int(x_blocks.shape[0]))
        jacobian_indices = rng.choice(x_blocks.shape[0], size=jacobian_count, replace=False)
        ratios, antisym_norms, jacobian_norms = jacobian_antisymmetry(
            model,
            x_blocks,
            jacobian_indices,
            args.jacobian_epsilon,
        )
        finite = np.isfinite(ratios)
        ratios = ratios[finite]
        antisym_norms = antisym_norms[finite]
        jacobian_norms = jacobian_norms[finite]

        jacobian_metrics_path = output_dir / "jacobian_antisymmetry_metrics.tsv"
        jacobian_plot_path = output_dir / "jacobian_antisymmetry_histogram.png"
        jacobian_samples_path = output_dir / "jacobian_antisymmetry_samples.npz"
        write_jacobian_metrics(
            jacobian_metrics_path,
            summarize(ratios),
            summarize(antisym_norms),
            summarize(jacobian_norms),
            args,
        )
        plot_jacobian_histogram(jacobian_plot_path, ratios, args.bins, args.percentile)
        np.savez_compressed(
            jacobian_samples_path,
            antisymmetry_ratio=ratios,
            antisymmetric_norm=antisym_norms,
            jacobian_norm=jacobian_norms,
        )
        print(f"Saved {jacobian_metrics_path}")
        print(f"Saved {jacobian_plot_path}")
        print(f"Saved {jacobian_samples_path}")


if __name__ == "__main__":
    main()
