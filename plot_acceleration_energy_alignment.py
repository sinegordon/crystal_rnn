"""Compare acceleration/displacement and acceleration/velocity alignment.

The main question answered by this diagnostic is whether the model suppresses
large displacements by producing forces that inject kinetic energy.  The sign of
``a dot v`` is a direct proxy for instantaneous power: positive values mean the
model acceleration points along the velocity, while negative values mean it
opposes the velocity.  The sign of ``a dot u`` shows whether acceleration is
restoring relative to the displacement from the reference lattice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CU_MASS_AMU = 63.546
AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM = 1.0364269656262175e-4


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=ASE_NPZ",
        help="ASE inference output to compare. May be specified multiple times.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the PNG and metrics table.")
    parser.add_argument("--bins", type=int, default=220, help="Histogram bin count.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Symmetric histogram range percentile.")
    parser.add_argument("--title", default="Acceleration alignment diagnostics", help="Figure title.")
    return parser.parse_args()


def parse_runs(values):
    """Return ``(label, path)`` pairs from LABEL=PATH command-line values."""
    runs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must have LABEL=PATH format, got {value!r}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--run must have LABEL=PATH format, got {value!r}")
        runs.append((label, path))
    return runs


def as_cell_matrix(value):
    """Return a 3x3 cell matrix."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.ndim == 2 and cell.shape[1] == 3 and cell.shape[0] != 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must resolve to shape (3, 3), got {cell.shape}")
    return cell


def minimum_image(delta, cell):
    """Wrap flat position differences into the nearest periodic image."""
    fractional = np.asarray(delta, dtype=np.float64) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def crystal_to_flat_values(crystal_values, atom_order):
    """Convert crystal-shaped values into flat ASE atom order."""
    crystal_values = np.asarray(crystal_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    frames = int(crystal_values.shape[0])
    atom_count = int(atom_order.size)
    flat = np.empty((frames, atom_count, 3), dtype=np.float64)
    flat[:, atom_order.reshape(-1), :] = crystal_values.reshape(frames, atom_count, 3)
    return flat


def discrete_acceleration(displacements):
    """Return discrete accelerations aligned with central frames."""
    return displacements[2:] - 2.0 * displacements[1:-1] + displacements[:-2]


def central_velocity(displacements, dt_ps):
    """Return centered finite-difference velocities aligned with accelerations."""
    return (displacements[2:] - displacements[:-2]) / (2.0 * float(dt_ps))


def force_to_discrete_acceleration(forces, dt_ps):
    """Convert ASE forces back to model discrete accelerations."""
    factor = AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM / float(dt_ps) ** 2
    return np.asarray(forces, dtype=np.float64) / (CU_MASS_AMU * factor)


def vector_dot(a, b):
    """Return per-atom vector dot products."""
    frames = min(a.shape[0], b.shape[0])
    return np.sum(a[:frames] * b[:frames], axis=-1).reshape(-1)


def summarize(label, acceleration, displacement, velocity):
    """Return alignment samples and scalar summary metrics."""
    frames = min(acceleration.shape[0], displacement.shape[0], velocity.shape[0])
    acceleration = np.asarray(acceleration[:frames], dtype=np.float64)
    displacement = np.asarray(displacement[:frames], dtype=np.float64)
    velocity = np.asarray(velocity[:frames], dtype=np.float64)
    dot_au = vector_dot(acceleration, displacement)
    dot_av = vector_dot(acceleration, velocity)
    positive_power = dot_av[dot_av > 0.0].sum()
    negative_power = -dot_av[dot_av < 0.0].sum()
    power_total = positive_power + negative_power
    return {
        "label": label,
        "dot_au": dot_au,
        "dot_av": dot_av,
        "frames": frames,
        "u_rms": float(np.sqrt(np.mean(displacement**2))),
        "v_component_std": float(velocity.reshape(-1).std()),
        "a_component_rms": float(np.sqrt(np.mean(acceleration**2))),
        "au_mean": float(dot_au.mean()),
        "au_positive_fraction": float(np.mean(dot_au > 0.0)),
        "av_mean": float(dot_av.mean()),
        "av_positive_fraction": float(np.mean(dot_av > 0.0)),
        "av_positive_abs_fraction": float(positive_power / power_total) if power_total > 0.0 else float("nan"),
    }


def write_metrics(path, summaries):
    """Write scalar metrics as a tab-separated table."""
    fields = [
        "label",
        "frames",
        "u_rms",
        "v_component_std",
        "a_component_rms",
        "au_mean",
        "au_positive_fraction",
        "av_mean",
        "av_positive_fraction",
        "av_positive_abs_fraction",
    ]
    lines = ["\t".join(fields)]
    for summary in summaries:
        lines.append("\t".join(str(summary[field]) for field in fields))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_alignment(path, summaries, bins, percentile, title):
    """Plot overlaid histograms of ``a dot u`` and ``a dot v``."""
    dot_au = [summary["dot_au"] for summary in summaries]
    dot_av = [summary["dot_av"] for summary in summaries]
    au_limit = float(np.percentile(np.abs(np.concatenate(dot_au)), percentile))
    av_limit = float(np.percentile(np.abs(np.concatenate(dot_av)), percentile))
    if not np.isfinite(au_limit) or au_limit <= 0.0:
        au_limit = 1.0
    if not np.isfinite(av_limit) or av_limit <= 0.0:
        av_limit = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for summary in summaries:
        axes[0].hist(
            summary["dot_au"],
            bins=bins,
            range=(-au_limit, au_limit),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=summary["label"],
        )
        axes[1].hist(
            summary["dot_av"],
            bins=bins,
            range=(-av_limit, av_limit),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=summary["label"],
        )
    axes[0].axvline(0.0, color="black", linewidth=0.9)
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[0].set_title("Restoring alignment: a dot u")
    axes[0].set_xlabel("a.u, A^2/step^2")
    axes[1].set_title("Power alignment: a dot v")
    axes[1].set_xlabel("a.v, A^2/(step^2 ps)")
    for axis in axes:
        axis.set_ylabel("density")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Run the comparison."""
    args = parse_args()
    runs = parse_runs(args.run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)

    summaries = []
    reference_added = False
    for label, ase_path in runs:
        ase = np.load(ase_path)
        cell = as_cell_matrix(ase["cell"])
        dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64))
        start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0

        positions = np.asarray(ase["positions"], dtype=np.float64)
        velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
        forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
        displacements = minimum_image(positions - reference_positions[None, :, :], cell)
        acceleration = force_to_discrete_acceleration(forces, dt_ps)

        if not reference_added:
            reference_displacements = crystal_to_flat_values(
                np.asarray(data["displacements"][start:], dtype=np.float64),
                atom_order,
            )
            summaries.append(
                summarize(
                    "reference",
                    discrete_acceleration(reference_displacements),
                    reference_displacements[1:-1],
                    central_velocity(reference_displacements, dt_ps),
                )
            )
            reference_added = True

        summaries.append(summarize(label, acceleration[1:-1], displacements[1:-1], velocities[1:-1]))

    plot_path = output_dir / "acceleration_energy_alignment_comparison.png"
    metrics_path = output_dir / "acceleration_energy_alignment_comparison.txt"
    plot_alignment(plot_path, summaries, args.bins, args.percentile, args.title)
    write_metrics(metrics_path, summaries)
    print(f"Saved {plot_path}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
