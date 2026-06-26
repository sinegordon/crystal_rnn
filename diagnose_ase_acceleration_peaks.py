"""Diagnose structured peaks in ASE/RNN acceleration histograms.

The ASE trajectory stores both positions and effective forces.  This script
compares accelerations reconstructed from the saved trajectory,

    a_traj[t] = u[t + 1] - 2 u[t] + u[t - 1],

with accelerations reconstructed from the saved RNN forces.  If histogram
peaks appear in ``a_traj`` but not in ``a_force``, they are produced by the
integrator/thermostat/post-processing path.  If they appear in both, they are
already present in the model force field.
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
    parser.add_argument("--ase-path", required=True, help="ASE trajectory .npz.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostics.")
    parser.add_argument("--title", default="ASE acceleration peak diagnostics")
    parser.add_argument("--bins", type=int, default=260)
    parser.add_argument("--percentile", type=float, default=99.7)
    parser.add_argument("--max-frames", type=int, default=0, help="Use only the first N frames; 0 means all frames.")
    parser.add_argument("--peak-count", type=int, default=12, help="Number of strongest histogram peaks to report.")
    parser.add_argument(
        "--plot-sample-size",
        type=int,
        default=1200000,
        help="Maximum number of component pairs used in hexbin plots. Metrics still use all frames.",
    )
    return parser.parse_args()


def as_cell_matrix(value):
    """Return a 3x3 cell matrix."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,) or (3, 3)")
    return cell


def cell_matrix(data, ase):
    """Return the simulation cell matrix from ASE output or source data."""
    if "cell" in ase.files:
        return as_cell_matrix(ase["cell"])
    if "cell" in data.files:
        return as_cell_matrix(data["cell"])
    if "box_lengths" in data.files:
        return as_cell_matrix(data["box_lengths"])
    raise ValueError("No cell or box_lengths found")


def minimum_image(delta, cell):
    """Wrap position differences into the nearest periodic image."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(delta, dtype=np.float64) @ inverse_cell
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
    """Return discrete accelerations from displacement frames."""
    displacements = np.asarray(displacements, dtype=np.float64)
    if displacements.shape[0] < 3:
        raise ValueError("At least three displacement frames are required")
    return displacements[2:] - 2.0 * displacements[1:-1] + displacements[:-2]


def force_to_discrete_acceleration(forces, masses, dt_ps):
    """Convert ASE forces back to model discrete accelerations."""
    factor = AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM / float(dt_ps) ** 2
    masses = np.asarray(masses, dtype=np.float64)
    return np.asarray(forces, dtype=np.float64) / (masses[None, :, None] * factor)


def components(values):
    """Return flattened vector components."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def sample_xy(x, y, max_points):
    """Return deterministic downsampled x/y pairs for expensive scatter-like plots."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    count = min(x.size, y.size)
    x = x[:count]
    y = y[:count]
    if max_points is None or max_points <= 0 or count <= max_points:
        return x, y
    indices = np.linspace(0, count - 1, int(max_points), dtype=np.int64)
    return x[indices], y[indices]


def magnitudes(values):
    """Return flattened vector magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def robust_symmetric_range(*arrays, percentile):
    """Return a symmetric robust histogram range."""
    values = np.concatenate([np.abs(components(array)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def robust_positive_range(*arrays, percentile):
    """Return a positive robust histogram range."""
    values = np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    limit = float(np.percentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return 0.0, limit


def rms(values):
    """Return root-mean-square over all entries."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def correlation(a, b):
    """Return Pearson correlation between flattened arrays."""
    a = components(a)
    b = components(b)
    if a.size != b.size:
        frames = min(a.size, b.size)
        a = a[:frames]
        b = b[:frames]
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def magnitude_correlation(a, b):
    """Return Pearson correlation between flattened vector magnitudes."""
    return correlation(magnitudes(a), magnitudes(b))


def frame_rms(values):
    """Return component RMS for every time frame."""
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.mean(values.reshape(values.shape[0], -1) ** 2, axis=1))


def histogram_peaks(values, bins, hist_range, count):
    """Return strongest unsmoothed one-dimensional histogram peaks."""
    hist, edges = np.histogram(components(values), bins=bins, range=hist_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_indices = []
    for index in range(1, len(hist) - 1):
        if hist[index] > hist[index - 1] and hist[index] >= hist[index + 1]:
            peak_indices.append(index)
    if not peak_indices:
        peak_indices = list(np.argsort(hist)[-count:])
    peak_indices = sorted(peak_indices, key=lambda index: hist[index], reverse=True)[:count]
    peak_indices = sorted(peak_indices, key=lambda index: centers[index])
    return [(float(centers[index]), float(hist[index])) for index in peak_indices]


def peak_spacing(peaks):
    """Estimate median spacing between same-sign neighboring peaks."""
    centers = np.asarray([center for center, _height in peaks], dtype=np.float64)
    centers = centers[np.abs(centers) > 0]
    if centers.size < 3:
        return float("nan")
    positive = np.sort(centers[centers > 0])
    negative = np.sort(-centers[centers < 0])
    spacings = []
    if positive.size >= 2:
        spacings.extend(np.diff(positive).tolist())
    if negative.size >= 2:
        spacings.extend(np.diff(negative).tolist())
    if not spacings:
        return float("nan")
    return float(np.median(spacings))


def metric_lines(label, values):
    """Return compact scalar metrics for a vector quantity."""
    comp = components(values)
    mag = magnitudes(values)
    return [
        f"{label}_component_mean = {comp.mean():.10g}",
        f"{label}_component_std = {comp.std():.10g}",
        f"{label}_component_rms = {rms(comp):.10g}",
        f"{label}_magnitude_mean = {mag.mean():.10g}",
        f"{label}_magnitude_p95 = {np.percentile(mag, 95):.10g}",
        f"{label}_magnitude_p99 = {np.percentile(mag, 99):.10g}",
    ]


def plot_overlay(output_path, title, acceleration_sets, bins, percentile):
    """Plot component and magnitude acceleration histograms."""
    component_range = robust_symmetric_range(*[values for _label, values in acceleration_sets], percentile=percentile)
    magnitude_range = robust_positive_range(
        *[magnitudes(values) for _label, values in acceleration_sets],
        percentile=percentile,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for label, values in acceleration_sets:
        axes[0].hist(
            components(values),
            bins=bins,
            range=component_range,
            density=True,
            alpha=0.42,
            label=label,
        )
        axes[1].hist(
            magnitudes(values),
            bins=bins,
            range=magnitude_range,
            density=True,
            alpha=0.42,
            label=label,
        )
    axes[0].set_title("Acceleration components")
    axes[0].set_xlabel("a component, A/step^2")
    axes[1].set_title("Acceleration magnitudes")
    axes[1].set_xlabel("|a|, A/step^2")
    for ax in axes:
        ax.set_ylabel("density")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_components(output_path, title, acceleration_sets, bins, percentile):
    """Plot x/y/z component histograms separately."""
    component_range = robust_symmetric_range(*[values for _label, values in acceleration_sets], percentile=percentile)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for component_index, ax in enumerate(axes):
        for label, values in acceleration_sets:
            ax.hist(
                np.asarray(values, dtype=np.float64)[..., component_index].reshape(-1),
                bins=bins,
                range=component_range,
                density=True,
                alpha=0.42,
                label=label,
            )
        ax.set_title(f"component {component_index}")
        ax.set_ylabel("density")
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("a component, A/step^2")
    axes[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_force_vs_trajectory(output_path, title, trajectory_acceleration, force_acceleration, plot_sample_size):
    """Plot direct comparison between trajectory and force-derived accelerations."""
    frames = min(trajectory_acceleration.shape[0], force_acceleration.shape[0])
    a_force_all = components(force_acceleration[:frames])
    a_traj_all = components(trajectory_acceleration[:frames])
    a_force, a_traj = sample_xy(a_force_all, a_traj_all, plot_sample_size)
    limit = float(np.percentile(np.abs(np.concatenate([a_traj, a_force])), 99.5))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    residual = a_traj_all - a_force_all

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hexbin(a_force, a_traj, gridsize=90, extent=(-limit, limit, -limit, limit), mincnt=1, cmap="viridis")
    axes[0].plot([-limit, limit], [-limit, limit], color="white", linewidth=1.0)
    axes[0].set_title("a_traj vs a_force")
    axes[0].set_xlabel("a_force component, A/step^2")
    axes[0].set_ylabel("a_traj component, A/step^2")
    axes[0].grid(alpha=0.18)

    residual_limit = float(np.percentile(np.abs(residual), 99.5))
    if not np.isfinite(residual_limit) or residual_limit <= 0:
        residual_limit = 1.0
    axes[1].hist(residual, bins=180, range=(-residual_limit, residual_limit), density=True, color="tab:red", alpha=0.7)
    axes[1].set_title("a_traj - a_force")
    axes[1].set_xlabel("residual component, A/step^2")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.22)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_series(output_path, title, acceleration_sets):
    """Plot frame-wise RMS for acceleration sources."""
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for label, values in acceleration_sets:
        ax.plot(frame_rms(values), linewidth=1.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("acceleration frame")
    ax.set_ylabel("component RMS, A/step^2")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def top_frame_lines(label, values, count=10):
    """Return rows for frames with the largest acceleration RMS."""
    values = np.asarray(values, dtype=np.float64)
    scores = frame_rms(values)
    indices = np.argsort(scores)[-count:][::-1]
    lines = [f"{label}_top_frames\tframe_index\tcomponent_rms\tmax_abs_component"]
    for index in indices:
        frame = values[index]
        lines.append(f"{label}_top_frames\t{int(index)}\t{scores[index]:.10g}\t{np.max(np.abs(frame)):.10g}")
    return lines


def central_velocity(displacements, dt_ps):
    """Return centered finite-difference velocities aligned with discrete accelerations."""
    displacements = np.asarray(displacements, dtype=np.float64)
    return (displacements[2:] - displacements[:-2]) / (2.0 * float(dt_ps))


def vector_dot(a, b):
    """Return per-atom vector dot products over all frames."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    frames = min(a.shape[0], b.shape[0])
    return np.sum(a[:frames] * b[:frames], axis=-1)


def scalar_stats(label, values):
    """Return common statistics for a scalar sample."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    positive = values[values > 0]
    negative = values[values < 0]
    abs_sum = np.sum(np.abs(values))
    positive_sum = np.sum(positive) if positive.size else 0.0
    negative_abs_sum = -np.sum(negative) if negative.size else 0.0
    if abs_sum > 0:
        positive_abs_fraction = positive_sum / abs_sum
        negative_abs_fraction = negative_abs_sum / abs_sum
    else:
        positive_abs_fraction = float("nan")
        negative_abs_fraction = float("nan")
    return [
        f"{label}_mean = {values.mean():.10g}",
        f"{label}_std = {values.std():.10g}",
        f"{label}_p05 = {np.percentile(values, 5):.10g}",
        f"{label}_p50 = {np.percentile(values, 50):.10g}",
        f"{label}_p95 = {np.percentile(values, 95):.10g}",
        f"{label}_positive_fraction = {np.mean(values > 0):.10g}",
        f"{label}_positive_abs_fraction = {positive_abs_fraction:.10g}",
        f"{label}_negative_abs_fraction = {negative_abs_fraction:.10g}",
    ]


def alignment_lines(label, acceleration, displacements, velocities):
    """Return metrics showing whether acceleration restores or pumps motion."""
    frames = min(acceleration.shape[0], displacements.shape[0], velocities.shape[0])
    acceleration = np.asarray(acceleration[:frames], dtype=np.float64)
    displacements = np.asarray(displacements[:frames], dtype=np.float64)
    velocities = np.asarray(velocities[:frames], dtype=np.float64)
    dot_au = vector_dot(acceleration, displacements)
    dot_av = vector_dot(acceleration, velocities)
    u2 = np.sum(displacements * displacements)
    v2 = np.sum(velocities * velocities)
    stiffness_proxy = -float(np.sum(acceleration * displacements)) / u2 if u2 > 0 else float("nan")
    damping_proxy = -float(np.sum(acceleration * velocities)) / v2 if v2 > 0 else float("nan")
    lines = [
        f"{label}_component_corr_a_u = {correlation(acceleration, displacements):.10g}",
        f"{label}_component_corr_a_v = {correlation(acceleration, velocities):.10g}",
        f"{label}_magnitude_corr_abs_a_abs_u = {magnitude_correlation(acceleration, displacements):.10g}",
        f"{label}_magnitude_corr_abs_a_abs_v = {magnitude_correlation(acceleration, velocities):.10g}",
        f"{label}_stiffness_proxy_minus_au_over_u2 = {stiffness_proxy:.10g}",
        f"{label}_damping_proxy_minus_av_over_v2 = {damping_proxy:.10g}",
    ]
    lines.extend(scalar_stats(f"{label}_dot_a_u", dot_au))
    lines.extend(scalar_stats(f"{label}_dot_a_v", dot_av))
    return lines


def plot_alignment_histograms(output_path, title, alignment_sets, bins, percentile):
    """Plot distributions of a.u and a.v for each acceleration source."""
    dot_au_values = [vector_dot(acceleration, displacement) for _label, acceleration, displacement, _velocity in alignment_sets]
    dot_av_values = [vector_dot(acceleration, velocity) for _label, acceleration, _displacement, velocity in alignment_sets]
    au_range = robust_symmetric_range(*dot_au_values, percentile=percentile)
    av_range = robust_symmetric_range(*dot_av_values, percentile=percentile)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for label, acceleration, displacement, velocity in alignment_sets:
        axes[0].hist(
            vector_dot(acceleration, displacement).reshape(-1),
            bins=bins,
            range=au_range,
            density=True,
            alpha=0.42,
            label=label,
        )
        axes[1].hist(
            vector_dot(acceleration, velocity).reshape(-1),
            bins=bins,
            range=av_range,
            density=True,
            alpha=0.42,
            label=label,
        )
    axes[0].axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Restoring alignment: a dot u")
    axes[0].set_xlabel("a.u, A^2/step^2")
    axes[1].set_title("Power alignment: a dot v")
    axes[1].set_xlabel("a.v, A^2/(step^2 ps)")
    for ax in axes:
        ax.set_ylabel("density")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_component_alignment(output_path, title, alignment_sets, plot_sample_size):
    """Plot acceleration components against displacement and velocity components."""
    fig, axes = plt.subplots(2, len(alignment_sets), figsize=(5 * len(alignment_sets), 9), constrained_layout=True)
    if len(alignment_sets) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (label, acceleration, displacement, velocity) in enumerate(alignment_sets):
        a_comp_all = components(acceleration)
        u_comp_all = components(displacement)
        v_comp_all = components(velocity)
        u_comp, a_comp_u = sample_xy(u_comp_all, a_comp_all, plot_sample_size)
        v_comp, a_comp_v = sample_xy(v_comp_all, a_comp_all, plot_sample_size)
        u_limit = float(np.percentile(np.abs(u_comp_all), 99.5))
        v_limit = float(np.percentile(np.abs(v_comp_all), 99.5))
        a_limit = float(np.percentile(np.abs(a_comp_all), 99.5))
        if not np.isfinite(u_limit) or u_limit <= 0:
            u_limit = 1.0
        if not np.isfinite(v_limit) or v_limit <= 0:
            v_limit = 1.0
        if not np.isfinite(a_limit) or a_limit <= 0:
            a_limit = 1.0

        axes[0, column].hexbin(
            u_comp,
            a_comp_u,
            gridsize=90,
            extent=(-u_limit, u_limit, -a_limit, a_limit),
            mincnt=1,
            cmap="viridis",
        )
        axes[0, column].axhline(0.0, color="white", linewidth=0.8, alpha=0.8)
        axes[0, column].axvline(0.0, color="white", linewidth=0.8, alpha=0.8)
        axes[0, column].set_title(f"{label}: a vs u")
        axes[0, column].set_xlabel("u component, A")
        axes[0, column].set_ylabel("a component, A/step^2")
        axes[0, column].grid(alpha=0.16)

        axes[1, column].hexbin(
            v_comp,
            a_comp_v,
            gridsize=90,
            extent=(-v_limit, v_limit, -a_limit, a_limit),
            mincnt=1,
            cmap="magma",
        )
        axes[1, column].axhline(0.0, color="white", linewidth=0.8, alpha=0.8)
        axes[1, column].axvline(0.0, color="white", linewidth=0.8, alpha=0.8)
        axes[1, column].set_title(f"{label}: a vs v")
        axes[1, column].set_xlabel("v component, A/ps")
        axes[1, column].set_ylabel("a component, A/step^2")
        axes[1, column].grid(alpha=0.16)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    """Run acceleration peak diagnostics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if args.plot_sample_size < 0:
        raise ValueError("plot-sample-size must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    cell = cell_matrix(data, ase)
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0

    positions = np.asarray(ase["positions"], dtype=np.float64)
    forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
    if args.max_frames > 0:
        positions = positions[: args.max_frames]
        forces = forces[: args.max_frames]
    masses = np.full(positions.shape[1], CU_MASS_AMU, dtype=np.float64)

    inference_displacements = minimum_image(positions - reference_positions[None, :, :], cell)
    trajectory_acceleration_full = discrete_acceleration(inference_displacements)
    force_acceleration = force_to_discrete_acceleration(forces, masses, dt_ps)
    force_aligned_full = force_acceleration[1:-1]
    inference_displacements_aligned_full = inference_displacements[1:-1]
    if "velocities_ang_per_ps" in ase.files:
        velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
        if args.max_frames > 0:
            velocities = velocities[: args.max_frames]
        inference_velocities_aligned_full = velocities[1:-1]
    else:
        inference_velocities_aligned_full = central_velocity(inference_displacements, dt_ps)

    reference_crystal = np.asarray(data["displacements"][start:], dtype=np.float64)
    if args.max_frames > 0:
        reference_crystal = reference_crystal[: args.max_frames]
    reference_displacements = crystal_to_flat_values(reference_crystal, atom_order)
    reference_acceleration = discrete_acceleration(reference_displacements)
    reference_displacements_aligned = reference_displacements[1:-1]
    reference_velocities_aligned = central_velocity(reference_displacements, dt_ps)

    full_frames = min(trajectory_acceleration_full.shape[0], force_aligned_full.shape[0])
    trajectory_acceleration_full = trajectory_acceleration_full[:full_frames]
    force_aligned_full = force_aligned_full[:full_frames]
    inference_displacements_aligned_full = inference_displacements_aligned_full[:full_frames]
    inference_velocities_aligned_full = inference_velocities_aligned_full[:full_frames]

    frames = min(full_frames, reference_acceleration.shape[0])
    trajectory_acceleration = trajectory_acceleration_full[:frames]
    force_aligned = force_aligned_full[:frames]
    inference_displacements_aligned = inference_displacements_aligned_full[:frames]
    inference_velocities_aligned = inference_velocities_aligned_full[:frames]
    reference_acceleration = reference_acceleration[:frames]
    reference_displacements_aligned = reference_displacements_aligned[:frames]
    reference_velocities_aligned = reference_velocities_aligned[:frames]

    acceleration_sets = [
        ("reference discrete", reference_acceleration),
        ("ASE trajectory diff", trajectory_acceleration),
        ("ASE force/model", force_aligned),
    ]
    plot_overlay(
        output_dir / "acceleration_source_overlay.png",
        args.title,
        acceleration_sets,
        args.bins,
        args.percentile,
    )
    plot_components(
        output_dir / "acceleration_components_by_source.png",
        args.title,
        acceleration_sets,
        args.bins,
        args.percentile,
    )
    plot_force_vs_trajectory(
        output_dir / "trajectory_vs_force_acceleration.png",
        args.title,
        trajectory_acceleration,
        force_aligned,
        args.plot_sample_size,
    )
    plot_time_series(
        output_dir / "acceleration_frame_rms.png",
        args.title,
        acceleration_sets,
    )
    full_acceleration_sets = [
        ("ASE trajectory diff full", trajectory_acceleration_full),
        ("ASE force/model full", force_aligned_full),
    ]
    plot_overlay(
        output_dir / "acceleration_inference_full_overlay.png",
        f"{args.title}: full inference",
        full_acceleration_sets,
        args.bins,
        args.percentile,
    )
    plot_force_vs_trajectory(
        output_dir / "trajectory_vs_force_acceleration_full.png",
        f"{args.title}: full inference",
        trajectory_acceleration_full,
        force_aligned_full,
        args.plot_sample_size,
    )
    alignment_sets = [
        ("reference discrete", reference_acceleration, reference_displacements_aligned, reference_velocities_aligned),
        ("ASE trajectory diff", trajectory_acceleration, inference_displacements_aligned, inference_velocities_aligned),
        ("ASE force/model", force_aligned, inference_displacements_aligned, inference_velocities_aligned),
    ]
    plot_alignment_histograms(
        output_dir / "acceleration_energy_alignment.png",
        args.title,
        alignment_sets,
        args.bins,
        args.percentile,
    )
    plot_component_alignment(
        output_dir / "acceleration_component_alignment.png",
        args.title,
        alignment_sets,
        args.plot_sample_size,
    )
    full_alignment_sets = [
        (
            "ASE trajectory diff full",
            trajectory_acceleration_full,
            inference_displacements_aligned_full,
            inference_velocities_aligned_full,
        ),
        ("ASE force/model full", force_aligned_full, inference_displacements_aligned_full, inference_velocities_aligned_full),
    ]
    plot_alignment_histograms(
        output_dir / "acceleration_energy_alignment_full.png",
        f"{args.title}: full inference",
        full_alignment_sets,
        args.bins,
        args.percentile,
    )
    plot_component_alignment(
        output_dir / "acceleration_component_alignment_full.png",
        f"{args.title}: full inference",
        full_alignment_sets,
        args.plot_sample_size,
    )

    hist_range = robust_symmetric_range(
        reference_acceleration,
        trajectory_acceleration,
        force_aligned,
        percentile=args.percentile,
    )
    lines = [
        f"ase_path = {args.ase_path}",
        f"data_path = {args.data_path}",
        f"frames_compared = {frames}",
        f"full_inference_frames_compared = {full_frames}",
        f"dt_ps = {dt_ps:.10g}",
        f"histogram_component_range = {hist_range[0]:.10g}\t{hist_range[1]:.10g}",
        f"trajectory_force_component_correlation = {correlation(trajectory_acceleration, force_aligned):.10g}",
        f"trajectory_force_component_correlation_full = {correlation(trajectory_acceleration_full, force_aligned_full):.10g}",
        f"trajectory_reference_component_correlation = {correlation(trajectory_acceleration, reference_acceleration):.10g}",
        f"force_reference_component_correlation = {correlation(force_aligned, reference_acceleration):.10g}",
    ]
    for label, values in acceleration_sets:
        normalized = label.lower().replace("/", "_").replace(" ", "_")
        lines.extend(metric_lines(normalized, values))
        peaks = histogram_peaks(values, args.bins, hist_range, args.peak_count)
        lines.append(f"{normalized}_peak_spacing_estimate = {peak_spacing(peaks):.10g}")
        lines.append(f"{normalized}_strong_component_peaks\tcenter\tdensity")
        lines.extend([f"{normalized}_strong_component_peaks\t{center:.10g}\t{height:.10g}" for center, height in peaks])
        lines.extend(top_frame_lines(normalized, values))
    for label, acceleration, displacement, velocity in alignment_sets:
        normalized = label.lower().replace("/", "_").replace(" ", "_")
        lines.extend(alignment_lines(normalized, acceleration, displacement, velocity))
    full_hist_range = robust_symmetric_range(
        trajectory_acceleration_full,
        force_aligned_full,
        percentile=args.percentile,
    )
    lines.append(f"full_inference_histogram_component_range = {full_hist_range[0]:.10g}\t{full_hist_range[1]:.10g}")
    for label, values in full_acceleration_sets:
        normalized = label.lower().replace("/", "_").replace(" ", "_")
        lines.extend(metric_lines(normalized, values))
        peaks = histogram_peaks(values, args.bins, full_hist_range, args.peak_count)
        lines.append(f"{normalized}_peak_spacing_estimate = {peak_spacing(peaks):.10g}")
        lines.append(f"{normalized}_strong_component_peaks\tcenter\tdensity")
        lines.extend([f"{normalized}_strong_component_peaks\t{center:.10g}\t{height:.10g}" for center, height in peaks])
        lines.extend(top_frame_lines(normalized, values))
    for label, acceleration, displacement, velocity in full_alignment_sets:
        normalized = label.lower().replace("/", "_").replace(" ", "_")
        lines.extend(alignment_lines(normalized, acceleration, displacement, velocity))

    metrics_path = output_dir / "acceleration_peak_diagnostics.txt"
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_dir / 'acceleration_source_overlay.png'}")
    print(f"Saved {output_dir / 'acceleration_components_by_source.png'}")
    print(f"Saved {output_dir / 'trajectory_vs_force_acceleration.png'}")
    print(f"Saved {output_dir / 'acceleration_frame_rms.png'}")
    print(f"Saved {output_dir / 'acceleration_inference_full_overlay.png'}")
    print(f"Saved {output_dir / 'trajectory_vs_force_acceleration_full.png'}")
    print(f"Saved {output_dir / 'acceleration_energy_alignment.png'}")
    print(f"Saved {output_dir / 'acceleration_component_alignment.png'}")
    print(f"Saved {output_dir / 'acceleration_energy_alignment_full.png'}")
    print(f"Saved {output_dir / 'acceleration_component_alignment_full.png'}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
