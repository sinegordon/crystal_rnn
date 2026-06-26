#!/usr/bin/env python3
"""Diagnose ASE trajectory power and local history-Jacobian phases.

The diagnostic is intended for long ASE inference runs.  It compares beginning,
middle, and end windows of one trajectory and estimates the local velocity
response used by the history-damping correction:

    G = d a / d(u_t - u_{t-1}) at fixed u_t ~= -d a / d u_{t-1}

For every sampled local patch the script reports the positive symmetric
response ``G_pos`` and the local powers before and after the eta correction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_classes import CU_MASS_AMU, forces_to_discrete_accelerations, positions_to_crystal_displacements


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Saved edge/pair-force RNN model path.")
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and metrics.")
    parser.add_argument("--eta", type=float, default=8.0, help="History damping eta used for corrected local power.")
    parser.add_argument("--window-frames", type=int, default=1000, help="Frames per beginning/middle/end power window.")
    parser.add_argument("--jacobian-samples", type=int, default=240, help="Local patches sampled per phase.")
    parser.add_argument("--batch-size", type=int, default=16, help="Autograd batch size.")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--percentile", type=float, default=99.0)
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
    return torch.device(device)


def load_model(path, device):
    """Load a saved model and move it to the requested device."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(model, "_center_acceleration_from_patch_tensor"):
        raise TypeError("Model must provide _center_acceleration_from_patch_tensor")
    if hasattr(model, "to"):
        model.to(device)
    elif hasattr(model, "model"):
        model.model.to(device)
        model.torch_device = device
    if hasattr(model, "model"):
        model.model.eval()
    return model


def as_box_lengths(data, ase):
    """Return orthorhombic box lengths from data or ASE output."""
    if "box_lengths" in data.files:
        box = np.asarray(data["box_lengths"], dtype=np.float64)
        return box[0] if box.ndim == 2 else box
    cell = np.asarray(ase["cell"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.shape == (3,):
        return cell
    return np.diag(cell)


def phase_slices(frame_count, window):
    """Return beginning, middle, and end frame slices."""
    if window <= 0:
        raise ValueError("window-frames must be positive")
    if frame_count < 3:
        raise ValueError("ASE trajectory needs at least three frames")
    window = min(int(window), frame_count - 2)
    begin = slice(2, 2 + window)
    middle_start = max(2, frame_count // 2 - window // 2)
    middle_start = min(middle_start, frame_count - window)
    middle = slice(middle_start, middle_start + window)
    end = slice(frame_count - window, frame_count)
    return {"begin": begin, "middle": middle, "end": end}


def extract_patch(history, center):
    """Extract one periodic 3x3x3 local patch from a displacement history."""
    crystal_shape = history.shape[1:4]
    axes = []
    for value, size in zip(center, crystal_shape):
        axes.append((np.arange(value - 1, value + 2, dtype=np.int64) % size).astype(np.int64))
    return history[(slice(None), *np.ix_(axes[0], axes[1], axes[2]), slice(None), slice(None))]


def model_jacobians(model, patches, device):
    """Return model accelerations and d a / d(u_t-u_{t-1}) local Jacobians."""
    patches_tensor = torch.as_tensor(patches, dtype=torch.float32, device=device).detach().clone().requires_grad_(True)
    batch_size = int(patches.shape[0])
    output_size = int(patches.shape[5] * 3)
    with torch.backends.cudnn.flags(enabled=False):
        acceleration = model._center_acceleration_from_patch_tensor(patches_tensor).reshape(batch_size, output_size)
        rows = []
        for output_index in range(output_size):
            gradient = torch.autograd.grad(
                acceleration[:, output_index].sum(),
                patches_tensor,
                create_graph=False,
                retain_graph=True,
            )[0]
            rows.append((-gradient[:, -2, 1, 1, 1]).reshape(batch_size, output_size))
    response = torch.stack(rows, dim=1)
    return (
        acceleration.detach().cpu().numpy().astype(np.float64),
        response.detach().cpu().numpy().astype(np.float64),
    )


def summarize(values):
    """Return scalar summary statistics for one distribution."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "std", "rms", "p05", "p50", "p95", "p99", "positive_fraction")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def append_summary(rows, phase, quantity, values):
    """Append a summary row."""
    summary = summarize(values)
    rows.append({"phase": phase, "quantity": quantity, "count": int(np.asarray(values).size), **summary})


def phase_power(ase, data, atom_order, phases):
    """Return per-atom corrected-force power distributions for each phase."""
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    mass = float(np.asarray(data["atom_mass_amu"], dtype=np.float64)) if "atom_mass_amu" in data.files else CU_MASS_AMU
    forces = np.asarray(ase["forces_ev_per_ang"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    acceleration = forces_to_discrete_accelerations(forces, dt_ps=dt_ps, atom_mass_amu=mass).astype(np.float64)
    power = {}
    normalized = {}
    for name, slc in phases.items():
        a = acceleration[slc]
        v = velocities[slc]
        dot = np.sum(a * v, axis=-1)
        power[name] = dot.reshape(-1)
        crystal_a = a[:, atom_order, :]
        crystal_v = v[:, atom_order, :]
        numerator = float(np.mean(crystal_a * crystal_v))
        denominator = max(float(np.sqrt(np.mean(crystal_a**2)) * np.sqrt(np.mean(crystal_v**2))), 1e-30)
        normalized[name] = numerator / denominator
    return power, normalized


def phase_jacobian(model, device, displacements, atom_order, phases, samples, batch_size, eta, rng):
    """Sample local patches and return Jacobian/power distributions."""
    centers = list(np.ndindex(atom_order.shape[:3]))
    output_size = int(atom_order.shape[3] * 3)
    results = {}
    for name, slc in phases.items():
        frame_indices = np.arange(max(2, slc.start), slc.stop, dtype=np.int64)
        sample_count = min(int(samples), int(frame_indices.size * len(centers)))
        sampled_frames = rng.choice(frame_indices, size=sample_count, replace=True)
        sampled_centers = rng.integers(0, len(centers), size=sample_count)
        values = {
            "model_power_step": [],
            "corrected_model_power_step": [],
            "positive_response_power_step": [],
            "g_sym_quadratic_norm": [],
            "g_pos_quadratic_norm": [],
            "g_max_eig": [],
            "g_positive_trace": [],
        }
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            patches = []
            for frame, center_index in zip(sampled_frames[start:end], sampled_centers[start:end]):
                center = centers[int(center_index)]
                patches.append(extract_patch(displacements[frame - 2 : frame + 1], center))
            patches = np.asarray(patches, dtype=np.float32)
            acceleration, response = model_jacobians(model, patches, device)
            velocity = (patches[:, -1, 1, 1, 1] - patches[:, -2, 1, 1, 1]).reshape(len(patches), output_size)
            velocity_norm2 = np.sum(velocity**2, axis=1) + 1e-30
            for local_index in range(len(patches)):
                sym = 0.5 * (response[local_index] + response[local_index].T)
                eigenvalues, eigenvectors = np.linalg.eigh(sym)
                positive = np.clip(eigenvalues, 0.0, None)
                positive_response = (eigenvectors * positive[None, :]) @ eigenvectors.T
                v = velocity[local_index]
                a = acceleration[local_index]
                g_power = float(v @ positive_response @ v)
                sym_power = float(v @ sym @ v)
                corrected = a - float(eta) * (positive_response @ v)
                values["model_power_step"].append(float(np.sum(a * v)))
                values["corrected_model_power_step"].append(float(np.sum(corrected * v)))
                values["positive_response_power_step"].append(g_power)
                values["g_sym_quadratic_norm"].append(sym_power / float(velocity_norm2[local_index]))
                values["g_pos_quadratic_norm"].append(g_power / float(velocity_norm2[local_index]))
                values["g_max_eig"].append(float(np.max(eigenvalues)))
                values["g_positive_trace"].append(float(np.sum(positive)))
        results[name] = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    return results


def write_metrics(path, rows):
    """Write summary rows to TSV."""
    fields = [
        "phase",
        "quantity",
        "count",
        "mean",
        "std",
        "rms",
        "p05",
        "p50",
        "p95",
        "p99",
        "positive_fraction",
        "normalized_power",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def robust_range(series, percentile, symmetric=True):
    """Return robust histogram range."""
    values = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        limit = float(np.percentile(np.abs(values), percentile))
        return (-limit, limit) if limit > 0 else (-1.0, 1.0)
    low = float(np.percentile(values, 100 - percentile))
    high = float(np.percentile(values, percentile))
    return (low, high) if high > low else (0.0, 1.0)


def plot_diagnostics(path, phase_power_values, jacobian_values, bins, percentile):
    """Plot phase histograms."""
    phase_names = ["begin", "middle", "end"]
    panels = [
        ("corrected force a·v", [phase_power_values[p] for p in phase_names], True),
        ("model local a·du", [jacobian_values[p]["model_power_step"] for p in phase_names], True),
        ("eta-corrected local a·du", [jacobian_values[p]["corrected_model_power_step"] for p in phase_names], True),
        ("v.T G_pos v / |v|^2", [jacobian_values[p]["g_pos_quadratic_norm"] for p in phase_names], False),
        ("max eig sym(G)", [jacobian_values[p]["g_max_eig"] for p in phase_names], True),
        ("positive trace sym(G)", [jacobian_values[p]["g_positive_trace"] for p in phase_names], False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    colors = {"begin": "tab:blue", "middle": "tab:orange", "end": "tab:green"}
    for axis, (title, arrays, symmetric) in zip(axes.ravel(), panels):
        hist_range = robust_range(arrays, percentile, symmetric=symmetric)
        for phase, values in zip(phase_names, arrays):
            axis.hist(
                values,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=phase,
                color=colors[phase],
            )
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Run phase diagnostics."""
    args = parse_args()
    if args.eta < 0:
        raise ValueError("eta must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.jacobian_samples <= 0:
        raise ValueError("jacobian-samples must be positive")

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = load_model(args.model_path, device)
    ase = np.load(args.ase_path)
    data = np.load(args.data_path)

    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    box_lengths = as_box_lengths(data, ase)
    displacements = positions_to_crystal_displacements(
        np.asarray(ase["positions"], dtype=np.float32),
        np.asarray(data["reference_positions"], dtype=np.float32),
        atom_order,
        box_lengths,
    )
    phases = phase_slices(displacements.shape[0], args.window_frames)
    force_power, normalized_power = phase_power(ase, data, atom_order, phases)
    jacobian = phase_jacobian(
        model=model,
        device=device,
        displacements=displacements,
        atom_order=atom_order,
        phases=phases,
        samples=args.jacobian_samples,
        batch_size=args.batch_size,
        eta=args.eta,
        rng=rng,
    )

    rows = []
    for phase in ("begin", "middle", "end"):
        append_summary(rows, phase, "corrected_force_power", force_power[phase])
        rows[-1]["normalized_power"] = normalized_power[phase]
        for quantity, values in jacobian[phase].items():
            append_summary(rows, phase, quantity, values)
            rows[-1]["normalized_power"] = ""

    metrics_path = output_dir / "ase_phase_jacobian_metrics.tsv"
    plot_path = output_dir / "ase_phase_jacobian_histograms.png"
    samples_path = output_dir / "ase_phase_jacobian_samples.npz"
    write_metrics(metrics_path, rows)
    plot_diagnostics(plot_path, force_power, jacobian, args.bins, args.percentile)
    np.savez_compressed(
        samples_path,
        **{f"{phase}_force_power": force_power[phase] for phase in force_power},
        **{f"{phase}_{key}": value for phase, values in jacobian.items() for key, value in values.items()},
    )
    print(f"Saved {metrics_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {samples_path}")


if __name__ == "__main__":
    main()
