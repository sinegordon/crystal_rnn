#!/usr/bin/env python3
"""Diagnose history/velocity anti-damping in a trained local RNN model.

The edge and pair-force RNNs consume a short displacement history, so their
acceleration is not just a static force field ``a(u_t)``.  This diagnostic
estimates local Jacobians with respect to the previous and current central-cell
displacements and derives an effective velocity response

    G = d a / d v,  v = u_t - u_{t-1}

at fixed current displacement.  Positive values of ``v.T @ sym(G) @ v`` are a
direct local anti-damping signal: the history-dependent part of the model
pushes along the current velocity.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Saved torch model path.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal npz with X_blocks.")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and metrics.")
    parser.add_argument("--samples", type=int, default=500, help="Number of sampled local blocks.")
    parser.add_argument("--batch-size", type=int, default=16, help="Autograd batch size.")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=140)
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
    if device.startswith("mps") and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise ValueError("MPS requested but unavailable")
    return torch.device(device)


def load_model(path, device):
    """Load a saved model and move it to the requested device."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(model, "_center_acceleration_from_patch_tensor"):
        raise TypeError("Model must provide _center_acceleration_from_patch_tensor for autograd diagnostics")
    if hasattr(model, "to"):
        model.to(device)
    elif hasattr(model, "model"):
        model.model.to(device)
        model.torch_device = device
    else:
        raise TypeError("Loaded object does not look like a supported model")
    if hasattr(model, "model"):
        model.model.eval()
    return model


def central_force_targets(data):
    """Return optional central-cell force targets aligned with X_blocks."""
    if "force_acceleration_blocks" not in data.files:
        return None
    target = np.asarray(data["force_acceleration_blocks"], dtype=np.float64)
    return target[:, 1, 1, 1]


def jacobians_for_batch(model, patches, device):
    """Return model accelerations and Jacobians wrt previous/current centers."""
    patches_tensor = torch.as_tensor(patches, dtype=torch.float32, device=device).detach().clone().requires_grad_(True)
    unit_cell_atoms = int(patches.shape[5])
    output_size = unit_cell_atoms * 3
    with torch.backends.cudnn.flags(enabled=False):
        acceleration = model._center_acceleration_from_patch_tensor(patches_tensor).reshape(patches.shape[0], output_size)
        previous_rows = []
        current_rows = []
        for output_index in range(output_size):
            gradient = torch.autograd.grad(
                acceleration[:, output_index].sum(),
                patches_tensor,
                create_graph=False,
                retain_graph=True,
            )[0]
            previous_rows.append(gradient[:, -2, 1, 1, 1].reshape(patches.shape[0], output_size))
            current_rows.append(gradient[:, -1, 1, 1, 1].reshape(patches.shape[0], output_size))
    previous_jacobian = torch.stack(previous_rows, dim=1)
    current_jacobian = torch.stack(current_rows, dim=1)
    return (
        acceleration.detach().cpu().numpy().astype(np.float64),
        previous_jacobian.detach().cpu().numpy().astype(np.float64),
        current_jacobian.detach().cpu().numpy().astype(np.float64),
    )


def summarize(values):
    """Return scalar summary statistics."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    abs_values = np.abs(finite)
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "mean_abs": float(np.mean(abs_values)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "positive_fraction": float(np.mean(finite > 0.0)),
    }


def write_metrics(path, rows, args):
    """Write summary metrics as TSV."""
    fields = [
        "quantity",
        "count",
        "mean",
        "std",
        "rms",
        "mean_abs",
        "p50",
        "p95",
        "p99",
        "positive_fraction",
        "samples",
        "model_path",
        "data_path",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for quantity, values in rows.items():
            row = {"quantity": quantity, **summarize(values)}
            row.update({"samples": args.samples, "model_path": args.model_path, "data_path": args.data_path})
            writer.writerow(row)


def robust_symmetric_range(values, percentile):
    """Return robust symmetric histogram limits."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    limit = float(np.percentile(np.abs(finite), percentile)) if finite.size else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def plot_histograms(path, values, bins, percentile):
    """Plot diagnostic histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = [
        ("model a · v", values["model_power"], "tab:orange", True),
        ("reference a · v", values.get("reference_power"), "tab:blue", True),
        ("v.T sym(G_fixed_current) v / |v|^2", values["g_fixed_current_quadratic_norm"], "tab:red", True),
        ("max eig sym(G_fixed_current)", values["g_fixed_current_max_eig"], "tab:purple", False),
    ]
    for ax, (title, series, color, symmetric) in zip(axes.ravel(), panels):
        if series is None:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            ax.set_axis_off()
            continue
        series = np.asarray(series, dtype=np.float64)
        finite = series[np.isfinite(series)]
        if symmetric:
            hist_range = robust_symmetric_range(finite, percentile)
        else:
            upper = float(np.percentile(finite, percentile)) if finite.size else 1.0
            lower = float(np.percentile(finite, 100.0 - percentile)) if finite.size else 0.0
            if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
                lower, upper = 0.0, 1.0
            hist_range = (lower, upper)
        ax.hist(finite, bins=bins, range=hist_range, density=True, color=color, alpha=0.72)
        ax.axvline(0.0, color="black", linewidth=0.9)
        ax.set_title(title)
        ax.grid(alpha=0.22)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Run the history/velocity Jacobian diagnostic."""
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    model = load_model(args.model_path, device)
    data = np.load(args.data_path)
    x_blocks = np.asarray(data["X_blocks"], dtype=np.float32)
    if x_blocks.ndim != 7 or x_blocks.shape[1] < 2:
        raise ValueError("X_blocks must have shape (samples, sequence>=2, 3, 3, 3, atoms, 3)")

    sample_count = min(int(args.samples), int(x_blocks.shape[0]))
    sample_indices = rng.choice(x_blocks.shape[0], size=sample_count, replace=False)
    target = central_force_targets(data)

    values = {
        "model_power": [],
        "g_fixed_current_quadratic": [],
        "g_fixed_current_quadratic_norm": [],
        "g_fixed_current_max_eig": [],
        "g_fixed_current_positive_trace": [],
        "g_midpoint_quadratic_norm": [],
        "g_midpoint_max_eig": [],
    }
    if target is not None:
        values["reference_power"] = []

    output_size = int(x_blocks.shape[5] * 3)
    for start in range(0, sample_count, args.batch_size):
        batch_indices = sample_indices[start : start + args.batch_size]
        patches = x_blocks[batch_indices]
        acceleration, previous_jacobian, current_jacobian = jacobians_for_batch(model, patches, device)
        velocity = (patches[:, -1, 1, 1, 1] - patches[:, -2, 1, 1, 1]).reshape(len(batch_indices), output_size)
        velocity_norm2 = np.sum(velocity**2, axis=1) + 1e-30

        g_fixed_current = -previous_jacobian
        g_midpoint = 0.5 * (current_jacobian - previous_jacobian)
        for local_index, global_index in enumerate(batch_indices):
            v = velocity[local_index]
            a = acceleration[local_index]
            sym_fixed = 0.5 * (g_fixed_current[local_index] + g_fixed_current[local_index].T)
            sym_midpoint = 0.5 * (g_midpoint[local_index] + g_midpoint[local_index].T)
            fixed_eig = np.linalg.eigvalsh(sym_fixed)
            midpoint_eig = np.linalg.eigvalsh(sym_midpoint)
            fixed_quad = float(v @ sym_fixed @ v)
            midpoint_quad = float(v @ sym_midpoint @ v)

            values["model_power"].append(float(np.sum(a * v)))
            values["g_fixed_current_quadratic"].append(fixed_quad)
            values["g_fixed_current_quadratic_norm"].append(fixed_quad / float(velocity_norm2[local_index]))
            values["g_fixed_current_max_eig"].append(float(np.max(fixed_eig)))
            values["g_fixed_current_positive_trace"].append(float(np.sum(np.maximum(fixed_eig, 0.0))))
            values["g_midpoint_quadratic_norm"].append(midpoint_quad / float(velocity_norm2[local_index]))
            values["g_midpoint_max_eig"].append(float(np.max(midpoint_eig)))
            if target is not None:
                values["reference_power"].append(float(np.sum(target[global_index].reshape(output_size) * v)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(series, dtype=np.float64) for key, series in values.items()}
    metrics_path = output_dir / "history_velocity_jacobian_metrics.tsv"
    plot_path = output_dir / "history_velocity_jacobian_histograms.png"
    samples_path = output_dir / "history_velocity_jacobian_samples.npz"
    write_metrics(metrics_path, arrays, args)
    plot_histograms(plot_path, arrays, args.bins, args.percentile)
    np.savez_compressed(samples_path, **arrays)
    print(f"Saved {metrics_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {samples_path}")


if __name__ == "__main__":
    main()
