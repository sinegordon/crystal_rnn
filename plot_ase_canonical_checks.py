"""Build canonical-ensemble diagnostics for an ASE RNN inference run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Robust histogram percentile.")
    parser.add_argument(
        "--frame-fraction-start",
        type=float,
        default=0.0,
        help="Start fraction of the ASE trajectory used for diagnostics, e.g. 0.5 for the second half.",
    )
    parser.add_argument(
        "--frame-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of the ASE trajectory used for diagnostics.",
    )
    parser.add_argument(
        "--frame-fraction-scope",
        choices=["ase", "overlap"],
        default="ase",
        help=(
            "Frame count used to convert frame fractions to indices. "
            "'overlap' means the reference-overlapping segment."
        ),
    )
    parser.add_argument(
        "--reference-window",
        choices=["same", "full"],
        default="same",
        help="'same' uses the aligned reference window; 'full' compares the inference window with all reference frames.",
    )
    parser.add_argument("--q-decimals", type=int, default=8, help="Decimal rounding used to group |q| shells.")
    parser.add_argument(
        "--exclude-q-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude spatial q=0 from q-mode metrics.",
    )
    parser.add_argument("--title", default="ASE canonical-ensemble checks")
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
        raise ValueError("cell must have shape (3,) or (3, 3)")
    return cell


def cell_matrix(data, ase) -> np.ndarray:
    """Return the simulation cell matrix from ASE output or source dataset."""
    if "cell" in ase.files:
        return as_cell_matrix(ase["cell"])
    if "cell" in data.files:
        return as_cell_matrix(data["cell"])
    if "box_lengths" in data.files:
        return as_cell_matrix(data["box_lengths"])
    raise ValueError("No cell or box_lengths found")


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap flat position differences into the nearest periodic image."""
    inverse_cell = np.linalg.inv(cell)
    fractional = np.asarray(delta, dtype=np.float64) @ inverse_cell
    fractional -= np.round(fractional)
    return fractional @ cell


def flat_to_crystal(flat_values: np.ndarray, atom_order: np.ndarray) -> np.ndarray:
    """Convert flat ASE atom order values into crystal layout."""
    flat_values = np.asarray(flat_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    return flat_values[:, atom_order.reshape(-1), :].reshape(flat_values.shape[0], *atom_order.shape, 3)


def crystal_to_flat(crystal_values: np.ndarray, atom_order: np.ndarray) -> np.ndarray:
    """Convert crystal-shaped values into flat ASE atom order."""
    crystal_values = np.asarray(crystal_values, dtype=np.float64)
    atom_order = np.asarray(atom_order, dtype=np.int64)
    flat = np.empty((crystal_values.shape[0], atom_order.size, 3), dtype=np.float64)
    flat[:, atom_order.reshape(-1), :] = crystal_values.reshape(crystal_values.shape[0], atom_order.size, 3)
    return flat


def q_grid(shape: tuple[int, int, int], cell: np.ndarray) -> np.ndarray:
    """Return |q| values in FFT index order for a rectangular supercell."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=np.float64), axis=1)
    q_axes = [2.0 * np.pi * np.fft.fftfreq(int(n), d=float(length) / int(n)) for n, length in zip(shape, lengths)]
    qx, qy, qz = np.meshgrid(q_axes[0], q_axes[1], q_axes[2], indexing="ij")
    return np.sqrt(qx**2 + qy**2 + qz**2)


def fft_power_by_q(values: np.ndarray) -> np.ndarray:
    """Return per-grid-point mean squared Fourier amplitude over cells and channels."""
    values = np.asarray(values, dtype=np.float64)
    cell_count = int(np.prod(values.shape[1:4]))
    spectrum = np.fft.fftn(values, axes=(1, 2, 3)) / np.sqrt(cell_count)
    return np.mean(np.abs(spectrum) ** 2, axis=(0, 4, 5))


def shell_average(q_abs: np.ndarray, values: np.ndarray, decimals: int, exclude_q_zero: bool) -> list[dict[str, float]]:
    """Average grid-point values over equal-|q| shells."""
    q_flat = np.asarray(q_abs, dtype=np.float64).reshape(-1)
    value_flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if exclude_q_zero:
        mask = q_flat > 0
        q_flat = q_flat[mask]
        value_flat = value_flat[mask]
    rounded = np.round(q_flat, decimals=int(decimals))
    rows = []
    for value in np.unique(rounded):
        mask = rounded == value
        rows.append(
            {
                "q": float(np.mean(q_flat[mask])),
                "mean": float(np.mean(value_flat[mask])),
                "std": float(np.std(value_flat[mask])),
                "count": int(np.count_nonzero(mask)),
            }
        )
    rows.sort(key=lambda row: row["q"])
    return rows


def rows_to_arrays(predicted_rows: list[dict[str, float]], reference_rows: list[dict[str, float]]):
    """Return common q and predicted/reference shell arrays."""
    if len(predicted_rows) != len(reference_rows):
        raise ValueError("Predicted/reference q shell counts differ")
    q = np.asarray([row["q"] for row in predicted_rows], dtype=np.float64)
    reference_q = np.asarray([row["q"] for row in reference_rows], dtype=np.float64)
    if not np.allclose(q, reference_q):
        raise ValueError("Predicted/reference q shell grids differ")
    predicted = np.asarray([row["mean"] for row in predicted_rows], dtype=np.float64)
    reference = np.asarray([row["mean"] for row in reference_rows], dtype=np.float64)
    counts = np.asarray([row["count"] for row in predicted_rows], dtype=np.int64)
    return q, predicted, reference, counts


def robust_symmetric_limit(*arrays: np.ndarray, percentile: float) -> float:
    """Return a symmetric robust histogram limit."""
    values = np.concatenate([np.abs(np.asarray(array, dtype=np.float64).reshape(-1)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(values, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def robust_positive_limit(*arrays: np.ndarray, percentile: float) -> float:
    """Return a robust positive histogram upper limit."""
    values = np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(values, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def component_values(values: np.ndarray) -> np.ndarray:
    """Return flattened vector components."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def speed_values(values: np.ndarray) -> np.ndarray:
    """Return flattened vector magnitudes."""
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    """Return compact scalar summary statistics."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "std", "rms", "p05", "p50", "p95", "p99")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def metric_row(quantity: str, source: str, values: np.ndarray, extra: dict[str, float] | None = None):
    """Return one metrics table row."""
    row = {"quantity": quantity, "source": source, "count": int(np.asarray(values).size), **summarize(values)}
    if extra:
        row.update(extra)
    return row


def load_canonical_arrays(ase_path: Path, data_path: Path):
    """Load inference arrays and the available stationary reference arrays."""
    ase = np.load(ase_path)
    data = np.load(data_path)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    cell = cell_matrix(data, ase)
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0

    positions = np.asarray(ase["positions"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    reference_displacements_all = np.asarray(data["displacements"], dtype=np.float64)
    available_reference_frames = max(0, int(reference_displacements_all.shape[0]) - start)
    overlap_frames = min(int(positions.shape[0]), available_reference_frames)
    if overlap_frames < 3:
        raise ValueError(
            f"Need at least three overlapping frames, got {overlap_frames}: "
            f"positions={positions.shape[0]}, reference_available={available_reference_frames}, start={start}"
        )

    predicted_displacements_flat = minimum_image(positions - reference_positions[None, :, :], cell)
    predicted_displacements = flat_to_crystal(predicted_displacements_flat, atom_order)
    reference_displacements = reference_displacements_all[start:]

    reference_flat = crystal_to_flat(reference_displacements, atom_order)
    reference_velocities_flat = np.diff(reference_flat, axis=0) / dt_ps
    reference_velocities = flat_to_crystal(reference_velocities_flat, atom_order)
    predicted_velocities = flat_to_crystal(velocities, atom_order)
    overlap_velocity_frames = min(int(predicted_velocities.shape[0]), int(reference_velocities.shape[0]))

    return {
        "ase": ase,
        "data": data,
        "cell": cell,
        "start": start,
        "frames": int(positions.shape[0]),
        "reference_frames": int(reference_displacements.shape[0]),
        "overlap_frames": overlap_frames,
        "velocity_frames": int(predicted_velocities.shape[0]),
        "reference_velocity_frames": int(reference_velocities.shape[0]),
        "overlap_velocity_frames": overlap_velocity_frames,
        "steps": np.asarray(ase["steps"], dtype=np.int64) if "steps" in ase.files else np.arange(positions.shape[0]),
        "temperature": np.asarray(ase["temperature_k"], dtype=np.float64),
        "predicted_velocities_full": velocities,
        "predicted_velocities": predicted_velocities,
        "reference_velocities": reference_velocities,
        "predicted_displacements": predicted_displacements,
        "reference_displacements": reference_displacements,
        "atom_order": atom_order,
    }


def slice_frames(
    arrays: dict[str, object],
    start_fraction: float,
    end_fraction: float,
    fraction_scope: str,
    reference_window: str,
) -> dict[str, object]:
    """Restrict diagnostics to a fractional ASE frame window."""
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("frame fractions must satisfy 0 <= start < end <= 1")

    if fraction_scope == "ase":
        total_frames = int(arrays["temperature"].shape[0])
    elif fraction_scope == "overlap":
        total_frames = int(arrays["overlap_frames"])
    else:
        raise ValueError(f"Unsupported frame fraction scope: {fraction_scope!r}")
    start_index = int(np.floor(total_frames * start_fraction))
    end_index = int(np.ceil(total_frames * end_fraction))
    start_index = min(max(start_index, 0), total_frames - 1)
    end_index = min(max(end_index, start_index + 3), total_frames)
    if end_index - start_index < 3:
        raise ValueError("Selected frame window must contain at least three frames")

    sliced = dict(arrays)
    sliced["frame_window_start"] = start_index
    sliced["frame_window_end"] = end_index
    sliced["temperature"] = np.asarray(arrays["temperature"])[start_index:end_index]
    sliced["steps"] = np.asarray(arrays["steps"])[start_index:end_index]
    sliced["predicted_velocities_full"] = np.asarray(arrays["predicted_velocities_full"])[start_index:end_index]

    predicted_end = min(end_index, int(arrays["frames"]))
    if predicted_end - start_index < 3:
        raise ValueError(
            "Selected inference window has fewer than three displacement frames: "
            f"window=({start_index}, {end_index}), inference_frames={arrays['frames']}"
        )

    predicted_velocity_end = min(end_index, int(arrays["velocity_frames"]))
    if predicted_velocity_end - start_index < 2:
        raise ValueError(
            "Selected inference window has fewer than two velocity frames: "
            f"window=({start_index}, {end_index}), inference_velocity_frames={arrays['velocity_frames']}"
        )

    sliced["predicted_displacements"] = np.asarray(arrays["predicted_displacements"])[start_index:predicted_end]
    sliced["predicted_velocities"] = np.asarray(arrays["predicted_velocities"])[start_index:predicted_velocity_end]

    if reference_window == "same":
        reference_end = min(end_index, int(arrays["reference_frames"]))
        reference_velocity_end = min(end_index, int(arrays["reference_velocity_frames"]))
        if reference_end - start_index < 3:
            raise ValueError(
                "Selected window has fewer than three aligned reference frames. "
                "Use --reference-window full for ensemble comparison against all reference frames."
            )
        if reference_velocity_end - start_index < 2:
            raise ValueError(
                "Selected window has fewer than two aligned reference velocity frames. "
                "Use --reference-window full for ensemble comparison against all reference frames."
            )
        reference_start = start_index
        reference_velocity_start = start_index
    elif reference_window == "full":
        reference_start = 0
        reference_end = int(arrays["reference_frames"])
        reference_velocity_start = 0
        reference_velocity_end = int(arrays["reference_velocity_frames"])
    else:
        raise ValueError(f"Unsupported reference window: {reference_window!r}")

    sliced["reference_displacements"] = np.asarray(arrays["reference_displacements"])[reference_start:reference_end]
    sliced["reference_velocities"] = np.asarray(arrays["reference_velocities"])[
        reference_velocity_start:reference_velocity_end
    ]
    sliced["frames"] = int(predicted_end - start_index)
    sliced["reference_frames"] = int(reference_end - reference_start)
    sliced["velocity_frames"] = int(predicted_velocity_end - start_index)
    sliced["reference_velocity_frames"] = int(reference_velocity_end - reference_velocity_start)
    sliced["reference_window"] = reference_window
    return sliced


def q_power_curves(values_pred: np.ndarray, values_ref: np.ndarray, cell: np.ndarray, decimals: int, exclude_q_zero: bool):
    """Return q shell arrays for mean squared mode amplitudes."""
    q_abs = q_grid(values_pred.shape[1:4], cell)
    predicted_rows = shell_average(q_abs, fft_power_by_q(values_pred), decimals, exclude_q_zero)
    reference_rows = shell_average(q_abs, fft_power_by_q(values_ref), decimals, exclude_q_zero)
    return rows_to_arrays(predicted_rows, reference_rows)


def rms_relative_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Return RMS relative curve error."""
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = predicted / reference - 1.0
    relative = relative[np.isfinite(relative)]
    if relative.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(relative**2)))


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write metrics as TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "quantity",
        "source",
        "count",
        "mean",
        "std",
        "rms",
        "p05",
        "p50",
        "p95",
        "p99",
        "ratio",
        "expected",
        "score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_hist_pair(ax, predicted: np.ndarray, reference: np.ndarray, title: str, xlabel: str, bins: int, percentile: float):
    """Plot a predicted/reference component histogram pair."""
    limit = robust_symmetric_limit(predicted, reference, percentile=percentile)
    ax.hist(component_values(reference), bins=bins, range=(-limit, limit), density=True, alpha=0.48, label="reference")
    ax.hist(component_values(predicted), bins=bins, range=(-limit, limit), density=True, alpha=0.48, label="ASE")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)


def plot_speed_pair(ax, predicted: np.ndarray, reference: np.ndarray, title: str, bins: int, percentile: float):
    """Plot a predicted/reference speed histogram pair."""
    limit = robust_positive_limit(speed_values(predicted), speed_values(reference), percentile=percentile)
    ax.hist(speed_values(reference), bins=bins, range=(0, limit), density=True, alpha=0.48, label="reference")
    ax.hist(speed_values(predicted), bins=bins, range=(0, limit), density=True, alpha=0.48, label="ASE")
    ax.set_title(title)
    ax.set_xlabel("|v|, A/ps")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)


def plot_q_ratio(ax, q: np.ndarray, predicted: np.ndarray, reference: np.ndarray, title: str):
    """Plot q-resolved ASE/reference mode-power ratio."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = predicted / reference
    ax.axhline(1.0, color="black", lw=1.0, alpha=0.65)
    ax.plot(q, ratio, "o-", ms=4, color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("|q|, 1/A")
    ax.set_ylabel("ASE/reference")
    ax.grid(alpha=0.25)


def main() -> int:
    """Build canonical checks and save plots/metrics."""
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    if not 0 < args.percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    arrays = load_canonical_arrays(Path(args.ase_path), Path(args.data_path))
    arrays = slice_frames(
        arrays,
        args.frame_fraction_start,
        args.frame_fraction_end,
        args.frame_fraction_scope,
        args.reference_window,
    )
    temperature = arrays["temperature"]
    predicted_v = arrays["predicted_velocities"]
    reference_v = arrays["reference_velocities"]
    predicted_u = arrays["predicted_displacements"]
    reference_u = arrays["reference_displacements"]
    q_u, pred_u_power, ref_u_power, _ = q_power_curves(
        predicted_u,
        reference_u,
        arrays["cell"],
        args.q_decimals,
        args.exclude_q_zero,
    )
    q_v, pred_v_power, ref_v_power, _ = q_power_curves(
        predicted_v,
        reference_v,
        arrays["cell"],
        args.q_decimals,
        args.exclude_q_zero,
    )

    target_temperature = float(np.asarray(arrays["ase"]["temperature_target_k"])) if "temperature_target_k" in arrays["ase"].files else 300.0
    dof = max(1, 3 * int(arrays["predicted_velocities_full"].shape[1]))
    expected_temperature_std = target_temperature * np.sqrt(2.0 / dof)
    rows = [
        {
            "quantity": "frame_window",
            "source": "ASE",
            "count": int(arrays["frame_window_end"] - arrays["frame_window_start"]),
            "mean": float(arrays["frame_window_start"]),
            "std": float(arrays["frame_window_end"]),
            "rms": float(args.frame_fraction_start),
            "p05": float(args.frame_fraction_end),
        },
        {
            "quantity": "reference_window",
            "source": str(arrays["reference_window"]),
            "count": int(arrays["reference_frames"]),
            "mean": float(arrays["reference_velocity_frames"]),
        },
        metric_row(
            "temperature_k",
            "ASE_full",
            temperature,
            {
                "ratio": float(np.mean(temperature) / target_temperature),
                "expected": target_temperature,
                "score": float(np.std(temperature) / expected_temperature_std),
            },
        ),
        metric_row(
            "velocity_component",
            "ASE_overlap",
            component_values(predicted_v),
            {"ratio": float(np.std(component_values(predicted_v)) / np.std(component_values(reference_v)))},
        ),
        metric_row("velocity_component", "reference_overlap", component_values(reference_v)),
        metric_row(
            "speed",
            "ASE_overlap",
            speed_values(predicted_v),
            {"ratio": float(np.mean(speed_values(predicted_v)) / np.mean(speed_values(reference_v)))},
        ),
        metric_row("speed", "reference_overlap", speed_values(reference_v)),
        metric_row(
            "displacement_component",
            "ASE_overlap",
            component_values(predicted_u),
            {"ratio": float(np.std(component_values(predicted_u)) / np.std(component_values(reference_u)))},
        ),
        metric_row("displacement_component", "reference_overlap", component_values(reference_u)),
        {
            "quantity": "q_displacement_power_ratio",
            "source": "ASE/reference",
            "count": int(q_u.size),
            "mean": float(np.mean(pred_u_power / ref_u_power)),
            "std": float(np.std(pred_u_power / ref_u_power)),
            "rms": float(np.sqrt(np.mean((pred_u_power / ref_u_power) ** 2))),
            "score": rms_relative_error(pred_u_power, ref_u_power),
        },
        {
            "quantity": "q_velocity_power_ratio",
            "source": "ASE/reference",
            "count": int(q_v.size),
            "mean": float(np.mean(pred_v_power / ref_v_power)),
            "std": float(np.std(pred_v_power / ref_v_power)),
            "rms": float(np.sqrt(np.mean((pred_v_power / ref_v_power) ** 2))),
            "score": rms_relative_error(pred_v_power, ref_v_power),
        },
    ]

    time_ps = np.arange(temperature.size, dtype=np.float64)
    if "dt_ps" in arrays["ase"].files:
        time_ps = np.asarray(arrays["steps"], dtype=np.float64) * float(np.asarray(arrays["ase"]["dt_ps"]))

    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time_ps, temperature, lw=0.9, color="tab:red")
    ax.axhline(target_temperature, color="black", lw=1.0, ls="--", alpha=0.65)
    ax.set_title("temperature trace")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("T, K")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.hist(temperature, bins=args.bins, density=True, alpha=0.7, color="tab:red")
    ax.axvline(target_temperature, color="black", lw=1.0, ls="--", alpha=0.65, label="target")
    ax.axvspan(
        target_temperature - expected_temperature_std,
        target_temperature + expected_temperature_std,
        color="black",
        alpha=0.08,
        label="ideal sigma_T",
    )
    ax.set_title("temperature distribution")
    ax.set_xlabel("T, K")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22)

    plot_hist_pair(
        axes[0, 2],
        predicted_v,
        reference_v,
        "velocity components",
        "v component, A/ps",
        args.bins,
        args.percentile,
    )
    axes[0, 2].legend(fontsize=8)

    plot_speed_pair(axes[1, 0], predicted_v, reference_v, "speed distribution", args.bins, args.percentile)
    axes[1, 0].legend(fontsize=8)
    plot_hist_pair(
        axes[1, 1],
        predicted_u,
        reference_u,
        "displacement components",
        "u component, A",
        args.bins,
        args.percentile,
    )
    plot_q_ratio(axes[1, 2], q_u, pred_u_power, ref_u_power, "q-mode <|u(q)|^2>")
    plot_q_ratio(axes[2, 0], q_v, pred_v_power, ref_v_power, "q-mode <|v(q)|^2>")

    axes[2, 1].axis("off")
    summary = [
        f"frame window: {arrays['frame_window_start']}:{arrays['frame_window_end']}",
        f"frames: ASE {temperature.size}, ref {arrays['reference_frames']}",
        f"reference window: {arrays['reference_window']}",
        f"T mean/target = {np.mean(temperature):.3g} / {target_temperature:.3g}",
        f"T sigma / ideal sigma = {np.std(temperature) / expected_temperature_std:.3g}",
        f"v std ratio = {rows[3]['ratio']:.4g}",
        f"u std ratio = {rows[7]['ratio']:.4g}",
        f"q-u RMS rel. error = {rows[9]['score']:.4g}",
        f"q-v RMS rel. error = {rows[10]['score']:.4g}",
    ]
    axes[2, 1].text(0.02, 0.98, "\n".join(summary), va="top", ha="left", family="monospace", fontsize=11)
    axes[2, 1].set_title("summary")

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.98,
        "Notes:\n"
        "Velocity and displacement histograms use\n"
        "the selected inference window and selected\n"
        "reference window.\n"
        "Temperature uses the full ASE trajectory.\n"
        "q-mode panels compare shell-averaged powers.",
        va="top",
        ha="left",
        fontsize=10,
    )
    axes[2, 2].set_title("scope")

    fig.suptitle(args.title, fontsize=16)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")

    if args.metrics_path:
        write_metrics(Path(args.metrics_path), rows)
        print(f"Saved {args.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
