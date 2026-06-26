"""Compare velocity autocorrelation functions for ASE inference and reference."""

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
    parser.add_argument("--max-lag-frames", type=int, default=2500, help="Maximum VACF lag in frames.")
    parser.add_argument("--fft-chunk-size", type=int, default=512, help="Flattened velocity columns per FFT chunk.")
    parser.add_argument(
        "--remove-frame-com",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract the instantaneous center-of-mass velocity before VACF calculation.",
    )
    parser.add_argument("--title", default="ASE NVT velocity autocorrelation")
    return parser.parse_args()


def load_reference_velocities(data: np.lib.npyio.NpzFile, start: int, dt_ps: float) -> np.ndarray:
    """Return reference velocities in flat atom order."""
    reference_displacements = np.asarray(data["displacements"][start:], dtype=np.float32)
    if reference_displacements.shape[0] < 2:
        raise ValueError("Need at least two reference frames for VACF.")
    reference_positions = crystal_frames_to_flat_positions(
        reference_displacements,
        data["reference_positions"],
        data["atom_order"],
    ).reshape(reference_displacements.shape[0], -1, 3)
    return np.diff(reference_positions, axis=0).astype(np.float64) / float(dt_ps)


def preprocess_velocities(velocities: np.ndarray, remove_frame_com: bool) -> np.ndarray:
    """Return velocities ready for autocorrelation."""
    values = np.asarray(velocities, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Velocities must have shape (frames, atoms, 3), got {values.shape}")
    if remove_frame_com:
        values = values - np.mean(values, axis=1, keepdims=True)
    return values.reshape(values.shape[0], -1)


def autocorrelation_fft(values: np.ndarray, max_lag: int, chunk_size: int) -> np.ndarray:
    """Return unbiased mean velocity autocorrelation over flattened velocity columns."""
    if values.ndim != 2:
        raise ValueError("values must be a 2D time-by-channel array")
    frame_count, channel_count = values.shape
    if frame_count < 2:
        raise ValueError("Need at least two velocity frames for VACF.")
    if chunk_size <= 0:
        raise ValueError("fft-chunk-size must be positive")
    max_lag = min(int(max_lag), frame_count - 1)
    fft_length = 1 << (2 * frame_count - 1).bit_length()
    sums = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, channel_count, chunk_size):
        chunk = np.ascontiguousarray(values[:, start : start + chunk_size])
        spectrum = np.fft.rfft(chunk, n=fft_length, axis=0)
        corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_length, axis=0)[: max_lag + 1]
        sums += np.sum(corr, axis=1)
    counts = (frame_count - np.arange(max_lag + 1, dtype=np.float64)) * channel_count
    return sums / counts


def normalized_vacf(velocities: np.ndarray, max_lag: int, chunk_size: int, remove_frame_com: bool):
    """Return lag indices, raw VACF, and normalized VACF."""
    prepared = preprocess_velocities(velocities, remove_frame_com)
    raw = autocorrelation_fft(prepared, max_lag=max_lag, chunk_size=chunk_size)
    norm = raw / raw[0] if raw[0] != 0 else np.full_like(raw, np.nan)
    return np.arange(raw.shape[0]), raw, norm


def first_zero_crossing(time_ps: np.ndarray, values: np.ndarray) -> float:
    """Return the first zero-crossing time, or NaN if absent."""
    for index in range(1, values.shape[0]):
        if values[index - 1] >= 0 and values[index] < 0:
            x0, x1 = time_ps[index - 1], time_ps[index]
            y0, y1 = values[index - 1], values[index]
            if y1 == y0:
                return float(x1)
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float("nan")


def summarize(source: str, time_ps: np.ndarray, raw: np.ndarray, normalized: np.ndarray) -> dict[str, float | str]:
    """Return compact VACF summary metrics."""
    finite = np.isfinite(normalized)
    integral = float(np.trapezoid(normalized[finite], time_ps[finite])) if np.any(finite) else float("nan")
    minimum_index = int(np.nanargmin(normalized)) if np.any(finite) else 0
    return {
        "source": source,
        "frames": int(time_ps.shape[0]),
        "c0": float(raw[0]),
        "normalized_integral_ps": integral,
        "first_zero_ps": first_zero_crossing(time_ps, normalized),
        "first_min_ps": float(time_ps[minimum_index]),
        "first_min_value": float(normalized[minimum_index]),
    }


def write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write TSV metrics."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build a VACF comparison plot and metrics."""
    args = parse_args()
    if args.max_lag_frames <= 0:
        raise ValueError("max-lag-frames must be positive")

    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    start = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0

    predicted_velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    reference_velocities = load_reference_velocities(data, start=start, dt_ps=dt_ps)

    pred_lag, pred_raw, pred_norm = normalized_vacf(
        predicted_velocities,
        max_lag=args.max_lag_frames,
        chunk_size=args.fft_chunk_size,
        remove_frame_com=args.remove_frame_com,
    )
    ref_lag, ref_raw, ref_norm = normalized_vacf(
        reference_velocities,
        max_lag=args.max_lag_frames,
        chunk_size=args.fft_chunk_size,
        remove_frame_com=args.remove_frame_com,
    )

    common = min(pred_norm.shape[0], ref_norm.shape[0])
    common_time = np.arange(common, dtype=np.float64) * dt_ps
    corr = float(np.corrcoef(pred_norm[:common], ref_norm[:common])[0, 1]) if common > 2 else float("nan")
    rms_diff = float(np.sqrt(np.mean((pred_norm[:common] - ref_norm[:common]) ** 2))) if common else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].plot(pred_lag * dt_ps, pred_norm, label="ASE inference", color="tab:orange")
    axes[0].plot(ref_lag * dt_ps, ref_norm, label="reference", color="tab:blue")
    axes[0].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[0].set_xlabel("lag, ps")
    axes[0].set_ylabel("normalized VACF")
    axes[0].set_title(f"VACF, corr={corr:.3f}, RMS diff={rms_diff:.3f}")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(pred_lag * dt_ps, pred_raw, label="ASE inference", color="tab:orange")
    axes[1].plot(ref_lag * dt_ps, ref_raw, label="reference", color="tab:blue")
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[1].set_xlabel("lag, ps")
    axes[1].set_ylabel("raw VACF, (A/ps)^2")
    axes[1].set_title("Raw velocity autocorrelation")
    axes[1].grid(alpha=0.2)

    fig.suptitle(args.title)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    rows = [
        {
            "source": "comparison",
            "frames": common,
            "c0": "",
            "normalized_integral_ps": "",
            "first_zero_ps": "",
            "first_min_ps": "",
            "first_min_value": "",
            "correlation": corr,
            "rms_diff": rms_diff,
            "dt_ps": dt_ps,
            "max_lag_frames": common - 1,
            "remove_frame_com": int(args.remove_frame_com),
        },
        {
            **summarize("ASE", pred_lag * dt_ps, pred_raw, pred_norm),
            "correlation": "",
            "rms_diff": "",
            "dt_ps": dt_ps,
            "max_lag_frames": pred_norm.shape[0] - 1,
            "remove_frame_com": int(args.remove_frame_com),
        },
        {
            **summarize("reference", ref_lag * dt_ps, ref_raw, ref_norm),
            "correlation": "",
            "rms_diff": "",
            "dt_ps": dt_ps,
            "max_lag_frames": ref_norm.shape[0] - 1,
            "remove_frame_com": int(args.remove_frame_com),
        },
    ]

    print(f"Saved {output_path}")
    print(f"VACF correlation = {corr:.6g}")
    print(f"VACF RMS diff = {rms_diff:.6g}")
    for row in rows[1:]:
        print(
            f"{row['source']} VACF integral/zero/min = "
            f"{row['normalized_integral_ps']:.6g} ps / "
            f"{row['first_zero_ps']:.6g} ps / "
            f"{row['first_min_value']:.6g} at {row['first_min_ps']:.6g} ps"
        )
    if args.metrics_path:
        write_metrics(Path(args.metrics_path), rows)


if __name__ == "__main__":
    main()
