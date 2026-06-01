"""Estimate the longitudinal sound speed along the crystal OX axis.

The diagnostic uses the longitudinal current

    J_L(q, t) = sum_i (v_i(t) . q_hat) exp(-i q . r_i(t))

for wave vectors q = n b_x, finds the main low-energy spectral peak, and fits
E(q) = hbar c q for the first few q values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_sqw_comparison import crystal_frames_to_flat_positions


H_PLANCK_EV_S = 4.135667696e-15
HBAR_EV_S = 6.582119569e-16
MEV_ANGSTROM_PER_HBAR_TO_M_PER_S = 1.0e-3 * 1.0e-10 / HBAR_EV_S


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", required=True, help="ASE inference .npz file.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal reference .npz.")
    parser.add_argument("--output-path", required=True, help="Output PNG path.")
    parser.add_argument("--metrics-path", default=None, help="Optional TSV metrics path.")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x", help="Crystal axis used for q.")
    parser.add_argument("--q-index-min", type=int, default=1, help="Smallest non-zero q harmonic.")
    parser.add_argument(
        "--q-index-max",
        type=int,
        default=None,
        help="Largest q harmonic. Defaults to min(5, floor(N_axis / 2)).",
    )
    parser.add_argument("--fit-q-count", type=int, default=3, help="Number of lowest-q peaks used in the fit.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame before the temporal FFT.",
    )
    parser.add_argument(
        "--ase-frame-fraction-start",
        type=float,
        default=0.0,
        help="Start fraction of the ASE trajectory used for the spectrum.",
    )
    parser.add_argument(
        "--ase-frame-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of the ASE trajectory used for the spectrum.",
    )
    parser.add_argument(
        "--reference-frame-fraction-start",
        type=float,
        default=0.0,
        help="Start fraction of the reference trajectory used for the spectrum.",
    )
    parser.add_argument(
        "--reference-frame-fraction-end",
        type=float,
        default=1.0,
        help="End fraction of the reference trajectory used for the spectrum.",
    )
    parser.add_argument("--min-energy-mev", type=float, default=0.25, help="Ignore peaks below this energy.")
    parser.add_argument(
        "--max-energy-mev",
        type=float,
        default=45.0,
        help="Ignore peaks above this energy. Use a non-positive value to disable.",
    )
    parser.add_argument(
        "--smooth-bins",
        type=int,
        default=5,
        help="Odd moving-average width used before peak picking.",
    )
    parser.add_argument(
        "--frame-chunk",
        type=int,
        default=512,
        help="Frame chunk size used while building the longitudinal current.",
    )
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap name.")
    parser.add_argument("--title", default="Longitudinal sound speed along OX")
    return parser.parse_args()


def axis_index(axis: str) -> int:
    """Return the integer index for an axis name."""
    return {"x": 0, "y": 1, "z": 2}[axis]


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
        raise ValueError("cell must have shape (3,), (frames, 3), or (3, 3)")
    return cell


def load_cell(data: np.lib.npyio.NpzFile, ase: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return the simulation cell matrix."""
    if "cell" in ase.files:
        return as_cell_matrix(ase["cell"])
    if "cell" in data.files:
        return as_cell_matrix(data["cell"])
    if "box_lengths" in data.files:
        return as_cell_matrix(data["box_lengths"])
    raise ValueError("No cell, box_lengths, or ASE cell array found")


def reciprocal_axis_vector(cell: np.ndarray, axis: int, harmonic: int) -> np.ndarray:
    """Return the Cartesian wave vector for one reciprocal-axis harmonic."""
    inverse_cell = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    return 2.0 * np.pi * int(harmonic) * inverse_cell[:, axis]


def validate_fraction_window(start: float, end: float) -> None:
    """Validate a fractional frame window."""
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Frame fractions must satisfy 0 <= start < end <= 1")


def fraction_slice(frame_count: int, start: float, end: float) -> slice:
    """Convert fractional bounds into a frame slice."""
    validate_fraction_window(start, end)
    first = int(np.floor(frame_count * start))
    last = int(np.ceil(frame_count * end))
    first = min(max(first, 0), frame_count - 2)
    last = min(max(last, first + 2), frame_count)
    return slice(first, last)


def load_reference_positions_and_velocities(
    data: np.lib.npyio.NpzFile,
    start_frame: int,
    dt_ps: float,
    frame_fraction_start: float,
    frame_fraction_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat reference positions and central-difference velocities."""
    required = ["displacements", "reference_positions", "atom_order"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing reference arrays in data file: {missing}")

    displacements = np.asarray(data["displacements"][start_frame:], dtype=np.float32)
    frame_slice = fraction_slice(int(displacements.shape[0]), frame_fraction_start, frame_fraction_end)
    displacements = displacements[frame_slice]
    if displacements.shape[0] < 3:
        raise ValueError("Reference window must contain at least three frames")

    flat = crystal_frames_to_flat_positions(displacements, data["reference_positions"], data["atom_order"]).reshape(
        displacements.shape[0],
        -1,
        3,
    )
    velocities = (flat[2:] - flat[:-2]) / (2.0 * dt_ps)
    positions = flat[1:-1]
    return positions.astype(np.float64, copy=False), velocities.astype(np.float64, copy=False)


def load_ase_positions_and_velocities(
    ase: np.lib.npyio.NpzFile,
    frame_fraction_start: float,
    frame_fraction_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ASE positions and saved velocities for the selected frame window."""
    positions = np.asarray(ase["positions"], dtype=np.float64)
    velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)
    frame_count = min(int(positions.shape[0]), int(velocities.shape[0]))
    frame_slice = fraction_slice(frame_count, frame_fraction_start, frame_fraction_end)
    positions = positions[frame_slice]
    velocities = velocities[frame_slice]
    if positions.shape[0] < 3:
        raise ValueError("ASE window must contain at least three frames")
    return positions, velocities


def apply_frame_stride(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt_ps: float,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downsample positions/velocities and return the effective time step."""
    if frame_stride <= 0:
        raise ValueError("frame-stride must be positive")
    positions = positions[::frame_stride]
    velocities = velocities[::frame_stride]
    if positions.shape[0] < 8:
        raise ValueError("Need at least eight frames after frame-stride")
    return positions, velocities, dt_ps * frame_stride


def q_vectors(cell: np.ndarray, axis: int, q_index_min: int, q_index_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Return q harmonics and Cartesian q vectors."""
    if q_index_min <= 0:
        raise ValueError("q-index-min must be positive")
    if q_index_max < q_index_min:
        raise ValueError("q-index-max must be >= q-index-min")
    q_indices = np.arange(q_index_min, q_index_max + 1, dtype=np.int64)
    vectors = np.asarray([reciprocal_axis_vector(cell, axis, value) for value in q_indices], dtype=np.float64)
    return q_indices, vectors


def longitudinal_current(
    positions: np.ndarray,
    velocities: np.ndarray,
    q_vectors_cart: np.ndarray,
    frame_chunk: int,
) -> np.ndarray:
    """Compute J_L(q, t) for all requested q vectors."""
    if frame_chunk <= 0:
        raise ValueError("frame-chunk must be positive")
    frame_count = int(min(positions.shape[0], velocities.shape[0]))
    q_count = int(q_vectors_cart.shape[0])
    current = np.empty((frame_count, q_count), dtype=np.complex128)
    for start in range(0, frame_count, frame_chunk):
        end = min(start + frame_chunk, frame_count)
        pos = positions[start:end]
        vel = velocities[start:end]
        for q_index, q_vec in enumerate(q_vectors_cart):
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm <= 0:
                raise ValueError("q vector norm must be positive")
            q_unit = q_vec / q_norm
            phase = np.exp(-1j * np.tensordot(pos, q_vec, axes=([2], [0])))
            longitudinal_velocity = np.tensordot(vel, q_unit, axes=([2], [0]))
            current[start:end, q_index] = np.sum(longitudinal_velocity * phase, axis=1)
    current -= np.mean(current, axis=0, keepdims=True)
    return current


def smooth_spectrum(spectrum: np.ndarray, smooth_bins: int) -> np.ndarray:
    """Smooth spectra along the energy axis with a small moving average."""
    if smooth_bins <= 1:
        return spectrum
    width = int(smooth_bins)
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float64) / width
    smoothed = np.empty_like(spectrum)
    for column in range(spectrum.shape[1]):
        smoothed[:, column] = np.convolve(spectrum[:, column], kernel, mode="same")
    return smoothed


def current_spectrum(
    current: np.ndarray,
    dt_ps: float,
    smooth_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-energy longitudinal-current spectra."""
    frame_count = int(current.shape[0])
    window = np.hanning(frame_count)[:, None]
    weighted = current * window
    fft_values = np.fft.fft(weighted, axis=0)
    frequencies_hz = np.fft.fftfreq(frame_count, d=dt_ps * 1.0e-12)
    mask = frequencies_hz > 0
    energies_mev = frequencies_hz[mask] * H_PLANCK_EV_S * 1000.0
    spectrum = np.abs(fft_values[mask]) ** 2
    maxima = np.max(spectrum, axis=0, keepdims=True)
    spectrum = spectrum / np.maximum(maxima, 1.0e-300)
    spectrum = smooth_spectrum(spectrum, smooth_bins)
    return energies_mev.astype(np.float64, copy=False), spectrum.astype(np.float64, copy=False)


def quadratic_peak(x_values: np.ndarray, y_values: np.ndarray, index: int) -> tuple[float, float]:
    """Refine a peak position with a three-point quadratic fit."""
    if index <= 0 or index >= len(x_values) - 1:
        return float(x_values[index]), float(y_values[index])
    xs = np.asarray(x_values[index - 1 : index + 2], dtype=np.float64)
    ys = np.asarray(y_values[index - 1 : index + 2], dtype=np.float64)
    try:
        a, b, c = np.polyfit(xs, ys, 2)
    except np.linalg.LinAlgError:
        return float(x_values[index]), float(y_values[index])
    if a >= 0:
        return float(x_values[index]), float(y_values[index])
    peak_x = -b / (2.0 * a)
    if peak_x < xs[0] or peak_x > xs[-1]:
        return float(x_values[index]), float(y_values[index])
    peak_y = a * peak_x**2 + b * peak_x + c
    return float(peak_x), float(peak_y)


def pick_peaks(
    energies_mev: np.ndarray,
    spectrum: np.ndarray,
    min_energy_mev: float,
    max_energy_mev: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the strongest spectral peak inside the requested acoustic window."""
    if min_energy_mev < 0:
        raise ValueError("min-energy-mev must be non-negative")
    mask = energies_mev >= float(min_energy_mev)
    if max_energy_mev is not None:
        mask &= energies_mev <= float(max_energy_mev)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Peak-picking energy window contains fewer than three bins")

    energies_window = energies_mev[mask]
    spectrum_window = spectrum[mask]
    peak_energies = np.empty(spectrum.shape[1], dtype=np.float64)
    peak_intensities = np.empty(spectrum.shape[1], dtype=np.float64)
    for column in range(spectrum.shape[1]):
        index = int(np.argmax(spectrum_window[:, column]))
        peak_energy, peak_intensity = quadratic_peak(energies_window, spectrum_window[:, column], index)
        peak_energies[column] = peak_energy
        peak_intensities[column] = peak_intensity
    return peak_energies, peak_intensities


def fit_sound_speed(q_abs: np.ndarray, peak_energies: np.ndarray, fit_q_count: int) -> dict[str, float]:
    """Fit the low-q acoustic slope and return speed metrics."""
    if fit_q_count <= 0:
        raise ValueError("fit-q-count must be positive")
    finite = np.isfinite(q_abs) & np.isfinite(peak_energies)
    q = q_abs[finite][:fit_q_count]
    energy = peak_energies[finite][:fit_q_count]
    if q.size < 2:
        raise ValueError("Need at least two finite peak points for sound-speed fit")
    slope_origin = float(np.dot(q, energy) / np.dot(q, q))
    if q.size >= 2:
        slope, intercept = np.polyfit(q, energy, 1)
        fitted = slope * q + intercept
        residual = energy - fitted
        total = energy - np.mean(energy)
        r2 = 1.0 - float(np.sum(residual**2) / max(np.sum(total**2), 1.0e-300))
    else:
        slope = slope_origin
        intercept = 0.0
        r2 = float("nan")
    return {
        "fit_q_count": int(q.size),
        "slope_origin_mev_ang": slope_origin,
        "speed_origin_m_per_s": slope_origin * MEV_ANGSTROM_PER_HBAR_TO_M_PER_S,
        "slope_intercept_mev_ang": float(slope),
        "intercept_mev": float(intercept),
        "speed_intercept_m_per_s": float(slope) * MEV_ANGSTROM_PER_HBAR_TO_M_PER_S,
        "r2": float(r2),
    }


def analyze_source(
    label: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    dt_ps: float,
    q_vectors_cart: np.ndarray,
    frame_chunk: int,
    smooth_bins: int,
    min_energy_mev: float,
    max_energy_mev: float | None,
    fit_q_count: int,
) -> dict[str, object]:
    """Compute spectra, peaks, and a low-q sound-speed fit."""
    current = longitudinal_current(positions, velocities, q_vectors_cart, frame_chunk)
    energies_mev, spectrum = current_spectrum(current, dt_ps, smooth_bins)
    peak_energies, peak_intensities = pick_peaks(energies_mev, spectrum, min_energy_mev, max_energy_mev)
    q_abs = np.linalg.norm(q_vectors_cart, axis=1)
    fit = fit_sound_speed(q_abs, peak_energies, fit_q_count)
    return {
        "label": label,
        "frame_count": int(current.shape[0]),
        "dt_ps": float(dt_ps),
        "energies_mev": energies_mev,
        "spectrum": spectrum,
        "peak_energies": peak_energies,
        "peak_intensities": peak_intensities,
        "fit": fit,
    }


def metrics_rows(
    q_indices: np.ndarray,
    q_abs: np.ndarray,
    analyses: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return TSV metrics rows."""
    rows: list[dict[str, object]] = []
    for analysis in analyses:
        label = str(analysis["label"])
        peak_energies = np.asarray(analysis["peak_energies"], dtype=np.float64)
        peak_intensities = np.asarray(analysis["peak_intensities"], dtype=np.float64)
        for index, q_value, energy, intensity in zip(q_indices, q_abs, peak_energies, peak_intensities):
            rows.append(
                {
                    "row_type": "peak",
                    "source": label,
                    "q_index": int(index),
                    "q_1_per_ang": float(q_value),
                    "peak_energy_mev": float(energy),
                    "peak_intensity": float(intensity),
                }
            )
        fit = dict(analysis["fit"])
        rows.append(
            {
                "row_type": "fit",
                "source": label,
                "frame_count": int(analysis["frame_count"]),
                "dt_ps": float(analysis["dt_ps"]),
                **fit,
            }
        )

    if len(analyses) >= 2:
        first = dict(analyses[0]["fit"])
        second = dict(analyses[1]["fit"])
        rows.append(
            {
                "row_type": "comparison",
                "source": f"{analyses[0]['label']}/{analyses[1]['label']}",
                "speed_origin_ratio": first["speed_origin_m_per_s"] / second["speed_origin_m_per_s"],
                "speed_intercept_ratio": first["speed_intercept_m_per_s"] / second["speed_intercept_m_per_s"],
            }
        )
    return rows


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write metrics rows as TSV."""
    fields = [
        "row_type",
        "source",
        "q_index",
        "q_1_per_ang",
        "peak_energy_mev",
        "peak_intensity",
        "frame_count",
        "dt_ps",
        "fit_q_count",
        "slope_origin_mev_ang",
        "speed_origin_m_per_s",
        "slope_intercept_mev_ang",
        "intercept_mev",
        "speed_intercept_m_per_s",
        "r2",
        "speed_origin_ratio",
        "speed_intercept_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_spectrum_map(
    ax: plt.Axes,
    q_abs: np.ndarray,
    energies_mev: np.ndarray,
    spectrum: np.ndarray,
    peak_energies: np.ndarray,
    title: str,
    cmap: str,
) -> None:
    """Plot a q-energy spectral intensity map."""
    q_edges = q_centers_to_edges(q_abs)
    energy_edges = centers_to_edges(energies_mev)
    mesh = ax.pcolormesh(q_edges, energy_edges, spectrum, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.plot(q_abs, peak_energies, "wo", ms=4, mec="black", mew=0.7, label="picked peak")
    ax.set_title(title)
    ax.set_xlabel("q_x, 1/A")
    ax.set_ylabel("Energy, meV")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="upper right")
    plt.colorbar(mesh, ax=ax, label="normalized intensity")


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    """Return bin edges for monotonic bin centers."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        width = 1.0
        return np.array([values[0] - width / 2.0, values[0] + width / 2.0])
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def q_centers_to_edges(q_abs: np.ndarray) -> np.ndarray:
    """Return q bin edges for a small monotonic q grid."""
    return centers_to_edges(np.asarray(q_abs, dtype=np.float64))


def plot_dispersion(ax: plt.Axes, q_abs: np.ndarray, analyses: list[dict[str, object]]) -> None:
    """Plot picked dispersion points and fitted acoustic slopes."""
    colors = {"ASE": "tab:blue", "reference": "tab:orange"}
    q_line = np.linspace(0.0, float(np.max(q_abs)) * 1.05, 200)
    for analysis in analyses:
        label = str(analysis["label"])
        color = colors.get(label, None)
        peaks = np.asarray(analysis["peak_energies"], dtype=np.float64)
        fit = dict(analysis["fit"])
        ax.plot(q_abs, peaks, "o", color=color, label=f"{label} peaks")
        ax.plot(
            q_line,
            fit["slope_origin_mev_ang"] * q_line,
            "-",
            color=color,
            alpha=0.75,
            label=f"{label} fit: {fit['speed_origin_m_per_s']:.0f} m/s",
        )
    ax.set_title("Low-q longitudinal dispersion")
    ax.set_xlabel("q_x, 1/A")
    ax.set_ylabel("Peak energy, meV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")


def plot_summary(ax: plt.Axes, args: argparse.Namespace, analyses: list[dict[str, object]]) -> None:
    """Write a compact text summary."""
    ax.axis("off")
    lines = [
        f"axis: O{args.axis.upper()}",
        f"frame stride: {args.frame_stride}",
        f"peak window: {args.min_energy_mev:g}..{args.max_energy_mev:g} meV",
        f"fit q count: {args.fit_q_count}",
        "",
    ]
    for analysis in analyses:
        fit = dict(analysis["fit"])
        lines.extend(
            [
                f"{analysis['label']}:",
                f"  frames = {analysis['frame_count']}",
                f"  c_origin = {fit['speed_origin_m_per_s']:.1f} m/s",
                f"  c_intercept = {fit['speed_intercept_m_per_s']:.1f} m/s",
                f"  intercept = {fit['intercept_mev']:.3g} meV",
                f"  R2 = {fit['r2']:.4f}",
                "",
            ]
        )
    if len(analyses) >= 2:
        first = dict(analyses[0]["fit"])
        second = dict(analyses[1]["fit"])
        lines.append(f"speed ratio {analyses[0]['label']}/{analyses[1]['label']}:")
        lines.append(f"  origin fit = {first['speed_origin_m_per_s'] / second['speed_origin_m_per_s']:.4f}")
        lines.append(f"  intercept fit = {first['speed_intercept_m_per_s'] / second['speed_intercept_m_per_s']:.4f}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    ax.set_title("summary")


def plot_results(
    output_path: Path,
    q_abs: np.ndarray,
    analyses: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    """Save the sound-speed diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, analysis in zip(axes[0], analyses):
        plot_spectrum_map(
            ax,
            q_abs,
            np.asarray(analysis["energies_mev"], dtype=np.float64),
            np.asarray(analysis["spectrum"], dtype=np.float64),
            np.asarray(analysis["peak_energies"], dtype=np.float64),
            f"{analysis['label']} longitudinal spectrum",
            args.cmap,
        )
        if args.max_energy_mev and args.max_energy_mev > 0:
            ax.set_ylim(0.0, args.max_energy_mev)
    plot_dispersion(axes[1, 0], q_abs, analyses)
    if args.max_energy_mev and args.max_energy_mev > 0:
        axes[1, 0].set_ylim(0.0, args.max_energy_mev)
    plot_summary(axes[1, 1], args, analyses)
    fig.suptitle(args.title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Run the OX sound-speed diagnostic."""
    args = parse_args()
    ase = np.load(args.ase_path)
    data = np.load(args.data_path)
    dt_ps = float(np.asarray(ase["dt_ps"], dtype=np.float64)) if "dt_ps" in ase.files else 0.002
    start_frame = int(ase["initial_frames"][-1]) if "initial_frames" in ase.files else 0
    cell = load_cell(data, ase)
    axis = axis_index(args.axis)
    crystal_shape = np.asarray(data["crystal_shape"], dtype=np.int64) if "crystal_shape" in data.files else np.asarray(data["atom_order"].shape[:3], dtype=np.int64)
    default_q_max = min(5, int(crystal_shape[axis]) // 2)
    q_index_max = default_q_max if args.q_index_max is None else int(args.q_index_max)
    q_indices, q_vectors_cart = q_vectors(cell, axis, args.q_index_min, q_index_max)
    q_abs = np.linalg.norm(q_vectors_cart, axis=1)
    max_energy = None if args.max_energy_mev <= 0 else float(args.max_energy_mev)

    ase_positions, ase_velocities = load_ase_positions_and_velocities(
        ase,
        args.ase_frame_fraction_start,
        args.ase_frame_fraction_end,
    )
    ref_positions, ref_velocities = load_reference_positions_and_velocities(
        data,
        start_frame,
        dt_ps,
        args.reference_frame_fraction_start,
        args.reference_frame_fraction_end,
    )
    ase_positions, ase_velocities, ase_dt = apply_frame_stride(
        ase_positions,
        ase_velocities,
        dt_ps,
        args.frame_stride,
    )
    ref_positions, ref_velocities, ref_dt = apply_frame_stride(
        ref_positions,
        ref_velocities,
        dt_ps,
        args.frame_stride,
    )

    analyses = [
        analyze_source(
            "ASE",
            ase_positions,
            ase_velocities,
            ase_dt,
            q_vectors_cart,
            args.frame_chunk,
            args.smooth_bins,
            args.min_energy_mev,
            max_energy,
            args.fit_q_count,
        ),
        analyze_source(
            "reference",
            ref_positions,
            ref_velocities,
            ref_dt,
            q_vectors_cart,
            args.frame_chunk,
            args.smooth_bins,
            args.min_energy_mev,
            max_energy,
            args.fit_q_count,
        ),
    ]

    output_path = Path(args.output_path)
    plot_results(output_path, q_abs, analyses, args)
    print(f"Saved {output_path}")

    rows = metrics_rows(q_indices, q_abs, analyses)
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        write_metrics(metrics_path, rows)
        print(f"Saved {metrics_path}")
    for analysis in analyses:
        fit = dict(analysis["fit"])
        print(
            f"{analysis['label']} sound speed origin/intercept = "
            f"{fit['speed_origin_m_per_s']:.6g} / {fit['speed_intercept_m_per_s']:.6g} m/s"
        )
    if len(analyses) >= 2:
        first = dict(analyses[0]["fit"])
        second = dict(analyses[1]["fit"])
        print(
            "sound speed ratio ASE/reference = "
            f"{first['speed_origin_m_per_s'] / second['speed_origin_m_per_s']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
