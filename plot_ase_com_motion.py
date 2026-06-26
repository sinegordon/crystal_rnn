#!/usr/bin/env python3
"""Plot center-of-mass displacement and velocity for a saved ASE trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import format as npy_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-npz", required=True, help="ASE trajectory .npz produced by run_ase_copper_nvt.py.")
    parser.add_argument("--output-prefix", required=True, help="Output path prefix for .png, .tsv, and .txt files.")
    parser.add_argument("--chunk-frames", type=int, default=2048, help="Frames per streaming read chunk.")
    parser.add_argument(
        "--mass-weighted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use masses from the .npz file if present. For pure Cu this is equivalent to an arithmetic mean.",
    )
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return values.mean(axis=1)
    normalized = weights / np.sum(weights)
    return np.einsum("tac,a->tc", values, normalized)


def finite_difference(values: np.ndarray, time_ps: np.ndarray) -> np.ndarray:
    derivative = np.empty_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (time_ps[2:, None] - time_ps[:-2, None])
    derivative[0] = (values[1] - values[0]) / (time_ps[1] - time_ps[0])
    derivative[-1] = (values[-1] - values[-2]) / (time_ps[-1] - time_ps[-2])
    return derivative


def read_small_array(path: Path, name: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if name not in data.files:
            return None
        return np.asarray(data[name])


def read_npy_header(handle):
    version = npy_format.read_magic(handle)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(handle)
    if version == (2, 0):
        return npy_format.read_array_header_2_0(handle)
    raise ValueError(f"Unsupported .npy version {version}")


def npz_member_shape(path: Path, name: str) -> tuple[int, ...] | None:
    member = f"{name}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            return None
        with archive.open(member) as handle:
            shape, _fortran_order, _dtype = read_npy_header(handle)
    return tuple(int(dim) for dim in shape)


def stream_frame_means(path: Path, name: str, weights: np.ndarray | None, chunk_frames: int) -> np.ndarray | None:
    """Return per-frame atom means without loading the full trajectory array."""
    member = f"{name}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            return None
        with archive.open(member) as handle:
            shape, fortran_order, dtype = read_npy_header(handle)
            if fortran_order:
                raise ValueError(f"{name} is Fortran-ordered; streaming reader expects C-order arrays")
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"{name} must have shape (frames, atoms, 3), got {shape}")
            frames, atoms, components = (int(shape[0]), int(shape[1]), int(shape[2]))
            if weights is not None and weights.shape != (atoms,):
                raise ValueError("masses must have shape (atoms,)")
            normalized = None if weights is None else weights / np.sum(weights)
            result = np.empty((frames, components), dtype=np.float64)
            values_per_frame = atoms * components
            bytes_per_frame = values_per_frame * np.dtype(dtype).itemsize
            for start in range(0, frames, chunk_frames):
                count = min(chunk_frames, frames - start)
                raw = handle.read(count * bytes_per_frame)
                values = np.frombuffer(raw, dtype=dtype, count=count * values_per_frame)
                if values.size != count * values_per_frame:
                    raise ValueError(f"Unexpected end of {name} data stream")
                block = values.reshape(count, atoms, components).astype(np.float64, copy=False)
                if normalized is None:
                    result[start : start + count] = block.mean(axis=1)
                else:
                    result[start : start + count] = np.einsum("tac,a->tc", block, normalized)
    return result


def load_com_arrays(path: Path, mass_weighted: bool, chunk_frames: int):
    with np.load(path, allow_pickle=False) as data:
        position_shape = npz_member_shape(path, "positions")
        if position_shape is None:
            raise ValueError(f"{path} does not contain positions")
        steps = np.asarray(data["steps"], dtype=np.float64) if "steps" in data.files else np.arange(position_shape[0])
        dt_ps = float(np.asarray(data["dt_ps"]).reshape(())) if "dt_ps" in data.files else 1.0
        time_ps = steps * dt_ps
        weights = None
        if mass_weighted and "masses" in data.files:
            weights = np.asarray(data["masses"], dtype=np.float64)
            if weights.shape != (position_shape[1],):
                raise ValueError("masses must have shape (atoms,)")

    com = stream_frame_means(path, "positions", weights, chunk_frames)
    v_com = stream_frame_means(path, "velocities_ang_per_ps", weights, chunk_frames)
    if v_com is not None and v_com.shape != com.shape:
        raise ValueError("velocities_ang_per_ps shape does not match positions")
    return com, v_com, steps, time_ps


def save_table(
    output_tsv: Path,
    steps: np.ndarray,
    time_ps: np.ndarray,
    com: np.ndarray,
    com_shift: np.ndarray,
    v_com: np.ndarray,
    v_com_fd: np.ndarray,
) -> None:
    columns = np.column_stack(
        [
            steps,
            time_ps,
            com,
            com_shift,
            np.linalg.norm(com_shift, axis=1),
            v_com,
            np.linalg.norm(v_com, axis=1),
            v_com_fd,
            np.linalg.norm(v_com_fd, axis=1),
        ]
    )
    header = (
        "step\ttime_ps\tcom_x_A\tcom_y_A\tcom_z_A\t"
        "com_shift_x_A\tcom_shift_y_A\tcom_shift_z_A\tcom_shift_norm_A\t"
        "vcom_x_A_per_ps\tvcom_y_A_per_ps\tvcom_z_A_per_ps\tvcom_norm_A_per_ps\t"
        "fd_vcom_x_A_per_ps\tfd_vcom_y_A_per_ps\tfd_vcom_z_A_per_ps\tfd_vcom_norm_A_per_ps"
    )
    np.savetxt(output_tsv, columns, delimiter="\t", header=header, comments="")


def save_summary(output_txt: Path, time_ps: np.ndarray, com_shift: np.ndarray, v_com: np.ndarray, v_com_fd: np.ndarray):
    initial_v = v_com[0]
    final_v = v_com[-1]
    mean_v = np.mean(v_com, axis=0)
    final_shift = com_shift[-1]
    span = time_ps[-1] - time_ps[0]
    linear_from_initial = initial_v * span
    linear_from_mean = mean_v * span
    mean_fd = np.mean(v_com_fd, axis=0)
    lines = [
        f"frames = {len(time_ps)}",
        f"time_start_ps = {time_ps[0]:.12g}",
        f"time_end_ps = {time_ps[-1]:.12g}",
        "",
        "COM velocity from saved velocities [A/ps]",
        f"initial = {initial_v[0]:.12e} {initial_v[1]:.12e} {initial_v[2]:.12e}; norm = {np.linalg.norm(initial_v):.12e}",
        f"mean    = {mean_v[0]:.12e} {mean_v[1]:.12e} {mean_v[2]:.12e}; norm = {np.linalg.norm(mean_v):.12e}",
        f"final   = {final_v[0]:.12e} {final_v[1]:.12e} {final_v[2]:.12e}; norm = {np.linalg.norm(final_v):.12e}",
        "",
        "COM velocity from finite-difference positions [A/ps]",
        f"initial = {v_com_fd[0,0]:.12e} {v_com_fd[0,1]:.12e} {v_com_fd[0,2]:.12e}; norm = {np.linalg.norm(v_com_fd[0]):.12e}",
        f"mean    = {mean_fd[0]:.12e} {mean_fd[1]:.12e} {mean_fd[2]:.12e}; norm = {np.linalg.norm(mean_fd):.12e}",
        f"final   = {v_com_fd[-1,0]:.12e} {v_com_fd[-1,1]:.12e} {v_com_fd[-1,2]:.12e}; norm = {np.linalg.norm(v_com_fd[-1]):.12e}",
        "",
        "COM displacement relative to first saved frame [A]",
        f"final = {final_shift[0]:.12e} {final_shift[1]:.12e} {final_shift[2]:.12e}; norm = {np.linalg.norm(final_shift):.12e}",
        "",
        "Linear displacement estimates over the same time span [A]",
        f"from_initial_v = {linear_from_initial[0]:.12e} {linear_from_initial[1]:.12e} {linear_from_initial[2]:.12e}; norm = {np.linalg.norm(linear_from_initial):.12e}",
        f"from_mean_v    = {linear_from_mean[0]:.12e} {linear_from_mean[1]:.12e} {linear_from_mean[2]:.12e}; norm = {np.linalg.norm(linear_from_mean):.12e}",
        "",
        "Unit conversion",
        "1 A/ps = 100 m/s",
        f"initial_speed = {np.linalg.norm(initial_v) * 100.0:.12e} m/s",
    ]
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(output_png: Path, time_ps: np.ndarray, com_shift: np.ndarray, v_com: np.ndarray, v_com_fd: np.ndarray):
    labels = ("x", "y", "z")
    colors = ("tab:blue", "tab:orange", "tab:green")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    for axis, label, color in zip(range(3), labels, colors):
        ax.plot(time_ps, com_shift[:, axis], label=label, color=color, lw=1.2)
    ax.set_title("COM displacement components")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("R_COM(t) - R_COM(0), A")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(time_ps, np.linalg.norm(com_shift, axis=1), color="black", lw=1.2)
    ax.set_title("COM displacement norm")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("|R_COM(t) - R_COM(0)|, A")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for axis, label, color in zip(range(3), labels, colors):
        ax.plot(time_ps, v_com[:, axis], label=f"{label} saved", color=color, lw=1.0)
        ax.plot(time_ps, v_com_fd[:, axis], color=color, lw=0.7, ls="--", alpha=0.55)
    ax.set_title("COM velocity components")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("V_COM, A/ps")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(time_ps, np.linalg.norm(v_com, axis=1), color="tab:red", lw=1.0, label="saved velocities")
    ax.plot(time_ps, np.linalg.norm(v_com_fd, axis=1), color="black", lw=0.8, ls="--", alpha=0.6, label="finite difference")
    ax.set_title("COM speed")
    ax.set_xlabel("time, ps")
    ax.set_ylabel("|V_COM|, A/ps")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Center-of-mass motion diagnostic", fontsize=16)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    com, v_com, steps, time_ps = load_com_arrays(Path(args.ase_npz), args.mass_weighted, args.chunk_frames)
    com_shift = com - com[0]
    v_com_fd = finite_difference(com, time_ps)
    if v_com is None:
        v_com = v_com_fd

    save_table(output_prefix.with_suffix(".tsv"), steps, time_ps, com, com_shift, v_com, v_com_fd)
    save_summary(output_prefix.with_suffix(".txt"), time_ps, com_shift, v_com, v_com_fd)
    plot(output_prefix.with_suffix(".png"), time_ps, com_shift, v_com, v_com_fd)


if __name__ == "__main__":
    main()
