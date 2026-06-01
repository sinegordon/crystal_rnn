#!/usr/bin/env python3
"""Submit, monitor, and fetch the ASE NVT 1055 inference run from the cluster."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parent

DEFAULT_HOST = "sinegordon@cluster.vstu.ru"
DEFAULT_PORT = "57322"
DEFAULT_KEY = "~/.ssh/id_ed25519_cluster_vstu"
DEFAULT_REMOTE_WORKDIR = "~/crystal_rnn_accnorm"
DEFAULT_NODELIST = "node54.cluster"
DEFAULT_PARTITION = "gold-batch"
DEFAULT_CONDA_ENV = "torch"
DEFAULT_MODEL = (
    "models333_field_rnn_accnorm_cl2_d30k_ep100/"
    "mean_norm_0.9217232867679755_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth"
)
DEFAULT_DATA = "data1055.npz"


class CommandError(RuntimeError):
    """Raised when an external command fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Run ASE FieldRNN NVT inference on data1055.npz at the cluster, wait for SLURM, "
            "fetch plots/metrics, and print the final summary."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH destination, e.g. user@host.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="SSH port.")
    parser.add_argument("--identity-file", default=DEFAULT_KEY, help="SSH private key path.")
    parser.add_argument("--remote-workdir", default=DEFAULT_REMOTE_WORKDIR, help="Remote repository/work dir.")
    parser.add_argument("--partition", default=DEFAULT_PARTITION, help="SLURM partition.")
    parser.add_argument("--nodelist", default=DEFAULT_NODELIST, help="SLURM node constraint. Empty disables it.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Remote conda environment.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="Remote model path, relative to remote workdir.")
    parser.add_argument("--data-path", default=DEFAULT_DATA, help="Remote dataset path, relative to remote workdir.")
    parser.add_argument("--label", default=f"ase1055_{timestamp}", help="Run label used in the output path.")
    parser.add_argument("--output-root", default=None, help="Remote output directory. Defaults to label-based path.")
    parser.add_argument("--local-output-root", default=None, help="Local fetch directory. Defaults to output-root.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of ASE MD steps.")
    parser.add_argument("--dt-ps", type=float, default=0.002, help="MD/model time step in ps.")
    parser.add_argument("--taut-fs", type=float, default=20.0, help="Bussi thermostat coupling time in fs.")
    parser.add_argument("--temperature-k", type=float, default=300.0, help="Target temperature in K.")
    parser.add_argument("--patch-batch-size", type=int, default=250, help="Patch batch size for GPU inference.")
    parser.add_argument("--device", default="cuda", help="Inference device passed to the ASE script.")
    parser.add_argument("--ncells", type=int, default=10, help="S(q,w) ncells value.")
    parser.add_argument("--kcount", type=int, default=10, help="S(q,w) kcount value.")
    parser.add_argument("--sqw-step", type=int, default=10, help="Frame stride used for S(q,w).")
    parser.add_argument("--velocity-window", type=int, default=10, help="Velocity histogram window size.")
    parser.add_argument("--velocity-bins", type=int, default=120, help="Velocity histogram bin count.")
    parser.add_argument("--phase-window-frames", type=int, default=1000, help="Frames per phase diagnostics window.")
    parser.add_argument("--phase-bins", type=int, default=120, help="Phase diagnostics histogram bin count.")
    parser.add_argument("--sound-frame-stride", type=int, default=1, help="Frame stride for OX sound-speed FFT.")
    parser.add_argument("--sound-fit-q-count", type=int, default=3, help="Low-q peak count used for sound-speed fit.")
    parser.add_argument("--sound-q-index-min", type=int, default=1, help="Smallest q harmonic for sound-speed analysis.")
    parser.add_argument("--sound-q-index-max", type=int, default=None, help="Largest q harmonic for sound-speed analysis.")
    parser.add_argument("--sound-min-energy-mev", type=float, default=0.25, help="Lower peak-picking energy cutoff.")
    parser.add_argument("--sound-max-energy-mev", type=float, default=45.0, help="Upper peak-picking energy cutoff.")
    parser.add_argument("--sound-smooth-bins", type=int, default=5, help="Moving-average width for sound spectra.")
    parser.add_argument("--path-energy-slabs", type=int, default=10, help="Number of OX slabs for path-energy diagnostics.")
    parser.add_argument("--path-energy-plot-stride", type=int, default=1, help="Frame stride for path-energy plots.")
    parser.add_argument("--path-energy-fit-fraction-start", type=float, default=0.1, help="Energy-drift fit start fraction.")
    parser.add_argument("--path-energy-fit-fraction-end", type=float, default=1.0, help="Energy-drift fit end fraction.")
    parser.add_argument("--heat-capacity-frame-fraction-start", type=float, default=0.0)
    parser.add_argument("--heat-capacity-frame-fraction-end", type=float, default=1.0)
    parser.add_argument("--heat-capacity-detrend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heat-capacity-block-sizes", type=int, nargs="*", default=[250, 500, 1000, 2500, 5000])
    parser.add_argument(
        "--q-zero-mode",
        choices=["none", "initial", "constant_velocity", "zero"],
        default="none",
        help="Spatial q=0 correction mode passed to run_ase_copper_nvt.py.",
    )
    parser.add_argument(
        "--low-q-correction-mode",
        choices=["none", "reference"],
        default="none",
        help="Optional Fourier-space low-q correction mode passed to the ASE calculator.",
    )
    parser.add_argument(
        "--low-q-correction-max-q",
        type=float,
        default=0.0,
        help="Largest |q| in 1/A corrected by the low-q mode.",
    )
    parser.add_argument(
        "--low-q-correction-blend",
        type=float,
        default=1.0,
        help="Blend from model acceleration to elastic low-q acceleration.",
    )
    parser.add_argument(
        "--low-q-correction-exclude-q-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude spatial q=0 from the low-q correction.",
    )
    parser.add_argument(
        "--low-q-reference-max-frames",
        type=int,
        default=None,
        help="Optional frame cap when estimating reference low-q stiffness on the cluster.",
    )
    parser.add_argument(
        "--low-q-stiffness-epsilon",
        type=float,
        default=1e-30,
        help="Numerical floor for reference low-q stiffness estimation.",
    )
    parser.add_argument(
        "--history-damping-mode",
        choices=["none", "local-positive"],
        default="none",
        help="Optional history-damping correction mode.",
    )
    parser.add_argument("--history-damping-eta", type=float, default=0.0, help="Initial history-damping eta.")
    parser.add_argument("--history-damping-interval", type=int, default=10)
    parser.add_argument("--history-damping-batch-size", type=int, default=32)
    parser.add_argument("--history-damping-adaptive-gain", type=float, default=0.0)
    parser.add_argument("--history-damping-adaptive-cooling-gain", type=float, default=None)
    parser.add_argument("--history-damping-adaptive-interval", type=int, default=100)
    parser.add_argument("--history-damping-adaptive-min-eta", type=float, default=0.0)
    parser.add_argument("--history-damping-adaptive-max-eta", type=float, default=None)
    parser.add_argument("--history-damping-adaptive-target-power", type=float, default=0.0)
    parser.add_argument("--history-damping-adaptive-ema", type=float, default=0.05)
    parser.add_argument(
        "--temperature-eta-adaptive-mode",
        choices=["none", "log"],
        default="none",
        help="Adapt history-damping eta from smoothed ASE temperature.",
    )
    parser.add_argument("--temperature-eta-adaptive-gain", type=float, default=0.0)
    parser.add_argument("--temperature-eta-adaptive-interval", type=int, default=100)
    parser.add_argument("--temperature-eta-adaptive-ema", type=float, default=0.002)
    parser.add_argument("--temperature-eta-adaptive-min-eta", type=float, default=0.0)
    parser.add_argument("--temperature-eta-adaptive-max-eta", type=float, default=None)
    parser.add_argument("--temperature-eta-adaptive-deadband", type=float, default=0.0)
    parser.add_argument(
        "--power-bias-correction-mode",
        choices=["none", "global", "global-positive"],
        default="none",
    )
    parser.add_argument("--power-bias-correction-alpha", type=float, default=1.0)
    parser.add_argument("--power-bias-correction-epsilon", type=float, default=1e-30)
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Polling interval in seconds.")
    parser.add_argument("--tail-lines", type=int, default=12, help="Tail lines printed from remote logs while waiting.")
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not sync local scripts/base_classes before submitting.",
    )
    parser.add_argument(
        "--install-ase-if-missing",
        action="store_true",
        help="Install ASE into the remote conda environment if the import check fails.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--open-images", action="store_true", help="Open fetched PNG plots on macOS after completion.")
    return parser.parse_args()


def expanded_key(path: str) -> str:
    """Return an expanded SSH identity path."""
    return str(Path(path).expanduser())


def remote_cd_path(path: str) -> str:
    """Quote a remote path while preserving tilde expansion."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def remote_join(workdir: str, relative_path: str) -> str:
    """Join a remote workdir and relative path for rsync remote specs."""
    return workdir.rstrip("/") + "/" + relative_path.strip("/")


def command_text(command: list[str]) -> str:
    """Return a shell-like command string for logging."""
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a local command and return the completed process."""
    print(f"$ {command_text(command)}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        message = [f"Command failed with exit code {proc.returncode}: {command_text(command)}"]
        if proc.stdout:
            message.append("--- stdout ---")
            message.append(proc.stdout.rstrip())
        if proc.stderr:
            message.append("--- stderr ---")
            message.append(proc.stderr.rstrip())
        raise CommandError("\n".join(message))
    return proc


def ssh_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    """Build an SSH command."""
    return [
        "ssh",
        "-i",
        expanded_key(args.identity_file),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        args.host,
        remote_command,
    ]


def ssh(args: argparse.Namespace, remote_command: str, *, check: bool = True) -> str:
    """Run a remote shell command over SSH."""
    proc = run_command(ssh_command(args, remote_command), dry_run=args.dry_run, check=check)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout.strip():
        print(stdout.rstrip(), flush=True)
    if stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    return stdout


def rsync_command(args: argparse.Namespace, sources: list[str], destination: str) -> list[str]:
    """Build an rsync command using the configured SSH transport."""
    ssh_transport = f"ssh -i {shlex.quote(expanded_key(args.identity_file))} -p {args.port} -o BatchMode=yes"
    return ["rsync", "-avz", "-e", ssh_transport, *sources, destination]


def rsync(args: argparse.Namespace, sources: list[str], destination: str) -> None:
    """Run rsync."""
    proc = run_command(rsync_command(args, sources, destination), dry_run=args.dry_run, check=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip(), flush=True)
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr, flush=True)


def ensure_remote_directory(args: argparse.Namespace, relative_path: str) -> None:
    """Create a remote directory relative to the remote workdir."""
    command = f"cd {remote_cd_path(args.remote_workdir)} && mkdir -p {shlex.quote(relative_path)}"
    ssh(args, command)


def sync_required_files(args: argparse.Namespace) -> None:
    """Sync local code needed by the remote SLURM script."""
    if args.no_sync:
        print("Skipping code sync (--no-sync).", flush=True)
        return

    required_paths = [
        LOCAL_ROOT / "base_classes",
        LOCAL_ROOT / "ase_copper_calculator.py",
        LOCAL_ROOT / "run_ase_copper_nvt.py",
        LOCAL_ROOT / "plot_sqw_comparison.py",
        LOCAL_ROOT / "plot_ase_velocity_histograms.py",
        LOCAL_ROOT / "plot_ase_temperature_trace.py",
        LOCAL_ROOT / "plot_ase_phase_histograms.py",
        LOCAL_ROOT / "plot_ase_canonical_checks.py",
        LOCAL_ROOT / "plot_ase_sound_speed_ox.py",
        LOCAL_ROOT / "plot_ase_path_energy.py",
        LOCAL_ROOT / "plot_ase_total_energy.py",
        LOCAL_ROOT / "plot_ase_heat_capacity.py",
        LOCAL_ROOT / "plot_ase_etot_heat_capacity.py",
        LOCAL_ROOT / "plot_ase_etot_cumulative.py",
        LOCAL_ROOT / "cluster" / "run_ase_1055_nvt.slurm",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required local files:\n" + "\n".join(missing))

    ensure_remote_directory(args, "base_classes")
    ensure_remote_directory(args, "cluster")

    remote_root = f"{args.host}:{args.remote_workdir.rstrip('/')}/"
    remote_base = f"{args.host}:{remote_join(args.remote_workdir, 'base_classes')}/"
    remote_cluster = f"{args.host}:{remote_join(args.remote_workdir, 'cluster')}/"

    rsync(args, [str(LOCAL_ROOT / "base_classes") + "/"], remote_base)
    rsync(
        args,
        [
            str(LOCAL_ROOT / "ase_copper_calculator.py"),
            str(LOCAL_ROOT / "run_ase_copper_nvt.py"),
            str(LOCAL_ROOT / "plot_sqw_comparison.py"),
            str(LOCAL_ROOT / "plot_ase_velocity_histograms.py"),
            str(LOCAL_ROOT / "plot_ase_temperature_trace.py"),
            str(LOCAL_ROOT / "plot_ase_phase_histograms.py"),
            str(LOCAL_ROOT / "plot_ase_canonical_checks.py"),
            str(LOCAL_ROOT / "plot_ase_sound_speed_ox.py"),
            str(LOCAL_ROOT / "plot_ase_path_energy.py"),
            str(LOCAL_ROOT / "plot_ase_total_energy.py"),
            str(LOCAL_ROOT / "plot_ase_heat_capacity.py"),
            str(LOCAL_ROOT / "plot_ase_etot_heat_capacity.py"),
            str(LOCAL_ROOT / "plot_ase_etot_cumulative.py"),
        ],
        remote_root,
    )
    rsync(args, [str(LOCAL_ROOT / "cluster" / "run_ase_1055_nvt.slurm")], remote_cluster)


def check_remote_ase(args: argparse.Namespace) -> None:
    """Check that ASE is importable in the remote conda environment."""
    command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"conda run -n {shlex.quote(args.conda_env)} "
        "python -c 'import ase; import torch; print(\"ase ok\", ase.__version__); print(\"torch ok\", torch.__version__)'"
    )
    proc = run_command(ssh_command(args, command), dry_run=args.dry_run, check=False)
    if args.dry_run:
        return
    if proc.returncode == 0:
        print(proc.stdout.rstrip(), flush=True)
        return
    print(proc.stdout.rstrip(), flush=True)
    print(proc.stderr.rstrip(), file=sys.stderr, flush=True)
    if not args.install_ase_if_missing:
        raise CommandError(
            "ASE import failed on the cluster. Re-run with --install-ase-if-missing "
            "or install ASE manually in the remote conda environment."
        )
    install = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"conda run -n {shlex.quote(args.conda_env)} python -m pip install ase"
    )
    ssh(args, install)


def output_root(args: argparse.Namespace) -> str:
    """Return the remote output root."""
    if args.output_root:
        return args.output_root.strip("/")
    return f"inference_outputs/ase_nvt_1055/{args.label}".strip("/")


def local_output_dir(args: argparse.Namespace, remote_output_root: str) -> Path:
    """Return the local result directory."""
    if args.local_output_root:
        path = Path(args.local_output_root)
        return path if path.is_absolute() else LOCAL_ROOT / path
    return LOCAL_ROOT / remote_output_root


def submit_job(args: argparse.Namespace, remote_output_root: str) -> str:
    """Submit the SLURM job and return its id."""
    env = {
        "CONDA_ENV": args.conda_env,
        "MODEL_PATH": args.model_path,
        "DATA_PATH": args.data_path,
        "OUTPUT_ROOT": remote_output_root,
        "STEPS": str(args.steps),
        "DT_PS": str(args.dt_ps),
        "TAUT_FS": str(args.taut_fs),
        "TEMPERATURE_K": str(args.temperature_k),
        "PATCH_BATCH_SIZE": str(args.patch_batch_size),
        "DEVICE": args.device,
        "NCELLS": str(args.ncells),
        "KCOUNT": str(args.kcount),
        "SQW_STEP": str(args.sqw_step),
        "VELOCITY_WINDOW": str(args.velocity_window),
        "VELOCITY_BINS": str(args.velocity_bins),
        "PHASE_WINDOW_FRAMES": str(args.phase_window_frames),
        "PHASE_BINS": str(args.phase_bins),
        "SOUND_FRAME_STRIDE": str(args.sound_frame_stride),
        "SOUND_FIT_Q_COUNT": str(args.sound_fit_q_count),
        "SOUND_Q_INDEX_MIN": str(args.sound_q_index_min),
        "SOUND_MIN_ENERGY_MEV": str(args.sound_min_energy_mev),
        "SOUND_MAX_ENERGY_MEV": str(args.sound_max_energy_mev),
        "SOUND_SMOOTH_BINS": str(args.sound_smooth_bins),
        "PATH_ENERGY_SLABS": str(args.path_energy_slabs),
        "PATH_ENERGY_PLOT_STRIDE": str(args.path_energy_plot_stride),
        "PATH_ENERGY_FIT_FRACTION_START": str(args.path_energy_fit_fraction_start),
        "PATH_ENERGY_FIT_FRACTION_END": str(args.path_energy_fit_fraction_end),
        "HEAT_CAPACITY_FRAME_FRACTION_START": str(args.heat_capacity_frame_fraction_start),
        "HEAT_CAPACITY_FRAME_FRACTION_END": str(args.heat_capacity_frame_fraction_end),
        "HEAT_CAPACITY_DETREND": "1" if args.heat_capacity_detrend else "0",
        "HEAT_CAPACITY_BLOCK_SIZES": " ".join(str(value) for value in args.heat_capacity_block_sizes),
        "Q_ZERO_MODE": args.q_zero_mode,
        "LOW_Q_CORRECTION_MODE": args.low_q_correction_mode,
        "LOW_Q_CORRECTION_MAX_Q": str(args.low_q_correction_max_q),
        "LOW_Q_CORRECTION_BLEND": str(args.low_q_correction_blend),
        "LOW_Q_CORRECTION_EXCLUDE_Q_ZERO": "1" if args.low_q_correction_exclude_q_zero else "0",
        "LOW_Q_STIFFNESS_EPSILON": str(args.low_q_stiffness_epsilon),
        "HISTORY_DAMPING_MODE": args.history_damping_mode,
        "HISTORY_DAMPING_ETA": str(args.history_damping_eta),
        "HISTORY_DAMPING_INTERVAL": str(args.history_damping_interval),
        "HISTORY_DAMPING_BATCH_SIZE": str(args.history_damping_batch_size),
        "HISTORY_DAMPING_ADAPTIVE_GAIN": str(args.history_damping_adaptive_gain),
        "HISTORY_DAMPING_ADAPTIVE_INTERVAL": str(args.history_damping_adaptive_interval),
        "HISTORY_DAMPING_ADAPTIVE_MIN_ETA": str(args.history_damping_adaptive_min_eta),
        "HISTORY_DAMPING_ADAPTIVE_TARGET_POWER": str(args.history_damping_adaptive_target_power),
        "HISTORY_DAMPING_ADAPTIVE_EMA": str(args.history_damping_adaptive_ema),
        "TEMPERATURE_ETA_ADAPTIVE_MODE": args.temperature_eta_adaptive_mode,
        "TEMPERATURE_ETA_ADAPTIVE_GAIN": str(args.temperature_eta_adaptive_gain),
        "TEMPERATURE_ETA_ADAPTIVE_INTERVAL": str(args.temperature_eta_adaptive_interval),
        "TEMPERATURE_ETA_ADAPTIVE_EMA": str(args.temperature_eta_adaptive_ema),
        "TEMPERATURE_ETA_ADAPTIVE_MIN_ETA": str(args.temperature_eta_adaptive_min_eta),
        "TEMPERATURE_ETA_ADAPTIVE_DEADBAND": str(args.temperature_eta_adaptive_deadband),
        "POWER_BIAS_CORRECTION_MODE": args.power_bias_correction_mode,
        "POWER_BIAS_CORRECTION_ALPHA": str(args.power_bias_correction_alpha),
        "POWER_BIAS_CORRECTION_EPSILON": str(args.power_bias_correction_epsilon),
    }
    if args.low_q_reference_max_frames is not None:
        env["LOW_Q_REFERENCE_MAX_FRAMES"] = str(args.low_q_reference_max_frames)
    if args.history_damping_adaptive_cooling_gain is not None:
        env["HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN"] = str(args.history_damping_adaptive_cooling_gain)
    if args.history_damping_adaptive_max_eta is not None:
        env["HISTORY_DAMPING_ADAPTIVE_MAX_ETA"] = str(args.history_damping_adaptive_max_eta)
    if args.temperature_eta_adaptive_max_eta is not None:
        env["TEMPERATURE_ETA_ADAPTIVE_MAX_ETA"] = str(args.temperature_eta_adaptive_max_eta)
    if args.sound_q_index_max is not None:
        env["SOUND_Q_INDEX_MAX"] = str(args.sound_q_index_max)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    sbatch_options = [
        f"--partition={args.partition}",
    ]
    if args.nodelist:
        sbatch_options.append(f"--nodelist={args.nodelist}")
    sbatch = " ".join(shlex.quote(option) for option in sbatch_options)
    command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"{assignments} sbatch {sbatch} cluster/run_ase_1055_nvt.slurm"
    )
    stdout = ssh(args, command)
    if args.dry_run:
        print("Dry run: sbatch was not executed.", flush=True)
        return "DRYRUN"
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job id from sbatch output:\n{stdout}")
    job_id = match.group(1)
    print(f"Submitted job {job_id}", flush=True)
    return job_id


def poll_job(args: argparse.Namespace, job_id: str, remote_output_root: str) -> None:
    """Wait until the SLURM job disappears from the queue."""
    print(f"Waiting for job {job_id}. Poll interval: {args.poll_interval:g} s", flush=True)
    previous_status = None
    while True:
        queue_command = f"squeue -h -j {shlex.quote(job_id)} -o '%i|%T|%M|%R'"
        stdout = ssh(args, queue_command, check=False)
        status = stdout.strip()
        if not status:
            print(f"Job {job_id} is no longer in squeue.", flush=True)
            break
        if status != previous_status:
            print(f"Queue: {status}", flush=True)
            previous_status = status
        else:
            print(f"Queue: {status}", flush=True)

        if args.tail_lines > 0:
            tail_command = (
                f"cd {remote_cd_path(args.remote_workdir)} && "
                f"tail -n {int(args.tail_lines)} "
                f"logs/ase1055_nvt_{shlex.quote(job_id)}.out "
                f"{shlex.quote(remote_output_root)}/ase_nvt_1055.log 2>/dev/null || true"
            )
            ssh(args, tail_command, check=False)

        if args.dry_run:
            break
        time.sleep(args.poll_interval)


def fetch_results(args: argparse.Namespace, job_id: str, remote_output_root: str, local_dir: Path) -> None:
    """Fetch result files and SLURM logs."""
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_result = f"{args.host}:{remote_join(args.remote_workdir, remote_output_root)}/"
    rsync(args, [remote_result], str(local_dir) + "/")

    log_dir = local_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    remote_log_base = remote_join(args.remote_workdir, f"logs/ase1055_nvt_{job_id}")
    rsync(
        args,
        [
            f"{args.host}:{remote_log_base}.out",
            f"{args.host}:{remote_log_base}.err",
        ],
        str(log_dir) + "/",
    )


def read_text(path: Path) -> str:
    """Read text if a file exists."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def print_summary(local_dir: Path, job_id: str) -> None:
    """Print the fetched metrics in the same style used during manual checks."""
    sqw_text = read_text(local_dir / "ase_nvt_1055_sqw.txt")
    velocity_text = read_text(local_dir / "ase_nvt_1055_velocity_histograms.txt")
    temperature_text = read_text(local_dir / "ase_nvt_1055_temperature_trace.txt")
    phase_text = read_text(local_dir / "ase_nvt_1055_phase_power_acceleration.tsv")
    canonical_text = read_text(local_dir / "ase_nvt_1055_canonical_checks.tsv")
    sound_text = read_text(local_dir / "ase_nvt_1055_sound_speed_ox.tsv")
    path_energy_text = read_text(local_dir / "ase_nvt_1055_path_energy.tsv")
    etot_text = read_text(local_dir / "ase_nvt_1055_etot_trace.tsv")
    heat_capacity_text = read_text(local_dir / "ase_nvt_1055_heat_capacity.tsv")
    etot_heat_capacity_text = read_text(local_dir / "ase_nvt_1055_etot_heat_capacity.tsv")
    stdout_text = read_text(local_dir / "logs" / f"ase1055_nvt_{job_id}.out")
    stderr_text = read_text(local_dir / "logs" / f"ase1055_nvt_{job_id}.err")

    correlation = re.search(r"correlation\s*=\s*([0-9.eE+-]+)", sqw_text)
    temperature = re.search(r"temperature first/last/mean/min/max = ([^\n]+)", temperature_text)
    if temperature is None:
        temperature = re.search(r"temperature first/last/mean/min/max = ([^\n]+)", stdout_text)

    print("\n=== ASE 1055 cluster result ===")
    if correlation:
        print(f"S(q,w) correlation: {correlation.group(1)}")
    else:
        print("S(q,w) correlation: not found")

    if temperature:
        print(f"temperature first/last/mean/min/max: {temperature.group(1)}")
    else:
        print("temperature first/last/mean/min/max: not found")

    print("\nVelocity metrics:")
    if velocity_text.strip():
        print(velocity_text.strip())
    else:
        print("not found")

    if phase_text.strip():
        print("\nPhase power/acceleration metrics:")
        print("\n".join(phase_text.strip().splitlines()[:13]))

    if canonical_text.strip():
        print("\nCanonical-check metrics:")
        print("\n".join(canonical_text.strip().splitlines()[:10]))

    if sound_text.strip():
        print("\nOX sound-speed metrics:")
        print("\n".join(sound_text.strip().splitlines()[-3:]))

    if path_energy_text.strip():
        print("\nPath-energy metrics:")
        rows = path_energy_text.strip().splitlines()
        print(rows[0])
        for row in rows[1:4]:
            print(row)

    if etot_text.strip():
        print("\nDirect E_tot metrics:")
        rows = etot_text.strip().splitlines()
        print(rows[0])
        for row in rows[1:4]:
            print(row)

    if heat_capacity_text.strip():
        print("\nPath-energy heat-capacity diagnostic metrics:")
        for row in heat_capacity_text.strip().splitlines():
            if "cv_per_atom" in row or "classical_dulong_petit" in row:
                print(row)

    if etot_heat_capacity_text.strip():
        print("\nDirect E_tot heat-capacity diagnostic metrics:")
        for row in etot_heat_capacity_text.strip().splitlines():
            if "cv_per_atom" in row or "classical_dulong_petit" in row:
                print(row)

    if stderr_text.strip():
        print("\nSLURM stderr:")
        print(stderr_text.strip())

    sqw_plot = local_dir / "ase_nvt_1055_sqw.png"
    velocity_plot = local_dir / "ase_nvt_1055_velocity_histograms.png"
    temperature_plot = local_dir / "ase_nvt_1055_temperature_trace.png"
    phase_plot = local_dir / "ase_nvt_1055_phase_power_acceleration.png"
    canonical_plot = local_dir / "ase_nvt_1055_canonical_checks.png"
    sound_plot = local_dir / "ase_nvt_1055_sound_speed_ox.png"
    path_energy_plot = local_dir / "ase_nvt_1055_path_energy.png"
    etot_plot = local_dir / "ase_nvt_1055_etot_trace.png"
    heat_capacity_plot = local_dir / "ase_nvt_1055_heat_capacity.png"
    etot_heat_capacity_plot = local_dir / "ase_nvt_1055_etot_heat_capacity.png"
    print("\nFetched files:")
    for path in [
        local_dir / "ase_nvt_1055.npz",
        sqw_plot,
        velocity_plot,
        temperature_plot,
        phase_plot,
        canonical_plot,
        sound_plot,
        path_energy_plot,
        etot_plot,
        heat_capacity_plot,
        etot_heat_capacity_plot,
        local_dir / "ase_nvt_1055_sqw.txt",
        local_dir / "ase_nvt_1055_velocity_histograms.txt",
        local_dir / "ase_nvt_1055_temperature_trace.txt",
        local_dir / "ase_nvt_1055_phase_power_acceleration.tsv",
        local_dir / "ase_nvt_1055_canonical_checks.tsv",
        local_dir / "ase_nvt_1055_sound_speed_ox.tsv",
        local_dir / "ase_nvt_1055_path_energy.tsv",
        local_dir / "ase_nvt_1055_etot_trace.tsv",
        local_dir / "ase_nvt_1055_heat_capacity.tsv",
        local_dir / "ase_nvt_1055_etot_heat_capacity.tsv",
        local_dir / "logs" / f"ase1055_nvt_{job_id}.out",
        local_dir / "logs" / f"ase1055_nvt_{job_id}.err",
    ]:
        print(path)


def open_images(local_dir: Path) -> None:
    """Open fetched images on macOS."""
    for path in [
        local_dir / "ase_nvt_1055_sqw.png",
        local_dir / "ase_nvt_1055_velocity_histograms.png",
        local_dir / "ase_nvt_1055_temperature_trace.png",
        local_dir / "ase_nvt_1055_phase_power_acceleration.png",
        local_dir / "ase_nvt_1055_canonical_checks.png",
        local_dir / "ase_nvt_1055_sound_speed_ox.png",
        local_dir / "ase_nvt_1055_path_energy.png",
        local_dir / "ase_nvt_1055_etot_trace.png",
        local_dir / "ase_nvt_1055_heat_capacity.png",
        local_dir / "ase_nvt_1055_etot_heat_capacity.png",
    ]:
        if path.exists():
            subprocess.run(["open", str(path)], check=False)


def main() -> int:
    """Run the full local submit-wait-fetch workflow."""
    args = parse_args()
    remote_output_root = output_root(args)
    local_dir = local_output_dir(args, remote_output_root)

    print("ASE 1055 cluster workflow")
    print(f"remote: {args.host}:{args.remote_workdir}")
    print(f"node: {args.nodelist or '(partition default)'}")
    print(f"model: {args.model_path}")
    print(f"remote output: {remote_output_root}")
    print(f"local output: {local_dir}")

    sync_required_files(args)
    check_remote_ase(args)
    job_id = submit_job(args, remote_output_root)
    poll_job(args, job_id, remote_output_root)
    fetch_results(args, job_id, remote_output_root, local_dir)
    print_summary(local_dir, job_id)
    if args.open_images:
        open_images(local_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
