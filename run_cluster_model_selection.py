#!/usr/bin/env python3
"""Submit, monitor, and fetch a cluster FieldRNN model-selection sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from pipelines.shared.cluster.config import defaults_from_argv


LOCAL_ROOT = Path(__file__).resolve().parent


class CommandError(RuntimeError):
    """Raised when an external command fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_parent, cluster = defaults_from_argv()
    parser = argparse.ArgumentParser(
        description=(
            "Run cluster FieldRNN model selection through the existing Slurm array script, "
            "wait for array+collector completion, fetch results, and print top models."
        ),
        parents=[cluster_parent],
    )
    parser.add_argument("--host", default=cluster["host"], help="SSH destination, e.g. user@host.")
    parser.add_argument("--port", default=cluster["port"], help="SSH port.")
    parser.add_argument("--identity-file", default=cluster["identity_file"], help="SSH private key path.")
    parser.add_argument("--remote-workdir", default=cluster["remote_workdir"], help="Remote repository/work dir.")
    parser.add_argument("--conda-env", default=cluster["conda_env"], help="Remote conda environment.")
    parser.add_argument("--run-label", default=f"field_rnn_selection_{timestamp}", help="Run label.")
    parser.add_argument("--output-root", default=None, help="Remote output dir. Defaults to inference_outputs/RUN_LABEL.")
    parser.add_argument("--models-dir", default=None, help="Remote models dir. Defaults to models333_RUN_LABEL.")
    parser.add_argument("--local-output-root", default=None, help="Local result dir. Defaults to remote output-root.")
    parser.add_argument("--local-models-dir", default=None, help="Local models dir. Defaults to remote models-dir.")
    parser.add_argument("--data-path", default="data333.npz", help="Remote training data path.")
    parser.add_argument("--model-count", type=int, default=100, help="Number of independent models to train.")
    parser.add_argument("--base-seed", type=int, default=20260801, help="Base seed; task id is added to it.")
    parser.add_argument("--count-steps", type=int, default=2000, help="Autoregressive rollout length for scoring.")
    parser.add_argument("--count-run", type=int, default=1, help="Number of scoring rollouts per model.")
    parser.add_argument("--delta-frames", type=int, default=30000, help="Consecutive frames used for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--data-len", type=float, default=1.0, help="Fraction of sampled training data to use.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Training learning rate.")
    parser.add_argument("--encoder-channels", type=int, default=32)
    parser.add_argument("--rnn-hidden-size", type=int, default=64)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--conv-layers", type=int, default=2)
    parser.add_argument("--acceleration-normalization", default="channel")
    parser.add_argument("--loss-region", default="all")
    parser.add_argument("--input-transform", default="absolute")
    parser.add_argument("--input-relative-scale", type=float, default=1.0)
    parser.add_argument("--force-balance-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--acceleration-rms-loss-weight",
        type=float,
        default=0.0,
        help="Training penalty for predicted/reference physical acceleration RMS mismatch.",
    )
    parser.add_argument(
        "--velocity-rms-loss-weight",
        type=float,
        default=0.0,
        help="Training penalty for predicted/reference next-step velocity RMS mismatch.",
    )
    parser.add_argument(
        "--rms-loss-epsilon",
        type=float,
        default=1e-12,
        help="Numerical floor used by RMS-ratio training penalties.",
    )
    parser.add_argument(
        "--low-q-stiffness-loss-weight",
        type=float,
        default=0.0,
        help="Training penalty for low-q acceleration response per displacement amplitude.",
    )
    parser.add_argument(
        "--low-q-stiffness-max-shell",
        type=int,
        default=1,
        help="Largest integer Fourier shell used by the low-q stiffness penalty.",
    )
    parser.add_argument(
        "--low-q-stiffness-epsilon",
        type=float,
        default=1e-8,
        help="Numerical floor used by the low-q stiffness training penalty.",
    )
    parser.add_argument("--velocity-score-weight", type=float, default=1.0)
    parser.add_argument("--velocity-window-frames", type=int, default=10)
    parser.add_argument("--velocity-hist-bins", type=int, default=80)
    parser.add_argument("--velocity-max-end-speed-ratio", default="inf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-partition", default=cluster["train_partition"])
    parser.add_argument("--collect-partition", default=cluster["collect_partition"])
    parser.add_argument("--train-nodelist", default=cluster["train_nodelist"], help="Node for array jobs. Empty disables.")
    parser.add_argument("--collect-nodelist", default="", help="Optional node for collector job.")
    parser.add_argument("--train-time", default="06:00:00")
    parser.add_argument("--collect-time", default="00:20:00")
    parser.add_argument("--train-mem", default="16G")
    parser.add_argument("--collect-mem", default="4G")
    parser.add_argument("--train-gres", default="gpu:1")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=120.0, help="Queue polling interval in seconds.")
    parser.add_argument("--tail-lines", type=int, default=0, help="Tail collector/array logs while waiting.")
    parser.add_argument("--top", type=int, default=10, help="Number of best models to print.")
    parser.add_argument("--fetch-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fetch-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fetch-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-sync", action="store_true", help="Do not sync code before submitting.")
    parser.add_argument(
        "--install-ase-if-missing",
        action="store_true",
        help="Install ASE into the remote conda environment if it is missing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def expanded_key(path: str) -> str:
    """Return expanded SSH identity path."""
    return str(Path(path).expanduser())


def remote_cd_path(path: str) -> str:
    """Quote a remote path while preserving tilde expansion."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def remote_join(workdir: str, relative_path: str) -> str:
    """Join a remote workdir and a relative path."""
    return workdir.rstrip("/") + "/" + relative_path.strip("/")


def command_text(command: list[str]) -> str:
    """Return shell-like command text."""
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a local command."""
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
    """Build an rsync command."""
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
    ssh(args, f"cd {remote_cd_path(args.remote_workdir)} && mkdir -p {shlex.quote(relative_path)}")


def sync_required_files(args: argparse.Namespace) -> None:
    """Sync local code needed by the model-selection run."""
    if args.no_sync:
        print("Skipping code sync (--no-sync).", flush=True)
        return

    required_files = [
        "find_field_rnn_models.py",
        "plot_sqw_comparison.py",
        "cluster/run_field_rnn_velocity_selection_100.sh",
        "cluster/collect_field_rnn_metrics.py",
    ]
    required_paths = [LOCAL_ROOT / path for path in required_files] + [LOCAL_ROOT / "base_classes"]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required local files:\n" + "\n".join(missing))

    ensure_remote_directory(args, "base_classes")
    ensure_remote_directory(args, "cluster")

    remote_root = f"{args.host}:{args.remote_workdir.rstrip('/')}/"
    remote_base = f"{args.host}:{remote_join(args.remote_workdir, 'base_classes')}/"
    remote_cluster = f"{args.host}:{remote_join(args.remote_workdir, 'cluster')}/"

    rsync(args, [str(LOCAL_ROOT / "base_classes") + "/"], remote_base)
    rsync(args, [str(LOCAL_ROOT / "find_field_rnn_models.py"), str(LOCAL_ROOT / "plot_sqw_comparison.py")], remote_root)
    rsync(
        args,
        [
            str(LOCAL_ROOT / "cluster" / "run_field_rnn_velocity_selection_100.sh"),
            str(LOCAL_ROOT / "cluster" / "collect_field_rnn_metrics.py"),
        ],
        remote_cluster,
    )


def check_remote_environment(args: argparse.Namespace) -> None:
    """Check the remote conda environment has the needed imports."""
    command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"conda run -n {shlex.quote(args.conda_env)} "
        "python -c 'import torch, numpy, matplotlib; print(\"torch\", torch.__version__)'"
    )
    proc = run_command(ssh_command(args, command), dry_run=args.dry_run, check=False)
    if args.dry_run:
        return
    if proc.returncode != 0:
        message = "Remote Python environment check failed."
        if proc.stdout:
            message += "\n" + proc.stdout.rstrip()
        if proc.stderr:
            message += "\n" + proc.stderr.rstrip()
        raise CommandError(message)
    print(proc.stdout.rstrip(), flush=True)

    ase_command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"conda run -n {shlex.quote(args.conda_env)} python -c 'import ase; print(\"ase\", ase.__version__)'"
    )
    ase_proc = run_command(ssh_command(args, ase_command), dry_run=args.dry_run, check=False)
    if ase_proc.returncode == 0:
        print(ase_proc.stdout.rstrip(), flush=True)
        return
    if not args.install_ase_if_missing:
        print("ASE is not importable, but it is not required for training/model selection.", flush=True)
        return
    install = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"conda run -n {shlex.quote(args.conda_env)} python -m pip install ase"
    )
    ssh(args, install)


def output_root(args: argparse.Namespace) -> str:
    """Return remote output root."""
    return (args.output_root or f"inference_outputs/{args.run_label}").strip("/")


def models_dir(args: argparse.Namespace) -> str:
    """Return remote models dir."""
    return (args.models_dir or f"models333_{args.run_label}").strip("/")


def local_output_dir(args: argparse.Namespace, remote_output_root: str) -> Path:
    """Return local output directory."""
    if args.local_output_root:
        path = Path(args.local_output_root)
        return path if path.is_absolute() else LOCAL_ROOT / path
    return LOCAL_ROOT / remote_output_root


def local_models_dir(args: argparse.Namespace, remote_models_dir: str) -> Path:
    """Return local models directory."""
    if args.local_models_dir:
        path = Path(args.local_models_dir)
        return path if path.is_absolute() else LOCAL_ROOT / path
    return LOCAL_ROOT / remote_models_dir


def env_assignments(args: argparse.Namespace, remote_output_root: str, remote_models_dir: str) -> dict[str, str]:
    """Build environment variables consumed by the remote shell script."""
    return {
        "CONDA_ENV": args.conda_env,
        "RUN_LABEL": args.run_label,
        "OUTPUT_ROOT": remote_output_root,
        "MODELS_DIR": remote_models_dir,
        "MODEL_COUNT": str(args.model_count),
        "BASE_SEED": str(args.base_seed),
        "DATA_PATH": args.data_path,
        "COUNT_STEPS": str(args.count_steps),
        "COUNT_RUN": str(args.count_run),
        "DELTA_FRAMES": str(args.delta_frames),
        "EPOCHS": str(args.epochs),
        "BATCH_SIZE": str(args.batch_size),
        "DATA_LEN": str(args.data_len),
        "LEARNING_RATE": str(args.learning_rate),
        "ENCODER_CHANNELS": str(args.encoder_channels),
        "RNN_HIDDEN_SIZE": str(args.rnn_hidden_size),
        "RNN_LAYERS": str(args.rnn_layers),
        "CONV_LAYERS": str(args.conv_layers),
        "ACCELERATION_NORMALIZATION": args.acceleration_normalization,
        "LOSS_REGION": args.loss_region,
        "INPUT_TRANSFORM": args.input_transform,
        "INPUT_RELATIVE_SCALE": str(args.input_relative_scale),
        "FORCE_BALANCE_LOSS_WEIGHT": str(args.force_balance_loss_weight),
        "ACCELERATION_RMS_LOSS_WEIGHT": str(args.acceleration_rms_loss_weight),
        "VELOCITY_RMS_LOSS_WEIGHT": str(args.velocity_rms_loss_weight),
        "RMS_LOSS_EPSILON": str(args.rms_loss_epsilon),
        "LOW_Q_STIFFNESS_LOSS_WEIGHT": str(args.low_q_stiffness_loss_weight),
        "LOW_Q_STIFFNESS_MAX_SHELL": str(args.low_q_stiffness_max_shell),
        "LOW_Q_STIFFNESS_EPSILON": str(args.low_q_stiffness_epsilon),
        "VELOCITY_SCORE_WEIGHT": str(args.velocity_score_weight),
        "VELOCITY_WINDOW_FRAMES": str(args.velocity_window_frames),
        "VELOCITY_HIST_BINS": str(args.velocity_hist_bins),
        "VELOCITY_MAX_END_SPEED_RATIO": str(args.velocity_max_end_speed_ratio),
        "DEVICE": args.device,
        "TRAIN_PARTITION": args.train_partition,
        "COLLECT_PARTITION": args.collect_partition,
        "TRAIN_NODELIST": args.train_nodelist,
        "COLLECT_NODELIST": args.collect_nodelist,
        "TRAIN_TIME": args.train_time,
        "COLLECT_TIME": args.collect_time,
        "TRAIN_MEM": args.train_mem,
        "COLLECT_MEM": args.collect_mem,
        "TRAIN_GRES": args.train_gres,
        "CPUS_PER_TASK": str(args.cpus_per_task),
    }


def submit_jobs(args: argparse.Namespace, remote_output_root: str, remote_models_dir: str) -> tuple[str, str]:
    """Submit the array and collector jobs."""
    env = env_assignments(args, remote_output_root, remote_models_dir)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"{assignments} bash cluster/run_field_rnn_velocity_selection_100.sh submit"
    )
    stdout = ssh(args, command)
    if args.dry_run:
        print("Dry run: submit was not executed.", flush=True)
        return "DRYRUN_ARRAY", "DRYRUN_COLLECT"

    array_match = re.search(r"ARRAY_JOB_ID=(\S+)", stdout)
    collect_match = re.search(r"COLLECT_JOB_ID=(\S+)", stdout)
    if not array_match or not collect_match:
        raise RuntimeError(f"Could not parse job ids from submit output:\n{stdout}")
    array_job_id = array_match.group(1)
    collect_job_id = collect_match.group(1)
    print(f"Submitted array job {array_job_id}; collector job {collect_job_id}", flush=True)
    return array_job_id, collect_job_id


def poll_jobs(args: argparse.Namespace, array_job_id: str, collect_job_id: str, remote_output_root: str) -> None:
    """Wait until both array and collector disappear from squeue."""
    print(f"Waiting for array {array_job_id} and collector {collect_job_id}.", flush=True)
    ids = f"{array_job_id},{collect_job_id}"
    while True:
        queue_command = f"squeue -h -j {shlex.quote(ids)} -o '%i|%T|%M|%R'"
        queue = ssh(args, queue_command, check=False).strip()
        count_command = (
            f"cd {remote_cd_path(args.remote_workdir)} && "
            f"find {shlex.quote(remote_output_root)} -maxdepth 1 -type f -name 'metrics_*.tsv' "
            "2>/dev/null | wc -l"
        )
        count = ssh(args, count_command, check=False).strip()
        print(f"Metrics files: {count or '0'} / {args.model_count}", flush=True)
        if queue:
            print("Queue:")
            print(queue)
            if args.tail_lines > 0:
                tail_logs(args, array_job_id, collect_job_id)
            if args.dry_run:
                break
            time.sleep(args.poll_interval)
            continue
        print("Jobs are no longer in squeue.", flush=True)
        break


def tail_logs(args: argparse.Namespace, array_job_id: str, collect_job_id: str) -> None:
    """Print short tails from remote logs."""
    command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"echo COLLECT_LOG && tail -n {int(args.tail_lines)} logs/{shlex.quote(args.run_label)}_collect_{shlex.quote(collect_job_id)}.out "
        "2>/dev/null || true && "
        "echo ARRAY_LOGS && "
        f"for f in $(ls -t logs/{shlex.quote(args.run_label)}_{shlex.quote(array_job_id)}_*.out 2>/dev/null | head -n 2); "
        f"do echo $f; tail -n {int(args.tail_lines)} $f; done"
    )
    ssh(args, command, check=False)


def fetch_results(
    args: argparse.Namespace,
    array_job_id: str,
    collect_job_id: str,
    remote_output_root: str,
    remote_models_dir: str,
    local_out: Path,
    local_models: Path,
) -> None:
    """Fetch outputs, models, and logs from the cluster."""
    local_out.mkdir(parents=True, exist_ok=True)
    remote_output = f"{args.host}:{remote_join(args.remote_workdir, remote_output_root)}/"
    if args.fetch_plots:
        rsync(args, [remote_output], str(local_out) + "/")
    else:
        ssh_transport = f"ssh -i {shlex.quote(expanded_key(args.identity_file))} -p {args.port} -o BatchMode=yes"
        command = [
            "rsync",
            "-avz",
            "-e",
            ssh_transport,
            "--include",
            "summary.tsv",
            "--include",
            "top10.txt",
            "--include",
            "metrics_*.tsv",
            "--exclude",
            "*",
            remote_output,
            str(local_out) + "/",
        ]
        proc = run_command(command, dry_run=args.dry_run, check=True)
        if proc.stdout.strip():
            print(proc.stdout.rstrip(), flush=True)
        if proc.stderr.strip():
            print(proc.stderr.rstrip(), file=sys.stderr, flush=True)

    if args.fetch_models:
        local_models.mkdir(parents=True, exist_ok=True)
        remote_models = f"{args.host}:{remote_join(args.remote_workdir, remote_models_dir)}/"
        rsync(args, [remote_models], str(local_models) + "/")

    if args.fetch_logs:
        logs_dir = local_out / "logs"
        logs_dir.mkdir(exist_ok=True)
        remote_logs = f"{args.host}:{remote_join(args.remote_workdir, 'logs')}/"
        include = [
            "--include",
            f"{args.run_label}_{array_job_id}_*.out",
            "--include",
            f"{args.run_label}_{array_job_id}_*.err",
            "--include",
            f"{args.run_label}_collect_{collect_job_id}.out",
            "--include",
            f"{args.run_label}_collect_{collect_job_id}.err",
            "--exclude",
            "*",
        ]
        ssh_transport = f"ssh -i {shlex.quote(expanded_key(args.identity_file))} -p {args.port} -o BatchMode=yes"
        command = ["rsync", "-avz", "-e", ssh_transport, *include, remote_logs, str(logs_dir) + "/"]
        proc = run_command(command, dry_run=args.dry_run, check=True)
        if proc.stdout.strip():
            print(proc.stdout.rstrip(), flush=True)
        if proc.stderr.strip():
            print(proc.stderr.rstrip(), file=sys.stderr, flush=True)


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    """Read summary.tsv rows."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def row_score(row: dict[str, str]) -> float:
    """Return preferred selection score."""
    value = row.get("selection_score") or row.get("sqw_norm") or "inf"
    return float(value)


def print_summary(local_out: Path, local_models: Path, top: int) -> None:
    """Print a compact top-model summary."""
    summary_path = local_out / "summary.tsv"
    top10_path = local_out / "top10.txt"
    rows = sorted(read_summary_rows(summary_path), key=row_score)

    print("\n=== Model-selection result ===")
    print(f"summary: {summary_path}")
    print(f"top10:   {top10_path}")
    print(f"models:  {local_models}")

    if top10_path.exists():
        print("\nTop10 file:")
        print(top10_path.read_text(encoding="utf-8", errors="replace").strip())
    elif rows:
        print(f"\nTop {min(top, len(rows))} models:")
        for row in rows[:top]:
            velocity = row.get("velocity_score", "")
            if velocity:
                acceleration_ratio = row.get("acceleration_rms_ratio", "")
                acceleration_text = (
                    f"acc_rms={float(acceleration_ratio):.6g}\t"
                    if acceleration_ratio
                    else ""
                )
                print(
                    f"selection={row_score(row):.6g}\t"
                    f"sqw={float(row['sqw_norm']):.6g}\t"
                    f"velocity={float(velocity):.6g}\t"
                    f"{acceleration_text}"
                    f"{row.get('model_path', '')}"
                )
            else:
                print(f"{row_score(row):.6g}\t{row.get('model_path', '')}")
    else:
        print("No summary rows found.")

    metrics_count = len(list(local_out.glob("metrics_*.tsv")))
    plots_count = len(list(local_out.glob("plots_*")))
    model_count = len(list(local_models.glob("*.pth"))) if local_models.exists() else 0
    print("\nFetched counts:")
    print(f"metrics files: {metrics_count}")
    print(f"plot dirs:      {plots_count}")
    print(f"model files:    {model_count}")


def main() -> int:
    """Run the full submit-wait-fetch workflow."""
    args = parse_args()
    if args.model_count <= 0:
        raise ValueError("--model-count must be positive")
    if args.acceleration_rms_loss_weight < 0:
        raise ValueError("--acceleration-rms-loss-weight must be non-negative")
    if args.velocity_rms_loss_weight < 0:
        raise ValueError("--velocity-rms-loss-weight must be non-negative")
    if args.rms_loss_epsilon <= 0:
        raise ValueError("--rms-loss-epsilon must be positive")
    if args.low_q_stiffness_loss_weight < 0:
        raise ValueError("--low-q-stiffness-loss-weight must be non-negative")
    if args.low_q_stiffness_max_shell <= 0:
        raise ValueError("--low-q-stiffness-max-shell must be positive")
    if args.low_q_stiffness_epsilon <= 0:
        raise ValueError("--low-q-stiffness-epsilon must be positive")
    remote_output_root = output_root(args)
    remote_models_dir = models_dir(args)
    local_out = local_output_dir(args, remote_output_root)
    local_models = local_models_dir(args, remote_models_dir)

    print("FieldRNN cluster model selection")
    print(f"remote: {args.host}:{args.remote_workdir}")
    print(f"run label: {args.run_label}")
    print(f"model count: {args.model_count}")
    print(f"remote output: {remote_output_root}")
    print(f"remote models: {remote_models_dir}")
    print(f"local output: {local_out}")
    print(f"local models: {local_models}")

    sync_required_files(args)
    check_remote_environment(args)
    array_job_id, collect_job_id = submit_jobs(args, remote_output_root, remote_models_dir)
    poll_jobs(args, array_job_id, collect_job_id, remote_output_root)
    fetch_results(args, array_job_id, collect_job_id, remote_output_root, remote_models_dir, local_out, local_models)
    print_summary(local_out, local_models, args.top)
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
