#!/usr/bin/env python3
"""Submit a pair-energy model-search Slurm array."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
from pathlib import Path

from pipelines.shared.cluster.config import defaults_from_argv


LOCAL_ROOT = Path(__file__).resolve().parents[3]


def parse_args(default_architecture: str | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    cluster_parent, cluster = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, parents=[cluster_parent])
    parser.add_argument("--architecture", choices=["pair-energy"], default=default_architecture or "pair-energy")
    parser.add_argument("--host", default=cluster["host"])
    parser.add_argument("--port", default=cluster["port"])
    parser.add_argument("--identity-file", default=cluster["identity_file"])
    parser.add_argument("--remote-workdir", default=cluster["remote_workdir"])
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--model-count", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260531)
    parser.add_argument("--data-path", default="data333_force.npz")
    parser.add_argument("--delta-frames", type=int, default=90000)
    parser.add_argument("--count-steps", type=int, default=2000)
    parser.add_argument("--count-run", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-len", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--rnn-type", choices=["RNN", "GRU", "LSTM"], default="RNN")
    parser.add_argument("--rnn-readout-mode", choices=["last-output", "final-hidden"], default="final-hidden")
    parser.add_argument("--neighbor-shells", type=int, default=2)
    parser.add_argument("--cutoff-scale", type=float, default=1.05)
    parser.add_argument("--acceleration-normalization", choices=["none", "global", "channel"], default="global")
    parser.add_argument("--training-target", choices=["displacement", "force"], default="force")
    parser.add_argument("--velocity-score-weight", type=float, default=0.0)
    parser.add_argument("--acceleration-score-weight", type=float, default=0.0)
    parser.add_argument("--velocity-window-frames", type=int, default=10)
    parser.add_argument("--velocity-hist-bins", type=int, default=80)
    parser.add_argument("--power-mean-loss-weight", type=float, default=0.1)
    parser.add_argument("--q-power-loss-weight", type=float, default=0.0)
    parser.add_argument("--q-power-loss-mode", choices=["match", "positive-excess"], default="positive-excess")
    parser.add_argument("--q-power-loss-sample-count", type=int, default=2)
    parser.add_argument("--q-power-loss-interval", type=int, default=10)
    parser.add_argument("--q-power-loss-margin", type=float, default=0.0)
    parser.add_argument("--q-power-loss-epsilon", type=float, default=1e-12)
    parser.add_argument("--q-power-loss-exclude-q-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--acceleration-over-rms-loss-weight", type=float, default=0.1)
    parser.add_argument("--acceleration-under-rms-loss-weight", type=float, default=0.01)
    parser.add_argument("--train-partition", default=cluster["train_partition"])
    parser.add_argument("--train-nodelist", default=cluster["train_nodelist"])
    parser.add_argument("--train-time", default="12:00:00")
    parser.add_argument("--train-mem", default="16G")
    parser.add_argument("--train-gres", default="gpu:1")
    parser.add_argument("--collect-partition", default=cluster["collect_partition"])
    parser.add_argument("--collect-time", default="00:20:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return args


def expanded(path: str) -> str:
    """Return a user-expanded local path."""
    return str(Path(path).expanduser())


def remote_cd_path(path: str) -> str:
    """Quote a remote cd path while preserving tilde expansion."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def default_run_label(args: argparse.Namespace) -> str:
    """Build a stable default run label."""
    architecture = args.architecture.replace("-", "_")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    q_power_tag = f"_qpow_w{args.q_power_loss_weight:g}" if args.q_power_loss_weight > 0 else ""
    return (
        f"{architecture}_rnn_finalhidden_rl{args.rnn_layers}_force"
        f"_pmean_w{args.power_mean_loss_weight:g}"
        f"{q_power_tag}"
        f"_aover_w{args.acceleration_over_rms_loss_weight:g}"
        f"_aunder_w{args.acceleration_under_rms_loss_weight:g}"
        f"_d{args.delta_frames // 1000}k_{args.model_count}_{timestamp}"
    ).replace(".", "p")


def run(args: argparse.Namespace, command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a local command."""
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if args.dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main(default_architecture: str | None = None) -> int:
    """Submit the remote Slurm array."""
    args = parse_args(default_architecture)
    run_label = args.run_label or default_run_label(args)
    state_path = Path(args.state_path) if args.state_path else LOCAL_ROOT / "logs" / f"{run_label}_search_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "RUN_LABEL": run_label,
        "MODEL_COUNT": str(args.model_count),
        "BASE_SEED": str(args.base_seed),
        "DATA_PATH": args.data_path,
        "ARCHITECTURE": args.architecture,
        "TRAINING_TARGET": args.training_target,
        "RNN_TYPE": args.rnn_type,
        "RNN_READOUT_MODE": args.rnn_readout_mode,
        "HIDDEN_SIZE": str(args.hidden_size),
        "RNN_LAYERS": str(args.rnn_layers),
        "NEIGHBOR_SHELLS": str(args.neighbor_shells),
        "CUTOFF_SCALE": str(args.cutoff_scale),
        "ACCELERATION_NORMALIZATION": args.acceleration_normalization,
        "DELTA_FRAMES": str(args.delta_frames),
        "DATA_LEN": str(args.data_len),
        "EPOCHS": str(args.epochs),
        "BATCH_SIZE": str(args.batch_size),
        "LEARNING_RATE": str(args.learning_rate),
        "COUNT_STEPS": str(args.count_steps),
        "COUNT_RUN": str(args.count_run),
        "VELOCITY_SCORE_WEIGHT": str(args.velocity_score_weight),
        "ACCELERATION_SCORE_WEIGHT": str(args.acceleration_score_weight),
        "VELOCITY_WINDOW_FRAMES": str(args.velocity_window_frames),
        "VELOCITY_HIST_BINS": str(args.velocity_hist_bins),
        "POWER_MEAN_LOSS_WEIGHT": str(args.power_mean_loss_weight),
        "Q_POWER_LOSS_WEIGHT": str(args.q_power_loss_weight),
        "Q_POWER_LOSS_MODE": args.q_power_loss_mode,
        "Q_POWER_LOSS_SAMPLE_COUNT": str(args.q_power_loss_sample_count),
        "Q_POWER_LOSS_INTERVAL": str(args.q_power_loss_interval),
        "Q_POWER_LOSS_MARGIN": str(args.q_power_loss_margin),
        "Q_POWER_LOSS_EPSILON": str(args.q_power_loss_epsilon),
        "Q_POWER_LOSS_EXCLUDE_Q_ZERO": "true" if args.q_power_loss_exclude_q_zero else "false",
        "ACCELERATION_OVER_RMS_LOSS_WEIGHT": str(args.acceleration_over_rms_loss_weight),
        "ACCELERATION_UNDER_RMS_LOSS_WEIGHT": str(args.acceleration_under_rms_loss_weight),
        "TRAIN_PARTITION": args.train_partition,
        "TRAIN_NODELIST": args.train_nodelist,
        "TRAIN_TIME": args.train_time,
        "TRAIN_MEM": args.train_mem,
        "TRAIN_GRES": args.train_gres,
        "COLLECT_PARTITION": args.collect_partition,
        "COLLECT_TIME": args.collect_time,
        "CPUS_PER_TASK": str(args.cpus_per_task),
    }
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    remote_command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"{assignments} bash cluster/run_edge_rnn_search.sh submit"
    )
    command = [
        "ssh",
        "-i",
        expanded(args.identity_file),
        "-p",
        args.port,
        "-o",
        "BatchMode=yes",
        args.host,
        remote_command,
    ]
    proc = run(args, command)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        return proc.returncode

    array_match = re.search(r"ARRAY_JOB_ID=(\S+)", proc.stdout)
    collect_match = re.search(r"COLLECT_JOB_ID=(\S+)", proc.stdout)
    output_match = re.search(r"OUTPUT_ROOT=(\S+)", proc.stdout)
    models_match = re.search(r"MODELS_DIR=(\S+)", proc.stdout)
    state = {
        "host": args.host,
        "port": args.port,
        "identity_file": args.identity_file,
        "remote_workdir": args.remote_workdir,
        "run_label": run_label,
        "architecture": args.architecture,
        "array_job_id": array_match.group(1) if array_match else "",
        "collect_job_id": collect_match.group(1) if collect_match else "",
        "output_root": output_match.group(1) if output_match else f"inference_outputs/{run_label}",
        "models_dir": models_match.group(1) if models_match else f"models333_{run_label}",
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved state: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
