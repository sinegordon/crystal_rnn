#!/usr/bin/env python3
"""Submit an ASE 1055 run to the cluster.

The script submits the SLURM job and writes a small local state file.  Use
fetch_ase_1055.py to check completion and fetch post-processing outputs without
downloading large NPZ trajectory files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
from pathlib import Path

from pipelines.shared.cluster.config import defaults_from_argv


LOCAL_ROOT = Path(__file__).resolve().parent

DEFAULT_PAIR_FORCE_MODEL_PATH = (
    "models333_edge_pair_force_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_10/"
    "mean_norm_0.7968641992137379_rnn_pair_force_rnn_acceleration_h128_rl1_"
    "readoutfinalhidden_bidir_shells2_n18_targetforce_accnormglobal_"
    "pmean0.1_aover0.1_aunder0.01_op2_up2.pth"
)
DEFAULT_PAIR_ENERGY_MODEL_PATH = (
    "models333_pair_energy_rnn_finalhidden_rl1_force_pmean_w01_aover_w01_aunder_w001_d90k_30/"
    "mean_norm_0.6663052760722262_rnn_pair_energy_rnn_acceleration_h128_rl1_"
    "readoutfinalhidden_bidir_shells2_n18_targetforce_accnormglobal_"
    "pmean0.1_aover0.1_aunder0.01_op2_up2.pth"
)
DEFAULT_DATA_PATH = "data1055.npz"
DEFAULT_PAIR_FORCE_LABEL = "pair_force_finalhidden_rl1_sqw0797_tau50_qnone_50000"
DEFAULT_PAIR_ENERGY_LABEL = "pair_energy_finalhidden_sqw0666_bussi200_qnone_nointernal_10000"
DEFAULT_STATE_PATH = LOCAL_ROOT / "logs/ase1055_rl1_50000_last_job.json"


def preset_defaults(preset: str) -> dict[str, object]:
    """Return model and thermostat defaults for a named ASE run preset."""
    if preset == "pair-force":
        return {
            "model_path": DEFAULT_PAIR_FORCE_MODEL_PATH,
            "label": DEFAULT_PAIR_FORCE_LABEL,
            "steps": 10000,
            "taut_fs": 500.0,
            "q_zero_mode": "none",
            "history_damping_mode": "none",
            "history_damping_eta": 0.0,
            "history_damping_interval": 10,
            "history_damping_batch_size": 32,
            "history_damping_adaptive_gain": 0.0,
            "temperature_eta_adaptive_mode": "none",
            "temperature_eta_adaptive_gain": 0.0,
            "temperature_eta_adaptive_interval": 100,
            "temperature_eta_adaptive_ema": 0.002,
            "temperature_eta_adaptive_min_eta": 0.0,
            "temperature_eta_adaptive_max_eta": None,
            "temperature_eta_adaptive_deadband": 0.0,
            "power_bias_correction_mode": "none",
            "power_bias_correction_alpha": 1.0,
            "power_bias_correction_epsilon": 1e-30,
            "acceleration_scale": 1.0,
        }
    if preset == "pair-energy":
        return {
            "model_path": DEFAULT_PAIR_ENERGY_MODEL_PATH,
            "label": DEFAULT_PAIR_ENERGY_LABEL,
            "steps": 10000,
            "taut_fs": 200.0,
            "q_zero_mode": "none",
            "history_damping_mode": "none",
            "history_damping_eta": 0.0,
            "history_damping_interval": 10,
            "history_damping_batch_size": 32,
            "history_damping_adaptive_gain": 0.0,
            "temperature_eta_adaptive_mode": "none",
            "temperature_eta_adaptive_gain": 0.0,
            "temperature_eta_adaptive_interval": 100,
            "temperature_eta_adaptive_ema": 0.002,
            "temperature_eta_adaptive_min_eta": 0.0,
            "temperature_eta_adaptive_max_eta": None,
            "temperature_eta_adaptive_deadband": 0.0,
            "power_bias_correction_mode": "none",
            "power_bias_correction_alpha": 1.0,
            "power_bias_correction_epsilon": 1e-30,
            "acceleration_scale": 1.0,
        }
    raise ValueError(f"Unknown preset: {preset}")


def parse_args() -> argparse.Namespace:
    cluster_parent, cluster = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, parents=[cluster_parent])
    parser.add_argument(
        "--preset",
        choices=["pair-force", "pair-energy"],
        default="pair-force",
        help=(
            "Named run preset. 'pair-force' preserves the previous RL1 defaults; "
            "'pair-energy' uses the current conservative energy model with clean Bussi NVT defaults."
        ),
    )
    parser.add_argument("--host", default=cluster["host"])
    parser.add_argument("--port", default=cluster["port"])
    parser.add_argument("--identity-file", default=cluster["identity_file"])
    parser.add_argument("--remote-workdir", default=cluster["remote_workdir"])
    parser.add_argument("--partition", default=cluster["partition"])
    parser.add_argument("--nodelist", default=cluster["nodelist"])
    parser.add_argument("--conda-env", default=cluster["conda_env"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--taut-fs", type=float, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--q-zero-mode", choices=["none", "initial", "constant_velocity", "zero"], default=None)
    parser.add_argument("--acceleration-scale", type=float, default=None)
    parser.add_argument("--history-damping-mode", choices=["none", "local-positive"], default=None)
    parser.add_argument("--history-damping-eta", type=float, default=None)
    parser.add_argument("--history-damping-interval", type=int, default=None)
    parser.add_argument("--history-damping-batch-size", type=int, default=None)
    parser.add_argument("--history-damping-adaptive-gain", type=float, default=None)
    parser.add_argument(
        "--temperature-eta-adaptive-mode",
        choices=["none", "log"],
        default=None,
        help="Adapt history-damping eta from smoothed temperature.",
    )
    parser.add_argument("--temperature-eta-adaptive-gain", type=float, default=None)
    parser.add_argument("--temperature-eta-adaptive-interval", type=int, default=None)
    parser.add_argument("--temperature-eta-adaptive-ema", type=float, default=None)
    parser.add_argument("--temperature-eta-adaptive-min-eta", type=float, default=None)
    parser.add_argument("--temperature-eta-adaptive-max-eta", type=float, default=None)
    parser.add_argument("--temperature-eta-adaptive-deadband", type=float, default=None)
    parser.add_argument("--power-bias-correction-mode", choices=["none", "global", "global-positive"], default=None)
    parser.add_argument("--power-bias-correction-alpha", type=float, default=None)
    parser.add_argument("--power-bias-correction-epsilon", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_run_config(args: argparse.Namespace) -> dict[str, object]:
    """Merge preset defaults with explicit command-line overrides."""
    defaults = preset_defaults(args.preset)
    config = dict(defaults)
    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def expanded(path: str) -> str:
    return str(Path(path).expanduser())


def remote_cd_path(path: str) -> str:
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def run(command: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "Submitted batch job DRYRUN\n", "")
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout.strip():
        print(proc.stdout.rstrip(), flush=True)
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def ssh(args: argparse.Namespace, remote_command: str) -> str:
    command = [
        "ssh",
        "-i",
        expanded(args.identity_file),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        args.host,
        remote_command,
    ]
    return run(command, dry_run=args.dry_run).stdout


def output_root(args: argparse.Namespace, config: dict[str, object]) -> str:
    if args.output_root:
        return args.output_root.strip("/")
    return f"inference_outputs/ase_nvt_1055/{config['label']}"


def build_env(args: argparse.Namespace, config: dict[str, object], remote_output_root: str) -> dict[str, str]:
    """Build environment variables consumed by cluster/run_ase_1055_nvt.slurm."""
    env = {
        "CONDA_ENV": args.conda_env,
        "MODEL_PATH": str(config["model_path"]),
        "DATA_PATH": args.data_path,
        "OUTPUT_ROOT": remote_output_root,
        "STEPS": str(config["steps"]),
        "DT_PS": str(args.dt_ps),
        "TEMPERATURE_K": str(args.temperature_k),
        "TAUT_FS": str(config["taut_fs"]),
        "PATCH_BATCH_SIZE": str(args.patch_batch_size),
        "DEVICE": args.device,
        "NCELLS": "10",
        "KCOUNT": "10",
        "SQW_STEP": "10",
        "VELOCITY_WINDOW": "10",
        "VELOCITY_BINS": "120",
        "PHASE_WINDOW_FRAMES": "1000",
        "PHASE_BINS": "120",
        "SOUND_FRAME_STRIDE": "1",
        "SOUND_FIT_Q_COUNT": "3",
        "SOUND_Q_INDEX_MIN": "1",
        "SOUND_MAX_ENERGY_MEV": "45.0",
        "SOUND_SMOOTH_BINS": "5",
        "PATH_ENERGY_SLABS": "10",
        "PATH_ENERGY_PLOT_STRIDE": "1",
        "PATH_ENERGY_FIT_FRACTION_START": "0.1",
        "PATH_ENERGY_FIT_FRACTION_END": "1.0",
        "HEAT_CAPACITY_FRAME_FRACTION_START": "0.0",
        "HEAT_CAPACITY_FRAME_FRACTION_END": "1.0",
        "HEAT_CAPACITY_DETREND": "1",
        "Q_ZERO_MODE": str(config["q_zero_mode"]),
        "ACCELERATION_SCALE": str(config["acceleration_scale"]),
        "HISTORY_DAMPING_MODE": str(config["history_damping_mode"]),
        "TEMPERATURE_ETA_ADAPTIVE_MODE": str(config["temperature_eta_adaptive_mode"]),
        "POWER_BIAS_CORRECTION_MODE": str(config["power_bias_correction_mode"]),
    }

    # Keep the default pair-energy path clean: optional correction parameters are
    # exported only when their corresponding mode is active.
    if config["history_damping_mode"] != "none":
        env.update(
            {
                "HISTORY_DAMPING_ETA": str(config["history_damping_eta"]),
                "HISTORY_DAMPING_INTERVAL": str(config["history_damping_interval"]),
                "HISTORY_DAMPING_BATCH_SIZE": str(config["history_damping_batch_size"]),
                "HISTORY_DAMPING_ADAPTIVE_GAIN": str(config["history_damping_adaptive_gain"]),
            }
        )
    if config["temperature_eta_adaptive_mode"] != "none":
        env.update(
            {
                "TEMPERATURE_ETA_ADAPTIVE_GAIN": str(config["temperature_eta_adaptive_gain"]),
                "TEMPERATURE_ETA_ADAPTIVE_INTERVAL": str(config["temperature_eta_adaptive_interval"]),
                "TEMPERATURE_ETA_ADAPTIVE_EMA": str(config["temperature_eta_adaptive_ema"]),
                "TEMPERATURE_ETA_ADAPTIVE_MIN_ETA": str(config["temperature_eta_adaptive_min_eta"]),
                "TEMPERATURE_ETA_ADAPTIVE_DEADBAND": str(config["temperature_eta_adaptive_deadband"]),
            }
        )
    if config["temperature_eta_adaptive_mode"] != "none" and config["temperature_eta_adaptive_max_eta"] is not None:
        env["TEMPERATURE_ETA_ADAPTIVE_MAX_ETA"] = str(config["temperature_eta_adaptive_max_eta"])
    if config["power_bias_correction_mode"] != "none":
        env.update(
            {
                "POWER_BIAS_CORRECTION_ALPHA": str(config["power_bias_correction_alpha"]),
                "POWER_BIAS_CORRECTION_EPSILON": str(config["power_bias_correction_epsilon"]),
            }
        )
    return env


def save_state(
    args: argparse.Namespace,
    config: dict[str, object],
    env: dict[str, str],
    job_id: str,
    remote_output_root: str,
) -> Path:
    state_path = Path(args.state_path)
    if not state_path.is_absolute():
        state_path = LOCAL_ROOT / state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "job_id": job_id,
        "preset": args.preset,
        "label": config["label"],
        "remote_output_root": remote_output_root,
        "local_output_root": str(LOCAL_ROOT / remote_output_root),
        "host": args.host,
        "port": str(args.port),
        "identity_file": args.identity_file,
        "remote_workdir": args.remote_workdir,
        "model_path": config["model_path"],
        "data_path": args.data_path,
        "steps": config["steps"],
        "dt_ps": args.dt_ps,
        "temperature_k": args.temperature_k,
        "taut_fs": config["taut_fs"],
        "run_config": config,
        "slurm_env": env,
        "submitted_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state_path


def main() -> int:
    args = parse_args()
    config = resolved_run_config(args)
    remote_output_root = output_root(args, config)
    env = build_env(args, config, remote_output_root)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    sbatch_options = [f"--partition={args.partition}"]
    if args.nodelist:
        sbatch_options.append(f"--nodelist={args.nodelist}")
    command = (
        f"cd {remote_cd_path(args.remote_workdir)} && "
        f"{assignments} sbatch {' '.join(shlex.quote(value) for value in sbatch_options)} "
        "cluster/run_ase_1055_nvt.slurm"
    )
    stdout = ssh(args, command)
    match = re.search(r"Submitted batch job\s+(\d+|DRYRUN)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output:\n{stdout}")
    job_id = match.group(1)
    if args.dry_run:
        print("\nDry run only; state file was not updated.")
        print(f"preset: {args.preset}")
        print(f"remote_output_root: {remote_output_root}")
        return 0
    state_path = save_state(args, config, env, job_id, remote_output_root)
    print("\nSubmitted ASE 1055 run")
    print(f"preset: {args.preset}")
    print(f"job_id: {job_id}")
    print(f"remote_output_root: {remote_output_root}")
    print(f"state_path: {state_path}")
    print("\nCheck/fetch command:")
    print(f"python fetch_ase_1055.py --state-path {shlex.quote(str(state_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
