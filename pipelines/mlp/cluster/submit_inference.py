#!/usr/bin/env python3
"""Synchronize code and submit one MLP ASE inference plus postprocessing job."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.mlp.cluster._remote import remote_cd, run, ssh_command, sync_code, write_state  # noqa: E402
from pipelines.shared.cluster.config import defaults_from_argv  # noqa: E402


def parse_args():
    """Parse remote ASE inference options."""
    parent, cluster = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, parents=[parent])
    parser.add_argument("--host", default=cluster["host"])
    parser.add_argument("--port", default=cluster["port"])
    parser.add_argument("--identity-file", default=cluster["identity_file"])
    parser.add_argument("--remote-workdir", default=cluster["remote_workdir"])
    parser.add_argument("--partition", default=cluster["partition"])
    parser.add_argument("--nodelist", default=cluster["nodelist"])
    parser.add_argument("--conda-env", default=cluster["conda_env"])
    parser.add_argument("--model-path", required=True, help="Checkpoint path relative to remote-workdir.")
    parser.add_argument("--data-path", required=True, help="Inference NPZ path relative to remote-workdir.")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--initial-frames", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--taut-fs", type=float, default=200.0)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument("--block-batch-size", type=int, default=250)
    parser.add_argument("--ncells", type=int, default=10)
    parser.add_argument("--kcount", type=int, default=10)
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--mem", default="24G")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--sync-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Submit MLP NVT inference and all standard diagnostics as one job."""
    args = parse_args()
    if args.sync_code:
        sync_code(args)
    label = args.run_label or f"mlp_nvt_{args.steps}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_root = f"inference_outputs/{label}"
    log_root = f"logs/{label}"
    script = f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --partition={args.partition}
#SBATCH --nodelist={args.nodelist}
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --gres={args.gres}
#SBATCH --output={log_root}/run.out
#SBATCH --error={log_root}/run.err
set -euo pipefail
cd {args.remote_workdir}
eval "$(conda shell.bash hook)"
conda activate {shlex.quote(args.conda_env)}
mkdir -p {output_root}/postprocess {log_root}
python pipelines/mlp/ase/run_nvt.py \
  --model-path {shlex.quote(args.model_path)} \
  --data-path {shlex.quote(args.data_path)} \
  --output-npz {output_root}/trajectory.npz \
  --steps {args.steps} --initial-frames {args.initial_frames[0]} {args.initial_frames[1]} \
  --temperature-k {args.temperature_k} --taut-fs {args.taut_fs} --dt-ps {args.dt_ps} \
  --record-interval {args.record_interval} --block-batch-size {args.block_batch_size} \
  --device auto --periodic --remove-initial-com-velocity
python pipelines/mlp/postprocess/run_all.py \
  --ase-path {output_root}/trajectory.npz \
  --data-path {shlex.quote(args.data_path)} \
  --output-dir {output_root}/postprocess \
  --ncells {args.ncells} --kcount {args.kcount}
"""
    encoded = base64.b64encode(script.encode()).decode()
    remote = (
        f"cd {remote_cd(args.remote_workdir)} && mkdir -p {shlex.quote(log_root)} "
        f"{shlex.quote(output_root + '/postprocess')} && "
        f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(log_root + '/inference.slurm')} && "
        f"sbatch {shlex.quote(log_root + '/inference.slurm')}"
    )
    result = run(ssh_command(args, remote), dry_run=args.dry_run)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if result.returncode:
        return result.returncode
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    state_path = Path(args.state_path or ROOT / "logs" / f"{label}_inference_state.json")
    write_state(
        state_path,
        {
            "kind": "mlp-inference",
            "host": args.host,
            "port": str(args.port),
            "identity_file": args.identity_file,
            "remote_workdir": args.remote_workdir,
            "run_label": label,
            "job_id": "" if match is None else match.group(1),
            "output_root": output_root,
            "model_path": args.model_path,
            "data_path": args.data_path,
        },
    )
    print(f"Saved state: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
