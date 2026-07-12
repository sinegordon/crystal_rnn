#!/usr/bin/env python3
"""Synchronize code and submit a standalone MLP model-search array."""

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
    """Parse Slurm search options."""
    parent, cluster = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, parents=[parent])
    parser.add_argument("--host", default=cluster["host"])
    parser.add_argument("--port", default=cluster["port"])
    parser.add_argument("--identity-file", default=cluster["identity_file"])
    parser.add_argument("--remote-workdir", default=cluster["remote_workdir"])
    parser.add_argument("--partition", default=cluster["train_partition"])
    parser.add_argument("--nodelist", default=cluster["train_nodelist"])
    parser.add_argument("--conda-env", default=cluster["conda_env"])
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--model-count", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260712)
    parser.add_argument("--data-path", default="data333_mlp_force.npz")
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--delta-frames", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--count-steps", type=int, default=2000)
    parser.add_argument("--count-run", type=int, default=3)
    parser.add_argument("--velocity-score-weight", type=float, default=0.0)
    parser.add_argument("--reference-pressure-loss-weight", type=float, default=0.0)
    parser.add_argument("--reference-pressure-target", type=float, default=0.0)
    parser.add_argument("--reference-pressure-loss-scale", type=float, default=1.0)
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--sync-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Submit one independent candidate per Slurm array task."""
    args = parse_args()
    if args.model_count <= 0:
        raise ValueError("model-count must be positive")
    if args.sync_code:
        sync_code(args)
    label = args.run_label or f"mlp_d{args.delta_frames}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_root = f"inference_outputs/{label}"
    models_root = f"models333_{label}"
    log_root = f"logs/{label}"
    eval_option = "" if args.eval_data_path is None else f" --eval-data-path {shlex.quote(args.eval_data_path)}"
    script = f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --partition={args.partition}
#SBATCH --nodelist={args.nodelist}
#SBATCH --array=0-{args.model_count - 1}
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --gres={args.gres}
#SBATCH --output={log_root}/%A_%a.out
#SBATCH --error={log_root}/%A_%a.err
set -euo pipefail
cd {args.remote_workdir}
eval "$(conda shell.bash hook)"
conda activate {shlex.quote(args.conda_env)}
TASK=${{SLURM_ARRAY_TASK_ID}}
SEED=$(({args.base_seed} + TASK))
mkdir -p {output_root}/task_${{TASK}} {models_root}/task_${{TASK}} {log_root}
python pipelines/mlp/search/find_models.py 1 \
  --data-path {shlex.quote(args.data_path)}{eval_option} \
  --models-dir {models_root}/task_${{TASK}} \
  --metrics-path {output_root}/task_${{TASK}}/metrics.tsv \
  --plot-output-dir {output_root}/task_${{TASK}} \
  --size {args.size} --delta-frames {args.delta_frames} \
  --epochs {args.epochs} --batch-size {args.batch_size} --learning-rate {args.learning_rate} \
  --count-steps {args.count_steps} --count-run {args.count_run} \
  --velocity-score-weight {args.velocity_score_weight} \
  --reference-pressure-loss-weight {args.reference_pressure_loss_weight} \
  --reference-pressure-target {args.reference_pressure_target} \
  --reference-pressure-loss-scale {args.reference_pressure_loss_scale} \
  --seed ${{SEED}} --save-all --plot-all --device auto
"""
    encoded = base64.b64encode(script.encode()).decode()
    remote = (
        f"cd {remote_cd(args.remote_workdir)} && mkdir -p {shlex.quote(log_root)} "
        f"{shlex.quote(output_root)} {shlex.quote(models_root)} && "
        f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(log_root + '/search.slurm')} && "
        f"sbatch {shlex.quote(log_root + '/search.slurm')}"
    )
    result = run(ssh_command(args, remote), dry_run=args.dry_run)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if result.returncode:
        return result.returncode
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    state_path = Path(args.state_path or ROOT / "logs" / f"{label}_search_state.json")
    write_state(
        state_path,
        {
            "kind": "mlp-search",
            "host": args.host,
            "port": str(args.port),
            "identity_file": args.identity_file,
            "remote_workdir": args.remote_workdir,
            "run_label": label,
            "job_id": "" if match is None else match.group(1),
            "output_root": output_root,
            "models_root": models_root,
        },
    )
    print(f"Saved state: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
