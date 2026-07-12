#!/usr/bin/env python3
"""Check a standalone MLP Slurm search and print its current top models."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.mlp.cluster._remote import remote_cd, run, ssh_command  # noqa: E402


def main():
    """Read a search state file and query remote queue and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_path")
    parser.add_argument("--tail-lines", type=int, default=20)
    args = parser.parse_args()
    state = json.loads(Path(args.state_path).read_text(encoding="utf-8"))
    for key, value in state.items():
        setattr(args, key, value)
    remote_python = r'''import csv, glob
rows=[]
for path in glob.glob("''' + state["output_root"] + r'''/task_*/metrics.tsv"):
    with open(path, encoding="utf-8") as stream:
        rows.extend(csv.DictReader(stream, delimiter="\t"))
rows.sort(key=lambda row: float(row["selection_score"]))
print("completed =", len(rows))
print("rank\tselection\tsqw\tvelocity\tmodel")
for rank, row in enumerate(rows[:10], 1):
    print(f'{rank}\t{float(row["selection_score"]):.6f}\t{float(row["sqw_norm"]):.6f}\t{float(row["velocity_score"]):.6f}\t{row["model_path"]}')
'''
    parts = [f"cd {remote_cd(state['remote_workdir'])}"]
    job_ids = ",".join(
        value for value in (state.get("job_id", ""), state.get("collect_job_id", "")) if value
    )
    if job_ids:
        parts.append(f"squeue -j {shlex.quote(job_ids)} -o '%i|%T|%M|%R|%j' || true")
    parts.append(f"python -c {shlex.quote(remote_python)}")
    if args.tail_lines:
        parts.append(
            f"for f in $(ls -t logs/{shlex.quote(state['run_label'])}/*.out 2>/dev/null | head -n 2); "
            f"do echo ===$f; tail -n {args.tail_lines} $f; done"
        )
    result = run(ssh_command(args, " && ".join(parts)))
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
