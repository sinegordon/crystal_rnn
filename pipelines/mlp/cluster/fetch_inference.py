#!/usr/bin/env python3
"""Check an MLP inference job and optionally fetch its complete output directory."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.mlp.cluster._remote import identity_path, remote_cd, run, ssh_command  # noqa: E402


def main():
    """Print remote state and rsync finished trajectory and diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail-lines", type=int, default=30)
    args = parser.parse_args()
    state = json.loads(Path(args.state_path).read_text(encoding="utf-8"))
    for key, value in state.items():
        setattr(args, key, value)
    parts = [f"cd {remote_cd(state['remote_workdir'])}"]
    if state.get("job_id"):
        parts.append(f"squeue -j {shlex.quote(state['job_id'])} -o '%i|%T|%M|%R|%j' || true")
        parts.append(f"sacct -j {shlex.quote(state['job_id'])} --format=JobID,State,Elapsed,ExitCode -n || true")
    parts.append(f"find {shlex.quote(state['output_root'] + '/postprocess')} -maxdepth 1 -type f 2>/dev/null | sort || true")
    if args.tail_lines:
        parts.append(f"tail -n {args.tail_lines} logs/{shlex.quote(state['run_label'])}/run.out 2>/dev/null || true")
        parts.append(f"tail -n {args.tail_lines} logs/{shlex.quote(state['run_label'])}/run.err 2>/dev/null || true")
    status = run(ssh_command(args, " && ".join(parts)))
    if status.stdout:
        print(status.stdout.rstrip())
    if status.stderr:
        print(status.stderr.rstrip())
    if status.returncode or not args.fetch:
        return status.returncode
    destination = Path(args.output_dir or ROOT / "outputs" / state["run_label"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    transport = f"ssh -i {shlex.quote(identity_path(state['identity_file']))} -p {state['port']} -o BatchMode=yes"
    fetch = run(
        [
            "rsync",
            "-az",
            "-e",
            transport,
            f"{state['host']}:{state['remote_workdir'].rstrip('/')}/{state['output_root']}/",
            f"{destination}/",
        ],
        capture=False,
    )
    if fetch.returncode == 0:
        print(f"Fetched to {destination}")
    return fetch.returncode


if __name__ == "__main__":
    raise SystemExit(main())
