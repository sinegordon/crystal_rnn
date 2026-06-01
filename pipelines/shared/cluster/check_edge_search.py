#!/usr/bin/env python3
"""Check pair-force or pair-energy Slurm-array search progress."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from pipelines.shared.cluster.config import defaults_from_argv


LOCAL_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    cluster_parent, cluster = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, parents=[cluster_parent])
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--identity-file", default=None)
    parser.add_argument("--remote-workdir", default=None)
    parser.add_argument("--array-job-id", default="")
    parser.add_argument("--collect-job-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--tail-lines", type=int, default=30)
    args = parser.parse_args()
    args.cluster_defaults = cluster
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


def load_state(path: str | None) -> dict[str, str]:
    """Load a saved search state."""
    if not path:
        return {}
    state_path = Path(path)
    if not state_path.is_absolute():
        state_path = LOCAL_ROOT / state_path
    if not state_path.exists():
        raise SystemExit(f"Missing state file: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def merge_config(args: argparse.Namespace, state: dict[str, str]) -> dict[str, str]:
    """Merge explicit CLI values with saved state."""
    cluster = args.cluster_defaults
    return {
        "host": args.host or state.get("host", cluster["host"]),
        "port": args.port or state.get("port", cluster["port"]),
        "identity_file": args.identity_file or state.get("identity_file", cluster["identity_file"]),
        "remote_workdir": args.remote_workdir or state.get("remote_workdir", cluster["remote_workdir"]),
        "array_job_id": args.array_job_id or state.get("array_job_id", ""),
        "collect_job_id": args.collect_job_id or state.get("collect_job_id", ""),
        "output_root": args.output_root or state.get("output_root", ""),
        "models_dir": args.models_dir or state.get("models_dir", ""),
    }


def ssh(config: dict[str, str], remote_command: str) -> str:
    """Run a remote command."""
    command = [
        "ssh",
        "-i",
        expanded(config["identity_file"]),
        "-p",
        config["port"],
        "-o",
        "BatchMode=yes",
        config["host"],
        remote_command,
    ]
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.stderr.strip():
        print(proc.stderr.rstrip())
    return proc.stdout


def main() -> int:
    """Print queue state, metrics count, and current top models."""
    args = parse_args()
    config = merge_config(args, load_state(args.state_path))
    job_ids = ",".join(value for value in [config["array_job_id"], config["collect_job_id"]] if value)
    if not config["output_root"]:
        raise SystemExit("--output-root or --state-path is required")
    if not config["models_dir"]:
        raise SystemExit("--models-dir or --state-path is required")

    remote_parts = [f"cd {remote_cd_path(config['remote_workdir'])}"]
    if job_ids:
        remote_parts.append(f"echo '=== queue ==='; squeue -j {shlex.quote(job_ids)} -o '%i|%T|%M|%R|%j' || true")
    remote_parts.extend(
        [
            "echo '=== metrics count ==='",
            f"find {shlex.quote(config['output_root'])} -maxdepth 1 -name 'metrics_*.tsv' 2>/dev/null | wc -l",
            "echo '=== models count ==='",
            f"find {shlex.quote(config['models_dir'])} -maxdepth 1 -name '*.pth' 2>/dev/null | wc -l",
            "echo '=== top10 ==='",
            f"cat {shlex.quote(config['output_root'])}/top10.txt 2>/dev/null || true",
        ]
    )
    if args.tail_lines > 0:
        label = Path(config["output_root"]).name
        remote_parts.extend(
            [
                "echo '=== recent logs ==='",
                (
                    f"for f in $(ls -t logs/{shlex.quote(label)}_*.out 2>/dev/null | head -n 3); "
                    f"do echo --- $f; tail -n {int(args.tail_lines)} $f; done"
                ),
                "echo '=== recent errors ==='",
                (
                    f"for f in $(ls -t logs/{shlex.quote(label)}_*.err 2>/dev/null | head -n 3); "
                    f"do if [ -s $f ]; then echo --- $f; tail -n {int(args.tail_lines)} $f; fi; done"
                ),
            ]
        )
    print(ssh(config, " && ".join(remote_parts)).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
