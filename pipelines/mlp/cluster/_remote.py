"""Shared SSH and rsync helpers for the standalone MLP pipeline."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def identity_path(path):
    """Expand a local SSH identity path."""
    return str(Path(path).expanduser())


def remote_cd(path):
    """Quote a remote directory while preserving tilde expansion."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def ssh_command(args, remote_command):
    """Return the SSH command for configured cluster access."""
    return [
        "ssh",
        "-i",
        identity_path(args.identity_file),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        args.host,
        remote_command,
    ]


def run(command, dry_run=False, capture=True):
    """Print and execute one local command."""
    print("$ " + " ".join(shlex.quote(str(part)) for part in command), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )


def sync_code(args):
    """Synchronize source code while leaving remote datasets and models intact."""
    mkdir = run(
        ssh_command(args, f"mkdir -p {remote_cd(args.remote_workdir)}"),
        dry_run=args.dry_run,
    )
    if mkdir.returncode:
        raise SystemExit(mkdir.stderr)
    ssh_transport = f"ssh -i {shlex.quote(identity_path(args.identity_file))} -p {args.port} -o BatchMode=yes"
    command = [
        "rsync",
        "-az",
        "--exclude=.git/",
        "--exclude=*.npz",
        "--exclude=*.pth",
        "--exclude=outputs/",
        "--exclude=inference_outputs/",
        "--exclude=models*/",
        "--exclude=logs/",
        "-e",
        ssh_transport,
        f"{ROOT}/",
        f"{args.host}:{args.remote_workdir.rstrip('/')}/",
    ]
    result = run(command, dry_run=args.dry_run, capture=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def write_state(path, state):
    """Write a local JSON state file."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
