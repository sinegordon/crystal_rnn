"""Cluster configuration helpers for local pipeline launchers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ENV_CONFIG_PATH = "CRYSTAL_RNN_CLUSTER_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("cluster_config.json")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"Cluster config must be a JSON object: {path}")
    return loaded


def _require(config: dict[str, Any], *path: str) -> Any:
    """Return a required nested config value."""
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise SystemExit(f"Missing cluster config key: {dotted}")
        current = current[key]
    return current


def load_cluster_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load cluster settings from JSON.

    The repository default is used as a base config.  A user-supplied config can
    override only the keys that differ from the default.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        raise SystemExit(f"Default cluster config does not exist: {DEFAULT_CONFIG_PATH}")
    config = _read_json_object(DEFAULT_CONFIG_PATH)
    selected = config_path or os.environ.get(ENV_CONFIG_PATH)
    if selected:
        path = Path(selected).expanduser()
        if not path.exists():
            raise SystemExit(f"Cluster config does not exist: {path}")
        config = _deep_merge(config, _read_json_object(path))
    return config


def cluster_defaults(config_path: str | Path | None = None) -> dict[str, str]:
    """Return flattened defaults used by argparse-based pipeline scripts."""
    config = load_cluster_config(config_path)
    return {
        "host": str(_require(config, "ssh", "host")),
        "port": str(_require(config, "ssh", "port")),
        "identity_file": str(_require(config, "ssh", "identity_file")),
        "remote_workdir": str(_require(config, "paths", "remote_workdir")),
        "partition": str(_require(config, "slurm", "default_partition")),
        "nodelist": str(_require(config, "slurm", "gpu_nodelist")),
        "train_partition": str(_require(config, "slurm", "train_partition")),
        "train_nodelist": str(_require(config, "slurm", "train_nodelist")),
        "collect_partition": str(_require(config, "slurm", "collect_partition")),
        "conda_env": str(_require(config, "slurm", "conda_env")),
    }


def cluster_config_parent_parser() -> argparse.ArgumentParser:
    """Return a reusable parent parser that exposes the config-file option."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cluster-config",
        default=None,
        help=(
            "Cluster JSON config path. Defaults to "
            f"{DEFAULT_CONFIG_PATH}; can also be set with {ENV_CONFIG_PATH}."
        ),
    )
    return parser


def defaults_from_argv(argv: list[str] | None = None) -> tuple[argparse.ArgumentParser, dict[str, str]]:
    """Pre-parse --cluster-config and return a parent parser plus defaults."""
    parent = cluster_config_parent_parser()
    known, _ = parent.parse_known_args(argv)
    return parent, cluster_defaults(known.cluster_config)
