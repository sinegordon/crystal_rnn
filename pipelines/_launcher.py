"""Small helpers for safe pipeline wrappers.

The original scripts stay at the repository root for backwards compatibility.
Pipeline wrappers set paradigm defaults, add the repository root to
``sys.path``, and execute the original implementation.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root regardless of wrapper depth."""
    return Path(__file__).resolve().parents[1]


def _option_index(args: list[str], option: str) -> int | None:
    """Return the index of an option written as ``--x value`` or ``--x=value``."""
    for index, arg in enumerate(args):
        if arg == option or arg.startswith(option + "="):
            return index
    return None


def _option_value(args: list[str], option: str) -> str | None:
    """Return a command-line option value when present."""
    index = _option_index(args, option)
    if index is None:
        return None
    arg = args[index]
    if arg.startswith(option + "="):
        return arg.split("=", 1)[1]
    if index + 1 >= len(args):
        raise SystemExit(f"{option} requires a value")
    return args[index + 1]


def ensure_default(args: list[str], option: str, value: str) -> None:
    """Append an option only when the caller did not provide it."""
    if _option_index(args, option) is None:
        args.extend([option, value])


def ensure_fixed(args: list[str], option: str, value: str) -> None:
    """Set a fixed option and reject contradictory explicit values."""
    existing = _option_value(args, option)
    if existing is None:
        args.extend([option, value])
    elif existing != value:
        raise SystemExit(f"{option} must be {value!r} for this pipeline, got {existing!r}")


def run_root_script(script_name: str, args: list[str] | None = None) -> None:
    """Execute a root-level Python script as ``__main__``."""
    root = repo_root()
    script_path = root / script_name
    if not script_path.exists():
        raise SystemExit(f"Missing root script: {script_path}")
    sys.path.insert(0, str(root))
    sys.argv = [str(script_path), *(sys.argv[1:] if args is None else args)]
    runpy.run_path(str(script_path), run_name="__main__")
