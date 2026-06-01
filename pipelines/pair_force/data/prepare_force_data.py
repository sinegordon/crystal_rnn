#!/usr/bin/env python3
"""Prepare force-enabled crystal data for pair-force models."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


args = list(sys.argv[1:])
if "--include-forces" not in args:
    args.append("--include-forces")
if "--require-forces" not in args:
    args.append("--require-forces")
run_root_script("prepare_crystal_data.py", args)
