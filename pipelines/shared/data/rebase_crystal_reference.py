#!/usr/bin/env python3
"""Rebase crystal displacements to a different reference structure."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("rebase_crystal_reference.py")
