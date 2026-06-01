#!/usr/bin/env python3
"""Check/fetch pair-force ASE 1055 inference outputs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("fetch_ase_1055.py")
