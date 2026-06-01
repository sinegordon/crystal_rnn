#!/usr/bin/env python3
"""Submit/wait/fetch a generic ASE 1055 cluster inference workflow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("run_cluster_ase_1055.py")
