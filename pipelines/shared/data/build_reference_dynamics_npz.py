#!/usr/bin/env python3
"""Build reference-dynamics arrays used by diagnostics."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("build_reference_dynamics_npz.py")
