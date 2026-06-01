#!/usr/bin/env python3
"""Check atom ordering in LAMMPS dumps or prepared crystal data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("check_atom_order.py")
