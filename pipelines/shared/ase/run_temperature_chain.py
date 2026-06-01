#!/usr/bin/env python3
"""Run temperature-chain ASE copper NVT inference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("run_ase_copper_nvt_temperature_chain.py")
