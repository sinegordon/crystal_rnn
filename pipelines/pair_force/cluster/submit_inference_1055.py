#!/usr/bin/env python3
"""Submit pair-force ASE 1055 inference to the cluster."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import ensure_fixed, run_root_script


args = list(sys.argv[1:])
ensure_fixed(args, "--preset", "pair-force")
run_root_script("submit_ase_1055.py", args)
