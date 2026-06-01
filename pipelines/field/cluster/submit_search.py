#!/usr/bin/env python3
"""Submit, wait for, and fetch FieldRNN cluster model selection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("run_cluster_model_selection.py")
