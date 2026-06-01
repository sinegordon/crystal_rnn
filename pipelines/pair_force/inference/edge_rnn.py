#!/usr/bin/env python3
"""Run pair/edge RNN inference helper."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("infer_edge_rnn.py")
