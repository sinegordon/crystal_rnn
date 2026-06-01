#!/usr/bin/env python3
"""Search conservative pair-energy RNN models."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import ensure_default, ensure_fixed, run_root_script


args = list(sys.argv[1:])
ensure_fixed(args, "--architecture", "pair-energy")
ensure_default(args, "--training-target", "force")
ensure_default(args, "--rnn-readout-mode", "final-hidden")
ensure_default(args, "--acceleration-normalization", "global")
run_root_script("find_edge_rnn_models.py", args)
