#!/usr/bin/env python3
"""Submit pair-energy model search to the cluster."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.shared.cluster.submit_edge_search import main


raise SystemExit(main(default_architecture="pair-energy"))
