#!/usr/bin/env python3
"""Check pair-force model-search progress."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.shared.cluster.check_edge_search import main


raise SystemExit(main())
