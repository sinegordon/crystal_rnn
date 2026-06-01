#!/usr/bin/env python3
"""Plot OX sound-speed diagnostic."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines._launcher import run_root_script


run_root_script("plot_ase_sound_speed_ox.py")
