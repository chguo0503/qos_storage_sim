#!/usr/bin/env python3
"""Reuse the unchanged parent pilot, writing only in this subdirectory."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_pilot
run_pilot.HERE=HERE
if __name__=='__main__':run_pilot.main()
