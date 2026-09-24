#!/usr/bin/env python3
"""Create the reusable memory-mapped dataset for the multiclass proton MVA."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.MVA.run_dijet_mva_multiclass_protons import main


if __name__ == "__main__":
    if "--prepare-only" not in sys.argv:
        sys.argv.append("--prepare-only")
    main()
