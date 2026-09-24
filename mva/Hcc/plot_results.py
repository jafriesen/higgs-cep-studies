#!/usr/bin/env python3
"""Plot the compact H(cc) MVA result artifact."""

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.plotting import plot_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(SCRIPT_DIR / "results" / "nominal_full"))
    args = parser.parse_args()
    plot_results(args.result_dir)


if __name__ == "__main__":
    main()
