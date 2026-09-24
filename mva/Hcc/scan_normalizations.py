#!/usr/bin/env python3
"""Scan H(cc) survival and tag normalizations with the score held fixed."""

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.scans import run_normalization_scans  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(SCRIPT_DIR / "results" / "nominal_full"))
    parser.add_argument("--baseline-survival", type=float, default=0.03)
    parser.add_argument("--min-survival", type=float, default=0.01)
    parser.add_argument("--max-survival", type=float, default=0.30)
    parser.add_argument("--survival-points", type=int, default=41)
    parser.add_argument("--tag-points", type=int, default=31)
    parser.add_argument("--tag-min-scale", type=float, default=0.5)
    parser.add_argument("--tag-max-scale", type=float, default=1.5)
    parser.add_argument("--eff-c", type=float, default=0.65)
    parser.add_argument("--mistag-b-to-c", type=float, default=0.055)
    parser.add_argument("--eff-b", type=float, default=0.85)
    parser.add_argument("--mistag-c-to-b", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.0 < args.min_survival <= args.baseline_survival <= args.max_survival:
        raise ValueError("Require min survival <= baseline <= max survival")
    if args.survival_points < 2 or args.tag_points < 2:
        raise ValueError("Every scan needs at least two points")
    run_normalization_scans(args.result_dir, args)


if __name__ == "__main__":
    main()
