#!/usr/bin/env python3
"""Train and evaluate a prepared central-event H(cc) dataset."""

import argparse
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.training import train_and_evaluate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=str(SCRIPT_DIR / "data" / "fsr_mtd_exclusive_gg_full")
    )
    parser.add_argument(
        "--result-dir", default=str(SCRIPT_DIR / "results" / "fsr_mtd_exclusive_gg")
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--nonexclusive-train-cap", type=int, default=250000)
    parser.add_argument("--stop-cap", type=int, default=40000)
    parser.add_argument("--grid-cells", type=int, default=256)
    parser.add_argument("--features", nargs="+", default=None,
                        help="Train on this subset of the stored features")
    parser.add_argument("--evaluation-chunk", type=int, default=20000)
    parser.add_argument("--intensity-chunk", type=int, default=100000)
    parser.add_argument("--batch-rows", type=int, default=250000)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--monitor-rounds", type=int, default=50)
    parser.add_argument("--support-floor", type=float, default=50.0)
    parser.add_argument("--score-bins", type=int, default=320)
    parser.add_argument("--score-min", type=float, default=-30.0)
    parser.add_argument("--score-max", type=float, default=30.0)
    parser.add_argument("--scan-points", type=int, default=96)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--vertex-bins", type=int, default=8)
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7)
    parser.add_argument("--pps-time-ps", type=float, default=10.0)
    parser.add_argument("--pv-time-ps", type=float, default=7.1)
    parser.add_argument("--pv-z-resolution-cm", type=float, default=0.001)
    parser.add_argument("--pps-time-scan", type=float, nargs="+", default=(5.0, 10.0, 20.0, 30.0))
    parser.add_argument(
        "--pv-time-scan", type=float, nargs="+", default=(7.1, 10.0, 20.0, 30.0)
    )
    parser.add_argument(
        "--require-truth-matched", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--skip-hash", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    positive = (
        "n_estimators", "nonexclusive_train_cap", "stop_cap", "grid_cells",
        "evaluation_chunk", "intensity_chunk", "batch_rows", "jobs", "score_bins",
        "scan_points", "ladder_bins", "vertex_bins",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("All count and chunk arguments must be positive")
    if args.score_min >= args.score_max:
        raise ValueError("--score-min must be below --score-max")
    args.data_dir = Path(args.data_dir).resolve()
    args.result_dir = Path(args.result_dir).resolve()
    with open(args.data_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    selection = metadata["selection"]
    required_by_dataset = bool(selection["require_truth_matched_at_training"])
    if args.require_truth_matched == "auto":
        args.require_truth_matched = required_by_dataset
    else:
        args.require_truth_matched = args.require_truth_matched == "on"
    if required_by_dataset and not args.require_truth_matched:
        raise ValueError("A truth-jet dataset must be trained and evaluated with truth_matched required")
    args.mass_window = tuple(float(value) for value in selection["mass_window_gev"])
    args.max_delta_y = float(selection["max_abs_rapidity_difference"])
    args.pileup_mu = float(metadata["protons"].get("pileup_mu", 200.0))
    with open(args.data_dir / "channel_config.yaml", encoding="utf-8") as handle:
        channel = yaml.safe_load(handle)
    args.pps_config = channel["pps_config"]
    args.repo = REPO
    train_and_evaluate(args.data_dir, args.result_dir, args)


if __name__ == "__main__":
    main()
