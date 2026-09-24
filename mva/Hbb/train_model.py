#!/usr/bin/env python3
"""Train and evaluate a prepared central-event H(bb) dataset."""

import argparse
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.training import train_and_evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "data" / "nominal_three_class"))
    parser.add_argument("--result-dir", default=str(SCRIPT_DIR / "results" / "nominal_three_class"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--nonexclusive-train-cap", type=int, default=250000)
    parser.add_argument("--stop-cap", type=int, default=40000)
    parser.add_argument("--grid-cells", type=int, default=256)
    parser.add_argument("--features", nargs="+", default=None,
                        help="Train on this subset of the stored features")
    parser.add_argument(
        "--feature-set",
        help="Named feature set from the dataset's channel_config.yaml",
    )
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
    parser.add_argument("--require-truth-matched", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    count_names = (
        "n_estimators", "nonexclusive_train_cap", "stop_cap", "grid_cells",
        "evaluation_chunk", "intensity_chunk", "batch_rows", "jobs", "score_bins",
        "scan_points", "ladder_bins", "vertex_bins",
    )
    if any(getattr(args, name) <= 0 for name in count_names):
        raise ValueError("All count and chunk arguments must be positive")
    args.data_dir = Path(args.data_dir).resolve()
    args.result_dir = Path(args.result_dir).resolve()
    if args.features and args.feature_set:
        raise ValueError("Use either --features or --feature-set, not both")
    with open(args.data_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    required = bool(metadata["selection"]["require_truth_matched_at_training"])
    args.require_truth_matched = required if args.require_truth_matched == "auto" else args.require_truth_matched == "on"
    if required and not args.require_truth_matched:
        raise ValueError("A truth-jet dataset requires truth_matched during training and evaluation")
    args.mass_window = tuple(metadata["selection"]["mass_window_gev"])
    args.max_delta_y = float(metadata["selection"]["max_abs_rapidity_difference"])
    args.pileup_mu = float(metadata["protons"].get("pileup_mu", 200.0))
    with open(args.data_dir / "channel_config.yaml", encoding="utf-8") as handle:
        channel_config = yaml.safe_load(handle)
    if args.feature_set:
        try:
            args.features = list(channel_config["feature_sets"][args.feature_set])
        except KeyError as error:
            raise ValueError(f"Unknown feature set: {args.feature_set}") from error
    args.pps_config = channel_config["pps_config"]
    args.repo = REPO
    train_and_evaluate(args.data_dir, args.result_dir, args)


if __name__ == "__main__":
    main()
