#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import analyzer
import plot_random_protons


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count random minbias BXs with smeared PPS proton-pair masses in 100-150 GeV."
    )
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file with beam.sqrt_s_gev, pps.xi_ranges, pps.xi_res, and random.seed.",
    )
    parser.add_argument(
        "--minbias-campaign",
        default=None,
        help="Minbias campaign key. Defaults to config.yaml minbias.default_campaign.",
    )
    parser.add_argument(
        "--minbias-input",
        default=None,
        help="Override minbias input .npz/.parquet file or directory. Defaults to the minbias campaign directory.",
    )
    parser.add_argument(
        "--max-minbias-files",
        type=int,
        default=None,
        help="Maximum number of minbias .npz/.parquet files to load.",
    )
    parser.add_argument("--mu", type=float, default=200.0, help="Mean interactions per synthetic BX")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to random.seed from --pps-config.")
    parser.add_argument("--max-bx", type=int, default=None, help="Maximum number of synthetic BXs to build.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_minbias_files is not None and args.max_minbias_files <= 0:
        raise RuntimeError("--max-minbias-files must be > 0")
    if args.mu <= 0.0:
        raise RuntimeError("--mu must be > 0")
    if args.max_bx is not None and args.max_bx <= 0:
        raise RuntimeError("--max-bx must be > 0")

    analyzer.ensure_analysis_runtime(Path(__file__), sys.argv[1:])
    _ak, np, _plt, _uproot = analyzer.import_libraries()
    pps_config = analyzer.load_pps_config(analyzer.resolve_path(args.pps_config, base=analyzer.ROOT))
    pps_config["mu"] = float(args.mu)

    minbias = plot_random_protons.load_minbias(np, args, pps_config)
    rng = np.random.default_rng(args.seed if args.seed is not None else pps_config["seed"])

    bx_with_pairs = 0
    while args.max_bx is None or minbias["bx_built"] < args.max_bx:
        pairs = plot_random_protons.random_bx_pairs(np, minbias, pps_config, rng)
        if pairs is None:
            break
        if pairs["mx"].size:
            bx_with_pairs += 1

    total_bx = minbias["bx_built"]
    total_pairs = minbias["pairs_kept"]
    multi_pair_bx = minbias["multi_pair_bx"]
    fraction = bx_with_pairs / total_bx if total_bx else 0.0
    multi_fraction = multi_pair_bx / total_bx if total_bx else 0.0

    print(f"Minbias campaign: {minbias['campaign']}")
    print(f"Files loaded: {len(minbias['files'])}")
    print(f"Synthetic BX built: {total_bx}")
    print(f"Interactions consumed: {minbias['interactions_consumed']}")
    print(f"BX with >=1 pair in 100-150 GeV: {bx_with_pairs} ({fraction:.6g})")
    print(f"BX with >1 pair in 100-150 GeV: {multi_pair_bx} ({multi_fraction:.6g})")
    print(f"Total pairs in 100-150 GeV: {total_pairs}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
