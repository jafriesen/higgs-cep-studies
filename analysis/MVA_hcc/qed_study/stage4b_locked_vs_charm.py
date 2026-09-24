#!/usr/bin/env python3
"""Stage 4b: the H(bb)-derived locked 20 against the charm-ranked top 20.

Nine of the twenty locked features do not appear in the charm permutation
ranking's top twenty. This compares the two sets head to head at identical
settings, which stage 4's feature-count scan cannot do because it only ever
uses charm-ranked prefixes.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.common import LOCKED_FEATURES  # noqa: E402
from analysis.MVA_hcc.qed_study.common import DEFAULT_DATA, DEFAULT_OUTPUT, write_yaml  # noqa: E402
from analysis.MVA_hcc.qed_study import harness  # noqa: E402

CLASS_MAP = (0, 1, 1, 2)
CLASS_NAMES = ("Hcc", "exclusive_continuum", "nonexclusive_QCD")
PAIR_FEATURE = "yx_minus_dijet_rapidity"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "cc_wide"))
    parser.add_argument("--stage4-dir", default=str(DEFAULT_OUTPUT / "stage4"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage4b"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--grid-cells", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--stop-cap", type=int, default=40000)
    parser.add_argument("--ladder-bins", type=int, default=6)
    return parser.parse_args()


def evaluate(data_dir, features, args, offset):
    evaluation = harness.build_evaluation(
        data_dir, CLASS_MAP, CLASS_NAMES, seed=args.seed + 1000 * offset,
        grid_cells=args.grid_cells, n_estimators=args.n_estimators,
        feature_names=features, stop_cap=args.stop_cap, verbose=False,
    )
    row_score, cell_score = harness.scores_from(evaluation)
    edges = harness.signal_quantile_edges(row_score, evaluation, args.ladder_bins)
    mass, effective = harness.ladder_mass(evaluation, row_score, cell_score, edges)
    total, _per = harness.significance(mass)
    del evaluation
    return total, float(effective.min())


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(args.stage4_dir) / "stage4_report.yaml", encoding="utf-8") as handle:
        ranked = yaml.safe_load(handle)["ranked_features"]

    sets = {
        "locked_20_from_Hbb": list(LOCKED_FEATURES),
        "charm_ranked_20": ranked[:20],
        "union": sorted(set(LOCKED_FEATURES) | set(ranked[:20])),
    }
    for name, features in sets.items():
        if PAIR_FEATURE not in features:
            raise RuntimeError(f"{name} lacks the proton observable {PAIR_FEATURE}")

    results = {}
    for name, features in sets.items():
        values, supports = [], []
        for offset in range(args.seeds):
            significance, support = evaluate(args.data_dir, features, args, offset)
            values.append(significance)
            supports.append(support)
            print(f"  {name:22s} seed+{offset}: Z={significance:.4f} min neff={support:.0f}",
                  flush=True)
        array = np.asarray(values)
        results[name] = {
            "n_features": len(features),
            "features": features,
            "significances": values,
            "mean_significance": float(array.mean()),
            "seed_spread": float(array.std(ddof=1)) if array.size > 1 else None,
            "min_effective_pooled_events": supports,
        }
    print()
    for name, block in results.items():
        spread = block["seed_spread"]
        print(f"{name:22s} n={block['n_features']:3d} "
              f"Z={block['mean_significance']:.4f} +/- "
              f"{spread:.4f}" if spread else "", flush=True)

    write_yaml(output_dir / "stage4b_report.yaml", {
        "description": "Locked H(bb) feature set against the charm-ranked set, matched settings",
        "data_dir": str(Path(args.data_dir).resolve()),
        "class_names": list(CLASS_NAMES),
        "ladder_bins": args.ladder_bins,
        "grid_cells": args.grid_cells,
        "seeds": args.seeds,
        "hyperparameters": {**harness.PRODUCTION_PARAMS, "n_estimators": args.n_estimators},
        "results": results,
        "locked_not_in_charm_top20": sorted(set(LOCKED_FEATURES) - set(ranked[:20])),
        "charm_top20_not_in_locked": sorted(set(ranked[:20]) - set(LOCKED_FEATURES)),
        "runtime_seconds": time.perf_counter() - started,
    })
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
