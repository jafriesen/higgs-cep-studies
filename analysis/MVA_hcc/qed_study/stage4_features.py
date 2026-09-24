#!/usr/bin/env python3
"""Stage 4: which variables the charm channel actually needs.

The 20 locked features were selected from an H(bb) permutation ranking against
an H(bb) MadGraph sample. This stage redoes the ranking for charm, targeting the
non-exclusive MadGraph QCDcc background, and runs a nested feature-count scan
with the full pool-integrated metric.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.common import plugin_score  # noqa: E402
from analysis.MVA_hcc.qed_study.common import (  # noqa: E402
    DEFAULT_DATA, DEFAULT_OUTPUT, MASS_BINS, write_yaml,
)
from analysis.MVA_hcc.qed_study import harness  # noqa: E402

CLASS_MAP = (0, 1, 1, 2)
CLASS_NAMES = ("Hcc", "exclusive_continuum", "nonexclusive_QCD")
PAIR_FEATURE = "yx_minus_dijet_rapidity"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "cc_wide"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage4"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--grid-cells", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--train-cap", type=int, default=250000)
    parser.add_argument("--stop-cap", type=int, default=40000)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20, 30, 55])
    parser.add_argument("--rank-rows", type=int, default=1200000)
    parser.add_argument("--rank-ladder-bins", type=int, default=4)
    return parser.parse_args()


def row_level_ladder_z(evaluation, row_score, n_bins):
    """Ladder Z from the sampled-pair rows only.

    A cheap stand-in for the pool-integrated metric, used to rank features. It
    keeps the same weights and mass bins, so relative comparisons are fair, but
    it does not replace the integrated metric used for the size scan.
    """
    signal_rows = evaluation["component"] == 0
    edges = np.unique(np.r_[
        -np.inf,
        np.quantile(row_score[signal_rows & np.isfinite(row_score)],
                    np.linspace(0.0, 1.0, n_bins + 1)[1:-1]),
        np.inf,
    ])
    category = harness.bin_index(row_score, edges)
    total = 0.0
    for cell in range(edges.size - 1):
        keep = category == cell
        signal, _ = np.histogram(evaluation["mx"][keep & signal_rows], MASS_BINS,
                                 weights=evaluation["weight"][keep & signal_rows])
        background, _ = np.histogram(evaluation["mx"][keep & ~signal_rows], MASS_BINS,
                                     weights=evaluation["weight"][keep & ~signal_rows])
        combined = signal + background
        total += np.sum(np.divide(signal * signal, combined,
                                  out=np.zeros_like(signal), where=combined > 0.0))
    return float(np.sqrt(total))


def permutation_ranking(data_dir, features, args):
    """Drop in the row-level ladder Z when each feature alone is shuffled."""
    evaluation = harness.build_evaluation(
        data_dir, CLASS_MAP, CLASS_NAMES, seed=args.seed, grid_cells=2,
        cap=args.train_cap, n_estimators=args.n_estimators, stop_cap=args.stop_cap,
    )
    matrix = harness.read_columns(
        harness.load_dataset(data_dir, mmap=True)["x"],
        np.arange(len(features)),
    )
    rng = np.random.default_rng(args.seed)
    rows = np.arange(matrix.shape[0])
    scale = 1.0
    exclusive = np.flatnonzero(~evaluation["is_pooled"])
    pooled = np.flatnonzero(evaluation["is_pooled"])
    if pooled.size > args.rank_rows:
        drawn = rng.choice(pooled, size=args.rank_rows, replace=False)
        rows = np.sort(np.r_[exclusive, drawn])
        # Uniform row subsampling: the retained pooled rows have to carry the
        # weight of the ones dropped, or the pooled yield is understated by the
        # sampling fraction and the metric becomes meaningless.
        scale = pooled.size / drawn.size
    block = {key: evaluation[key][rows] for key in ("component", "mx", "weight", "folds")}
    block["weight"] = block["weight"].copy()
    block["weight"][evaluation["is_pooled"][rows]] *= scale
    subset = {**evaluation, **block, "is_pooled": evaluation["is_pooled"][rows]}
    print(f"  ranking on {rows.size:,} rows (pooled weight scale {scale:.3f})", flush=True)

    def score_of(sample):
        output = np.full(sample.shape[0], np.nan)
        for fold, (model, calibrator) in enumerate(evaluation["fits"]):
            select = block["folds"] == fold
            probability = harness.calibrated_probabilities(
                model, calibrator, sample[select], len(CLASS_NAMES)
            )
            output[select] = plugin_score(probability, evaluation["kappas"])
        return output

    sample = matrix[rows]
    baseline = row_level_ladder_z(subset, score_of(sample), args.rank_ladder_bins)
    ranking = []
    for index, name in enumerate(features):
        keep = sample[:, index].copy()
        sample[:, index] = keep[rng.permutation(keep.size)]
        shuffled = row_level_ladder_z(subset, score_of(sample), args.rank_ladder_bins)
        sample[:, index] = keep
        ranking.append({"feature": name, "impact": baseline - shuffled})
        print(f"  {name:38s} impact={baseline - shuffled:+.5f}", flush=True)
    ranking.sort(key=lambda item: -item["impact"])
    return baseline, ranking


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = harness.load_dataset(args.data_dir, mmap=True)["features"]
    print(f"ranking {len(features)} features", flush=True)
    baseline, ranking = permutation_ranking(args.data_dir, features, args)
    ordered = [item["feature"] for item in ranking]
    if PAIR_FEATURE not in ordered[:min(args.sizes)]:
        ordered = [PAIR_FEATURE] + [name for name in ordered if name != PAIR_FEATURE]
    with open(output_dir / "permutation_ranking.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature", "impact"])
        for rank, item in enumerate(ranking):
            writer.writerow([rank, item["feature"], item["impact"]])

    scan = {}
    for size in args.sizes:
        selected = ordered[:size]
        values = []
        for offset in range(args.seeds):
            evaluation = harness.build_evaluation(
                args.data_dir, CLASS_MAP, CLASS_NAMES, seed=args.seed + 1000 * offset,
                grid_cells=args.grid_cells, cap=args.train_cap,
                n_estimators=args.n_estimators, feature_names=selected,
                stop_cap=args.stop_cap, verbose=False,
            )
            row_score, cell_score = harness.scores_from(evaluation)
            edges = harness.signal_quantile_edges(row_score, evaluation, args.ladder_bins)
            mass, effective = harness.ladder_mass(evaluation, row_score, cell_score, edges)
            total, _per = harness.significance(mass)
            values.append({"seed_offset": offset, "significance": total,
                           "min_effective_pooled_events": float(effective.min())})
            print(f"  {size:3d} features seed+{offset}: Z={total:.4f} "
                  f"min neff={effective.min():.0f}", flush=True)
            del evaluation
        significances = np.asarray([item["significance"] for item in values])
        scan[size] = {
            "features": selected,
            "runs": values,
            "mean_significance": float(significances.mean()),
            "seed_spread": float(significances.std(ddof=1)) if significances.size > 1 else None,
        }

    write_yaml(output_dir / "stage4_report.yaml", {
        "description": "Charm-specific feature ranking and nested feature-count scan",
        "data_dir": str(Path(args.data_dir).resolve()),
        "class_map": list(CLASS_MAP),
        "class_names": list(CLASS_NAMES),
        "ranking_baseline_row_level_z": baseline,
        "ranking": ranking,
        "ranked_features": ordered,
        "feature_count_scan": scan,
        "grid_cells": args.grid_cells,
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "note": "Ranking uses a row-level ladder metric for affordability; the size "
                "scan uses the pool-integrated metric. Judge sizes against the seed "
                "spread, not a bootstrap sigma.",
    })
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
