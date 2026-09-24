#!/usr/bin/env python3
"""Optimize the multiclass proton MVA architecture on a cached dataset.

Two questions, one script:
  1. which XGBoost hyperparameters are actually best, and
  2. how many of the 51 nonconstant input variables are actually needed.

Neither has ever been answered on cache_v4_all_campaigns: the configuration
used by significance_v3_all_campaigns was inherited through two --quick-
significance runs from a study performed on a different cache, and that study
selected its hyperparameters while evaluating on the same split it used for
early stopping and calibration.

Selection metric: the mass-binned significance evaluated over a ladder of
score bins rather than a single threshold,
    Z = sqrt( sum_scorebins sum_massbins s^2 / (s + b) ).
Bin edges come from signal-score quantiles only, so no background-driven
selection enters the binning and the per-model score scale divides out. This
avoids the threshold scan, whose optimum pins itself to the MC-statistics
safety boundary and is correspondingly noisy. Every result carries
min_madgraph_groups_per_bin so an unsupported number is visible as such.

Split hygiene (the point of the exercise):
  split 0 train | split 1 early stopping + calibration | split 2 evaluation
  split 3 is read exactly once, for the winning configuration only.

The cache is memory-mapped and each split is strided through the whole file,
so every split is read into RAM once up front rather than re-read per fit.

Run under the analysis environment:
  source setup_env.sh
  python3 analysis/MVA/study_mva_architecture.py \
      --cache-dir analysis/MVA/output/mva_bb_multiclass_protons/cache_v4_all_campaigns \
      --output-dir analysis/MVA/output/mva_bb_multiclass_protons/architecture_study_v1
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.MVA.optimize_dijet_mva_multiclass_protons import (  # noqa: E402
    CONSTANT_FEATURES,
    MASS_BINS,
    assert_group_safe,
    cap_groups,
    class_balanced_weights,
    correlation_screen,
    development_indices,
    load_cache,
    physical_kappas,
    plugin_score,
    split_weight,
)

# Hyperparameters of the configuration currently in production, used as the
# stage 1/2 baseline so feature ranking happens at a known-good working point
# and as configuration 0 of the search so every result has a reference.
BASELINE_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "max_bin": 256,
}
N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 40
BATCH_ROWS = 250000
SECONDARY_LADDER_BINS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--ladder-bins", type=int, default=6,
        help="Score bins in the selection metric. Six keeps a usable number "
        "of MadGraph groups per bin on split 2 alone; the ten-bin value is "
        "also recorded for every configuration.",
    )
    parser.add_argument(
        "--madgraph-train-cap", type=int, default=60000,
        help="MadGraph hard-event groups used for training during stages 1-3.",
    )
    parser.add_argument(
        "--madgraph-confirm-cap", type=int, default=250000,
        help="MadGraph training groups for the stage 4 confirmation refits.",
    )
    parser.add_argument(
        "--madgraph-stop-cap", type=int, default=150000,
        help="MadGraph groups kept in the early-stopping/calibration split "
        "during stages 1-3. Stage 4 uses the full split.",
    )
    parser.add_argument("--max-configs", type=int, default=40)
    parser.add_argument("--bootstrap-replicas", type=int, default=200)
    parser.add_argument(
        "--nested-sizes", default="6,10,15,20,25,30,40",
        help="Comma-separated feature-set sizes to scan; the full survivor "
        "set is always appended.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse any stage checkpoint already present in --output-dir.",
    )
    parser.add_argument("--stages", default="0,1,2,3,4")
    return parser.parse_args()


# --------------------------------------------------------------------------
# model construction
#
# optimize_dijet_mva_multiclass_protons.make_model hardcodes n_estimators,
# subsample, colsample_bytree, max_bin and early_stopping_rounds, so passing
# any of them through **params raises TypeError. This study varies them, hence
# a local builder.
# --------------------------------------------------------------------------
def build_model(params, seed):
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=16,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=seed,
        **params,
    )


class RowBlock:
    """One split held in RAM. The cache is memory-mapped and every split is
    strided through the whole 3.2 GB file, so reading each split once here
    turns a per-fit I/O cost into a one-off."""

    def __init__(self, data, indices, name):
        started = time.perf_counter()
        self.name = name
        self.indices = indices
        self.x = np.empty((indices.size, data["x"].shape[1]), dtype=np.float32)
        for start in range(0, indices.size, BATCH_ROWS):
            stop = min(start + BATCH_ROWS, indices.size)
            self.x[start:stop] = data["x"][indices[start:stop]]
        self.classes = np.asarray(data["class"][indices])
        self.groups = np.asarray(data["group_id"][indices])
        self.mixture = np.asarray(
            data["training_mixture_weight"][indices], dtype=np.float64
        )
        self.mass = np.asarray(data["mx"][indices], dtype=np.float64)
        self.weights = split_weight(data, indices)
        print(
            f"  loaded {name}: {indices.size} rows "
            f"({self.x.nbytes / 1e6:.0f} MB, {time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    def columns(self, features):
        return np.ascontiguousarray(self.x[:, features])


def fit_calibrated(train, stop, features, params, seed):
    train_weights = class_balanced_weights(train.classes, train.mixture)
    stop_weights = class_balanced_weights(stop.classes, stop.mixture)
    stop_x = stop.columns(features)
    model = build_model(params, seed)
    model.fit(
        train.columns(features), train.classes,
        sample_weight=train_weights,
        eval_set=[(stop_x, stop.classes)],
        sample_weight_eval_set=[stop_weights],
        verbose=False,
    )
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
    calibrator.fit(
        model.predict(stop_x, output_margin=True), stop.classes,
        sample_weight=stop_weights,
    )
    return model, calibrator


def calibrated_probabilities(model, calibrator, matrix):
    """Batched so the margin and probability buffers stay small."""
    out = np.empty((matrix.shape[0], 3), dtype=np.float64)
    for start in range(0, matrix.shape[0], BATCH_ROWS):
        stop = min(start + BATCH_ROWS, matrix.shape[0])
        margins = model.predict(matrix[start:stop], output_margin=True)
        out[start:stop] = calibrator.predict_proba(margins)
    return out


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------
def ladder_edges(score, classes, n_bins):
    signal = score[(classes == 0) & np.isfinite(score)]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.r_[-np.inf, np.quantile(signal, quantiles), np.inf])


def ladder_significance(classes, mass, weights, score, edges, groups=None):
    """Mass-binned significance summed over a fixed ladder of score bins."""
    finite = np.isfinite(score) & np.isfinite(mass)
    signal = classes == 0
    z2 = 0.0
    support = []
    for index in range(edges.size - 1):
        inside = finite & (score >= edges[index]) & (score < edges[index + 1])
        in_signal = inside & signal
        in_background = inside & ~signal
        s, _ = np.histogram(mass[in_signal], MASS_BINS, weights=weights[in_signal])
        b, _ = np.histogram(
            mass[in_background], MASS_BINS, weights=weights[in_background]
        )
        total = s + b
        z2 += np.sum(
            np.divide(s * s, total, out=np.zeros_like(s), where=total > 0.0)
        )
        if groups is not None:
            support.append(int(np.unique(groups[inside & (classes == 2)]).size))
    return float(np.sqrt(z2)), support


def bootstrap_sigma(block, score, edges, replicas, seed):
    """Group-level multinomial resampling at fixed bin edges. The offset of
    the median from the point estimate measures the upward bias of
    sum s^2/(s+b) under per-bin background fluctuations."""
    rng = np.random.default_rng(seed)
    maps = []
    for class_id in range(3):
        rows = np.flatnonzero(block.classes == class_id)
        unique, inverse = np.unique(block.groups[rows], return_inverse=True)
        maps.append((rows, unique.size, inverse))
    values = []
    for _ in range(replicas):
        multiplier = np.zeros(block.classes.size, dtype=np.float64)
        for rows, size, inverse in maps:
            counts = rng.multinomial(size, np.full(size, 1.0 / size))
            multiplier[rows] = counts[inverse]
        values.append(ladder_significance(
            block.classes, block.mass, block.weights * multiplier, score, edges
        )[0])
    low, median, high = np.percentile(np.asarray(values), [16, 50, 84])
    return {
        "replicas": int(replicas),
        "median": float(median),
        "percent_68": [float(low), float(high)],
        "sigma": float(0.5 * (high - low)),
    }


def evaluate(block, model, calibrator, features, kappas, n_bins):
    probabilities = calibrated_probabilities(model, calibrator, block.columns(features))
    score = plugin_score(probabilities, kappas)
    edges = ladder_edges(score, block.classes, n_bins)
    z, support = ladder_significance(
        block.classes, block.mass, block.weights, score, edges, block.groups
    )
    secondary_edges = ladder_edges(score, block.classes, SECONDARY_LADDER_BINS)
    secondary, _ = ladder_significance(
        block.classes, block.mass, block.weights, score, secondary_edges
    )
    return {
        "significance": z,
        f"significance_{SECONDARY_LADDER_BINS}bin": secondary,
        "score_bins": int(edges.size - 1),
        "min_madgraph_groups_per_bin": int(min(support)) if support else 0,
        "best_iteration": int(model.best_iteration),
    }, score, edges


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def load_checkpoint(path, resume):
    if resume and path.is_file():
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return None


def capped_indices(data, split_id, limit, seed):
    """Split indices with the MadGraph class capped by hard-event group.
    Returns the indices and the weight correction restoring the yield."""
    indices = np.flatnonzero(data["split"] == split_id)
    if limit is None:
        return indices, 1.0
    return cap_groups(
        indices, data["class"], data["group_id"], 2, limit,
        np.random.default_rng(seed),
    )


def training_indices(data, cap, seed):
    indices, _correction = development_indices(data, cap, seed)
    # The correction that development_indices returns is a constant factor on
    # the MadGraph training mixture, and class_balanced_weights renormalizes
    # each class total afterwards, so it cancels exactly. Not applied here.
    return indices


def build_blocks(data, args, train_cap, stop_cap):
    train = RowBlock(data, training_indices(data, train_cap, args.seed), "train")
    stop_indices, _ = capped_indices(data, 1, stop_cap, args.seed + 31)
    stop = RowBlock(data, stop_indices, "stop")
    return train, stop


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage0_setup(args, data, output_dir, blocks):
    train, stop, evaluation = blocks
    nonconstant = np.asarray([
        index for index, name in enumerate(data["feature_names"])
        if name not in CONSTANT_FEATURES
    ])
    kappas, yields = physical_kappas(evaluation.classes, evaluation.weights)

    started = time.perf_counter()
    model, calibrator = fit_calibrated(train, stop, nonconstant, BASELINE_PARAMS, args.seed)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics, _score, _edges = evaluate(
        evaluation, model, calibrator, nonconstant, kappas, args.ladder_bins
    )
    eval_seconds = time.perf_counter() - started

    cycle = fit_seconds + eval_seconds
    projected = (
        fit_seconds + eval_seconds * len(nonconstant)          # stage 1
        + cycle * 9                                            # stage 2
        + cycle * (args.max_configs + 1)                       # stage 3
        + cycle * 3.5 * 11                                     # stage 4, bigger caps
    )
    payload = {
        "cache_dir": args.cache_dir,
        "seed": args.seed,
        "ladder_bins": args.ladder_bins,
        "nonconstant_features": data["feature_names"][nonconstant].tolist(),
        "nonconstant_feature_indices": nonconstant.tolist(),
        "constant_features": sorted(CONSTANT_FEATURES),
        "rows": {
            "train": int(train.indices.size),
            "stop": int(stop.indices.size),
            "eval": int(evaluation.indices.size),
        },
        "kappas": kappas.tolist(),
        "preselection_yields": yields.tolist(),
        "baseline_params": BASELINE_PARAMS,
        "baseline_metrics": metrics,
        "timing_seconds": {
            "one_fit": fit_seconds,
            "one_evaluation": eval_seconds,
            "projected_total": projected,
        },
    }
    write_yaml(output_dir / "stage0_setup.yaml", payload)
    print(
        f"stage0: fit={fit_seconds:.0f}s eval={eval_seconds:.0f}s "
        f"baseline Z={metrics['significance']:.4f} "
        f"(min MG groups/bin={metrics['min_madgraph_groups_per_bin']}); "
        f"projected total {projected / 3600.0:.2f} h",
        flush=True,
    )
    return payload


def stage1_ranking(args, data, output_dir, setup, blocks):
    train, stop, evaluation = blocks
    nonconstant = np.asarray(setup["nonconstant_feature_indices"], dtype=int)
    kappas = np.asarray(setup["kappas"], dtype=np.float64)

    model, calibrator = fit_calibrated(train, stop, nonconstant, BASELINE_PARAMS, args.seed)
    baseline, _score, edges = evaluate(
        evaluation, model, calibrator, nonconstant, kappas, args.ladder_bins
    )
    print(
        f"stage1: ranking baseline Z={baseline['significance']:.4f} on the full "
        f"evaluation split ({evaluation.indices.size} rows)",
        flush=True,
    )

    matrix = evaluation.columns(nonconstant)
    rng = np.random.default_rng(args.seed + 1)
    impacts = {}
    for local, global_index in enumerate(nonconstant):
        name = str(data["feature_names"][global_index])
        original = matrix[:, local].copy()
        rng.shuffle(matrix[:, local])
        score = plugin_score(
            calibrated_probabilities(model, calibrator, matrix), kappas
        )
        matrix[:, local] = original
        permuted, _ = ladder_significance(
            evaluation.classes, evaluation.mass, evaluation.weights, score, edges
        )
        drop = max(
            (baseline["significance"] - permuted) / baseline["significance"], 0.0
        )
        impacts[name] = {"combined": float(drop), "permuted_significance": permuted}
        print(f"  {name:38s} impact={drop:+.5f}", flush=True)
    del matrix

    survivors, correlated_removed = correlation_screen(
        data, nonconstant, impacts, train.indices
    )
    by_impact = lambda index: impacts[str(data["feature_names"][index])]["combined"]
    ranked = sorted(survivors, key=by_impact, reverse=True)
    with open(output_dir / "feature_ranking.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "feature", "permutation_impact", "survived_correlation_screen"]
        )
        for position, index in enumerate(
            sorted(nonconstant, key=by_impact, reverse=True), start=1
        ):
            name = str(data["feature_names"][index])
            writer.writerow(
                [position, name, impacts[name]["combined"], index in survivors]
            )

    payload = {
        "baseline_metrics": baseline,
        "ranking_rows": int(evaluation.indices.size),
        "permutation_impact": impacts,
        "correlation_removed": correlated_removed,
        "ranked_feature_indices": [int(index) for index in ranked],
        "ranked_features": [str(data["feature_names"][index]) for index in ranked],
    }
    write_yaml(output_dir / "stage1_ranking.yaml", payload)
    print(
        f"stage1: {len(nonconstant)} -> {len(ranked)} after correlation screen "
        f"(removed {len(correlated_removed)})",
        flush=True,
    )
    return payload


def stage2_feature_sets(args, data, output_dir, setup, ranking, blocks):
    train, stop, evaluation = blocks
    kappas = np.asarray(setup["kappas"], dtype=np.float64)
    ranked = np.asarray(ranking["ranked_feature_indices"], dtype=int)

    sizes = sorted({
        size for size in (int(value) for value in args.nested_sizes.split(","))
        if 0 < size < ranked.size
    } | {int(ranked.size)})
    results = []
    best_score, best_edges, best_value = None, None, -1.0
    for size in sizes:
        features = ranked[:size]
        model, calibrator = fit_calibrated(
            train, stop, features, BASELINE_PARAMS, args.seed + size
        )
        metrics, score, edges = evaluate(
            evaluation, model, calibrator, features, kappas, args.ladder_bins
        )
        results.append({
            "size": int(size),
            "features": [str(data["feature_names"][i]) for i in features],
            "feature_indices": features.tolist(),
            **metrics,
        })
        if metrics["significance"] > best_value:
            best_value = metrics["significance"]
            best_score, best_edges = score, edges
        print(
            f"stage2: {size:3d} features Z={metrics['significance']:.4f} "
            f"(10-bin {metrics['significance_10bin']:.4f}, "
            f"min MG groups/bin={metrics['min_madgraph_groups_per_bin']}, "
            f"iter={metrics['best_iteration']})",
            flush=True,
        )

    best = max(results, key=lambda item: item["significance"])
    uncertainty = bootstrap_sigma(
        evaluation, best_score, best_edges, args.bootstrap_replicas, args.seed + 501
    )
    threshold = best["significance"] - uncertainty["sigma"]
    selected = min(
        (item for item in results if item["significance"] >= threshold),
        key=lambda item: item["size"],
    )

    _plot_feature_scan(results, selected, uncertainty, output_dir)
    payload = {
        "sizes": sizes,
        "results": results,
        "best": {"size": best["size"], "significance": best["significance"]},
        "bootstrap_at_best": uncertainty,
        "selection_rule": "smallest feature set within one bootstrap sigma of the best",
        "selection_threshold": float(threshold),
        "selected": selected,
    }
    write_yaml(output_dir / "stage2_feature_sets.yaml", payload)
    print(
        f"stage2: best Z={best['significance']:.4f} at {best['size']} features, "
        f"sigma={uncertainty['sigma']:.4f}; selected {selected['size']} features "
        f"(Z={selected['significance']:.4f})",
        flush=True,
    )
    return payload


def sample_params(rng):
    return {
        "max_depth": int(rng.choice([3, 4, 5, 6, 8])),
        "learning_rate": round(float(np.exp(rng.uniform(np.log(0.02), np.log(0.25)))), 4),
        "min_child_weight": int(rng.choice([1, 5, 20, 50, 200])),
        "subsample": float(rng.choice([0.6, 0.8, 1.0])),
        "colsample_bytree": float(rng.choice([0.5, 0.7, 0.9])),
        "reg_lambda": round(float(np.exp(rng.uniform(np.log(0.5), np.log(50.0)))), 3),
        "gamma": float(rng.choice([0.0, 0.5, 2.0])),
        "max_bin": int(rng.choice([256, 512])),
    }


def stage3_search(args, data, output_dir, setup, feature_sets, blocks):
    train, stop, evaluation = blocks
    kappas = np.asarray(setup["kappas"], dtype=np.float64)
    features = np.asarray(feature_sets["selected"]["feature_indices"], dtype=int)

    checkpoint = output_dir / "stage3_search.yaml"
    existing = load_checkpoint(checkpoint, args.resume) or {}
    results = existing.get("results", [])
    rng = np.random.default_rng(args.seed + 77)
    candidates = [dict(BASELINE_PARAMS)] + [
        sample_params(rng) for _ in range(args.max_configs)
    ]
    for position, params in enumerate(candidates):
        if position < len(results):
            continue
        started = time.perf_counter()
        model, calibrator = fit_calibrated(train, stop, features, params, args.seed)
        metrics, _score, _edges = evaluate(
            evaluation, model, calibrator, features, kappas, args.ladder_bins
        )
        results.append({
            "index": position,
            "params": params,
            **metrics,
            "seconds": time.perf_counter() - started,
        })
        write_yaml(checkpoint, {"features": features.tolist(), "results": results})
        print(
            f"stage3: [{position:2d}/{len(candidates) - 1}] "
            f"Z={metrics['significance']:.4f} "
            f"d{params['max_depth']} lr={params['learning_rate']:.3f} "
            f"mcw={params['min_child_weight']} sub={params['subsample']} "
            f"col={params['colsample_bytree']} lam={params['reg_lambda']:.2f} "
            f"g={params['gamma']} bin={params['max_bin']} "
            f"({results[-1]['seconds']:.0f}s)",
            flush=True,
        )

    ordered = sorted(results, key=lambda item: item["significance"], reverse=True)
    _plot_search(results, output_dir)
    payload = {
        "features": features.tolist(),
        "feature_names": [str(data["feature_names"][i]) for i in features],
        "results": results,
        "top": ordered[:5],
    }
    write_yaml(checkpoint, payload)
    print(f"stage3: best Z={ordered[0]['significance']:.4f} {ordered[0]['params']}", flush=True)
    return payload


def stage4_confirm(args, data, output_dir, setup, search, evaluation):
    kappas = np.asarray(setup["kappas"], dtype=np.float64)
    features = np.asarray(search["features"], dtype=int)
    nonconstant = np.asarray(setup["nonconstant_feature_indices"], dtype=int)
    train, stop = build_blocks(data, args, args.madgraph_confirm_cap, None)

    ordered = sorted(search["results"], key=lambda item: item["significance"], reverse=True)
    confirmations = []
    for rank, candidate in enumerate(ordered[:3]):
        values = []
        for offset in (0, 1, 2):
            model, calibrator = fit_calibrated(
                train, stop, features, candidate["params"], args.seed + 900 + offset
            )
            metrics, _score, _edges = evaluate(
                evaluation, model, calibrator, features, kappas, args.ladder_bins
            )
            values.append(metrics["significance"])
            print(f"stage4: rank {rank} seed {offset} Z={values[-1]:.4f}", flush=True)
        confirmations.append({
            "rank": rank,
            "params": candidate["params"],
            "search_significance": candidate["significance"],
            "confirmed_significance_per_seed": [float(value) for value in values],
            "confirmed_mean": float(np.mean(values)),
            "confirmed_seed_spread": float(np.std(values)),
        })

    winner = max(confirmations, key=lambda item: item["confirmed_mean"])
    model, calibrator = fit_calibrated(
        train, stop, features, winner["params"], args.seed + 900
    )
    metrics, score, edges = evaluate(
        evaluation, model, calibrator, features, kappas, args.ladder_bins
    )
    uncertainty = bootstrap_sigma(
        evaluation, score, edges, args.bootstrap_replicas, args.seed + 777
    )

    baseline_model, baseline_calibrator = fit_calibrated(
        train, stop, nonconstant, BASELINE_PARAMS, args.seed + 900
    )
    baseline_metrics, baseline_score, baseline_edges = evaluate(
        evaluation, baseline_model, baseline_calibrator, nonconstant, kappas,
        args.ladder_bins,
    )
    baseline_uncertainty = bootstrap_sigma(
        evaluation, baseline_score, baseline_edges, args.bootstrap_replicas,
        args.seed + 778,
    )

    test_block = RowBlock(data, np.flatnonzero(data["split"] == 3), "test")
    test_metrics, _tscore, _tedges = evaluate(
        test_block, model, calibrator, features, kappas, args.ladder_bins
    )
    model.get_booster().save_model(output_dir / "architecture_winner.json")

    payload = {
        "features": [str(data["feature_names"][i]) for i in features],
        "feature_indices": features.tolist(),
        "feature_count": int(features.size),
        "baseline_feature_count": int(nonconstant.size),
        "baseline_params": BASELINE_PARAMS,
        "baseline_metrics": baseline_metrics,
        "baseline_bootstrap": baseline_uncertainty,
        "confirmations": confirmations,
        "winner_params": winner["params"],
        "winner_metrics": metrics,
        "winner_bootstrap": uncertainty,
        "untouched_test_split": test_metrics,
        "madgraph_train_cap": args.madgraph_confirm_cap,
        "note": (
            "Classifier-only optimization. Z is stat-only with a perfectly "
            "known background and does not address the combinatorial "
            "acceptance factor, which remains the dominant uncertainty on the "
            "physics result."
        ),
    }
    write_yaml(output_dir / "summary_architecture.yaml", payload)
    print(
        f"stage4: winner Z={metrics['significance']:.4f} "
        f"+/- {uncertainty['sigma']:.4f} with {features.size} features; "
        f"baseline ({nonconstant.size} features, production params) "
        f"Z={baseline_metrics['significance']:.4f} "
        f"+/- {baseline_uncertainty['sigma']:.4f}; "
        f"untouched test split Z={test_metrics['significance']:.4f}",
        flush=True,
    )
    return payload


def stage5_feature_size_seeds(args, data, output_dir, setup, ranking, summary, blocks):
    """Re-scan the feature-set size with seed averaging, at the tuned working
    point.

    Stage 2 scans one seed per size and selects against the bootstrap sigma,
    which only measures MC resampling at a fixed model. Refitting on a
    different feature set also changes the model, and that variation turned
    out to be larger: the single-seed stage 2 scan scattered by +/-0.3 while
    its bootstrap sigma was 0.19. Feature count therefore has to be decided
    against the seed-to-seed spread, and at the tuned hyperparameters rather
    than the untuned baseline.
    """
    train, stop, evaluation = blocks
    kappas = np.asarray(setup["kappas"], dtype=np.float64)
    ranked = np.asarray(ranking["ranked_feature_indices"], dtype=int)
    params = summary["winner_params"]
    seeds = (args.seed + 900, args.seed + 901, args.seed + 902)

    sizes = sorted({
        size for size in (int(value) for value in args.nested_sizes.split(","))
        if 0 < size < ranked.size
    } | {int(ranked.size)})
    results = []
    for size in sizes:
        features = ranked[:size]
        values, support = [], []
        for seed in seeds:
            model, calibrator = fit_calibrated(train, stop, features, params, seed)
            metrics, _score, _edges = evaluate(
                evaluation, model, calibrator, features, kappas, args.ladder_bins
            )
            values.append(metrics["significance"])
            support.append(metrics["min_madgraph_groups_per_bin"])
        results.append({
            "size": int(size),
            "per_seed": [float(value) for value in values],
            "mean": float(np.mean(values)),
            "seed_spread": float(np.std(values)),
            "min_madgraph_groups_per_bin": int(min(support)),
            "feature_indices": features.tolist(),
            "features": [str(data["feature_names"][i]) for i in features],
        })
        print(
            f"stage5: {size:3d} features Z={np.mean(values):.4f} "
            f"+/- {np.std(values):.4f} (seeds {['%.3f' % v for v in values]}, "
            f"min MG groups/bin={min(support)})",
            flush=True,
        )

    best = max(results, key=lambda item: item["mean"])
    # Compare against the spread of the best point, floored by the typical
    # spread across the scan so a lucky low-variance point cannot set an
    # unreachably tight bar.
    tolerance = max(
        best["seed_spread"],
        float(np.median([item["seed_spread"] for item in results])),
    )
    threshold = best["mean"] - tolerance
    selected = min(
        (item for item in results if item["mean"] >= threshold),
        key=lambda item: item["size"],
    )
    _plot_seed_scan(results, selected, threshold, output_dir)
    payload = {
        "winner_params": params,
        "seeds": list(seeds),
        "results": results,
        "best": {"size": best["size"], "mean": best["mean"]},
        "tolerance": float(tolerance),
        "selection_threshold": float(threshold),
        "selection_rule": (
            "smallest feature set whose 3-seed mean is within one seed spread "
            "of the best"
        ),
        "selected_size": selected["size"],
        "selected_features": selected["features"],
        "selected_feature_indices": selected["feature_indices"],
    }
    write_yaml(output_dir / "stage5_feature_size_seeds.yaml", payload)
    print(
        f"stage5: best {best['size']} features (Z={best['mean']:.4f}); "
        f"selected {selected['size']} features "
        f"(Z={selected['mean']:.4f} +/- {selected['seed_spread']:.4f})",
        flush=True,
    )
    return payload


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _plot_seed_scan(results, selected, threshold, output_dir):
    sizes = [item["size"] for item in results]
    means = [item["mean"] for item in results]
    spreads = [item["seed_spread"] for item in results]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.errorbar(sizes, means, yerr=spreads, fmt="o-", capsize=3, color="#0072B2")
    axis.axhline(threshold, color="gray", linestyle="--", label="best - 1 seed spread")
    axis.axvline(
        selected["size"], color="#D55E00", linestyle=":",
        label=f"selected ({selected['size']})",
    )
    axis.set_xlabel("Number of input variables")
    axis.set_ylabel("Ladder mass-binned significance (3-seed mean)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "feature_size_seed_scan.png", dpi=180)
    plt.close(figure)


def _plot_feature_scan(results, selected, uncertainty, output_dir):
    sizes = [item["size"] for item in results]
    values = [item["significance"] for item in results]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(sizes, values, "o-", color="#0072B2")
    axis.axhline(
        max(values) - uncertainty["sigma"], color="gray", linestyle="--",
        label="best - 1 sigma",
    )
    axis.axvline(
        selected["size"], color="#D55E00", linestyle=":",
        label=f"selected ({selected['size']})",
    )
    axis.set_xlabel("Number of input variables")
    axis.set_ylabel("Ladder mass-binned significance")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "feature_set_scan.png", dpi=180)
    plt.close(figure)


def _plot_search(results, output_dir):
    values = [item["significance"] for item in results]
    figure, axis = plt.subplots(figsize=(9.5, 4.5))
    axis.scatter(range(len(values)), values, color="#0072B2")
    axis.scatter([0], [values[0]], color="#D55E00", zorder=3, label="production baseline")
    axis.axhline(values[0], color="#D55E00", linestyle=":", alpha=0.6)
    axis.set_xlabel("Random-search configuration")
    axis.set_ylabel("Ladder mass-binned significance")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "hyperparameter_search.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = {int(value) for value in args.stages.split(",")}

    data = load_cache(args.cache_dir)
    assert_group_safe(data["split"], data["group_id"])

    # Every split is strided through the whole memory-mapped cache, so each is
    # read into RAM once here and shared by every stage.
    evaluation = RowBlock(data, np.flatnonzero(data["split"] == 2), "eval")
    blocks = None
    if stages & {0, 1, 2, 3, 5}:
        train, stop = build_blocks(
            data, args, args.madgraph_train_cap, args.madgraph_stop_cap
        )
        blocks = (train, stop, evaluation)

    setup = load_checkpoint(output_dir / "stage0_setup.yaml", args.resume)
    if setup is None and 0 in stages:
        setup = stage0_setup(args, data, output_dir, blocks)

    ranking = load_checkpoint(output_dir / "stage1_ranking.yaml", args.resume)
    if ranking is None and 1 in stages:
        ranking = stage1_ranking(args, data, output_dir, setup, blocks)

    feature_sets = load_checkpoint(output_dir / "stage2_feature_sets.yaml", args.resume)
    if feature_sets is None and 2 in stages:
        feature_sets = stage2_feature_sets(args, data, output_dir, setup, ranking, blocks)

    search = None
    if 3 in stages:
        search = stage3_search(args, data, output_dir, setup, feature_sets, blocks)
    summary = None
    if 4 in stages:
        if search is None:
            search = load_checkpoint(output_dir / "stage3_search.yaml", True)
        summary = stage4_confirm(args, data, output_dir, setup, search, evaluation)
    if 5 in stages:
        if summary is None:
            summary = load_checkpoint(output_dir / "summary_architecture.yaml", True)
        stage5_feature_size_seeds(
            args, data, output_dir, setup, ranking, summary, blocks
        )
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
