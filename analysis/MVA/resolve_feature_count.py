#!/usr/bin/env python3
"""Resolve how many input variables the multiclass proton MVA actually needs.

The architecture study could not separate 15 / 20 / 42 variables: on its
untouched split they gave 1.133 / 1.180 / 1.291 against a bootstrap sigma of
~0.18. The limitation was MadGraph statistics in the signal region, and three
specific things were throwing them away. This script fixes all of them.

  1. Two-fold out-of-fold scoring, folds assigned by hard-event group, so the
     comparison uses every MadGraph group instead of a 15% split.
  2. The MadGraph mass template is the full inclusive mx distribution taken
     before any scoring, normalised to the yield in each score bin, rather
     than being histogrammed from the ~30 events that survive per bin. The
     score is near-independent of mx for this background: in the min-bias pool
     corr(yx, mx) = -0.021, and tightening to |yx| < 0.2 moves <mx> by 0.15 GeV
     against a 4.6 GeV RMS.
  3. The proton pair is integrated over the min-bias pool rather than sampled
     four times. The pair reaches the model through exactly one feature,
     yx_minus_dijet_rapidity, so for a fixed central event the score is a
     function of one variable and can be integrated on a grid.

The v01 MadGraph campaign is dropped: it contributes 249,708 rows at ~116
expected events each and no survivors past the selection, which wrecks the
effective statistics for no benefit. Stitching weights are re-derived over
v02 + v03 only. The v01-exclusive region (parton pT 15-25, |eta| up to 3) is
consequently unmodelled here, by choice.

Run under the analysis environment:
  source setup_env.sh
  python3 analysis/MVA/resolve_feature_count.py \
      --cache-dir analysis/MVA/output/mva_bb_multiclass_protons/cache_v4_all_campaigns \
      --ranking analysis/MVA/output/mva_bb_multiclass_protons/architecture_study_v1/stage1_ranking.yaml \
      --output-dir analysis/MVA/output/mva_bb_multiclass_protons/feature_count_v1
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.MVA.optimize_dijet_mva_multiclass_protons import (  # noqa: E402
    MASS_BINS,
    assert_group_safe,
    cap_groups,
    class_balanced_weights,
    physical_kappas,
    plugin_score,
)
from analysis.MVA.run_dijet_mva_multiclass_protons import (  # noqa: E402
    MAX_ABS_RAPIDITY_DIFFERENCE,
)
from analysis.MVA.study_mva_architecture import (  # noqa: E402
    build_model,
    calibrated_probabilities,
    ladder_edges,
    write_yaml,
)
from common.config_utils import resolve_minbias_campaign  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

DROP_CAMPAIGN = "QCDbb__v01"
DIJET_RAPIDITY_FEATURE = "dijet_rapidity"
PAIR_FEATURE = "yx_minus_dijet_rapidity"
PRODUCTION_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "max_bin": 256,
}
BATCH_ROWS = 250000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ranking", required=True, help="stage1_ranking.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-sets", default="15,20,42")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument(
        "--grid-cells", type=int, default=32,
        help="Cells used to integrate the proton pool over the deltaY band.",
    )
    parser.add_argument("--madgraph-train-cap", type=int, default=250000)
    parser.add_argument("--bootstrap-replicas", type=int, default=200)
    return parser.parse_args()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_reweighted(cache_dir):
    """Load the cache, drop the v01 MadGraph campaign, and re-derive the
    stitching weights over v02 + v03 only.

    The cached central_stitch_weight_fb is exactly 1/sum(L_i) over the campaigns
    covering each event's truth phase-space point, so removing a campaign means
    recomputing that sum without it. Verified to machine precision below.
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    read = lambda name: np.asarray(  # noqa: E731
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
    )

    classes = read("class")
    campaign = read("source_campaign")
    coverage = read("coverage_mask")
    stitch = read("central_stitch_weight_fb")
    band = read("band_probability").astype(np.float64)
    weight = read("physical_weight")
    mixture = read("training_mixture_weight")

    luminosity = {
        name: block["effective_mc_luminosity_fb_inv"]
        for name, block in metadata["madgraph_stitching"].items()
    }
    bit = metadata["madgraph_coverage_bits"]
    madgraph = classes == 2

    covered = np.zeros(int(madgraph.sum()), dtype=np.float64)
    for name, value in luminosity.items():
        covered += value * ((coverage[madgraph] & (1 << bit[name])) > 0)
    if not np.allclose(1.0 / covered, stitch[madgraph], rtol=1e-9):
        raise RuntimeError(
            "Cached central_stitch_weight_fb is not 1/sum(effective MC "
            "luminosity) over covered campaigns; the reweighting below would "
            "be invalid for this cache."
        )
    scale = weight[madgraph] / (stitch[madgraph] * band[madgraph] / 4.0)
    if not np.allclose(scale, scale[0], rtol=1e-6):
        raise RuntimeError("Physical weight scale is not constant across MadGraph rows")
    physical_scale = float(scale[0])

    keep = ~(madgraph & (campaign == DROP_CAMPAIGN))
    kept_madgraph = madgraph[keep]
    retained = [name for name in luminosity if name != DROP_CAMPAIGN]
    recovered = np.zeros(int(kept_madgraph.sum()), dtype=np.float64)
    for name in retained:
        recovered += luminosity[name] * (
            (coverage[keep][kept_madgraph] & (1 << bit[name])) > 0
        )
    if np.any(recovered <= 0.0):
        raise RuntimeError("A retained MadGraph event is outside every retained region")
    new_stitch = 1.0 / recovered
    factor = new_stitch / stitch[keep][kept_madgraph]

    weight = weight[keep].copy()
    mixture = mixture[keep].copy()
    weight[kept_madgraph] *= factor
    mixture[kept_madgraph] *= factor

    data = {
        "x": np.load(cache_dir / "x.npy", mmap_mode="r"),
        "keep": np.flatnonzero(keep),
        "class": classes[keep],
        "group_id": read("group_id")[keep],
        "mx": read("mx")[keep].astype(np.float64),
        "physical_weight": weight,
        "training_mixture_weight": mixture,
        "band_probability": band[keep],
        "central_stitch_new": np.zeros(keep.sum(), dtype=np.float64),
        "feature_names": np.asarray(metadata["features"]),
        "physical_scale": physical_scale,
        "dropped_campaign": DROP_CAMPAIGN,
        "retained_campaigns": retained,
    }
    data["central_stitch_new"][kept_madgraph] = new_stitch
    print(
        f"loaded: {keep.sum()} rows kept of {keep.size} "
        f"(dropped {DROP_CAMPAIGN}); MadGraph groups "
        f"{np.unique(data['group_id'][data['class'] == 2]).size}; "
        f"MadGraph preselection yield "
        f"{data['physical_weight'][data['class'] == 2].sum():.4g}; "
        f"physical_scale={physical_scale:.6f}",
        flush=True,
    )
    return data


def gather(data, rows, features):
    """x[rows][:, features] in batches; the cache is memory-mapped."""
    source = data["keep"][rows]
    out = np.empty((rows.size, len(features)), dtype=np.float32)
    for start in range(0, rows.size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, rows.size)
        out[start:stop] = data["x"][source[start:stop]][:, features]
    return out


def load_pool():
    """Min-bias proton-pair pool, sorted by yx with a cumulative weight array.

    The cumulative array turns "pool probability inside a yx window" into two
    lookups, which is what the deltaY integral needs once per event per cell.
    """
    campaign_dir, campaign = resolve_minbias_campaign(None)
    table = pq.read_table(
        campaign_dir / "pairs" / "proton_pairs.parquet", columns=["yx", "weight"]
    )
    yx = np.asarray(table["yx"], dtype=np.float64)
    weight = np.asarray(table["weight"], dtype=np.float64)
    order = np.argsort(yx, kind="stable")
    yx = yx[order]
    cumulative = np.r_[0.0, np.cumsum(weight[order])]
    print(f"pool: {yx.size} pairs from {campaign}", flush=True)
    return {"yx": yx, "cumulative": cumulative, "total": float(cumulative[-1])}


def pool_mass_between(pool, low, high):
    """Pool weight with yx in [low, high), vectorised over events."""
    left = np.searchsorted(pool["yx"], low, side="left")
    right = np.searchsorted(pool["yx"], high, side="left")
    return pool["cumulative"][right] - pool["cumulative"][left]


# --------------------------------------------------------------------------
# folds and fitting
# --------------------------------------------------------------------------
def assign_folds(groups, seed, n_folds=2):
    values = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    lookup = np.empty(int(values.max()) + 1, dtype=np.int8)
    lookup[values] = np.arange(values.size) % n_folds
    folds = lookup[groups]
    assert_group_safe(folds, groups)
    return folds


def fold_training_rows(data, folds, fold, cap, seed):
    """Rows for training and for early stopping/calibration within one fold.

    One row per MadGraph group for training (the copies are the same hard
    event), a deterministic 10% inner group holdout for early stopping, and a
    uniform cap on MadGraph training groups so every feature set is compared at
    the same training size.
    """
    available = np.flatnonzero(folds != fold)
    groups = data["group_id"]
    inner = np.unique(groups[available])
    stop_groups = inner[np.asarray(inner, dtype=np.uint64) % np.uint64(10) == 0]
    is_stop = np.isin(groups[available], stop_groups)
    stop_rows = available[is_stop]
    candidates = available[~is_stop]

    madgraph = data["class"][candidates] == 2
    _unique, first = np.unique(groups[candidates[madgraph]], return_index=True)
    train_rows = np.concatenate([
        candidates[~madgraph], candidates[madgraph][first]
    ])
    train_rows, _correction = cap_groups(
        train_rows, data["class"], groups, 2, cap, np.random.default_rng(seed + 5)
    )
    return np.sort(train_rows), np.sort(stop_rows)


def fit_calibrated(data, features, train_rows, stop_rows, params, seed):
    train_weights = class_balanced_weights(
        data["class"][train_rows], data["training_mixture_weight"][train_rows]
    )
    stop_weights = class_balanced_weights(
        data["class"][stop_rows], data["training_mixture_weight"][stop_rows]
    )
    stop_x = gather(data, stop_rows, features)
    model = build_model(params, seed)
    model.fit(
        gather(data, train_rows, features), data["class"][train_rows],
        sample_weight=train_weights,
        eval_set=[(stop_x, data["class"][stop_rows])],
        sample_weight_eval_set=[stop_weights],
        verbose=False,
    )
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
    calibrator.fit(
        model.predict(stop_x, output_margin=True), data["class"][stop_rows],
        sample_weight=stop_weights,
    )
    return model, calibrator


# --------------------------------------------------------------------------
# the proton-pool integral
# --------------------------------------------------------------------------
def integrate_pool(
    matrix, rapidity, pool, model, calibrator, kappas, pair_column, cells, edges
):
    """Probability that each central event lands in each score bin.

    The sampled pair reaches the model through one feature only, so for a fixed
    central event the score is a function of deltaY alone. Sweeping deltaY on a
    grid and weighting each cell by the pool probability inside the
    corresponding yx window replaces the four random draws with a deterministic
    integral. Cell probabilities sum to the event's band probability by
    construction, which is the normalisation the sampled scheme used.
    """
    n_bins = edges.size - 1
    probability = np.zeros((matrix.shape[0], n_bins), dtype=np.float64)
    bounds = np.linspace(
        -MAX_ABS_RAPIDITY_DIFFERENCE, MAX_ABS_RAPIDITY_DIFFERENCE, cells + 1
    )
    for index in range(cells):
        low, high = bounds[index], bounds[index + 1]
        matrix[:, pair_column] = 0.5 * (low + high)
        score = plugin_score(
            calibrated_probabilities(model, calibrator, matrix), kappas
        )
        cell = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_bins - 1)
        mass = pool_mass_between(pool, rapidity + low, rapidity + high) / pool["total"]
        probability[np.arange(matrix.shape[0]), cell] += mass
    return probability


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------
def significance(signal, superchic, madgraph_yield, shape):
    """Z = sqrt(sum over score bins and mass bins of s^2/(s+b))."""
    background = superchic + madgraph_yield[:, None] * shape[None, :]
    total = signal + background
    return float(np.sqrt(np.sum(
        np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0)
    )))


def event_cells(data, score, edges, class_id):
    """Flat (score bin, mass bin) index and weight for one class of events.

    Keeping the per-event assignment rather than only the histogram is what
    lets the bootstrap resample events instead of bins.
    """
    n_bins = edges.size - 1
    n_mass = MASS_BINS.size - 1
    rows = np.flatnonzero(data["class"] == class_id)
    score_bin = np.clip(
        np.searchsorted(edges, score[rows], side="right") - 1, 0, n_bins - 1
    )
    mass_bin = np.searchsorted(MASS_BINS, data["mx"][rows], side="right") - 1
    inside = (mass_bin >= 0) & (mass_bin < n_mass)
    flat = score_bin[inside] * n_mass + mass_bin[inside]
    return flat, data["physical_weight"][rows][inside], n_bins * n_mass


def histogram_from_cells(flat, weight, size, shape2d):
    return np.bincount(flat, weights=weight, minlength=size).reshape(shape2d)


def build_templates(data, score, edges, probability, madgraph_weight):
    """Per-score-bin mass templates.

    Signal and SuperChic QCDbb keep their per-event mx: their protons are real.
    MadGraph contributes only a yield per score bin, which is multiplied by the
    inclusive mass shape by the caller.
    """
    n_bins = edges.size - 1
    n_mass = MASS_BINS.size - 1
    cells = {
        class_id: event_cells(data, score, edges, class_id) for class_id in (0, 1)
    }
    signal = histogram_from_cells(*cells[0], (n_bins, n_mass))
    superchic = histogram_from_cells(*cells[1], (n_bins, n_mass))
    return signal, superchic, probability.T @ madgraph_weight, cells


def bootstrap(cells, probability, madgraph_weight, shape, edges, replicas, seed):
    """Resample Hbb events, QCDbb events and MadGraph central events."""
    n_bins = edges.size - 1
    n_mass = MASS_BINS.size - 1
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicas):
        replicas_2d = []
        for class_id in (0, 1):
            flat, weight, size = cells[class_id]
            draw = rng.multinomial(weight.size, np.full(weight.size, 1.0 / weight.size))
            replicas_2d.append(
                histogram_from_cells(flat, weight * draw, size, (n_bins, n_mass))
            )
        count = madgraph_weight.size
        draw = rng.multinomial(count, np.full(count, 1.0 / count))
        values.append(significance(
            replicas_2d[0], replicas_2d[1],
            probability.T @ (madgraph_weight * draw), shape,
        ))
    low, median, high = np.percentile(np.asarray(values), [16, 50, 84])
    return {
        "replicas": int(replicas),
        "median": float(median),
        "percent_68": [float(low), float(high)],
        "sigma": float(0.5 * (high - low)),
    }


def effective_counts(probability, madgraph_weight):
    contribution = probability * madgraph_weight[:, None]
    total = contribution.sum(axis=0)
    squared = np.sum(contribution * contribution, axis=0)
    return np.divide(
        total * total, squared, out=np.zeros_like(total), where=squared > 0.0
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def evaluate_feature_set(args, data, pool, features, seed, kappas, shape):
    names = data["feature_names"]
    pair_column = int(np.flatnonzero(names[features] == PAIR_FEATURE)[0])
    rapidity_index = int(np.flatnonzero(names == DIJET_RAPIDITY_FEATURE)[0])

    folds = assign_folds(data["group_id"], seed)
    madgraph_rows = np.flatnonzero(data["class"] == 2)
    _groups, first = np.unique(data["group_id"][madgraph_rows], return_index=True)
    central = madgraph_rows[first]
    central_matrix = gather(data, central, features)
    rapidity = gather(data, central, [rapidity_index])[:, 0].astype(np.float64)
    madgraph_weight = data["central_stitch_new"][central] * data["physical_scale"]
    band = data["band_probability"][central]

    # Fit both folds first and score every real-proton event out of fold. The
    # two fold models have different score scales, so the ladder edges must come
    # from the combined out-of-fold signal scores, not from one fold's model.
    score = np.full(data["class"].size, np.nan)
    fits = []
    for fold in range(2):
        train_rows, stop_rows = fold_training_rows(
            data, folds, fold, args.madgraph_train_cap, seed
        )
        model, calibrator = fit_calibrated(
            data, features, train_rows, stop_rows, PRODUCTION_PARAMS, seed + fold
        )
        fits.append((model, calibrator))
        held_out = np.flatnonzero(folds == fold)
        real = held_out[data["class"][held_out] != 2]
        score[real] = plugin_score(
            calibrated_probabilities(model, calibrator, gather(data, real, features)),
            kappas,
        )
    signal_score = score[data["class"] == 0]
    edges = ladder_edges(
        signal_score, np.zeros(signal_score.size, dtype=np.int8), args.ladder_bins
    )

    probability = np.zeros((central.size, edges.size - 1), dtype=np.float64)
    residual = 0.0
    for fold, (model, calibrator) in enumerate(fits):
        selected = folds[central] == fold
        probability[selected] = integrate_pool(
            central_matrix[selected], rapidity[selected], pool, model, calibrator,
            kappas, pair_column, args.grid_cells, edges,
        )
        residual = max(residual, float(np.max(np.abs(
            probability[selected].sum(axis=1) - band[selected]
        ))))

    signal, superchic, madgraph_yield, cells = build_templates(
        data, score, edges, probability, madgraph_weight
    )
    return {
        "significance": significance(signal, superchic, madgraph_yield, shape),
        "band_probability_residual": residual,
        "madgraph_yield_per_bin": madgraph_yield.tolist(),
        "effective_central_events_per_bin": effective_counts(
            probability, madgraph_weight
        ).tolist(),
    }, (cells, probability, madgraph_weight, edges)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_reweighted(args.cache_dir)
    pool = load_pool()
    with open(args.ranking, encoding="utf-8") as handle:
        ranked = np.asarray(yaml.safe_load(handle)["ranked_feature_indices"], dtype=int)

    kappas, yields = physical_kappas(data["class"], data["physical_weight"])
    madgraph = data["class"] == 2
    shape, _ = np.histogram(
        data["mx"][madgraph], MASS_BINS, weights=data["physical_weight"][madgraph]
    )
    shape = shape / shape.sum()
    print(f"kappas={kappas}  preselection yields={yields}", flush=True)

    results = []
    for size in (int(value) for value in args.feature_sets.split(",")):
        features = ranked[:size]
        per_seed = []
        last = None
        for offset in range(args.seeds):
            started = time.perf_counter()
            metrics, pieces = evaluate_feature_set(
                args, data, pool, features, args.base_seed + 100 * offset, kappas, shape
            )
            per_seed.append(metrics["significance"])
            last = (metrics, pieces)
            print(
                f"  {size:3d} features seed {offset}: Z={metrics['significance']:.4f} "
                f"(band residual {metrics['band_probability_residual']:.2e}, "
                f"{time.perf_counter() - started:.0f}s)",
                flush=True,
            )
        metrics, (cells, probability, madgraph_weight, edges) = last
        uncertainty = bootstrap(
            cells, probability, madgraph_weight, shape, edges,
            args.bootstrap_replicas, args.base_seed + 7,
        )
        results.append({
            "size": int(size),
            "features": data["feature_names"][features].tolist(),
            "per_seed": [float(value) for value in per_seed],
            "mean": float(np.mean(per_seed)),
            "seed_spread": float(np.std(per_seed)),
            "bootstrap": uncertainty,
            "madgraph_yield_per_bin": metrics["madgraph_yield_per_bin"],
            "effective_central_events_per_bin": metrics[
                "effective_central_events_per_bin"
            ],
            "band_probability_residual": metrics["band_probability_residual"],
        })
        print(
            f"{size:3d} features: Z={np.mean(per_seed):.4f} "
            f"+/- {np.std(per_seed):.4f} (seed), "
            f"+/- {uncertainty['sigma']:.4f} (bootstrap); "
            f"min effective central events/bin="
            f"{min(metrics['effective_central_events_per_bin']):.0f}",
            flush=True,
        )

    write_yaml(output_dir / "feature_count_comparison.yaml", {
        "cache_dir": args.cache_dir,
        "ranking": args.ranking,
        "dropped_campaign": data["dropped_campaign"],
        "retained_campaigns": data["retained_campaigns"],
        "hyperparameters": PRODUCTION_PARAMS,
        "folds": 2,
        "grid_cells": args.grid_cells,
        "ladder_bins": args.ladder_bins,
        "seeds": args.seeds,
        "madgraph_train_cap": args.madgraph_train_cap,
        "kappas": kappas.tolist(),
        "preselection_yields": yields.tolist(),
        "madgraph_mass_shape": shape.tolist(),
        "results": results,
        "note": (
            "Classifier-only. Z is stat-only with a perfectly known background "
            "and does not address the combinatorial acceptance factor, which "
            "is a detector design parameter rather than an uncertainty."
        ),
    })
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
