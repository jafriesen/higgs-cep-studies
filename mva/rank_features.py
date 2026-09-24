#!/usr/bin/env python3
"""Rank central features by recursive elimination with refit.

Stage 1 drops one member of each strongly correlated pair; stage 2 removes the
weakest remaining feature by permutation impact, refits, and repeats.  The
metric is the mass-binned significance evaluated with ONE sampled proton cell
per pooled event rather than the production grid, which is ~250x cheaper and
affects only the proton axis.  Absolute values here are not sensitivities; only
the ordering and the nesting curve are used.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mva.common.protons import build_pair_density, load_pps_config  # noqa: E402
from mva.common.training import (  # noqa: E402
    assign_folds,
    balanced_weights,
    calibrated_probabilities,
    fit_calibrated,
    load_dataset,
    sample_training_cells,
)
from mva.common.weights import plugin_score  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[12345, 20260101, 20260202])
    p.add_argument("--correlation-max", type=float, default=0.95)
    p.add_argument("--floor", type=int, default=10, help="smallest set to evaluate")
    p.add_argument("--train-cap", type=int, default=250000)
    p.add_argument("--stop-cap", type=int, default=40000)
    p.add_argument("--eval-cap", type=int, default=400000)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--grid-cells", type=int, default=256)
    p.add_argument("--batch-rows", type=int, default=250000)
    p.add_argument("--monitor-rounds", type=int, default=0)
    p.add_argument("--mass-bins", type=int, default=16)
    p.add_argument("--ladder-bins", type=int, default=6)
    p.add_argument("--fine-below", type=int, default=25,
                   help="remove one feature at a time below this size")
    p.add_argument("--skip-hash", action="store_true", default=True)
    return p.parse_args()


class Fit:
    """Container mirroring the production trainer's arguments."""

    def __init__(self, args, seed):
        self.n_estimators = args.n_estimators
        self.jobs = args.jobs
        self.seed = seed
        self.batch_rows = args.batch_rows
        self.monitor_rounds = args.monitor_rounds


def build_matrix(data):
    meta = data["metadata"]
    central = list(meta["central_features"])
    names = central + ["yx_minus_dijet_rapidity"]
    rows = data["x"].shape[0]
    matrix = np.empty((rows, len(names)), dtype=np.float32)
    for start in range(0, rows, 500000):
        stop = min(start + 500000, rows)
        matrix[start:stop, : len(central)] = np.asarray(data["x"][start:stop])
        delta = data["proton_yx"][start:stop] - data["dijet_rapidity"][start:stop]
        matrix[start:stop, len(central)] = np.nan_to_num(delta, nan=0.0)
    return matrix, names


def correlation_prune(matrix, names, limit, sample=400000, seed=0):
    """Drop constant features and one member of each strongly correlated pair.

    NaN is meaningful here -- XGBoost learns a default direction for it, and
    several activity variables are undefined by construction (no interjet
    track, no outer track jet).  So the statistics are nan-aware, and only
    features with no variation among their finite entries count as dead.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(matrix.shape[0])
    if rows.size > sample:
        rows = np.sort(rng.choice(rows, size=sample, replace=False))
    block = np.asarray(matrix[rows], dtype=np.float64)
    spread = np.nanstd(block, axis=0)
    dead = [i for i in range(len(names)) if not np.isfinite(spread[i]) or spread[i] == 0.0]
    keep = [i for i in range(len(names)) if i not in dead]

    # correlation only needs to spot near-duplicates; impute NaN at the median
    filled = block[:, keep].copy()
    for column in range(filled.shape[1]):
        values = filled[:, column]
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.nanmedian(values)
    corr = np.corrcoef(filled, rowvar=False)

    dropped = []
    while True:
        worst = None
        for a in range(len(keep)):
            for b in range(a + 1, len(keep)):
                value = corr[a, b]
                if np.isfinite(value) and abs(value) >= limit:
                    if worst is None or abs(value) > worst[0]:
                        worst = (abs(value), a, b)
        if worst is None:
            break
        _, a, b = worst
        loser = b if spread[keep[b]] <= spread[keep[a]] else a
        winner = a if loser == b else b
        dropped.append((names[keep[loser]], names[keep[winner]], float(worst[0])))
        keep.pop(loser)
        filled = np.delete(filled, loser, axis=1)
        corr = np.corrcoef(filled, rowvar=False)
    return keep, dropped, [names[i] for i in dead]


def significance(score, mass_bin, weight, is_signal, n_mass, n_categories):
    """Ladder significance over signal-score quantile categories.

    A single cut scanned for its maximum is upward biased here: the pooled
    background is a rescaled subsample, so its tail is a few very heavy events
    and the maximum lands on whichever threshold happens to exclude them.  The
    ladder places every event in a category instead, which is both the
    production metric and stable enough to rank features with.
    """
    if not np.any(is_signal):
        return 0.0
    quantiles = np.linspace(0.0, 1.0, n_categories + 1)[1:-1]
    edges = np.quantile(score[is_signal], quantiles)
    category = np.searchsorted(edges, score, side="right")
    flat = category * n_mass + mass_bin
    size = n_categories * n_mass
    signal = np.bincount(flat[is_signal], weights=weight[is_signal], minlength=size)
    background = np.bincount(flat[~is_signal], weights=weight[~is_signal], minlength=size)
    total = signal + background
    return float(np.sqrt(np.sum(np.divide(signal * signal, total,
                                          out=np.zeros_like(total), where=total > 0.0))))


def fit_folds(matrix, columns, folds, eligible, pooled, labels, mixture,
              n_classes, args, seed):
    """Fit both cross-fit folds once on `columns`."""
    from mva.common.training import fold_rows

    sub = np.ascontiguousarray(matrix[:, columns])
    fits = []
    for fold in range(2):
        train, stop = fold_rows(folds, fold, eligible, pooled, args.train_cap,
                                args.stop_cap, seed)
        fits.append(fit_calibrated(sub, labels, mixture, train, stop, n_classes,
                                   Fit(args, seed), fold))
    return sub, fits


def score_z(sub, fits, folds, eval_rows, weight, mass_bin, is_signal, n_classes, args,
            permute_column=None, rng=None):
    """Score the fixed evaluation rows with already-fitted models."""
    if permute_column is not None:
        saved = sub[eval_rows, permute_column].copy()
        sub[eval_rows, permute_column] = saved[rng.permutation(saved.size)]
    try:
        score = np.full(sub.shape[0], np.nan, dtype=np.float64)
        kappas = np.ones(n_classes - 1, dtype=np.float64)
        for fold, (model, calibrator) in enumerate(fits):
            rows = eval_rows[folds[eval_rows] == fold]
            if rows.size == 0:
                continue
            prob = calibrated_probabilities(model, calibrator, sub[rows], args.batch_rows)
            score[rows] = plugin_score(prob, kappas)
        rows = eval_rows[np.isfinite(score[eval_rows])]
        return significance(score[rows], mass_bin[rows], weight[rows], is_signal[rows],
                            args.mass_bins, args.ladder_bins)
    finally:
        if permute_column is not None:
            sub[eval_rows, permute_column] = saved


def main():
    args = parse_args()
    started = time.perf_counter()
    data = load_dataset(args.data_dir)
    meta = data["metadata"]
    n_classes = len(meta["classes"])
    components = meta["components"]
    print(f"{meta['channel']} {meta['profile']}: {meta['rows']:,} events, "
          f"{len(meta['central_features'])} central features", flush=True)

    matrix, names = build_matrix(data)
    labels = np.asarray(data["class"], dtype=np.int32)
    component = np.asarray(data["component"], dtype=np.int32)
    mixture = np.asarray(data["training_mixture_weight"], dtype=np.float64)
    physical = np.asarray(data["physical_weight"], dtype=np.float64)
    band = np.asarray(data["pair_band_intensity"], dtype=np.float64)
    groups = np.asarray(data["group_id"])
    real = np.array([bool(c["real_protons"]) for c in components])
    pooled = ~real[component]
    eligible = np.isfinite(physical) & (physical > 0.0)
    eligible &= np.where(pooled, band > 0.0, True)

    # one sampled proton cell per pooled event, exactly as the trainer fits on
    pps = load_pps_config(meta["pps_config"]) if "pps_config" in meta else load_pps_config(
        yaml.safe_load(open(Path(args.data_dir) / "channel_config.yaml"))["pps_config"])
    channel = yaml.safe_load(open(Path(args.data_dir) / "channel_config.yaml"))
    pairs = build_pair_density(channel["minbias_path"], pps, seed=args.seeds[0],
                               verify_hash=not args.skip_hash)
    delta = float(channel["max_abs_rapidity_difference"])
    cell_edges = np.linspace(-delta, delta, args.grid_cells + 1)
    pair_column = len(names) - 1
    pooled_rows = np.flatnonzero(pooled & eligible)
    sample_training_cells(matrix, pair_column, pooled_rows,
                          np.asarray(data["dijet_rapidity"]), pairs, cell_edges,
                          tuple(float(v) for v in channel["mass_window_gev"]),
                          float(channel["pileup_mu"]), args.seeds[0], 500000)

    # evaluation weights and mass bins; pooled mass is uniform across the window
    window = tuple(float(v) for v in channel["mass_window_gev"])
    weight = np.where(pooled, physical * band, physical)
    mx = np.asarray(data["proton_mx"], dtype=np.float64)
    edges = np.linspace(window[0], window[1], args.mass_bins + 1)
    mass_bin = np.clip(np.digitize(mx, edges) - 1, 0, args.mass_bins - 1)
    rng = np.random.default_rng(4242)
    # a pooled event carries its full weight into one randomly drawn mass bin;
    # that already realises the uniform spread, so the weight is NOT divided.
    mass_bin[pooled] = rng.integers(0, args.mass_bins, size=int(pooled.sum()))
    is_signal = labels == 0

    # fixed evaluation subsample, pooled rows capped and rescaled (never subsample
    # without rescaling: that is the 137x trap)
    real_rows = np.flatnonzero(real[component] & eligible)
    if pooled_rows.size > args.eval_cap:
        drawn = rng.choice(pooled_rows, size=args.eval_cap, replace=False)
        weight[drawn] *= pooled_rows.size / drawn.size
    else:
        drawn = pooled_rows
    eval_rows = np.sort(np.concatenate([real_rows, drawn]))
    print(f"evaluation rows: {eval_rows.size:,} "
          f"({real_rows.size:,} real + {drawn.size:,} pooled of {pooled_rows.size:,})", flush=True)

    keep, dropped, dead = correlation_prune(matrix, names, args.correlation_max)
    print(f"\nstage 1: {len(names)} -> {len(keep)} features")
    for loser, winner, value in dropped:
        print(f"   drop {loser:<34s} corr {value:.3f} with {winner}")
    if dead:
        print(f"   drop (no variation): {dead}")

    history = []
    for seed in args.seeds:
        folds = assign_folds(groups, seed)
        columns = list(keep)
        while len(columns) >= args.floor:
            sub, fits = fit_folds(matrix, columns, folds, eligible, pooled, labels,
                                  mixture, n_classes, args, seed)
            base = score_z(sub, fits, folds, eval_rows, weight, mass_bin, is_signal,
                           n_classes, args)
            impact = {}
            for position, column in enumerate(columns):
                impact[names[column]] = base - score_z(
                    sub, fits, folds, eval_rows, weight, mass_bin, is_signal,
                    n_classes, args, permute_column=position, rng=rng)
            batch = max(1, (len(columns) - args.floor) // 8) if len(columns) > args.fine_below else 1
            order = sorted(impact, key=impact.get)
            weakest = order[:batch]
            history.append({"seed": int(seed), "size": len(columns), "z": float(base),
                            "features": [names[c] for c in columns],
                            "impact": {k: float(v) for k, v in impact.items()},
                            "removed": list(weakest)})
            print(f"seed {seed}: n={len(columns):3d} Z={base:.5f} drop "
                  f"{', '.join(weakest)} (impact {impact[weakest[0]]:+.5f}) "
                  f"[{time.perf_counter()-started:.0f}s]", flush=True)
            columns = [c for c in columns if names[c] not in set(weakest)]
            with open(args.output, "w", encoding="utf-8") as handle:
                yaml.safe_dump({"data_dir": str(args.data_dir), "channel": meta["channel"],
                                "stage1_dropped": dropped, "history": history},
                               handle, sort_keys=False)
    print(f"\nwrote {args.output} in {time.perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()
