#!/usr/bin/env python3
"""Train and evaluate the standalone locked 20-feature proton MVA."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import Booster, DMatrix, XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data"
DEFAULT_RESULTS = SCRIPT_DIR / "results"
CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
MASS_BINS = np.arange(117.0, 134.0, 1.0)
MAX_ABS_RAPIDITY_DIFFERENCE = 0.2
PAIR_FEATURE = "yx_minus_dijet_rapidity"
DIJET_RAPIDITY_FEATURE = "dijet_rapidity"
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
N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 40
BATCH_ROWS = 250000
SCORE_RANGE_BINS = 320
N_SCAN = 96
MIN_EFFECTIVE_CENTRAL = 50.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--grid-cells", type=int, default=32)
    parser.add_argument("--madgraph-train-cap", type=int, default=250000)
    return parser.parse_args()


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(plain(payload), handle, sort_keys=False)


def load_data(data_dir):
    with open(data_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if metadata.get("format_version") != 1:
        raise RuntimeError("Unsupported reduced-dataset format")
    names = np.asarray(metadata["features"])
    if names.size != 20:
        raise RuntimeError(f"Expected 20 features, found {names.size}")
    for required in (PAIR_FEATURE, DIJET_RAPIDITY_FEATURE):
        if required not in names:
            raise RuntimeError(f"Reduced dataset is missing {required}")
    arrays = {
        name: np.load(data_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "x",
            "class",
            "group_id",
            "mx",
            "physical_weight",
            "training_mixture_weight",
            "band_probability",
            "central_weight",
        )
    }
    rows = arrays["class"].size
    if arrays["x"].shape != (rows, 20):
        raise RuntimeError("Reduced feature matrix has the wrong shape")
    for name, values in arrays.items():
        if name != "x" and values.shape != (rows,):
            raise RuntimeError(f"Reduced array {name} has the wrong shape")
    arrays["feature_names"] = names
    arrays["metadata"] = metadata
    return arrays


def load_pool(data_dir):
    table = pq.read_table(
        data_dir / "proton_pairs.parquet", columns=["yx", "weight"]
    )
    yx = np.asarray(table["yx"], dtype=np.float64)
    weight = np.asarray(table["weight"], dtype=np.float64)
    order = np.argsort(yx, kind="stable")
    yx = yx[order]
    cumulative = np.r_[0.0, np.cumsum(weight[order])]
    if cumulative[-1] <= 0.0:
        raise RuntimeError("Proton-pair pool has non-positive total weight")
    return {"yx": yx, "cumulative": cumulative, "total": float(cumulative[-1])}


def gather(data, rows):
    output = np.empty((rows.size, data["x"].shape[1]), dtype=np.float32)
    for start in range(0, rows.size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, rows.size)
        output[start:stop] = data["x"][rows[start:stop]]
    return output


def pool_mass_between(pool, low, high):
    left = np.searchsorted(pool["yx"], low, side="left")
    right = np.searchsorted(pool["yx"], high, side="left")
    return pool["cumulative"][right] - pool["cumulative"][left]


def assert_group_safe(labels, groups):
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_labels = labels[order]
    if np.any(
        (sorted_groups[1:] == sorted_groups[:-1])
        & (sorted_labels[1:] != sorted_labels[:-1])
    ):
        raise RuntimeError("A hard-event group crosses data folds")


def assign_folds(groups, seed, n_folds=2):
    values = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    lookup = np.empty(int(values.max()) + 1, dtype=np.int8)
    lookup[values] = np.arange(values.size) % n_folds
    folds = lookup[groups]
    assert_group_safe(folds, groups)
    return folds


def cap_groups(indices, classes, groups, class_id, limit, rng):
    mask = classes[indices] == class_id
    candidates = np.unique(groups[indices[mask]])
    if limit is None or candidates.size <= limit:
        return indices
    selected = rng.choice(candidates, size=limit, replace=False)
    return indices[~mask | np.isin(groups[indices], selected)]


def fold_training_rows(data, folds, fold, cap, seed):
    available = np.flatnonzero(folds != fold)
    groups = data["group_id"]
    inner = np.unique(groups[available])
    stop_groups = inner[np.asarray(inner, dtype=np.uint64) % np.uint64(10) == 0]
    is_stop = np.isin(groups[available], stop_groups)
    stop_rows = available[is_stop]
    candidates = available[~is_stop]

    madgraph = data["class"][candidates] == 2
    _unique, first = np.unique(groups[candidates[madgraph]], return_index=True)
    train_rows = np.concatenate((candidates[~madgraph], candidates[madgraph][first]))
    train_rows = cap_groups(
        train_rows,
        data["class"],
        groups,
        2,
        cap,
        np.random.default_rng(seed + 5),
    )
    return np.sort(train_rows), np.sort(stop_rows)


def class_balanced_weights(classes, mixture):
    weights = np.asarray(mixture, dtype=np.float64).copy()
    target = classes.size / len(CLASS_ORDER)
    for class_id in range(len(CLASS_ORDER)):
        mask = classes == class_id
        total = np.sum(weights[mask])
        if total <= 0.0:
            raise RuntimeError(f"Class {CLASS_ORDER[class_id]} has no training weight")
        weights[mask] *= target / total
    return weights


def build_model(seed):
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=16,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=seed,
        **PRODUCTION_PARAMS,
    )


def fit_calibrated(data, train_rows, stop_rows, seed):
    train_weights = class_balanced_weights(
        data["class"][train_rows], data["training_mixture_weight"][train_rows]
    )
    stop_weights = class_balanced_weights(
        data["class"][stop_rows], data["training_mixture_weight"][stop_rows]
    )
    stop_x = gather(data, stop_rows)
    model = build_model(seed)
    model.fit(
        gather(data, train_rows),
        data["class"][train_rows],
        sample_weight=train_weights,
        eval_set=[(stop_x, data["class"][stop_rows])],
        sample_weight_eval_set=[stop_weights],
        verbose=False,
    )
    calibrator = LogisticRegression(
        solver="lbfgs", max_iter=500, random_state=seed
    )
    calibrator.fit(
        model.predict(stop_x, output_margin=True),
        data["class"][stop_rows],
        sample_weight=stop_weights,
    )
    return model, calibrator


def calibrated_probabilities(model, calibrator, matrix):
    output = np.empty((matrix.shape[0], len(CLASS_ORDER)), dtype=np.float64)
    for start in range(0, matrix.shape[0], BATCH_ROWS):
        stop = min(start + BATCH_ROWS, matrix.shape[0])
        margins = model.predict(matrix[start:stop], output_margin=True)
        output[start:stop] = calibrator.predict_proba(margins)
    return output


def calibration_payload(calibrator):
    return {
        "classes": calibrator.classes_.astype(int),
        "coefficients": calibrator.coef_,
        "intercepts": calibrator.intercept_,
        "iterations": calibrator.n_iter_.astype(int),
    }


def portable_calibration(margins, payload):
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    intercepts = np.asarray(payload["intercepts"], dtype=np.float64)
    logits = np.asarray(margins, dtype=np.float64) @ coefficients.T + intercepts
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def physical_kappas(classes, weights):
    yields = np.bincount(classes, weights=weights, minlength=3)
    if yields[0] <= 0.0:
        raise RuntimeError("Signal has non-positive physical yield")
    return yields[1:] / yields[0], yields


def plugin_score(probabilities, kappas):
    probabilities = np.clip(probabilities, 1.0e-12, 1.0)
    background = kappas[0] * probabilities[:, 1] + kappas[1] * probabilities[:, 2]
    return np.log(probabilities[:, 0]) - np.log(background)


def ladder_edges(signal_score, n_bins):
    signal_score = signal_score[np.isfinite(signal_score)]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.r_[-np.inf, np.quantile(signal_score, quantiles), np.inf])


def sweep_pool(
    matrix, rapidity, weight, pool, model, calibrator, kappas, pair_column, cells
):
    score = np.empty((matrix.shape[0], cells), dtype=np.float32)
    contribution = np.empty((matrix.shape[0], cells), dtype=np.float32)
    covered = np.zeros(matrix.shape[0])
    bounds = np.linspace(
        -MAX_ABS_RAPIDITY_DIFFERENCE,
        MAX_ABS_RAPIDITY_DIFFERENCE,
        cells + 1,
    )
    for index in range(cells):
        low, high = bounds[index], bounds[index + 1]
        matrix[:, pair_column] = 0.5 * (low + high)
        score[:, index] = plugin_score(
            calibrated_probabilities(model, calibrator, matrix), kappas
        )
        mass = pool_mass_between(pool, rapidity + low, rapidity + high) / pool["total"]
        covered += mass
        contribution[:, index] = weight * mass
    return score, contribution, covered


def accumulate_regions(score, contribution, edges):
    size = edges.size - 1
    cell = np.clip(
        np.searchsorted(edges, score.ravel(), side="right") - 1, 0, size - 1
    )
    rows = np.repeat(np.arange(score.shape[0]), score.shape[1])
    per_event = np.bincount(
        rows * size + cell,
        weights=contribution.ravel().astype(np.float64),
        minlength=score.shape[0] * size,
    ).reshape(score.shape[0], size)
    return per_event.sum(axis=0), np.sum(per_event * per_event, axis=0)


def accumulate_above(score, contribution, thresholds):
    total = np.zeros(thresholds.size)
    squared = np.zeros(thresholds.size)
    for index, threshold in enumerate(thresholds):
        per_event = np.sum(
            np.where(score >= threshold, contribution, 0.0),
            axis=1,
            dtype=np.float64,
        )
        total[index] = per_event.sum()
        squared[index] = np.sum(per_event * per_event)
    return total, squared


def class_score_mass(data, score, class_id, score_edges):
    rows = np.flatnonzero(data["class"] == class_id)
    histogram, _, _ = np.histogram2d(
        score[rows],
        data["mx"][rows],
        bins=(np.r_[score_edges, np.inf], MASS_BINS),
        weights=data["physical_weight"][rows],
    )
    return histogram


def significance(signal, superchic, madgraph_yield, madgraph_shape):
    background = superchic + madgraph_yield * madgraph_shape
    total = signal + background
    return float(
        np.sqrt(
            np.sum(
                np.divide(
                    signal * signal,
                    total,
                    out=np.zeros_like(signal),
                    where=total > 0.0,
                )
            )
        )
    )


def cumulative_from_top(values):
    return np.cumsum(values[::-1], axis=0)[::-1]


def save_and_check_model(model, calibrator, sample, fold, model_dir):
    model_path = model_dir / f"fold_{fold}_model.json"
    calibration_path = model_dir / f"fold_{fold}_calibration.yaml"
    model.get_booster().save_model(model_path)
    payload = calibration_payload(calibrator)
    write_yaml(calibration_path, payload)

    restored = Booster()
    restored.load_model(model_path)
    original_margin = model.predict(sample, output_margin=True)
    restored_margin = restored.predict(
        DMatrix(sample),
        output_margin=True,
        iteration_range=(0, int(model.best_iteration) + 1),
    )
    expected = calibrator.predict_proba(original_margin)
    actual = portable_calibration(restored_margin, payload)
    return {
        "fold": fold,
        "model": str(model_path.name),
        "calibration": str(calibration_path.name),
        "best_iteration": int(model.best_iteration),
        "reload_margin_max_abs_difference": float(
            np.max(np.abs(original_margin - restored_margin))
        ),
        "reload_probability_max_abs_difference": float(
            np.max(np.abs(expected - actual))
        ),
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    model_dir = result_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = load_data(data_dir)
    pool = load_pool(data_dir)
    names = data["feature_names"]
    pair_column = int(np.flatnonzero(names == PAIR_FEATURE)[0])
    rapidity_column = int(np.flatnonzero(names == DIJET_RAPIDITY_FEATURE)[0])
    kappas, preselection_yields = physical_kappas(
        data["class"], data["physical_weight"]
    )
    madgraph_mask = data["class"] == 2
    madgraph_shape, _ = np.histogram(
        data["mx"][madgraph_mask],
        MASS_BINS,
        weights=data["physical_weight"][madgraph_mask],
    )
    madgraph_shape = madgraph_shape / madgraph_shape.sum()
    print(
        f"Loaded {data['class'].size:,} rows; pool has {pool['yx'].size:,} pairs",
        flush=True,
    )

    folds = assign_folds(data["group_id"], args.seed)
    score = np.full(data["class"].size, np.nan)
    fits = []
    model_records = []
    for fold in range(2):
        train_rows, stop_rows = fold_training_rows(
            data, folds, fold, args.madgraph_train_cap, args.seed
        )
        model, calibrator = fit_calibrated(
            data, train_rows, stop_rows, args.seed + fold
        )
        fits.append((model, calibrator))
        held_out = np.flatnonzero(folds == fold)
        real = held_out[data["class"][held_out] != 2]
        score[real] = plugin_score(
            calibrated_probabilities(model, calibrator, gather(data, real)), kappas
        )
        sample = gather(data, stop_rows[: min(64, stop_rows.size)])
        model_records.append(
            save_and_check_model(model, calibrator, sample, fold, model_dir)
        )
        print(f"fold {fold}: trained on {train_rows.size:,} rows", flush=True)

    ladder = ladder_edges(score[data["class"] == 0], args.ladder_bins)
    finite = np.isfinite(score)
    score_edges = np.linspace(
        float(np.floor(np.nanpercentile(score[finite], 0.05))),
        float(np.ceil(np.nanmax(score[finite]))),
        SCORE_RANGE_BINS + 1,
    )
    scan_thresholds = np.linspace(score_edges[0], score_edges[-1] - 1e-6, N_SCAN)

    madgraph_rows = np.flatnonzero(madgraph_mask)
    _groups, first = np.unique(data["group_id"][madgraph_rows], return_index=True)
    central = madgraph_rows[first]
    central_matrix = gather(data, central)
    rapidity = np.asarray(
        data["x"][central, rapidity_column], dtype=np.float64
    )
    central_weight = np.asarray(data["central_weight"][central], dtype=np.float64)

    ladder_yield = np.zeros(ladder.size - 1)
    ladder_squared = np.zeros(ladder.size - 1)
    above_yield = np.zeros(N_SCAN)
    above_squared = np.zeros(N_SCAN)
    residual = 0.0
    for fold, (model, calibrator) in enumerate(fits):
        selected = folds[central] == fold
        cell_score, cell_contribution, covered = sweep_pool(
            central_matrix[selected],
            rapidity[selected],
            central_weight[selected],
            pool,
            model,
            calibrator,
            kappas,
            pair_column,
            args.grid_cells,
        )
        residual = max(
            residual,
            float(
                np.max(
                    np.abs(
                        covered
                        - np.asarray(data["band_probability"][central][selected])
                    )
                )
            ),
        )
        block_yield, block_squared = accumulate_regions(
            cell_score, cell_contribution, ladder
        )
        ladder_yield += block_yield
        ladder_squared += block_squared
        block_yield, block_squared = accumulate_above(
            cell_score, cell_contribution, scan_thresholds
        )
        above_yield += block_yield
        above_squared += block_squared
        del cell_score, cell_contribution
    print(f"band-probability residual: {residual:.3e}", flush=True)

    signal_map = class_score_mass(data, score, 0, scan_thresholds)
    superchic_map = class_score_mass(data, score, 1, scan_thresholds)
    signal_cumulative = cumulative_from_top(signal_map)
    superchic_cumulative = cumulative_from_top(superchic_map)
    effective = np.divide(
        above_yield * above_yield,
        above_squared,
        out=np.zeros_like(above_yield),
        where=above_squared > 0.0,
    )
    scan_significance = np.asarray(
        [
            significance(
                signal_cumulative[index],
                superchic_cumulative[index],
                above_yield[index],
                madgraph_shape,
            )
            for index in range(N_SCAN)
        ]
    )
    scan_yields = np.vstack(
        (signal_cumulative.sum(axis=1), superchic_cumulative.sum(axis=1), above_yield)
    )
    valid = np.flatnonzero(effective >= MIN_EFFECTIVE_CENTRAL)
    if valid.size == 0:
        raise RuntimeError("No scan threshold satisfies the MadGraph support floor")
    best = valid[int(np.argmax(scan_significance[valid]))]
    operating = {
        "threshold": float(scan_thresholds[best]),
        "significance": float(scan_significance[best]),
        "signal_yield": float(scan_yields[0, best]),
        "superchic_yield": float(scan_yields[1, best]),
        "madgraph_yield": float(scan_yields[2, best]),
        "madgraph_effective_central_events": float(effective[best]),
    }
    background = operating["superchic_yield"] + operating["madgraph_yield"]
    operating["signal_over_background"] = operating["signal_yield"] / background
    operating["counting_significance"] = operating["signal_yield"] / np.sqrt(
        operating["signal_yield"] + background
    )

    with open(
        result_dir / "single_cut_scan.csv", "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "threshold",
                "mass_binned_significance",
                "signal_yield",
                "superchic_yield",
                "madgraph_yield",
                "madgraph_effective_central_events",
            ]
        )
        for index in range(N_SCAN):
            writer.writerow(
                [
                    scan_thresholds[index],
                    scan_significance[index],
                    scan_yields[0, index],
                    scan_yields[1, index],
                    scan_yields[2, index],
                    effective[index],
                ]
            )

    category_mass = []
    category_significance = []
    category_low = []
    category_high = []
    categories = []
    for index in range(ladder.size - 1):
        low, high = ladder[index], ladder[index + 1]
        rows = np.flatnonzero((score >= low) & (score < high))
        signal_rows = rows[data["class"][rows] == 0]
        superchic_rows = rows[data["class"][rows] == 1]
        signal_hist, _ = np.histogram(
            data["mx"][signal_rows],
            MASS_BINS,
            weights=data["physical_weight"][signal_rows],
        )
        superchic_hist, _ = np.histogram(
            data["mx"][superchic_rows],
            MASS_BINS,
            weights=data["physical_weight"][superchic_rows],
        )
        madgraph_hist = ladder_yield[index] * madgraph_shape
        spread = ladder_squared[index]
        display_low = float(low) if np.isfinite(low) else float(score_edges[0])
        display_high = float(high) if np.isfinite(high) else float(score_edges[-1])
        category_z = significance(
            signal_hist, superchic_hist, ladder_yield[index], madgraph_shape
        )
        histograms = np.stack((signal_hist, superchic_hist, madgraph_hist))
        category_mass.append(histograms)
        category_significance.append(category_z)
        category_low.append(display_low)
        category_high.append(display_high)
        categories.append(
            {
                "index": index,
                "score_range": [display_low, display_high],
                "signal_yield": float(signal_hist.sum()),
                "superchic_yield": float(superchic_hist.sum()),
                "madgraph_yield": float(madgraph_hist.sum()),
                "significance": category_z,
                "madgraph_effective_central_events": (
                    float(ladder_yield[index] ** 2 / spread) if spread > 0.0 else 0.0
                ),
                "mass_bins_gev": MASS_BINS,
                "signal_mass": signal_hist,
                "superchic_mass": superchic_hist,
                "madgraph_mass": madgraph_hist,
            }
        )
    category_mass = np.stack(category_mass)
    category_significance = np.asarray(category_significance)
    ladder_z = float(np.sqrt(np.sum(category_significance**2)))

    presel_signal, _ = np.histogram(
        data["mx"][data["class"] == 0],
        MASS_BINS,
        weights=data["physical_weight"][data["class"] == 0],
    )
    presel_superchic, _ = np.histogram(
        data["mx"][data["class"] == 1],
        MASS_BINS,
        weights=data["physical_weight"][data["class"] == 1],
    )
    presel_madgraph = ladder_yield.sum() * madgraph_shape
    preselection_mass = np.stack((presel_signal, presel_superchic, presel_madgraph))
    selected_mass = np.stack(
        (
            signal_cumulative[best],
            superchic_cumulative[best],
            above_yield[best] * madgraph_shape,
        )
    )
    score_yields = np.vstack(
        (
            signal_map.sum(axis=1),
            superchic_map.sum(axis=1),
            -np.diff(np.r_[above_yield, 0.0]),
        )
    )

    report = {
        "n_features": int(names.size),
        "features": names,
        "hyperparameters": PRODUCTION_PARAMS,
        "folds": 2,
        "grid_cells": args.grid_cells,
        "seed": args.seed,
        "kappas": kappas,
        "band_probability_residual": residual,
        "preselection_yields": {
            "signal": float(presel_signal.sum()),
            "superchic_qcdbb": float(presel_superchic.sum()),
            "madgraph_qcdbb": float(presel_madgraph.sum()),
        },
        "single_cut_operating_point": operating,
        "ladder_significance": ladder_z,
        "categories": categories,
        "note": (
            "Classifier-only, stat-only, perfectly known background. The v01 "
            "MadGraph campaign is dropped, so the parton pT 15-25 / |eta| up "
            "to 3 region is unmodelled. The combinatorial acceptance factor "
            "is a detector design parameter, not an uncertainty."
        ),
    }
    write_yaml(result_dir / "report.yaml", report)
    np.savez_compressed(
        result_dir / "report_data.npz",
        mass_bins=MASS_BINS,
        score_axis=scan_thresholds,
        score_yields=score_yields,
        scan_thresholds=scan_thresholds,
        scan_significance=scan_significance,
        scan_yields=scan_yields,
        scan_signal_mass=signal_cumulative,
        scan_superchic_mass=superchic_cumulative,
        madgraph_mass_shape=madgraph_shape,
        scan_madgraph_effective=effective,
        support_floor=np.asarray(MIN_EFFECTIVE_CENTRAL),
        operating_index=np.asarray(best),
        preselection_mass=preselection_mass,
        selected_mass=selected_mass,
        category_mass=category_mass,
        category_low=np.asarray(category_low),
        category_high=np.asarray(category_high),
        category_significance=category_significance,
    )
    write_yaml(
        model_dir / "model_manifest.yaml",
        {
            "purpose": "Archived two-fold out-of-fold evaluation models",
            "production_inference_model": False,
            "classes": CLASS_ORDER,
            "features": names,
            "fold_assignment_seed": args.seed,
            "folds": 2,
            "hyperparameters": {
                "n_estimators": N_ESTIMATORS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                **PRODUCTION_PARAMS,
            },
            "artifacts": model_records,
        },
    )
    print(
        f"operating point {operating['threshold']:.3f}: "
        f"S={operating['signal_yield']:.2f} B={background:.1f} "
        f"Z={operating['significance']:.4f}; ladder Z={ladder_z:.4f}; "
        f"done in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    print(f"Wrote {result_dir}", flush=True)


if __name__ == "__main__":
    main()
