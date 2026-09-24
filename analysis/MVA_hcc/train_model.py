#!/usr/bin/env python3
"""Train, calibrate, and evaluate the five-class H(cc) proton MVA."""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import Booster, DMatrix, XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from analysis.MVA_hcc.common import plugin_score, write_yaml


DEFAULT_DATA = SCRIPT_DIR / "data"
DEFAULT_RESULTS = SCRIPT_DIR / "results"
MASS_BINS = np.arange(117.0, 134.0, 1.0)
MAX_ABS_RAPIDITY_DIFFERENCE = 0.2
PAIR_FEATURE = "yx_minus_dijet_rapidity"
RAPIDITY_FEATURE = "dijet_rapidity"
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
SCORE_BINS = 320
N_SCAN = 96


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--grid-cells", type=int, default=32)
    parser.add_argument("--nonexclusive-train-cap", type=int, default=250000)
    parser.add_argument("--support-floor", type=float, default=50.0)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    return parser.parse_args()


def validate_args(args):
    if args.ladder_bins <= 0 or args.grid_cells <= 0:
        raise ValueError("Ladder bins and grid cells must be positive")
    if args.nonexclusive_train_cap <= 0 or args.support_floor < 0.0:
        raise ValueError("Training cap must be positive and support floor non-negative")
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be positive")


def load_data(data_dir):
    with open(data_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if metadata.get("format_version") != 1:
        raise RuntimeError("Unsupported H(cc) dataset format")
    names = np.asarray(metadata["features"])
    if names.size != 20:
        raise RuntimeError(f"Expected 20 locked features, found {names.size}")
    if tuple(component["id"] for component in metadata["components"]) != tuple(
        range(len(metadata["components"]))
    ):
        raise RuntimeError("Component IDs must be contiguous and metadata ordered")
    arrays = {
        name: np.load(data_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "x", "class", "component", "group_id", "mx", "physical_weight",
            "training_mixture_weight", "central_weight", "band_probability",
        )
    }
    rows = arrays["class"].size
    if arrays["x"].shape != (rows, names.size):
        raise RuntimeError("Feature matrix has the wrong shape")
    for name, values in arrays.items():
        if name != "x" and values.shape != (rows,):
            raise RuntimeError(f"Array {name} has the wrong shape")
    n_classes = len(metadata["classes"])
    n_components = len(metadata["components"])
    if np.any(np.bincount(arrays["class"], minlength=n_classes) == 0):
        raise RuntimeError("Every training class must be nonempty")
    if np.any(np.bincount(arrays["component"], minlength=n_components) == 0):
        raise RuntimeError("Every physical component must be nonempty")
    arrays["feature_names"] = names
    arrays["metadata"] = metadata
    return arrays


def load_pool(data_dir):
    table = pq.read_table(data_dir / "proton_pairs.parquet", columns=["yx", "weight"])
    yx = np.asarray(table["yx"], dtype=np.float64)
    weight = np.asarray(table["weight"], dtype=np.float64)
    order = np.argsort(yx, kind="stable")
    cumulative = np.r_[0.0, np.cumsum(weight[order])]
    if yx.size == 0 or cumulative[-1] <= 0.0:
        raise RuntimeError("The proton-pair pool is empty or has non-positive weight")
    return {"yx": yx[order], "cumulative": cumulative, "total": cumulative[-1]}


def gather(data, rows):
    output = np.empty((rows.size, data["x"].shape[1]), dtype=np.float32)
    for start in range(0, rows.size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, rows.size)
        output[start:stop] = data["x"][rows[start:stop]]
    return output


def assert_group_safe(labels, groups):
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_labels = labels[order]
    if np.any(
        (sorted_groups[1:] == sorted_groups[:-1])
        & (sorted_labels[1:] != sorted_labels[:-1])
    ):
        raise RuntimeError("A hard-event group crosses folds")


def assign_folds(groups, seed, n_folds=2):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    assignment = np.arange(unique.size, dtype=np.int8) % n_folds
    rng.shuffle(assignment)
    folds = assignment[inverse]
    assert_group_safe(folds, groups)
    return folds


def cap_groups(rows, groups, limit, rng):
    candidates = np.unique(groups[rows])
    if candidates.size <= limit:
        return rows
    selected = rng.choice(candidates, size=limit, replace=False)
    return rows[np.isin(groups[rows], selected)]


def fold_training_rows(data, folds, fold, nonexclusive_class, cap, seed):
    available = np.flatnonzero(folds != fold)
    groups = data["group_id"]
    unique_groups = np.unique(groups[available])
    stop_groups = unique_groups[np.asarray(unique_groups, dtype=np.uint64) % np.uint64(10) == 0]
    is_stop = np.isin(groups[available], stop_groups)
    stop_rows = available[is_stop]
    candidates = available[~is_stop]
    nonexclusive = data["class"][candidates] == nonexclusive_class
    _groups, first = np.unique(groups[candidates[nonexclusive]], return_index=True)
    train_rows = np.concatenate((candidates[~nonexclusive], candidates[nonexclusive][first]))
    exclusive_rows = train_rows[data["class"][train_rows] != nonexclusive_class]
    nonexclusive_rows = train_rows[data["class"][train_rows] == nonexclusive_class]
    nonexclusive_rows = cap_groups(
        nonexclusive_rows, groups, cap, np.random.default_rng(seed + fold)
    )
    train_rows = np.sort(np.concatenate((exclusive_rows, nonexclusive_rows)))
    if train_rows.size == 0 or stop_rows.size == 0:
        raise RuntimeError("Training or calibration split is empty")
    return train_rows, np.sort(stop_rows)


def class_balanced_weights(classes, mixture, n_classes):
    weights = np.asarray(mixture, dtype=np.float64).copy()
    target = classes.size / n_classes
    for class_id in range(n_classes):
        mask = classes == class_id
        total = weights[mask].sum()
        if total <= 0.0:
            raise RuntimeError(f"Class {class_id} has no positive training weight")
        weights[mask] *= target / total
    return weights


def build_model(seed, n_estimators):
    return XGBClassifier(
        n_estimators=n_estimators,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=16,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=seed,
        **PRODUCTION_PARAMS,
    )


def fit_calibrated(data, train_rows, stop_rows, seed, n_estimators, n_classes):
    train_weight = class_balanced_weights(
        data["class"][train_rows], data["training_mixture_weight"][train_rows], n_classes
    )
    stop_weight = class_balanced_weights(
        data["class"][stop_rows], data["training_mixture_weight"][stop_rows], n_classes
    )
    stop_x = gather(data, stop_rows)
    model = build_model(seed, n_estimators)
    model.fit(
        gather(data, train_rows),
        data["class"][train_rows],
        sample_weight=train_weight,
        eval_set=[(stop_x, data["class"][stop_rows])],
        sample_weight_eval_set=[stop_weight],
        verbose=False,
    )
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
    calibrator.fit(
        model.predict(
            stop_x,
            output_margin=True,
            iteration_range=(0, int(model.best_iteration) + 1),
        ),
        data["class"][stop_rows],
        sample_weight=stop_weight,
    )
    if not np.array_equal(calibrator.classes_, np.arange(n_classes)):
        raise RuntimeError("Calibration split does not contain every class")
    return model, calibrator


def calibrated_probabilities(model, calibrator, matrix, n_classes):
    output = np.empty((matrix.shape[0], n_classes), dtype=np.float64)
    for start in range(0, matrix.shape[0], BATCH_ROWS):
        stop = min(start + BATCH_ROWS, matrix.shape[0])
        margins = model.predict(
            matrix[start:stop],
            output_margin=True,
            iteration_range=(0, int(model.best_iteration) + 1),
        )
        output[start:stop] = calibrator.predict_proba(margins)
    return output


def calibration_payload(calibrator):
    return {
        "classes": calibrator.classes_.astype(int),
        "dtype": str(calibrator.coef_.dtype),
        "coefficients": calibrator.coef_,
        "intercepts": calibrator.intercept_,
        "iterations": calibrator.n_iter_.astype(int),
    }


def portable_calibration(margins, payload):
    dtype = np.dtype(payload.get("dtype", "float64"))
    coefficients = np.asarray(payload["coefficients"], dtype=dtype)
    intercepts = np.asarray(payload["intercepts"], dtype=dtype)
    logits = np.asarray(margins, dtype=dtype) @ coefficients.T + intercepts
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def save_and_check_model(model, calibrator, sample, fold, model_dir):
    model_path = model_dir / f"fold_{fold}_model.json"
    calibration_path = model_dir / f"fold_{fold}_calibration.yaml"
    model.get_booster().save_model(model_path)
    payload = calibration_payload(calibrator)
    write_yaml(calibration_path, payload)
    restored = Booster()
    restored.load_model(model_path)
    original_margin = model.predict(
        sample,
        output_margin=True,
        iteration_range=(0, int(model.best_iteration) + 1),
    )
    restored_margin = restored.predict(
        DMatrix(sample),
        output_margin=True,
        iteration_range=(0, int(model.best_iteration) + 1),
    )
    expected = calibrator.predict_proba(original_margin)
    actual = portable_calibration(restored_margin, payload)
    return {
        "fold": fold,
        "model": model_path.name,
        "calibration": calibration_path.name,
        "best_iteration": int(model.best_iteration),
        "reload_margin_max_abs_difference": np.max(np.abs(original_margin - restored_margin)),
        "reload_probability_max_abs_difference": np.max(np.abs(expected - actual)),
    }


def pool_mass_between(pool, low, high):
    left = np.searchsorted(pool["yx"], low, side="left")
    right = np.searchsorted(pool["yx"], high, side="left")
    return pool["cumulative"][right] - pool["cumulative"][left]


def sweep_pool(matrix, rapidity, weight, pool, model, calibrator, kappas, pair_column, cells, n_classes):
    score = np.empty((matrix.shape[0], cells), dtype=np.float32)
    contribution = np.empty((matrix.shape[0], cells), dtype=np.float64)
    covered = np.zeros(matrix.shape[0], dtype=np.float64)
    bounds = np.linspace(-MAX_ABS_RAPIDITY_DIFFERENCE, MAX_ABS_RAPIDITY_DIFFERENCE, cells + 1)
    for index, (low, high) in enumerate(zip(bounds[:-1], bounds[1:])):
        matrix[:, pair_column] = 0.5 * (low + high)
        score[:, index] = plugin_score(
            calibrated_probabilities(model, calibrator, matrix, n_classes), kappas
        )
        mass = pool_mass_between(pool, rapidity + low, rapidity + high) / pool["total"]
        covered += mass
        contribution[:, index] = weight * mass
    return score, contribution, covered


def accumulate_regions(score, contribution, edges):
    size = edges.size - 1
    cell = np.clip(np.searchsorted(edges, score.ravel(), side="right") - 1, 0, size - 1)
    rows = np.repeat(np.arange(score.shape[0]), score.shape[1])
    per_event = np.bincount(
        rows * size + cell,
        weights=contribution.ravel(),
        minlength=score.shape[0] * size,
    ).reshape(score.shape[0], size)
    return per_event.sum(axis=0), np.sum(per_event * per_event, axis=0)


def accumulate_above(score, contribution, thresholds):
    total = np.zeros(thresholds.size)
    squared = np.zeros(thresholds.size)
    for index, threshold in enumerate(thresholds):
        per_event = np.where(score >= threshold, contribution, 0.0).sum(axis=1)
        total[index] = per_event.sum()
        squared[index] = np.sum(per_event * per_event)
    return total, squared


def ladder_edges(signal_score, n_bins):
    finite = signal_score[np.isfinite(signal_score)]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.r_[-np.inf, np.quantile(finite, quantiles), np.inf])


def cumulative_from_top(values):
    return np.cumsum(values[::-1], axis=0)[::-1]


def mass_significance(component_mass):
    signal = component_mass[0]
    background = component_mass[1:].sum(axis=0)
    total = signal + background
    return float(
        np.sqrt(
            np.sum(np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0))
        )
    )


def component_score_mass(data, score, component_id, score_edges):
    rows = np.flatnonzero(data["component"] == component_id)
    histogram, _, _ = np.histogram2d(
        score[rows], data["mx"][rows], bins=(np.r_[score_edges, np.inf], MASS_BINS),
        weights=data["physical_weight"][rows],
    )
    return histogram


def probability_histograms(data, probabilities, n_classes, bins):
    output = np.zeros((n_classes, n_classes, bins.size - 1), dtype=np.float64)
    for truth in range(n_classes):
        rows = data["class"] == truth
        for predicted in range(n_classes):
            output[truth, predicted], _ = np.histogram(
                probabilities[rows, predicted], bins=bins,
                weights=data["physical_weight"][rows],
            )
    return output


def main():
    args = parse_args()
    validate_args(args)
    data_dir = Path(args.data_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    model_dir = result_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = load_data(data_dir)
    pool = load_pool(data_dir)
    metadata = data["metadata"]
    class_names = metadata["classes"]
    components = metadata["components"]
    n_classes, n_components = len(class_names), len(components)
    nonexclusive_class = class_names.index("nonexclusive_QCD")
    pair_column = int(np.flatnonzero(data["feature_names"] == PAIR_FEATURE)[0])
    rapidity_column = int(np.flatnonzero(data["feature_names"] == RAPIDITY_FEATURE)[0])
    class_yields = np.bincount(
        data["class"], weights=data["physical_weight"], minlength=n_classes
    )
    if class_yields[0] <= 0.0:
        raise RuntimeError("Signal has non-positive yield")
    kappas = class_yields[1:] / class_yields[0]

    folds = assign_folds(data["group_id"], args.seed)
    score = np.full(data["class"].size, np.nan, dtype=np.float64)
    probabilities = np.full((data["class"].size, n_classes), np.nan, dtype=np.float32)
    fits = []
    model_records = []
    for fold in range(2):
        train_rows, stop_rows = fold_training_rows(
            data, folds, fold, nonexclusive_class, args.nonexclusive_train_cap, args.seed
        )
        model, calibrator = fit_calibrated(
            data, train_rows, stop_rows, args.seed + fold, args.n_estimators, n_classes
        )
        fits.append((model, calibrator))
        held_out = np.flatnonzero(folds == fold)
        held_prob = calibrated_probabilities(
            model, calibrator, gather(data, held_out), n_classes
        )
        probabilities[held_out] = held_prob
        score[held_out] = plugin_score(held_prob, kappas)
        sample = gather(data, stop_rows[: min(64, stop_rows.size)])
        model_records.append(save_and_check_model(model, calibrator, sample, fold, model_dir))
        print(f"fold {fold}: trained on {train_rows.size:,} rows", flush=True)
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(score)):
        raise RuntimeError("Cross-fit scoring produced non-finite results")

    signal_rows = data["component"] == 0
    ladder = ladder_edges(score[signal_rows], args.ladder_bins)
    if ladder.size != args.ladder_bins + 1:
        raise RuntimeError("Signal score quantiles did not produce distinct category edges")
    score_min = float(np.floor(np.percentile(score, 0.05)))
    score_max = float(np.ceil(np.max(score)))
    score_edges = np.linspace(score_min, score_max, SCORE_BINS)
    thresholds = np.linspace(score_min, score_max - 1.0e-6, N_SCAN)

    is_madgraph_component = np.asarray([item["generator"] == "madgraph" for item in components])
    madgraph_rows = np.flatnonzero(is_madgraph_component[data["component"]])
    _groups, first = np.unique(data["group_id"][madgraph_rows], return_index=True)
    central = madgraph_rows[first]
    central_matrix = gather(data, central)
    rapidity = np.asarray(data["x"][central, rapidity_column], dtype=np.float64)
    central_weight = np.asarray(data["central_weight"][central], dtype=np.float64)
    above_yield = np.zeros((n_components, thresholds.size))
    above_squared = np.zeros((n_components, thresholds.size))
    ladder_yield = np.zeros((n_components, ladder.size - 1))
    ladder_squared = np.zeros((n_components, ladder.size - 1))
    residual = 0.0
    for fold, (model, calibrator) in enumerate(fits):
        for component_id in np.flatnonzero(is_madgraph_component):
            selected = (folds[central] == fold) & (data["component"][central] == component_id)
            if not np.any(selected):
                raise RuntimeError(f"Fold {fold} has no central events for component {component_id}")
            cell_score, contribution, covered = sweep_pool(
                central_matrix[selected].copy(), rapidity[selected], central_weight[selected], pool,
                model, calibrator, kappas, pair_column, args.grid_cells, n_classes,
            )
            residual = max(
                residual,
                float(np.max(np.abs(covered - data["band_probability"][central][selected]))),
            )
            ladder_yield[component_id], ladder_squared[component_id] = (
                np.asarray((ladder_yield[component_id], ladder_squared[component_id]))
                + np.asarray(accumulate_regions(cell_score, contribution, ladder))
            )
            above_yield[component_id], above_squared[component_id] = (
                np.asarray((above_yield[component_id], above_squared[component_id]))
                + np.asarray(accumulate_above(cell_score, contribution, thresholds))
            )
    print(f"band-probability residual: {residual:.3e}", flush=True)

    component_mass_shape = np.zeros((n_components, MASS_BINS.size - 1))
    for component_id in np.flatnonzero(is_madgraph_component):
        rows = data["component"] == component_id
        shape, _ = np.histogram(
            data["mx"][rows], MASS_BINS, weights=data["physical_weight"][rows]
        )
        if shape.sum() <= 0.0:
            raise RuntimeError(f"MadGraph component {component_id} has no mass support")
        component_mass_shape[component_id] = shape / shape.sum()

    score_mass = np.zeros((n_components, score_edges.size, MASS_BINS.size - 1))
    scan_mass = np.zeros((thresholds.size, n_components, MASS_BINS.size - 1))
    score_yields = np.zeros((n_components, score_edges.size))
    for component_id in range(n_components):
        if is_madgraph_component[component_id]:
            interpolated = np.vstack(
                [np.interp(score_edges, thresholds, above_yield[component_id])]
            )[0]
            score_yields[component_id] = -np.diff(np.r_[interpolated, 0.0])
            scan_mass[:, component_id] = (
                above_yield[component_id, :, None] * component_mass_shape[component_id]
            )
        else:
            score_mass[component_id] = component_score_mass(
                data, score, component_id, score_edges
            )
            score_yields[component_id] = score_mass[component_id].sum(axis=1)
            cumulative = cumulative_from_top(
                component_score_mass(data, score, component_id, thresholds)
            )
            scan_mass[:, component_id] = cumulative

    total_above = above_yield.sum(axis=0)
    total_squared = above_squared.sum(axis=0)
    effective = np.divide(
        total_above**2, total_squared, out=np.zeros_like(total_above), where=total_squared > 0.0
    )
    scan_significance = np.asarray([mass_significance(item) for item in scan_mass])
    scan_component_yields = scan_mass.sum(axis=2).T
    valid = np.flatnonzero(effective >= args.support_floor)
    if valid.size == 0:
        raise RuntimeError("No threshold satisfies the MadGraph support floor")
    best = valid[int(np.argmax(scan_significance[valid]))]
    selected_yields = scan_component_yields[:, best]
    signal_yield = selected_yields[0]
    background_yield = selected_yields[1:].sum()
    operating = {
        "threshold": thresholds[best],
        "significance": scan_significance[best],
        "signal_yield": signal_yield,
        "background_yield": background_yield,
        "signal_over_background": signal_yield / background_yield,
        "counting_significance": signal_yield / np.sqrt(signal_yield + background_yield),
        "madgraph_effective_central_events": effective[best],
        "component_yields": {
            component["name"]: selected_yields[index]
            for index, component in enumerate(components)
        },
    }

    preselection_mass = np.zeros((n_components, MASS_BINS.size - 1))
    selected_mass = scan_mass[best]
    for component_id in range(n_components):
        if is_madgraph_component[component_id]:
            preselection_mass[component_id] = (
                ladder_yield[component_id].sum() * component_mass_shape[component_id]
            )
        else:
            rows = data["component"] == component_id
            preselection_mass[component_id], _ = np.histogram(
                data["mx"][rows], MASS_BINS, weights=data["physical_weight"][rows]
            )

    category_mass = np.zeros((ladder.size - 1, n_components, MASS_BINS.size - 1))
    category_significance = np.zeros(ladder.size - 1)
    for category, (low, high) in enumerate(zip(ladder[:-1], ladder[1:])):
        for component_id in range(n_components):
            if is_madgraph_component[component_id]:
                category_mass[category, component_id] = (
                    ladder_yield[component_id, category] * component_mass_shape[component_id]
                )
            else:
                rows = (
                    (data["component"] == component_id) & (score >= low) & (score < high)
                )
                category_mass[category, component_id], _ = np.histogram(
                    data["mx"][rows], MASS_BINS, weights=data["physical_weight"][rows]
                )
        category_significance[category] = mass_significance(category_mass[category])

    probability_bins = np.linspace(0.0, 1.0, 51)
    probability_hist = probability_histograms(data, probabilities, n_classes, probability_bins)
    component_names = {component["name"] for component in components}
    notes = ["Classifier-only, stat-only, perfectly known nominal backgrounds."]
    if metadata.get("profile") == "cc_only":
        notes.append("Charm-only comparison; all bottom-flavor components are excluded.")
    elif "QCDbb_madgraph" in component_names:
        notes.append("QCDbb MadGraph v01 is excluded.")
    if "QCDcc_madgraph" in component_names:
        notes.append("QCDcc MadGraph coverage is limited to v01.")
    report = {
        "format_version": 1,
        "profile": metadata.get("profile", "nominal_five_class"),
        "features": metadata["features"],
        "classes": class_names,
        "components": components,
        "hyperparameters": {**PRODUCTION_PARAMS, "n_estimators": args.n_estimators},
        "folds": 2,
        "seed": args.seed,
        "grid_cells": args.grid_cells,
        "support_floor": args.support_floor,
        "kappas": kappas,
        "preselection_class_yields": dict(zip(class_names, class_yields)),
        "preselection_component_yields": metadata["physical_yields_per_component"],
        "band_probability_residual": residual,
        "single_cut_operating_point": operating,
        "ladder_significance": np.sqrt(np.sum(category_significance**2)),
        "tagging": metadata["tagging"],
        "note": " ".join(notes),
    }
    write_yaml(result_dir / "report.yaml", report)
    np.savez_compressed(
        result_dir / "oof_results.npz",
        probabilities=probabilities,
        score=score,
        fold=folds,
        class_id=np.asarray(data["class"]),
        component_id=np.asarray(data["component"]),
        group_id=np.asarray(data["group_id"]),
        mx=np.asarray(data["mx"]),
        physical_weight=np.asarray(data["physical_weight"]),
    )
    np.savez_compressed(
        result_dir / "report_data.npz",
        mass_bins=MASS_BINS,
        score_axis=score_edges,
        score_yields=score_yields,
        probability_bins=probability_bins,
        probability_histograms=probability_hist,
        scan_thresholds=thresholds,
        scan_significance=scan_significance,
        scan_component_yields=scan_component_yields,
        scan_component_mass=scan_mass,
        scan_madgraph_effective=effective,
        support_floor=np.asarray(args.support_floor),
        operating_index=np.asarray(best),
        preselection_mass=preselection_mass,
        selected_mass=selected_mass,
        category_mass=category_mass,
        category_low=ladder[:-1],
        category_high=ladder[1:],
        category_significance=category_significance,
    )
    with open(result_dir / "single_cut_scan.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["threshold", "mass_binned_significance", "madgraph_effective_central_events"]
            + [component["name"] for component in components]
        )
        for index, threshold in enumerate(thresholds):
            writer.writerow(
                [threshold, scan_significance[index], effective[index]]
                + scan_component_yields[:, index].tolist()
            )
    write_yaml(
        model_dir / "model_manifest.yaml",
        {
            "purpose": "Archived two-fold out-of-fold H(cc) evaluation models",
            "production_inference_model": False,
            "classes": class_names,
            "features": metadata["features"],
            "fold_assignment_seed": args.seed,
            "artifacts": model_records,
        },
    )
    print(
        f"operating point {operating['threshold']:.3f}: S={signal_yield:.3g} "
        f"B={background_yield:.3g} Z={operating['significance']:.4f}; "
        f"done in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    print(f"Wrote {result_dir}", flush=True)


if __name__ == "__main__":
    main()
