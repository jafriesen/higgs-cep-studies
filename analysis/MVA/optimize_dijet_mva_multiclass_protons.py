#!/usr/bin/env python3
"""Train and evaluate the cached multiclass proton MVA by mass significance."""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.MVA.run_dijet_mva_multiclass_protons import (
    CLASS_ORDER,
    MASS_WINDOW_GEV,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    distance_correlation,
)

CONSTANT_FEATURES = {
    "jet1_ptd", "jet2_ptd", "jet1_width", "jet2_width",
}
BACKGROUND_CLASSES = (1, 2)
MASS_BINS = np.arange(MASS_WINDOW_GEV[0], MASS_WINDOW_GEV[1] + 1.0, 1.0)
NESTED_SIZES = (8, 12, 16, 24)
MIN_BACKGROUND_GROUPS = 50
MIN_BACKGROUND_NEFF = 25.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("develop", "final"), default="develop")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-madgraph-train-groups", type=int, default=250000)
    parser.add_argument(
        "--permutation-groups",
        type=int,
        default=None,
        help="Optional diagnostic-only cap for permutation screening; by default "
        "all validation groups are retained.",
    )
    parser.add_argument("--bootstrap-replicas", type=int, default=200)
    parser.add_argument(
        "--quick-significance",
        action="store_true",
        help="Skip feature and hyperparameter studies; fit one configured "
        "nonconstant feature set with the previously selected hyperparameters.",
    )
    parser.add_argument(
        "--quick-feature-set",
        choices=("all", "projected"),
        default="all",
        help="In quick mode, optionally replace the three legacy eta-strip/"
        "bridge features with their projected counterparts.",
    )
    parser.add_argument(
        "--model-config-summary",
        default=None,
        help="Previous summary supplying only selected_features and "
        "selected_hyperparameters for a locked quick retrain.",
    )
    parser.add_argument(
        "--locked-summary",
        default=None,
        help="Develop-mode summary containing the locked final configuration.",
    )
    parser.add_argument(
        "--resume-features",
        action="store_true",
        help="Reuse output-dir/checkpoint_features.yaml and continue with the search.",
    )
    parser.add_argument(
        "--resume-search",
        action="store_true",
        help="Reuse output-dir/checkpoint_search.yaml and refit only its winner.",
    )
    parser.add_argument(
        "--configuration-study-cache",
        default=None,
        help="Optional cache path that produced reused feature/search checkpoints, "
        "recorded for provenance.",
    )
    parser.add_argument(
        "--reuse-crossfit",
        action="store_true",
        help="In final mode, reuse crossfit_probabilities.npy and crossfit_folds.npy "
        "and only reevaluate the locked score regions.",
    )
    return parser.parse_args()


def load_cache(path):
    path = Path(path)
    with open(path / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "x", "class", "physical_weight", "training_mixture_weight",
            "group_id", "split", "mx", "source_campaign",
        )
    }
    arrays["feature_names"] = np.asarray(metadata["features"])
    arrays["metadata"] = metadata
    return arrays


def effective_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    denominator = np.sum(weights * weights)
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0.0 else 0.0


def assert_group_safe(labels, groups):
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_labels = labels[order]
    boundary = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], sorted_groups.size]
    for start, end in zip(starts, ends):
        if np.unique(sorted_labels[start:end]).size != 1:
            raise RuntimeError(
                f"Hard-event group {sorted_groups[start]} crosses data partitions"
            )


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


def cap_groups(indices, classes, groups, class_id, limit, rng):
    mask = classes[indices] == class_id
    candidates = np.unique(groups[indices[mask]])
    if limit is None or candidates.size <= limit:
        return indices, 1.0
    selected = rng.choice(candidates, size=limit, replace=False)
    keep = ~mask | np.isin(groups[indices], selected)
    return indices[keep], float(candidates.size / limit)


def development_indices(data, limit, seed, first_pass=False):
    indices = np.flatnonzero(data["split"] == 0)
    mg_rows = indices[data["class"][indices] == 2]
    _mg_groups, first = np.unique(data["group_id"][mg_rows], return_index=True)
    keep_mg_rows = mg_rows[first]
    indices = np.concatenate([
        indices[data["class"][indices] != 2],
        keep_mg_rows,
    ])
    cap = min(limit, 100000) if first_pass else limit
    return cap_groups(
        indices, data["class"], data["group_id"], 2, cap,
        np.random.default_rng(seed + (11 if first_pass else 17)),
    )


def make_model(params, seed):
    return XGBClassifier(
        n_estimators=800,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        max_bin=256,
        n_jobs=16,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=25,
        random_state=seed,
        **params,
    )


def fit_calibrated(
    data, feature_indices, train_indices, stop_indices, params, seed,
    train_mixture=None,
):
    train_weights = class_balanced_weights(
        data["class"][train_indices],
        (
            data["training_mixture_weight"][train_indices]
            if train_mixture is None
            else train_mixture
        ),
    )
    stop_weights = class_balanced_weights(
        data["class"][stop_indices],
        data["training_mixture_weight"][stop_indices],
    )
    model = make_model(params, seed)
    model.fit(
        data["x"][train_indices][:, feature_indices],
        data["class"][train_indices],
        sample_weight=train_weights,
        eval_set=[(
            data["x"][stop_indices][:, feature_indices],
            data["class"][stop_indices],
        )],
        sample_weight_eval_set=[stop_weights],
        verbose=False,
    )
    margins = model.predict(
        data["x"][stop_indices][:, feature_indices],
        output_margin=True,
    )
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        random_state=seed,
    )
    calibrator.fit(
        margins,
        data["class"][stop_indices],
        sample_weight=stop_weights,
    )
    return model, calibrator


def calibrated_probabilities(model, calibrator, x):
    margins = model.predict(x, output_margin=True)
    return calibrator.predict_proba(margins)


def calibrated_probabilities_for_indices(
    data, indices, feature_indices, model, calibrator, batch_size=250000,
    output=None,
):
    if output is None:
        output = np.empty((indices.size, len(CLASS_ORDER)), dtype=np.float32)
    if output.shape != (indices.size, len(CLASS_ORDER)):
        raise RuntimeError("Calibrated-probability output has the wrong shape")
    for start in range(0, indices.size, batch_size):
        stop = min(start + batch_size, indices.size)
        rows = indices[start:stop]
        output[start:stop] = calibrated_probabilities(
            model, calibrator, data["x"][rows][:, feature_indices]
        )
    return output


def split_weight(data, indices):
    split_id = int(data["split"][indices[0]])
    return np.asarray(data["physical_weight"][indices], dtype=np.float64) / SPLIT_FRACTIONS[split_id]


def physical_kappas(classes, weights):
    yields = np.bincount(classes, weights=weights, minlength=3)
    if yields[0] <= 0.0:
        raise RuntimeError("The evaluation split has non-positive signal yield")
    return yields[1:] / yields[0], yields


def plugin_score(probabilities, kappas):
    probabilities = np.clip(probabilities, 1.0e-12, 1.0)
    background = kappas[0] * probabilities[:, 1] + kappas[1] * probabilities[:, 2]
    return np.log(probabilities[:, 0]) - np.log(background)


def group_stats(classes, groups, weights, selection, class_id):
    mask = selection & (classes == class_id)
    if not np.any(mask):
        return {"groups": 0, "neff": 0.0}
    selected_groups, inverse = np.unique(groups[mask], return_inverse=True)
    group_weight = np.bincount(inverse, weights=weights[mask])
    return {
        "groups": int(selected_groups.size),
        "neff": effective_size(group_weight),
    }


def eligible(classes, groups, weights, selections, safety_factor=1.0):
    diagnostics = []
    valid = True
    for selection in selections:
        category = {}
        for class_id in BACKGROUND_CLASSES:
            stats = group_stats(classes, groups, weights, selection, class_id)
            category[CLASS_ORDER[class_id]] = stats
            if (
                stats["groups"] < safety_factor * MIN_BACKGROUND_GROUPS
                or stats["neff"] < safety_factor * MIN_BACKGROUND_NEFF
            ):
                valid = False
        diagnostics.append(category)
    return valid, diagnostics


def histogram_significance(classes, mass, weights, selections):
    z2 = 0.0
    yields = np.zeros(3, dtype=np.float64)
    for selection in selections:
        signal, _ = np.histogram(
            mass[selection & (classes == 0)], MASS_BINS,
            weights=weights[selection & (classes == 0)],
        )
        background, _ = np.histogram(
            mass[selection & (classes != 0)], MASS_BINS,
            weights=weights[selection & (classes != 0)],
        )
        denominator = signal + background
        z2 += np.sum(
            np.divide(
                signal * signal,
                denominator,
                out=np.zeros_like(signal),
                where=denominator > 0.0,
            )
        )
        for class_id in range(3):
            yields[class_id] += np.sum(weights[selection & (classes == class_id)])
    return float(np.sqrt(z2)), yields


def candidate_thresholds(score, classes):
    finite = score[np.isfinite(score)]
    quantiles = np.r_[
        np.linspace(0.0, 0.99, 51),
        0.995, 0.999, 0.9995, 0.9999, 1.0,
    ]
    sources = [np.quantile(finite, quantiles)]
    for class_id in range(len(CLASS_ORDER)):
        class_score = score[(classes == class_id) & np.isfinite(score)]
        if class_score.size:
            sources.append(np.quantile(class_score, quantiles))
    return np.unique(np.r_[-np.inf, *sources])


def optimize_single_cut(
    classes, groups, mass, weights, score, safety_factor=1.0,
    return_scan=False,
):
    best = None
    scan = []
    for threshold in candidate_thresholds(score, classes):
        selection = np.isfinite(mass) & (score >= threshold)
        valid, stats = eligible(
            classes, groups, weights, [selection], safety_factor
        )
        significance, yields = histogram_significance(
            classes, mass, weights, [selection]
        )
        candidate = {
            "threshold": float(threshold),
            "significance": significance,
            "yields": yields.tolist(),
            "background_mc": stats[0],
            "valid": bool(valid),
        }
        scan.append(candidate)
        if valid and (best is None or significance > best["significance"]):
            best = candidate
    if best is None:
        raise RuntimeError("No single-cut threshold meets group-level MC requirements")
    return (best, scan) if return_scan else best


def optimize_categories(
    classes, groups, mass, weights, score, safety_factor=1.0,
    return_scan=False, extra_boundaries=(),
):
    finite = score[np.isfinite(score)]
    quantiles = np.linspace(0.10, 0.90, 7)
    boundary_sources = [np.quantile(finite, quantiles)]
    for class_id in BACKGROUND_CLASSES:
        class_score = score[(classes == class_id) & np.isfinite(score)]
        if class_score.size:
            boundary_sources.append(np.quantile(class_score, quantiles))
    boundaries = np.unique(np.r_[np.concatenate(boundary_sources), extra_boundaries])
    best = None
    scan = []
    for low_index in range(boundaries.size - 1):
        for high_index in range(low_index + 1, boundaries.size):
            low, high = boundaries[low_index], boundaries[high_index]
            selections = [
                np.isfinite(mass) & (score < low),
                np.isfinite(mass) & (score >= low) & (score < high),
                np.isfinite(mass) & (score >= high),
            ]
            valid, stats = eligible(
                classes, groups, weights, selections, safety_factor
            )
            significance, yields = histogram_significance(
                classes, mass, weights, selections
            )
            candidate = {
                "boundaries": [float(low), float(high)],
                "significance": significance,
                "yields": yields.tolist(),
                "background_mc": stats,
                "valid": bool(valid),
            }
            scan.append(candidate)
            if valid and (best is None or significance > best["significance"]):
                best = candidate
    if best is None:
        raise RuntimeError("No three-category boundaries meet group-level MC requirements")
    return (best, scan) if return_scan else best


def locked_metrics(classes, groups, mass, weights, score, threshold, boundaries):
    single_selection = np.isfinite(mass) & (score >= threshold)
    single_valid, single_stats = eligible(
        classes, groups, weights, [single_selection]
    )
    low, high = boundaries
    categories = [
        np.isfinite(mass) & (score < low),
        np.isfinite(mass) & (score >= low) & (score < high),
        np.isfinite(mass) & (score >= high),
    ]
    category_valid, category_stats = eligible(
        classes, groups, weights, categories
    )
    if not single_valid or not category_valid:
        raise RuntimeError(
            "Locked score regions fail group-level MC requirements "
            f"(single_valid={single_valid}, category_valid={category_valid}, "
            f"single={single_stats}, categories={category_stats})"
        )
    single_z, single_yields = histogram_significance(
        classes, mass, weights, [single_selection]
    )
    category_z, category_yields = histogram_significance(
        classes, mass, weights, categories
    )
    s, b = single_yields[0], np.sum(single_yields[1:])
    return {
        "single_cut": {
            "threshold": float(threshold),
            "significance": single_z,
            "global_window_significance": float(s / np.sqrt(s + b)) if s + b > 0 else 0.0,
            "yields": single_yields.tolist(),
            "background_mc": single_stats[0],
        },
        "categories": {
            "boundaries": [float(low), float(high)],
            "significance": category_z,
            "yields": category_yields.tolist(),
            "background_mc": category_stats,
        },
    }


def optimize_metrics(
    data, indices, probabilities, safety_factor=1.0, include_scans=False
):
    classes = np.asarray(data["class"][indices])
    weights = split_weight(data, indices)
    kappas, yields = physical_kappas(classes, weights)
    score = plugin_score(probabilities, kappas)
    single_result = optimize_single_cut(
        classes, data["group_id"][indices], data["mx"][indices], weights, score,
        safety_factor, return_scan=include_scans,
    )
    if include_scans:
        single, single_scan = single_result
    else:
        single = single_result
    category_result = optimize_categories(
        classes, data["group_id"][indices], data["mx"][indices], weights, score,
        safety_factor, return_scan=include_scans,
        extra_boundaries=(single["threshold"],),
    )
    if include_scans:
        categories, category_scan = category_result
    else:
        categories = category_result
    result = {
        "single_cut": single,
        "categories": categories,
        "kappas": kappas.tolist(),
        "preselection_yields": yields.tolist(),
    }
    if include_scans:
        result["single_cut_scan"] = single_scan
        result["category_scan"] = category_scan
    return result, score


def model_metrics(data, feature_indices, train_indices, stop_indices, eval_indices, params, seed):
    model, calibrator = fit_calibrated(
        data, feature_indices, train_indices, stop_indices, params, seed
    )
    probabilities = calibrated_probabilities(
        model, calibrator, data["x"][eval_indices][:, feature_indices]
    )
    metrics, score = optimize_metrics(data, eval_indices, probabilities)
    metrics["best_iteration"] = int(model.best_iteration)
    return metrics, model, calibrator, probabilities, score


def metric_pair(metrics):
    return np.asarray([
        metrics["single_cut"]["significance"],
        metrics["categories"]["significance"],
    ])


def select_maximin(results):
    values = np.asarray([metric_pair(item["metrics"]) for item in results])
    best = np.max(values, axis=0)
    quality = np.min(values / best, axis=1)
    return int(np.argmax(quality))


def calibration_summary(calibrator):
    return {
        "classes": calibrator.classes_.astype(int).tolist(),
        "coefficients": calibrator.coef_.tolist(),
        "intercepts": calibrator.intercept_.tolist(),
        "iterations": calibrator.n_iter_.astype(int).tolist(),
    }


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_optimization_scans(output_dir, locked):
    with open(
        output_dir / "single_cut_scan_optimization.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = (
            "threshold", "valid", "significance", "signal_yield",
            "superchic_yield", "madgraph_yield", "superchic_groups",
            "superchic_neff", "madgraph_groups", "madgraph_neff",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in locked["single_cut_scan"]:
            writer.writerow({
                "threshold": point["threshold"],
                "valid": point["valid"],
                "significance": point["significance"],
                "signal_yield": point["yields"][0],
                "superchic_yield": point["yields"][1],
                "madgraph_yield": point["yields"][2],
                "superchic_groups": point["background_mc"]["QCDbb"]["groups"],
                "superchic_neff": point["background_mc"]["QCDbb"]["neff"],
                "madgraph_groups": point["background_mc"]["QCDbb_madgraph"]["groups"],
                "madgraph_neff": point["background_mc"]["QCDbb_madgraph"]["neff"],
            })
    with open(
        output_dir / "category_scan_optimization.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "low_boundary", "high_boundary", "valid", "significance",
            "signal_yield", "superchic_yield", "madgraph_yield",
        ]
        for category in range(3):
            for background in ("superchic", "madgraph"):
                fieldnames.extend([
                    f"category_{category}_{background}_groups",
                    f"category_{category}_{background}_neff",
                ])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in locked["category_scan"]:
            row = {
                "low_boundary": point["boundaries"][0],
                "high_boundary": point["boundaries"][1],
                "valid": point["valid"],
                "significance": point["significance"],
                "signal_yield": point["yields"][0],
                "superchic_yield": point["yields"][1],
                "madgraph_yield": point["yields"][2],
            }
            for category, stats in enumerate(point["background_mc"]):
                for key, label in (
                    ("QCDbb", "superchic"),
                    ("QCDbb_madgraph", "madgraph"),
                ):
                    row[f"category_{category}_{label}_groups"] = stats[key]["groups"]
                    row[f"category_{category}_{label}_neff"] = stats[key]["neff"]
            writer.writerow(row)


def permutation_ranking(
    data, feature_indices, model, calibrator, eval_indices, baseline, seed
):
    rng = np.random.default_rng(seed)
    x = np.array(data["x"][eval_indices][:, feature_indices], copy=True)
    impacts = {}
    baseline_pair = metric_pair(baseline)
    classes = np.asarray(data["class"][eval_indices])
    mass = np.asarray(data["mx"][eval_indices])
    weights = split_weight(data, eval_indices)
    kappas = np.asarray(baseline["kappas"])
    threshold = baseline["single_cut"]["threshold"]
    low, high = baseline["categories"]["boundaries"]
    for local_index, global_index in enumerate(feature_indices):
        original = x[:, local_index].copy()
        rng.shuffle(x[:, local_index])
        probabilities = calibrated_probabilities(model, calibrator, x)
        score = plugin_score(probabilities, kappas)
        single = np.isfinite(mass) & (score >= threshold)
        categories = [
            np.isfinite(mass) & (score < low),
            np.isfinite(mass) & (score >= low) & (score < high),
            np.isfinite(mass) & (score >= high),
        ]
        permuted_pair = np.asarray([
            histogram_significance(classes, mass, weights, [single])[0],
            histogram_significance(classes, mass, weights, categories)[0],
        ])
        impact = np.maximum(
            (baseline_pair - permuted_pair) / baseline_pair,
            0.0,
        )
        impacts[str(data["feature_names"][global_index])] = {
            "single_cut": float(impact[0]),
            "categories": float(impact[1]),
            "combined": float(np.mean(impact)),
        }
        x[:, local_index] = original
    return impacts


def correlation_screen(data, feature_indices, impacts, train_indices):
    rng = np.random.default_rng(923)
    rows = train_indices
    if rows.size > 100000:
        rows = rng.choice(rows, size=100000, replace=False)
    values = np.asarray(data["x"][rows][:, feature_indices], dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    missing = ~np.isfinite(values)
    values[missing] = np.take(medians, np.nonzero(missing)[1])
    correlation = np.corrcoef(values, rowvar=False)
    parent = np.arange(len(feature_indices))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left in range(len(feature_indices)):
        for right in range(left + 1, len(feature_indices)):
            if np.isfinite(correlation[left, right]) and abs(correlation[left, right]) >= 0.95:
                root_left, root_right = find(left), find(right)
                parent[root_right] = root_left
    components = {}
    for local_index, global_index in enumerate(feature_indices):
        components.setdefault(find(local_index), []).append(global_index)
    kept, removed = [], []
    for members in components.values():
        winner = max(
            members,
            key=lambda index: impacts[str(data["feature_names"][index])]["combined"],
        )
        kept.append(winner)
        removed.extend(index for index in members if index != winner)
    return np.asarray(kept, dtype=int), [
        str(data["feature_names"][index]) for index in removed
    ]


def feature_study(data, train_indices, stop_indices, eval_indices, seed):
    nonconstant = np.asarray([
        index for index, name in enumerate(data["feature_names"])
        if name not in CONSTANT_FEATURES
    ])
    default_params = {
        "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 20,
    }
    baseline, baseline_model, baseline_calibrator, _prob, _score = model_metrics(
        data, nonconstant, train_indices, stop_indices, eval_indices,
        default_params, seed,
    )
    print(
        "feature baseline: "
        f"Zcut={baseline['single_cut']['significance']:.4f} "
        f"Zcat={baseline['categories']['significance']:.4f} "
        f"iter={baseline['best_iteration']}",
        flush=True,
    )
    impacts = permutation_ranking(
        data, nonconstant, baseline_model, baseline_calibrator,
        eval_indices, baseline, seed + 1,
    )
    survivors, correlated_removed = correlation_screen(
        data, nonconstant, impacts, train_indices
    )
    ranked = sorted(
        survivors,
        key=lambda index: impacts[str(data["feature_names"][index])]["combined"],
        reverse=True,
    )
    sizes = [size for size in NESTED_SIZES if size < len(ranked)] + [len(ranked)]
    results = []
    for size in sizes:
        selected = np.asarray(ranked[:size])
        metrics, _model, _calibrator, _prob, _score = model_metrics(
            data, selected, train_indices, stop_indices, eval_indices,
            default_params, seed + size,
        )
        results.append({
            "size": size,
            "features": data["feature_names"][selected].tolist(),
            "metrics": metrics,
        })
        print(
            f"feature set {size}: "
            f"Zcut={metrics['single_cut']['significance']:.4f} "
            f"Zcat={metrics['categories']['significance']:.4f} "
            f"iter={metrics['best_iteration']}",
            flush=True,
        )
    values = np.asarray([metric_pair(item["metrics"]) for item in results])
    best = np.max(values, axis=0)
    within = np.flatnonzero(np.all(values >= 0.99 * best, axis=1))
    selected_result = (
        min((results[index] for index in within), key=lambda item: item["size"])
        if within.size
        else results[select_maximin(results)]
    )
    selected_indices = np.asarray([
        int(np.flatnonzero(data["feature_names"] == name)[0])
        for name in selected_result["features"]
    ])
    compact_passes_baseline = bool(
        np.all(metric_pair(selected_result["metrics"]) >= 0.99 * metric_pair(baseline))
    )
    if not compact_passes_baseline:
        selected_indices = nonconstant
        selected_result = {
            "size": int(nonconstant.size),
            "features": data["feature_names"][nonconstant].tolist(),
            "metrics": baseline,
        }
    return selected_indices, {
        "constant_removed": sorted(CONSTANT_FEATURES),
        "correlation_removed": correlated_removed,
        "permutation_impact": impacts,
        "baseline_all": baseline,
        "baseline_feature_count": int(nonconstant.size),
        "nested_results": results,
        "selection": selected_result,
        "compact_within_one_percent_both": bool(within.size),
        "compact_within_one_percent_of_full_baseline": compact_passes_baseline,
    }


def hyperparameter_study(
    data, features, first_train_indices, full_train_indices, stop_indices, seed
):
    first_results = []
    for depth in (3, 4, 5):
        for child in (1, 20):
            params = {
                "max_depth": depth,
                "learning_rate": 0.05,
                "min_child_weight": child,
            }
            metrics, _model, _cal, _prob, _score = model_metrics(
                data, features, first_train_indices, stop_indices, stop_indices,
                params, seed + depth * 10 + child,
            )
            first_results.append({"params": params, "metrics": metrics})
            print(
                f"search pass1 depth={depth} child={child}: "
                f"Zcut={metrics['single_cut']['significance']:.4f} "
                f"Zcat={metrics['categories']['significance']:.4f}",
                flush=True,
            )
    first_values = np.asarray([metric_pair(item["metrics"]) for item in first_results])
    first_best = np.max(first_values, axis=0)
    quality = np.min(first_values / first_best, axis=1)
    top = np.argsort(quality)[-2:][::-1]

    second_results = []
    selected_artifacts = []
    for rank, first_index in enumerate(top):
        base = first_results[first_index]["params"]
        for rate in (0.03, 0.05, 0.1):
            params = {**base, "learning_rate": rate}
            metrics, model, calibrator, probabilities, score = model_metrics(
                data, features, full_train_indices, stop_indices, stop_indices,
                params, seed + 100 + rank * 10 + int(rate * 100),
            )
            second_results.append({"params": params, "metrics": metrics})
            selected_artifacts.append((model, calibrator, probabilities, score))
            print(
                f"search pass2 depth={params['max_depth']} child="
                f"{params['min_child_weight']} lr={rate}: "
                f"Zcut={metrics['single_cut']['significance']:.4f} "
                f"Zcat={metrics['categories']['significance']:.4f}",
                flush=True,
            )
    best_index = select_maximin(second_results)
    return (
        second_results[best_index]["params"],
        selected_artifacts[best_index],
        {"first_pass": first_results, "second_pass": second_results},
    )


def pairwise_diagnostics(classes, probabilities, weights, score, mass, seed):
    diagnostics = {}
    for background in BACKGROUND_CLASSES:
        mask = (classes == 0) | (classes == background)
        label = (classes[mask] == 0).astype(np.int8)
        pair_score = probabilities[mask, 0] / (
            probabilities[mask, 0] + probabilities[mask, background]
        )
        auc_weights = np.asarray(weights[mask]).copy()
        for value in (0, 1):
            auc_weights[label == value] /= np.sum(auc_weights[label == value])
        auc = roc_auc_score(label, pair_score, sample_weight=auc_weights)
        tails = {}
        signal_score = pair_score[label == 1]
        signal_weight = auc_weights[label == 1]
        order = np.argsort(signal_score)[::-1]
        cumulative = np.cumsum(signal_weight[order])
        for efficiency in (0.5, 0.2, 0.1):
            threshold = signal_score[order][
                min(np.searchsorted(cumulative, efficiency), order.size - 1)
            ]
            background_mask = label == 0
            tails[str(efficiency)] = float(
                np.sum(auc_weights[background_mask & (pair_score >= threshold)])
                / np.sum(auc_weights[background_mask])
            )
        diagnostics[CLASS_ORDER[background]] = {
            "auc": float(auc),
            "background_efficiency_at_signal_efficiency": tails,
        }
    rng = np.random.default_rng(seed)
    diagnostics["score_mass_correlation"] = {
        CLASS_ORDER[class_id]: distance_correlation(
            score[classes == class_id],
            mass[classes == class_id],
            rng,
        )
        for class_id in BACKGROUND_CLASSES
    }
    return diagnostics


def bootstrap_interval(classes, groups, mass, weights, score, threshold, boundaries, replicas, seed):
    rng = np.random.default_rng(seed)
    group_maps = {}
    for class_id in range(3):
        rows = np.flatnonzero(classes == class_id)
        unique_groups, inverse = np.unique(groups[rows], return_inverse=True)
        group_maps[class_id] = (rows, unique_groups, inverse)
    single, category = [], []
    low, high = boundaries
    for _ in range(replicas):
        multiplier = np.zeros(classes.size, dtype=np.float64)
        for class_id, (rows, unique_groups, inverse) in group_maps.items():
            counts = rng.multinomial(
                unique_groups.size,
                np.full(unique_groups.size, 1.0 / unique_groups.size),
            )
            multiplier[rows] = counts[inverse]
        replica_weight = weights * multiplier
        selection = np.isfinite(mass) & (score >= threshold)
        categories = [
            np.isfinite(mass) & (score < low),
            np.isfinite(mass) & (score >= low) & (score < high),
            np.isfinite(mass) & (score >= high),
        ]
        single.append(histogram_significance(
            classes, mass, replica_weight, [selection]
        )[0])
        category.append(histogram_significance(
            classes, mass, replica_weight, categories
        )[0])
    return {
        "replicas": replicas,
        "single_cut_68_percent": np.percentile(single, [16, 50, 84]).tolist(),
        "categories_68_percent": np.percentile(category, [16, 50, 84]).tolist(),
    }


def normalization_scan(classes, groups, mass, weights, probabilities, nominal_kappas, threshold, boundaries):
    results = {}
    for scale in (0.5, 1.0, 2.0):
        scaled_weights = weights.copy()
        scaled_weights[classes == 2] *= scale
        kappas = np.asarray(nominal_kappas).copy()
        kappas[1] *= scale
        score = plugin_score(probabilities, kappas)
        results[str(scale)] = locked_metrics(
            classes, groups, mass, scaled_weights, score, threshold, boundaries
        )
    return results


def evaluate_locked(data, indices, probabilities, locked, replicas, seed):
    classes = np.asarray(data["class"][indices])
    groups = np.asarray(data["group_id"][indices])
    mass = np.asarray(data["mx"][indices])
    weights = split_weight(data, indices)
    score = plugin_score(probabilities, np.asarray(locked["kappas"]))
    metrics = locked_metrics(
        classes, groups, mass, weights, score,
        locked["single_cut"]["threshold"],
        locked["categories"]["boundaries"],
    )
    metrics["bootstrap"] = bootstrap_interval(
        classes, groups, mass, weights, score,
        locked["single_cut"]["threshold"],
        locked["categories"]["boundaries"],
        replicas, seed,
    )
    metrics["diagnostics"] = pairwise_diagnostics(
        classes, probabilities, weights, score, mass, seed + 1
    )
    metrics["madgraph_normalization_scan"] = normalization_scan(
        classes, groups, mass, weights, probabilities, locked["kappas"],
        locked["single_cut"]["threshold"],
        locked["categories"]["boundaries"],
    )
    return metrics


def data_summary(data):
    result = {}
    for split_id, split_name in enumerate(SPLIT_NAMES):
        split_mask = data["split"] == split_id
        result[split_name] = {}
        for class_id, class_name in enumerate(CLASS_ORDER):
            mask = split_mask & (data["class"] == class_id)
            group_weight = []
            if np.any(mask):
                _groups, inverse = np.unique(
                    data["group_id"][mask], return_inverse=True
                )
                group_weight = np.bincount(
                    inverse, weights=data["physical_weight"][mask]
                )
            result[split_name][class_name] = {
                "rows": int(np.sum(mask)),
                "groups": int(np.unique(data["group_id"][mask]).size),
                "effective_groups": effective_size(group_weight),
            }
    return result


def develop(args, data, output_dir):
    runtime = {}
    stage = time.perf_counter()
    train_indices, correction = development_indices(
        data, args.max_madgraph_train_groups, args.seed
    )
    first_train_indices, _first_correction = development_indices(
        data, args.max_madgraph_train_groups, args.seed, first_pass=True
    )
    stop_indices = np.flatnonzero(data["split"] == 1)
    optimization_indices = np.flatnonzero(data["split"] == 2)
    test_indices = np.flatnonzero(data["split"] == 3)
    # The cap is uniform in hard groups. Correct only the sampled MG training
    # mixture; physical evaluation weights are untouched.
    training_mixture = np.asarray(data["training_mixture_weight"])
    if correction != 1.0:
        training_mixture = training_mixture.copy()
        training_mixture[
            (data["split"] == 0) & (data["class"] == 2)
        ] *= correction
        data = {**data, "training_mixture_weight": training_mixture}
    runtime["load_and_split"] = time.perf_counter() - stage

    checkpoint_path = output_dir / "checkpoint_features.yaml"
    if args.resume_features:
        with open(checkpoint_path, encoding="utf-8") as handle:
            checkpoint = yaml.safe_load(handle)
        if checkpoint.get("available_features") != data["feature_names"].tolist():
            raise RuntimeError(
                "Feature checkpoint schema does not match this cache; rerun "
                "without --resume-features."
            )
        feature_report = checkpoint["feature_study"]
        selected_metrics = feature_report["selection"]["metrics"]
        baseline_metrics = feature_report["baseline_all"]
        compact_passes = bool(
            np.all(metric_pair(selected_metrics) >= 0.99 * metric_pair(baseline_metrics))
        )
        if compact_passes:
            features = np.asarray(checkpoint["selected_feature_indices"], dtype=int)
        else:
            features = np.asarray([
                index for index, name in enumerate(data["feature_names"])
                if name not in CONSTANT_FEATURES
            ])
            feature_report["selection"] = {
                "size": int(features.size),
                "features": data["feature_names"][features].tolist(),
                "metrics": baseline_metrics,
            }
        feature_report["compact_within_one_percent_of_full_baseline"] = compact_passes
        runtime.update(checkpoint.get("runtime_seconds", {}))
        print(
            f"resumed feature checkpoint: selected {features.size} variables",
            flush=True,
        )
    else:
        stage = time.perf_counter()
        feature_eval_indices, feature_eval_correction = cap_groups(
            stop_indices, data["class"], data["group_id"], 2,
            args.permutation_groups, np.random.default_rng(args.seed + 23),
        )
        feature_data = data
        if feature_eval_correction != 1.0:
            feature_physical_weight = np.asarray(data["physical_weight"]).copy()
            feature_physical_weight[
                (data["split"] == 1) & (data["class"] == 2)
            ] *= feature_eval_correction
            feature_data = {**data, "physical_weight": feature_physical_weight}
        features, feature_report = feature_study(
            feature_data, first_train_indices, stop_indices,
            feature_eval_indices, args.seed
        )
        feature_report["madgraph_evaluation_group_cap"] = args.permutation_groups
        feature_report["madgraph_evaluation_weight_correction"] = feature_eval_correction
        runtime["feature_study"] = time.perf_counter() - stage
        write_yaml(
            checkpoint_path,
            {
                "available_features": data["feature_names"].tolist(),
                "selected_features": data["feature_names"][features].tolist(),
                "selected_feature_indices": features.tolist(),
                "feature_study": feature_report,
                "runtime_seconds": runtime,
            },
        )

    stage = time.perf_counter()
    search_checkpoint_path = output_dir / "checkpoint_search.yaml"
    if args.resume_search:
        with open(search_checkpoint_path, encoding="utf-8") as handle:
            search_checkpoint = yaml.safe_load(handle)
        if search_checkpoint.get("available_features") != data["feature_names"].tolist():
            raise RuntimeError(
                "Search checkpoint schema does not match this cache; rerun "
                "without --resume-search."
            )
        params = search_checkpoint["selected_hyperparameters"]
        search_report = search_checkpoint["hyperparameter_search"]
        model, calibrator = fit_calibrated(
            data, features, train_indices, stop_indices, params, args.seed + 700
        )
        print(f"resumed search winner: {params}", flush=True)
        runtime["hyperparameter_search"] = search_checkpoint.get(
            "runtime_seconds", {}
        ).get(
            "full_hyperparameter_search",
            search_checkpoint.get("runtime_seconds", {}).get(
                "hyperparameter_search", float("nan")
            ),
        )
        runtime["selected_model_refit"] = time.perf_counter() - stage
    else:
        params, artifacts, search_report = hyperparameter_study(
            data, features, first_train_indices, train_indices, stop_indices, args.seed
        )
        model, calibrator, _stop_probabilities, _stop_score = artifacts
        runtime["hyperparameter_search"] = time.perf_counter() - stage
        runtime["full_hyperparameter_search"] = runtime[
            "hyperparameter_search"
        ]
    write_yaml(
        search_checkpoint_path,
        {
            "available_features": data["feature_names"].tolist(),
            "selected_features": data["feature_names"][features].tolist(),
            "selected_feature_indices": features.tolist(),
            "selected_hyperparameters": params,
            "hyperparameter_search": search_report,
            "runtime_seconds": runtime,
        },
    )

    stage = time.perf_counter()
    optimization_probabilities = calibrated_probabilities(
        model, calibrator, data["x"][optimization_indices][:, features]
    )
    locked, _optimization_score = optimize_metrics(
        data, optimization_indices, optimization_probabilities,
        safety_factor=3.0, include_scans=True,
    )
    write_optimization_scans(output_dir, locked)
    test_probabilities = calibrated_probabilities(
        model, calibrator, data["x"][test_indices][:, features]
    )
    test_metrics = evaluate_locked(
        data, test_indices, test_probabilities, locked,
        args.bootstrap_replicas, args.seed + 400,
    )
    runtime["locked_evaluation"] = time.perf_counter() - stage

    model.get_booster().save_model(output_dir / "develop_model.json")
    summary = {
        "mode": "develop",
        "configuration_study_cache": (
            args.configuration_study_cache or args.cache_dir
        ),
        "evaluation_cache": args.cache_dir,
        "cache_metadata": data["metadata"],
        "selected_features": data["feature_names"][features].tolist(),
        "selected_feature_indices": features.tolist(),
        "selected_hyperparameters": params,
        "feature_study": feature_report,
        "hyperparameter_search": search_report,
        "calibration": calibration_summary(calibrator),
        "locked_optimization": locked,
        "untouched_test": test_metrics,
        "training": {
            "madgraph_group_cap": args.max_madgraph_train_groups,
            "madgraph_sampling_weight_correction": correction,
            "best_iteration": int(model.best_iteration),
        },
        "data": data_summary(data),
        "runtime_seconds": runtime,
        "seed": args.seed,
    }
    return summary


def develop_quick(args, data, output_dir):
    """Fit one calibrated model and perform only locked significance evaluation."""
    runtime = {}
    stage = time.perf_counter()
    train_indices, correction = development_indices(
        data, args.max_madgraph_train_groups, args.seed
    )
    stop_indices = np.flatnonzero(data["split"] == 1)
    optimization_indices = np.flatnonzero(data["split"] == 2)
    test_indices = np.flatnonzero(data["split"] == 3)
    training_mixture = np.asarray(data["training_mixture_weight"])
    if correction != 1.0:
        training_mixture = training_mixture.copy()
        training_mixture[
            (data["split"] == 0) & (data["class"] == 2)
        ] *= correction
        data = {**data, "training_mixture_weight": training_mixture}
    feature_names = np.asarray(data["feature_names"])
    if args.model_config_summary:
        with open(args.model_config_summary, encoding="utf-8") as handle:
            model_config = yaml.safe_load(handle)
        selected_names = model_config["selected_features"]
        missing = sorted(set(selected_names) - set(feature_names.tolist()))
        if missing:
            raise RuntimeError(
                "Locked model features are absent from the cache: "
                + ", ".join(missing)
            )
        lookup = {name: index for index, name in enumerate(feature_names)}
        features = np.asarray([lookup[name] for name in selected_names], dtype=int)
        params = dict(model_config["selected_hyperparameters"])
    else:
        features = np.asarray([
            index for index, name in enumerate(data["feature_names"])
            if name not in CONSTANT_FEATURES
        ])
        if args.quick_feature_set == "projected":
            legacy = {
                "n_tracks_interjet",
                "sum_track_pt_interjet",
                "interjet_bridge_asymmetry",
            }
            features = features[
                ~np.isin(feature_names[features], list(legacy))
            ]
        params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_weight": 1,
        }
    runtime["load_and_split"] = time.perf_counter() - stage

    stage = time.perf_counter()
    model, calibrator = fit_calibrated(
        data, features, train_indices, stop_indices, params, args.seed + 700
    )
    runtime["single_model_fit"] = time.perf_counter() - stage

    stage = time.perf_counter()
    optimization_probabilities = calibrated_probabilities_for_indices(
        data, optimization_indices, features, model, calibrator
    )
    locked, _optimization_score = optimize_metrics(
        data, optimization_indices, optimization_probabilities,
        safety_factor=3.0, include_scans=True,
    )
    write_optimization_scans(output_dir, locked)
    test_probabilities = calibrated_probabilities_for_indices(
        data, test_indices, features, model, calibrator
    )
    test_metrics = evaluate_locked(
        data, test_indices, test_probabilities, locked,
        args.bootstrap_replicas, args.seed + 400,
    )
    runtime["locked_evaluation"] = time.perf_counter() - stage

    model.get_booster().save_model(output_dir / "quick_model.json")
    return {
        "mode": "develop",
        "workflow": "quick_significance",
        "quick_feature_set": args.quick_feature_set,
        "model_config_summary": args.model_config_summary,
        "evaluation_cache": args.cache_dir,
        "cache_metadata": data["metadata"],
        "selected_features": data["feature_names"][features].tolist(),
        "selected_feature_indices": features.tolist(),
        "selected_hyperparameters": params,
        "feature_study": {
            "skipped": True,
            "reason": (
                "--quick-significance uses the configured nonconstant feature set"
            ),
        },
        "hyperparameter_search": {
            "skipped": True,
            "reason": "reused the previous study winner",
        },
        "calibration": calibration_summary(calibrator),
        "locked_optimization": locked,
        "untouched_test": test_metrics,
        "training": {
            "madgraph_group_cap": args.max_madgraph_train_groups,
            "madgraph_sampling_weight_correction": correction,
            "best_iteration": int(model.best_iteration),
        },
        "data": data_summary(data),
        "runtime_seconds": runtime,
        "seed": args.seed,
    }


def final_crossfit(args, data, output_dir, locked_summary):
    features = np.asarray(locked_summary["selected_feature_indices"], dtype=int)
    params = locked_summary["selected_hyperparameters"]
    locked = locked_summary["locked_optimization"]
    classes = np.asarray(data["class"])
    groups = np.asarray(data["group_id"])
    started = time.perf_counter()
    if args.reuse_crossfit:
        probabilities = np.load(
            output_dir / "crossfit_probabilities.npy", mmap_mode="r"
        )
        folds = np.load(output_dir / "crossfit_folds.npy", mmap_mode="r")
        with open(output_dir / "summary_final.yaml", encoding="utf-8") as handle:
            previous_summary = yaml.safe_load(handle)
        fold_summary = previous_summary["crossfit"]
        if probabilities.shape != (classes.size, 3) or folds.shape != classes.shape:
            raise RuntimeError("Stored cross-fit arrays do not match the cache")
        assert_group_safe(folds, groups)
    else:
        group_values = np.unique(groups)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(group_values)
        group_fold = np.empty(int(np.max(group_values)) + 1, dtype=np.int8)
        group_fold[group_values] = np.arange(group_values.size) % 3
        folds = group_fold[groups]
        assert_group_safe(folds, groups)
        temporary_probabilities = output_dir / ".crossfit_probabilities.tmp.npy"
        probabilities = np.lib.format.open_memmap(
            temporary_probabilities,
            mode="w+",
            dtype=np.float32,
            shape=(classes.size, 3),
        )
        probabilities[:] = np.nan
        group_mixture_weight = np.bincount(
            groups,
            weights=np.asarray(data["training_mixture_weight"], dtype=np.float64),
        )
        fold_summary = []
        for fold in range(3):
            test_indices = np.flatnonzero(folds == fold)
            available = np.flatnonzero(folds != fold)
            # Deterministic inner 10% group holdout for early stopping/calibration.
            inner_groups = np.unique(groups[available])
            stop_groups = inner_groups[
                np.asarray(inner_groups, dtype=np.uint64) % np.uint64(10) == 0
            ]
            stop_indices = available[np.isin(groups[available], stop_groups)]
            train_candidates = available[~np.isin(groups[available], stop_groups)]
            _train_groups, first = np.unique(
                groups[train_candidates], return_index=True
            )
            train_indices = train_candidates[first]
            model, calibrator = fit_calibrated(
                data, features, train_indices, stop_indices,
                params, args.seed + fold,
                train_mixture=group_mixture_weight[groups[train_indices]],
            )
            fold_output = np.empty((test_indices.size, 3), dtype=np.float32)
            calibrated_probabilities_for_indices(
                data, test_indices, features, model, calibrator,
                output=fold_output,
            )
            probabilities[test_indices] = fold_output
            probabilities.flush()
            fold_summary.append({
                "fold": fold,
                "train_rows": int(train_indices.size),
                "calibration_rows": int(stop_indices.size),
                "test_rows": int(test_indices.size),
                "best_iteration": int(model.best_iteration),
                "calibration": calibration_summary(calibrator),
            })
            model.get_booster().save_model(output_dir / f"crossfit_fold_{fold}.json")
        probabilities.flush()
        temporary_probabilities.replace(output_dir / "crossfit_probabilities.npy")
        probabilities = np.load(
            output_dir / "crossfit_probabilities.npy", mmap_mode="r"
        )
    if np.any(~np.isfinite(probabilities)):
        raise RuntimeError("Final cross-fit left rows without probabilities")
    # All-fold weights already represent the full expected yield.
    kappas = np.asarray(locked["kappas"])
    score = plugin_score(probabilities, kappas)
    physical_weight = np.asarray(data["physical_weight"], dtype=np.float64)
    metrics = locked_metrics(
        classes, groups, data["mx"], physical_weight, score,
        locked["single_cut"]["threshold"],
        locked["categories"]["boundaries"],
    )
    metrics["bootstrap"] = bootstrap_interval(
        classes, groups, data["mx"], physical_weight, score,
        locked["single_cut"]["threshold"],
        locked["categories"]["boundaries"],
        args.bootstrap_replicas, args.seed + 500,
    )
    metrics["diagnostics"] = pairwise_diagnostics(
        classes, probabilities, physical_weight, score, data["mx"], args.seed + 501
    )
    metrics["madgraph_normalization_scan"] = normalization_scan(
        classes, groups, data["mx"], physical_weight, probabilities, kappas,
        locked["single_cut"]["threshold"],
        locked["categories"]["boundaries"],
    )
    if not args.reuse_crossfit:
        np.save(output_dir / "crossfit_folds.npy", folds)
    return {
        "mode": "final",
        "locked_from": args.locked_summary,
        "configuration_study_cache": locked_summary.get(
            "configuration_study_cache"
        ),
        "evaluation_cache": args.cache_dir,
        "selected_features": locked_summary["selected_features"],
        "selected_feature_indices": features.tolist(),
        "selected_hyperparameters": params,
        "locked_optimization": locked,
        "crossfit": fold_summary,
        "all_campaign_crossfit_evaluation": metrics,
        "data": data_summary(data),
        "runtime_seconds": {"three_fold_crossfit": time.perf_counter() - started},
        "crossfit_probabilities_reused": args.reuse_crossfit,
        "seed": args.seed,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_cache(args.cache_dir)
    assert_group_safe(data["split"], data["group_id"])
    if args.mode == "develop":
        summary = (
            develop_quick(args, data, output_dir)
            if args.quick_significance
            else develop(args, data, output_dir)
        )
        summary_path = output_dir / "summary_develop.yaml"
    else:
        if args.locked_summary is None:
            raise RuntimeError("--locked-summary is required in final mode")
        with open(args.locked_summary, encoding="utf-8") as handle:
            locked_summary = yaml.safe_load(handle)
        summary = final_crossfit(args, data, output_dir, locked_summary)
        summary_path = output_dir / "summary_final.yaml"
    write_yaml(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
