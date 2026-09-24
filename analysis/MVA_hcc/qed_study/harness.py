"""Fast evaluation harness for H(cc) score-architecture variants.

The expensive part of the nominal workflow is scoring every MadGraph central
event once per proton-pool grid cell. That cost does not depend on how the class
probabilities are later combined into a score, so this module does it once and
caches the calibrated probabilities per (central event, grid cell, class). Every
score definition, kappa choice and categorisation afterwards is a numpy pass.

The training class assignment is a parameter (`class_map`), so alternative class
structures are compared without rebuilding the dataset.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parents[2]) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.common import plugin_score  # noqa: E402
from analysis.MVA_hcc.qed_study.common import (  # noqa: E402
    MASS_BINS, MAX_ABS_RAPIDITY_DIFFERENCE, assign_folds, balanced_weights, load_dataset,
)

PRODUCTION_PARAMS = {
    "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "gamma": 0.0, "max_bin": 256,
}
N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 40
BATCH_ROWS = 250000
EVENT_CHUNK = 200000
PAIR_FEATURE = "yx_minus_dijet_rapidity"
RAPIDITY_FEATURE = "dijet_rapidity"


# ---------------------------------------------------------------- data access

def read_columns(matrix, columns):
    """Sequential read of a column subset into RAM."""
    columns = np.asarray(columns)
    output = np.empty((matrix.shape[0], columns.size), dtype=np.float32)
    for start in range(0, matrix.shape[0], BATCH_ROWS):
        stop = min(start + BATCH_ROWS, matrix.shape[0])
        output[start:stop] = np.asarray(matrix[start:stop])[:, columns]
    return output


def load_pool(data_dir):
    table = pq.read_table(Path(data_dir) / "proton_pairs.parquet", columns=["yx", "weight"])
    yx = np.asarray(table["yx"], dtype=np.float64)
    weight = np.asarray(table["weight"], dtype=np.float64)
    order = np.argsort(yx, kind="stable")
    cumulative = np.r_[0.0, np.cumsum(weight[order])]
    if yx.size == 0 or cumulative[-1] <= 0.0:
        raise RuntimeError("The proton-pair pool is empty or has non-positive weight")
    return {"yx": yx[order], "cumulative": cumulative, "total": cumulative[-1]}


def pool_mass_between(pool, low, high):
    left = np.searchsorted(pool["yx"], low, side="left")
    right = np.searchsorted(pool["yx"], high, side="left")
    return pool["cumulative"][right] - pool["cumulative"][left]


# -------------------------------------------------------------------- fitting

def cap_pooled_groups(rows, groups, is_pooled_class, cap, rng):
    """Keep every non-pooled row and at most `cap` pooled hard-event groups."""
    pooled = rows[is_pooled_class[rows]]
    other = rows[~is_pooled_class[rows]]
    candidates = np.unique(groups[pooled])
    if candidates.size > cap:
        keep = rng.choice(candidates, size=cap, replace=False)
        pooled = pooled[np.isin(groups[pooled], keep)]
    return np.sort(np.concatenate((other, pooled)))


def fold_training_rows(data, folds, fold, is_pooled_class, cap, seed, stop_cap=None):
    """Training and early-stopping/calibration rows for one fold.

    Mirrors analysis/MVA_hcc/train_model.py: a deterministic every-tenth-group
    inner holdout, one row per pooled-background group, and a group cap.
    """
    available = np.flatnonzero(folds != fold)
    groups = data["group_id"]
    unique_groups = np.unique(groups[available])
    stop_groups = unique_groups[np.asarray(unique_groups, dtype=np.uint64) % np.uint64(10) == 0]
    is_stop = np.isin(groups[available], stop_groups)
    stop_rows = available[is_stop]
    candidates = available[~is_stop]
    pooled = is_pooled_class[candidates]
    _groups, first = np.unique(groups[candidates[pooled]], return_index=True)
    train_rows = np.concatenate((candidates[~pooled], candidates[pooled][first]))
    train_rows = cap_pooled_groups(train_rows, groups, is_pooled_class, cap,
                                   np.random.default_rng(seed + fold))
    # The early-stopping and calibration split is otherwise ~90% pooled rows and
    # is re-evaluated every boosting round, which dominates the fit. Neither use
    # needs the full sample.
    if stop_cap is not None:
        stop_rows = cap_pooled_groups(stop_rows, groups, is_pooled_class, stop_cap,
                                      np.random.default_rng(seed + 500 + fold))
    if train_rows.size == 0 or stop_rows.size == 0:
        raise RuntimeError("Training or calibration split is empty")
    return train_rows, np.sort(stop_rows)


def fit_calibrated(matrix, labels, mixture, train_rows, stop_rows, n_classes, seed, n_estimators):
    train_weight = balanced_weights(labels[train_rows], mixture[train_rows], n_classes)
    stop_weight = balanced_weights(labels[stop_rows], mixture[stop_rows], n_classes)
    stop_x = matrix[stop_rows]
    model = XGBClassifier(
        n_estimators=n_estimators, objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", n_jobs=16, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=seed, **PRODUCTION_PARAMS,
    )
    model.fit(matrix[train_rows], labels[train_rows], sample_weight=train_weight,
              eval_set=[(stop_x, labels[stop_rows])], sample_weight_eval_set=[stop_weight],
              verbose=False)
    margins = model.predict(stop_x, output_margin=True,
                            iteration_range=(0, int(model.best_iteration) + 1))
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
    calibrator.fit(margins, labels[stop_rows], sample_weight=stop_weight)
    if not np.array_equal(calibrator.classes_, np.arange(n_classes)):
        raise RuntimeError("Calibration split does not contain every class")
    return model, calibrator


def calibrated_probabilities(model, calibrator, matrix, n_classes):
    output = np.empty((matrix.shape[0], n_classes), dtype=np.float64)
    for start in range(0, matrix.shape[0], BATCH_ROWS):
        stop = min(start + BATCH_ROWS, matrix.shape[0])
        margins = model.predict(matrix[start:stop], output_margin=True,
                                iteration_range=(0, int(model.best_iteration) + 1))
        output[start:stop] = calibrator.predict_proba(margins)
    return output


# ------------------------------------------------------------------ the cache

def build_evaluation(data_dir, class_map, class_names, seed=12345, grid_cells=32,
                     cap=250000, n_estimators=N_ESTIMATORS, feature_names=None,
                     stop_cap=40000, train_rows_mask=None, verbose=True):
    """Fit both folds and cache everything a score definition needs.

    class_map[component_id] -> training class id. Class 0 must be the signal.

    train_rows_mask, if given, restricts which rows may enter training and
    calibration. Scoring is unaffected, so a model can be trained on one region
    and applied to another.
    """
    started = time.perf_counter()
    mark = [time.perf_counter()]

    def lap(name):
        now = time.perf_counter()
        if verbose:
            print(f"    [{name}] {now - mark[0]:.1f}s", flush=True)
        mark[0] = now

    data = load_dataset(data_dir, mmap=True)
    metadata = data["metadata"]
    components = metadata["components"]
    n_components = len(components)
    class_map = np.asarray(class_map, dtype=np.int8)
    if class_map.size != n_components:
        raise RuntimeError("class_map must have one entry per physical component")
    n_classes = len(class_names)

    all_features = data["features"]
    columns = (np.arange(len(all_features)) if feature_names is None
               else np.asarray([all_features.index(name) for name in feature_names]))
    pair_column = int(np.flatnonzero(columns == all_features.index(PAIR_FEATURE))[0])
    rapidity_source = all_features.index(RAPIDITY_FEATURE)

    component = np.asarray(data["component"])
    labels = class_map[component].astype(np.int32)
    weight = np.asarray(data["physical_weight"])
    mixture = np.asarray(data["training_mixture_weight"])
    groups = np.asarray(data["group_id"])
    rapidity_all = np.asarray(data["x"][:, rapidity_source], dtype=np.float64)
    lap("open")
    matrix = read_columns(data["x"], columns)
    lap("read features")
    if verbose:
        print(f"  feature matrix {matrix.shape} in RAM "
              f"({matrix.nbytes / 2**30:.2f} GiB)", flush=True)

    class_yields = np.bincount(labels, weights=weight, minlength=n_classes)
    if class_yields[0] <= 0.0:
        raise RuntimeError("Class 0 must be the signal and must have positive yield")
    kappas = class_yields[1:] / class_yields[0]

    is_pooled_component = np.asarray([item["generator"] == "madgraph" for item in components])
    is_pooled = is_pooled_component[component]
    folds = assign_folds(groups, seed)
    lap("folds")

    # --- fit
    fits = []
    row_probabilities = np.full((labels.size, n_classes), np.nan, dtype=np.float32)
    for fold in (0, 1):
        train_rows, stop_rows = fold_training_rows(
            {"group_id": groups}, folds, fold, is_pooled, cap, seed, stop_cap
        )
        if train_rows_mask is not None:
            train_rows = train_rows[np.asarray(train_rows_mask)[train_rows]]
            stop_rows = stop_rows[np.asarray(train_rows_mask)[stop_rows]]
            if train_rows.size == 0 or stop_rows.size == 0:
                raise RuntimeError("train_rows_mask emptied the training or calibration split")
        lap(f"fold {fold} rows")
        model, calibrator = fit_calibrated(
            matrix, labels, mixture, train_rows, stop_rows, n_classes, seed + fold, n_estimators
        )
        lap(f"fold {fold} fit")
        fits.append((model, calibrator))
        held = np.flatnonzero((folds == fold) & ~is_pooled)
        row_probabilities[held] = calibrated_probabilities(
            model, calibrator, matrix[held], n_classes
        )
        lap(f"fold {fold} oof")
        if verbose:
            print(f"  fold {fold}: {train_rows.size:,} training rows, "
                  f"{stop_rows.size:,} early-stopping rows, "
                  f"best_iteration={int(model.best_iteration)}", flush=True)
    if np.any(~np.isfinite(row_probabilities[~is_pooled])):
        raise RuntimeError("Out-of-fold scoring left non-pooled rows unscored")

    # --- pooled central events, one calibrated probability vector per grid cell
    pool = load_pool(data_dir)
    pooled_rows = np.flatnonzero(is_pooled)
    _unique, first = np.unique(groups[pooled_rows], return_index=True)
    central = pooled_rows[first]
    central_matrix = matrix[central]
    rapidity = rapidity_all[central]
    central_weight = np.asarray(data["central_weight"])[central]
    lap("central setup")
    central_fold = folds[central]
    central_component = component[central]

    cells = np.empty((central.size, grid_cells, n_classes), dtype=np.float32)
    contribution = np.empty((central.size, grid_cells), dtype=np.float64)
    covered = np.zeros(central.size, dtype=np.float64)
    bounds = np.linspace(-MAX_ABS_RAPIDITY_DIFFERENCE, MAX_ABS_RAPIDITY_DIFFERENCE, grid_cells + 1)
    for fold, (model, calibrator) in enumerate(fits):
        rows = np.flatnonzero(central_fold == fold)
        if rows.size == 0:
            raise RuntimeError(f"Fold {fold} has no pooled central events")
        working = central_matrix[rows]
        for index, (low, high) in enumerate(zip(bounds[:-1], bounds[1:])):
            working[:, pair_column] = 0.5 * (low + high)
            cells[rows, index] = calibrated_probabilities(model, calibrator, working, n_classes)
            if fold == 0:
                mass = pool_mass_between(pool, rapidity + low, rapidity + high) / pool["total"]
                covered += mass
                contribution[:, index] = central_weight * mass
        lap(f"fold {fold} sweep")
        if verbose:
            print(f"  fold {fold}: cached {rows.size:,} x {grid_cells} cells", flush=True)
    residual = float(np.max(np.abs(covered - np.asarray(data["band_probability"])[central])))

    # --- mass templates: pooled backgrounds use their inclusive shape
    mass_shape = np.zeros((n_components, MASS_BINS.size - 1))
    for component_id in np.flatnonzero(is_pooled_component):
        rows = component == component_id
        shape, _ = np.histogram(np.asarray(data["mx"])[rows], MASS_BINS,
                                weights=weight[rows])
        if shape.sum() <= 0.0:
            raise RuntimeError(f"Pooled component {component_id} has no mass support")
        mass_shape[component_id] = shape / shape.sum()
    lap("mass templates")

    if verbose:
        print(f"  band-probability residual {residual:.3e}; "
              f"built in {time.perf_counter() - started:.0f}s", flush=True)
    return {
        "metadata": metadata,
        "components": components,
        "component_names": [item["name"] for item in components],
        "class_names": list(class_names),
        "class_map": class_map,
        "kappas": kappas,
        "class_yields": class_yields,
        "component": component,
        "mx": np.asarray(data["mx"]),
        "weight": weight,
        "folds": folds,
        "is_pooled_component": is_pooled_component,
        "is_pooled": is_pooled,
        "row_probabilities": row_probabilities,
        "cell_probabilities": cells,
        "cell_contribution": contribution,
        "central_component": central_component,
        "central_fold": central_fold,
        "central_rows": central,
        "mass_shape": mass_shape,
        "band_probability_residual": residual,
        "grid_cells": grid_cells,
        "stop_cap": stop_cap,
        "train_rows_restricted": train_rows_mask is not None,
        "features": [all_features[index] for index in columns],
        "fits": fits,
        "build_seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------- score & bins

def scores_from(evaluation, kappas=None, columns=None):
    """Plug-in score for rows and for every (central event, grid cell)."""
    kappas = evaluation["kappas"] if kappas is None else np.asarray(kappas, dtype=np.float64)
    row_probability = evaluation["row_probabilities"]
    cells = evaluation["cell_probabilities"]
    if columns is not None:
        keep = np.asarray(columns)
        row_probability = row_probability[:, keep]
        cells = cells[:, :, keep]
    row_score = np.full(row_probability.shape[0], np.nan)
    good = np.isfinite(row_probability[:, 0])
    row_score[good] = plugin_score(row_probability[good].astype(np.float64), kappas)
    shape = cells.shape
    cell_score = plugin_score(
        cells.reshape(-1, shape[2]).astype(np.float64), kappas
    ).reshape(shape[0], shape[1])
    return row_score, cell_score


def signal_quantile_edges(row_score, evaluation, n_bins):
    signal = row_score[(evaluation["component"] == 0) & np.isfinite(row_score)]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.r_[-np.inf, np.quantile(signal, quantiles), np.inf])
    if edges.size != n_bins + 1:
        raise RuntimeError("Signal score quantiles did not produce distinct edges")
    return edges


def accumulate_categories(category, contribution, n_categories):
    """Yield and sum-of-squared-per-event-yield for each category.

    The statistical unit is the central event, so an event's cells are summed
    before squaring; squaring per cell would inflate the effective count by
    roughly the number of contributing cells.
    """
    total = np.zeros(n_categories)
    squared = np.zeros(n_categories)
    for start in range(0, category.shape[0], EVENT_CHUNK):
        stop = min(start + EVENT_CHUNK, category.shape[0])
        block = category[start:stop]
        rows = np.repeat(np.arange(stop - start), block.shape[1])
        per_event = np.bincount(
            rows * n_categories + block.ravel(),
            weights=contribution[start:stop].ravel(),
            minlength=(stop - start) * n_categories,
        ).reshape(stop - start, n_categories)
        total += per_event.sum(axis=0)
        squared += np.sum(per_event * per_event, axis=0)
    return total, squared


def accumulate_above(cell_score, contribution, thresholds):
    """Yield above each threshold, bucketing once instead of rescanning per cut."""
    n = thresholds.size
    total = np.zeros(n)
    squared = np.zeros(n)
    for start in range(0, cell_score.shape[0], EVENT_CHUNK):
        stop = min(start + EVENT_CHUNK, cell_score.shape[0])
        bucket = np.searchsorted(thresholds, cell_score[start:stop], side="right") - 1
        inside = bucket >= 0
        rows = np.repeat(np.arange(stop - start), cell_score.shape[1])[inside.ravel()]
        per_event = np.bincount(
            rows * n + bucket.ravel()[inside.ravel()],
            weights=contribution[start:stop].ravel()[inside.ravel()],
            minlength=(stop - start) * n,
        ).reshape(stop - start, n)
        above = np.cumsum(per_event[:, ::-1], axis=1)[:, ::-1]
        total += above.sum(axis=0)
        squared += np.sum(above * above, axis=0)
    return total, squared


def partition_mass(evaluation, row_category, cell_category, n_categories):
    """(n_components, n_categories, n_mass_bins) yields for an arbitrary partition.

    row_category applies to components with a real proton pair; cell_category is
    (central event, grid cell) for the pooled combinatorial backgrounds.
    """
    n_components = len(evaluation["components"])
    output = np.zeros((n_components, n_categories, MASS_BINS.size - 1))
    pooled_total = np.zeros(n_categories)
    pooled_squared = np.zeros(n_categories)
    for component_id in range(n_components):
        if evaluation["is_pooled_component"][component_id]:
            select = evaluation["central_component"] == component_id
            total, squared = accumulate_categories(
                cell_category[select], evaluation["cell_contribution"][select], n_categories
            )
            output[component_id] = total[:, None] * evaluation["mass_shape"][component_id]
            pooled_total += total
            pooled_squared += squared
        else:
            rows = np.flatnonzero(evaluation["component"] == component_id)
            category = row_category[rows]
            for cell in range(n_categories):
                keep = rows[category == cell]
                output[component_id, cell], _ = np.histogram(
                    evaluation["mx"][keep], MASS_BINS, weights=evaluation["weight"][keep]
                )
    effective = np.divide(pooled_total**2, pooled_squared,
                          out=np.zeros_like(pooled_total), where=pooled_squared > 0.0)
    return output, effective


def bin_index(values, edges):
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)


def ladder_mass(evaluation, row_score, cell_score, edges):
    return partition_mass(evaluation, bin_index(row_score, edges),
                          bin_index(cell_score, edges), edges.size - 1)


def grid_mass(evaluation, row_scores, cell_scores, edge_sets):
    """Product partition over several score axes."""
    sizes = [edges.size - 1 for edges in edge_sets]
    row_category = np.zeros(row_scores[0].shape, dtype=np.int64)
    cell_category = np.zeros(cell_scores[0].shape, dtype=np.int64)
    for axis, edges in enumerate(edge_sets):
        row_category = row_category * sizes[axis] + bin_index(row_scores[axis], edges)
        cell_category = cell_category * sizes[axis] + bin_index(cell_scores[axis], edges)
    return partition_mass(evaluation, row_category, cell_category, int(np.prod(sizes)))


def factorised_grid_mass(evaluation, row_scores, cell_scores, edge_sets):
    """Two-axis partition with the pooled background factorised across axis 1.

    The pooled combinatorial background runs out of MC in the signal-like corner
    of a two-dimensional partition. Its axis-1 shape is measured inclusively
    (every pooled event contributes) and applied inside each axis-0 bin, exactly
    as the mass template already is. `factorisation_residual` reports how far
    that independence assumption is from the directly counted two-dimensional
    yields, so the approximation is checked rather than assumed.
    """
    sizes = [edges.size - 1 for edges in edge_sets]
    n_categories = int(np.prod(sizes))
    direct, _direct_effective = grid_mass(evaluation, row_scores, cell_scores, edge_sets)
    output = direct.copy()
    residual = {}
    marginal_total = np.zeros(sizes[0])
    marginal_squared = np.zeros(sizes[0])
    for component_id in np.flatnonzero(evaluation["is_pooled_component"]):
        select = evaluation["central_component"] == component_id
        contribution = evaluation["cell_contribution"][select]
        axis0 = bin_index(cell_scores[0][select], edge_sets[0])
        axis1 = bin_index(cell_scores[1][select], edge_sets[1])
        marginal0, squared0 = accumulate_categories(axis0, contribution, sizes[0])
        marginal1, _ = accumulate_categories(axis1, contribution, sizes[1])
        marginal_total += marginal0
        marginal_squared += squared0
        shape1 = marginal1 / marginal1.sum() if marginal1.sum() > 0.0 else marginal1
        estimate = (marginal0[:, None] * shape1[None, :]).reshape(n_categories)
        counted = direct[component_id].sum(axis=1)
        name = evaluation["component_names"][component_id]
        scale = counted.sum()
        residual[name] = float(np.max(np.abs(estimate - counted)) / scale) if scale > 0 else np.nan
        output[component_id] = estimate[:, None] * evaluation["mass_shape"][component_id]
    # The factorised estimate's variance comes from the axis-0 marginal, which every
    # pooled event contributes to, so that is the support to quote -- not the direct
    # two-dimensional count it replaces.
    marginal_effective = np.divide(marginal_total**2, marginal_squared,
                                   out=np.zeros_like(marginal_total),
                                   where=marginal_squared > 0.0)
    effective = np.repeat(marginal_effective, sizes[1])
    return output, effective, residual


def threshold_scan(evaluation, row_score, cell_score, thresholds):
    """Component yields per mass bin above each threshold, plus pooled n_eff."""
    n_components = len(evaluation["components"])
    scan = np.zeros((thresholds.size, n_components, MASS_BINS.size - 1))
    total = np.zeros(thresholds.size)
    squared = np.zeros(thresholds.size)
    for component_id in range(n_components):
        if evaluation["is_pooled_component"][component_id]:
            select = evaluation["central_component"] == component_id
            above, above_squared = accumulate_above(
                cell_score[select], evaluation["cell_contribution"][select], thresholds
            )
            scan[:, component_id] = above[:, None] * evaluation["mass_shape"][component_id]
            total += above
            squared += above_squared
        else:
            rows = np.flatnonzero(evaluation["component"] == component_id)
            histogram, _, _ = np.histogram2d(
                row_score[rows], evaluation["mx"][rows],
                bins=(np.r_[thresholds, np.inf], MASS_BINS),
                weights=evaluation["weight"][rows],
            )
            scan[:, component_id] = np.cumsum(histogram[::-1], axis=0)[::-1]
    effective = np.divide(total**2, squared, out=np.zeros_like(total), where=squared > 0.0)
    return scan, effective


def significance(component_mass, signal_components=(0,)):
    signal = component_mass[list(signal_components)].sum(axis=0)
    background = component_mass.sum(axis=0) - signal
    total = signal + background
    per_category = np.sqrt(np.sum(
        np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0),
        axis=-1,
    ))
    return float(np.sqrt(np.sum(per_category**2))), per_category
