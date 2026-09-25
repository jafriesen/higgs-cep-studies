"""Two-fold calibrated training and compact analytic-proton evaluation."""

import copy
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from minbias import vertex_likelihood_table
from mva.common.protons import (
    LegacyPairDensity,
    build_pair_density,
    load_bootstrap_pool,
    load_pps_config,
    proton_cell_intensities,
)
from mva.common.weights import plugin_score


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
EARLY_STOPPING_ROUNDS = 40
TAIL_SIGNAL_EFFICIENCIES = (0.20, 0.15, 0.10, 0.05)
PAIR_FEATURE = "yx_minus_dijet_rapidity"
MASS_ESTIMATOR_FEATURE = "jet1_mass_estimator"
MASS_ESTIMATOR_INPUTS = ("jet1_mt", "jet1_rapidity")
# must match PROTON_FEATURE_NAMES in mva/common/features.py (not imported: it needs fastjet)
PROTON_FEATURE_NAMES = (PAIR_FEATURE, MASS_ESTIMATOR_FEATURE)


def load_dataset(data_dir):
    data_dir = Path(data_dir).resolve()
    with open(data_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if metadata.get("format_version") != 2:
        raise RuntimeError("Expected central-event dataset format version 2")
    arrays = {}
    for name in metadata["arrays"]:
        arrays[name] = np.load(data_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    rows = int(metadata["rows"])
    if arrays["x"].shape != (rows, len(metadata["central_features"])):
        raise RuntimeError("Central feature matrix shape does not match metadata")
    for name, values in arrays.items():
        if name not in {"x", "parton"} and values.shape != (rows,):
            raise RuntimeError(f"Array {name} is not aligned to central events")
    arrays["metadata"] = metadata
    arrays["data_dir"] = data_dir
    return arrays


def mass_estimator_values(estimator, rows, delta_y):
    """2 m_T cosh(y_j1 - y_X) with y_X = dijet rapidity + delta-y, for dataset rows."""
    y_x = estimator["dijet_rapidity"][rows] + delta_y
    return 2.0 * estimator["jet1_mt"][rows] * np.cosh(estimator["jet1_rapidity"][rows] - y_x)


def read_logical_matrix(data, batch_rows=250000, features=None):
    """Read the selected columns sequentially and insert the proton features.

    `features` selects a subset of what the dataset stores, so a feature-set
    comparison can reuse one build instead of rebuilding per candidate set.
    Returns the matrix, the delta-y column, and the mass-estimator description
    (None unless jet1_mass_estimator is a feature). Pooled rows have no proton pair,
    so their proton features are placeholders at delta-y = 0 until a cell is chosen.
    """
    metadata = data["metadata"]
    logical = list(features) if features else metadata["features"]
    central = metadata["central_features"]
    available = set(central) | set(PROTON_FEATURE_NAMES)
    missing = [name for name in logical if name not in available]
    if missing:
        raise RuntimeError(f"Dataset does not store these features: {missing}")
    estimator = None
    if MASS_ESTIMATOR_FEATURE in logical:
        absent = [name for name in MASS_ESTIMATOR_INPUTS if name not in central]
        if absent:
            raise RuntimeError(
                f"{MASS_ESTIMATOR_FEATURE} needs central features {absent}; rebuild the dataset"
            )
        estimator = {
            "column": logical.index(MASS_ESTIMATOR_FEATURE),
            "jet1_mt": np.empty(data["x"].shape[0], dtype=np.float64),
            "jet1_rapidity": np.empty(data["x"].shape[0], dtype=np.float64),
            "dijet_rapidity": np.asarray(data["dijet_rapidity"], dtype=np.float64),
        }
    matrix = np.empty((data["x"].shape[0], len(logical)), dtype=np.float32)
    central_indices = {name: index for index, name in enumerate(central)}
    pair_column = logical.index(PAIR_FEATURE)
    for start in range(0, matrix.shape[0], batch_rows):
        stop = min(start + batch_rows, matrix.shape[0])
        source = np.asarray(data["x"][start:stop])
        delta_y = np.nan_to_num(
            data["proton_yx"][start:stop] - data["dijet_rapidity"][start:stop], nan=0.0
        )
        if estimator is not None:
            for name in MASS_ESTIMATOR_INPUTS:
                estimator[name][start:stop] = source[:, central_indices[name]]
        for target, name in enumerate(logical):
            if name == PAIR_FEATURE:
                matrix[start:stop, target] = delta_y
            elif name == MASS_ESTIMATOR_FEATURE:
                matrix[start:stop, target] = mass_estimator_values(
                    estimator, slice(start, stop), delta_y
                )
            else:
                matrix[start:stop, target] = source[:, central_indices[name]]
        print(f"  read features {stop:,}/{matrix.shape[0]:,}", flush=True)
    return matrix, pair_column, estimator


def set_proton_cells(working, pair_column, estimator, rows, delta_y):
    """Write delta-y, and the mass estimator it implies, into rows of `working`.

    `working` row i belongs to dataset row rows[i] at delta-y delta_y[i].
    """
    working[:, pair_column] = delta_y
    if estimator is not None:
        working[:, estimator["column"]] = mass_estimator_values(estimator, rows, delta_y)


def assign_folds(groups, seed, n_folds=2):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    assignment = np.arange(unique.size, dtype=np.int8) % n_folds
    rng.shuffle(assignment)
    folds = assignment[inverse]
    order = np.argsort(groups, kind="stable")
    if np.any(
        (groups[order][1:] == groups[order][:-1])
        & (folds[order][1:] != folds[order][:-1])
    ):
        raise RuntimeError("A central-event group crosses folds")
    return folds


def balanced_weights(labels, weights, n_classes):
    output = np.asarray(weights, dtype=np.float64).copy()
    target = labels.size / n_classes
    for class_id in range(n_classes):
        selected = labels == class_id
        total = output[selected].sum()
        if total <= 0.0:
            raise RuntimeError(f"Class {class_id} has no positive training weight")
        output[selected] *= target / total
    return output


def fixed_efficiency_thresholds(scores, labels, weights, efficiencies):
    """Return score thresholds giving the requested weighted signal efficiencies."""
    signal = labels == 0
    if not np.any(signal) or np.sum(weights[signal]) <= 0.0:
        raise RuntimeError("Tail selection needs positive-weight signal events")
    order = np.argsort(scores[signal])[::-1]
    signal_scores = scores[signal][order]
    cumulative = np.cumsum(weights[signal][order])
    targets = np.asarray(efficiencies) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return signal_scores[np.minimum(indices, signal_scores.size - 1)]


def tail_rejection_metric(scores, labels, weights, nonexclusive, efficiencies):
    """Mean log nonexclusive efficiency at fixed weighted signal efficiencies."""
    thresholds = fixed_efficiency_thresholds(scores, labels, weights, efficiencies)
    selected = np.asarray(nonexclusive, dtype=bool)
    total = np.sum(weights[selected])
    if total <= 0.0:
        raise RuntimeError("Tail selection needs positive-weight nonexclusive events")
    largest = np.max(weights[selected])
    efficiencies_out = np.asarray(
        [
            min(
                1.0,
                (np.sum(weights[selected & (scores >= threshold)]) + largest) / total,
            )
            for threshold in thresholds
        ]
    )
    return float(np.mean(np.log(efficiencies_out))), thresholds, efficiencies_out


def hard_negative_factors(scores, rows, nonexclusive, campaigns, fraction, boost):
    """Upweight the hardest requested fraction within each nonexclusive campaign."""
    scores = np.asarray(scores)
    rows = np.asarray(rows)
    factors = np.ones(scores.shape[0], dtype=np.float64)
    selected_rows = []
    diagnostics = []
    candidates = rows[nonexclusive[rows] & np.isfinite(scores[rows])]
    for campaign in np.unique(campaigns[candidates]):
        campaign_rows = candidates[campaigns[candidates] == campaign]
        count = max(1, int(np.ceil(fraction * campaign_rows.size)))
        order = np.argsort(scores[campaign_rows])
        hard = campaign_rows[order[-count:]]
        factors[hard] = boost
        selected_rows.append(hard)
        diagnostics.append(
            {
                "campaign_id": int(campaign),
                "candidates": int(campaign_rows.size),
                "selected": int(hard.size),
                "score_threshold": float(np.min(scores[hard])),
            }
        )
    selected = np.sort(np.concatenate(selected_rows)) if selected_rows else np.array([], dtype=int)
    return factors, selected, diagnostics


def _cap_rows(rows, pooled, limit, rng):
    pooled_rows = rows[pooled[rows]]
    other = rows[~pooled[rows]]
    if limit is not None and pooled_rows.size > limit:
        pooled_rows = rng.choice(pooled_rows, size=limit, replace=False)
    return np.sort(np.concatenate([other, pooled_rows]))


def _stratified_group_stop(available, labels, groups, components, seed):
    """Reserve about ten percent of each class without splitting event groups."""
    stop_parts = []
    rng = np.random.default_rng(seed)
    for class_id in np.unique(labels[available]):
        rows = available[labels[available] == class_id]
        keys = np.column_stack((components[rows], groups[rows]))
        _unique, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        if counts.size < 2:
            raise RuntimeError(
                f"Class {class_id} needs at least two event groups for stopping"
            )
        order = rng.permutation(counts.size)
        cumulative = np.cumsum(counts[order])
        target = 0.1 * rows.size
        choices = np.arange(1, counts.size)
        chosen_count = choices[np.argmin(np.abs(cumulative[choices - 1] - target))]
        chosen = np.zeros(counts.size, dtype=bool)
        chosen[order[:chosen_count]] = True
        stop_parts.append(rows[chosen[inverse]])
    return np.sort(np.concatenate(stop_parts))


def fold_rows(
    folds,
    fold,
    eligible,
    pooled,
    cap,
    stop_cap,
    seed,
    labels=None,
    groups=None,
    components=None,
):
    available = np.flatnonzero((folds != fold) & eligible)
    if labels is None:
        stop = available[np.asarray(available, dtype=np.uint64) % np.uint64(10) == 0]
    else:
        if groups is None or components is None:
            raise ValueError("Stratified stopping needs groups and components")
        stop = _stratified_group_stop(
            available, labels, groups, components, seed + fold
        )
    train = np.setdiff1d(available, stop, assume_unique=True)
    train = _cap_rows(train, pooled, cap, np.random.default_rng(seed + fold))
    stop = _cap_rows(stop, pooled, stop_cap, np.random.default_rng(seed + 500 + fold))
    if train.size == 0 or stop.size == 0:
        raise RuntimeError("Training or early-stopping rows are empty")
    return train, stop


def architecture_classes(classes, architecture):
    """Return analysis labels and names for a requested classifier architecture."""
    classes = list(classes)
    if architecture == "standard":
        return np.arange(len(classes), dtype=np.int32), classes
    if not classes:
        raise ValueError("Classifier architecture requires a signal class")
    signal_name = classes[0]
    if architecture in {
        "binary",
        "nonexclusive_specialist",
        "exclusive_specialist",
    }:
        mapping = np.asarray([0] + [1] * (len(classes) - 1), dtype=np.int32)
        background_name = {
            "binary": "all_backgrounds",
            "nonexclusive_specialist": "nonexclusive_background",
            "exclusive_specialist": "combined_exclusive_background",
        }[architecture]
        return mapping, [signal_name, background_name]
    if architecture in {"merged_exclusive", "hierarchical"}:
        mapping = []
        for name in classes:
            if name == signal_name:
                mapping.append(0)
            elif name.startswith("exclusive_"):
                mapping.append(1)
            elif name.startswith("nonexclusive_"):
                mapping.append(2)
            else:
                raise ValueError(f"Cannot classify architecture component: {name}")
        if set(mapping) != {0, 1, 2}:
            raise ValueError(
                "Merged architectures need signal, exclusive, and nonexclusive classes"
            )
        return np.asarray(mapping, dtype=np.int32), [
            signal_name,
            "combined_exclusive_background",
            "nonexclusive_background",
        ]
    raise ValueError(f"Unknown classifier architecture: {architecture}")


def architecture_training_mask(original_labels, original_classes, architecture):
    """Select components used to fit an architecture; evaluation remains inclusive."""
    labels = np.asarray(original_labels, dtype=np.int32)
    if architecture == "nonexclusive_specialist":
        allowed = {
            index
            for index, name in enumerate(original_classes)
            if index == 0 or name.startswith("nonexclusive_")
        }
    elif architecture == "exclusive_specialist":
        allowed = {
            index
            for index, name in enumerate(original_classes)
            if index == 0 or name.startswith("exclusive_")
        }
    else:
        return np.ones(labels.shape, dtype=bool)
    if len(allowed) < 2:
        raise ValueError(f"Architecture {architecture} did not resolve signal and background")
    return np.isin(labels, list(allowed))


def hierarchical_probabilities(stage_a, stage_b, class_yields):
    """Recover equal-prior H/exclusive/nonexclusive densities from two stages."""
    stage_a = np.clip(np.asarray(stage_a, dtype=np.float64), 1.0e-12, 1.0)
    stage_b = np.clip(np.asarray(stage_b, dtype=np.float64), 1.0e-12, 1.0)
    yields = np.asarray(class_yields, dtype=np.float64)
    if stage_a.shape != stage_b.shape or stage_a.ndim != 2 or stage_a.shape[1] != 2:
        raise ValueError("Both hierarchy stages must contain two probability columns")
    if yields.shape != (3,) or np.any(yields <= 0.0):
        raise ValueError("Hierarchy needs three positive physical class yields")
    h_over_x = stage_b[:, 0] / stage_b[:, 1]
    e_over_n = stage_a[:, 0] / stage_a[:, 1]
    f_h = h_over_x
    f_x = np.ones_like(f_h)
    f_e = (yields[0] * f_h + yields[1] * f_x) / (yields[0] + yields[1])
    f_n = f_e / e_over_n
    density = np.column_stack((f_h, f_x, f_n))
    return density / density.sum(axis=1, keepdims=True)


def sample_training_cells(
    matrix,
    pair_column,
    pooled_rows,
    rapidity,
    pairs,
    cell_edges,
    mass_window,
    pileup_mu,
    seed,
    chunk_rows,
    estimator=None,
):
    """Choose one analytic cell per pooled event for fitting, never for yields."""
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for start in range(0, pooled_rows.size, chunk_rows):
        stop = min(start + chunk_rows, pooled_rows.size)
        rows = pooled_rows[start:stop]
        intensity = proton_cell_intensities(
            pairs, rapidity[rows], cell_edges, mass_window, pileup_mu
        )
        cumulative = np.cumsum(intensity, axis=1)
        total = cumulative[:, -1]
        quantile = rng.random(rows.size) * total
        cell = np.sum(cumulative < quantile[:, np.newaxis], axis=1)
        cell = np.minimum(cell, cell_edges.size - 2)
        delta_y = 0.5 * (cell_edges[cell] + cell_edges[cell + 1])
        matrix[rows, pair_column] = delta_y
        if estimator is not None:
            matrix[rows, estimator["column"]] = mass_estimator_values(estimator, rows, delta_y)
        elapsed = time.perf_counter() - started
        print(
            f"  sampled weighted proton training cells {stop:,}/{pooled_rows.size:,} "
            f"elapsed={elapsed:.1f}s eta={elapsed / stop * (pooled_rows.size - stop):.1f}s",
            flush=True,
        )


def analysis_state(data_dir, args, features=None):
    """Load aligned arrays and derive the deterministic train/evaluation split."""
    data = load_dataset(data_dir)
    metadata = copy.deepcopy(data["metadata"])
    architecture = getattr(args, "architecture", "standard")
    original_class_names = list(metadata["classes"])
    class_mapping, class_names = architecture_classes(original_class_names, architecture)
    metadata["classes"] = class_names
    components = metadata["components"]
    for component in components:
        component["class_id"] = int(class_mapping[component["class_id"]])
    logical_features = list(features) if features else list(metadata["features"])
    available = set(metadata["central_features"]) | set(PROTON_FEATURE_NAMES)
    missing = [name for name in logical_features if name not in available]
    if missing:
        raise RuntimeError(f"Dataset does not store these features: {missing}")
    if PAIR_FEATURE not in logical_features:
        raise RuntimeError("Training features must include yx_minus_dijet_rapidity")

    original_labels = np.asarray(data["class"], dtype=np.int32)
    labels = class_mapping[original_labels]
    groups = np.asarray(data["group_id"])
    physical = np.asarray(data["physical_weight"], dtype=np.float64)
    mixture = np.asarray(data["training_mixture_weight"], dtype=np.float64)
    band = np.asarray(data["pair_band_intensity"], dtype=np.float64)
    real_component = np.asarray([item["real_protons"] for item in components], dtype=bool)
    pooled = ~real_component[np.asarray(data["component"])]
    effective_physical = physical * np.where(pooled, band, 1.0)
    effective_mixture = mixture * np.where(pooled, band, 1.0)
    eligible = effective_physical > 0.0
    if args.require_truth_matched:
        eligible &= np.asarray(data["truth_matched"])
    training_eligible = eligible & architecture_training_mask(
        original_labels, original_class_names, architecture
    )
    n_classes = len(class_names)
    class_yields = np.bincount(
        labels[training_eligible],
        weights=effective_physical[training_eligible],
        minlength=n_classes,
    )
    if class_yields[0] <= 0.0 or np.any(class_yields <= 0.0):
        raise RuntimeError("Every configured class needs positive eligible yield")
    return {
        "data": data,
        "metadata": metadata,
        "components": components,
        "n_classes": n_classes,
        "n_components": len(components),
        "logical_features": logical_features,
        "labels": labels,
        "groups": groups,
        "component_ids": np.asarray(data["component"], dtype=np.int32),
        "architecture": architecture,
        "physical": physical,
        "effective_physical": effective_physical,
        "effective_mixture": effective_mixture,
        "real_component": real_component,
        "pooled": pooled,
        "eligible": eligible,
        "training_eligible": training_eligible,
        "class_yields": class_yields,
        "kappas": class_yields[1:] / class_yields[0],
        "folds": assign_folds(groups, args.seed),
    }


def prepare_training_state(data_dir, args, features=None):
    """Materialize logical features and sample analytic cells for model fitting."""
    state = analysis_state(data_dir, args, features)
    matrix, pair_column, estimator = read_logical_matrix(
        state["data"], args.batch_rows, state["logical_features"]
    )
    pairs = None
    cell_edges = None
    if np.any(state["pooled"]):
        metadata = state["metadata"]
        if metadata["protons"]["backend"] == "analytic":
            pps = load_pps_config(args.repo / args.pps_config)
            pairs = build_pair_density(
                metadata["protons"]["path"],
                pps,
                seed=args.seed,
                verify_hash=not args.skip_hash,
            )
        elif metadata["protons"]["backend"] == "legacy_pool":
            pool = load_bootstrap_pool()
            if Path(pool["path"]).resolve() != Path(metadata["protons"]["path"]).resolve():
                raise RuntimeError("Configured legacy pool changed since dataset preparation")
            pairs = LegacyPairDensity(pool)
        else:
            raise RuntimeError(f"Unknown proton backend: {metadata['protons']['backend']}")
        cell_edges = np.linspace(-args.max_delta_y, args.max_delta_y, args.grid_cells + 1)
        sample_training_cells(
            matrix,
            pair_column,
            np.flatnonzero(state["pooled"] & state["eligible"]),
            np.asarray(state["data"]["dijet_rapidity"]),
            pairs,
            cell_edges,
            args.mass_window,
            args.pileup_mu,
            args.seed,
            args.intensity_chunk,
            estimator,
        )
    state.update(
        {
            "matrix": matrix,
            "pair_column": pair_column,
            "estimator": estimator,
            "pairs": pairs,
            "cell_edges": cell_edges,
        }
    )
    return state


def _select_tail_iteration(model, matrix, labels, weights, stop_rows, nonexclusive, args):
    coarse = np.arange(19, args.n_estimators, 20, dtype=int)
    if coarse.size == 0 or coarse[-1] != args.n_estimators - 1:
        coarse = np.r_[coarse, args.n_estimators - 1]

    def evaluate(iterations):
        rows = []
        for iteration in iterations:
            score = -model.predict(
                matrix[stop_rows],
                output_margin=True,
                iteration_range=(0, int(iteration) + 1),
            )
            metric, thresholds, efficiencies = tail_rejection_metric(
                score,
                labels[stop_rows],
                weights,
                nonexclusive[stop_rows],
                args.tail_signal_efficiencies,
            )
            rows.append((int(iteration), metric, thresholds, efficiencies))
        return rows

    coarse_rows = evaluate(coarse)
    coarse_best = min(coarse_rows, key=lambda item: item[1])[0]
    low = max(0, coarse_best - 19)
    high = min(args.n_estimators - 1, coarse_best + 19)
    fine = np.arange(low, high + 1, 2, dtype=int)
    rows = coarse_rows + evaluate(fine[~np.isin(fine, coarse)])
    best = min(rows, key=lambda item: item[1])
    model.get_booster().set_attr(best_iteration=str(best[0]), best_score=str(best[1]))
    return {
        "criterion": "mean_log_nonexclusive_efficiency",
        "iteration": best[0],
        "metric": best[1],
        "signal_efficiencies": [float(value) for value in args.tail_signal_efficiencies],
        "thresholds": [float(value) for value in best[2]],
        "nonexclusive_efficiencies": [float(value) for value in best[3]],
    }


def fit_calibrated(
    matrix,
    labels,
    mixture,
    train_rows,
    stop_rows,
    n_classes,
    args,
    fold,
    nonexclusive=None,
):
    train_weight = balanced_weights(labels[train_rows], mixture[train_rows], n_classes)
    stop_weight = balanced_weights(labels[stop_rows], mixture[stop_rows], n_classes)
    tail_selection = getattr(args, "model_selection", "logloss") == "nonexclusive_tail"
    if tail_selection and (n_classes != 2 or nonexclusive is None):
        raise ValueError("Nonexclusive-tail model selection requires a binary classifier")
    model_options = dict(
        n_estimators=args.n_estimators,
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        tree_method="hist",
        n_jobs=args.jobs,
        random_state=args.seed + fold,
        **PRODUCTION_PARAMS,
    )
    if not tail_selection:
        model_options["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = XGBClassifier(**model_options)
    print(
        f"  fold {fold}: fit rows={train_rows.size:,} stop={stop_rows.size:,} "
        f"matrix={train_rows.size * matrix.shape[1] * matrix.dtype.itemsize / 2**20:.1f} MiB",
        flush=True,
    )
    model.fit(
        matrix[train_rows],
        labels[train_rows],
        sample_weight=train_weight,
        eval_set=[(matrix[stop_rows], labels[stop_rows])],
        sample_weight_eval_set=[stop_weight],
        verbose=args.monitor_rounds,
    )
    selection = None
    if tail_selection:
        selection = _select_tail_iteration(
            model,
            matrix,
            labels,
            stop_weight,
            stop_rows,
            nonexclusive,
            args,
        )
        print(
            f"  fold {fold}: tail-selected iteration={selection['iteration']} "
            f"nonexclusive_efficiencies={selection['nonexclusive_efficiencies']}",
            flush=True,
        )
    margins = model.predict(
        matrix[stop_rows],
        output_margin=True,
        iteration_range=(0, int(model.best_iteration) + 1),
    )
    if margins.ndim == 1:
        margins = margins.reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500, random_state=args.seed + fold)
    calibrator.fit(margins, labels[stop_rows], sample_weight=stop_weight)
    if not np.array_equal(calibrator.classes_, np.arange(n_classes)):
        raise RuntimeError("Calibration split does not contain every class")
    model.tail_selection = selection
    return model, calibrator


def calibrated_probabilities(model, calibrator, matrix, batch_rows=250000):
    output = np.empty((matrix.shape[0], calibrator.classes_.size), dtype=np.float32)
    for start in range(0, matrix.shape[0], batch_rows):
        stop = min(start + batch_rows, matrix.shape[0])
        margin = model.predict(
            matrix[start:stop],
            output_margin=True,
            iteration_range=(0, int(model.best_iteration) + 1),
        )
        if margin.ndim == 1:
            margin = margin.reshape(-1, 1)
        output[start:stop] = calibrator.predict_proba(margin)
    return output


def hierarchy_calibrated_probabilities(
    stage_a_model,
    stage_a_calibrator,
    stage_b_model,
    stage_b_calibrator,
    class_yields,
    matrix,
    batch_rows=250000,
):
    stage_a = calibrated_probabilities(
        stage_a_model, stage_a_calibrator, matrix, batch_rows
    )
    stage_b = calibrated_probabilities(
        stage_b_model, stage_b_calibrator, matrix, batch_rows
    )
    return hierarchical_probabilities(stage_a, stage_b, class_yields).astype(np.float32)


def _vertex_terms(table, hypothesis):
    """Bin probabilities and log-LR shifts for one vertex hypothesis.

    "none" is the neutral element: a single bin of unit weight and zero
    shift, used when the vertex acceptance is already in the weights.
    """
    if hypothesis == "none":
        return np.array([1.0]), np.array([0.0])
    return table[f"{hypothesis}_probability"], table["log_likelihood_ratio"]


def _score_bins(values, edges):
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)


def _mass_bins(values, edges):
    return np.searchsorted(edges, values, side="right") - 1


def _histogram_tail_bins(histogram, efficiencies):
    above = np.cumsum(np.asarray(histogram)[::-1])[::-1]
    target = np.asarray(efficiencies) * above[0]
    return np.asarray([np.argmin(np.abs(above - value)) for value in target], dtype=int)


def _add_real_histograms(
    accumulator, probability_histograms, component_id, class_id, scores, probabilities,
    mass, weights, vertex_probability, vertex_log_lr, score_edges, mass_edges, probability_edges,
):
    for predicted in range(probabilities.shape[1]):
        probability_histograms[class_id, predicted] += np.histogram(
            probabilities[:, predicted], probability_edges, weights=weights
        )[0]
    mass_bin = _mass_bins(mass, mass_edges)
    valid_mass = (mass_bin >= 0) & (mass_bin < mass_edges.size - 1)
    for probability, shift in zip(vertex_probability, vertex_log_lr):
        score_bin = _score_bins(scores + shift, score_edges)
        combined = score_bin[valid_mass] * (mass_edges.size - 1) + mass_bin[valid_mass]
        accumulator[component_id] += np.bincount(
            combined,
            weights=weights[valid_mass] * probability,
            minlength=(score_edges.size - 1) * (mass_edges.size - 1),
        ).reshape(score_edges.size - 1, mass_edges.size - 1)


def _madgraph_mass_shape(pairs, rapidity, central_weight, mass_edges, max_delta_y, pileup_mu, chunk):
    histogram = np.zeros(mass_edges.size - 1, dtype=np.float64)
    started = time.perf_counter()
    for mass_index, mass_range in enumerate(zip(mass_edges[:-1], mass_edges[1:])):
        for start in range(0, rapidity.size, chunk):
            stop = min(start + chunk, rapidity.size)
            y = rapidity[start:stop]
            intensity = pileup_mu**2 * pairs.integrate_yx_ranges(
                mass_range, y - max_delta_y, y + max_delta_y
            )
            histogram[mass_index] += np.sum(central_weight[start:stop] * intensity)
        elapsed = time.perf_counter() - started
        completed = mass_index + 1
        print(
            f"    mass template {completed}/{histogram.size} elapsed={elapsed:.1f}s "
            f"eta={elapsed / completed * (histogram.size - completed):.1f}s",
            flush=True,
        )
    total = histogram.sum()
    return histogram / total if total > 0.0 else histogram


def _significance(component_mass):
    signal = component_mass[:, 0]
    background = component_mass[:, 1:].sum(axis=1)
    total = signal + background
    return np.sqrt(
        np.sum(
            np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0),
            axis=1,
        )
    )


def _weighted_quantiles(values, weights, quantiles=(0.1, 0.5, 0.9)):
    positive = np.asarray(weights) > 0.0
    if not np.any(positive):
        return [float("nan")] * len(quantiles)
    values = np.asarray(values)[positive]
    weights = np.asarray(weights)[positive]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    targets = np.asarray(quantiles) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return [float(value) for value in values[order][np.minimum(indices, values.size - 1)]]


def _vertex_convolution(base, components, table, score_edges):
    output = np.zeros_like(base)
    centers = 0.5 * (score_edges[:-1] + score_edges[1:])
    mass_bins = base.shape[2]
    for component_id, component in enumerate(components):
        probabilities, shifts = _vertex_terms(table, component["vertex_hypothesis"])
        for probability, shift in zip(probabilities, shifts):
            target = _score_bins(centers + shift, score_edges)
            for source, destination in enumerate(target):
                output[component_id, destination] += probability * base[component_id, source]
    return output


def evaluate_fold(state, model, calibrator, args, fold, auxiliary_fit=None):
    """Evaluate one held-out fold and return additive report accumulators."""
    data = state["data"]
    components = state["components"]
    n_classes = state["n_classes"]
    n_components = state["n_components"]
    matrix = state["matrix"]
    folds = state["folds"]
    eligible = state["eligible"]
    physical = state["physical"]
    pairs = state["pairs"]
    cell_edges = state["cell_edges"]
    pair_column = state["pair_column"]
    estimator = state.get("estimator")
    kappas = state["kappas"]

    def predict(values):
        if auxiliary_fit is None:
            return calibrated_probabilities(model, calibrator, values, args.batch_rows)
        auxiliary_model, auxiliary_calibrator = auxiliary_fit
        return hierarchy_calibrated_probabilities(
            model,
            calibrator,
            auxiliary_model,
            auxiliary_calibrator,
            class_yields=state["class_yields"],
            matrix=values,
            batch_rows=args.batch_rows,
        )

    score_edges = np.linspace(args.score_min, args.score_max, args.score_bins + 1)
    mass_edges = np.linspace(args.mass_window[0], args.mass_window[1], 17)
    probability_edges = np.linspace(0.0, 1.0, 51)
    base_mass = np.zeros((n_components, args.score_bins, mass_edges.size - 1))
    nominal_mass = np.zeros_like(base_mass)
    probability_histograms = np.zeros(
        (n_classes, n_classes, probability_edges.size - 1)
    )
    vertex = vertex_likelihood_table(
        bins=args.vertex_bins,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        single_arm_time_resolution_ps=args.pps_time_ps,
        pv_z_resolution_cm=args.pv_z_resolution_cm,
        pv_time_resolution_ps=args.pv_time_ps,
    )
    scan_bins = np.unique(
        np.linspace(0, args.score_bins - 1, args.scan_points).astype(int)
    )
    mg_above_squared = np.zeros(scan_bins.size)
    mg_base_above_squared = np.zeros(args.score_bins)
    tail_efficiencies = np.asarray(
        getattr(args, "tail_signal_efficiencies", TAIL_SIGNAL_EFFICIENCIES),
        dtype=np.float64,
    )
    tail_bins = None
    tail_signal_yields = None
    tail_rows = []
    tail_event_yields = []

    for component_id, component in enumerate(components):
        rows = np.flatnonzero(
            (folds == fold)
            & eligible
            & (np.asarray(data["component"]) == component_id)
        )
        if rows.size == 0:
            raise RuntimeError(f"Fold {fold} has no eligible {component['name']} events")
        print(
            f"  score fold={fold} component={component['name']} events={rows.size:,}",
            flush=True,
        )
        if component["real_protons"]:
            probabilities = predict(matrix[rows])
            base_score = plugin_score(probabilities, kappas)
            _add_real_histograms(
                nominal_mass,
                probability_histograms,
                component_id,
                component["class_id"],
                base_score,
                probabilities,
                np.asarray(data["proton_mx"])[rows],
                physical[rows],
                *_vertex_terms(vertex, component["vertex_hypothesis"]),
                score_edges,
                mass_edges,
                probability_edges,
            )
            _add_real_histograms(
                base_mass,
                np.zeros_like(probability_histograms),
                component_id,
                component["class_id"],
                base_score,
                probabilities,
                np.asarray(data["proton_mx"])[rows],
                physical[rows],
                np.array([1.0]),
                np.array([0.0]),
                score_edges,
                mass_edges,
                probability_edges,
            )
            if component_id == 0:
                signal_histogram = nominal_mass[component_id].sum(axis=1)
                tail_bins = _histogram_tail_bins(signal_histogram, tail_efficiencies)
                tail_signal_yields = np.cumsum(signal_histogram[::-1])[::-1][tail_bins]
            continue

        score_yield = np.zeros(args.score_bins)
        base_score_yield = np.zeros(args.score_bins)
        scoring_started = time.perf_counter()
        for start in range(0, rows.size, args.evaluation_chunk):
            stop = min(start + args.evaluation_chunk, rows.size)
            selected = rows[start:stop]
            working = np.repeat(matrix[selected], args.grid_cells, axis=0)
            delta = np.tile(
                0.5 * (cell_edges[:-1] + cell_edges[1:]), selected.size
            )
            set_proton_cells(
                working, pair_column, estimator,
                np.repeat(selected, args.grid_cells), delta,
            )
            probabilities = predict(working)
            base_score = plugin_score(probabilities, kappas).reshape(
                selected.size, args.grid_cells
            )
            intensity = proton_cell_intensities(
                pairs,
                np.asarray(data["dijet_rapidity"])[selected],
                cell_edges,
                args.mass_window,
                args.pileup_mu,
            )
            central_weight = physical[selected, np.newaxis]
            base_contribution = central_weight * intensity
            for predicted in range(n_classes):
                probability_histograms[component["class_id"], predicted] += np.histogram(
                    probabilities[:, predicted],
                    probability_edges,
                    weights=base_contribution.ravel(),
                )[0]
            base_bucket = _score_bins(base_score, score_edges)
            base_score_yield += np.bincount(
                base_bucket.ravel(),
                weights=base_contribution.ravel(),
                minlength=args.score_bins,
            )
            per_event_base = np.zeros((selected.size, args.score_bins), dtype=np.float64)
            np.add.at(
                per_event_base,
                (np.repeat(np.arange(selected.size), args.grid_cells), base_bucket.ravel()),
                base_contribution.ravel(),
            )
            base_above = np.cumsum(per_event_base[:, ::-1], axis=1)[:, ::-1]
            mg_base_above_squared += np.sum(base_above**2, axis=0)
            per_event_bucket = np.zeros(
                (selected.size, args.score_bins), dtype=np.float64
            )
            for vertex_probability, shift in zip(
                *_vertex_terms(vertex, component["vertex_hypothesis"])
            ):
                bucket = _score_bins(base_score + shift, score_edges)
                contribution = base_contribution * vertex_probability
                score_yield += np.bincount(
                    bucket.ravel(),
                    weights=contribution.ravel(),
                    minlength=args.score_bins,
                )
                np.add.at(
                    per_event_bucket,
                    (
                        np.repeat(np.arange(selected.size), args.grid_cells),
                        bucket.ravel(),
                    ),
                    contribution.ravel(),
                )
            above = np.cumsum(per_event_bucket[:, ::-1], axis=1)[:, ::-1]
            mg_above_squared += np.sum(above[:, scan_bins] ** 2, axis=0)
            if tail_bins is None:
                raise RuntimeError("Signal must precede pooled backgrounds in component order")
            tail_rows.append(selected)
            tail_event_yields.append(above[:, tail_bins])
            elapsed = time.perf_counter() - scoring_started
            print(
                f"    cells {stop:,}/{rows.size:,} elapsed={elapsed:.1f}s "
                f"eta={elapsed / stop * (rows.size - stop):.1f}s",
                flush=True,
            )

        rapidity = np.asarray(data["dijet_rapidity"])[rows]
        shape = _madgraph_mass_shape(
            pairs,
            rapidity,
            physical[rows],
            mass_edges,
            args.max_delta_y,
            args.pileup_mu,
            args.intensity_chunk,
        )
        nominal_mass[component_id] += score_yield[:, np.newaxis] * shape
        base_mass[component_id] += base_score_yield[:, np.newaxis] * shape

    return {
        "base_mass": base_mass,
        "nominal_mass": nominal_mass,
        "probability_histograms": probability_histograms,
        "mg_above_squared": mg_above_squared,
        "mg_base_above_squared": mg_base_above_squared,
        # empty when the profile has no accidental-proton (pooled) component
        "tail_rows": np.concatenate(tail_rows) if tail_rows else np.zeros(0, dtype=np.int64),
        "tail_event_yields": (
            np.concatenate(tail_event_yields)
            if tail_event_yields
            else np.zeros((0, tail_efficiencies.size))
        ),
        "tail_signal_yields": tail_signal_yields,
        "tail_thresholds": score_edges[tail_bins],
    }


def finish_evaluation(
    state,
    result_dir,
    args,
    partials,
    best_iterations,
    started,
    models_saved=False,
    orchestration=None,
):
    """Merge additive fold outputs and write the compact physics report."""
    data = state["data"]
    metadata = state["metadata"]
    components = state["components"]
    logical_features = state["logical_features"]
    real_component = state["real_component"]
    eligible = state["eligible"]
    pooled = state["pooled"]
    effective_physical = state["effective_physical"]
    class_yields = state["class_yields"]
    kappas = state["kappas"]
    base_mass = np.sum([item["base_mass"] for item in partials], axis=0)
    nominal_mass = np.sum([item["nominal_mass"] for item in partials], axis=0)
    probability_histograms = np.sum(
        [item["probability_histograms"] for item in partials], axis=0
    )
    mg_above_squared = np.sum(
        [item["mg_above_squared"] for item in partials], axis=0
    )
    mg_base_above_squared = np.sum(
        [item["mg_base_above_squared"] for item in partials], axis=0
    )
    tail_rows = np.concatenate([item["tail_rows"] for item in partials])
    tail_event_yields = np.concatenate(
        [item["tail_event_yields"] for item in partials], axis=0
    )
    tail_signal_yields = np.sum(
        [item["tail_signal_yields"] for item in partials], axis=0
    )
    tail_thresholds = np.stack([item["tail_thresholds"] for item in partials])

    score_edges = np.linspace(args.score_min, args.score_max, args.score_bins + 1)
    mass_edges = np.linspace(args.mass_window[0], args.mass_window[1], 17)
    probability_edges = np.linspace(0.0, 1.0, 51)
    scan_bins = np.unique(
        np.linspace(0, args.score_bins - 1, args.scan_points).astype(int)
    )
    vertex = vertex_likelihood_table(
        bins=args.vertex_bins,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        single_arm_time_resolution_ps=args.pps_time_ps,
        pv_z_resolution_cm=args.pv_z_resolution_cm,
        pv_time_resolution_ps=args.pv_time_ps,
    )
    cumulative = np.cumsum(nominal_mass[:, ::-1, :], axis=1)[:, ::-1, :]
    scan_component_mass = np.transpose(cumulative[:, scan_bins, :], (1, 0, 2))
    scan_significance = _significance(scan_component_mass)
    scan_total_mg = scan_component_mass[:, ~real_component].sum(axis=(1, 2))
    scan_effective = np.divide(
        scan_total_mg**2,
        mg_above_squared,
        out=np.zeros_like(scan_total_mg),
        where=mg_above_squared > 0.0,
    )
    valid = np.flatnonzero(scan_effective >= args.support_floor)
    if valid.size == 0:
        valid = np.arange(scan_significance.size)
        support_warning = "No scan point reached the requested MadGraph support floor"
    else:
        support_warning = None
    operating = valid[np.argmax(scan_significance[valid])]

    signal_score_yield = nominal_mass[0].sum(axis=1)
    signal_cdf = np.cumsum(signal_score_yield)
    quantiles = np.arange(1, args.ladder_bins) / args.ladder_bins * signal_cdf[-1]
    category_bins = np.r_[
        0, np.searchsorted(signal_cdf, quantiles), args.score_bins
    ]
    category_bins = np.unique(category_bins)
    category_mass = np.stack(
        [
            nominal_mass[:, low:high].sum(axis=1)
            for low, high in zip(category_bins[:-1], category_bins[1:])
        ]
    )
    category_significance = _significance(category_mass)
    ladder_significance = float(np.sqrt(np.sum(category_significance**2)))

    resolution_rows = []
    for pps_time in args.pps_time_scan:
        for pv_time in args.pv_time_scan:
            table = vertex_likelihood_table(
                bins=args.vertex_bins,
                beam_sigma_z_cm=args.beam_sigma_z_cm,
                single_arm_time_resolution_ps=pps_time,
                pv_z_resolution_cm=args.pv_z_resolution_cm,
                pv_time_resolution_ps=pv_time,
            )
            if np.isclose(pps_time, args.pps_time_ps) and np.isclose(
                pv_time, args.pv_time_ps
            ):
                varied = nominal_mass
            else:
                varied = _vertex_convolution(base_mass, components, table, score_edges)
            varied_cumulative = np.cumsum(
                varied[:, ::-1, :], axis=1
            )[:, ::-1, :]
            varied_scan = np.transpose(
                varied_cumulative[:, scan_bins], (1, 0, 2)
            )
            varied_z = _significance(varied_scan)
            best = int(np.argmax(varied_z))
            resolution_rows.append(
                {
                    "pps_single_arm_time_ps": float(pps_time),
                    "pv_time_ps": float(pv_time),
                    "best_significance": float(varied_z[best]),
                    "threshold": float(score_edges[scan_bins[best]]),
                    "nominal_resolution": bool(
                        np.isclose(pps_time, args.pps_time_ps)
                        and np.isclose(pv_time, args.pv_time_ps)
                    ),
                }
            )

    component_diagnostics = []
    dijet_edges = np.linspace(50.0, 150.0, 101)
    component_array = np.asarray(data["component"])
    for component_id, component in enumerate(components):
        selected = eligible & (component_array == component_id)
        selected_weight = effective_physical[selected]
        dijet_histogram = np.histogram(
            np.asarray(data["dijet_mass"])[selected],
            dijet_edges,
            weights=selected_weight,
        )[0]
        peak_index = (
            int(np.argmax(dijet_histogram))
            if np.any(dijet_histogram > 0.0)
            else 0
        )
        total = float(np.sum(selected_weight))
        squared = float(np.sum(selected_weight**2))
        inputs = metadata["inputs"][component["name"]]
        component_diagnostics.append(
            {
                "name": component["name"],
                "stored_events": int(np.sum(component_array == component_id)),
                "evaluated_events": int(np.sum(selected)),
                "physical_yield": float(nominal_mass[component_id].sum()),
                "central_effective_events": (
                    total**2 / squared if squared > 0.0 else 0.0
                ),
                "dijet_mass_peak_gev": float(
                    0.5 * (dijet_edges[peak_index] + dijet_edges[peak_index + 1])
                ),
                "truth_matched_fraction_stored": inputs["truth_matched_fraction"],
                "truth_is_leading_fraction_stored": inputs[
                    "truth_is_leading_fraction"
                ],
                "correction_invalid_fraction": inputs[
                    "correction_invalid_fraction"
                ],
            }
        )

    tail_targets = np.asarray(
        getattr(args, "tail_signal_efficiencies", TAIL_SIGNAL_EFFICIENCIES)
    )
    tail_campaigns = []
    campaign_array = np.asarray(data["campaign"])
    for component_id, component in enumerate(components):
        if component["real_protons"]:
            continue
        component_rows = tail_rows[
            np.asarray(data["component"])[tail_rows] == component_id
        ]
        names = component.get("campaigns") or [component["name"]]
        for campaign_id in np.unique(campaign_array[component_rows]):
            selected = (np.asarray(data["component"])[tail_rows] == component_id) & (
                campaign_array[tail_rows] == campaign_id
            )
            event_yields = tail_event_yields[selected]
            yields = np.sum(event_yields, axis=0)
            squared = np.sum(event_yields**2, axis=0)
            campaign_rows = component_rows[
                campaign_array[component_rows] == campaign_id
            ]
            preselection_yield = float(np.sum(effective_physical[campaign_rows]))
            tail_campaigns.append(
                {
                    "component": component["name"],
                    "campaign": names[int(campaign_id)],
                    "stored_tail_events": int(event_yields.shape[0]),
                    "preselection_yield": preselection_yield,
                    "selected_yields": [float(value) for value in yields],
                    "survival_efficiencies": [
                        float(value / preselection_yield)
                        for value in yields
                    ],
                    "effective_central_events": [
                        float(total**2 / square) if square > 0.0 else 0.0
                        for total, square in zip(yields, squared)
                    ],
                }
            )

    tail_feature_names = [
        name
        for name in (
            "dijet_mass",
            "dijet_pt",
            "dijet_rapidity",
            "delta_phi_jj",
            "pt_asymmetry",
            "jet1_pt",
            "jet2_pt",
            "jet_multiplicity",
            "n_tracks_outside_jets",
            "sum_track_pt_outside_jets",
            "n_tracks_transverse",
            "sum_track_pt_transverse",
        )
        if name in metadata["central_features"]
    ]
    central_indices = {
        name: index for index, name in enumerate(metadata["central_features"])
    }
    tail_feature_quantiles = {}
    preselection_weight = effective_physical[tail_rows]
    for name in tail_feature_names:
        values = np.asarray(data["x"])[tail_rows, central_indices[name]]
        tail_feature_quantiles[name] = {
            "preselection_q10_q50_q90": _weighted_quantiles(
                values, preselection_weight
            ),
            "selected_q10_q50_q90": [
                _weighted_quantiles(values, tail_event_yields[:, index])
                for index in range(tail_targets.size)
            ],
        }
    signal_total = float(nominal_mass[0].sum())
    tail_total_yields = np.sum(tail_event_yields, axis=0)
    tail_total_squared = np.sum(tail_event_yields**2, axis=0)
    tail_diagnostics = {
        "target_signal_efficiencies": [float(value) for value in tail_targets],
        "actual_signal_efficiencies": [
            float(value / signal_total) for value in tail_signal_yields
        ],
        "fold_score_thresholds": tail_thresholds.tolist(),
        "selected_nonexclusive_yields": [
            float(value) for value in tail_total_yields
        ],
        "effective_central_events": [
            float(total**2 / square) if square > 0.0 else 0.0
            for total, square in zip(tail_total_yields, tail_total_squared)
        ],
        "campaigns": tail_campaigns,
        "central_feature_quantiles": tail_feature_quantiles,
    }

    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    score_centers = 0.5 * (score_edges[:-1] + score_edges[1:])
    category_low = score_edges[category_bins[:-1]]
    category_high = score_edges[category_bins[1:]]
    np.savez_compressed(
        result_dir / "report_data.npz",
        mass_bins=mass_edges,
        score_axis=score_centers,
        score_edges=score_edges,
        score_yields=nominal_mass.sum(axis=2),
        probability_bins=probability_edges,
        probability_histograms=probability_histograms,
        scan_thresholds=score_edges[scan_bins],
        scan_significance=scan_significance,
        scan_component_mass=scan_component_mass,
        scan_component_yields=scan_component_mass.sum(axis=2),
        scan_madgraph_effective=scan_effective,
        support_floor=np.asarray(args.support_floor),
        operating_index=np.asarray(operating),
        preselection_mass=nominal_mass.sum(axis=1),
        base_mass=base_mass,
        madgraph_base_above_squared=mg_base_above_squared,
        selected_mass=scan_component_mass[operating],
        category_mass=category_mass,
        category_low=category_low,
        category_high=category_high,
        category_significance=category_significance,
        tail_target_signal_efficiencies=tail_targets,
        tail_actual_signal_efficiencies=tail_signal_yields / signal_total,
        tail_fold_score_thresholds=tail_thresholds,
        tail_event_rows=tail_rows,
        tail_event_yields=tail_event_yields,
    )
    report = {
        "format_version": 2,
        "channel": metadata["channel"],
        "profile": metadata["profile"],
        "classes": metadata["classes"],
        "components": components,
        "tagging": metadata.get("tagging", {}),
        "feature_set": getattr(args, "feature_set", None)
        or metadata.get("feature_set"),
        "features": logical_features,
        "truth_matched_required": args.require_truth_matched,
        "class_yields": class_yields,
        "kappas": kappas,
        "component_diagnostics": component_diagnostics,
        "nonexclusive_tail_diagnostics": tail_diagnostics,
        "architecture": state.get("architecture", "standard"),
        "folds": [
            {
                "fold": fold,
                "best_iteration": (
                    {name: int(value) for name, value in iteration.items()}
                    if isinstance(iteration, dict)
                    else int(iteration)
                ),
            }
            for fold, iteration in enumerate(best_iterations)
        ],
        "single_cut_operating_point": {
            "threshold": float(score_edges[scan_bins[operating]]),
            "significance": float(scan_significance[operating]),
            "madgraph_effective_central_events": float(scan_effective[operating]),
        },
        "ladder_significance": ladder_significance,
        "category_count": int(category_mass.shape[0]),
        "support_warning": support_warning,
        "vertex": {
            "bins": args.vertex_bins,
            "pps_single_arm_time_ps": args.pps_time_ps,
            "pv_time_ps": args.pv_time_ps,
            "table": vertex,
            "resolution_scan": resolution_rows,
        },
        "runtime_seconds": time.perf_counter() - started,
        "monitoring": {
            "stored_central_events": int(metadata["rows"]),
            "logical_full_grid_rows": int(
                np.sum(pooled & eligible) * args.grid_cells
                + np.sum(~pooled & eligible)
            ),
            "saved_full_oof_probabilities": False,
            "saved_per_fold_models": bool(models_saved),
        },
    }
    if orchestration is not None:
        report["orchestration"] = orchestration
    from mva.common.dataset import write_yaml

    write_yaml(result_dir / "report.yaml", report)
    print(
        f"Wrote compact report to {result_dir}; "
        f"Z={scan_significance[operating]:.4f}, "
        f"categories={ladder_significance:.4f}, "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return report


def train_and_evaluate(data_dir, result_dir, args):
    started = time.perf_counter()
    data = load_dataset(data_dir)
    metadata = data["metadata"]
    components = metadata["components"]
    n_classes = len(metadata["classes"])
    n_components = len(components)
    selected = list(args.features) if getattr(args, "features", None) else None
    matrix, pair_column, estimator = read_logical_matrix(data, args.batch_rows, selected)
    logical_features = selected or metadata["features"]
    labels = np.asarray(data["class"], dtype=np.int32)
    groups = np.asarray(data["group_id"])
    physical = np.asarray(data["physical_weight"], dtype=np.float64)
    mixture = np.asarray(data["training_mixture_weight"], dtype=np.float64)
    band = np.asarray(data["pair_band_intensity"], dtype=np.float64)
    real_component = np.asarray([item["real_protons"] for item in components], dtype=bool)
    pooled = ~real_component[np.asarray(data["component"])]
    effective_physical = physical * np.where(pooled, band, 1.0)
    effective_mixture = mixture * np.where(pooled, band, 1.0)
    eligible = effective_physical > 0.0
    require_truth = args.require_truth_matched
    if require_truth:
        eligible &= np.asarray(data["truth_matched"])
    class_yields = np.bincount(labels[eligible], weights=effective_physical[eligible], minlength=n_classes)
    if class_yields[0] <= 0.0 or np.any(class_yields <= 0.0):
        raise RuntimeError("Every configured class needs positive eligible yield")
    kappas = class_yields[1:] / class_yields[0]
    folds = assign_folds(groups, args.seed)

    pairs = None
    if np.any(pooled):
        if metadata["protons"]["backend"] == "analytic":
            pps = load_pps_config(args.repo / args.pps_config)
            pairs = build_pair_density(
                metadata["protons"]["path"], pps, seed=args.seed, verify_hash=not args.skip_hash
            )
        elif metadata["protons"]["backend"] == "legacy_pool":
            pool = load_bootstrap_pool()
            if Path(pool["path"]).resolve() != Path(metadata["protons"]["path"]).resolve():
                raise RuntimeError("Configured legacy pool changed since dataset preparation")
            pairs = LegacyPairDensity(pool)
        else:
            raise RuntimeError(f"Unknown proton backend: {metadata['protons']['backend']}")
        cell_edges = np.linspace(-args.max_delta_y, args.max_delta_y, args.grid_cells + 1)
        sample_training_cells(
            matrix,
            pair_column,
            np.flatnonzero(pooled & eligible),
            np.asarray(data["dijet_rapidity"]),
            pairs,
            cell_edges,
            args.mass_window,
            args.pileup_mu,
            args.seed,
            args.intensity_chunk,
            estimator,
        )
    else:
        cell_edges = None

    fits = []
    for fold in range(2):
        train_rows, stop_rows = fold_rows(
            folds,
            fold,
            eligible,
            pooled,
            args.nonexclusive_train_cap,
            args.stop_cap,
            args.seed,
        )
        fold_started = time.perf_counter()
        fits.append(
            fit_calibrated(
                matrix, labels, effective_mixture, train_rows, stop_rows, n_classes, args, fold
            )
        )
        print(
            f"fold {fold}: best_iteration={fits[-1][0].best_iteration} "
            f"elapsed={time.perf_counter() - fold_started:.1f}s",
            flush=True,
        )

    score_edges = np.linspace(args.score_min, args.score_max, args.score_bins + 1)
    mass_edges = np.linspace(args.mass_window[0], args.mass_window[1], 17)
    probability_edges = np.linspace(0.0, 1.0, 51)
    base_mass = np.zeros((n_components, args.score_bins, mass_edges.size - 1))
    nominal_mass = np.zeros_like(base_mass)
    probability_histograms = np.zeros((n_classes, n_classes, probability_edges.size - 1))
    vertex = vertex_likelihood_table(
        bins=args.vertex_bins,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        single_arm_time_resolution_ps=args.pps_time_ps,
        pv_z_resolution_cm=args.pv_z_resolution_cm,
        pv_time_resolution_ps=args.pv_time_ps,
    )
    scan_bins = np.unique(np.linspace(0, args.score_bins - 1, args.scan_points).astype(int))
    mg_above_squared = np.zeros(scan_bins.size)
    mg_base_above_squared = np.zeros(args.score_bins)

    for fold, (model, calibrator) in enumerate(fits):
        for component_id, component in enumerate(components):
            rows = np.flatnonzero(
                (folds == fold) & eligible & (np.asarray(data["component"]) == component_id)
            )
            if rows.size == 0:
                raise RuntimeError(f"Fold {fold} has no eligible {component['name']} events")
            print(
                f"  score fold={fold} component={component['name']} events={rows.size:,}",
                flush=True,
            )
            if component["real_protons"]:
                probabilities = calibrated_probabilities(model, calibrator, matrix[rows], args.batch_rows)
                base_score = plugin_score(probabilities, kappas)
                _add_real_histograms(
                    nominal_mass,
                    probability_histograms,
                    component_id,
                    component["class_id"],
                    base_score,
                    probabilities,
                    np.asarray(data["proton_mx"])[rows],
                    physical[rows],
                    *_vertex_terms(vertex, component["vertex_hypothesis"]),
                    score_edges,
                    mass_edges,
                    probability_edges,
                )
                _add_real_histograms(
                    base_mass,
                    np.zeros_like(probability_histograms),
                    component_id,
                    component["class_id"],
                    base_score,
                    probabilities,
                    np.asarray(data["proton_mx"])[rows],
                    physical[rows],
                    np.array([1.0]),
                    np.array([0.0]),
                    score_edges,
                    mass_edges,
                    probability_edges,
                )
                continue

            score_yield = np.zeros(args.score_bins)
            base_score_yield = np.zeros(args.score_bins)
            scoring_started = time.perf_counter()
            for start in range(0, rows.size, args.evaluation_chunk):
                stop = min(start + args.evaluation_chunk, rows.size)
                selected = rows[start:stop]
                working = np.repeat(matrix[selected], args.grid_cells, axis=0)
                delta = np.tile(0.5 * (cell_edges[:-1] + cell_edges[1:]), selected.size)
                set_proton_cells(
                    working, pair_column, estimator,
                    np.repeat(selected, args.grid_cells), delta,
                )
                probabilities = calibrated_probabilities(model, calibrator, working, args.batch_rows)
                base_score = plugin_score(probabilities, kappas).reshape(selected.size, args.grid_cells)
                intensity = proton_cell_intensities(
                    pairs,
                    np.asarray(data["dijet_rapidity"])[selected],
                    cell_edges,
                    args.mass_window,
                    args.pileup_mu,
                )
                central_weight = physical[selected, np.newaxis]
                base_contribution = central_weight * intensity
                for predicted in range(n_classes):
                    probability_histograms[component["class_id"], predicted] += np.histogram(
                        probabilities[:, predicted],
                        probability_edges,
                        weights=base_contribution.ravel(),
                    )[0]
                base_bucket = _score_bins(base_score, score_edges)
                base_score_yield += np.bincount(
                    base_bucket.ravel(), weights=base_contribution.ravel(), minlength=args.score_bins
                )
                per_event_base = np.zeros((selected.size, args.score_bins), dtype=np.float64)
                np.add.at(
                    per_event_base,
                    (np.repeat(np.arange(selected.size), args.grid_cells), base_bucket.ravel()),
                    base_contribution.ravel(),
                )
                base_above = np.cumsum(per_event_base[:, ::-1], axis=1)[:, ::-1]
                mg_base_above_squared += np.sum(base_above**2, axis=0)
                per_event_bucket = np.zeros((selected.size, args.score_bins), dtype=np.float64)
                for vertex_probability, shift in zip(
                    *_vertex_terms(vertex, component["vertex_hypothesis"])
                ):
                    bucket = _score_bins(base_score + shift, score_edges)
                    contribution = base_contribution * vertex_probability
                    score_yield += np.bincount(
                        bucket.ravel(), weights=contribution.ravel(), minlength=args.score_bins
                    )
                    np.add.at(
                        per_event_bucket,
                        (np.repeat(np.arange(selected.size), args.grid_cells), bucket.ravel()),
                        contribution.ravel(),
                    )
                above = np.cumsum(per_event_bucket[:, ::-1], axis=1)[:, ::-1]
                mg_above_squared += np.sum(above[:, scan_bins] ** 2, axis=0)
                elapsed = time.perf_counter() - scoring_started
                print(
                    f"    cells {stop:,}/{rows.size:,} elapsed={elapsed:.1f}s "
                    f"eta={elapsed / stop * (rows.size - stop):.1f}s",
                    flush=True,
                )

            rapidity = np.asarray(data["dijet_rapidity"])[rows]
            shape = _madgraph_mass_shape(
                pairs,
                rapidity,
                physical[rows],
                mass_edges,
                args.max_delta_y,
                args.pileup_mu,
                args.intensity_chunk,
            )
            nominal_mass[component_id] += score_yield[:, np.newaxis] * shape
            base_mass[component_id] += base_score_yield[:, np.newaxis] * shape

    cumulative = np.cumsum(nominal_mass[:, ::-1, :], axis=1)[:, ::-1, :]
    scan_component_mass = np.transpose(cumulative[:, scan_bins, :], (1, 0, 2))
    scan_significance = _significance(scan_component_mass)
    scan_total_mg = scan_component_mass[:, ~real_component].sum(axis=(1, 2))
    scan_effective = np.divide(
        scan_total_mg**2,
        mg_above_squared,
        out=np.zeros_like(scan_total_mg),
        where=mg_above_squared > 0.0,
    )
    valid = np.flatnonzero(scan_effective >= args.support_floor)
    if valid.size == 0:
        valid = np.arange(scan_significance.size)
        support_warning = "No scan point reached the requested MadGraph support floor"
    else:
        support_warning = None
    operating = valid[np.argmax(scan_significance[valid])]

    signal_score_yield = nominal_mass[0].sum(axis=1)
    signal_cdf = np.cumsum(signal_score_yield)
    quantiles = np.arange(1, args.ladder_bins) / args.ladder_bins * signal_cdf[-1]
    category_bins = np.r_[0, np.searchsorted(signal_cdf, quantiles), args.score_bins]
    category_bins = np.unique(category_bins)
    category_mass = np.stack(
        [nominal_mass[:, low:high].sum(axis=1) for low, high in zip(category_bins[:-1], category_bins[1:])]
    )
    category_significance = _significance(category_mass)
    ladder_significance = float(np.sqrt(np.sum(category_significance**2)))

    resolution_rows = []
    for pps_time in args.pps_time_scan:
        for pv_time in args.pv_time_scan:
            table = vertex_likelihood_table(
                bins=args.vertex_bins,
                beam_sigma_z_cm=args.beam_sigma_z_cm,
                single_arm_time_resolution_ps=pps_time,
                pv_z_resolution_cm=args.pv_z_resolution_cm,
                pv_time_resolution_ps=pv_time,
            )
            if np.isclose(pps_time, args.pps_time_ps) and np.isclose(
                pv_time, args.pv_time_ps
            ):
                varied = nominal_mass
            else:
                varied = _vertex_convolution(base_mass, components, table, score_edges)
            varied_cumulative = np.cumsum(varied[:, ::-1, :], axis=1)[:, ::-1, :]
            varied_scan = np.transpose(varied_cumulative[:, scan_bins], (1, 0, 2))
            varied_z = _significance(varied_scan)
            best = int(np.argmax(varied_z))
            resolution_rows.append(
                {
                    "pps_single_arm_time_ps": float(pps_time),
                    "pv_time_ps": float(pv_time),
                    "best_significance": float(varied_z[best]),
                    "threshold": float(score_edges[scan_bins[best]]),
                    "nominal_resolution": bool(
                        np.isclose(pps_time, args.pps_time_ps)
                        and np.isclose(pv_time, args.pv_time_ps)
                    ),
                }
            )

    component_diagnostics = []
    dijet_edges = np.linspace(50.0, 150.0, 101)
    component_array = np.asarray(data["component"])
    for component_id, component in enumerate(components):
        selected = eligible & (component_array == component_id)
        selected_weight = effective_physical[selected]
        dijet_histogram = np.histogram(
            np.asarray(data["dijet_mass"])[selected],
            dijet_edges,
            weights=selected_weight,
        )[0]
        peak_index = int(np.argmax(dijet_histogram)) if np.any(dijet_histogram > 0.0) else 0
        total = float(np.sum(selected_weight))
        squared = float(np.sum(selected_weight**2))
        inputs = metadata["inputs"][component["name"]]
        component_diagnostics.append(
            {
                "name": component["name"],
                "stored_events": int(np.sum(component_array == component_id)),
                "evaluated_events": int(np.sum(selected)),
                "physical_yield": float(nominal_mass[component_id].sum()),
                "central_effective_events": total**2 / squared if squared > 0.0 else 0.0,
                "dijet_mass_peak_gev": float(
                    0.5 * (dijet_edges[peak_index] + dijet_edges[peak_index + 1])
                ),
                "truth_matched_fraction_stored": inputs["truth_matched_fraction"],
                "truth_is_leading_fraction_stored": inputs["truth_is_leading_fraction"],
                "correction_invalid_fraction": inputs["correction_invalid_fraction"],
            }
        )

    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    score_centers = 0.5 * (score_edges[:-1] + score_edges[1:])
    category_low = score_edges[category_bins[:-1]]
    category_high = score_edges[category_bins[1:]]
    np.savez_compressed(
        result_dir / "report_data.npz",
        mass_bins=mass_edges,
        score_axis=score_centers,
        score_edges=score_edges,
        score_yields=nominal_mass.sum(axis=2),
        probability_bins=probability_edges,
        probability_histograms=probability_histograms,
        scan_thresholds=score_edges[scan_bins],
        scan_significance=scan_significance,
        scan_component_mass=scan_component_mass,
        scan_component_yields=scan_component_mass.sum(axis=2),
        scan_madgraph_effective=scan_effective,
        support_floor=np.asarray(args.support_floor),
        operating_index=np.asarray(operating),
        preselection_mass=nominal_mass.sum(axis=1),
        base_mass=base_mass,
        madgraph_base_above_squared=mg_base_above_squared,
        selected_mass=scan_component_mass[operating],
        category_mass=category_mass,
        category_low=category_low,
        category_high=category_high,
        category_significance=category_significance,
    )
    report = {
        "format_version": 2,
        "channel": metadata["channel"],
        "profile": metadata["profile"],
        "classes": metadata["classes"],
        "components": components,
        "tagging": metadata.get("tagging", {}),
        "feature_set": getattr(args, "feature_set", None) or metadata.get("feature_set"),
        "features": logical_features,
        "truth_matched_required": require_truth,
        "class_yields": class_yields,
        "kappas": kappas,
        "component_diagnostics": component_diagnostics,
        "folds": [
            {"fold": fold, "best_iteration": int(model.best_iteration)}
            for fold, (model, _calibrator) in enumerate(fits)
        ],
        "single_cut_operating_point": {
            "threshold": float(score_edges[scan_bins[operating]]),
            "significance": float(scan_significance[operating]),
            "madgraph_effective_central_events": float(scan_effective[operating]),
        },
        "ladder_significance": ladder_significance,
        "category_count": int(category_mass.shape[0]),
        "support_warning": support_warning,
        "vertex": {
            "bins": args.vertex_bins,
            "pps_single_arm_time_ps": args.pps_time_ps,
            "pv_time_ps": args.pv_time_ps,
            "table": vertex,
            "resolution_scan": resolution_rows,
        },
        "runtime_seconds": time.perf_counter() - started,
        "monitoring": {
            "stored_central_events": int(metadata["rows"]),
            "logical_full_grid_rows": int(np.sum(pooled & eligible) * args.grid_cells + np.sum(~pooled & eligible)),
            "saved_full_oof_probabilities": False,
            "saved_per_fold_models": False,
        },
    }
    from mva.common.dataset import write_yaml

    write_yaml(result_dir / "report.yaml", report)
    print(
        f"Wrote compact report to {result_dir}; Z={scan_significance[operating]:.4f}, "
        f"categories={ladder_significance:.4f}, elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return report
