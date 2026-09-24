#!/usr/bin/env python3
"""Multiclass (softmax) central-only H->bb MVA over the cached feature dataset.

Trains one 3-class XGBoost model (Hbb / SuperChic QCDbb / MadGraph QCDbb) on
the uncut dataset produced by run_dijet_mva_exclusive_central.py, with equal
training weight per class. Unlike a binary signal-vs-mixture classifier, the
softmax model estimates each process density separately, so every pairwise
likelihood ratio is recoverable from one model:

  score_vs_b = p_sig / (p_sig + p_b)

and the analysis discriminant with plug-in background normalizations kappa_b,

  D = p_sig / (p_sig + sum_b kappa_b p_b),

can be re-weighted after training when normalizations change (e.g. when the
proton coincidence suppression of the MadGraph background comes back in),
without retraining. Reported metrics: OOF pairwise AUCs vs each background
(the numbers to compare with the dedicated binary trainings) and the AUC of D
with physical-weight kappas as one example working point.

Run under the analysis environment:
  source setup_env.sh
  python3 analysis/MVA/run_dijet_mva_multiclass.py
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
CLASS_LABELS = {
    "Hbb": "H->bb (SuperChic)",
    "QCDbb": "SuperChic QCDbb",
    "QCDbb_madgraph": "MadGraph QCDbb (inclusive)",
}
CV_FOLDS = 5
SEARCH_MAX_ROWS = 300000

PARAM_GRID = tuple(
    {"max_depth": depth, "learning_rate": rate, "min_child_weight": child}
    for depth in (4, 6, 8)
    for rate in (0.02, 0.05)
    for child in (1, 20)
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-from",
        default=str(REPO / "analysis/MVA/output/mva_bb_exclusive_central/dataset.npz"),
        help="Uncut feature dataset from run_dijet_mva_exclusive_central.py.",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip the hyperparameter grid search and use the default configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "analysis/MVA/output/mva_bb_multiclass"),
        help="Output directory for model, scores, plots, and summary.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    return parser.parse_args()


def load_dataset(path):
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise RuntimeError(
            f"Missing dataset {dataset_path}; run run_dijet_mva_exclusive_central.py first"
        )
    source = np.load(dataset_path, allow_pickle=True)
    dataset = {name: source[name] for name in source.files}
    process = dataset["process"].astype(str)
    missing = [name for name in CLASS_ORDER if not np.any(process == name)]
    if missing:
        raise RuntimeError(f"Dataset lacks events for: {', '.join(missing)}")
    class_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    dataset["process"] = process
    dataset["class"] = np.array([class_index[name] for name in process], dtype=np.int8)
    return dataset


def class_balanced_weights(classes):
    """Equal total weight per class, normalized to mean 1 per event."""
    weights = np.zeros(classes.shape[0], dtype=np.float64)
    n_classes = len(np.unique(classes))
    for value in np.unique(classes):
        mask = classes == value
        weights[mask] = 1.0 / (n_classes * np.sum(mask))
    return weights * classes.shape[0]


def pair_balanced_weights(labels):
    weights = np.ones(labels.shape[0], dtype=np.float64)
    for label in (0, 1):
        weights[labels == label] /= np.sum(labels == label)
    return weights


def pairwise_scores(probabilities, background_class):
    with np.errstate(divide="ignore", invalid="ignore"):
        return probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, background_class])


def pairwise_auc(roc_auc_score, dataset, probabilities, background):
    background_class = CLASS_ORDER.index(background)
    mask = (dataset["process"] == "Hbb") | (dataset["process"] == background)
    labels = (dataset["process"][mask] == "Hbb").astype(np.int8)
    scores = pairwise_scores(probabilities[mask], background_class)
    return float(
        roc_auc_score(labels, scores, sample_weight=pair_balanced_weights(labels))
    )


def plug_in_discriminant(probabilities, kappas):
    background_sum = sum(
        kappa * probabilities[:, CLASS_ORDER.index(name)] for name, kappa in kappas.items()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return probabilities[:, 0] / (probabilities[:, 0] + background_sum)


def make_model(XGBClassifier, params, seed):
    return XGBClassifier(
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=8,
        tree_method="hist",
        early_stopping_rounds=30,
        **params,
    )


def fit_model(XGBClassifier, train_test_split, x, classes, weights, params, seed):
    fit_idx, stop_idx = train_test_split(
        np.arange(classes.shape[0]), train_size=0.9, random_state=seed, stratify=classes
    )
    model = make_model(XGBClassifier, params, seed)
    model.fit(
        x[fit_idx],
        classes[fit_idx],
        sample_weight=weights[fit_idx],
        eval_set=[(x[stop_idx], classes[stop_idx])],
        sample_weight_eval_set=[weights[stop_idx]],
        verbose=False,
    )
    return model


def search_metric(roc_auc_score, dataset, probabilities, mask):
    subset = {"process": dataset["process"][mask]}
    auc_sc = pairwise_auc(roc_auc_score, subset, probabilities, "QCDbb")
    auc_mg = pairwise_auc(roc_auc_score, subset, probabilities, "QCDbb_madgraph")
    return 0.5 * (auc_sc + auc_mg), auc_sc, auc_mg


def grid_search(XGBClassifier, train_test_split, roc_auc_score, dataset, weights, seed, rng):
    x, classes = dataset["x"], dataset["class"]
    rows = np.arange(classes.shape[0])
    if rows.size > SEARCH_MAX_ROWS:
        rows = rng.choice(rows, size=SEARCH_MAX_ROWS, replace=False)
    train_idx, test_idx = train_test_split(
        rows, train_size=0.75, random_state=seed, stratify=classes[rows]
    )
    results = []
    for params in PARAM_GRID:
        model = fit_model(
            XGBClassifier, train_test_split, x[train_idx], classes[train_idx],
            weights[train_idx], params, seed,
        )
        probabilities = model.predict_proba(x[test_idx])
        mean_auc, auc_sc, auc_mg = search_metric(roc_auc_score, dataset, probabilities, test_idx)
        results.append(
            {**params, "mean_pairwise_auc": float(mean_auc), "auc_sc": float(auc_sc),
             "auc_mg": float(auc_mg), "best_iteration": int(model.best_iteration)}
        )
        print(
            f"search: depth={params['max_depth']} lr={params['learning_rate']} "
            f"min_child_weight={params['min_child_weight']} mean_auc={mean_auc:.5f} "
            f"(sc={auc_sc:.5f}, mg={auc_mg:.5f}) best_iter={model.best_iteration}",
            flush=True,
        )
    best = max(results, key=lambda item: item["mean_pairwise_auc"])
    print(f"best configuration: {best}", flush=True)
    return {k: best[k] for k in ("max_depth", "learning_rate", "min_child_weight")}, results


def train_oof(XGBClassifier, StratifiedKFold, train_test_split, dataset, weights, params, seed):
    x, classes = dataset["x"], dataset["class"]
    oof = np.full((classes.shape[0], len(CLASS_ORDER)), np.nan)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    best_iterations = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, classes)):
        model = fit_model(
            XGBClassifier, train_test_split, x[train_idx], classes[train_idx],
            weights[train_idx], params, seed,
        )
        oof[test_idx] = model.predict_proba(x[test_idx])
        best_iterations.append(int(model.best_iteration))
        print(f"fold_{fold}: train={train_idx.size} test={test_idx.size} best_iter={model.best_iteration}", flush=True)
    if np.any(np.isnan(oof)):
        raise RuntimeError("Cross-validation left events without an out-of-fold score")
    final_model = fit_model(XGBClassifier, train_test_split, x, classes, weights, params, seed)
    return oof, best_iterations, final_model


def plot_outputs(roc_curve, dataset, oof, model, output_dir):
    styles = {
        "Hbb": ("#0072B2", "-"),
        "QCDbb": ("#E69F00", "--"),
        "QCDbb_madgraph": ("#D55E00", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    bins = np.linspace(0.0, 1.0, 51)
    for ax, background in zip(axes, ("QCDbb", "QCDbb_madgraph")):
        background_class = CLASS_ORDER.index(background)
        scores = pairwise_scores(oof, background_class)
        for name, (color, linestyle) in styles.items():
            mask = dataset["process"] == name
            ax.hist(
                scores[mask], bins=bins, density=True, histtype="step",
                color=color, linestyle=linestyle, linewidth=1.6, label=CLASS_LABELS[name],
            )
        ax.set_xlabel(f"p(sig) / (p(sig) + p({background}))")
        ax.set_ylabel("Normalized events / bin")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_pairwise_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for background in ("QCDbb", "QCDbb_madgraph"):
        color, linestyle = styles[background]
        background_class = CLASS_ORDER.index(background)
        mask = (dataset["process"] == "Hbb") | (dataset["process"] == background)
        labels = (dataset["process"][mask] == "Hbb").astype(np.int8)
        fpr, tpr, _ = roc_curve(
            labels, pairwise_scores(oof[mask], background_class),
            sample_weight=pair_balanced_weights(labels),
        )
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, label=f"Hbb vs {CLASS_LABELS[background]}")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc.png", dpi=160)
    plt.close(fig)

    names = [str(name) for name in dataset["feature_names"]]
    importance = model.get_booster().get_score(importance_type="gain")
    gains = np.array([importance.get(f"f{i}", 0.0) for i in range(len(names))])
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1.5))
    ax.barh(np.arange(len(names)), gains[order], color="#0072B2")
    ax.set_yticks(np.arange(len(names)), [names[i] for i in order], fontsize=8)
    ax.set_xlabel("XGBoost gain")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from xgboost import XGBClassifier

    dataset = load_dataset(args.dataset_from)
    weights = class_balanced_weights(dataset["class"])
    for name in CLASS_ORDER:
        print(f"{name}: events={int(np.sum(dataset['process'] == name))}", flush=True)
    print(f"dataset: {dataset['x'].shape[0]} events, {dataset['x'].shape[1]} features", flush=True)

    rng = np.random.default_rng(args.seed)
    if args.skip_search:
        best_params = {"max_depth": 6, "learning_rate": 0.02, "min_child_weight": 20}
        search_results = []
    else:
        best_params, search_results = grid_search(
            XGBClassifier, train_test_split, roc_auc_score, dataset, weights, args.seed, rng
        )

    oof, best_iterations, model = train_oof(
        XGBClassifier, StratifiedKFold, train_test_split, dataset, weights, best_params, args.seed
    )
    auc_sc = pairwise_auc(roc_auc_score, dataset, oof, "QCDbb")
    auc_mg = pairwise_auc(roc_auc_score, dataset, oof, "QCDbb_madgraph")
    print(f"OOF pairwise AUC Hbb vs SuperChic QCDbb: {auc_sc:.5f}")
    print(f"OOF pairwise AUC Hbb vs MadGraph QCDbb:  {auc_mg:.5f}")

    # example plug-in working point: physical background yields per event
    physical = dataset["physical_weight"]
    signal_total = float(np.sum(physical[dataset["process"] == "Hbb"]))
    kappas = {
        name: float(np.sum(physical[dataset["process"] == name])) / signal_total
        for name in CLASS_ORDER[1:]
    }
    discriminant = plug_in_discriminant(oof, kappas)
    labels = (dataset["process"] == "Hbb").astype(np.int8)
    auc_plug_in = float(
        roc_auc_score(labels, discriminant, sample_weight=pair_balanced_weights(labels))
    )
    print(f"plug-in D (physical kappas {kappas}): class-balanced AUC {auc_plug_in:.5f}")

    np.savez_compressed(
        output_dir / "scores.npz",
        oof_probabilities=oof,
        process=dataset["process"],
        training_weight=weights,
    )
    model._estimator_type = "classifier"
    model.save_model(output_dir / "model.json")
    plot_outputs(roc_curve, dataset, oof, model, output_dir)

    summary = {
        "dataset_from": str(args.dataset_from),
        "class_order": list(CLASS_ORDER),
        "training_weights": "equal total weight per class, mean 1 per event",
        "events_per_class": {
            name: int(np.sum(dataset["process"] == name)) for name in CLASS_ORDER
        },
        "hyperparameter_search": search_results,
        "best_params": best_params,
        "cv_best_iterations": best_iterations,
        "auc_oof_pairwise_hbb_vs_superchic_qcdbb": auc_sc,
        "auc_oof_pairwise_hbb_vs_madgraph_qcdbb": auc_mg,
        "plug_in_kappas_physical": kappas,
        "auc_oof_plug_in_physical": auc_plug_in,
        "seed": args.seed,
    }
    with open(output_dir / "summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"Wrote model, scores, plots, and summary to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
