#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


FEATURE_NAMES = (
    "jet1_pt_over_mjj",
    "jet2_pt_over_mjj",
    "jet1_eta",
    "jet2_eta",
    "delta_r_jj",
    "dijet_pt",
    "dijet_eta",
)
PROCESS_ORDER = (
    "qcd_gg",
    "qcd_qq",
    "qcd_bb",
    "qcd_cc",
    "qed_bb",
    "qed_cc",
    "h_bb",
    "h_cc",
)
PROCESS_LABELS = {
    "qcd_gg": "QCD gg",
    "qcd_qq": "QCD qq",
    "qcd_bb": "QCD bb",
    "qcd_cc": "QCD cc",
    "qed_bb": "QED bb",
    "qed_cc": "QED cc",
    "h_bb": "H->bb",
    "h_cc": "H->cc",
}
COLORS = {
    "h_bb": "#0072B2",
    "h_cc": "#D55E00",
    "qed_bb": "#009E73",
    "qed_cc": "#CC79A7",
    "qcd_bb": "#E69F00",
    "qcd_cc": "#56B4E9",
    "qcd_qq": "#000000",
    "qcd_gg": "#F0E442",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate dijet MVA plots from cached run_dijet_mva.py outputs."
    )
    parser.add_argument(
        "--input-dir",
        "--output-dir",
        dest="input_dir",
        required=True,
        help="Directory containing dataset.npz, splits.npz, scores.npz, and model.json.",
    )
    return parser.parse_args()


def import_libraries():
    import matplotlib
    import numpy as np
    from xgboost import XGBClassifier

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return np, plt, XGBClassifier


def load_required_npz(np, path):
    if not path.is_file():
        raise RuntimeError(f"Missing required input: {path}")
    return np.load(path, allow_pickle=False)


def normalized_class_weights(np, labels, weights):
    normalized = np.asarray(weights, dtype=np.float64).copy()
    for label in (0, 1):
        mask = labels == label
        total = float(np.sum(normalized[mask]))
        if total <= 0.0:
            raise RuntimeError(f"Class {label} has non-positive total weight")
        normalized[mask] /= total
    return normalized


def plot_validation_score(
    np,
    plt,
    output_path,
    val_scores,
    val_labels,
    val_weights,
    train_scores,
    train_labels,
    train_weights,
):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bins = np.linspace(0.0, 1.0, 41)
    normalized_val = normalized_class_weights(np, val_labels, val_weights)
    normalized_train = normalized_class_weights(np, train_labels, train_weights)
    for label, text, color in ((0, "Background", "#D55E00"), (1, "Signal", "#0072B2")):
        mask = val_labels == label
        ax.hist(
            val_scores[mask],
            bins=bins,
            weights=normalized_val[mask],
            histtype="step",
            linewidth=1.5,
            color=color,
            label=f"{text} validation",
        )
        train_mask = train_labels == label
        ax.hist(
            train_scores[train_mask],
            bins=bins,
            weights=normalized_train[train_mask],
            histtype="step",
            linewidth=1.5,
            linestyle="--",
            color=color,
            label=f"{text} train",
        )
    ax.set_xlabel("XGBoost score")
    ax.set_ylabel("Normalized events / bin")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc(plt, output_path, fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(fpr, tpr, color="#0072B2", linewidth=1.8, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(np, plt, output_path, model):
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.barh(np.arange(len(FEATURE_NAMES)), importances[order], color="#009E73")
    ax.set_yticks(np.arange(len(FEATURE_NAMES)), [FEATURE_NAMES[index] for index in order])
    ax.set_xlabel("XGBoost feature importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_tmva_score(np, plt, output_path, scores, labels, weights, processes, log_y=False):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bins = np.linspace(0.0, 1.0, 41)
    positive_counts = []
    for process_name in PROCESS_ORDER:
        mask = processes == process_name
        if not np.any(mask):
            continue
        linestyle = "-" if np.any(labels[mask] == 1) else "--"
        counts, _edges, _patches = ax.hist(
            scores[mask],
            bins=bins,
            weights=weights[mask],
            histtype="step",
            linewidth=1.4,
            linestyle=linestyle,
            color=COLORS.get(process_name),
            label=f"{PROCESS_LABELS.get(process_name, process_name)} ({np.sum(weights[mask]):.3g})",
        )
        positive_counts.extend(counts[counts > 0.0])
    ax.set_xlabel("XGBoost score")
    ax.set_ylabel("Expected events / bin")
    ax.set_xlim(0.0, 1.0)
    if log_y:
        ax.set_yscale("log")
        if positive_counts:
            ax.set_ylim(bottom=float(np.min(positive_counts)) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_plots(np, plt, XGBClassifier, input_dir):
    dataset = load_required_npz(np, input_dir / "dataset.npz")
    splits = load_required_npz(np, input_dir / "splits.npz")
    scores = load_required_npz(np, input_dir / "scores.npz")
    model_path = input_dir / "model.json"
    if not model_path.is_file():
        raise RuntimeError(f"Missing required input: {model_path}")

    model = XGBClassifier()
    model.load_model(model_path)

    train_idx = splits["train"]
    val_idx = splits["validation"]
    labels = dataset["y"]
    weights = dataset["physical_weight"]
    processes = dataset["process"]
    all_scores = scores["all"] if "all" in scores.files else model.predict_proba(dataset["x"])[:, 1]

    outputs = {
        "validation_score": input_dir / "validation_score.png",
        "roc": input_dir / "roc.png",
        "feature_importance": input_dir / "feature_importance.png",
        "tmva_score": input_dir / "tmva_score.png",
        "tmva_score_log": input_dir / "tmva_score_log.png",
    }
    plot_validation_score(
        np,
        plt,
        outputs["validation_score"],
        scores["validation"],
        labels[val_idx],
        weights[val_idx],
        scores["train"],
        labels[train_idx],
        weights[train_idx],
    )
    plot_roc(plt, outputs["roc"], scores["fpr"], scores["tpr"], float(scores["roc_auc"]))
    plot_feature_importance(np, plt, outputs["feature_importance"], model)
    plot_tmva_score(
        np,
        plt,
        outputs["tmva_score"],
        all_scores,
        labels,
        weights,
        processes,
    )
    plot_tmva_score(
        np,
        plt,
        outputs["tmva_score_log"],
        all_scores,
        labels,
        weights,
        processes,
        log_y=True,
    )
    return outputs


def main():
    args = parse_args()
    np, plt, XGBClassifier = import_libraries()
    input_dir = Path(args.input_dir)
    outputs = write_plots(np, plt, XGBClassifier, input_dir)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
