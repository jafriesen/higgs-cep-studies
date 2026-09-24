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
    "delta_phi_jj",
    "delta_eta_jj",
    "dijet_pt",
    "dijet_eta",
    "jet_multiplicity",
    "track_multiplicity_r_gt_0p4_pt1",
    "dijet_mass",
    "mx_minus_dijet_mass",
    "yx_minus_dijet_rapidity",
)
PROCESS_ORDER = (
    "Hbb",
    "QCDbb",
    "QCDbb_madgraph_comb",
)
PROCESS_LABELS = {
    "Hbb": "H->bb",
    "QCDbb": "SuperChic QCDbb",
    "QCDbb_madgraph_comb": "MadGraph QCDbb + min-bias protons",
}
COLORS = {
    "Hbb": "#0072B2",
    "QCDbb": "#E69F00",
    "QCDbb_madgraph_comb": "#D55E00",
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
        help="Directory containing dataset.npz, scores.npz, and model.json.",
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


def logit_scores(np, scores):
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def binned_significance(np, values, labels, weights, bins):
    signal_counts, _ = np.histogram(
        values[labels == 1], bins=bins, weights=weights[labels == 1]
    )
    background_counts, _ = np.histogram(
        values[labels == 0], bins=bins, weights=weights[labels == 0]
    )
    denominator = signal_counts + background_counts
    z_bins = np.zeros_like(signal_counts, dtype=np.float64)
    nonzero = denominator > 0.0
    z_bins[nonzero] = signal_counts[nonzero] / np.sqrt(denominator[nonzero])
    return float(np.sqrt(np.sum(z_bins * z_bins)))


def plot_oof_score(
    np,
    plt,
    output_path,
    oof_scores,
    train_final_scores,
    labels,
    weights,
    logit_x=False,
):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    if logit_x:
        oof_scores = logit_scores(np, oof_scores)
        train_final_scores = logit_scores(np, train_final_scores)
        low = float(min(oof_scores.min(), train_final_scores.min()))
        high = float(max(oof_scores.max(), train_final_scores.max()))
        bins = np.linspace(low, high, 100)
    else:
        bins = np.linspace(0.0, 1.0, 100)
    normalized = normalized_class_weights(np, labels, weights)
    for label, text, color in ((0, "Background", "#D55E00"), (1, "Signal", "#0072B2")):
        mask = labels == label
        ax.hist(
            oof_scores[mask],
            bins=bins,
            weights=normalized[mask],
            histtype="step",
            linewidth=1.5,
            color=color,
            label=f"{text} out-of-fold",
        )
        ax.hist(
            train_final_scores[mask],
            bins=bins,
            weights=normalized[mask],
            histtype="step",
            linewidth=1.5,
            linestyle="--",
            color=color,
            label=f"{text} final model on train",
        )
    ax.set_xlabel("logit(XGBoost score)" if logit_x else "XGBoost score")
    ax.set_ylabel("Normalized events / bin")
    ax.set_xlim(float(bins[0]), float(bins[-1]))
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


def plot_feature_importance(np, plt, output_path, model, feature_names):
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(feature_names) != importances.shape[0]:
        raise RuntimeError(
            f"Feature-name count {len(feature_names)} does not match model importance count "
            f"{importances.shape[0]}"
        )
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.barh(np.arange(len(feature_names)), importances[order], color="#009E73")
    ax.set_yticks(np.arange(len(feature_names)), [feature_names[index] for index in order])
    ax.set_xlabel("XGBoost feature importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_tmva_score(np, plt, output_path, scores, labels, weights, processes, log_y=False, logit_x=False):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    if logit_x:
        scores = logit_scores(np, scores)
        bins = np.linspace(float(scores.min()), float(scores.max()), 100)
    else:
        bins = np.linspace(0.5, 1.0, 100)
    significance = binned_significance(np, scores, labels, weights, bins)
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
    ax.set_xlabel("logit(XGBoost score)" if logit_x else "XGBoost score")
    ax.set_ylabel("Expected events / bin")
    ax.set_xlim(bins[0], bins[-1])
    if log_y:
        ax.set_yscale("log")
        if positive_counts:
            ax.set_ylim(bottom=float(np.min(positive_counts)) * 0.5)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none"))
    legend_labels.append(f"Binned MVA Z = {significance:.3g}")
    ax.legend(handles, legend_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_plots(np, plt, XGBClassifier, input_dir):
    dataset = load_required_npz(np, input_dir / "dataset.npz")
    scores = load_required_npz(np, input_dir / "scores.npz")
    model_path = input_dir / "model.json"
    if not model_path.is_file():
        raise RuntimeError(f"Missing required input: {model_path}")

    model = XGBClassifier()
    model._estimator_type = "classifier"
    model.load_model(model_path)
    model.n_classes_ = 2

    labels = dataset["y"]
    weights = dataset["physical_weight"]
    processes = dataset["process"]
    if "training_feature_names" in dataset.files:
        feature_names = tuple(str(name) for name in dataset["training_feature_names"])
    elif "feature_names" in dataset.files:
        feature_names = tuple(str(name) for name in dataset["feature_names"])
    else:
        feature_names = FEATURE_NAMES
    all_scores = scores["all"] if "all" in scores.files else model.predict_proba(dataset["x"])[:, 1]
    train_final = (
        scores["train_final"]
        if "train_final" in scores.files
        else model.predict_proba(dataset["x"])[:, 1]
    )

    outputs = {
        "oof_score": input_dir / "oof_score.png",
        "oof_score_logit": input_dir / "oof_score_logit.png",
        "roc": input_dir / "roc.png",
        "feature_importance": input_dir / "feature_importance.png",
        "tmva_score": input_dir / "tmva_score.png",
        "tmva_score_log": input_dir / "tmva_score_log.png",
        "tmva_score_logit": input_dir / "tmva_score_logit.png",
        "tmva_score_logit_log": input_dir / "tmva_score_logit_log.png",
    }
    plot_oof_score(
        np,
        plt,
        outputs["oof_score"],
        all_scores,
        train_final,
        labels,
        weights,
    )
    plot_oof_score(
        np,
        plt,
        outputs["oof_score_logit"],
        all_scores,
        train_final,
        labels,
        weights,
        logit_x=True,
    )
    plot_roc(plt, outputs["roc"], scores["fpr"], scores["tpr"], float(scores["roc_auc"]))
    plot_feature_importance(np, plt, outputs["feature_importance"], model, feature_names)
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
    plot_tmva_score(
        np,
        plt,
        outputs["tmva_score_logit"],
        all_scores,
        labels,
        weights,
        processes,
        logit_x=True,
    )
    plot_tmva_score(
        np,
        plt,
        outputs["tmva_score_logit_log"],
        all_scores,
        labels,
        weights,
        processes,
        log_y=True,
        logit_x=True,
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
