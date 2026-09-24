#!/usr/bin/env python3
"""Create the standard diagnostic report for the proton multiclass MVA."""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
LABELS = (
    r"$H\rightarrow b\bar{b}$",
    "SuperChic QCD",
    "MadGraph QCD",
)
COLORS = ("#0072B2", "#E69F00", "#D55E00")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--madgraph-neff-marker", type=float, default=200.0)
    return parser.parse_args()


def plugin_score(probabilities, kappas):
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    return np.log(clipped[:, 0]) - np.log(
        kappas[0] * clipped[:, 1] + kappas[1] * clipped[:, 2]
    )


def score_only_significance(classes, weights, selections):
    z2 = 0.0
    regions = []
    for selection in selections:
        yields = [
            float(np.sum(weights[selection & (classes == class_id)]))
            for class_id in range(3)
        ]
        signal = yields[0]
        background = yields[1] + yields[2]
        z = signal / np.sqrt(signal + background) if signal + background else 0.0
        z2 += z * z
        regions.append({"yields": yields, "significance": float(z)})
    return float(np.sqrt(z2)), regions


def histogram_density(values, bins, weights):
    histogram, _ = np.histogram(values, bins=bins, weights=weights)
    integral = np.sum(histogram)
    if integral > 0.0:
        histogram = histogram / (integral * np.diff(bins))
    return histogram


def plot_scores(classes, weights, probabilities, score, summary, result_dir):
    threshold = float(summary["locked_optimization"]["single_cut"]["threshold"])
    low, high = summary["locked_optimization"]["categories"]["boundaries"]
    score_bins = np.linspace(np.floor(np.min(score)), np.ceil(np.max(score)), 101)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    for class_id, (label, color) in enumerate(zip(LABELS, COLORS)):
        mask = classes == class_id
        density = histogram_density(score[mask], score_bins, weights[mask])
        expected, _ = np.histogram(score[mask], score_bins, weights=weights[mask])
        axes[0].stairs(
            density, score_bins, color=color, linewidth=1.8, label=label
        )
        axes[1].stairs(
            expected, score_bins, color=color, linewidth=1.8, label=label
        )
    for axis in axes:
        axis.axvline(threshold, color="black", linestyle="--", label="Locked cut")
        axis.axvline(low, color="#009E73", linestyle=":")
        axis.axvline(
            high, color="#009E73", linestyle=":", label="Category boundaries"
        )
        axis.set_yscale("log")
        axis.set_xlabel(r"Physical score $T$")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Area-normalized density")
    axes[1].set_ylabel("Expected events / bin")
    fig.suptitle("Out-of-fold physical score distributions")
    fig.tight_layout()
    fig.savefig(result_dir / "mva_score_distributions.png", dpi=180)
    plt.close(fig)

    probability_bins = np.linspace(0.0, 1.0, 81)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=True)
    for output_class, axis in enumerate(axes):
        for truth_class, (label, color) in enumerate(zip(LABELS, COLORS)):
            mask = classes == truth_class
            density = histogram_density(
                probabilities[mask, output_class],
                probability_bins,
                weights[mask],
            )
            axis.stairs(
                density, probability_bins, color=color, linewidth=1.7, label=label
            )
        axis.set_yscale("log")
        axis.set_xlabel(rf"Calibrated $p({CLASS_ORDER[output_class]})$")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Area-normalized density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Out-of-fold calibrated multiclass probabilities")
    fig.tight_layout()
    fig.savefig(result_dir / "multiclass_probability_distributions.png", dpi=180)
    plt.close(fig)


def plot_roc_tails(classes, weights, probabilities, summary, result_dir):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    diagnostics = summary["all_campaign_crossfit_evaluation"]["diagnostics"]
    for background_class, color in ((1, COLORS[1]), (2, COLORS[2])):
        signal_mask = classes == 0
        background_mask = classes == background_class
        pair_logit = np.log(np.clip(probabilities[:, 0], 1.0e-12, 1.0)) - np.log(
            np.clip(probabilities[:, background_class], 1.0e-12, 1.0)
        )
        bins = np.linspace(-30.0, 30.0, 3001)
        signal_hist, _ = np.histogram(
            pair_logit[signal_mask], bins, weights=weights[signal_mask]
        )
        background_hist, _ = np.histogram(
            pair_logit[background_mask], bins, weights=weights[background_mask]
        )
        signal_efficiency = np.cumsum(signal_hist[::-1])[::-1]
        background_efficiency = np.cumsum(background_hist[::-1])[::-1]
        signal_efficiency /= signal_efficiency[0]
        background_efficiency /= background_efficiency[0]
        name = CLASS_ORDER[background_class]
        auc = diagnostics[name]["auc"]
        ax.plot(
            signal_efficiency,
            background_efficiency,
            color=color,
            linewidth=1.8,
            label=f"{LABELS[background_class]} (AUC={auc:.4f})",
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0e-7, 1.0)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_title("Pairwise out-of-fold ROC tails")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_roc_tail_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_features(summary, result_dir):
    study = summary["feature_study"]
    impacts = study["permutation_impact"]
    ranked = sorted(
        impacts.items(), key=lambda item: item[1]["combined"], reverse=True
    )
    selected = set(summary["selected_features"])
    with open(
        result_dir / "mva_feature_list.csv", "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank", "feature", "used_in_locked_model", "single_cut_impact",
            "category_impact", "combined_impact",
        ])
        for rank, (name, impact) in enumerate(ranked, start=1):
            writer.writerow([
                rank, name, name in selected, impact["single_cut"],
                impact["categories"], impact["combined"],
            ])

    shown = ranked[:20][::-1]
    names = [item[0] for item in shown]
    single = [item[1]["single_cut"] for item in shown]
    categories = [item[1]["categories"] for item in shown]
    positions = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    ax.barh(
        positions - 0.19, single, height=0.38,
        color="#009E73", label="Single-cut significance impact",
    )
    ax.barh(
        positions + 0.19, categories, height=0.38,
        color="#CC79A7", label="Category significance impact",
    )
    ax.set_yticks(positions, names)
    ax.set_xlabel("Permutation decrease in validation significance")
    ax.set_title(
        f"Top feature impacts; locked model uses {len(selected)} variables"
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_feature_importance.png", dpi=180)
    fig.savefig(result_dir / "mva_feature_permutation_impact.png", dpi=180)
    plt.close(fig)

    nested = study["nested_results"]
    sizes = [item["size"] for item in nested]
    single = [
        item["metrics"]["single_cut"]["significance"] for item in nested
    ]
    categories = [
        item["metrics"]["categories"]["significance"] for item in nested
    ]
    baseline = study.get("baseline_all", study.get("baseline_43"))
    baseline_size = study.get("baseline_feature_count", 43)
    if baseline_size not in sizes:
        sizes.append(baseline_size)
        single.append(baseline["single_cut"]["significance"])
        categories.append(baseline["categories"]["significance"])
    order = np.argsort(sizes)
    sizes = np.asarray(sizes)[order]
    single = np.asarray(single)[order]
    categories = np.asarray(categories)[order]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(sizes, single, "o-", color="#009E73", label="Single cut")
    ax.plot(sizes, categories, "s-", color="#CC79A7", label="Three categories")
    ax.axhline(
        0.99 * baseline["single_cut"]["significance"],
        color="#009E73", linestyle=":", linewidth=1,
    )
    ax.axhline(
        0.99 * baseline["categories"]["significance"],
        color="#CC79A7", linestyle=":", linewidth=1,
    )
    ax.scatter(
        [baseline_size, baseline_size],
        [
            baseline["single_cut"]["significance"],
            baseline["categories"]["significance"],
        ],
        marker="*", s=150, color="black", zorder=5,
        label=f"Full {baseline_size}-feature baseline",
    )
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Optimization-split mass-binned significance")
    ax.set_title("Nested feature-set scan")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_feature_set_scan.png", dpi=180)
    plt.close(fig)


def plot_locked_features(summary, result_dir):
    from xgboost import Booster

    names = list(summary["selected_features"])
    gains = np.zeros(len(names), dtype=np.float64)
    models = sorted(result_dir.glob("crossfit_fold_*.json"))
    for path in models:
        booster = Booster()
        booster.load_model(path)
        importance = booster.get_score(importance_type="gain")
        gains += np.asarray([
            importance.get(f"f{index}", 0.0) for index in range(len(names))
        ])
    if models:
        gains /= len(models)
    order = np.argsort(gains)[::-1]
    with open(
        result_dir / "mva_feature_list.csv", "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature", "used_in_locked_model", "mean_crossfit_gain"])
        for rank, index in enumerate(order, start=1):
            writer.writerow([rank, names[index], True, gains[index]])
    shown = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    ax.barh(np.arange(shown.size), gains[shown], color="#0072B2")
    ax.set_yticks(np.arange(shown.size), [names[index] for index in shown])
    ax.set_xlabel("Mean XGBoost gain across cross-fit folds")
    ax.set_title(f"Locked projected feature set ({len(names)} variables)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_feature_importance.png", dpi=180)
    fig.savefig(result_dir / "mva_feature_permutation_impact.png", dpi=180)
    plt.close(fig)


def plot_locked_hyperparameters(summary, result_dir):
    params = summary["selected_hyperparameters"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.axis("off")
    text = "\n".join([
        "Locked model configuration",
        f"max_depth = {params['max_depth']}",
        f"learning_rate = {params['learning_rate']}",
        f"min_child_weight = {params['min_child_weight']}",
        "Features and hyperparameters reused; all models and thresholds retrained.",
    ])
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_hyperparameter_search.png", dpi=180)
    plt.close(fig)


def plot_hyperparameters(summary, result_dir):
    search = summary["hyperparameter_search"]
    points = []
    for pass_name in ("first_pass", "second_pass"):
        for result in search[pass_name]:
            params = result["params"]
            points.append({
                "pass": pass_name.replace("_", " "),
                "label": (
                    f"d{params['max_depth']}, lr={params['learning_rate']}, "
                    f"mcw={params['min_child_weight']}"
                ),
                "single": result["metrics"]["single_cut"]["significance"],
                "categories": result["metrics"]["categories"]["significance"],
                "params": params,
            })
    positions = np.arange(len(points))
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.scatter(
        positions - 0.12, [point["single"] for point in points],
        color="#009E73", marker="o", label="Single cut",
    )
    ax.scatter(
        positions + 0.12, [point["categories"] for point in points],
        color="#CC79A7", marker="s", label="Three categories",
    )
    ax.axvline(len(search["first_pass"]) - 0.5, color="gray", linestyle=":")
    selected = summary["selected_hyperparameters"]
    selected_indices = [
        index for index, point in enumerate(points)
        if point["params"] == selected
    ]
    for offset, metric in ((-0.12, "single"), (0.12, "categories")):
        ax.scatter(
            np.asarray(selected_indices) + offset,
            [points[index][metric] for index in selected_indices],
            s=100, facecolors="none", edgecolors="black", linewidths=1.5,
            label="Locked configuration" if metric == "single" else None,
        )
    ax.set_xticks(
        positions,
        [f"{point['pass']}:\n{point['label']}" for point in points],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Validation mass-binned significance")
    ax.set_title(
        "Successive hyperparameter search "
        f"(locked: depth {selected['max_depth']}, "
        f"learning rate {selected['learning_rate']}, "
        f"child weight {selected['min_child_weight']})"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_hyperparameter_search.png", dpi=180)
    plt.close(fig)


def plot_score_significance(
    classes, weights, score, summary, result_dir, madgraph_neff_marker
):
    scan = np.genfromtxt(
        result_dir / "single_cut_scan_crossfit.csv",
        delimiter=",", names=True, encoding="utf-8",
    )
    valid = (
        (scan["superchic_groups"] >= 50)
        & (scan["madgraph_groups"] >= 50)
        & (scan["superchic_neff"] >= 25)
        & (scan["madgraph_neff"] >= 25)
    )
    threshold = float(summary["locked_optimization"]["single_cut"]["threshold"])
    locked_index = int(np.argmin(np.abs(scan["threshold"] - threshold)))
    valid_score_indices = np.flatnonzero(valid)
    best_score_index = valid_score_indices[
        np.argmax(scan["global_window_significance"][valid])
    ]
    best_mass_index = valid_score_indices[
        np.argmax(scan["mass_binned_significance"][valid])
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot(
        scan["threshold"], scan["global_window_significance"],
        color="#0072B2", linewidth=1.8,
        label=r"Score-only cut: $S/\sqrt{S+B}$",
    )
    ax.plot(
        scan["threshold"], scan["mass_binned_significance"],
        color="#009E73", linewidth=1.8,
        label=r"Score cut + 1 GeV $m_{pp}$ bins",
    )
    ax.scatter(
        [threshold],
        [scan["global_window_significance"][locked_index]],
        marker="*", s=150, color="#CC79A7", zorder=5, label="Locked cut",
    )
    ax.scatter(
        [scan["threshold"][best_score_index]],
        [scan["global_window_significance"][best_score_index]],
        marker="x", s=65, color="black", zorder=5,
        label="Post-lock score-only maximum",
    )
    ax.fill_between(
        scan["threshold"], 0.0, 1.3,
        where=~valid, color="gray", alpha=0.12,
        label="Fails full-sample MC support",
    )
    ax.fill_between(
        scan["threshold"], 0.0, 1.3,
        where=scan["madgraph_neff"] < madgraph_neff_marker,
        color="#E69F00", alpha=0.08,
        label=rf"MadGraph $N_{{eff}}<{madgraph_neff_marker:g}$ (diagnostic)",
    )
    ax.set_ylim(0.0, max(1.25, 1.05 * np.max(scan["mass_binned_significance"])))
    ax.set_xlabel(r"Score threshold $T$")
    ax.set_ylabel("Significance")
    ax.set_title("Out-of-fold significance from the MVA score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "mva_score_significance.png", dpi=180)
    plt.close(fig)

    low, high = summary["locked_optimization"]["categories"]["boundaries"]
    category_selections = (
        score < low,
        (score >= low) & (score < high),
        score >= high,
    )
    category_z, category_regions = score_only_significance(
        classes, weights, category_selections
    )
    single_z, single_regions = score_only_significance(
        classes, weights, (score >= threshold,)
    )
    evaluation = summary["all_campaign_crossfit_evaluation"]
    report = {
        "definition": (
            "Score-only results use S/sqrt(S+B) without proton-mass bins. "
            "Exclusive score categories are combined in quadrature."
        ),
        "locked_single_cut": {
            "threshold": threshold,
            "score_only_significance": single_z,
            "mass_binned_significance": evaluation["single_cut"]["significance"],
            **single_regions[0],
        },
        "locked_categories": {
            "boundaries": [float(low), float(high)],
            "score_only_significance": category_z,
            "mass_binned_significance": evaluation["categories"]["significance"],
            "regions": {
                name: region for name, region in zip(
                    ("low", "middle", "high"), category_regions
                )
            },
        },
        "post_lock_crossfit_diagnostic": {
            "best_score_only_cut": {
                "threshold": float(scan["threshold"][best_score_index]),
                "significance": float(
                    scan["global_window_significance"][best_score_index]
                ),
            },
            "best_mass_binned_cut": {
                "threshold": float(scan["threshold"][best_mass_index]),
                "significance": float(
                    scan["mass_binned_significance"][best_mass_index]
                ),
            },
        },
    }
    with open(
        result_dir / "mva_score_significance.yaml", "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    return report


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    result_dir = Path(args.result_dir)
    summary_path = (
        Path(args.summary) if args.summary else result_dir / "summary_final.yaml"
    )
    with open(summary_path, encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)
    study_summary = summary
    if "feature_study" not in study_summary:
        with open(
            result_dir / "summary_develop.yaml", encoding="utf-8"
        ) as handle:
            study_summary = yaml.safe_load(handle)
    classes = np.load(cache_dir / "class.npy", mmap_mode="r")
    weights = np.load(cache_dir / "physical_weight.npy", mmap_mode="r")
    probabilities = np.load(
        result_dir / "crossfit_probabilities.npy", mmap_mode="r"
    )
    score = plugin_score(
        probabilities,
        np.asarray(summary["locked_optimization"]["kappas"], dtype=np.float64),
    )

    plot_scores(classes, weights, probabilities, score, summary, result_dir)
    plot_roc_tails(classes, weights, probabilities, summary, result_dir)
    if study_summary.get("feature_study", {}).get("skipped"):
        plot_locked_features(summary, result_dir)
        plot_locked_hyperparameters(summary, result_dir)
    else:
        plot_features(study_summary, result_dir)
        plot_hyperparameters(study_summary, result_dir)
    report = plot_score_significance(
        classes, weights, score, summary, result_dir,
        args.madgraph_neff_marker,
    )
    print(
        "Wrote MVA report plots. "
        f"Locked score-only Z={report['locked_single_cut']['score_only_significance']:.6f}; "
        "locked score+mass "
        f"Z={report['locked_single_cut']['mass_binned_significance']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
