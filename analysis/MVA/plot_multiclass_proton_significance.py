#!/usr/bin/env python3
"""Plot locked multiclass-proton score and mass-significance diagnostics."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
LABELS = {
    "Hbb": r"$H\rightarrow b\bar{b}$",
    "QCDbb": "SuperChic QCD",
    "QCDbb_madgraph": "MadGraph QCD",
}
COLORS = {
    "Hbb": "#0072B2",
    "QCDbb": "#E69F00",
    "QCDbb_madgraph": "#D55E00",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--summary",
        default=None,
        help="Develop summary containing locked kappas and boundaries.",
    )
    return parser.parse_args()


def effective_size(weights):
    denominator = np.sum(weights * weights)
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0 else 0.0


def group_effective_size(groups, weights):
    if groups.size == 0:
        return 0.0
    _unique, inverse = np.unique(groups, return_inverse=True)
    group_weight = np.bincount(inverse, weights=weights)
    return effective_size(group_weight)


def significance(signal, background):
    denominator = signal + background
    return float(np.sqrt(np.sum(np.divide(
        signal * signal,
        denominator,
        out=np.zeros_like(signal),
        where=denominator > 0.0,
    ))))


def region_statistics(classes, groups, weights, selection):
    result = {}
    for class_id, name in enumerate(CLASS_ORDER):
        mask = selection & (classes == class_id)
        result[name] = {
            "candidate_rows": int(np.sum(mask)),
            "unique_hard_events": int(np.unique(groups[mask]).size),
            "expected_yield": float(np.sum(weights[mask])),
            "group_neff": group_effective_size(groups[mask], weights[mask]),
        }
    return result


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    result_dir = Path(args.result_dir)
    summary_path = (
        Path(args.summary) if args.summary else result_dir / "summary_develop.yaml"
    )
    with open(summary_path, encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    classes = np.load(cache_dir / "class.npy", mmap_mode="r")
    groups = np.load(cache_dir / "group_id.npy", mmap_mode="r")
    mass = np.load(cache_dir / "mx.npy", mmap_mode="r")
    weights = np.load(cache_dir / "physical_weight.npy", mmap_mode="r")
    probabilities = np.load(
        result_dir / "crossfit_probabilities.npy", mmap_mode="r"
    )
    kappas = np.asarray(summary["locked_optimization"]["kappas"])
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    score = np.log(clipped[:, 0]) - np.log(
        kappas[0] * clipped[:, 1] + kappas[1] * clipped[:, 2]
    )
    threshold = summary["locked_optimization"]["single_cut"]["threshold"]
    low, high = summary["locked_optimization"]["categories"]["boundaries"]
    finite_mass = np.isfinite(mass)
    regions = {
        "preselection": finite_mass,
        "single_cut": finite_mass & (score >= threshold),
        "category_low": finite_mass & (score < low),
        "category_middle": finite_mass & (score >= low) & (score < high),
        "category_high": finite_mass & (score >= high),
    }
    statistics = {
        name: region_statistics(classes, groups, weights, selection)
        for name, selection in regions.items()
    }

    mass_bins = np.arange(117.0, 134.0, 1.0)
    category_names = ("category_low", "category_middle", "category_high")
    for region_name in ("single_cut",) + category_names:
        selection = regions[region_name]
        signal, _ = np.histogram(
            mass[selection & (classes == 0)], mass_bins,
            weights=weights[selection & (classes == 0)],
        )
        backgrounds = []
        for class_id in (1, 2):
            hist, _ = np.histogram(
                mass[selection & (classes == class_id)], mass_bins,
                weights=weights[selection & (classes == class_id)],
            )
            backgrounds.append(hist)
        total_background = backgrounds[0] + backgrounds[1]
        statistics[region_name]["mass_binned_significance"] = significance(
            signal, total_background
        )
        signal_yield = float(np.sum(signal))
        background_yield = float(np.sum(total_background))
        statistics[region_name]["global_window_significance"] = (
            float(signal_yield / np.sqrt(signal_yield + background_yield))
            if signal_yield + background_yield > 0.0 else 0.0
        )

    with open(result_dir / "postcut_event_counts.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "score_definition": {
                    "kappa_superchic": float(kappas[0]),
                    "kappa_madgraph": float(kappas[1]),
                    "single_threshold": float(threshold),
                    "category_boundaries": [float(low), float(high)],
                },
                "regions": statistics,
            },
            handle,
            sort_keys=False,
        )
    with open(
        result_dir / "postcut_event_counts.csv", "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "region", "process", "candidate_rows", "unique_hard_events",
                "expected_yield", "group_neff",
            ),
        )
        writer.writeheader()
        for region_name, region in statistics.items():
            for process in CLASS_ORDER:
                if process not in region:
                    continue
                writer.writerow({
                    "region": region_name,
                    "process": process,
                    **region[process],
                })

    score_bins = np.linspace(-30.0, 0.0, 81)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharex=True)
    for class_id, name in enumerate(CLASS_ORDER):
        mask = classes == class_id
        axes[0].hist(
            score[mask], score_bins, weights=weights[mask], density=True,
            histtype="step", linewidth=1.8, color=COLORS[name], label=LABELS[name],
        )
        axes[1].hist(
            score[mask], score_bins, weights=weights[mask],
            histtype="step", linewidth=1.8, color=COLORS[name], label=LABELS[name],
        )
    for ax in axes:
        ax.axvline(threshold, color="black", linestyle="--", label="Single cut")
        ax.axvline(low, color="#009E73", linestyle=":")
        ax.axvline(high, color="#009E73", linestyle=":", label="Category boundaries")
        ax.set_yscale("log")
        ax.set_xlabel("Physical plug-in score T")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Area-normalized density")
    axes[1].set_ylabel("Expected events / score bin")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "locked_score_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharex=True)
    for ax, region_name, title in zip(
        axes,
        category_names,
        ("Low-score category", "Middle-score category", "High-score category"),
    ):
        selection = regions[region_name]
        for class_id, name in enumerate(CLASS_ORDER):
            hist, _ = np.histogram(
                mass[selection & (classes == class_id)], mass_bins,
                weights=weights[selection & (classes == class_id)],
            )
            ax.stairs(
                hist, mass_bins, linewidth=1.8, color=COLORS[name],
                label=f"{LABELS[name]} ({np.sum(hist):.3g})",
            )
        ax.set_yscale("log")
        ax.set_title(
            f"{title}\n"
            f"Z={statistics[region_name]['mass_binned_significance']:.3f}"
        )
        ax.set_xlabel(r"$m_{pp}$ [GeV]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Expected events / 1 GeV")
    fig.tight_layout()
    fig.savefig(result_dir / "locked_category_mass_templates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    selection = regions["single_cut"]
    for class_id, name in enumerate(CLASS_ORDER):
        hist, _ = np.histogram(
            mass[selection & (classes == class_id)], mass_bins,
            weights=weights[selection & (classes == class_id)],
        )
        ax.stairs(
            hist, mass_bins, linewidth=1.8, color=COLORS[name],
            label=f"{LABELS[name]} ({np.sum(hist):.3g})",
        )
    ax.set_yscale("log")
    ax.set_title(
        f"Locked single cut T >= {threshold:.3f}\n"
        f"Z={statistics['single_cut']['mass_binned_significance']:.3f}"
    )
    ax.set_xlabel(r"$m_{pp}$ [GeV]")
    ax.set_ylabel("Expected events / 1 GeV")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "locked_single_cut_mass_template.png", dpi=180)
    plt.close(fig)

    contribution = np.zeros((3, mass_bins.size - 1))
    for category_index, region_name in enumerate(category_names):
        selection = regions[region_name]
        signal, _ = np.histogram(
            mass[selection & (classes == 0)], mass_bins,
            weights=weights[selection & (classes == 0)],
        )
        background, _ = np.histogram(
            mass[selection & (classes != 0)], mass_bins,
            weights=weights[selection & (classes != 0)],
        )
        contribution[category_index] = np.divide(
            signal,
            np.sqrt(signal + background),
            out=np.zeros_like(signal),
            where=signal + background > 0.0,
        )
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    image = ax.imshow(
        contribution,
        aspect="auto",
        origin="lower",
        extent=[mass_bins[0], mass_bins[-1], 0, 3],
        cmap="viridis",
    )
    ax.set_yticks([0.5, 1.5, 2.5], ["Low", "Middle", "High"])
    ax.set_xlabel(r"$m_{pp}$ [GeV]")
    ax.set_ylabel("Score category")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"Per-bin $S/\sqrt{S+B}$")
    fig.tight_layout()
    fig.savefig(result_dir / "significance_contribution_map.png", dpi=180)
    plt.close(fig)

    single_scan_path = result_dir / "single_cut_scan_optimization.csv"
    category_scan_path = result_dir / "category_scan_optimization.csv"
    if not single_scan_path.exists() or not category_scan_path.exists():
        print(
            "Optimization scan CSV files are not available; wrote the locked "
            "event-count and mass-template plots only.",
            flush=True,
        )
        return

    single_scan = np.genfromtxt(
        single_scan_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    finite = np.isfinite(single_scan["threshold"])
    scan = single_scan[finite]
    order = np.argsort(scan["threshold"])
    scan = scan[order]
    valid = scan["valid"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharex=True)
    axes[0].plot(
        scan["threshold"][valid], scan["significance"][valid],
        color="#009E73", linewidth=1.8, marker=".", label="Valid scan points",
    )
    axes[0].scatter(
        scan["threshold"][~valid], scan["significance"][~valid],
        color="gray", marker="x", s=18, label="Fails MC requirement",
    )
    axes[0].axvline(threshold, color="black", linestyle="--")
    axes[0].scatter(
        [threshold],
        [summary["locked_optimization"]["single_cut"]["significance"]],
        color="#CC79A7", marker="*", s=120, zorder=5, label="Locked point",
    )
    axes[0].set_ylabel(r"Mass-binned $S/\sqrt{S+B}$")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        scan["threshold"], scan["signal_yield"],
        color=COLORS["Hbb"], label=LABELS["Hbb"],
    )
    axes[1].plot(
        scan["threshold"], scan["superchic_yield"],
        color=COLORS["QCDbb"], label=LABELS["QCDbb"],
    )
    axes[1].plot(
        scan["threshold"], scan["madgraph_yield"],
        color=COLORS["QCDbb_madgraph"], label=LABELS["QCDbb_madgraph"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Expected yield")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        scan["threshold"], scan["superchic_groups"],
        color=COLORS["QCDbb"], linestyle="-", label="SC groups",
    )
    axes[2].plot(
        scan["threshold"], scan["madgraph_groups"],
        color=COLORS["QCDbb_madgraph"], linestyle="-", label="MG groups",
    )
    axes[2].plot(
        scan["threshold"], scan["superchic_neff"],
        color=COLORS["QCDbb"], linestyle=":", label=r"SC $N_\mathrm{eff}$",
    )
    axes[2].plot(
        scan["threshold"], scan["madgraph_neff"],
        color=COLORS["QCDbb_madgraph"], linestyle=":", label=r"MG $N_\mathrm{eff}$",
    )
    axes[2].axhline(150, color="black", linestyle="--", linewidth=1, label="150 groups")
    axes[2].axhline(75, color="black", linestyle=":", linewidth=1, label=r"$N_\mathrm{eff}=75$")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Optimization MC support")
    axes[2].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Threshold T")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "single_cut_scan_optimization.png", dpi=180)
    plt.close(fig)

    category_scan = np.genfromtxt(
        category_scan_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    valid = category_scan["valid"]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    scatter = ax.scatter(
        category_scan["low_boundary"][valid],
        category_scan["high_boundary"][valid],
        c=category_scan["significance"][valid],
        cmap="viridis",
        s=45,
        label="Valid",
    )
    ax.scatter(
        category_scan["low_boundary"][~valid],
        category_scan["high_boundary"][~valid],
        color="gray",
        marker="x",
        s=25,
        label="Fails MC requirement",
    )
    ax.scatter(
        [low], [high], color="#CC79A7", marker="*", s=180,
        edgecolor="black", linewidth=0.5, label="Locked boundaries",
    )
    ax.set_xlabel("Low/middle boundary")
    ax.set_ylabel("Middle/high boundary")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(r"Mass-binned $S/\sqrt{S+B}$")
    fig.tight_layout()
    fig.savefig(result_dir / "category_boundary_scan_optimization.png", dpi=180)
    plt.close(fig)

    print(f"Wrote event counts and diagnostic plots to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
