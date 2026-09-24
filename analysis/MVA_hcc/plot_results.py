#!/usr/bin/env python3
"""Plot compact results from an H(cc) MVA dataset."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    return parser.parse_args()


def load_results(result_dir):
    with open(result_dir / "report.yaml", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    arrays = np.load(result_dir / "report_data.npz", allow_pickle=False)
    required = {
        "mass_bins", "score_axis", "score_yields", "probability_bins",
        "probability_histograms", "scan_thresholds", "scan_significance",
        "scan_component_yields", "scan_madgraph_effective", "support_floor",
        "operating_index", "preselection_mass", "selected_mass", "category_mass",
        "category_low", "category_high", "category_significance",
    }
    missing = required.difference(arrays.files)
    if missing:
        raise RuntimeError(f"report_data.npz is missing: {sorted(missing)}")
    return report, arrays


def labels_and_colors(report):
    labels = [component["name"].replace("_", " ") for component in report["components"]]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(labels)))
    return labels, colors


def plot_probabilities(report, arrays, result_dir):
    classes = report["classes"]
    edges = arrays["probability_bins"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    histograms = arrays["probability_histograms"]
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), sharex=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(classes)))
    for predicted, axis in enumerate(axes.flat):
        if predicted >= len(classes):
            axis.axis("off")
            continue
        for truth, (label, color) in enumerate(zip(classes, colors)):
            values = histograms[truth, predicted]
            total = values.sum()
            density = values / (total * np.diff(edges)) if total > 0.0 else values
            axis.plot(centers, density, color=color, label=label)
        axis.set_yscale("log")
        axis.set_title(f"p({classes[predicted]})")
        axis.set_xlabel("Calibrated probability")
        axis.set_ylabel("Area-normalized density")
        axis.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    figure.suptitle(f"{len(classes)}-class out-of-fold calibrated probabilities")
    figure.tight_layout()
    figure.savefig(result_dir / "class_probabilities.png", dpi=180)
    plt.close(figure)


def plot_scores(report, arrays, result_dir):
    labels, colors = labels_and_colors(report)
    score = arrays["score_axis"]
    yields = arrays["score_yields"]
    threshold = report["single_cut_operating_point"]["threshold"]
    width = float(np.median(np.diff(score)))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    for values, label, color in zip(yields, labels, colors):
        total = values.sum()
        density = values / (total * width) if total > 0.0 else values
        axes[0].plot(score, density, color=color, linewidth=1.6, label=label)
        axes[1].plot(score, values, color=color, linewidth=1.6, label=label)
    for axis in axes:
        axis.axvline(threshold, color="black", linestyle="--", label="Selected cut")
        axis.set_yscale("log")
        axis.set_xlabel("Physical plug-in score T")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Area-normalized density")
    axes[1].set_ylabel("Expected events / score bin")
    axes[1].legend(fontsize=7)
    figure.suptitle(
        f"{len(labels)} physical components in the {len(report['classes'])}-class H(cc) MVA"
    )
    figure.tight_layout()
    figure.savefig(result_dir / "score_distributions.png", dpi=180)
    plt.close(figure)


def plot_threshold_scan(report, arrays, result_dir):
    labels, colors = labels_and_colors(report)
    threshold = arrays["scan_thresholds"]
    significance = arrays["scan_significance"]
    yields = arrays["scan_component_yields"]
    effective = arrays["scan_madgraph_effective"]
    floor = float(arrays["support_floor"])
    operating = report["single_cut_operating_point"]
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), sharex=True)
    axes[0].plot(threshold, significance, color="#009E73", linewidth=1.8)
    axes[0].scatter(
        [operating["threshold"]], [operating["significance"]], marker="*", s=120,
        color="#CC79A7", zorder=5,
    )
    axes[0].set_ylabel("Mass-binned significance")
    for values, label, color in zip(yields, labels, colors):
        axes[1].plot(threshold, values, color=color, linewidth=1.4, label=label)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Expected events above threshold")
    axes[1].legend(fontsize=6)
    axes[2].plot(threshold, effective, color="#D55E00", linewidth=1.8)
    axes[2].axhline(floor, color="black", linestyle=":", label=f"Support floor ({floor:g})")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Combined MadGraph effective events")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.axvline(operating["threshold"], color="#CC79A7", linestyle=":")
        axis.set_xlabel("Threshold T")
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle("Locked-score threshold optimization")
    figure.tight_layout()
    figure.savefig(result_dir / "single_cut_scan.png", dpi=180)
    plt.close(figure)


def draw_component_mass(axis, histograms, mass_bins, labels, colors):
    for values, label, color in zip(histograms, labels, colors):
        axis.stairs(
            values, mass_bins, color=color, linewidth=1.5,
            label=f"{label} ({values.sum():.3g})",
        )
    axis.set_yscale("log")
    axis.set_xlabel("m_pp [GeV]")
    axis.set_ylabel("Expected events / 1 GeV")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=6)


def plot_mass_spectra(report, arrays, result_dir):
    labels, colors = labels_and_colors(report)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), sharex=True)
    draw_component_mass(axes[0], arrays["preselection_mass"], arrays["mass_bins"], labels, colors)
    draw_component_mass(axes[1], arrays["selected_mass"], arrays["mass_bins"], labels, colors)
    axes[0].set_title("Preselection")
    operating = report["single_cut_operating_point"]
    axes[1].set_title(
        f"T >= {operating['threshold']:.3f}; Z={operating['significance']:.3f}"
    )
    figure.suptitle("Proton mass spectra by physical component")
    figure.tight_layout()
    figure.savefig(result_dir / "mass_spectra.png", dpi=180)
    plt.close(figure)


def plot_categories(report, arrays, result_dir):
    labels, colors = labels_and_colors(report)
    category_mass = arrays["category_mass"]
    if category_mass.shape[0] != 6:
        raise RuntimeError("The category plot expects six score categories")
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 9.5), sharex=True)
    score_axis = arrays["score_axis"]
    for index, axis in enumerate(axes.flat):
        draw_component_mass(axis, category_mass[index], arrays["mass_bins"], labels, colors)
        low = arrays["category_low"][index]
        high = arrays["category_high"][index]
        low = low if np.isfinite(low) else score_axis[0]
        high = high if np.isfinite(high) else score_axis[-1]
        axis.set_title(
            f"Category {index}: {low:.2f} <= T < {high:.2f}\n"
            f"Z={arrays['category_significance'][index]:.3f}"
        )
    figure.suptitle(
        f"Mass spectra by score category; combined Z={report['ladder_significance']:.3f}"
    )
    figure.tight_layout()
    figure.savefig(result_dir / "mass_spectra_by_category.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    report, arrays = load_results(result_dir)
    plot_probabilities(report, arrays, result_dir)
    plot_scores(report, arrays, result_dir)
    plot_threshold_scan(report, arrays, result_dir)
    plot_mass_spectra(report, arrays, result_dir)
    plot_categories(report, arrays, result_dir)
    arrays.close()
    print(f"Wrote five H(cc) report plots to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
