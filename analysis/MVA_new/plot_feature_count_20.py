#!/usr/bin/env python3
"""Plot the standalone locked-feature report from its compact result files."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"
LABELS = (
    r"$H\rightarrow b\bar{b}$",
    "SuperChic QCD",
    "MadGraph QCD",
)
COLORS = ("#0072B2", "#E69F00", "#D55E00")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    return parser.parse_args()


def load_results(result_dir):
    with open(result_dir / "report.yaml", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    arrays = np.load(result_dir / "report_data.npz", allow_pickle=False)
    required = {
        "mass_bins",
        "score_axis",
        "score_yields",
        "scan_thresholds",
        "scan_significance",
        "scan_yields",
        "scan_madgraph_effective",
        "support_floor",
        "operating_index",
        "preselection_mass",
        "selected_mass",
        "category_mass",
        "category_low",
        "category_high",
        "category_significance",
    }
    missing = required.difference(arrays.files)
    if missing:
        raise RuntimeError(f"report_data.npz is missing: {sorted(missing)}")
    return report, arrays


def plot_scores(report, arrays, result_dir):
    score = arrays["score_axis"]
    yields = arrays["score_yields"]
    threshold = report["single_cut_operating_point"]["threshold"]
    bin_width = float(np.median(np.diff(score)))

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    for class_id, (label, color) in enumerate(zip(LABELS, COLORS)):
        total = yields[class_id].sum()
        density = (
            yields[class_id] / (total * bin_width)
            if total > 0.0
            else np.zeros_like(yields[class_id])
        )
        axes[0].plot(score, density, color=color, linewidth=1.8, label=label)
        axes[1].plot(
            score, yields[class_id], color=color, linewidth=1.8, label=label
        )
    for axis in axes:
        axis.axvline(threshold, color="black", linestyle="--", label="Selected cut")
        axis.set_yscale("log")
        axis.set_xlabel(r"Physical plug-in score $T$")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Area-normalized density")
    axes[1].set_ylabel("Expected events / score bin")
    figure.suptitle(
        f"Out-of-fold score distributions ({report['n_features']} features)"
    )
    figure.tight_layout()
    figure.savefig(result_dir / "score_distributions.png", dpi=180)
    plt.close(figure)


def plot_scan(report, arrays, result_dir):
    threshold = arrays["scan_thresholds"]
    significance = arrays["scan_significance"]
    yields = arrays["scan_yields"]
    effective = arrays["scan_madgraph_effective"]
    support_floor = float(arrays["support_floor"])
    operating = report["single_cut_operating_point"]

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharex=True)
    axes[0].plot(threshold, significance, color="#009E73", linewidth=1.8)
    axes[0].scatter(
        [operating["threshold"]],
        [operating["significance"]],
        marker="*",
        s=120,
        color="#CC79A7",
        zorder=5,
        label=(
            f"Selected $T={operating['threshold']:.3f}$, "
            f"$Z={operating['significance']:.3f}$"
        ),
    )
    axes[0].set_ylabel("Mass-binned significance")
    axes[0].legend(fontsize=8)

    for values, label, color in zip(yields, LABELS, COLORS):
        axes[1].plot(threshold, values, color=color, linewidth=1.8, label=label)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Expected events")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        threshold, effective, color=COLORS[2], linewidth=1.8,
        label="MadGraph effective central events",
    )
    axes[2].axhline(
        support_floor,
        color="black",
        linestyle=":",
        label=f"Support floor ({support_floor:.0f})",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"MadGraph $N_{\mathrm{eff}}$")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.axvline(operating["threshold"], color="#CC79A7", linestyle=":")
        axis.set_xlabel(r"Threshold $T$")
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(
        f"Two-fold cross-fit single-cut scan ({report['n_features']} features)"
    )
    figure.tight_layout()
    figure.savefig(result_dir / "single_cut_scan.png", dpi=180)
    plt.close(figure)


def draw_mass_lines(axis, histograms, mass_bins):
    for values, label, color in zip(histograms, LABELS, COLORS):
        axis.stairs(
            values,
            mass_bins,
            linewidth=1.8,
            color=color,
            label=f"{label} ({values.sum():.3g})",
        )
    axis.set_yscale("log")
    axis.set_xlabel(r"$m_{pp}$ [GeV]")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def plot_mass_spectra(report, arrays, result_dir):
    mass_bins = arrays["mass_bins"]
    operating = report["single_cut_operating_point"]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    draw_mass_lines(axes[0], arrays["preselection_mass"], mass_bins)
    draw_mass_lines(axes[1], arrays["selected_mass"], mass_bins)
    axes[0].set_title("Preselection")
    axes[1].set_title(
        rf"Selected $T\geq {operating['threshold']:.3f}$; "
        f"$Z={operating['significance']:.3f}$"
    )
    axes[0].set_ylabel("Expected events / 1 GeV")
    axes[1].set_ylabel("Expected events / 1 GeV")
    figure.suptitle("Proton mass spectra before and after the score selection")
    figure.tight_layout()
    figure.savefig(result_dir / "mass_spectra.png", dpi=180)
    plt.close(figure)


def plot_single_cut_mass_spectrum(report, arrays, result_dir):
    operating = report["single_cut_operating_point"]
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    draw_mass_lines(axis, arrays["selected_mass"], arrays["mass_bins"])
    axis.set_title(
        rf"Selected $T\geq {operating['threshold']:.3f}$; "
        f"$Z={operating['significance']:.3f}$"
    )
    axis.set_ylabel("Expected events / 1 GeV")
    figure.tight_layout()
    figure.savefig(result_dir / "single_cut_mass_spectrum.png", dpi=180)
    plt.close(figure)


def plot_category_masses(report, arrays, result_dir):
    mass_bins = arrays["mass_bins"]
    category_mass = arrays["category_mass"]
    lows = arrays["category_low"]
    highs = arrays["category_high"]
    significance = arrays["category_significance"]
    if category_mass.shape[0] != 6:
        raise RuntimeError("The plotting layout expects six score categories")

    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), sharex=True)
    for index, axis in enumerate(axes.flat):
        draw_mass_lines(axis, category_mass[index], mass_bins)
        axis.set_title(
            rf"Category {index}: {lows[index]:.2f} $\leq T <$ {highs[index]:.2f}" "\n"
            f"$Z={significance[index]:.3f}$"
        )
        axis.set_ylabel("Expected events / 1 GeV")
    figure.suptitle(
        f"Mass spectra by score category; combined "
        f"$Z={report['ladder_significance']:.3f}$"
    )
    figure.tight_layout()
    figure.savefig(result_dir / "mass_spectra_by_category.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    report, arrays = load_results(result_dir)
    plot_scores(report, arrays, result_dir)
    plot_scan(report, arrays, result_dir)
    plot_mass_spectra(report, arrays, result_dir)
    plot_single_cut_mass_spectrum(report, arrays, result_dir)
    plot_category_masses(report, arrays, result_dir)
    arrays.close()
    print(f"Wrote five report plots to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
