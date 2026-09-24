"""Compact diagnostic plots for unified MVA result artifacts."""

from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(result_dir):
    result_dir = Path(result_dir).resolve()
    with open(result_dir / "report.yaml", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    arrays = np.load(result_dir / "report_data.npz", allow_pickle=False)
    return result_dir, report, arrays


def _component_style(report):
    labels = [item["name"].replace("_", " ") for item in report["components"]]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(labels)))
    return labels, colors


def _draw_mass(axis, values, edges, labels, colors):
    for histogram, label, color in zip(values, labels, colors):
        axis.stairs(
            histogram,
            edges,
            color=color,
            linewidth=1.4,
            label=f"{label} ({histogram.sum():.3g})",
        )
    axis.set_yscale("log")
    axis.set_xlabel("m_pp [GeV]")
    axis.set_ylabel("Expected events / GeV")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=6)


def plot_results(result_dir):
    result_dir, report, arrays = _load(result_dir)
    labels, colors = _component_style(report)
    score = arrays["score_axis"]
    threshold = report["single_cut_operating_point"]["threshold"]

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    for values, label, color in zip(arrays["score_yields"], labels, colors):
        density = values / values.sum() if values.sum() > 0.0 else values
        axes[0].plot(score, density, color=color, label=label)
        axes[1].plot(score, values, color=color, label=label)
    for axis in axes:
        axis.axvline(threshold, color="black", linestyle="--")
        axis.set_yscale("log")
        axis.set_xlabel("Classifier + vertex log likelihood ratio")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Area-normalized yield")
    axes[1].set_ylabel("Expected events / score bin")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(result_dir / "score_distributions.png", dpi=180)
    plt.close(figure)
    print("  wrote score_distributions.png", flush=True)

    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), sharex=True)
    axes[0].plot(arrays["scan_thresholds"], arrays["scan_significance"], color="#009E73")
    for values, label, color in zip(arrays["scan_component_yields"].T, labels, colors):
        axes[1].plot(arrays["scan_thresholds"], values, color=color, label=label)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=6)
    axes[2].plot(arrays["scan_thresholds"], arrays["scan_madgraph_effective"], color="#D55E00")
    axes[2].axhline(float(arrays["support_floor"]), color="black", linestyle=":")
    axes[2].set_yscale("log")
    for axis, ylabel in zip(axes, ("Mass-binned Z", "Expected yield", "MadGraph n_eff")):
        axis.axvline(threshold, color="#CC79A7", linestyle=":")
        axis.set_xlabel("Score threshold")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(result_dir / "single_cut_scan.png", dpi=180)
    plt.close(figure)
    print("  wrote single_cut_scan.png", flush=True)

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), sharex=True)
    _draw_mass(axes[0], arrays["preselection_mass"], arrays["mass_bins"], labels, colors)
    _draw_mass(axes[1], arrays["selected_mass"], arrays["mass_bins"], labels, colors)
    axes[0].set_title("Preselection")
    axes[1].set_title(f"Selected, Z={report['single_cut_operating_point']['significance']:.3f}")
    figure.tight_layout()
    figure.savefig(result_dir / "mass_spectra.png", dpi=180)
    plt.close(figure)
    print("  wrote mass_spectra.png", flush=True)

    classes = report["classes"]
    count = len(classes)
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.3 * rows), squeeze=False)
    centers = 0.5 * (arrays["probability_bins"][:-1] + arrays["probability_bins"][1:])
    class_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, count))
    for predicted, axis in enumerate(axes.flat):
        if predicted >= count:
            axis.axis("off")
            continue
        for truth, (label, color) in enumerate(zip(classes, class_colors)):
            values = arrays["probability_histograms"][truth, predicted]
            density = values / values.sum() if values.sum() > 0.0 else values
            axis.plot(centers, density, color=color, label=label)
        axis.set_yscale("log")
        axis.set_title(f"p({classes[predicted]})")
        axis.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(result_dir / "class_probabilities.png", dpi=180)
    plt.close(figure)
    print("  wrote class_probabilities.png", flush=True)

    category = arrays["category_mass"]
    columns = min(3, category.shape[0])
    rows = int(np.ceil(category.shape[0] / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.7 * columns, 4.8 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= category.shape[0]:
            axis.axis("off")
            continue
        _draw_mass(axis, category[index], arrays["mass_bins"], labels, colors)
        axis.set_title(
            f"Category {index}: Z={arrays['category_significance'][index]:.3f}"
        )
    figure.tight_layout()
    figure.savefig(result_dir / "mass_spectra_by_category.png", dpi=180)
    plt.close(figure)
    arrays.close()
    print(f"Wrote five compact result plots to {result_dir}", flush=True)
