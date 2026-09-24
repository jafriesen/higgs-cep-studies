#!/usr/bin/env python3
"""Figures for the H(cc) QED-background study."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.qed_study.common import DEFAULT_DATA, DEFAULT_OUTPUT  # noqa: E402

# Okabe-Ito; validated for CVD separation against the light surface.
COLOR = {
    "Hcc": "#0072B2",
    "QEDcc_superchic": "#D55E00",
    "QCDcc_superchic": "#E69F00",
    "QCDcc_madgraph": "#009E73",
}
LABEL = {
    "Hcc": "H(cc) signal",
    "QEDcc_superchic": r"QED $\gamma\gamma\to c\bar c$",
    "QCDcc_superchic": r"exclusive QCD $c\bar c$",
    "QCDcc_madgraph": r"non-exclusive QCD $c\bar c$",
}
SURFACE = "#FCFCFB"
INK = "#1A1A1A"
MUTED = "#6B6B6B"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#C8C8C4", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#E6E6E2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "exclusive_wide"))
    return parser.parse_args()


def read_yaml(path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def weighted_shape(values, weights, bins):
    histogram, edges = np.histogram(values, bins, weights=weights)
    width = np.diff(edges)
    total = (histogram * width).sum()
    return edges, histogram / total if total > 0 else histogram


def step_series(axis, edges, shape, color, label):
    axis.step(edges, np.r_[shape, shape[-1]], where="post", color=color, label=label)


def plot_stage1(output_dir, data_dir):
    """The production angle is the whole exclusive-continuum discriminant."""
    from analysis.MVA_hcc.qed_study.common import component_rows, load_dataset
    report = read_yaml(Path(output_dir) / "stage1" / "stage1_report.yaml")
    if report is None:
        return
    data = load_dataset(data_dir, mmap=False)
    index = data["features"].index("delta_eta_jj")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    bins = np.linspace(0.0, 4.0, 41)
    for name in ("Hcc", "QEDcc_superchic", "QCDcc_superchic"):
        rows = component_rows(data, name)
        edges, shape = weighted_shape(np.asarray(data["x"])[rows, index],
                                      np.asarray(data["physical_weight"])[rows], bins)
        step_series(axes[0], edges, shape, COLOR[name], LABEL[name])
    axes[0].set_xlabel(r"$|\Delta\eta_{jj}|$")
    axes[0].set_ylabel("normalised density")
    axes[0].set_title("Production angle separates the signal from both continua")
    axes[0].legend(loc="upper right")

    ranking = np.load(Path(output_dir) / "stage1" / "stage1_arrays.npz", allow_pickle=True)
    order = np.argsort(-ranking["separation_qed"])[:12]
    position = np.arange(order.size)
    axes[1].barh(position + 0.2, ranking["separation_qed"][order], height=0.38,
                 color=COLOR["QEDcc_superchic"], label="raw separation")
    axes[1].barh(position - 0.2, np.nan_to_num(ranking["conditional_qed"][order]), height=0.38,
                 color=COLOR["QCDcc_madgraph"], label=r"after conditioning on $|\Delta\eta_{jj}|$")
    axes[1].axvline(0.5, color=MUTED, linewidth=1.0, linestyle=":")
    axes[1].set_yticks(position)
    axes[1].set_yticklabels([str(name) for name in ranking["features"][order]], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.45, 0.95)
    axes[1].set_xlabel("weighted separation (folded AUC) versus QEDcc")
    axes[1].set_title("Nothing survives once the angle is known")
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "stage1" / "qed_variables.png", dpi=160)
    plt.close(figure)


def plot_stage2(output_dir):
    """Truth-level proton pT, and the resolution it would need."""
    stage = Path(output_dir) / "stage2"
    report = read_yaml(stage / "stage2_report.yaml")
    if report is None:
        return
    arrays = np.load(stage / "stage2_arrays.npz")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    bins = np.linspace(0.0, 1.5, 46)
    for name in ("Hcc", "QEDcc_superchic", "QCDcc_superchic"):
        pt = 0.5 * (arrays[f"{name}_pt_left"] + arrays[f"{name}_pt_right"])
        edges, shape = weighted_shape(pt, arrays[f"{name}_weight"], bins)
        step_series(axes[0], edges, shape, COLOR[name], LABEL[name])
    axes[0].set_xlabel(r"mean outgoing proton $p_T$  [GeV]")
    axes[0].set_ylabel("normalised density")
    axes[0].set_title("Photon exchange leaves a much softer proton")
    axes[0].legend(loc="upper right")

    bins = np.linspace(0.0, np.pi, 41)
    for name in ("Hcc", "QEDcc_superchic", "QCDcc_superchic"):
        edges, shape = weighted_shape(arrays[f"{name}_delta_phi_pp"],
                                      arrays[f"{name}_weight"], bins)
        step_series(axes[1], edges, shape, COLOR[name], LABEL[name])
    axes[1].set_xlabel(r"$|\Delta\phi_{pp}|$  [rad]")
    axes[1].set_ylabel("normalised density")
    axes[1].set_title("Proton azimuthal correlation")
    axes[1].legend(loc="upper left")

    scan = report["separation"]["QEDcc_superchic"]["pt_pair_resolution_scan"]
    sigma = np.asarray(sorted(scan), dtype=float)
    for label, key, column, color, style in (
        (r"$p_T$ versus QEDcc", "QEDcc_superchic", "pt_pair_resolution_scan",
         COLOR["QEDcc_superchic"], "-"),
        (r"$\Delta\phi_{pp}$ versus QEDcc", "QEDcc_superchic", "delta_phi_resolution_scan",
         COLOR["QEDcc_superchic"], "--"),
        (r"$p_T$ versus exclusive QCD", "QCDcc_superchic", "pt_pair_resolution_scan",
         COLOR["QCDcc_superchic"], "-"),
        (r"$\Delta\phi_{pp}$ versus exclusive QCD", "QCDcc_superchic",
         "delta_phi_resolution_scan", COLOR["QCDcc_superchic"], "--"),
    ):
        block = report["separation"][key].get(column)
        if block is None:
            continue
        values = [block[s] for s in sigma]
        axes[2].plot(sigma * 1000.0, values, marker="o", markersize=5, color=color,
                     linestyle=style, label=label)
    angle = report["separation"]["QEDcc_superchic"]["delta_eta_jj"]
    axes[2].axhline(angle, color=MUTED, linewidth=1.2, linestyle="--")
    axes[2].annotate(r"central-detector ceiling ($|\Delta\eta_{jj}|$)", (12, angle + 0.008),
                     fontsize=8, color=MUTED)
    axes[2].set_xlabel(r"proton $p_T$ resolution  [MeV]")
    axes[2].set_ylabel("weighted separation")
    axes[2].set_title("How good a t-measurement would have to be")
    axes[2].legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(stage / "proton_pt.png", dpi=160)
    plt.close(figure)


def plot_stage3(output_dir):
    """What the score projection costs, and what categorisation recovers."""
    stage = Path(output_dir) / "stage3"
    report = read_yaml(stage / "stage3_report.yaml")
    if report is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    structures = list(report["structures"])
    width = 0.8 / len(structures)
    labels = None
    for offset, structure in enumerate(structures):
        block = report["structures"][structure]
        entries = [("single cut", block["single_cut"]["significance"])]
        entries += [(item["label"].replace("ladder_plugin_", "ladder ")
                     .replace("grid_factorised_", "grid* ").replace("grid_", "grid "),
                     item["significance"]) for item in block["partitions"]]
        labels = [name for name, _ in entries]
        position = np.arange(len(entries)) + (offset - 0.5 * (len(structures) - 1)) * width
        axes[0].bar(position, [value for _, value in entries], width=width * 0.9,
                    color=list(COLOR.values())[offset], label=structure.replace("_", " "))
    reference = read_yaml(Path(output_dir) / "baseline_cc_only" / "report.yaml")
    if reference is not None:
        value = reference["ladder_significance"]
        # labelled through the legend rather than in place: every bar sits near it
        axes[0].axhline(value, color=MUTED, linewidth=1.2, linestyle="--",
                        label=f"train_model.py ladder ({value:.4f})")
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("mass-binned significance")
    axes[0].set_title("Categorisation, not the cut, is where the gain is\n"
                      "(grid* = pooled background factorised across the continuum axis)",
                      fontsize=10)
    axes[0].legend(loc="upper left")

    first = report["structures"][structures[0]]["qed_efficiency_at_signal_efficiency"]
    axis_labels = {"plugin": "single plug-in score", "continuum": "continuum axis",
                   "pooled": "non-exclusive axis"}
    colors = {"plugin": COLOR["Hcc"], "continuum": COLOR["QEDcc_superchic"],
              "pooled": COLOR["QCDcc_madgraph"]}
    for name, values in first.items():
        efficiency = np.asarray(sorted(values))
        axes[1].plot(efficiency, [values[e] for e in efficiency], marker="o", markersize=6,
                     color=colors[name], label=axis_labels[name])
    axes[1].set_xlabel("signal efficiency")
    axes[1].set_ylabel("QEDcc efficiency")
    axes[1].set_title("The plug-in score discards the QED separation the model learned")
    axes[1].legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(stage / "score_architecture.png", dpi=160)
    plt.close(figure)


def plot_stage5(output_dir):
    """Per-category significance against the MonteCarlo support behind it."""
    stage = Path(output_dir) / "stage5"
    report = read_yaml(stage / "stage5_report.yaml")
    if report is None:
        return
    per_category = np.asarray(report["per_category_significance"])
    effective = np.asarray(report["per_category_effective_pooled_events"])
    position = np.arange(per_category.size)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    axes[0].bar(position, per_category, color=COLOR["Hcc"], width=0.7)
    axes[0].set_xlabel("score category (equal signal fraction)")
    axes[0].set_ylabel("mass-binned significance")
    axes[0].set_title(f"Ladder total Z = {report['ladder_significance']:.4f}")
    axes[1].bar(position, effective, color=COLOR["QCDcc_madgraph"], width=0.7)
    axes[1].axhline(50.0, color=MUTED, linestyle="--", linewidth=1.2,
                    label="support floor, 50 effective events")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=15.0)
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_xlabel("score category (equal signal fraction)")
    axes[1].set_ylabel("effective MadGraph central events")
    axes[1].set_title("The most significant categories rest on the least MonteCarlo")
    for cell in position:
        axes[1].annotate(f"{effective[cell]:.0f}", (cell, effective[cell]),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(stage / "mc_support.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    plot_stage1(args.output_dir, args.data_dir)
    plot_stage2(args.output_dir)
    plot_stage3(args.output_dir)
    plot_stage5(args.output_dir)
    print(f"Wrote figures under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
