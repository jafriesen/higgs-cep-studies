#!/usr/bin/env python3
"""Scan a single score cut using saved group-safe cross-fit predictions."""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
COLORS = ("#0072B2", "#E69F00", "#D55E00")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--threshold-min", type=float, default=-15.0)
    parser.add_argument("--threshold-max", type=float, default=-2.0)
    parser.add_argument("--points", type=int, default=261)
    parser.add_argument("--madgraph-neff-marker", type=float, default=200.0)
    return parser.parse_args()


def cumulative_group_support(score, groups, weights, thresholds):
    """Return group counts and group-weight Neff above every threshold."""
    threshold_bin = np.searchsorted(thresholds, score, side="right") - 1
    valid = threshold_bin >= 0
    threshold_bin = threshold_bin[valid]
    groups = groups[valid]
    weights = weights[valid]
    order = np.argsort(threshold_bin, kind="stable")
    threshold_bin = threshold_bin[order]
    groups = groups[order]
    weights = weights[order]
    offsets = np.searchsorted(
        threshold_bin, np.arange(thresholds.size + 1), side="left"
    )

    group_weight = np.zeros(int(np.max(groups)) + 1, dtype=np.float64)
    counts = np.zeros(thresholds.size, dtype=np.int64)
    neff = np.zeros(thresholds.size, dtype=np.float64)
    total_weight = 0.0
    squared_weight = 0.0
    group_count = 0
    for threshold_index in range(thresholds.size - 1, -1, -1):
        start = offsets[threshold_index]
        stop = offsets[threshold_index + 1]
        if start != stop:
            bin_groups, inverse = np.unique(
                groups[start:stop], return_inverse=True
            )
            increments = np.bincount(
                inverse, weights=weights[start:stop]
            )
            old_weight = group_weight[bin_groups]
            new_weight = old_weight + increments
            group_count += int(np.count_nonzero(old_weight == 0.0))
            total_weight += float(np.sum(increments))
            squared_weight += float(np.sum(
                new_weight * new_weight - old_weight * old_weight
            ))
            group_weight[bin_groups] = new_weight
        counts[threshold_index] = group_count
        if squared_weight > 0.0:
            neff[threshold_index] = total_weight * total_weight / squared_weight
    return counts, neff


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    result_dir = Path(args.result_dir)
    with open(result_dir / "summary_develop.yaml", encoding="utf-8") as handle:
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
    locked_threshold = float(
        summary["locked_optimization"]["single_cut"]["threshold"]
    )
    thresholds = np.unique(np.r_[
        np.linspace(args.threshold_min, args.threshold_max, args.points),
        locked_threshold,
    ])
    score_edges = np.r_[thresholds, np.inf]
    mass_bins = np.arange(117.0, 134.0, 1.0)

    cumulative_mass = []
    cumulative_yield = []
    cumulative_rows = []
    cumulative_groups = []
    cumulative_neff = []
    for class_id in range(3):
        mask = classes == class_id
        score_mass_hist, _, _ = np.histogram2d(
            score[mask],
            mass[mask],
            bins=(score_edges, mass_bins),
            weights=weights[mask],
        )
        cumulative_mass.append(np.cumsum(score_mass_hist[::-1], axis=0)[::-1])
        score_yield, _ = np.histogram(
            score[mask], score_edges, weights=weights[mask]
        )
        cumulative_yield.append(np.cumsum(score_yield[::-1])[::-1])
        score_rows, _ = np.histogram(score[mask], score_edges)
        cumulative_rows.append(np.cumsum(score_rows[::-1])[::-1])

        group_counts, group_neff = cumulative_group_support(
            score[mask], groups[mask], weights[mask], thresholds
        )
        cumulative_groups.append(group_counts)
        cumulative_neff.append(group_neff)

    cumulative_mass = np.asarray(cumulative_mass)
    cumulative_yield = np.asarray(cumulative_yield)
    cumulative_rows = np.asarray(cumulative_rows)
    cumulative_groups = np.asarray(cumulative_groups)
    cumulative_neff = np.asarray(cumulative_neff)
    signal = cumulative_mass[0]
    background = cumulative_mass[1] + cumulative_mass[2]
    denominator = signal + background
    mass_significance = np.sqrt(np.sum(np.divide(
        signal * signal,
        denominator,
        out=np.zeros_like(signal),
        where=denominator > 0.0,
    ), axis=1))
    total_signal = cumulative_yield[0]
    total_background = cumulative_yield[1] + cumulative_yield[2]
    global_significance = np.divide(
        total_signal,
        np.sqrt(total_signal + total_background),
        out=np.zeros_like(total_signal),
        where=total_signal + total_background > 0.0,
    )

    output_csv = result_dir / "single_cut_scan_crossfit.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "threshold", "mass_binned_significance",
            "global_window_significance", "signal_yield",
            "superchic_yield", "madgraph_yield", "signal_rows",
            "superchic_rows", "madgraph_rows", "signal_groups",
            "superchic_groups", "madgraph_groups", "signal_neff",
            "superchic_neff", "madgraph_neff",
        ])
        for index, threshold in enumerate(thresholds):
            writer.writerow([
                threshold,
                mass_significance[index],
                global_significance[index],
                cumulative_yield[0, index],
                cumulative_yield[1, index],
                cumulative_yield[2, index],
                cumulative_rows[0, index],
                cumulative_rows[1, index],
                cumulative_rows[2, index],
                cumulative_groups[0, index],
                cumulative_groups[1, index],
                cumulative_groups[2, index],
                cumulative_neff[0, index],
                cumulative_neff[1, index],
                cumulative_neff[2, index],
            ])

    locked_index = int(np.flatnonzero(thresholds == locked_threshold)[0])
    best_index = int(np.argmax(mass_significance))
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.8), sharex=True)
    axes[0].plot(
        thresholds, mass_significance, color="#009E73",
        linewidth=2.0, label="1 GeV mass bins",
    )
    axes[0].plot(
        thresholds, global_significance, color="#56B4E9",
        linewidth=1.5, linestyle="--", label="Global window",
    )
    axes[0].scatter(
        [locked_threshold], [mass_significance[locked_index]],
        color="#CC79A7", marker="*", s=140, zorder=5,
        label=f"Locked T={locked_threshold:.3f}",
    )
    axes[0].scatter(
        [thresholds[best_index]], [mass_significance[best_index]],
        color="black", marker="x", s=55, zorder=5,
        label="Cross-fit diagnostic maximum",
    )
    axes[0].set_ylabel("Significance")
    axes[0].legend(fontsize=8)

    for class_id, name in enumerate(CLASS_ORDER):
        axes[1].plot(
            thresholds, cumulative_yield[class_id],
            color=COLORS[class_id], label=name,
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Expected yield")
    axes[1].legend(fontsize=8)

    for class_id, name in enumerate(CLASS_ORDER):
        axes[2].plot(
            thresholds, cumulative_groups[class_id],
            color=COLORS[class_id], label=name,
        )
    axes[2].axhline(150, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(50, color="black", linestyle=":", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Unique hard-event groups")
    axes[2].legend(fontsize=8)
    for class_id, name in enumerate(CLASS_ORDER):
        axes[3].plot(
            thresholds, cumulative_neff[class_id],
            color=COLORS[class_id], label=name,
        )
    axes[3].axhline(
        args.madgraph_neff_marker,
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label=rf"MadGraph diagnostic $N_{{eff}}={args.madgraph_neff_marker:g}$",
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"Group-level $N_{eff}$")
    axes[3].legend(fontsize=7)
    for ax in axes:
        ax.axvline(locked_threshold, color="#CC79A7", linestyle=":", linewidth=1)
        ax.set_xlabel("Threshold T")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Saved three-fold cross-fit diagnostic scan "
        "(not used to retune the locked threshold)"
    )
    fig.tight_layout()
    fig.savefig(result_dir / "single_cut_scan_crossfit.png", dpi=180)
    plt.close(fig)

    print(
        f"Locked: T={locked_threshold:.6g}, Z={mass_significance[locked_index]:.6g}; "
        f"diagnostic maximum: T={thresholds[best_index]:.6g}, "
        f"Z={mass_significance[best_index]:.6g}",
        flush=True,
    )
    print(f"Wrote {output_csv}", flush=True)


if __name__ == "__main__":
    main()
