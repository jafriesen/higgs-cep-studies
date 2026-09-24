#!/usr/bin/env python3
"""Scan a common H(bb)/SuperChic survival factor on the locked MVA score."""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"
SIGNAL_COLOR = "#0072B2"
SUPERCHIC_COLOR = "#E69F00"
MADGRAPH_COLOR = "#D55E00"
SIGNIFICANCE_COLOR = "#009E73"
BASELINE_COLOR = "#CC79A7"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--baseline-survival", type=float, default=0.03)
    parser.add_argument("--min-survival", type=float, default=0.01)
    parser.add_argument("--max-survival", type=float, default=0.30)
    parser.add_argument(
        "--points",
        type=int,
        default=41,
        help="Number of logarithmically spaced scan points before adding baseline.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < args.min_survival <= args.baseline_survival <= args.max_survival:
        raise ValueError(
            "Require 0 < min-survival <= baseline-survival <= max-survival"
        )
    if args.points < 2:
        raise ValueError("--points must be at least 2")


def load_inputs(result_dir):
    with open(result_dir / "report.yaml", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    with np.load(result_dir / "report_data.npz", allow_pickle=False) as source:
        required = {
            "scan_thresholds",
            "scan_yields",
            "scan_signal_mass",
            "scan_superchic_mass",
            "madgraph_mass_shape",
            "scan_madgraph_effective",
            "support_floor",
            "category_mass",
        }
        missing = required.difference(source.files)
        if missing:
            raise RuntimeError(
                "report_data.npz lacks survival-scan inputs; rerun "
                f"train_feature_count_20.py. Missing: {sorted(missing)}"
            )
        arrays = {name: np.asarray(source[name]) for name in required}
    return report, arrays


def categorized_significance(category_mass, scale):
    signal = scale * category_mass[:, 0]
    background = scale * category_mass[:, 1] + category_mass[:, 2]
    total = signal + background
    return float(
        np.sqrt(
            np.sum(
                np.divide(
                    signal * signal,
                    total,
                    out=np.zeros_like(signal),
                    where=total > 0.0,
                )
            )
        )
    )


def scan_survival(survival, baseline, arrays):
    thresholds = arrays["scan_thresholds"]
    scan_yields = arrays["scan_yields"]
    effective = arrays["scan_madgraph_effective"]
    support_floor = float(arrays["support_floor"])
    signal_mass = arrays["scan_signal_mass"]
    superchic_mass = arrays["scan_superchic_mass"]
    madgraph_shape = arrays["madgraph_mass_shape"]
    valid = np.flatnonzero(effective >= support_floor)
    if valid.size == 0:
        raise RuntimeError("No score threshold satisfies the MadGraph support floor")

    rows = []
    for value in survival:
        scale = value / baseline
        signal = scale * signal_mass
        superchic = scale * superchic_mass
        madgraph = scan_yields[2, :, None] * madgraph_shape[None, :]
        total = signal + superchic + madgraph
        significance = np.sqrt(
            np.sum(
                np.divide(
                    signal * signal,
                    total,
                    out=np.zeros_like(signal),
                    where=total > 0.0,
                ),
                axis=1,
            )
        )
        best = valid[int(np.argmax(significance[valid]))]
        signal_yield = float(signal[best].sum())
        superchic_yield = float(superchic[best].sum())
        madgraph_yield = float(scan_yields[2, best])
        background = superchic_yield + madgraph_yield
        rows.append(
            {
                "survival_factor": float(value),
                "survival_percent": float(100.0 * value),
                "normalization_scale": float(scale),
                "threshold": float(thresholds[best]),
                "mass_binned_significance": float(significance[best]),
                "single_cut_mass_binned_significance": float(significance[best]),
                "category_mass_binned_significance": categorized_significance(
                    arrays["category_mass"], scale
                ),
                "signal_yield": signal_yield,
                "superchic_yield": superchic_yield,
                "madgraph_yield": madgraph_yield,
                "signal_over_background": float(signal_yield / background),
                "counting_significance": float(
                    signal_yield / np.sqrt(signal_yield + background)
                ),
                "madgraph_effective_central_events": float(effective[best]),
            }
        )
    return rows


def write_results(rows, args, report, result_dir):
    csv_path = result_dir / "survival_factor_scan.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline_index = int(
        np.argmin([abs(row["survival_factor"] - args.baseline_survival) for row in rows])
    )
    summary = {
        "interpretation": (
            "The classifier and physical plug-in score are held fixed. For each "
            "survival factor, H(bb) and SuperChic QCD yields are scaled together, "
            "and MadGraph is unchanged. The single-cut result selects the best "
            "existing threshold passing the MadGraph effective-event support floor. "
            "The categorized result combines the six locked score-category, "
            "mass-binned significances without changing their boundaries."
        ),
        "baseline_survival_factor": args.baseline_survival,
        "minimum_survival_factor": args.min_survival,
        "maximum_survival_factor": args.max_survival,
        "scan_points": len(rows),
        "baseline_result": rows[baseline_index],
        "reference_operating_point": report["single_cut_operating_point"],
        "results": rows,
    }
    with open(
        result_dir / "survival_factor_scan.yaml", "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def plot_results(rows, args, result_dir):
    survival_percent = np.asarray([row["survival_percent"] for row in rows])
    single_cut_significance = np.asarray(
        [row["single_cut_mass_binned_significance"] for row in rows]
    )
    category_significance = np.asarray(
        [row["category_mass_binned_significance"] for row in rows]
    )
    threshold = np.asarray([row["threshold"] for row in rows])
    yields = np.asarray(
        [
            [row["signal_yield"], row["superchic_yield"], row["madgraph_yield"]]
            for row in rows
        ]
    )

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharex=True)
    axes[0].plot(
        survival_percent,
        single_cut_significance,
        color=SIGNIFICANCE_COLOR,
        linewidth=1.8,
        label="Optimized single cut",
    )
    axes[0].plot(
        survival_percent,
        category_significance,
        color="#0072B2",
        linewidth=1.8,
        linestyle="--",
        label="Six locked categories",
    )
    axes[0].set_ylabel("Mass-binned significance")

    axes[1].plot(survival_percent, threshold, color="#0072B2", linewidth=1.8)
    axes[1].set_ylabel(r"Optimal locked-score threshold $T$")

    for index, (label, color) in enumerate(
        (
            (r"$H\rightarrow b\bar{b}$", SIGNAL_COLOR),
            ("SuperChic QCD", SUPERCHIC_COLOR),
            ("MadGraph QCD", MADGRAPH_COLOR),
        )
    ):
        axes[2].plot(
            survival_percent,
            yields[:, index],
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Expected events at optimal threshold")
    axes[2].legend(fontsize=8)

    baseline_percent = 100.0 * args.baseline_survival
    for axis in axes:
        axis.axvline(
            baseline_percent,
            color=BASELINE_COLOR,
            linestyle=":",
            label=f"Baseline ({baseline_percent:g}%)",
        )
        axis.set_xscale("log")
        axis.set_xlabel("Survival factor")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Locked-score optimization versus common survival factor")
    figure.tight_layout()
    figure.savefig(result_dir / "survival_factor_scan.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    result_dir = Path(args.result_dir).resolve()
    report, arrays = load_inputs(result_dir)
    survival = np.unique(
        np.r_[
            np.geomspace(args.min_survival, args.max_survival, args.points),
            args.baseline_survival,
        ]
    )
    rows = scan_survival(survival, args.baseline_survival, arrays)
    write_results(rows, args, report, result_dir)
    plot_results(rows, args, result_dir)

    baseline = min(
        rows, key=lambda row: abs(row["survival_factor"] - args.baseline_survival)
    )
    print(
        f"Scanned {len(rows)} survival factors from "
        f"{100.0 * args.min_survival:g}% to {100.0 * args.max_survival:g}%; "
        f"baseline {100.0 * args.baseline_survival:g}% gives "
        f"T={baseline['threshold']:.3f}, "
        f"Z(single cut)={baseline['single_cut_mass_binned_significance']:.4f}, "
        f"Z(categories)={baseline['category_mass_binned_significance']:.4f}",
        flush=True,
    )
    print(f"Wrote survival-factor scan to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
