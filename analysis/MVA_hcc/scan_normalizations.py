#!/usr/bin/env python3
"""Run locked-score survival and jet-tag normalization sensitivity scans."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from analysis.MVA_hcc.common import normalization_scales, write_yaml  # noqa: E402


DEFAULT_RESULTS = SCRIPT_DIR / "results"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--baseline-survival", type=float, default=0.03)
    parser.add_argument("--min-survival", type=float, default=0.01)
    parser.add_argument("--max-survival", type=float, default=0.30)
    parser.add_argument("--survival-points", type=int, default=41)
    parser.add_argument("--tag-points", type=int, default=31)
    parser.add_argument("--tag-min-scale", type=float, default=0.5)
    parser.add_argument("--tag-max-scale", type=float, default=1.5)
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < args.min_survival <= args.baseline_survival <= args.max_survival:
        raise ValueError("Require 0 < min survival <= baseline <= max survival")
    if args.survival_points < 2 or args.tag_points < 2:
        raise ValueError("Each scan needs at least two points")
    if not 0.0 <= args.tag_min_scale <= 1.0 <= args.tag_max_scale:
        raise ValueError("Tag scale range must contain the nominal scale 1")


def load_inputs(result_dir):
    with open(result_dir / "report.yaml", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    with np.load(result_dir / "report_data.npz", allow_pickle=False) as source:
        required = {
            "scan_thresholds", "scan_component_mass", "scan_madgraph_effective",
            "support_floor", "category_mass",
        }
        missing = required.difference(source.files)
        if missing:
            raise RuntimeError(f"report_data.npz lacks scan inputs: {sorted(missing)}")
        arrays = {name: np.asarray(source[name]) for name in required}
    return report, arrays


def significance(component_mass):
    signal = component_mass[:, 0]
    background = component_mass[:, 1:].sum(axis=1)
    total = signal + background
    return np.sqrt(
        np.sum(
            np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0),
            axis=1,
        )
    )


def categorized_significance(component_mass):
    return float(np.sqrt(np.sum(significance(component_mass) ** 2)))


def scan_values(kind, values, nominal, report, arrays, include_categories=False):
    valid = np.flatnonzero(arrays["scan_madgraph_effective"] >= float(arrays["support_floor"]))
    if valid.size == 0:
        raise RuntimeError("No score threshold satisfies the MadGraph support floor")
    rows = []
    base_mass = arrays["scan_component_mass"]
    for value in values:
        scales = normalization_scales(report["components"], kind, value, nominal)
        varied = base_mass * scales[None, :, None]
        scan_z = significance(varied)
        best = valid[int(np.argmax(scan_z[valid]))]
        component_yields = varied[best].sum(axis=1)
        signal_yield = float(component_yields[0])
        background_yield = float(component_yields[1:].sum())
        rows.append(
            {
                "value": float(value),
                "relative_to_nominal": float(value / nominal),
                "threshold": float(arrays["scan_thresholds"][best]),
                "mass_binned_significance": float(scan_z[best]),
                "single_cut_mass_binned_significance": float(scan_z[best]),
                "signal_yield": signal_yield,
                "background_yield": background_yield,
                "signal_over_background": float(signal_yield / background_yield),
                "counting_significance": float(
                    signal_yield / np.sqrt(signal_yield + background_yield)
                ),
                "madgraph_effective_central_events": float(
                    arrays["scan_madgraph_effective"][best]
                ),
            }
        )
        if include_categories:
            category_mass = arrays["category_mass"] * scales[None, :, None]
            rows[-1]["category_mass_binned_significance"] = categorized_significance(
                category_mass
            )
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scans(scans, result_dir):
    figure, axes = plt.subplots(1, len(scans), figsize=(5.7 * len(scans), 5.2))
    axes = np.atleast_1d(axes)
    labels = {
        "survival": "Survival factor",
        "eff_c": "Per-jet charm efficiency",
        "mistag_b_to_c": "Per-jet b-to-c mistag probability",
    }
    colors = {"survival": "#009E73", "eff_c": "#0072B2", "mistag_b_to_c": "#D55E00"}
    for axis, (kind, scan) in zip(axes, scans.items()):
        rows = scan["rows"]
        values = np.asarray([row["value"] for row in rows])
        z = np.asarray([row["mass_binned_significance"] for row in rows])
        threshold = np.asarray([row["threshold"] for row in rows])
        axis.plot(
            values,
            z,
            color=colors[kind],
            linewidth=1.8,
            label="Optimized single cut",
        )
        if kind == "survival":
            category_z = np.asarray(
                [row["category_mass_binned_significance"] for row in rows]
            )
            axis.plot(
                values,
                category_z,
                color="#0072B2",
                linewidth=1.8,
                linestyle="--",
                label="Six locked categories",
            )
        #twin = axis.twinx()
        #twin.plot(values, threshold, color="#666666", linestyle="--", label="Threshold")
        axis.axvline(scan["nominal"], color="black", linestyle=":")
        axis.set_xlabel(labels[kind])
        axis.set_ylabel("Mass-binned significance")
        #twin.set_ylabel("Optimal threshold T")
        axis.grid(True, alpha=0.25)
        lines, line_labels = axis.get_legend_handles_labels()
        #twin_lines, twin_labels = twin.get_legend_handles_labels()
        #axis.legend(lines + twin_lines, line_labels + twin_labels, fontsize=7)
        axis.legend(lines, line_labels, fontsize=7)
    figure.suptitle("Locked-score H(cc) normalization sensitivity scans")
    figure.tight_layout()
    figure.savefig(result_dir / "normalization_scans.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    result_dir = Path(args.result_dir).resolve()
    report, arrays = load_inputs(result_dir)
    eff_c = float(report["tagging"]["eff_c"])
    mistag = float(report["tagging"]["mistag_b_to_c"])
    survival_values = np.unique(
        np.r_[
            np.geomspace(args.min_survival, args.max_survival, args.survival_points),
            args.baseline_survival,
        ]
    )
    eff_values = np.unique(
        np.r_[
            np.linspace(
                max(0.0, args.tag_min_scale * eff_c),
                min(1.0, args.tag_max_scale * eff_c),
                args.tag_points,
            ),
            eff_c,
        ]
    )
    mistag_values = np.unique(
        np.r_[
            np.linspace(
                max(0.0, args.tag_min_scale * mistag),
                min(1.0, args.tag_max_scale * mistag),
                args.tag_points,
            ),
            mistag,
        ]
    )
    flavors = {component["source_flavor"] for component in report["components"]}
    scan_specs = [("survival", survival_values, args.baseline_survival)]
    if "cc" in flavors:
        scan_specs.append(("eff_c", eff_values, eff_c))
    if "bb" in flavors:
        scan_specs.append(("mistag_b_to_c", mistag_values, mistag))
    scans = {}
    for kind, values, nominal in scan_specs:
        rows = scan_values(
            kind,
            values,
            nominal,
            report,
            arrays,
            include_categories=(kind == "survival"),
        )
        scans[kind] = {"nominal": nominal, "rows": rows}
        write_csv(result_dir / f"{kind}_scan.csv", rows)
    write_yaml(
        result_dir / "normalization_scans.yaml",
        {
            "interpretation": (
                "Sensitivity scans, not assigned uncertainties. Classifier probabilities and "
                "the nominal plug-in score are held fixed; the supported optimal threshold is "
                "reselected after component yields are rescaled. The survival scan also "
                "combines the six locked score-category, mass-binned significances without "
                "changing their boundaries."
            ),
            "tag_scale_range": [args.tag_min_scale, args.tag_max_scale],
            "scans": scans,
        },
    )
    plot_scans(scans, result_dir)
    print(f"Wrote survival and tag-efficiency scans to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
