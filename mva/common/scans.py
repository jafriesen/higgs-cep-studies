"""Locked-score normalization scans for compact MVA results."""

import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mva.common.dataset import write_yaml
from mva.common.weights import normalization_scales


def _significance(component_mass):
    signal = component_mass[:, 0]
    background = component_mass[:, 1:].sum(axis=1)
    total = signal + background
    return np.sqrt(
        np.sum(
            np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0),
            axis=1,
        )
    )


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_normalization_scans(result_dir, args):
    result_dir = Path(result_dir).resolve()
    with open(result_dir / "report.yaml", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    with np.load(result_dir / "report_data.npz", allow_pickle=False) as source:
        scan_mass = np.asarray(source["scan_component_mass"])
        thresholds = np.asarray(source["scan_thresholds"])
        effective = np.asarray(source["scan_madgraph_effective"])
        support_floor = float(source["support_floor"])
        category_mass = np.asarray(source["category_mass"])
    valid = np.flatnonzero(effective >= support_floor)
    if valid.size == 0:
        valid = np.arange(thresholds.size)
    tagging = report.get("tagging") or {}
    specifications = [
        (
            "survival",
            np.unique(np.r_[np.geomspace(args.min_survival, args.max_survival, args.survival_points), args.baseline_survival]),
            args.baseline_survival,
        )
    ]
    flavors = {item["source_flavor"] for item in report["components"]}
    if report["channel"] == "Hcc" and "cc" in flavors:
        nominal = float(tagging.get("eff_c", args.eff_c))
        specifications.append(
            ("eff_c", np.unique(np.r_[np.linspace(max(0.0, nominal * args.tag_min_scale), min(1.0, nominal * args.tag_max_scale), args.tag_points), nominal]), nominal)
        )
    if report["channel"] == "Hcc" and "bb" in flavors:
        nominal = float(tagging.get("mistag_b_to_c", args.mistag_b_to_c))
        specifications.append(
            ("mistag_b_to_c", np.unique(np.r_[np.linspace(max(0.0, nominal * args.tag_min_scale), min(1.0, nominal * args.tag_max_scale), args.tag_points), nominal]), nominal)
        )
    if report["channel"] == "Hbb" and "bb" in flavors:
        nominal = float(tagging.get("eff_b", args.eff_b))
        specifications.append(
            ("eff_b", np.unique(np.r_[np.linspace(max(0.0, nominal * args.tag_min_scale), min(1.0, nominal * args.tag_max_scale), args.tag_points), nominal]), nominal)
        )
    if report["channel"] == "Hbb" and "cc" in flavors:
        nominal = float(tagging.get("mistag_c_to_b", args.mistag_c_to_b))
        specifications.append(
            ("mistag_c_to_b", np.unique(np.r_[np.linspace(max(0.0, nominal * args.tag_min_scale), min(1.0, nominal * args.tag_max_scale), args.tag_points), nominal]), nominal)
        )

    scans = {}
    for kind, values, nominal in specifications:
        rows = []
        for value in values:
            scales = normalization_scales(report["components"], kind, value, nominal)
            varied = scan_mass * scales[np.newaxis, :, np.newaxis]
            significance = _significance(varied)
            best = valid[np.argmax(significance[valid])]
            component_yields = varied[best].sum(axis=1)
            signal = float(component_yields[0])
            background = float(component_yields[1:].sum())
            category_varied = category_mass * scales[np.newaxis, :, np.newaxis]
            category_z = _significance(category_varied)
            rows.append(
                {
                    "value": float(value),
                    "relative_to_nominal": float(value / nominal),
                    "threshold": float(thresholds[best]),
                    "significance": float(significance[best]),
                    "category_significance": float(np.sqrt(np.sum(category_z**2))),
                    "signal_yield": signal,
                    "background_yield": background,
                    "signal_over_background": float(signal / background) if background > 0.0 else None,
                    "madgraph_effective_central_events": float(effective[best]),
                }
            )
        scans[kind] = {"nominal": nominal, "rows": rows}
        _write_csv(result_dir / f"{kind}_scan.csv", rows)
        print(f"  {kind}: evaluated {len(rows)} locked-score points", flush=True)

    write_yaml(
        result_dir / "normalization_scans.yaml",
        {
            "interpretation": (
                "The calibrated classifier and score bins are held fixed. Component yields "
                "are rescaled and the best supported stored threshold is reselected."
            ),
            "scans": scans,
        },
    )
    figure, axes = plt.subplots(1, len(scans), figsize=(5.7 * len(scans), 5.0), squeeze=False)
    for axis, (kind, scan) in zip(axes.flat, scans.items()):
        values = np.asarray([row["value"] for row in scan["rows"]])
        significance = np.asarray([row["significance"] for row in scan["rows"]])
        categories = np.asarray([row["category_significance"] for row in scan["rows"]])
        axis.plot(values, significance, label="Optimized single cut")
        axis.plot(values, categories, linestyle="--", label="Locked categories")
        axis.axvline(scan["nominal"], color="black", linestyle=":")
        axis.set_xlabel(kind)
        axis.set_ylabel("Mass-binned significance")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(result_dir / "normalization_scans.png", dpi=180)
    plt.close(figure)
    print(f"Wrote normalization scans to {result_dir}", flush=True)
    return scans
