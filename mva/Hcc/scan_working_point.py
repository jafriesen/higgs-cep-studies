#!/usr/bin/env python3
"""Compare H(cc) MVA setups at a different c-tag working point.

The trained score is held fixed and only the per-component tag factors change,
so thresholds, the ladder and the joint two-dimensional scan are re-optimised
under the new normalisation. This is the same locked-score approximation used
by scan_normalizations.py, extended to vary the gluon mistag as well.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

SETUPS = (
    "binary",
    "exclusive_specialist",
    "nonexclusive_specialist",
    "hierarchical",
    "multiclass",
    "merged_exclusive",
)
JOINT = "specialist_joint"


def significance(component_mass):
    """sqrt(sum_bins s^2/(s+b)), matching mva/common/scans.py."""
    signal = component_mass[..., 0, :]
    background = component_mass[..., 1:, :].sum(axis=-2)
    total = signal + background
    return np.sqrt(
        np.sum(
            np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0),
            axis=-1,
        )
    )


def flavor_scales(flavors, tagging, new_working_point):
    """Per-component yield scale: (new tag probability / old) ** 2."""
    old = {
        "cc": float(tagging["eff_c"]),
        "bb": float(tagging["mistag_b_to_c"]),
        "light": float(tagging["mistag_light_to_c"]),
    }
    return np.array(
        [(new_working_point[flavor] / old[flavor]) ** 2 for flavor in flavors],
        dtype=np.float64,
    )


def evaluate_setup(result_dir, new_working_point):
    report = yaml.safe_load((result_dir / "report.yaml").read_text())
    with np.load(result_dir / "report_data.npz", allow_pickle=False) as source:
        scan_mass = source["scan_component_mass"]
        thresholds = source["scan_thresholds"]
        effective = source["scan_madgraph_effective"]
        support_floor = float(source["support_floor"])
        category_mass = source["category_mass"]

    components = report["components"]
    flavors = [item["source_flavor"] for item in components]
    if new_working_point is None:
        scales = np.ones(len(components), dtype=np.float64)
    else:
        scales = flavor_scales(flavors, report["tagging"], new_working_point)

    scaled = scan_mass * scales[np.newaxis, :, np.newaxis]
    scan_z = significance(scaled)
    valid = np.flatnonzero(effective >= support_floor)
    if valid.size == 0:
        valid = np.arange(thresholds.size)
    best = valid[np.argmax(scan_z[valid])]
    yields = scaled[best].sum(axis=1)
    category_z = significance(category_mass * scales[np.newaxis, :, np.newaxis])

    return {
        "single_cut_significance": float(scan_z[best]),
        "ladder_significance": float(np.sqrt(np.sum(category_z**2))),
        "threshold": float(thresholds[best]),
        "madgraph_effective_central_events": float(effective[best]),
        "signal_yield": float(yields[0]),
        "background_yield": float(yields[1:].sum()),
        "component_yields": dict(zip((item["name"] for item in components), yields.tolist())),
        "reported_single_cut_significance": float(
            report["single_cut_operating_point"]["significance"]
        ),
        "reported_ladder_significance": float(report["ladder_significance"]),
    }


def evaluate_joint(result_dir, tagging, new_working_point):
    """The joint specialist stores a full two-dimensional threshold grid."""
    report = yaml.safe_load((result_dir / "report.yaml").read_text())
    best = report["best_two_dimensional"]
    with np.load(result_dir / "report_data.npz", allow_pickle=False) as source:
        cumulative = source["joint_cumulative_component_mass"]
        edges = source["score_edges"]

    names = list(best["component_yields"])
    flavors = {"Hcc": "cc", "QCDcc": "cc", "QEDcc": "cc", "QCDbb": "bb", "QCDgg": "light"}
    order = [flavors[name.split("_")[0]] for name in names]
    if new_working_point is None:
        scales = np.ones(len(names), dtype=np.float64)
    else:
        scales = flavor_scales(order, tagging, new_working_point)

    scaled = cumulative * scales[:, np.newaxis, np.newaxis, np.newaxis]
    grid = significance(np.moveaxis(scaled, 0, -2))
    row, column = np.unravel_index(np.argmax(grid), grid.shape)
    yields = scaled[:, row, column, :].sum(axis=1)

    return {
        "single_cut_significance": float(grid[row, column]),
        "ladder_significance": None,
        "nonexclusive_threshold": float(edges[row]),
        "exclusive_threshold": float(edges[column]),
        "signal_yield": float(yields[0]),
        "background_yield": float(yields[1:].sum()),
        "component_yields": dict(zip(names, yields.tolist())),
        "reported_single_cut_significance": float(best["significance"]),
        "reported_ladder_significance": None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--suffix", default="_7p1ps_allrows_gg_v04_g256_s12345")
    parser.add_argument("--eff-c", type=float, default=0.45)
    parser.add_argument("--mistag-b-to-c", type=float, default=0.025)
    parser.add_argument("--mistag-light-to-c", type=float, default=0.01)
    parser.add_argument("--output", default=None, help="Write the comparison to this CSV")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-evaluate at the stored working point and print the residual",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("eff_c", "mistag_b_to_c", "mistag_light_to_c"):
        if not 0.0 < getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in (0, 1]")
    results_dir = Path(args.results_dir).resolve()
    working_point = {
        "cc": args.eff_c,
        "bb": args.mistag_b_to_c,
        "light": args.mistag_light_to_c,
    }
    requested = None if args.check else working_point

    baseline_report = yaml.safe_load(
        (results_dir / f"fsr_mtd_{SETUPS[0]}{args.suffix}" / "report.yaml").read_text()
    )
    tagging = baseline_report["tagging"]

    rows = []
    for name in SETUPS:
        result = evaluate_setup(results_dir / f"fsr_mtd_{name}{args.suffix}", requested)
        result["setup"] = name
        rows.append(result)
    joint_dir = results_dir / f"fsr_mtd_{JOINT}{args.suffix}"
    if joint_dir.is_dir():
        result = evaluate_joint(joint_dir, tagging, requested)
        result["setup"] = JOINT
        rows.append(result)

    header = f"{'setup':26s} {'ladder':>10s} {'single cut':>11s} {'S':>9s} {'B':>12s}"
    print(header)
    for row in rows:
        ladder = row["ladder_significance"]
        ladder_text = "n/a" if ladder is None else f"{ladder:.6f}"
        line = (
            f"{row['setup']:26s} {ladder_text:>10s} "
            f"{row['single_cut_significance']:11.6f} "
            f"{row['signal_yield']:9.4g} {row['background_yield']:12.6g}"
        )
        if args.check:
            delta = (
                row["single_cut_significance"] - row["reported_single_cut_significance"]
            )
            line += f"   residual {delta:+.2e}"
        print(line)

    if args.output:
        fields = ["setup", "ladder_significance", "single_cut_significance",
                  "signal_yield", "background_yield"]
        with open(args.output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
        print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
