#!/usr/bin/env python3
"""Stage 3: how the class probabilities should be turned into a selection.

The five-class model already learns to separate H(cc) from the exclusive
continuum, but the plug-in score it is cut on is dominated by the non-exclusive
term (kappa ~ 2e6), so that separation is discarded by the one-dimensional
projection. This stage compares class structures, score definitions and
categorisations on the charm-only component set, reusing one cached set of
calibrated probabilities per class structure.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.common import plugin_score  # noqa: E402
from analysis.MVA_hcc.qed_study.common import DEFAULT_DATA, DEFAULT_OUTPUT, write_yaml  # noqa: E402
from analysis.MVA_hcc.qed_study import harness  # noqa: E402

# component order in the charm-only dataset
COMPONENTS = ("Hcc", "QCDcc_superchic", "QEDcc_superchic", "QCDcc_madgraph")
STRUCTURES = {
    "four_class": {
        "class_map": (0, 1, 2, 3),
        "class_names": ("Hcc", "exclusive_QCD", "exclusive_QED", "nonexclusive_QCD"),
        "continuum_classes": (1, 2),
        "pooled_classes": (3,),
    },
    "three_class": {
        "class_map": (0, 1, 1, 2),
        "class_names": ("Hcc", "exclusive_continuum", "nonexclusive_QCD"),
        "continuum_classes": (1,),
        "pooled_classes": (2,),
    },
}
N_SCAN = 96


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "cc_locked"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage3"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--grid-cells", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--train-cap", type=int, default=250000)
    parser.add_argument("--stop-cap", type=int, default=0,
                        help="Cap on pooled early-stopping groups; 0 keeps every row, "
                             "matching analysis/MVA_hcc/train_model.py.")
    parser.add_argument("--support-floor", type=float, default=50.0)
    parser.add_argument("--structures", nargs="+", default=list(STRUCTURES))
    return parser.parse_args()


def axis_scores(evaluation, structure):
    """Signal-versus-each-background-family log likelihood ratios.

    axis 'continuum': the exclusive QCD/QED continuum (real protons, no MadGraph
    statistics involved). axis 'pooled': the combinatorial background.
    """
    kappas = evaluation["kappas"]
    axes = {}
    for name, classes in (("continuum", structure["continuum_classes"]),
                          ("pooled", structure["pooled_classes"])):
        columns = np.r_[0, np.asarray(classes)]
        sub_kappas = kappas[np.asarray(classes) - 1]
        axes[name] = harness.scores_from(evaluation, kappas=sub_kappas, columns=columns)
    axes["plugin"] = harness.scores_from(evaluation)
    return axes


def report_partition(evaluation, mass, effective, label):
    total, per_category = harness.significance(mass)
    yields = mass.sum(axis=2)
    return {
        "label": label,
        "n_categories": int(mass.shape[1]),
        "significance": total,
        "per_category_significance": per_category,
        "per_category_effective_pooled_events": effective,
        "component_yields": {
            name: yields[index] for index, name in enumerate(evaluation["component_names"])
        },
    }


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    scan_rows = []

    for structure_name in args.structures:
        structure = STRUCTURES[structure_name]
        print(f"\n=== {structure_name} ===", flush=True)
        evaluation = harness.build_evaluation(
            args.data_dir, structure["class_map"], structure["class_names"],
            seed=args.seed, grid_cells=args.grid_cells, cap=args.train_cap,
            n_estimators=args.n_estimators, stop_cap=args.stop_cap or None,
        )
        if list(evaluation["component_names"]) != list(COMPONENTS):
            raise RuntimeError(f"Unexpected components {evaluation['component_names']}")
        axes = axis_scores(evaluation, structure)
        block = {
            "class_map": list(structure["class_map"]),
            "class_names": list(structure["class_names"]),
            "kappas": evaluation["kappas"],
            "band_probability_residual": evaluation["band_probability_residual"],
            "build_seconds": evaluation["build_seconds"],
            "preselection_component_yields": {
                name: float(evaluation["weight"][evaluation["component"] == index].sum())
                for index, name in enumerate(evaluation["component_names"])
            },
            "partitions": [],
        }

        # ---- single cut on the plug-in score, as the nominal workflow does
        row_score, cell_score = axes["plugin"]
        finite = row_score[np.isfinite(row_score)]
        low = float(np.floor(np.percentile(finite, 0.05)))
        high = float(np.ceil(np.max(finite)))
        thresholds = np.linspace(low, high - 1.0e-6, N_SCAN)
        scan, effective = harness.threshold_scan(evaluation, row_score, cell_score, thresholds)
        scan_z = np.asarray([harness.significance(item[:, None, :])[0] for item in scan])
        valid = np.flatnonzero(effective >= args.support_floor)
        if valid.size == 0:
            raise RuntimeError("No threshold satisfies the pooled-background support floor")
        best = valid[int(np.argmax(scan_z[valid]))]
        yields = scan[best].sum(axis=1)
        block["single_cut"] = {
            "threshold": float(thresholds[best]),
            "significance": float(scan_z[best]),
            "effective_pooled_events": float(effective[best]),
            "signal_yield": float(yields[0]),
            "background_yield": float(yields[1:].sum()),
            "component_yields": dict(zip(evaluation["component_names"], yields)),
            "support_floor_binding": bool(best == valid[-1]),
        }
        print(f"  single cut: T>={thresholds[best]:.3f} Z={scan_z[best]:.4f} "
              f"S={yields[0]:.3f} B={yields[1:].sum():.1f} neff={effective[best]:.0f}", flush=True)
        for index, threshold in enumerate(thresholds):
            scan_rows.append([structure_name, threshold, scan_z[index], effective[index]]
                             + scan[index].sum(axis=1).tolist())

        # ---- one-dimensional ladders on the plug-in score
        for n_bins in (6, 12):
            edges = harness.signal_quantile_edges(row_score, evaluation, n_bins)
            mass, effective_bins = harness.ladder_mass(evaluation, row_score, cell_score, edges)
            entry = report_partition(evaluation, mass, effective_bins, f"ladder_plugin_{n_bins}")
            block["partitions"].append(entry)
            print(f"  {entry['label']:26s} Z={entry['significance']:.4f} "
                  f"min neff={effective_bins.min():.0f}", flush=True)

        # ---- two-dimensional partitions: one axis per background family.
        # The continuum axis costs no pooled MC (QEDcc and QCDcc SuperChic carry
        # real protons), but the pooled component still has to be spread over the
        # product partition, so both the direct count and the factorised estimate
        # are reported.
        for n_pooled, n_continuum in ((6, 2), (6, 3), (6, 4), (8, 4)):
            pooled_edges = harness.signal_quantile_edges(axes["pooled"][0], evaluation, n_pooled)
            continuum_edges = harness.signal_quantile_edges(
                axes["continuum"][0], evaluation, n_continuum
            )
            row_axes = (axes["pooled"][0], axes["continuum"][0])
            cell_axes = (axes["pooled"][1], axes["continuum"][1])
            edge_sets = (pooled_edges, continuum_edges)
            mass, effective_bins = harness.grid_mass(evaluation, row_axes, cell_axes, edge_sets)
            entry = report_partition(
                evaluation, mass, effective_bins, f"grid_{n_pooled}x{n_continuum}"
            )
            entry["pooled_axis_bins"] = n_pooled
            entry["continuum_axis_bins"] = n_continuum
            entry["direct_count_usable"] = bool(effective_bins.min() >= args.support_floor)
            block["partitions"].append(entry)
            factorised, factor_effective, residual = harness.factorised_grid_mass(
                evaluation, row_axes, cell_axes, edge_sets
            )
            factor_entry = report_partition(
                evaluation, factorised, factor_effective,
                f"grid_factorised_{n_pooled}x{n_continuum}",
            )
            factor_entry["pooled_axis_bins"] = n_pooled
            factor_entry["continuum_axis_bins"] = n_continuum
            factor_entry["factorisation_residual"] = residual
            block["partitions"].append(factor_entry)
            print(f"  {entry['label']:26s} Z={entry['significance']:.4f} "
                  f"min neff={effective_bins.min():.0f}   "
                  f"factorised Z={factor_entry['significance']:.4f} "
                  f"(min neff={factor_effective.min():.0f}) "
                  f"residual={max(residual.values()):.3f}", flush=True)

        # ---- how much QED separation each score axis actually retains
        signal_rows = np.flatnonzero(evaluation["component"] == 0)
        qed_rows = np.flatnonzero(evaluation["component"] == 2)
        weight = evaluation["weight"]
        retention = {}
        for axis_name in ("plugin", "continuum", "pooled"):
            score = axes[axis_name][0]
            order = np.argsort(-score[signal_rows])
            cumulative = np.cumsum(weight[signal_rows][order]) / weight[signal_rows].sum()
            entry = {}
            for target in (0.2, 0.3, 0.5):
                cut = score[signal_rows][order][int(np.searchsorted(cumulative, target))]
                keep = score[qed_rows] >= cut
                entry[target] = float(weight[qed_rows][keep].sum() / weight[qed_rows].sum())
            retention[axis_name] = entry
        block["qed_efficiency_at_signal_efficiency"] = retention
        print("  QED efficiency at 30% signal efficiency: "
              + ", ".join(f"{name}={value[0.3]:.4f}" for name, value in retention.items()),
              flush=True)
        results[structure_name] = block
        del evaluation

    write_yaml(output_dir / "stage3_report.yaml", {
        "description": "H(cc) score construction and categorisation on the charm-only components",
        "data_dir": str(Path(args.data_dir).resolve()),
        "grid_cells": args.grid_cells,
        "support_floor": args.support_floor,
        "seed": args.seed,
        "hyperparameters": {**harness.PRODUCTION_PARAMS, "n_estimators": args.n_estimators},
        "stop_cap": args.stop_cap or None,
        "structures": results,
        "runtime_seconds": time.perf_counter() - started,
        "note": "Classifier-only, stat-only, perfectly known nominal backgrounds. "
                "The bb components are excluded; they are 1.5% of the background.",
    })
    with open(output_dir / "single_cut_scan.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure", "threshold", "significance", "effective_pooled_events"]
                        + list(COMPONENTS))
        writer.writerows(scan_rows)
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
