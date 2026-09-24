#!/usr/bin/env python3
"""Cross-evaluate H(bb) exclusive and nonexclusive specialist classifiers."""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]

import sys

sys.path.insert(0, str(REPO))

from minbias import vertex_likelihood_table  # noqa: E402
from mva.Hbb.train_model_condor import _training_args  # noqa: E402
from mva.common.dataset import write_yaml  # noqa: E402
from mva.common.protons import proton_cell_intensities  # noqa: E402
from mva.common.training import (  # noqa: E402
    _madgraph_mass_shape,
    _mass_bins,
    _score_bins,
    _significance,
    _vertex_terms,
    calibrated_probabilities,
    plugin_score,
    prepare_training_state,
    set_proton_cells,
)


def load_fit(result_dir, fold):
    model = XGBClassifier()
    model.load_model(result_dir / f"fold_{fold}_main_model.json")
    payload = np.load(result_dir / f"fold_{fold}_main_calibrator.npz")
    calibrator = LogisticRegression()
    calibrator.classes_ = payload["classes"]
    calibrator.coef_ = payload["coef"]
    calibrator.intercept_ = payload["intercept"]
    calibrator.n_features_in_ = calibrator.coef_.shape[1]
    return model, calibrator


def add_real(histogram, component, score_n, score_e, mass, weight, vertex, edges, mass_edges):
    mass_bin = _mass_bins(mass, mass_edges)
    valid = (mass_bin >= 0) & (mass_bin < mass_edges.size - 1)
    bins = edges.size - 1
    mass_bins = mass_edges.size - 1
    probabilities, shifts = _vertex_terms(vertex, component["vertex_hypothesis"])
    for probability, shift in zip(probabilities, shifts):
        bin_n = _score_bins(score_n + shift, edges)
        bin_e = _score_bins(score_e, edges)
        combined = (
            (bin_n[valid] * bins + bin_e[valid]) * mass_bins + mass_bin[valid]
        )
        histogram[component["id"]] += np.bincount(
            combined,
            weights=weight[valid] * probability,
            minlength=bins * bins * mass_bins,
        ).reshape(bins, bins, mass_bins)


def evaluate(args, state, result_n, result_e, kappa_n, kappa_e):
    bins = args.score_bins
    edges = np.linspace(args.score_min, args.score_max, bins + 1)
    mass_edges = np.linspace(args.mass_window[0], args.mass_window[1], 17)
    histogram = np.zeros((state["n_components"], bins, bins, 16), dtype=np.float64)
    data = state["data"]
    component_ids = np.asarray(data["component"])
    rapidity = np.asarray(data["dijet_rapidity"])
    matrix = state["matrix"]
    vertex = vertex_likelihood_table(
        bins=args.vertex_bins,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        single_arm_time_resolution_ps=args.pps_time_ps,
        pv_z_resolution_cm=args.pv_z_resolution_cm,
        pv_time_resolution_ps=args.pv_time_ps,
    )

    for fold in (0, 1):
        model_n, calibrator_n = load_fit(result_n, fold)
        model_e, calibrator_e = load_fit(result_e, fold)
        for component in state["components"]:
            rows = np.flatnonzero(
                (state["folds"] == fold)
                & state["eligible"]
                & (component_ids == component["id"])
            )
            print(
                f"fold={fold} component={component['name']} rows={rows.size:,}",
                flush=True,
            )
            if component["real_protons"]:
                probability_n = calibrated_probabilities(
                    model_n, calibrator_n, matrix[rows], args.batch_rows
                )
                probability_e = calibrated_probabilities(
                    model_e, calibrator_e, matrix[rows], args.batch_rows
                )
                add_real(
                    histogram,
                    component,
                    plugin_score(probability_n, [kappa_n]),
                    plugin_score(probability_e, [kappa_e]),
                    np.asarray(data["proton_mx"])[rows],
                    state["physical"][rows],
                    vertex,
                    edges,
                    mass_edges,
                )
                continue

            joint = np.zeros((bins, bins), dtype=np.float64)
            for start in range(0, rows.size, args.evaluation_chunk):
                stop = min(start + args.evaluation_chunk, rows.size)
                selected = rows[start:stop]
                working = np.repeat(matrix[selected], args.grid_cells, axis=0)
                delta_y = np.tile(
                    0.5 * (state["cell_edges"][:-1] + state["cell_edges"][1:]),
                    selected.size,
                )
                set_proton_cells(
                    working,
                    state["pair_column"],
                    state["estimator"],
                    np.repeat(selected, args.grid_cells),
                    delta_y,
                )
                probability_n = calibrated_probabilities(
                    model_n, calibrator_n, working, args.batch_rows
                )
                probability_e = calibrated_probabilities(
                    model_e, calibrator_e, working, args.batch_rows
                )
                score_n = plugin_score(probability_n, [kappa_n]).reshape(
                    selected.size, args.grid_cells
                )
                score_e = plugin_score(probability_e, [kappa_e]).reshape(
                    selected.size, args.grid_cells
                )
                intensity = proton_cell_intensities(
                    state["pairs"],
                    rapidity[selected],
                    state["cell_edges"],
                    args.mass_window,
                    args.pileup_mu,
                )
                base_weight = state["physical"][selected, np.newaxis] * intensity
                probabilities, shifts = _vertex_terms(
                    vertex, component["vertex_hypothesis"]
                )
                for probability, shift in zip(probabilities, shifts):
                    bin_n = _score_bins(score_n + shift, edges)
                    bin_e = _score_bins(score_e, edges)
                    combined = bin_n * bins + bin_e
                    joint += np.bincount(
                        combined.ravel(),
                        weights=(base_weight * probability).ravel(),
                        minlength=bins * bins,
                    ).reshape(bins, bins)
                print(f"  cells {stop:,}/{rows.size:,}", flush=True)
            shape = _madgraph_mass_shape(
                state["pairs"],
                rapidity[rows],
                state["physical"][rows],
                mass_edges,
                args.max_delta_y,
                args.pileup_mu,
                args.intensity_chunk,
            )
            histogram[component["id"]] += joint[:, :, np.newaxis] * shape
    return histogram, edges, mass_edges


def summarize(histogram, edges, components):
    cumulative = np.cumsum(
        np.cumsum(histogram[:, ::-1, ::-1], axis=1), axis=2
    )[:, ::-1, ::-1]
    scan = np.transpose(cumulative, (1, 2, 0, 3))
    significance = _significance(scan.reshape(-1, len(components), histogram.shape[-1]))
    best_flat = int(np.argmax(significance))
    best_n, best_e = np.unravel_index(best_flat, significance.reshape(scan.shape[:2]).shape)
    best_yields = cumulative[:, best_n, best_e].sum(axis=1)

    signal_total = cumulative[0, 0, 0].sum()
    efficiency_rows = []
    for efficiency in (0.5, 0.25, 0.1, 0.05):
        marginal_n = cumulative[0, :, 0].sum(axis=1) / signal_total
        marginal_e = cumulative[0, 0, :].sum(axis=1) / signal_total
        index_n = int(np.argmin(np.abs(marginal_n - efficiency)))
        index_e = int(np.argmin(np.abs(marginal_e - efficiency)))
        combined_yields = cumulative[:, index_n, index_e].sum(axis=1)
        efficiency_rows.append(
            {
                "target_signal_efficiency": efficiency,
                "nonexclusive_threshold": float(edges[index_n]),
                "exclusive_threshold": float(edges[index_e]),
                "nonexclusive_marginal_signal_efficiency": float(marginal_n[index_n]),
                "exclusive_marginal_signal_efficiency": float(marginal_e[index_e]),
                "both_signal_efficiency": float(combined_yields[0] / signal_total),
                "both_component_yields": {
                    component["name"]: float(value)
                    for component, value in zip(components, combined_yields)
                },
            }
        )
    report = {
        "format_version": 1,
        "warning": "Exploratory two-dimensional scan; no MadGraph effective-support constraint is applied.",
        "best_two_dimensional": {
            "nonexclusive_threshold": float(edges[best_n]),
            "exclusive_threshold": float(edges[best_e]),
            "significance": float(significance[best_flat]),
            "component_yields": {
                component["name"]: float(value)
                for component, value in zip(components, best_yields)
            },
        },
        "matched_marginal_efficiencies": efficiency_rows,
    }
    return report, cumulative, significance.reshape(scan.shape[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nonexclusive-result",
        type=Path,
        default=SCRIPT_DIR / "results/fsr_mtd_nonexclusive_specialist_allrows_g256_s12345",
    )
    parser.add_argument(
        "--exclusive-result",
        type=Path,
        default=SCRIPT_DIR / "results/fsr_mtd_exclusive_specialist_allrows_g256_s12345",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "condor/train_fsr_mtd_nonexclusive_specialist_allrows_g256_s12345/manifest.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results/fsr_mtd_specialist_joint_g256_s12345",
    )
    parser.add_argument("--score-bins", type=int, default=160)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite output directory: {options.output_dir}")
    manifest = yaml.safe_load(options.manifest.read_text(encoding="utf-8"))
    args = _training_args(manifest)
    args.score_bins = options.score_bins
    report_n = yaml.safe_load(
        (options.nonexclusive_result / "report.yaml").read_text(encoding="utf-8")
    )
    report_e = yaml.safe_load(
        (options.exclusive_result / "report.yaml").read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    state = prepare_training_state(manifest["data_dir"], args, manifest["features"])
    histogram, edges, mass_edges = evaluate(
        args,
        state,
        options.nonexclusive_result,
        options.exclusive_result,
        float(report_n["kappas"][0]),
        float(report_e["kappas"][0]),
    )
    report, cumulative, significance = summarize(
        histogram, edges, state["components"]
    )
    report["runtime_seconds"] = time.perf_counter() - started
    options.output_dir.mkdir(parents=True)
    write_yaml(options.output_dir / "report.yaml", report)
    np.savez_compressed(
        options.output_dir / "report_data.npz",
        score_edges=edges,
        mass_edges=mass_edges,
        joint_component_mass=histogram,
        joint_cumulative_component_mass=cumulative,
        joint_significance=significance,
    )
    print(f"Wrote {options.output_dir / 'report.yaml'}", flush=True)


if __name__ == "__main__":
    main()
