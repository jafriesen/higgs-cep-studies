#!/usr/bin/env python3
"""Stage 1: which observables separate H(cc) from the exclusive continuum.

Backgrounds here are SuperChic QEDcc (gamma-gamma -> ccbar) and SuperChic QCDcc
(exclusive gg -> ccbar) only. Both carry a real proton pair, so nothing in this
stage depends on the min-bias pool or on MadGraph statistics.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.qed_study.common import (  # noqa: E402
    DEFAULT_DATA, DEFAULT_OUTPUT, assign_folds, balanced_weights, component_rows,
    conditional_separation, held_out_threshold_z, load_dataset, mass_binned_z,
    mass_histogram, scan_threshold_z, separation, write_yaml,
)

CONDITION_ON = "delta_eta_jj"
LOCKED_20 = (
    "delta_phi_jj", "yx_minus_dijet_rapidity", "dijet_mass", "pt_asymmetry",
    "interjet_bridge_asymmetry_projected", "delta_eta_jj", "n_tracks_interjet",
    "sum_track_pt_outer_negative", "jet_multiplicity", "jet1_eta",
    "jet2_pull_magnitude", "n_vertices", "n_tracks_outer_positive",
    "interjet_bridge_asymmetry", "jet1_charged_fraction", "dijet_rapidity",
    "n_outer_track_jets", "sum_track_pt_outside_pt2", "minimum_gap_size",
    "n_tracks_projected_bridge",
)
PARAMS = {
    "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "gamma": 0.0, "max_bin": 256,
}
N_ESTIMATORS = 400


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "exclusive_wide"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage1"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 8])
    return parser.parse_args()


def oof_binary(matrix, labels, weights, groups, seed, n_estimators):
    """Two-fold group-safe out-of-fold signal probability."""
    folds = assign_folds(groups, seed)
    score = np.full(labels.size, np.nan)
    for fold in (0, 1):
        train = folds != fold
        model = XGBClassifier(
            n_estimators=n_estimators, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=16,
            random_state=seed + fold, **PARAMS,
        )
        model.fit(matrix[train], labels[train],
                  sample_weight=balanced_weights(labels[train], weights[train], 2))
        score[~train] = model.predict_proba(matrix[~train])[:, 1]
    if not np.all(np.isfinite(score)):
        raise RuntimeError("Out-of-fold scoring left rows unscored")
    return score


def background_efficiency(score, is_signal, weights, signal_efficiencies):
    order = np.argsort(-score[is_signal])
    cumulative = np.cumsum(weights[is_signal][order]) / weights[is_signal].sum()
    output = {}
    for target in signal_efficiencies:
        threshold = score[is_signal][order][min(int(np.searchsorted(cumulative, target)),
                                                cumulative.size - 1)]
        keep = score[~is_signal] >= threshold
        output[target] = float(weights[~is_signal][keep].sum() / weights[~is_signal].sum())
    return output


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data_dir, mmap=False)
    features = data["features"]
    matrix = np.asarray(data["x"], dtype=np.float32)
    mx = np.asarray(data["mx"])
    weight = np.asarray(data["physical_weight"])
    groups = np.asarray(data["group_id"])

    rows = {name: component_rows(data, name)
            for name in ("Hcc", "QEDcc_superchic", "QCDcc_superchic")}
    condition = features.index(CONDITION_ON)
    signal, qed, qcd = rows["Hcc"], rows["QEDcc_superchic"], rows["QCDcc_superchic"]
    print(f"rows Hcc={signal.size:,} QEDcc={qed.size:,} QCDcc_SC={qcd.size:,}", flush=True)

    # ---- per-variable separation, raw and conditioned on the production angle
    ranking = []
    for index, name in enumerate(features):
        entry = {"feature": name}
        for label, background in (("qed", qed), ("qcd", qcd)):
            entry[f"separation_{label}"] = separation(
                matrix[signal, index], weight[signal], matrix[background, index], weight[background]
            )
            entry[f"conditional_{label}"] = (
                np.nan if index == condition else conditional_separation(
                    matrix[signal, index], weight[signal],
                    matrix[background, index], weight[background],
                    matrix[signal, condition], matrix[background, condition],
                )
            )
        entry["nan_fraction_signal"] = float(np.mean(~np.isfinite(matrix[signal, index])))
        entry["nan_fraction_qed"] = float(np.mean(~np.isfinite(matrix[qed, index])))
        ranking.append(entry)
    ranking.sort(key=lambda item: -(item["separation_qed"] if np.isfinite(item["separation_qed"]) else 0))

    with open(output_dir / "variable_ranking.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    print(f"\n{'feature':38s} {'sep(QED)':>9s} {'cond|dEta':>10s} {'sep(QCD)':>9s} {'NaN(H)':>7s}")
    for entry in ranking[:14]:
        print(f"{entry['feature']:38s} {entry['separation_qed']:9.4f} "
              f"{entry['conditional_qed']:10.4f} {entry['separation_qcd']:9.4f} "
              f"{entry['nan_fraction_signal']:7.3f}")

    # ---- cos_theta_star is a monotone map of delta_eta_jj: check, do not assume
    monotonic = None
    if "cos_theta_star" in features:
        a = matrix[:, condition]
        b = matrix[:, features.index("cos_theta_star")]
        good = np.isfinite(a) & np.isfinite(b)
        order = np.argsort(a[good], kind="stable")
        monotonic = {
            "spearman": float(np.corrcoef(np.argsort(np.argsort(a[good])),
                                          np.argsort(np.argsort(b[good])))[0, 1]),
            "max_violation": float(np.min(np.diff(b[good][order]))),
        }
        print(f"\ncos_theta_star vs {CONDITION_ON}: spearman={monotonic['spearman']:.6f} "
              f"min_step={monotonic['max_violation']:.3e}", flush=True)

    # ---- classifiers on nested feature sets
    ordered = [entry["feature"] for entry in ranking]
    sets = {f"top_{k}": ordered[:k] for k in args.top_k}
    sets["delta_eta_only"] = [CONDITION_ON]
    sets["locked_20"] = [name for name in LOCKED_20 if name in features]
    sets["all_wide"] = list(features)

    results = {}
    for background_name, background in (("QEDcc_superchic", qed), ("QCDcc_superchic+QEDcc_superchic", None)):
        selected = qed if background is not None else np.r_[qed, qcd]
        use = np.r_[signal, selected]
        labels = np.r_[np.ones(signal.size, np.int32), np.zeros(selected.size, np.int32)]
        is_signal = labels == 1
        base_z = mass_binned_z(mass_histogram(mx[use], weight[use], is_signal),
                               mass_histogram(mx[use], weight[use], ~is_signal))
        block = {"preselection_z": base_z,
                 "preselection_signal": float(weight[signal].sum()),
                 "preselection_background": float(weight[selected].sum())}
        for name, columns in sets.items():
            index = [features.index(column) for column in columns]
            score = oof_binary(matrix[use][:, index], labels, weight[use], groups[use],
                               args.seed, args.n_estimators)
            _t, _v, cut, best = scan_threshold_z(score, mx[use], weight[use], is_signal)
            held_out = held_out_threshold_z(
                score, mx[use], weight[use], is_signal,
                assign_folds(groups[use], args.seed + 1),
            )
            block[name] = {
                "n_features": len(columns),
                "best_z": best,
                "held_out_z": held_out,
                "gain_over_preselection": best / base_z,
                "threshold": float(cut),
                "background_efficiency_at_signal_efficiency":
                    background_efficiency(score, is_signal, weight[use], (0.2, 0.3, 0.5, 0.7)),
            }
            print(f"  [{background_name:32s}] {name:16s} n={len(columns):3d} "
                  f"Z={best:.4f} held_out={held_out:.4f} (x{best / base_z:.3f})", flush=True)
        results[background_name] = block

    payload = {
        "description": "H(cc) versus the exclusive continuum: variable power and achievable Z",
        "data_dir": str(Path(args.data_dir).resolve()),
        "n_features": len(features),
        "rows": {name: int(value.size) for name, value in rows.items()},
        "conditioning_variable": CONDITION_ON,
        "cos_theta_star_monotonicity": monotonic,
        "feature_sets": {name: list(columns) for name, columns in sets.items()},
        "results": results,
        "hyperparameters": {**PARAMS, "n_estimators": args.n_estimators},
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "note": "Classifier-only, stat-only. No MadGraph and no min-bias pool enter this stage.",
    }
    write_yaml(output_dir / "stage1_report.yaml", payload)
    np.savez_compressed(
        output_dir / "stage1_arrays.npz",
        # every array below is in ranking order, including the names
        features=np.asarray([entry["feature"] for entry in ranking]),
        separation_qed=np.asarray([entry["separation_qed"] for entry in ranking]),
        conditional_qed=np.asarray([entry["conditional_qed"] for entry in ranking]),
        separation_qcd=np.asarray([entry["separation_qcd"] for entry in ranking]),
        input_order=np.asarray(features),
    )
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
