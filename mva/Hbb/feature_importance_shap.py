#!/usr/bin/env python3
"""TreeSHAP feature importance of an H(bb) binary result, in the selection tail.

Each saved fold model is re-evaluated on its held-out rows exactly as training
does: real-proton rows as stored, accidental-proton rows expanded over the
proton grid with their analytic pair intensity.  For the binary classifier the
BDT-only score is affine in the booster margin,
    score = -(coef * margin + intercept) - log(kappa),
so per-feature SHAP values in score units are -coef times the margin SHAP.

Importance is the mean |SHAP| weighted by each event's (or grid cell's)
expected contribution to the final selection: its weight times the probability
that its vertex-likelihood shift lifts it over the operating threshold.  So only
the tail that sets the sensitivity counts.  Feature groups are scored with
the summed SHAP of their members (additivity), so correlated partners in one
group do not dilute each other.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from minbias.vertex import vertex_likelihood_table  # noqa: E402
from mva.common.dataset import write_yaml  # noqa: E402
from mva.common.protons import build_pair_density, load_pps_config, proton_cell_intensities  # noqa: E402
from mva.common.training import analysis_state, read_logical_matrix, set_proton_cells  # noqa: E402
from mva.common.weights import plugin_score  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_model_condor", SCRIPT_DIR / "train_model_condor.py")
_condor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_condor)


def load_fold(result_dir, fold, report):
    booster = xgb.Booster()
    booster.load_model(str(result_dir / f"fold_{fold}_main_model.json"))
    best = next(item["best_iteration"] for item in report["folds"] if item["fold"] == fold)
    with np.load(result_dir / f"fold_{fold}_main_calibrator.npz") as source:
        coef = float(np.asarray(source["coef"]).ravel()[0])
        intercept = float(np.asarray(source["intercept"]).ravel()[0])
    return booster, int(best), coef, intercept


def pass_probability(base, hypothesis, table, threshold):
    """P(base + vertex shift >= threshold), the event's share of the final selection."""
    if hypothesis == "none":
        return (base >= threshold).astype(float)
    probability = np.asarray(table[f"{hypothesis}_probability"], dtype=float)
    shifts = np.asarray(table["log_likelihood_ratio"], dtype=float)
    return (probability[np.newaxis, :] * (base[..., np.newaxis] + shifts >= threshold)).sum(axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True, help="YAML: group name -> feature list")
    parser.add_argument("--chunk", type=int, default=4000, help="Accidental-proton rows per grid chunk")
    args_cli = parser.parse_args()
    result_dir = args_cli.result_dir.resolve()
    report = yaml.safe_load((result_dir / "report.yaml").read_text(encoding="utf-8"))
    if report.get("architecture") != "binary":
        raise SystemExit("Only the binary architecture has the affine score used here")
    manifest = yaml.safe_load(Path(report["orchestration"]["manifest"]).read_text(encoding="utf-8"))
    args = _condor._training_args(manifest)
    groups = yaml.safe_load(args_cli.groups.read_text(encoding="utf-8"))

    state = analysis_state(manifest["data_dir"], args, list(manifest["features"]))
    data = state["data"]
    features = state["logical_features"]
    missing = [name for group in groups.values() for name in group if name not in features]
    groups = {name: [f for f in members if f in features] for name, members in groups.items()}
    groups = {name: members for name, members in groups.items() if members}
    matrix, pair_column, estimator = read_logical_matrix(data, args.batch_rows, features)
    kappa = float(np.asarray(state["kappas"]).ravel()[0])
    threshold = float(report["single_cut_operating_point"]["threshold"])
    table = vertex_likelihood_table(
        bins=args.vertex_bins,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        single_arm_time_resolution_ps=args.pps_time_ps,
        pv_z_resolution_cm=args.pv_z_resolution_cm,
        pv_time_resolution_ps=args.pv_time_ps,
    )
    pairs = cell_centres = None
    if np.any(state["pooled"]):
        pairs = build_pair_density(
            state["metadata"]["protons"]["path"], load_pps_config(args.repo / args.pps_config), seed=args.seed
        )
        edges = np.linspace(-args.max_delta_y, args.max_delta_y, args.grid_cells + 1)
        cell_centres = 0.5 * (edges[:-1] + edges[1:])

    names = [item["name"] for item in state["components"]]
    n_features = len(features)
    sums = {n: np.zeros(n_features) for n in names}          # sum w |phi_f|
    signed = {n: np.zeros(n_features) for n in names}        # sum w phi_f
    group_sums = {n: np.zeros(len(groups)) for n in names}   # sum w |sum_{f in g} phi_f|
    totals = {n: 0.0 for n in names}
    counts = {n: 0 for n in names}
    group_index = [[features.index(f) for f in members] for members in groups.values()]
    closure = []

    for fold in (0, 1):
        booster, best, coef, intercept = load_fold(result_dir, fold, report)
        iteration = (0, best + 1)

        def contribute(name, hypothesis, x, weight):
            margin = booster.predict(xgb.DMatrix(x), output_margin=True, iteration_range=iteration)
            base = -(coef * margin + intercept) - np.log(kappa)
            share = weight * pass_probability(base, hypothesis, table, threshold)
            keep = share > 0.0
            if not np.any(keep):
                return
            phi = booster.predict(xgb.DMatrix(x[keep]), pred_contribs=True, iteration_range=iteration)
            phi = -coef * phi[:, :n_features]
            w = share[keep]
            sums[name] += w @ np.abs(phi)
            signed[name] += w @ phi
            group_sums[name] += w @ np.abs(np.stack([phi[:, idx].sum(axis=1) for idx in group_index], axis=1))
            totals[name] += float(w.sum())
            counts[name] += int(keep.sum())
            # the affine score must match the training's calibrated plugin score
            if len(closure) < 4:
                logit = coef * margin[keep][:200] + intercept
                probability = np.stack([1.0 / (1.0 + np.exp(logit)), 1.0 / (1.0 + np.exp(-logit))], axis=1)
                closure.append(float(np.max(np.abs(plugin_score(probability, [kappa]) - base[keep][:200]))))

        for component_id, component in enumerate(state["components"]):
            rows = np.flatnonzero(
                (state["folds"] == fold) & state["eligible"] & (state["component_ids"] == component_id)
            )
            if rows.size == 0:
                continue
            name = names[component_id]
            if component["real_protons"]:
                contribute(name, component["vertex_hypothesis"], np.asarray(matrix[rows]), state["effective_physical"][rows])
                continue
            for start in range(0, rows.size, args_cli.chunk):
                selected = rows[start : start + args_cli.chunk]
                working = np.repeat(matrix[selected], args.grid_cells, axis=0)
                set_proton_cells(
                    working, pair_column, estimator,
                    np.repeat(selected, args.grid_cells), np.tile(cell_centres, selected.size),
                )
                intensity = proton_cell_intensities(
                    pairs, np.asarray(data["dijet_rapidity"])[selected], edges, args.mass_window, args.pileup_mu
                )
                weight = (state["physical"][selected, np.newaxis] * intensity).ravel()
                contribute(name, component["vertex_hypothesis"], working, weight)
            print(f"  fold {fold} {name}: done", flush=True)

    if closure and max(closure) > 1e-3:
        raise SystemExit(f"Affine score disagrees with the plugin score ({max(closure):.2e})")
    kinds = {
        "signal": [n for n, c in zip(names, state["components"]) if c.get("signal")],
        "exclusive_background": [n for n, c in zip(names, state["components"]) if c["real_protons"] and not c.get("signal")],
        "inclusive_background": [n for n, c in zip(names, state["components"]) if not c["real_protons"]],
    }
    summary = {}
    for kind, members in kinds.items():
        total = sum(totals[n] for n in members)
        if total <= 0.0:
            continue
        mean_abs = sum(sums[n] for n in members) / total
        mean_signed = sum(signed[n] for n in members) / total
        group_abs = sum(group_sums[n] for n in members) / total
        order = np.argsort(-mean_abs)
        summary[kind] = {
            "selected_yield": total,
            "selected_mc_entries": int(sum(counts[n] for n in members)),
            "features": [
                {"feature": features[i], "mean_abs_shap": float(mean_abs[i]), "mean_shap": float(mean_signed[i])}
                for i in order
            ],
            "groups": sorted(
                ({"group": g, "mean_abs_shap": float(v)} for g, v in zip(groups, group_abs)),
                key=lambda item: -item["mean_abs_shap"],
            ),
        }
    write_yaml(
        result_dir / "feature_importance_shap.yaml",
        {
            "interpretation": __doc__.strip().splitlines()[0],
            "operating_threshold": threshold,
            "score_closure_max_abs": max(closure) if closure else None,
            "groups_not_in_model": missing,
            "summary": summary,
        },
    )
    for kind, item in summary.items():
        print(f"\n== {kind}: selected yield {item['selected_yield']:.4g} from {item['selected_mc_entries']} MC entries")
        print("  groups: " + ", ".join(f"{g['group']} {g['mean_abs_shap']:.3f}" for g in item["groups"]))
        for row in item["features"][:10]:
            print(f"  {row['feature']:36s} |phi| {row['mean_abs_shap']:.3f}  mean {row['mean_shap']:+.3f}")
    print(f"\nscore closure {max(closure) if closure else float('nan'):.1e}; wrote {result_dir / 'feature_importance_shap.yaml'}")


if __name__ == "__main__":
    main()
