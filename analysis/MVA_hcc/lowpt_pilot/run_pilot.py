#!/usr/bin/env python3
"""Where does the surviving MadGraph background live in the generation variables?

The QCD samples are generated inside boxes in parton pT, |eta| and m(jj), while
the analysis selection asks only for two jets above 15 GeV, roughly back to back,
plus a proton pair. Events outside the boxes are simply absent from the estimate.

QCDbb spans the union pT 15-100, |eta| < 3.0, m 50-180 across its three
campaigns, so the survival can be measured over the full region rather than
extrapolated. The measurement is done under the cc-trained model: Delphes does
not distinguish c from b jets -- flavour enters only as a per-event tag weight --
so the kinematic dependence of the survival transfers between flavours, and the
overlap region where both exist provides the check.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.qed_study import harness  # noqa: E402
from analysis.MVA_hcc.qed_study.common import DEFAULT_OUTPUT, write_yaml  # noqa: E402

PARTON_FIELDS = ("pt1", "pt2", "eta1", "eta2", "mass")
# Hcc / exclusive continuum / non-exclusive, the structure recommended by
# analysis/MVA_hcc/qed_study/FINDINGS.md.
CLASS_MAP = (0, 1, 1, 2, 2)
CLASS_NAMES = ("Hcc", "exclusive_continuum", "nonexclusive_QCD")
PT_EDGES = (15, 20, 25, 30, 40, 50, 65, 80, 120)
ETA_EDGES = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
MASS_EDGES = (40, 70, 100, 140, 180, 300)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "data" / "pilot"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--grid-cells", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--stop-cap", type=int, default=40000)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--train-on", default="narrow", choices=("narrow", "wide"),
                        help="narrow: train with the non-exclusive class restricted to the "
                             "current cc box, reproducing the model in use. wide: train on "
                             "everything, to see what the classifier learns when it is shown "
                             "the forward region.")
    return parser.parse_args()


def profile(name, values, edges, weight, surviving, label):
    """Preselected share, surviving share and relative survival per bin.

    `weight` and `surviving` are both per central event: the pooled backgrounds
    are scored through the proton-pool grid, so an event contributes a fraction
    of its weight rather than passing or failing.
    """
    rows = []
    base = None
    for low, high in zip(edges[:-1], edges[1:]):
        cell = (values >= low) & (values < high)
        if weight[cell].sum() <= 0.0:
            continue
        efficiency = surviving[cell].sum() / weight[cell].sum()
        if base is None:
            base = efficiency
        rows.append({"low": float(low), "high": float(high),
                     "preselection_share": float(weight[cell].sum() / weight.sum()),
                     "survivor_share": float(surviving[cell].sum() / surviving.sum()),
                     "survival": float(efficiency),
                     "relative_survival": float(efficiency / base) if base else np.nan,
                     "central_events": int(cell.sum())})
    print(f"\n{label} ({name})")
    print(f"{'bin':>12s} {'presel':>9s} {'survivors':>10s} {'survival':>10s} {'rel':>7s} {'events':>9s}")
    for r in rows:
        print(f"{r['low']:5.1f}-{r['high']:<6.1f} {r['preselection_share']:9.3f} "
              f"{r['survivor_share']:10.3f} {r['survival']:10.3e} "
              f"{r['relative_survival']:7.1f} {r['central_events']:9d}")
    return rows


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = harness.load_dataset(args.data_dir, mmap=True)
    names = [item["name"] for item in data["metadata"]["components"]]
    if len(CLASS_MAP) != len(names):
        raise RuntimeError(f"CLASS_MAP has {len(CLASS_MAP)} entries for components {names}")

    # Restricting the pooled training rows to the current cc box reproduces the
    # model the analysis actually uses; the forward region is then unseen.
    restrict = None
    if args.train_on == "narrow":
        parton = np.asarray(np.load(Path(args.data_dir) / "parton.npy", mmap_mode="r"))
        eta = np.maximum(np.abs(parton[:, 2]), np.abs(parton[:, 3]))
        soft = np.minimum(parton[:, 0], parton[:, 1])
        inside = (soft >= 30.0) & (eta <= 1.5) & (parton[:, 4] >= 70.0) & (parton[:, 4] <= 140.0)
        restrict = np.isnan(parton[:, 0]) | inside
        print(f"training restricted to the current cc box: "
              f"{100 * restrict.mean():.1f}% of rows retained", flush=True)

    evaluation = harness.build_evaluation(
        args.data_dir, CLASS_MAP, CLASS_NAMES, seed=args.seed,
        grid_cells=args.grid_cells, n_estimators=args.n_estimators,
        stop_cap=args.stop_cap, train_rows_mask=restrict,
    )
    row_score, cell_score = harness.scores_from(evaluation)
    edges = harness.signal_quantile_edges(row_score, evaluation, args.ladder_bins)
    mass, effective = harness.ladder_mass(evaluation, row_score, cell_score, edges)
    total, per_category = harness.significance(mass)
    print(f"\nladder Z = {total:.4f}; per-category n_eff {np.round(effective, 0)}", flush=True)

    parton = np.asarray(np.load(Path(args.data_dir) / "parton.npy", mmap_mode="r"))
    threshold = float(edges[-2])          # lower edge of the top ladder category

    report = {"description": __doc__.strip().splitlines()[0],
              "data_dir": str(Path(args.data_dir).resolve()),
              "train_on": args.train_on,
              "ladder_significance": total,
              "per_category_effective": effective,
              "top_category_threshold": threshold,
              "profiles": {}}

    contribution = evaluation["cell_contribution"]
    central_rows = evaluation["central_rows"]
    for cname in ("QCDbb_madgraph", "QCDcc_madgraph"):
        if cname not in names:
            continue
        select = evaluation["central_component"] == names.index(cname)
        block = parton[central_rows[select]]
        weights = contribution[select].sum(axis=1)
        surviving = np.where(cell_score[select] >= threshold,
                             contribution[select], 0.0).sum(axis=1)
        soft = np.minimum(block[:, 0], block[:, 1])
        eta = np.maximum(np.abs(block[:, 2]), np.abs(block[:, 3]))
        report["profiles"][cname] = {
            "parton_softer_pt": profile(cname, soft, PT_EDGES, weights, surviving, "parton softer pT"),
            "parton_max_abs_eta": profile(cname, eta, ETA_EDGES, weights, surviving, "parton max |eta|"),
            "parton_mass": profile(cname, block[:, 4], MASS_EDGES, weights, surviving, "parton m(jj)"),
            "preselected_yield": float(weights.sum()),
            "surviving_yield": float(surviving.sum()),
        }
        print(f"\n{cname}: preselected {weights.sum():.4g}, surviving {surviving.sum():.4g}")

        # How much of the true background can the current cc box even see?
        inside = ((soft >= 30.0) & (eta <= 1.5)
                  & (block[:, 4] >= 70.0) & (block[:, 4] <= 140.0))
        share = surviving[inside].sum() / surviving.sum() if surviving.sum() else np.nan
        report["profiles"][cname]["cc_box_share_of_survivors"] = float(share)
        report["profiles"][cname]["cc_box_share_of_preselection"] = float(
            weights[inside].sum() / weights.sum()
        )
        print(f"  inside the current cc box (pT>30, |eta|<1.5, 70<m<140): "
              f"{100 * share:.1f}% of survivors, "
              f"{100 * weights[inside].sum() / weights.sum():.1f}% of preselection")
        np.savez_compressed(
            output_dir / f"events_{args.train_on}_{cname}.npz",
            parton=block, weight=weights, surviving=surviving,
        )

    write_yaml(output_dir / f"pilot_{args.train_on}.yaml", report)
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
