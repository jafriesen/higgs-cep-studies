#!/usr/bin/env python3
"""Stage 5: how much of the result is MonteCarlo statistics, and what to generate.

The nominal H(cc) operating point sits where the effective MadGraph central-event
count falls off a cliff. This stage measures the support directly, bootstraps the
significance at the group level, and turns the shortfall into a concrete
generation request by locating the surviving events inside the generated phase
space of QCDcc__v01 (parton pT 30-100 GeV, |eta| < 1.5, m 70-140 GeV).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.qed_study.common import (  # noqa: E402
    DEFAULT_DATA, DEFAULT_OUTPUT, MASS_BINS, write_yaml,
)
from analysis.MVA_hcc.qed_study import harness  # noqa: E402

CLASS_MAP = (0, 1, 1, 2)
CLASS_NAMES = ("Hcc", "exclusive_continuum", "nonexclusive_QCD")
GENERATED_PHASE_SPACE = {"parton_pt_gev": (30.0, 100.0), "max_abs_eta": 1.5,
                         "dijet_mass_gev": (70.0, 140.0)}
EVENT_CHUNK = 200000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "cc_wide"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage5"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--grid-cells", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--train-cap", type=int, default=250000)
    parser.add_argument("--stop-cap", type=int, default=40000)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--replicas", type=int, default=300)
    parser.add_argument("--target-neff", type=float, default=200.0)
    return parser.parse_args()


def per_event_categories(category, contribution, n_categories):
    output = np.zeros((category.shape[0], n_categories))
    for start in range(0, category.shape[0], EVENT_CHUNK):
        stop = min(start + EVENT_CHUNK, category.shape[0])
        block = category[start:stop]
        rows = np.repeat(np.arange(stop - start), block.shape[1])
        output[start:stop] = np.bincount(
            rows * n_categories + block.ravel(),
            weights=contribution[start:stop].ravel(),
            minlength=(stop - start) * n_categories,
        ).reshape(stop - start, n_categories)
    return output


def significance_from(signal_mass, background_mass):
    total = signal_mass + background_mass
    return float(np.sqrt(np.sum(
        np.divide(signal_mass * signal_mass, total,
                  out=np.zeros_like(signal_mass), where=total > 0.0)
    )))


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = harness.build_evaluation(
        args.data_dir, CLASS_MAP, CLASS_NAMES, seed=args.seed, grid_cells=args.grid_cells,
        cap=args.train_cap, n_estimators=args.n_estimators, stop_cap=args.stop_cap,
    )
    names = evaluation["component_names"]
    row_score, cell_score = harness.scores_from(evaluation)
    edges = harness.signal_quantile_edges(row_score, evaluation, args.ladder_bins)
    n_categories = edges.size - 1
    mass, effective = harness.ladder_mass(evaluation, row_score, cell_score, edges)
    nominal, per_category = harness.significance(mass)
    print(f"\nladder Z={nominal:.4f}", flush=True)
    print(f"{'category':>9s} {'Z':>8s} {'neff':>9s} " + " ".join(f"{n:>14s}" for n in names))
    yields = mass.sum(axis=2)
    for cell in range(n_categories):
        print(f"{cell:9d} {per_category[cell]:8.4f} {effective[cell]:9.1f} "
              + " ".join(f"{yields[index, cell]:14.4g}" for index in range(len(names))))

    # ---- per-event contributions, for a group-level bootstrap
    pooled_id = int(np.flatnonzero(evaluation["is_pooled_component"])[0])
    select = evaluation["central_component"] == pooled_id
    pooled_category = harness.bin_index(cell_score[select], edges)
    pooled_events = per_event_categories(
        pooled_category, evaluation["cell_contribution"][select], n_categories
    )
    shape = evaluation["mass_shape"][pooled_id]

    exclusive = {}
    for component_id, name in enumerate(names):
        if evaluation["is_pooled_component"][component_id]:
            continue
        rows = np.flatnonzero(evaluation["component"] == component_id)
        category = harness.bin_index(row_score[rows], edges)
        mass_bin = np.clip(np.searchsorted(MASS_BINS, evaluation["mx"][rows], side="right") - 1,
                           0, MASS_BINS.size - 2)
        exclusive[name] = {
            "component_id": component_id,
            "cell": category * (MASS_BINS.size - 1) + mass_bin,
            "weight": evaluation["weight"][rows],
        }

    rng = np.random.default_rng(args.seed)
    n_cells = n_categories * (MASS_BINS.size - 1)
    replicas = np.empty(args.replicas)
    for replica in range(args.replicas):
        counts = rng.poisson(1.0, pooled_events.shape[0])
        pooled_yield = counts @ pooled_events
        background = pooled_yield[:, None] * shape
        signal = np.zeros((n_categories, MASS_BINS.size - 1))
        for name, block in exclusive.items():
            draw = rng.poisson(1.0, block["weight"].size)
            histogram = np.bincount(block["cell"], weights=draw * block["weight"],
                                    minlength=n_cells).reshape(n_categories, -1)
            if block["component_id"] == 0:
                signal += histogram
            else:
                background += histogram
        replicas[replica] = significance_from(signal, background)
    bootstrap = {
        "replicas": int(args.replicas),
        "sigma": float(replicas.std(ddof=1)),
        "median": float(np.median(replicas)),
        "median_minus_point": float(np.median(replicas) - nominal),
        "quantiles": {q: float(np.quantile(replicas, q)) for q in (0.16, 0.5, 0.84)},
    }
    print(f"\nbootstrap: sigma={bootstrap['sigma']:.4f} "
          f"median-point={bootstrap['median_minus_point']:+.4f}", flush=True)

    # ---- weight spread versus raw survivor count
    top = n_categories - 1
    contributing = pooled_events[:, top] > 0.0
    weights = pooled_events[contributing, top]
    support = {
        "top_category_effective_events": float(effective[top]),
        "top_category_contributing_events": int(contributing.sum()),
        "top_category_weight_concentration":
            float(effective[top] / contributing.sum()) if contributing.sum() else 0.0,
        "top_category_max_weight_share":
            float(weights.max() / weights.sum()) if weights.size else 0.0,
    }
    print(f"top category: {support['top_category_contributing_events']:,} contributing events, "
          f"neff {support['top_category_effective_events']:.1f} "
          f"(concentration {support['top_category_weight_concentration']:.3f})", flush=True)

    # ---- where the survivors sit inside the generated phase space
    features = evaluation["features"]
    location = None
    if {"jet1_pt_over_mjj", "jet2_pt_over_mjj", "jet1_eta", "jet2_eta"} <= set(features):
        matrix = harness.read_columns(
            harness.load_dataset(args.data_dir, mmap=True)["x"],
            [features.index(name) for name in
             ("jet1_pt_over_mjj", "jet2_pt_over_mjj", "jet1_eta", "jet2_eta", "dijet_mass")],
        )
        pooled_rows = np.flatnonzero(evaluation["is_pooled"])
        _u, first = np.unique(np.asarray(
            harness.load_dataset(args.data_dir, mmap=True)["group_id"])[pooled_rows],
            return_index=True)
        central = pooled_rows[first][select]
        block = matrix[central]
        jet1_pt = block[:, 0] * block[:, 4]
        jet2_pt = block[:, 1] * block[:, 4]
        soft_pt = np.minimum(jet1_pt, jet2_pt)
        max_eta = np.maximum(np.abs(block[:, 2]), np.abs(block[:, 3]))
        weight_top = pooled_events[:, top]
        total = weight_top.sum()

        def share(mask):
            return float(weight_top[mask].sum() / total) if total > 0 else np.nan

        pt_lo, pt_hi = GENERATED_PHASE_SPACE["parton_pt_gev"]
        eta_max = GENERATED_PHASE_SPACE["max_abs_eta"]
        location = {
            "reconstructed_softer_jet_pt_quantiles": {
                q: float(np.quantile(soft_pt[weight_top > 0], q)) for q in (0.05, 0.5, 0.95)
            },
            "reconstructed_max_abs_jet_eta_quantiles": {
                q: float(np.quantile(max_eta[weight_top > 0], q)) for q in (0.05, 0.5, 0.95)
            },
            "top_category_weight_within_5gev_of_pt_lower_edge": share(soft_pt < pt_lo + 5.0),
            "top_category_weight_within_0p2_of_eta_edge": share(max_eta > eta_max - 0.2),
            "generated_phase_space": GENERATED_PHASE_SPACE,
        }
        print(f"top category near pT lower edge: "
              f"{location['top_category_weight_within_5gev_of_pt_lower_edge']:.3f}; "
              f"near |eta| edge: {location['top_category_weight_within_0p2_of_eta_edge']:.3f}",
              flush=True)

    factor = args.target_neff / effective[top] if effective[top] > 0 else np.inf
    generated = evaluation["metadata"]["inputs"]["QCDcc_madgraph"]["campaigns"]["QCDcc__v01"]
    sizing = {
        "target_effective_events": args.target_neff,
        "current_effective_events": float(effective[top]),
        "required_factor": float(factor),
        "currently_generated": int(generated["generated"]),
        "required_generated": float(factor * generated["generated"]),
        "current_effective_mc_luminosity_fb_inv":
            float(generated["effective_mc_luminosity_fb_inv"]),
    }
    print(f"\nsizing: x{factor:.1f} more QCDcc events "
          f"({sizing['required_generated']:.3g} total) for neff >= {args.target_neff:.0f}",
          flush=True)

    write_yaml(output_dir / "stage5_report.yaml", {
        "description": "MonteCarlo support for the H(cc) selection and a generation sizing",
        "data_dir": str(Path(args.data_dir).resolve()),
        "ladder_bins": n_categories,
        "ladder_significance": nominal,
        "per_category_significance": per_category,
        "per_category_effective_pooled_events": effective,
        "per_category_component_yields": {
            name: yields[index] for index, name in enumerate(names)
        },
        "bootstrap": bootstrap,
        "support": support,
        "survivor_location": location,
        "generation_sizing": sizing,
        "grid_cells": args.grid_cells,
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "note": "Jet-level quantities are reconstructed, not parton-level; they locate "
                "the survivors relative to the generation cuts approximately.",
    })
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
