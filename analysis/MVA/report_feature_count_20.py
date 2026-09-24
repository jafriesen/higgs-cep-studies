#!/usr/bin/env python3
"""Full results for the locked 20-variable multiclass proton MVA.

Companion to resolve_feature_count.py, which established that 20 input
variables beat 15 (Z 1.190 vs 1.087, a real gap) and tie 42 (1.190 vs 1.201,
indistinguishable). This script re-fits that configuration and produces the
complete picture: score distributions, the scan across single cuts, the mass
spectra before and after selection, and the six-category breakdown.

Same treatment as the resolver, since the numbers must match:
  - the v01 MadGraph campaign is dropped and stitching weights re-derived,
  - two-fold out-of-fold scoring by hard-event group,
  - the proton pair is integrated over the min-bias pool rather than sampled,
  - the MadGraph mass template is the inclusive mx shape scaled to the yield in
    each score region, since the score is near-independent of mx for that
    background.

Signal and SuperChic QCDbb keep their per-event mx throughout: their protons
are real.

Run under the analysis environment:
  source setup_env.sh
  python3 analysis/MVA/report_feature_count_20.py \
      --cache-dir analysis/MVA/output/mva_bb_multiclass_protons/cache_v4_all_campaigns \
      --ranking analysis/MVA/output/mva_bb_multiclass_protons/architecture_study_v1/stage1_ranking.yaml \
      --output-dir analysis/MVA/output/mva_bb_multiclass_protons/feature_count_v1/report_20
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.MVA.optimize_dijet_mva_multiclass_protons import (  # noqa: E402
    MASS_BINS,
    physical_kappas,
    plugin_score,
)
from analysis.MVA.run_dijet_mva_multiclass_protons import (  # noqa: E402
    MAX_ABS_RAPIDITY_DIFFERENCE,
)
from analysis.MVA.resolve_feature_count import (  # noqa: E402
    DIJET_RAPIDITY_FEATURE,
    PAIR_FEATURE,
    PRODUCTION_PARAMS,
    assign_folds,
    fit_calibrated,
    fold_training_rows,
    gather,
    load_pool,
    load_reweighted,
    pool_mass_between,
)
from analysis.MVA.study_mva_architecture import (  # noqa: E402
    calibrated_probabilities,
    ladder_edges,
    write_yaml,
)

# Okabe-Ito, already this repo's plotting convention. Validated: lightness band,
# chroma floor, CVD separation and normal-vision floor all pass. The orange sits
# at 2.19:1 against white, so every series is direct-labelled and the same
# numbers ship as CSV.
COLORS = ("#0072B2", "#E69F00", "#D55E00")
LABELS = ("H->bb (SuperChic)", "QCDbb (SuperChic)", "QCDbb (MadGraph)")
INK = "#2b2b2b"
MUTED = "#767676"
SCORE_RANGE_BINS = 320
MIN_EFFECTIVE_CENTRAL = 50.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ladder-bins", type=int, default=6)
    parser.add_argument("--grid-cells", type=int, default=32)
    parser.add_argument("--madgraph-train-cap", type=int, default=250000)
    return parser.parse_args()


def sweep_pool(
    matrix, rapidity, weight, pool, model, calibrator, kappas, pair_column, cells
):
    """Per-cell score and expected-yield contribution for each central event.

    The pair reaches the model through one feature only, so sweeping deltaY on a
    grid and weighting each cell by the pool probability in the matching yx
    window replaces the sampled draws with a deterministic integral.

    Both arrays are returned per (event, cell) rather than pre-binned, because
    the statistical unit is the central event: an event's cells must be summed
    before squaring, or the effective count is inflated by roughly the number of
    contributing cells.
    """
    score = np.empty((matrix.shape[0], cells), dtype=np.float32)
    contribution = np.empty((matrix.shape[0], cells), dtype=np.float32)
    covered = np.zeros(matrix.shape[0])
    bounds = np.linspace(
        -MAX_ABS_RAPIDITY_DIFFERENCE, MAX_ABS_RAPIDITY_DIFFERENCE, cells + 1
    )
    for index in range(cells):
        low, high = bounds[index], bounds[index + 1]
        matrix[:, pair_column] = 0.5 * (low + high)
        score[:, index] = plugin_score(
            calibrated_probabilities(model, calibrator, matrix), kappas
        )
        mass = pool_mass_between(pool, rapidity + low, rapidity + high) / pool["total"]
        covered += mass
        contribution[:, index] = weight * mass
    return score, contribution, covered


def accumulate_regions(score, contribution, edges):
    """Yield and per-event-aggregated squared yield for a set of score bins."""
    size = edges.size - 1
    cell = np.clip(
        np.searchsorted(edges, score.ravel(), side="right") - 1, 0, size - 1
    )
    rows = np.repeat(np.arange(score.shape[0]), score.shape[1])
    per_event = np.bincount(
        rows * size + cell, weights=contribution.ravel().astype(np.float64),
        minlength=score.shape[0] * size,
    ).reshape(score.shape[0], size)
    return per_event.sum(axis=0), np.sum(per_event * per_event, axis=0)


def accumulate_above(score, contribution, thresholds):
    """Yield and per-event-aggregated squared yield above each threshold."""
    total = np.zeros(thresholds.size)
    squared = np.zeros(thresholds.size)
    for index, threshold in enumerate(thresholds):
        per_event = np.sum(
            np.where(score >= threshold, contribution, 0.0), axis=1, dtype=np.float64
        )
        total[index] = per_event.sum()
        squared[index] = np.sum(per_event * per_event)
    return total, squared


def plain(value):
    """Coerce numpy scalars and arrays to built-ins so yaml.safe_dump accepts
    the payload."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def class_score_mass(data, score, class_id, score_edges):
    """Weighted (score, mx) histogram for a class whose protons are real."""
    rows = np.flatnonzero(data["class"] == class_id)
    edges = np.r_[score_edges, np.inf]
    histogram, _, _ = np.histogram2d(
        score[rows], data["mx"][rows], bins=(edges, MASS_BINS),
        weights=data["physical_weight"][rows],
    )
    return histogram


def significance(signal, superchic, madgraph_yield, shape):
    background = superchic + madgraph_yield * shape
    total = signal + background
    return float(np.sqrt(np.sum(
        np.divide(signal * signal, total, out=np.zeros_like(signal), where=total > 0.0)
    )))


def cumulative_from_top(values):
    return np.cumsum(values[::-1], axis=0)[::-1]


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_scores(centers, spectra, output_dir):
    """Score shape per class. Areas normalised so shapes are comparable; the
    physical yields differ by five orders of magnitude and live in the scan."""
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    for values, color, label in zip(spectra, COLORS, LABELS):
        total = values.sum()
        if total <= 0:
            continue
        axis.step(centers, values / total, where="mid", color=color, linewidth=2.0,
                  label=label)
        peak = int(np.argmax(values))
        axis.annotate(
            label, (centers[peak], values[peak] / total), color=color,
            fontsize=8, xytext=(4, 6), textcoords="offset points",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Plug-in score  log p(H) - log(k1 p(QCD) + k2 p(MG))", color=INK)
    axis.set_ylabel("Fraction of class per bin", color=INK)
    axis.set_title("Out-of-fold score distributions, 20 variables", color=INK)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "score_distributions.png", dpi=180)
    plt.close(figure)


def plot_scan(scan, operating, output_dir):
    """Three stacked panels on one shared threshold axis.

    Deliberately not a dual-axis chart: significance, yields and MC support have
    unrelated scales, so they get their own panels rather than a second y-axis.
    """
    figure, axes = plt.subplots(
        3, 1, figsize=(8.4, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 0.9]},
    )
    threshold = scan["threshold"]

    axes[0].plot(threshold, scan["significance"], color="#009E73", linewidth=2.0)
    axes[0].axvline(operating["threshold"], color=MUTED, linestyle="--", linewidth=1.2)
    axes[0].annotate(
        f"operating point {operating['threshold']:.2f}\nZ = {operating['significance']:.3f}",
        (operating["threshold"], operating["significance"]), color=INK, fontsize=9,
        xytext=(8, -28), textcoords="offset points",
    )
    axes[0].set_ylabel("Mass-binned Z", color=INK)
    axes[0].set_title(
        "Single-cut scan, 20 variables (two-fold out-of-fold)", color=INK
    )

    for key, color, label in zip(
        ("signal_yield", "superchic_yield", "madgraph_yield"), COLORS, LABELS
    ):
        axes[1].plot(threshold, scan[key], color=color, linewidth=2.0, label=label)
        finite = np.flatnonzero(scan[key] > 0)
        if finite.size:
            spot = finite[len(finite) // 3]
            axes[1].annotate(
                label, (threshold[spot], scan[key][spot]), color=color, fontsize=8,
                xytext=(4, 4), textcoords="offset points",
            )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Expected events (3000 fb$^{-1}$)", color=INK)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    axes[2].plot(threshold, scan["madgraph_effective"], color=COLORS[2], linewidth=2.0)
    axes[2].axhline(
        MIN_EFFECTIVE_CENTRAL, color=MUTED, linestyle=":", linewidth=1.2,
    )
    axes[2].annotate(
        f"MC support floor ({MIN_EFFECTIVE_CENTRAL:.0f})", (threshold[2], MIN_EFFECTIVE_CENTRAL),
        color=MUTED, fontsize=8, xytext=(0, 5), textcoords="offset points",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Effective MadGraph\ncentral events", color=INK)
    axes[2].set_xlabel("Score threshold", color=INK)

    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.axvline(operating["threshold"], color=MUTED, linestyle="--", linewidth=1.0)
    figure.tight_layout()
    figure.savefig(output_dir / "single_cut_scan.png", dpi=180)
    plt.close(figure)


def plot_mass_spectra(preselection, selected, output_dir):
    """Mass spectra before and after the cut, background stacked with the signal
    overlaid at a stated multiplier (it is ~1% of the background)."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for axis, block, title in zip(
        axes, (preselection, selected),
        ("Preselection", "At the operating point"),
    ):
        signal, superchic, madgraph = block
        axis.stairs(
            madgraph, MASS_BINS, fill=True, color=COLORS[2], alpha=0.85,
            edgecolor="white", linewidth=1.5, label=LABELS[2],
        )
        axis.stairs(
            superchic + madgraph, MASS_BINS, baseline=madgraph, fill=True,
            color=COLORS[1], alpha=0.85, edgecolor="white", linewidth=1.5,
            label=LABELS[1],
        )
        scale = max(1.0, 10.0 ** np.floor(np.log10(
            max(superchic.sum() + madgraph.sum(), 1.0) / max(signal.sum(), 1e-9)
        )))
        axis.stairs(
            signal * scale, MASS_BINS, color=COLORS[0], linewidth=2.2,
            label=f"{LABELS[0]} x{scale:.0f}",
        )
        axis.set_yscale("log")
        axis.set_xlabel("m$_X$ from protons [GeV]", color=INK)
        axis.set_title(title, color=INK)
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.legend(
            handles=[
                Patch(facecolor=COLORS[2], label=LABELS[2]),
                Patch(facecolor=COLORS[1], label=LABELS[1]),
                Line2D([0], [0], color=COLORS[0], linewidth=2.2,
                       label=f"{LABELS[0]} x{scale:.0f}"),
            ],
            frameon=False, fontsize=8,
        )
    axes[0].set_ylabel("Expected events / GeV (3000 fb$^{-1}$)", color=INK)
    figure.tight_layout()
    figure.savefig(output_dir / "mass_spectra.png", dpi=180)
    plt.close(figure)


def plot_category_masses(categories, output_dir):
    """Small multiples: one mass spectrum per score category."""
    count = len(categories)
    columns = 3
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.2 * rows), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()
    for index, block in enumerate(categories):
        axis = axes[index]
        signal, superchic, madgraph = block["signal"], block["superchic"], block["madgraph"]
        axis.stairs(madgraph, MASS_BINS, fill=True, color=COLORS[2], alpha=0.85,
                    edgecolor="white", linewidth=1.0)
        axis.stairs(superchic + madgraph, MASS_BINS, baseline=madgraph, fill=True,
                    color=COLORS[1], alpha=0.85, edgecolor="white", linewidth=1.0)
        scale = max(1.0, 10.0 ** np.floor(np.log10(
            max(superchic.sum() + madgraph.sum(), 1.0) / max(signal.sum(), 1e-9)
        )))
        axis.stairs(signal * scale, MASS_BINS, color=COLORS[0], linewidth=2.0)
        axis.set_yscale("log")
        axis.set_title(
            f"category {index}  [{block['low']:.2f}, {block['high']:.2f})\n"
            f"S={signal.sum():.1f}  B={superchic.sum() + madgraph.sum():.0f}  "
            f"Z={block['significance']:.3f}  (signal x{scale:.0f})",
            fontsize=8, color=INK,
        )
        axis.grid(True, alpha=0.25, linewidth=0.6)
    for axis in axes[count:]:
        axis.axis("off")
    for axis in axes[:count]:
        axis.set_xlabel("m$_X$ [GeV]", color=INK)
    axes[0].set_ylabel("Expected events / GeV", color=INK)
    figure.suptitle(
        "Mass spectra by score category, 20 variables "
        "(MadGraph = inclusive shape scaled to its yield)",
        color=INK, fontsize=10,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[2], label=LABELS[2]),
            Patch(facecolor=COLORS[1], label=LABELS[1]),
            Line2D([0], [0], color=COLORS[0], linewidth=2.0,
                   label=f"{LABELS[0]} (scaled per panel)"),
        ],
        frameon=False, fontsize=9, ncol=3, loc="lower center",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(output_dir / "mass_spectra_by_category.png", dpi=180)
    plt.close(figure)


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = load_reweighted(args.cache_dir)
    pool = load_pool()
    with open(args.ranking, encoding="utf-8") as handle:
        ranked = np.asarray(yaml.safe_load(handle)["ranked_feature_indices"], dtype=int)
    features = ranked[: args.n_features]
    names = data["feature_names"]
    pair_column = int(np.flatnonzero(names[features] == PAIR_FEATURE)[0])
    rapidity_index = int(np.flatnonzero(names == DIJET_RAPIDITY_FEATURE)[0])

    kappas, yields = physical_kappas(data["class"], data["physical_weight"])
    madgraph_mask = data["class"] == 2
    shape, _ = np.histogram(
        data["mx"][madgraph_mask], MASS_BINS,
        weights=data["physical_weight"][madgraph_mask],
    )
    shape = shape / shape.sum()

    folds = assign_folds(data["group_id"], args.seed)
    score = np.full(data["class"].size, np.nan)
    fits = []
    for fold in range(2):
        train_rows, stop_rows = fold_training_rows(
            data, folds, fold, args.madgraph_train_cap, args.seed
        )
        model, calibrator = fit_calibrated(
            data, features, train_rows, stop_rows, PRODUCTION_PARAMS, args.seed + fold
        )
        fits.append((model, calibrator))
        held_out = np.flatnonzero(folds == fold)
        real = held_out[data["class"][held_out] != 2]
        score[real] = plugin_score(
            calibrated_probabilities(model, calibrator, gather(data, real, features)),
            kappas,
        )
        print(f"fold {fold}: trained on {train_rows.size} rows", flush=True)

    signal_score = score[data["class"] == 0]
    ladder = ladder_edges(
        signal_score, np.zeros(signal_score.size, dtype=np.int8), args.ladder_bins
    )
    finite = np.isfinite(score)
    score_edges = np.linspace(
        float(np.floor(np.nanpercentile(score[finite], 0.05))),
        float(np.ceil(np.nanmax(score[finite]))),
        SCORE_RANGE_BINS + 1,
    )

    madgraph_rows = np.flatnonzero(madgraph_mask)
    _groups, first = np.unique(data["group_id"][madgraph_rows], return_index=True)
    central = madgraph_rows[first]
    central_matrix = gather(data, central, features)
    rapidity = gather(data, central, [rapidity_index])[:, 0].astype(np.float64)
    central_weight = data["central_stitch_new"][central] * data["physical_scale"]

    # Sweep the pool once per fold, keeping per-(event, cell) scores and
    # contributions so every central event's cells can be summed before being
    # squared. Squaring per cell would inflate the effective count by roughly
    # the number of contributing cells.
    n_scan = 96
    scan_thresholds = np.linspace(score_edges[0], score_edges[-1] - 1e-6, n_scan)
    ladder_yield = np.zeros(ladder.size - 1)
    ladder_squared = np.zeros(ladder.size - 1)
    above_yield = np.zeros(n_scan)
    above_squared = np.zeros(n_scan)
    residual = 0.0
    for fold, (model, calibrator) in enumerate(fits):
        selected = folds[central] == fold
        cell_score, cell_contribution, covered = sweep_pool(
            central_matrix[selected], rapidity[selected], central_weight[selected],
            pool, model, calibrator, kappas, pair_column, args.grid_cells,
        )
        residual = max(residual, float(np.max(np.abs(
            covered - data["band_probability"][central][selected]
        ))))
        block_yield, block_squared = accumulate_regions(
            cell_score, cell_contribution, ladder
        )
        ladder_yield += block_yield
        ladder_squared += block_squared
        block_yield, block_squared = accumulate_above(
            cell_score, cell_contribution, scan_thresholds
        )
        above_yield += block_yield
        above_squared += block_squared
        del cell_score, cell_contribution
    print(f"band-probability residual: {residual:.3e}", flush=True)

    signal_map = class_score_mass(data, score, 0, scan_thresholds)
    superchic_map = class_score_mass(data, score, 1, scan_thresholds)
    signal_cumulative = cumulative_from_top(signal_map)
    superchic_cumulative = cumulative_from_top(superchic_map)
    effective = np.divide(
        above_yield * above_yield, above_squared,
        out=np.zeros_like(above_yield), where=above_squared > 0.0,
    )

    scan = {
        "threshold": scan_thresholds,
        "significance": np.array([
            significance(
                signal_cumulative[index], superchic_cumulative[index],
                above_yield[index], shape,
            ) for index in range(n_scan)
        ]),
        "signal_yield": signal_cumulative.sum(axis=1),
        "superchic_yield": superchic_cumulative.sum(axis=1),
        "madgraph_yield": above_yield,
        "madgraph_effective": effective,
    }
    valid = np.flatnonzero(effective >= MIN_EFFECTIVE_CENTRAL)
    best = valid[int(np.argmax(scan["significance"][valid]))]
    operating = {
        "threshold": float(scan["threshold"][best]),
        "significance": float(scan["significance"][best]),
        "signal_yield": float(scan["signal_yield"][best]),
        "superchic_yield": float(scan["superchic_yield"][best]),
        "madgraph_yield": float(scan["madgraph_yield"][best]),
        "madgraph_effective_central_events": float(effective[best]),
    }
    background = operating["superchic_yield"] + operating["madgraph_yield"]
    operating["signal_over_background"] = operating["signal_yield"] / background
    operating["counting_significance"] = operating["signal_yield"] / np.sqrt(
        operating["signal_yield"] + background
    )

    with open(output_dir / "single_cut_scan.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "threshold", "mass_binned_significance", "signal_yield",
            "superchic_yield", "madgraph_yield", "madgraph_effective_central_events",
        ])
        for index in range(n_scan):
            writer.writerow([
                scan["threshold"][index], scan["significance"][index],
                scan["signal_yield"][index], scan["superchic_yield"][index],
                scan["madgraph_yield"][index], effective[index],
            ])

    categories = []
    for index in range(ladder.size - 1):
        low, high = ladder[index], ladder[index + 1]
        rows = np.flatnonzero((score >= low) & (score < high))
        signal_rows = rows[data["class"][rows] == 0]
        superchic_rows = rows[data["class"][rows] == 1]
        signal_hist, _ = np.histogram(
            data["mx"][signal_rows], MASS_BINS,
            weights=data["physical_weight"][signal_rows],
        )
        superchic_hist, _ = np.histogram(
            data["mx"][superchic_rows], MASS_BINS,
            weights=data["physical_weight"][superchic_rows],
        )
        madgraph_hist = ladder_yield[index] * shape
        spread = ladder_squared[index]
        categories.append({
            "low": float(low) if np.isfinite(low) else float(score_edges[0]),
            "high": float(high) if np.isfinite(high) else float(score_edges[-1]),
            "signal": signal_hist, "superchic": superchic_hist,
            "madgraph": madgraph_hist,
            "significance": significance(
                signal_hist, superchic_hist, ladder_yield[index], shape
            ),
            "madgraph_effective_central_events": (
                float(ladder_yield[index] ** 2 / spread) if spread > 0.0 else 0.0
            ),
        })
    ladder_z = float(np.sqrt(sum(block["significance"] ** 2 for block in categories)))

    presel_signal, _ = np.histogram(
        data["mx"][data["class"] == 0], MASS_BINS,
        weights=data["physical_weight"][data["class"] == 0],
    )
    presel_superchic, _ = np.histogram(
        data["mx"][data["class"] == 1], MASS_BINS,
        weights=data["physical_weight"][data["class"] == 1],
    )
    presel_madgraph = ladder_yield.sum() * shape

    plot_scores(
        scan_thresholds,
        (signal_map.sum(axis=1), superchic_map.sum(axis=1), -np.diff(np.r_[above_yield, 0.0])),
        output_dir,
    )
    plot_scan(scan, operating, output_dir)
    plot_mass_spectra(
        (presel_signal, presel_superchic, presel_madgraph),
        (signal_cumulative[best], superchic_cumulative[best],
         above_yield[best] * shape),
        output_dir,
    )
    plot_category_masses(categories, output_dir)

    write_yaml(output_dir / "report.yaml", plain({
        "n_features": int(args.n_features),
        "features": names[features].tolist(),
        "hyperparameters": PRODUCTION_PARAMS,
        "folds": 2,
        "grid_cells": args.grid_cells,
        "seed": args.seed,
        "kappas": kappas.tolist(),
        "band_probability_residual": residual,
        "preselection_yields": {
            "signal": float(presel_signal.sum()),
            "superchic_qcdbb": float(presel_superchic.sum()),
            "madgraph_qcdbb": float(presel_madgraph.sum()),
        },
        "single_cut_operating_point": operating,
        "ladder_significance": ladder_z,
        "categories": [
            {
                "index": index,
                "score_range": [block["low"], block["high"]],
                "signal_yield": float(block["signal"].sum()),
                "superchic_yield": float(block["superchic"].sum()),
                "madgraph_yield": float(block["madgraph"].sum()),
                "significance": block["significance"],
                "madgraph_effective_central_events": block[
                    "madgraph_effective_central_events"
                ],
                "mass_bins_gev": MASS_BINS,
                "signal_mass": block["signal"],
                "superchic_mass": block["superchic"],
                "madgraph_mass": block["madgraph"],
            }
            for index, block in enumerate(categories)
        ],
        "note": (
            "Classifier-only, stat-only, perfectly known background. The v01 "
            "MadGraph campaign is dropped, so the parton pT 15-25 / |eta| up to "
            "3 region is unmodelled. The combinatorial acceptance factor is a "
            "detector design parameter, not an uncertainty."
        ),
    }))
    print(
        f"operating point {operating['threshold']:.3f}: "
        f"S={operating['signal_yield']:.2f} B={background:.1f} "
        f"Z={operating['significance']:.4f}; ladder Z={ladder_z:.4f}; "
        f"done in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
