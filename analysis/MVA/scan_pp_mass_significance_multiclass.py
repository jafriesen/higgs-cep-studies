#!/usr/bin/env python3
"""Significance scan for the 3-class central+proton MVA (mx peak fit).

Consumes run_dijet_mva_multiclass_protons.py output (dataset.npz with mx,
process, physical_weight; scores.npz with the per-class OOF softmax
probabilities). The softmax model is trained with equal class priors, so its
probability ratios estimate class likelihood ratios. The physical analysis
score folds the real background normalizations back in as plug-in priors
kappa_b = Y_b / Y_signal (expected-yield ratios):

  T = log(p_sig) - log(sum_b kappa_b p_b),

which is monotone in the true signal/background likelihood ratio, so cutting on
T is the near-optimal single-number selection. The scan then exploits the mx
peak exactly as scan_pp_mass_significance.py does: for each T cut, signal and
total background are histogrammed in mx over the signal window and combined as

  Z = sqrt( sum_i S_i^2 / (S_i + B_i) )        (per-bin S/sqrt(S+B) in quadrature)

which rewards the signal's 125 GeV peak against the smooth background. Yields
use the physical per-event weights, so the MadGraph combinatorial background
enters at its full (enormous) pre-cut rate and the scan shows how far the MVA
plus the mass peak suppress it.

Run under the analysis environment:
  source setup_env.sh
  python3 analysis/MVA/scan_pp_mass_significance_multiclass.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.config_utils import resolve_path  # noqa: E402

CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
BACKGROUNDS = ("QCDbb", "QCDbb_madgraph")
LABELS = {
    "Hbb": "H->bb (SuperChic protons)",
    "QCDbb": "SuperChic QCDbb",
    "QCDbb_madgraph": "MadGraph QCDbb + min-bias protons",
}
COLORS = {"Hbb": "#0072B2", "QCDbb": "#E69F00", "QCDbb_madgraph": "#D55E00"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "analysis/MVA/output/mva_bb_multiclass_protons"),
        help="Multiclass+protons MVA output directory (dataset.npz, scores.npz).",
    )
    parser.add_argument("--mass-range", default="117,133", help="mx window low,high in GeV.")
    parser.add_argument("--mass-bin-width", type=float, default=1.0, help="mx bin width in GeV.")
    parser.add_argument("--n-cuts", type=int, default=400, help="Number of T-cut scan points.")
    parser.add_argument(
        "--kappa",
        action="append",
        default=[],
        metavar="CHANNEL=FACTOR",
        help="Multiply a background's plug-in kappa (yield prior); may be repeated. "
        "Use to scan the MadGraph normalization uncertainty, e.g. QCDbb_madgraph=0.2.",
    )
    parser.add_argument(
        "--template-background",
        action="append",
        default=[],
        metavar="CHANNEL",
        help="Estimate this background's post-cut mx shape as its (fully sampled) "
        "pre-cut histogram scaled by the scalar cut efficiency, valid because the "
        "score is decorrelated from mx. Repeatable; defaults to QCDbb_madgraph. "
        "Fixes the few-survivor spiky-histogram bias in the significance.",
    )
    parser.add_argument(
        "--no-template", action="store_true",
        help="Disable shape templating; histogram post-cut MC directly (reproduces "
        "the MC-statistics-limited estimate).",
    )
    parser.add_argument(
        "--min-template-mc", type=int, default=20,
        help="Minimum surviving MC events a templated background must keep for a cut "
        "to be eligible as the optimum, so its scaled normalization is trustworthy.",
    )
    return parser.parse_args()


def parse_range(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise RuntimeError(f"Invalid --mass-range '{value}', expected low,high")
    low, high = float(parts[0]), float(parts[1])
    if high <= low:
        raise RuntimeError("--mass-range high must exceed low")
    return low, high


def parse_kappa_scales(entries):
    scales = {name: 1.0 for name in BACKGROUNDS}
    for entry in entries:
        if "=" not in entry:
            raise RuntimeError(f"Invalid --kappa '{entry}', expected CHANNEL=FACTOR")
        channel, value = (part.strip() for part in entry.split("=", 1))
        if channel not in scales:
            raise RuntimeError(f"Unknown --kappa channel '{channel}'; choose from {', '.join(BACKGROUNDS)}")
        scales[channel] = float(value)
    return scales


def load_output(output_dir):
    dataset_path, scores_path = output_dir / "dataset.npz", output_dir / "scores.npz"
    if not dataset_path.is_file() or not scores_path.is_file():
        raise RuntimeError(f"Missing dataset.npz/scores.npz in {output_dir}; run run_dijet_mva_multiclass_protons.py")
    dataset = np.load(dataset_path, allow_pickle=False)
    scores = np.load(scores_path, allow_pickle=False)
    for field in ("process", "mx", "physical_weight"):
        if field not in dataset.files:
            raise RuntimeError(f"{dataset_path} lacks '{field}'; rerun the MVA (it must store physical_weight)")
    if "oof_probabilities" not in scores.files:
        raise RuntimeError(f"{scores_path} lacks 'oof_probabilities'")
    probabilities = scores["oof_probabilities"]
    if probabilities.shape[1] != len(CLASS_ORDER):
        raise RuntimeError(f"Expected {len(CLASS_ORDER)} class columns, got {probabilities.shape[1]}")
    return {
        "process": dataset["process"].astype(str),
        "mx": np.asarray(dataset["mx"], dtype=np.float64),
        "weight": np.asarray(dataset["physical_weight"], dtype=np.float64),
        "probabilities": probabilities,
    }


def physical_kappas(data, kappa_scales):
    """kappa_b = (Y_b / Y_signal) * user scale, from finite-mx physical yields."""
    finite = np.isfinite(data["mx"])
    yields = {}
    for name in CLASS_ORDER:
        mask = (data["process"] == name) & finite
        yields[name] = float(np.sum(data["weight"][mask]))
    if yields["Hbb"] <= 0.0:
        raise RuntimeError("Non-positive signal yield")
    return {name: (yields[name] / yields["Hbb"]) * kappa_scales[name] for name in BACKGROUNDS}, yields


def plug_in_log_likelihood_score(probabilities, kappas):
    p_sig = probabilities[:, CLASS_ORDER.index("Hbb")]
    background = sum(
        kappas[name] * probabilities[:, CLASS_ORDER.index(name)] for name in BACKGROUNDS
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p_sig) - np.log(background)


def significance_from_hists(s_hist, b_hist):
    denom = s_hist + b_hist
    nonzero = denom > 0.0
    z2 = np.zeros_like(s_hist)
    z2[nonzero] = s_hist[nonzero] ** 2 / denom[nonzero]
    return float(np.sqrt(np.sum(z2)))


def background_hist(mass, weight, process, selection, mass_bins, template_backgrounds, precut):
    """Total background mx histogram. Templated backgrounds contribute their
    fully-sampled pre-cut shape scaled by the scalar post-cut efficiency
    (valid when the score is decorrelated from mx); others use post-cut MC."""
    total = np.zeros(len(mass_bins) - 1, dtype=np.float64)
    for name in BACKGROUNDS:
        mask = process == name
        if name in template_backgrounds:
            post_yield = float(np.sum(weight[selection & mask]))
            efficiency = post_yield / precut["yield"][name] if precut["yield"][name] > 0.0 else 0.0
            total = total + efficiency * precut["hist"][name]
        else:
            hist, _ = np.histogram(
                mass[selection & mask], bins=mass_bins, weights=weight[selection & mask]
            )
            total = total + hist
    return total


def precut_templates(mass, weight, process, mass_bins):
    hist, yields = {}, {}
    for name in BACKGROUNDS:
        mask = process == name
        hist[name], _ = np.histogram(mass[mask], bins=mass_bins, weights=weight[mask])
        yields[name] = float(np.sum(weight[mask]))
    return {"hist": hist, "yield": yields}


def scan(data, score, in_window, mass_bins, cuts, template_backgrounds):
    mass = data["mx"][in_window]
    weight = data["weight"][in_window]
    process = data["process"][in_window]
    is_signal = process == "Hbb"
    t = score[in_window]
    precut = precut_templates(mass, weight, process, mass_bins)
    results = []
    for cut in cuts:
        selected = t >= cut
        s_hist, _ = np.histogram(
            mass[selected & is_signal], bins=mass_bins, weights=weight[selected & is_signal]
        )
        b_hist = background_hist(
            mass, weight, process, selected, mass_bins, template_backgrounds, precut
        )
        results.append({
            "t_cut": float(cut),
            "significance": significance_from_hists(s_hist, b_hist),
            "signal_yield": float(np.sum(weight[selected & is_signal])),
            "sc_bkg_yield": float(np.sum(weight[selected & (process == "QCDbb")])),
            "mg_bkg_yield": float(np.sum(weight[selected & (process == "QCDbb_madgraph")])),
            "signal_events": int(np.sum(selected & is_signal)),
            "sc_mc_events": int(np.sum(selected & (process == "QCDbb"))),
            "mg_mc_events": int(np.sum(selected & (process == "QCDbb_madgraph"))),
            "background_events": int(np.sum(selected & ~is_signal)),
        })
    return results


def best_valid(results, template_backgrounds, min_template_mc):
    """Best cut whose templated backgrounds each retain >= min_template_mc MC
    survivors, so their scaled-template normalization is statistically meaningful."""
    key = {"QCDbb": "sc_mc_events", "QCDbb_madgraph": "mg_mc_events"}
    valid = [
        r for r in results
        if all(r[key[name]] >= min_template_mc for name in template_backgrounds)
    ]
    pool = valid if valid else results
    return max(pool, key=lambda r: r["significance"]), bool(valid)


def plot_probability_histograms(path, probabilities, process, score):
    """Per-process, area-normalized distributions of the three softmax class
    probabilities and of the plug-in log-likelihood score T. Unweighted, so
    these show the classifier's raw per-process separation independent of
    physical yields."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    prob_bins = np.linspace(0.0, 1.0, 51)
    panels = [
        (axes[0, 0], probabilities[:, CLASS_ORDER.index("Hbb")], "p(signal)", prob_bins, False),
        (axes[0, 1], probabilities[:, CLASS_ORDER.index("QCDbb")], "p(SuperChic QCDbb)", prob_bins, False),
        (axes[1, 0], probabilities[:, CLASS_ORDER.index("QCDbb_madgraph")], "p(MadGraph QCDbb)", prob_bins, False),
    ]
    finite_t = score[np.isfinite(score)]
    t_bins = np.linspace(*np.nanpercentile(finite_t, [0.1, 99.9]), 51)
    panels.append((axes[1, 1], score, "T", t_bins, False))
    for ax, values, xlabel, bins, log_x in panels:
        for name in CLASS_ORDER:
            mask = process == name
            ax.hist(values[mask], bins=bins, density=True, histtype="step", linewidth=1.6,
                    color=COLORS[name], linestyle="-" if name == "Hbb" else "--", label=LABELS[name])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized events / bin")
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path, results):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def plot_scan(path, results, score, process, weight, in_window):
    finite_t = score[in_window & np.isfinite(score)]
    t_low, t_high = np.min(finite_t), np.max(finite_t)
    t_bins = np.linspace(t_low, t_high, 61)
    finite_score = in_window & np.isfinite(score)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for name in CLASS_ORDER:
        mask = finite_score & (process == name)
        ax.hist(
            score[mask], bins=t_bins, density=True, histtype="step", linewidth=1.6,
            color=COLORS[name], linestyle="-" if name == "Hbb" else "--", label=LABELS[name],
        )
    ax.set_xlabel("Plug-in log-likelihood score T")
    ax.set_ylabel("Normalized events / bin")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    cuts = np.asarray([r["t_cut"] for r in results])
    significance = np.asarray([r["significance"] for r in results])
    finite_cuts = np.isfinite(cuts)
    sig_ax = ax.twinx()
    sig_ax.plot(
        cuts[finite_cuts], significance[finite_cuts], color="#009E73", linewidth=2.0,
        label="Binned mass significance",
    )
    signal = finite_score & (process == "Hbb")
    background = finite_score & (process != "Hbb")
    s_hist, _ = np.histogram(score[signal], bins=t_bins, weights=weight[signal])
    b_hist, _ = np.histogram(score[background], bins=t_bins, weights=weight[background])
    denominator = s_hist + b_hist
    t_bin_significance = np.divide(
        s_hist, np.sqrt(denominator), out=np.zeros_like(s_hist), where=denominator > 0.0,
    )
    combined_t_significance = float(np.sqrt(np.sum(t_bin_significance ** 2)))
    sig_ax.stairs(
        t_bin_significance, t_bins, color="#CC79A7", linewidth=1.8, linestyle="--",
        label=f"T-bin significance (combined Z={combined_t_significance:.3g})",
    )
    sig_ax.set_ylabel("Significance")
    handles, labels = ax.get_legend_handles_labels()
    sig_handles, sig_labels = sig_ax.get_legend_handles_labels()
    ax.legend(handles + sig_handles, labels + sig_labels, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return combined_t_significance


def plot_mass_by_process(path, data, in_window, mass_bins, selection, title, template_backgrounds=(), precut=None):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    mass = data["mx"][in_window]
    weight = data["weight"][in_window]
    process = data["process"][in_window]
    for name in CLASS_ORDER:
        mask = selection & (process == name)
        if not np.any(mask):
            continue
        if name in template_backgrounds and precut is not None and precut["yield"][name] > 0.0:
            efficiency = float(np.sum(weight[mask])) / precut["yield"][name]
            values = efficiency * precut["hist"][name]
            ax.stairs(values, mass_bins, color=COLORS[name], linewidth=1.5, linestyle="--",
                      label=f"{LABELS[name]} template ({np.sum(values):.3g})")
            continue
        ax.hist(mass[mask], bins=mass_bins, weights=weight[mask], histtype="step",
                linewidth=1.5, linestyle="-" if name == "Hbb" else "--",
                color=COLORS[name], label=f"{LABELS[name]} ({np.sum(weight[mask]):.3g})")
    ax.set_xlabel("$M_X$ [GeV]")
    ax.set_ylabel("Expected events / GeV")
    ax.set_yscale("log")
    ax.set_xlim(float(mass_bins[0]), float(mass_bins[-1]))
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if args.mass_bin_width <= 0.0 or args.n_cuts < 2:
        raise RuntimeError("--mass-bin-width must be > 0 and --n-cuts >= 2")
    output_dir = resolve_path(args.output_dir, base=REPO)
    mass_range = parse_range(args.mass_range)
    kappa_scales = parse_kappa_scales(args.kappa)

    template_backgrounds = [] if args.no_template else (args.template_background or ["QCDbb_madgraph"])
    for name in template_backgrounds:
        if name not in BACKGROUNDS:
            raise RuntimeError(f"Unknown --template-background '{name}'; choose from {', '.join(BACKGROUNDS)}")

    data = load_output(output_dir)
    kappas, yields = physical_kappas(data, kappa_scales)
    print("Physical yields in mx window (pre-cut):")
    for name in CLASS_ORDER:
        print(f"  {name}: {yields[name]:.6g}", flush=True)
    print(f"Plug-in kappas (Y_b/Y_sig x scale): {kappas}", flush=True)
    print(f"Templated backgrounds (pre-cut shape x efficiency): {template_backgrounds or 'none'}", flush=True)

    score = plug_in_log_likelihood_score(data["probabilities"], kappas)
    in_window = (
        np.isfinite(data["mx"]) & (data["mx"] >= mass_range[0]) & (data["mx"] <= mass_range[1])
    )
    if not np.any(in_window):
        raise RuntimeError(f"No events in mx window {mass_range}")
    mass_bins = np.arange(
        mass_range[0], mass_range[1] + 0.5 * args.mass_bin_width, args.mass_bin_width
    )
    # Scan the populated T range and add extra quantile resolution in the high-score tail.
    t_in = score[in_window]
    finite_t = t_in[np.isfinite(t_in)]
    if not finite_t.size:
        raise RuntimeError("No finite T scores in the mx window")
    t_lo = float(np.nanpercentile(finite_t, 1.0))
    t_hi = float(np.nanmax(finite_t))
    cuts = np.unique(np.concatenate([
        np.array([-np.inf]),
        np.linspace(t_lo, t_hi, args.n_cuts),
        np.nanpercentile(finite_t, np.linspace(50.0, 99.9, args.n_cuts // 2)),
    ]))
    results = scan(data, score, in_window, mass_bins, cuts, template_backgrounds)
    best, constrained = best_valid(results, template_backgrounds, args.min_template_mc)
    if template_backgrounds and not constrained:
        print(
            f"WARNING: no cut keeps >= {args.min_template_mc} MC survivors for the "
            "templated background(s); reported optimum is MC-statistics-limited.",
            flush=True,
        )
    no_cut = results[0]["significance"]

    print(f"No-cut binned Z (mx window only):        {no_cut:.6g}", flush=True)
    print(f"Best T cut:                              {best['t_cut']:.6g}", flush=True)
    print(f"Best combined significance Z:            {best['significance']:.6g}", flush=True)
    print(
        f"Best-cut yields: signal={best['signal_yield']:.6g}, "
        f"SC-QCDbb={best['sc_bkg_yield']:.6g}, MG-QCDbb={best['mg_bkg_yield']:.6g}",
        flush=True,
    )
    print(
        f"Best-cut MC events: signal={best['signal_events']}, "
        f"SC-QCDbb={best['sc_mc_events']}, MG-QCDbb={best['mg_mc_events']}",
        flush=True,
    )

    outputs = {
        "scan_csv": output_dir / "significance_scan.csv",
        "scan_plot": output_dir / "significance_scan.png",
        "class_probabilities": output_dir / "class_probabilities.png",
        "mass_before": output_dir / "mx_by_process_before_cut.png",
        "mass_after": output_dir / "mx_by_process_after_cut.png",
        "summary": output_dir / "significance_summary.yaml",
    }
    plot_probability_histograms(
        outputs["class_probabilities"], data["probabilities"], data["process"], score
    )
    precut = precut_templates(
        data["mx"][in_window], data["weight"][in_window], data["process"][in_window], mass_bins
    )
    write_csv(outputs["scan_csv"], results)
    t_binned_significance = plot_scan(
        outputs["scan_plot"], results, score, data["process"], data["weight"], in_window
    )
    all_sel = np.ones(int(np.sum(in_window)), dtype=bool)
    plot_mass_by_process(outputs["mass_before"], data, in_window, mass_bins, all_sel, "Before MVA cut")
    plot_mass_by_process(
        outputs["mass_after"], data, in_window, mass_bins,
        score[in_window] >= best["t_cut"], f"T >= {best['t_cut']:.4g}",
        template_backgrounds=template_backgrounds, precut=precut,
    )
    summary = {
        "output_dir": str(output_dir),
        "mass_range_gev": list(mass_range),
        "mass_bin_width_gev": args.mass_bin_width,
        "kappa_scales": kappa_scales,
        "plug_in_kappas": kappas,
        "templated_backgrounds": list(template_backgrounds),
        "prewindow_yields": yields,
        "no_cut_binned_significance": no_cut,
        "t_binned_significance": t_binned_significance,
        "best_t_cut": best["t_cut"],
        "best_significance": best["significance"],
        "best_signal_yield": best["signal_yield"],
        "best_sc_background_yield": best["sc_bkg_yield"],
        "best_mg_background_yield": best["mg_bkg_yield"],
        "best_signal_events": best["signal_events"],
        "best_background_events": best["background_events"],
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    with open(outputs["summary"], "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
