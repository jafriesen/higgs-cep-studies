#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


PROCESS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph_comb")
PROCESS_LABELS = {
    "Hbb": "H->bb (SuperChic protons)",
    "QCDbb": "SuperChic QCDbb",
    "QCDbb_madgraph_comb": "MadGraph QCDbb + min-bias protons",
}
COLORS = {
    "Hbb": "#0072B2",
    "QCDbb": "#E69F00",
    "QCDbb_madgraph_comb": "#D55E00",
}


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import resolve_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan MVA score cuts using cached PPS-selected event masses."
    )
    parser.add_argument("--flavor", choices=("bb",), default="bb", help="Signal flavor")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="MVA cache/output directory. Defaults to analysis/MVA/output/mva_bb.",
    )
    parser.add_argument(
        "--mass-range",
        default="117,133",
        help="Comma-separated cached pp mass range in GeV.",
    )
    parser.add_argument("--mass-bin-width", type=float, default=1.0, help="Mass bin width in GeV")
    parser.add_argument("--score-step", type=float, default=0.0001, help="MVA score scan step")
    parser.add_argument(
        "--channel-weight",
        action="append",
        default=[],
        metavar="CHANNEL=FACTOR",
        help=(
            "Scan-only channel weight multiplier; may be repeated. Channels: "
            + ", ".join(PROCESS_ORDER)
        ),
    )
    return parser.parse_args()


def ensure_delphes_python_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_ANALYZER_ENV") == "1":
        return
    lcg_view = Path(
        os.environ.get(
            "DELPHES_LCG_VIEW",
            "/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh",
        )
    )
    delphes_dir = Path(os.environ.get("DELPHES_DIR", "/home/jfriesen/Delphes"))
    if not lcg_view.is_file():
        raise RuntimeError(f"Delphes LCG view setup script does not exist: {lcg_view}")
    command = "\n".join(
        [
            f"source {shlex.quote(str(lcg_view))}",
            f"export DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"export LD_LIBRARY_PATH={shlex.quote(str(delphes_dir))}:$LD_LIBRARY_PATH",
            "export HIGGS_CEP_DELPHES_ANALYZER_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        ]
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


def import_libraries():
    import matplotlib
    import numpy as np
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return np, plt, yaml


def parse_range(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise RuntimeError(f"Invalid --mass-range '{value}', expected low,high")
    low, high = float(parts[0]), float(parts[1])
    if high <= low:
        raise RuntimeError("--mass-range high value must be larger than low value")
    return low, high


def parse_channel_weights(entries):
    weights = {process: 1.0 for process in PROCESS_ORDER}
    specified = set()
    for entry in entries:
        if "=" not in entry:
            raise RuntimeError(
                f"Invalid --channel-weight '{entry}', expected CHANNEL=FACTOR"
            )
        channel, value = (part.strip() for part in entry.split("=", 1))
        if channel not in weights:
            raise RuntimeError(
                f"Unknown channel '{channel}'; choose from {', '.join(PROCESS_ORDER)}"
            )
        if channel in specified:
            raise RuntimeError(f"Duplicate --channel-weight for {channel}")
        try:
            factor = float(value)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid weight factor for {channel}: '{value}'"
            ) from exc
        if not math.isfinite(factor) or factor < 0.0:
            raise RuntimeError(f"Weight factor for {channel} must be finite and >= 0")
        weights[channel] = factor
        specified.add(channel)
    return weights


def load_mva_cache(np, output_dir):
    dataset_path, scores_path = output_dir / "dataset.npz", output_dir / "scores.npz"
    if not dataset_path.is_file():
        raise RuntimeError(f"Missing MVA dataset cache: {dataset_path}")
    if not scores_path.is_file():
        raise RuntimeError(f"Missing MVA scores cache: {scores_path}")
    dataset = np.load(dataset_path, allow_pickle=False)
    scores = np.load(scores_path, allow_pickle=False)
    required = {
        "y", "physical_weight", "process", "mx", "yx", "xi_left", "xi_right",
        "dijet_mass", "dijet_rapidity", "proton_source",
    }
    missing = sorted(required.difference(dataset.files))
    if missing:
        raise RuntimeError(
            f"{dataset_path} is missing cached PPS fields ({', '.join(missing)}); "
            "rerun run_dijet_mva.py"
        )
    if "all" not in scores.files:
        raise RuntimeError(f"{scores_path} does not contain scores['all']; rerun run_dijet_mva.py")
    if scores["all"].shape[0] != dataset["y"].shape[0]:
        raise RuntimeError("MVA dataset and score row counts do not match")
    return dataset, scores


def cached_data(np, dataset, scores, mass_range):
    low, high = mass_range
    selected = (dataset["mx"] >= low) & (dataset["mx"] <= high)
    if not np.any(selected):
        raise RuntimeError(f"No cached events in mass range {low:g},{high:g} GeV")
    return {
        "mass": dataset["mx"][selected],
        "rapidity": dataset["yx"][selected],
        "score": scores["all"][selected],
        "weight": dataset["physical_weight"][selected],
        "label": dataset["y"][selected],
        "process": dataset["process"][selected],
        "proton_source": dataset["proton_source"][selected],
    }


def apply_channel_weights(np, data, channel_weights):
    weighted = dict(data)
    weighted["weight"] = data["weight"].copy()
    summary = {}
    for channel, factor in channel_weights.items():
        mask = data["process"] == channel
        before = float(np.sum(data["weight"][mask]))
        weighted["weight"][mask] *= factor
        summary[channel] = {
            "factor": factor,
            "events": int(np.sum(mask)),
            "yield_before": before,
            "yield_after": float(np.sum(weighted["weight"][mask])),
        }
    return weighted, summary


def scan_significance(np, data, mass_bins, score_cuts):
    results = []
    for cut in score_cuts:
        selected = data["score"] >= cut
        signal = selected & (data["label"] == 1)
        background = selected & (data["label"] == 0)
        signal_counts, _ = np.histogram(
            data["mass"][signal], bins=mass_bins, weights=data["weight"][signal]
        )
        background_counts, _ = np.histogram(
            data["mass"][background], bins=mass_bins, weights=data["weight"][background]
        )
        denominator = signal_counts + background_counts
        z_bins = np.zeros_like(signal_counts, dtype=np.float64)
        nonzero = denominator > 0.0
        z_bins[nonzero] = signal_counts[nonzero] / np.sqrt(denominator[nonzero])
        results.append(
            {
                "score_cut": float(cut),
                "significance": float(np.sqrt(np.sum(z_bins * z_bins))),
                "signal_yield": float(np.sum(signal_counts)),
                "background_yield": float(np.sum(background_counts)),
                "signal_events": int(np.sum(signal)),
                "background_events": int(np.sum(background)),
            }
        )
    return results


def write_scan_csv(path, results):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "score_cut", "significance", "signal_yield", "background_yield",
                "signal_events", "background_events",
            ),
        )
        writer.writeheader()
        writer.writerows(results)


def plot_scan(plt, path, results, best):
    cuts = [result["score_cut"] for result in results]
    significances = [result["significance"] for result in results]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(cuts, significances, color="#0072B2", linewidth=1.6)
    ax.axvline(best["score_cut"], color="#D55E00", linestyle="--", linewidth=1.2)
    ax.scatter([best["score_cut"]], [best["significance"]], color="#D55E00", zorder=3)
    ax.set_xlabel("XGBoost score cut")
    ax.set_ylabel("Combined binned S/sqrt(S+B)")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.05,
        0.95,
        f"best cut = {best['score_cut']:.3g}\nbest Z = {best['significance']:.3g}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mass_before_after(np, plt, path, data, mass_bins, best, log_y=False):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    selections = (
        ("Signal before", data["label"] == 1, "#0072B2", "-"),
        (
            "Signal after", (data["label"] == 1) & (data["score"] >= best["score_cut"]),
            "#0072B2", "--",
        ),
        ("Background before", data["label"] == 0, "#D55E00", "-"),
        (
            "Background after",
            (data["label"] == 0) & (data["score"] >= best["score_cut"]),
            "#D55E00", "--",
        ),
    )
    positive_counts = []
    for label, mask, color, linestyle in selections:
        weights = data["weight"][mask]
        counts, _, _ = ax.hist(
            data["mass"][mask],
            bins=mass_bins,
            weights=weights,
            histtype="step",
            linewidth=1.5,
            linestyle=linestyle,
            color=color,
            label=f"{label} ({np.sum(weights):.3g})",
        )
        positive_counts.extend(counts[counts > 0.0])
    ax.set_xlabel("$M_X$ [GeV]")
    ax.set_ylabel("Expected events / GeV")
    ax.set_xlim(float(mass_bins[0]), float(mass_bins[-1]))
    if log_y:
        ax.set_yscale("log")
        if positive_counts:
            ax.set_ylim(bottom=float(np.min(positive_counts)) * 0.5)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none"))
    legend_labels.append(f"Best-cut mass-binned Z = {best['significance']:.3g}")
    ax.legend(handles, legend_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def binned_significance(np, data, selection, mass_bins):
    signal = selection & (data["label"] == 1)
    background = selection & (data["label"] == 0)
    signal_counts, _ = np.histogram(
        data["mass"][signal], bins=mass_bins, weights=data["weight"][signal]
    )
    background_counts, _ = np.histogram(
        data["mass"][background], bins=mass_bins, weights=data["weight"][background]
    )
    denominator = signal_counts + background_counts
    nonzero = denominator > 0.0
    z_bins = np.zeros_like(signal_counts, dtype=np.float64)
    z_bins[nonzero] = signal_counts[nonzero] / np.sqrt(denominator[nonzero])
    return float(np.sqrt(np.sum(z_bins * z_bins)))


def sorted_processes_by_yield(np, data, selection):
    entries = []
    for process in PROCESS_ORDER:
        mask = selection & (data["process"] == process)
        if np.any(mask):
            entries.append((process, float(np.sum(data["weight"][mask])), bool(np.any(data["label"][mask] == 1))))
    return [item[0] for item in sorted(entries, key=lambda item: (0 if item[2] else 1, -item[1]))]


def plot_mass_by_process(np, plt, path, data, mass_bins, selection, title=None):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    significance = binned_significance(np, data, selection, mass_bins)
    for process in sorted_processes_by_yield(np, data, selection):
        mask = selection & (data["process"] == process)
        weights = data["weight"][mask]
        ax.hist(
            data["mass"][mask],
            bins=mass_bins,
            weights=weights,
            histtype="step",
            linewidth=1.5,
            linestyle="-" if np.any(data["label"][mask] == 1) else "--",
            color=COLORS[process],
            label=f"{PROCESS_LABELS[process]} ({np.sum(weights):.3g})",
        )
    if title:
        ax.set_title(title)
    ax.set_xlabel("$M_X$ [GeV]")
    ax.set_ylabel("Expected events / GeV")
    ax.set_xlim(float(mass_bins[0]), float(mass_bins[-1]))
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none"))
    labels.append(f"Z = {significance:.3g}")
    ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sample_summary(np, data):
    samples = []
    for process in PROCESS_ORDER:
        mask = data["process"] == process
        if not np.any(mask):
            continue
        sources = sorted(str(value) for value in np.unique(data["proton_source"][mask]))
        samples.append(
            {
                "process": process,
                "label": PROCESS_LABELS[process],
                "events": int(np.sum(mask)),
                "expected_yield": float(np.sum(data["weight"][mask])),
                "proton_sources": sources,
            }
        )
    return samples


def write_summary(
    np, yaml, path, args, mass_range, data, channel_weights, best, outputs,
):
    after = data["score"] >= best["score_cut"]
    summary = {
        "flavor": args.flavor,
        "proton_observables": "cached_from_run_dijet_mva",
        "mass_range_gev": list(mass_range),
        "scan_channel_weights": channel_weights,
        "best_score_cut": best["score_cut"],
        "best_significance": best["significance"],
        "best_signal_yield_in_mass_range": best["signal_yield"],
        "best_background_yield_in_mass_range": best["background_yield"],
        "best_signal_events_in_mass_range": best["signal_events"],
        "best_background_events_in_mass_range": best["background_events"],
        "before_signal_yield": float(np.sum(data["weight"][data["label"] == 1])),
        "before_background_yield": float(np.sum(data["weight"][data["label"] == 0])),
        "after_signal_yield": float(np.sum(data["weight"][after & (data["label"] == 1)])),
        "after_background_yield": float(np.sum(data["weight"][after & (data["label"] == 0)])),
        "samples": sample_summary(np, data),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def main():
    args = parse_args()
    if args.mass_bin_width <= 0.0:
        raise RuntimeError("--mass-bin-width must be > 0")
    if args.score_step <= 0.0 or args.score_step > 1.0:
        raise RuntimeError("--score-step must be in (0, 1]")
    ensure_delphes_python_runtime()
    np, plt, yaml = import_libraries()
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else ROOT / "analysis/MVA/output/mva_bb"
    )
    mass_range = parse_range(args.mass_range)
    channel_weights = parse_channel_weights(args.channel_weight)
    dataset, scores = load_mva_cache(np, output_dir)
    data = cached_data(np, dataset, scores, mass_range)
    data, channel_weight_summary = apply_channel_weights(np, data, channel_weights)
    for channel in PROCESS_ORDER:
        item = channel_weight_summary[channel]
        print(
            f"Scan channel weight: {channel} x {item['factor']:.6g}, "
            f"yield={item['yield_before']:.6g} -> {item['yield_after']:.6g}"
        )
    mass_bins = np.arange(
        mass_range[0], mass_range[1] + 0.5 * args.mass_bin_width, args.mass_bin_width
    )
    score_cuts = np.arange(0.8, 0.98 + 0.5 * args.score_step, args.score_step)
    results = scan_significance(np, data, mass_bins, score_cuts)
    best = max(results, key=lambda result: result["significance"])
    print(f"Best XGBoost score cut: {best['score_cut']:.6g}", flush=True)
    print(f"Best combined significance: {best['significance']:.6g}", flush=True)
    print(
        "Best-cut expected yields: "
        f"signal={best['signal_yield']:.6g}, "
        f"background={best['background_yield']:.6g}",
        flush=True,
    )
    print(
        "Best-cut actual MC events: "
        f"signal={best['signal_events']}, background={best['background_events']}",
        flush=True,
    )

    outputs = {
        "scan_csv": output_dir / "pp_mass_significance_scan.csv",
        "summary": output_dir / "pp_mass_significance_summary.yaml",
        "scan_plot": output_dir / "pp_mass_significance_scan.png",
        "mass_before_after": output_dir / "pp_mass_smeared_before_after_mva.png",
        "mass_before_after_log": output_dir / "pp_mass_smeared_before_after_mva_log.png",
        "mass_before_by_process": output_dir / "pp_mass_smeared_before_mva_by_process.png",
        "mass_after_by_process": output_dir / "pp_mass_smeared_after_mva_by_process.png",
    }
    write_scan_csv(outputs["scan_csv"], results)
    plot_scan(plt, outputs["scan_plot"], results, best)
    plot_mass_before_after(np, plt, outputs["mass_before_after"], data, mass_bins, best)
    plot_mass_before_after(
        np, plt, outputs["mass_before_after_log"], data, mass_bins, best, log_y=True
    )
    all_events = np.ones(data["mass"].shape, dtype=bool)
    plot_mass_by_process(np, plt, outputs["mass_before_by_process"], data, mass_bins, all_events)
    plot_mass_by_process(
        np,
        plt,
        outputs["mass_after_by_process"],
        data,
        mass_bins,
        data["score"] >= best["score_cut"],
        f"XGBoost score >= {best['score_cut']:.3g}",
    )
    write_summary(
        np, yaml, outputs["summary"], args, mass_range, data,
        channel_weight_summary, best, outputs
    )
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
