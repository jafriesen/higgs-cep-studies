#!/usr/bin/env python3
"""Compare leading JetPUPPI kinematics in H(bb) and HardQCD PU200 samples."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import jet_calibration
from common.config_utils import natural_key
from minbias.artifact import read_json
from trigger.dijet_rate import load_jet_correction
from trigger.jet_response import load_delphes_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_ROOT = (
    ROOT / "output-superchic/Hbb/Hbb__v01/sim-Delphes/Hbb_noFSR_200PU__v01/root"
)
DEFAULT_BACKGROUND_CAMPAIGN = ROOT / "output/trigger/hardqcd_pthat10_pilot"
DEFAULT_OUTPUT = ROOT / "trigger/output/trigger_jet_kinematics"
DEFAULT_CORRECTIONS = ROOT / "jet-energy/output/stage2_3/corrections.yaml"
COLORS = {"signal": "#0072B2", "background": "#D55E00"}


def leading_subleading_pt(pt, eta, eta_max):
    """Return the two leading accepted jet pT values, or ``None``."""
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    accepted = pt[np.abs(eta) < eta_max]
    if accepted.size < 2:
        return None
    leading = np.partition(accepted, -2)[-2:]
    leading.sort()
    return float(leading[1]), float(leading[0])


def corrected_jet_pt(pt, eta, correction_map):
    """Scale jet pT with the sample's own map and drop unsupported jets."""
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    if pt.size == 0:
        return pt, eta
    factors, valid = jet_calibration.correction_factors(pt, eta, correction_map)
    return pt[valid] * factors[valid], eta[valid]


def load_sample(files, *, max_events, eta_max, label, root_module, correction_map):
    """Load the two leading accepted, corrected JetPUPPI jets from Delphes files."""
    leading = []
    subleading = []
    loaded = 0
    selected = 0
    used_files = []
    for file_index, path in enumerate(files, start=1):
        remaining = None if max_events is None else max_events - loaded
        if remaining is not None and remaining <= 0:
            break
        print(f"  {label} file {file_index}/{len(files)}: {path.name}", flush=True)
        root_file = root_module.TFile.Open(str(path))
        tree = root_file.Get("Delphes") if root_file else None
        if not tree or not tree.GetBranch("JetPUPPI"):
            if root_file:
                root_file.Close()
            raise RuntimeError(f"Delphes JetPUPPI branch missing from {path}")
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("JetPUPPI*", 1)
        n_entries = int(tree.GetEntries())
        n_read = min(n_entries, remaining) if remaining is not None else n_entries
        for entry in range(n_read):
            tree.GetEntry(entry)
            collection = tree.JetPUPPI
            pts = []
            etas = []
            for jet_index in range(collection.GetEntriesFast()):
                jet = collection.At(jet_index)
                pts.append(float(jet.PT))
                etas.append(float(jet.Eta))
            pts, etas = corrected_jet_pt(pts, etas, correction_map)
            pair = leading_subleading_pt(pts, etas, eta_max)
            if pair is not None:
                leading.append(pair[0])
                subleading.append(pair[1])
                selected += 1
        root_file.Close()
        loaded += n_read
        used_files.append(str(path.resolve()))

    if loaded == 0:
        raise RuntimeError(f"No events loaded for {label}")
    if selected == 0:
        raise RuntimeError(f"No {label} events contain two accepted JetPUPPI jets")
    return {
        "label": label,
        "files": used_files,
        "events_loaded": loaded,
        "events_with_two_jets": selected,
        "leading_pt_gev": np.asarray(leading, dtype=np.float64),
        "subleading_pt_gev": np.asarray(subleading, dtype=np.float64),
    }


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, (0.01, 0.05, 0.5, 0.95, 0.99))
    return {
        "entries": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "quantiles": {
            "0.01": float(quantiles[0]),
            "0.05": float(quantiles[1]),
            "0.50": float(quantiles[2]),
            "0.95": float(quantiles[3]),
            "0.99": float(quantiles[4]),
        },
    }


def shared_range(samples, field):
    values = np.concatenate([sample[field] for sample in samples.values()])
    low = min(0.0, float(np.min(values)))
    high = float(np.max(values))
    if high <= low:
        high = low + 1.0
    return low, high * 1.02


def plot_distribution(samples, field, xlabel, output, bins, log_y):
    value_range = shared_range(samples, field)
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    positive = []
    for key, sample in samples.items():
        values = sample[field]
        counts, _ = np.histogram(values, bins=edges)
        fractions = counts / values.size
        errors = np.sqrt(counts) / values.size
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.stairs(
            fractions,
            edges,
            color=COLORS[key],
            linewidth=1.6,
            label=f"{sample['label']} (n={values.size:,})",
        )
        nonzero = counts > 0
        ax.errorbar(
            centers[nonzero],
            fractions[nonzero],
            yerr=errors[nonzero],
            fmt="none",
            ecolor=COLORS[key],
            elinewidth=0.9,
            capsize=1.2,
        )
        positive.extend(fractions[fractions > 0.0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of selected events / bin")
    ax.set_xlim(value_range)
    if log_y and positive:
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(positive) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output}")


def signal_files(root_dir, max_files):
    files = sorted(Path(root_dir).resolve().glob("*.root"), key=natural_key)
    files = files[:max_files]
    if not files:
        raise RuntimeError(f"No signal ROOT files found in {root_dir}")
    return files


def background_file(campaign):
    campaign = Path(campaign).resolve()
    metadata = read_json(campaign / "metadata.json")
    path = Path(metadata["delphes"]["path"])
    if not path.is_file():
        raise RuntimeError(f"HardQCD Delphes file not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-root-dir", type=Path, default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--signal-max-files", type=int, default=5)
    parser.add_argument(
        "--background-campaign", type=Path, default=DEFAULT_BACKGROUND_CAMPAIGN
    )
    parser.add_argument("--max-events", type=int, default=10_000)
    parser.add_argument("--eta-max", type=float, default=2.4)
    parser.add_argument("--jet-corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument(
        "--signal-fsr-state",
        choices=("FSR", "noFSR"),
        default="noFSR",
        help="FSR state of --signal-root-dir; selects the H(bb) correction map.",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.signal_max_files <= 0 or args.max_events <= 0 or args.bins <= 0:
        parser.error("file, event, and bin limits must be positive")
    if args.eta_max <= 0.0:
        parser.error("--eta-max must be positive")
    return args


def main():
    args = parse_args()
    root_module = load_delphes_root()
    signal_paths = signal_files(args.signal_root_dir, args.signal_max_files)
    background_path = background_file(args.background_campaign)
    signal_map, signal_info = load_jet_correction(
        args.jet_corrections, fsr_state=args.signal_fsr_state, sample="Hbb"
    )
    background_map, background_info = load_jet_correction(
        args.jet_corrections, sample="HardQCD"
    )
    print("Loading corrected H(bb) JetPUPPI jets...", flush=True)
    signal = load_sample(
        signal_paths,
        max_events=args.max_events,
        eta_max=args.eta_max,
        label=f"H(bb) {args.signal_fsr_state} + PU200",
        root_module=root_module,
        correction_map=signal_map,
    )
    print("Loading corrected HardQCD JetPUPPI jets...", flush=True)
    background = load_sample(
        [background_path],
        max_events=args.max_events,
        eta_max=args.eta_max,
        label="HardQCD pTHat > 10 GeV + PU200",
        root_module=root_module,
        correction_map=background_map,
    )
    samples = {"signal": signal, "background": background}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution(
        samples,
        "leading_pt_gev",
        r"Leading JetPUPPI $p_T$ [GeV]",
        output_dir / "leading_jet_pt.png",
        args.bins,
        args.log_y,
    )
    plot_distribution(
        samples,
        "subleading_pt_gev",
        r"Subleading JetPUPPI $p_T$ [GeV]",
        output_dir / "subleading_jet_pt.png",
        args.bins,
        args.log_y,
    )
    report = {
        "configuration": {
            "collection": "JetPUPPI",
            "signal_correction": signal_info,
            "background_correction": background_info,
            "eta_max": float(args.eta_max),
            "additional_pt_cut_gev": None,
            "signal_max_files": int(args.signal_max_files),
            "max_events_per_sample": int(args.max_events),
            "bins": int(args.bins),
        },
        "samples": {},
    }
    for key, sample in samples.items():
        report["samples"][key] = {
            "label": sample["label"],
            "files": sample["files"],
            "events_loaded": int(sample["events_loaded"]),
            "events_with_two_jets": int(sample["events_with_two_jets"]),
            "two_jet_fraction": float(
                sample["events_with_two_jets"] / sample["events_loaded"]
            ),
            "leading_pt_gev": distribution_summary(sample["leading_pt_gev"]),
            "subleading_pt_gev": distribution_summary(sample["subleading_pt_gev"]),
        }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote summary: {summary_path}")
    for sample in report["samples"].values():
        print(
            f"{sample['label']}: two jets={sample['events_with_two_jets']:,}/"
            f"{sample['events_loaded']:,} ({sample['two_jet_fraction']:.3%}); "
            "median leading/subleading pT="
            f"{sample['leading_pt_gev']['quantiles']['0.50']:.2f}/"
            f"{sample['subleading_pt_gev']['quantiles']['0.50']:.2f} GeV"
        )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
