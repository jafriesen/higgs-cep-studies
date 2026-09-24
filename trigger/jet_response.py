#!/usr/bin/env python3
"""Validate PU200 Delphes JetPUPPI response against hard-interaction GenJets."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


PT_BINS = np.array([5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 500], dtype=float)
ETA_BINS = np.array([0.0, 0.8, 1.6, 2.4], dtype=float)


def load_delphes_root():
    """Import ROOT with the Delphes class dictionary loaded."""
    import ROOT

    delphes_dir = Path(
        os.environ.get("DELPHES_DIR", Path(__file__).resolve().parents[2] / "delphes")
    )
    library = delphes_dir / "libDelphes.so"
    if not library.is_file() or ROOT.gSystem.Load(str(library)) < 0:
        raise RuntimeError(f"Could not load Delphes ROOT dictionary: {library}")
    return ROOT


def delta_phi(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


def delta_r(first, second):
    return math.hypot(
        first["eta"] - second["eta"], delta_phi(first["phi"], second["phi"])
    )


def read_jets(collection, eta_max=2.4):
    return [
        {
            "pt": float(collection.At(index).PT),
            "eta": float(collection.At(index).Eta),
            "phi": float(collection.At(index).Phi),
            "mass": float(collection.At(index).Mass),
        }
        for index in range(collection.GetEntriesFast())
        if abs(float(collection.At(index).Eta)) < eta_max
    ]


def match_jets(gen_jets, reco_jets, max_delta_r=0.2):
    candidates = sorted(
        (delta_r(gen, reco), gen_index, reco_index)
        for gen_index, gen in enumerate(gen_jets)
        for reco_index, reco in enumerate(reco_jets)
        if delta_r(gen, reco) < max_delta_r
    )
    matches = []
    used_gen, used_reco = set(), set()
    for distance, gen_index, reco_index in candidates:
        if gen_index in used_gen or reco_index in used_reco:
            continue
        used_gen.add(gen_index)
        used_reco.add(reco_index)
        matches.append((gen_index, reco_index, distance))
    return matches


def four_vector(jet):
    pt, eta, phi, mass = (jet[key] for key in ("pt", "eta", "phi", "mass"))
    px, py, pz = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
    energy = math.sqrt(max(mass * mass + px * px + py * py + pz * pz, 0.0))
    return px, py, pz, energy


def dijet_rapidity(first, second):
    vectors = [four_vector(first), four_vector(second)]
    pz = sum(vector[2] for vector in vectors)
    energy = sum(vector[3] for vector in vectors)
    if energy <= abs(pz):
        return math.copysign(math.inf, pz)
    return 0.5 * math.log((energy + pz) / (energy - pz))


def binned_summary(rows, pt_bins=PT_BINS, eta_bins=ETA_BINS, threshold=15.0):
    output = []
    for eta_low, eta_high in zip(eta_bins[:-1], eta_bins[1:]):
        for pt_low, pt_high in zip(pt_bins[:-1], pt_bins[1:]):
            gen = [
                row
                for row in rows
                if eta_low <= abs(row["gen_eta"]) < eta_high
                and pt_low <= row["gen_pt"] < pt_high
            ]
            responses = np.asarray(
                [row["response"] for row in gen if row["matched"]], dtype=float
            )
            passing = sum(row["matched"] and row["reco_pt"] > threshold for row in gen)
            median = float(np.median(responses)) if responses.size else None
            q16, q84 = (
                np.quantile(responses, [0.16, 0.84]).tolist()
                if responses.size
                else (None, None)
            )
            output.append(
                {
                    "pt_range_gev": [float(pt_low), float(pt_high)],
                    "abs_eta_range": [float(eta_low), float(eta_high)],
                    "gen_jets": len(gen),
                    "matched_jets": int(responses.size),
                    "matching_efficiency": float(responses.size / len(gen))
                    if gen
                    else None,
                    "threshold_efficiency": float(passing / len(gen)) if gen else None,
                    "median_response": median,
                    "response_q16": q16,
                    "response_q84": q84,
                    "relative_central68_resolution": (
                        float(0.5 * (q84 - q16) / median)
                        if median and q16 is not None
                        else None
                    ),
                }
            )
    return output


def analyze(path, max_events=None, max_delta_r=0.2, threshold=15.0, progress=False):
    ROOT = load_delphes_root()

    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("Delphes") if root_file else None
    if not tree or not tree.GetBranch("GenJet") or not tree.GetBranch("JetPUPPI"):
        raise RuntimeError(f"GenJet/JetPUPPI branches missing from {path}")
    entries = int(tree.GetEntries())
    if max_events is not None:
        entries = min(entries, int(max_events))
    rows, dijet_dy = [], []
    progress_every = max(1, entries // 10)
    for entry in range(entries):
        tree.GetEntry(entry)
        gen_jets = read_jets(tree.GenJet)
        reco_jets = read_jets(tree.JetPUPPI)
        matches = match_jets(gen_jets, reco_jets, max_delta_r)
        by_gen = {
            gen_index: (reco_index, distance)
            for gen_index, reco_index, distance in matches
        }
        for gen_index, gen in enumerate(gen_jets):
            matched = gen_index in by_gen
            reco = reco_jets[by_gen[gen_index][0]] if matched else None
            rows.append(
                {
                    "gen_pt": gen["pt"],
                    "gen_eta": gen["eta"],
                    "matched": matched,
                    "reco_pt": reco["pt"] if reco else 0.0,
                    "response": reco["pt"] / gen["pt"]
                    if reco and gen["pt"] > 0.0
                    else None,
                }
            )
        leading_gen = sorted(
            enumerate(gen_jets), key=lambda item: item[1]["pt"], reverse=True
        )[:2]
        if len(leading_gen) == 2 and all(
            index in by_gen for index, _jet in leading_gen
        ):
            reco_pair = [reco_jets[by_gen[index][0]] for index, _jet in leading_gen]
            gen_pair = [jet for _index, jet in leading_gen]
            dijet_dy.append(dijet_rapidity(*reco_pair) - dijet_rapidity(*gen_pair))
        if progress and ((entry + 1) % progress_every == 0 or entry + 1 == entries):
            print(f"  response events {entry + 1:,}/{entries:,}", flush=True)
    root_file.Close()
    dy = np.asarray(dijet_dy, dtype=float)
    return {
        "input": str(Path(path).resolve()),
        "events": entries,
        "match_delta_r": float(max_delta_r),
        "jet_threshold_gev": float(threshold),
        "bins": binned_summary(rows, threshold=threshold),
        "dijet_rapidity": {
            "matched_events": int(dy.size),
            "median_delta_y": float(np.median(dy)) if dy.size else None,
            "central68_half_width": float(
                0.5 * np.diff(np.quantile(dy, [0.16, 0.84]))[0]
            )
            if dy.size
            else None,
        },
    }


def make_plots(report, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for field, ylabel, filename in (
        ("median_response", "median JetPUPPI / GenJet pT", "response.png"),
        (
            "relative_central68_resolution",
            "central 68% half-width / median",
            "resolution.png",
        ),
        ("threshold_efficiency", "P(JetPUPPI pT > 15 GeV)", "turnon.png"),
    ):
        figure, axis = plt.subplots(figsize=(7, 5))
        eta_ranges = sorted({tuple(row["abs_eta_range"]) for row in report["bins"]})
        for eta_range in eta_ranges:
            selected = [
                row
                for row in report["bins"]
                if tuple(row["abs_eta_range"]) == eta_range and row[field] is not None
            ]
            x = [
                math.sqrt(row["pt_range_gev"][0] * row["pt_range_gev"][1])
                for row in selected
            ]
            axis.plot(
                x,
                [row[field] for row in selected],
                marker="o",
                label=f"{eta_range[0]:g} < |eta| < {eta_range[1]:g}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("GenJet pT [GeV]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--match-dr", type=float, default=0.2)
    parser.add_argument("--threshold-gev", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.campaign_dir / "jet_response"
    print(
        f"Reading raw JetPUPPI and GenJet collections from {args.campaign_dir}...",
        flush=True,
    )
    report = analyze(
        args.campaign_dir / "delphes.root",
        args.max_events,
        args.match_dr,
        args.threshold_gev,
        True,
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_plots(report, output)
    print(f"Wrote response report and plots to {output.resolve()}")


if __name__ == "__main__":
    main()
