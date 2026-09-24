#!/usr/bin/env python3
"""Compare additional-jet mass recovery in the Delphes 200PU H->bb samples."""

import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "output-superchic/Hbb/Hbb__v01/sim-Delphes"
DEFAULT_NO_FSR = CAMPAIGN / "Hbb_noFSR_200PU__v01/root"
DEFAULT_FSR = CAMPAIGN / "Hbb_FSR_200PU__v01/root"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/delphes_200PU"
ADDITIONAL_JET_PT_THRESHOLDS = (2.0, 5.0, 10.0)
STORED_JET_PT_MIN = 5.0
TRUTH_LEADING_PT_MIN = 20.0
TRUTH_SUBLEADING_PT_MIN = 15.0
LOCAL_JET_PT_MIN = 10.0
LOCAL_JET_DR_MAX = 0.8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare leading-dijet masses with additional stored R=0.4 PUPPI jets "
            "in the FSR and no-FSR Delphes 200PU samples."
        )
    )
    parser.add_argument("--no-fsr-dir", type=Path, default=DEFAULT_NO_FSR)
    parser.add_argument("--fsr-dir", type=Path, default=DEFAULT_FSR)
    parser.add_argument("--collection", default="JetPUPPI")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=2000, help="Events read per dataset")
    parser.add_argument("--main-jet-pt-min", type=float, default=20.0)
    parser.add_argument("--match-dr-max", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_FSR_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_DELPHES_FSR_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def natural_key(path):
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", path.name)]


def input_files(directory, max_files):
    directory = directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Input directory does not exist: {directory}")
    files = sorted(directory.glob("*.root"), key=natural_key)
    if not files:
        raise RuntimeError(f"No ROOT files found in {directory}")
    return files[:max_files]


def jet_p4(jet):
    pt = float(jet.PT)
    eta = float(jet.Eta)
    phi = float(jet.Phi)
    mass = float(jet.Mass)
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = math.sqrt(px * px + py * py + pz * pz + mass * mass)
    return px, py, pz, energy


def invariant_mass(jets):
    px = sum(jet["p4"][0] for jet in jets)
    py = sum(jet["p4"][1] for jet in jets)
    pz = sum(jet["p4"][2] for jet in jets)
    energy = sum(jet["p4"][3] for jet in jets)
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def delta_r(eta1, phi1, eta2, phi2):
    delta_phi = math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2))
    return math.hypot(eta1 - eta2, delta_phi)


def match_truth_bottoms(particles, jets, max_delta_r):
    b_quarks = [
        particle for particle in particles
        if int(particle.PID) == 5 and int(particle.Status) == 23 and not int(particle.IsPU)
    ]
    bbar_quarks = [
        particle for particle in particles
        if int(particle.PID) == -5 and int(particle.Status) == 23 and not int(particle.IsPU)
    ]
    if len(b_quarks) != 1 or len(bbar_quarks) != 1 or len(jets) < 2:
        return None, math.nan, math.nan

    best = None
    for b_index in range(len(jets)):
        for bbar_index in range(len(jets)):
            if b_index == bbar_index:
                continue
            b_distance = delta_r(
                float(b_quarks[0].Eta), float(b_quarks[0].Phi),
                jets[b_index]["eta"], jets[b_index]["phi"],
            )
            bbar_distance = delta_r(
                float(bbar_quarks[0].Eta), float(bbar_quarks[0].Phi),
                jets[bbar_index]["eta"], jets[bbar_index]["phi"],
            )
            candidate = (b_distance + bbar_distance, b_index, bbar_index, b_distance, bbar_distance)
            if best is None or candidate < best:
                best = candidate
    if best[3] >= max_delta_r or best[4] >= max_delta_r:
        return None, best[3], best[4]
    return (best[1], best[2]), best[3], best[4]


def add_mass_combinations(row, jets, pair, mass_key, count_key):
    row[mass_key] = invariant_mass([jets[index] for index in pair]) if pair is not None else math.nan
    for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
        label = f"pt{int(pt_min)}"
        additional = (
            [
                jet for index, jet in enumerate(jets)
                if index not in pair and jet["pt"] >= pt_min
            ]
            if pair is not None
            else []
        )
        row[f"{count_key}_{label}"] = len(additional)
        row[f"{mass_key}_plus_jets_{label}"] = (
            invariant_mass([jets[index] for index in pair] + additional)
            if pair is not None
            else math.nan
        )


def add_truth_local_recovery(row, jets, truth_pair):
    selected_pair = None
    if truth_pair is not None:
        selected_pair = tuple(sorted(truth_pair, key=lambda index: jets[index]["pt"], reverse=True))
        if (
            jets[selected_pair[0]]["pt"] <= TRUTH_LEADING_PT_MIN
            or jets[selected_pair[1]]["pt"] <= TRUTH_SUBLEADING_PT_MIN
        ):
            selected_pair = None

    row["m_truth_matched_pt20_15"] = (
        invariant_mass([jets[index] for index in selected_pair])
        if selected_pair is not None
        else math.nan
    )
    row["truth_local_candidate_count"] = 0
    row["truth_local_jet_found"] = 0
    row["truth_local_jet_pt"] = math.nan
    row["truth_local_jet_dr"] = math.nan
    row["m_truth_matched_pt20_15_plus_local_jet"] = math.nan
    if selected_pair is None:
        return

    candidates = []
    for index, jet in enumerate(jets):
        if index in selected_pair or jet["pt"] <= LOCAL_JET_PT_MIN:
            continue
        distance = min(
            delta_r(jet["eta"], jet["phi"], jets[pair_index]["eta"], jets[pair_index]["phi"])
            for pair_index in selected_pair
        )
        if distance < LOCAL_JET_DR_MAX:
            candidates.append((index, distance))

    row["truth_local_candidate_count"] = len(candidates)
    recovered_jets = [jets[index] for index in selected_pair]
    if candidates:
        recovery_index, recovery_distance = candidates[0]
        recovered_jets.append(jets[recovery_index])
        row["truth_local_jet_found"] = 1
        row["truth_local_jet_pt"] = jets[recovery_index]["pt"]
        row["truth_local_jet_dr"] = recovery_distance
    row["m_truth_matched_pt20_15_plus_local_jet"] = invariant_mass(recovered_jets)


def analyze_dataset(ROOT, dataset, files, args):
    rows = []
    read_events = 0
    for path in files:
        if read_events >= args.max_events:
            break
        root_file = ROOT.TFile.Open(str(path))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not open ROOT file: {path}")
        tree = root_file.Get("Delphes")
        if not tree:
            root_file.Close()
            raise RuntimeError(f"Could not find Delphes tree in {path}")
        if not tree.GetBranch(args.collection):
            root_file.Close()
            raise RuntimeError(f"Could not find {args.collection} branch in {path}")
        if not tree.GetBranch("Particle"):
            root_file.Close()
            raise RuntimeError(f"Could not find Particle branch in {path}")

        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus(f"{args.collection}*", 1)
        tree.SetBranchStatus("Particle*", 1)
        entries = min(int(tree.GetEntries()), args.max_events - read_events)
        print(f"Reading {dataset}: {path.name} ({entries} events)")
        for entry in range(entries):
            tree.GetEntry(entry)
            collection = getattr(tree, args.collection)
            jets = [
                {
                    "pt": float(collection.At(index).PT),
                    "eta": float(collection.At(index).Eta),
                    "phi": float(collection.At(index).Phi),
                    "p4": jet_p4(collection.At(index)),
                }
                for index in range(collection.GetEntriesFast())
            ]
            jets.sort(key=lambda jet: jet["pt"], reverse=True)
            read_events += 1
            row = {
                "dataset": dataset,
                "file": path.name,
                "event": read_events,
                "n_stored_jets": len(jets),
                "leading_pt": jets[0]["pt"] if jets else math.nan,
                "subleading_pt": jets[1]["pt"] if len(jets) >= 2 else math.nan,
            }
            leading_pair = (
                (0, 1)
                if len(jets) >= 2 and jets[1]["pt"] >= args.main_jet_pt_min
                else None
            )
            add_mass_combinations(row, jets, leading_pair, "m_dijet", "n_additional")

            particles = [tree.Particle.At(index) for index in range(tree.Particle.GetEntriesFast())]
            truth_pair, b_distance, bbar_distance = match_truth_bottoms(
                particles, jets, args.match_dr_max
            )
            row["truth_match_found"] = int(truth_pair is not None)
            row["truth_match_dr_b"] = b_distance
            row["truth_match_dr_bbar"] = bbar_distance
            row["truth_matched_b_pt"] = jets[truth_pair[0]]["pt"] if truth_pair is not None else math.nan
            row["truth_matched_bbar_pt"] = jets[truth_pair[1]]["pt"] if truth_pair is not None else math.nan
            add_mass_combinations(
                row, jets, truth_pair, "m_truth_matched", "n_truth_additional"
            )

            truth_pair_pt20 = (
                truth_pair
                if truth_pair is not None
                and all(jets[index]["pt"] >= args.main_jet_pt_min for index in truth_pair)
                else None
            )
            add_mass_combinations(
                row,
                jets,
                truth_pair_pt20,
                "m_truth_matched_pt20",
                "n_truth_pt20_additional",
            )
            add_truth_local_recovery(row, jets, truth_pair)
            rows.append(row)
        root_file.Close()

    if read_events == 0:
        raise RuntimeError(f"No events read for {dataset}")
    return rows, read_events


def mean(rows, key):
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else math.nan


def median(rows, key):
    values = sorted(float(row[key]) for row in rows if math.isfinite(float(row[key])))
    if not values:
        return math.nan
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])


def make_summary(by_dataset, totals):
    result = []
    for dataset, rows in by_dataset.items():
        summary = {
            "dataset": dataset,
            "events_read": totals[dataset],
            "truth_matches": sum(row["truth_match_found"] for row in rows),
        }
        configurations = (
            ("leading", "m_dijet", "n_additional"),
            ("truth", "m_truth_matched", "n_truth_additional"),
            ("truth_pt20", "m_truth_matched_pt20", "n_truth_pt20_additional"),
        )
        for configuration, mass_key, count_key in configurations:
            selected = sum(math.isfinite(row[mass_key]) for row in rows)
            summary[f"events_{configuration}"] = selected
            summary[f"efficiency_{configuration}"] = selected / totals[dataset]
            summary[f"mean_{mass_key}"] = mean(rows, mass_key)
            summary[f"median_{mass_key}"] = median(rows, mass_key)
            for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
                label = f"pt{int(pt_min)}"
                additional_mass_key = f"{mass_key}_plus_jets_{label}"
                summary[f"mean_{additional_mass_key}"] = mean(rows, additional_mass_key)
                summary[f"median_{additional_mass_key}"] = median(rows, additional_mass_key)
                summary[f"mean_{count_key}_{label}"] = mean(rows, count_key + "_" + label)
        local_mass_key = "m_truth_matched_pt20_15"
        recovered_mass_key = "m_truth_matched_pt20_15_plus_local_jet"
        local_selected = sum(math.isfinite(row[local_mass_key]) for row in rows)
        recovery_count = sum(row["truth_local_jet_found"] for row in rows)
        summary["events_truth_pt20_15"] = local_selected
        summary["efficiency_truth_pt20_15"] = local_selected / totals[dataset]
        summary["events_truth_pt20_15_with_local_jet"] = recovery_count
        summary["local_jet_fraction"] = recovery_count / local_selected if local_selected else math.nan
        for statistic in (mean, median):
            name = statistic.__name__
            summary[f"{name}_{local_mass_key}"] = statistic(rows, local_mass_key)
            summary[f"{name}_{recovered_mass_key}"] = statistic(rows, recovered_mass_key)
        result.append(summary)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries, args):
    print()
    print(
        f"Stored {args.collection} anti-kt R=0.4 jets; main jets pT >= "
        f"{args.main_jet_pt_min:g} GeV"
    )
    configurations = (
        ("leading", "m_dijet", "Two leading jets after pT cut"),
        ("truth", "m_truth_matched", f"Truth-matched jets within dR<{args.match_dr_max:g}, before pT cut"),
        ("truth_pt20", "m_truth_matched_pt20", "Truth-matched jets after pT cut"),
    )
    for configuration, mass_key, title in configurations:
        print()
        print(title)
        print(f"{'sample':<6} {'stat':<6} {'selected':>10} {'dijet':>9} {'+jets>2':>9} {'+jets>5':>9} {'+jets>10':>10}")
        for row in summaries:
            for statistic in ("mean", "median"):
                print(
                    f"{row['dataset']:<6} {statistic:<6} "
                    f"{row[f'events_{configuration}']:>5}/{row['events_read']:<4} "
                    f"{row[f'{statistic}_{mass_key}']:>9.2f} "
                    f"{row[f'{statistic}_{mass_key}_plus_jets_pt2']:>9.2f} "
                    f"{row[f'{statistic}_{mass_key}_plus_jets_pt5']:>9.2f} "
                    f"{row[f'{statistic}_{mass_key}_plus_jets_pt10']:>10.2f}"
                )
    print(
        f"NOTE: the 2 and 5 GeV columns are identical because {args.collection} "
        f"was stored with JetPTMin={STORED_JET_PT_MIN:g} GeV."
    )
    print()
    print(
        f"Truth-matched jets: leading pT > {TRUTH_LEADING_PT_MIN:g} GeV, "
        f"subleading pT > {TRUTH_SUBLEADING_PT_MIN:g} GeV; add at most one "
        f"highest-pT remaining jet with pT > {LOCAL_JET_PT_MIN:g} GeV and "
        f"dR < {LOCAL_JET_DR_MAX:g} from either matched jet"
    )
    print(f"{'sample':<6} {'stat':<6} {'selected':>10} {'with add':>10} {'dijet':>9} {'+local':>9} {'change':>9}")
    for row in summaries:
        for statistic in ("mean", "median"):
            base = row[f"{statistic}_m_truth_matched_pt20_15"]
            recovered = row[f"{statistic}_m_truth_matched_pt20_15_plus_local_jet"]
            print(
                f"{row['dataset']:<6} {statistic:<6} "
                f"{row['events_truth_pt20_15']:>5}/{row['events_read']:<4} "
                f"{row['events_truth_pt20_15_with_local_jet']:>5} "
                f"({row['local_jet_fraction']:>4.1%}) "
                f"{base:>9.2f} {recovered:>9.2f} {recovered - base:>+9.2f}"
            )


def make_plot(np, plt, by_dataset, output_dir, mass_key, filename, title):
    configurations = (
        (mass_key, "Main dijet"),
        (f"{mass_key}_plus_jets_pt2", "Main pair + jets above 2 GeV (stored >=5 GeV)"),
        (f"{mass_key}_plus_jets_pt5", "Main pair + jets above 5 GeV"),
        (f"{mass_key}_plus_jets_pt10", "Main pair + jets above 10 GeV"),
    )
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for axis, (key, panel_title) in zip(axes.flat, configurations):
        combined = np.asarray(
            [row[key] for rows in by_dataset.values() for row in rows if math.isfinite(row[key])],
            dtype=np.float64,
        )
        upper = max(160.0, 50.0 * math.ceil(float(np.quantile(combined, 0.995)) / 50.0))
        bins = np.linspace(0.0, upper, 76)
        for dataset, rows in by_dataset.items():
            values = [row[key] for row in rows if math.isfinite(row[key])]
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[dataset],
                label=dataset,
            )
        axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(panel_title)
        axis.set_xlabel("Reconstructed mass [GeV]")
        axis.set_ylabel("Normalized events")
        axis.set_xlim(0.0, upper)
        axis.legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=160)
    plt.close(figure)


def make_local_recovery_plot(np, plt, by_dataset, output_dir):
    configurations = (
        ("m_truth_matched_pt20_15", "Truth-matched dijet"),
        ("m_truth_matched_pt20_15_plus_local_jet", "Dijet + at most one local jet"),
    )
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(0.0, 250.0, 76)
    for axis, (key, panel_title) in zip(axes, configurations):
        for dataset, rows in by_dataset.items():
            values = [row[key] for row in rows if math.isfinite(row[key])]
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[dataset],
                label=dataset,
            )
        axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(panel_title)
        axis.set_xlabel("Reconstructed mass [GeV]")
        axis.set_ylabel("Normalized events")
        axis.legend(frameon=False)
    figure.suptitle(
        "Delphes 200PU, truth-matched R=0.4 jets: local additional-jet recovery"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "delphes_truth_matched_local_jet.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.max_files <= 0 or args.max_events <= 0:
        raise RuntimeError("--max-files and --max-events must be positive")
    if args.main_jet_pt_min < STORED_JET_PT_MIN:
        raise RuntimeError(
            f"--main-jet-pt-min must be at least the stored jet minimum of "
            f"{STORED_JET_PT_MIN:g} GeV"
        )
    if args.match_dr_max <= 0.0:
        raise RuntimeError("--match-dr-max must be positive")

    import ROOT
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ROOT.gSystem.Load("libDelphes") < 0:
        raise RuntimeError("Could not load libDelphes")

    selected = {
        "noFSR": input_files(args.no_fsr_dir, args.max_files),
        "FSR": input_files(args.fsr_dir, args.max_files),
    }
    totals = {}
    by_dataset = {}
    for dataset, files in selected.items():
        by_dataset[dataset], totals[dataset] = analyze_dataset(ROOT, dataset, files, args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = by_dataset["noFSR"] + by_dataset["FSR"]
    summaries = make_summary(by_dataset, totals)
    write_csv(output_dir / "event_metrics.csv", all_rows)
    write_csv(output_dir / "summary.csv", summaries)
    print_summary(summaries, args)
    make_plot(
        np,
        plt,
        by_dataset,
        output_dir,
        "m_dijet",
        "delphes_dijet_mass_additional_jets.png",
        "Delphes 200PU: two leading PUPPI anti-kt R=0.4 jets after pT cut",
    )
    make_plot(
        np,
        plt,
        by_dataset,
        output_dir,
        "m_truth_matched",
        "delphes_truth_matched_mass_additional_jets.png",
        "Delphes 200PU: truth-matched PUPPI R=0.4 jets before pT cut",
    )
    make_plot(
        np,
        plt,
        by_dataset,
        output_dir,
        "m_truth_matched_pt20",
        "delphes_truth_matched_pt20_mass_additional_jets.png",
        "Delphes 200PU: truth-matched PUPPI R=0.4 jets after pT cut",
    )
    make_local_recovery_plot(np, plt, by_dataset, output_dir)
    print(f"Wrote {len(all_rows)} event rows and plots to {output_dir}")


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
