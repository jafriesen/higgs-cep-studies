#!/usr/bin/env python3
"""Derive a simple JetPUPPI energy correction from matched Delphes GenJets."""

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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/delphes_jet_correction"
DEFAULT_PT_BINS = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 250.0)
DEFAULT_ETA_BINS = (0.0, 1.5, 2.5)
BOTTOM_MATCH_DR_MAX = 0.4
LEADING_PT_MIN = 20.0
SUBLEADING_PT_MIN = 15.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Match R=0.4 JetPUPPI jets to visible R=0.4 GenJets and derive a "
            "separate median jet-energy correction for the FSR and noFSR samples."
        )
    )
    parser.add_argument("--no-fsr-dir", type=Path, default=DEFAULT_NO_FSR)
    parser.add_argument("--fsr-dir", type=Path, default=DEFAULT_FSR)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=2000, help="Events read per dataset")
    parser.add_argument("--match-dr-max", type=float, default=0.2)
    parser.add_argument(
        "--gen-pt-min",
        type=float,
        default=5.0,
        help="Generator-jet minimum; default matches the stored GenJet threshold",
    )
    parser.add_argument("--pt-bins", type=float, nargs="+", default=DEFAULT_PT_BINS)
    parser.add_argument("--eta-bins", type=float, nargs="+", default=DEFAULT_ETA_BINS)
    parser.add_argument("--min-bin-entries", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_JEC_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_DELPHES_JEC_ENV=1",
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


def delta_r(first, second):
    delta_phi = math.atan2(
        math.sin(first["phi"] - second["phi"]),
        math.cos(first["phi"] - second["phi"]),
    )
    return math.hypot(first["eta"] - second["eta"], delta_phi)


def read_jets(collection):
    return [
        {
            "pt": float(collection.At(index).PT),
            "eta": float(collection.At(index).Eta),
            "phi": float(collection.At(index).Phi),
            "mass": float(collection.At(index).Mass),
        }
        for index in range(collection.GetEntriesFast())
    ]


def jet_p4(jet, scale=1.0):
    pt = scale * jet["pt"]
    mass = scale * jet["mass"]
    px = pt * math.cos(jet["phi"])
    py = pt * math.sin(jet["phi"])
    pz = pt * math.sinh(jet["eta"])
    energy = math.sqrt(px * px + py * py + pz * pz + mass * mass)
    return px, py, pz, energy


def invariant_mass(vectors):
    px = sum(vector[0] for vector in vectors)
    py = sum(vector[1] for vector in vectors)
    pz = sum(vector[2] for vector in vectors)
    energy = sum(vector[3] for vector in vectors)
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def match_jets(gen_jets, reco_jets, max_delta_r):
    candidates = []
    for gen_index, gen_jet in enumerate(gen_jets):
        for reco_index, reco_jet in enumerate(reco_jets):
            distance = delta_r(gen_jet, reco_jet)
            if distance < max_delta_r:
                candidates.append((distance, gen_index, reco_index))

    matches = []
    used_gen = set()
    used_reco = set()
    for distance, gen_index, reco_index in sorted(candidates):
        if gen_index in used_gen or reco_index in used_reco:
            continue
        used_gen.add(gen_index)
        used_reco.add(reco_index)
        matches.append((gen_index, reco_index, distance))
    return matches


def match_truth_bottoms(particles, reco_jets):
    bottoms = [
        {
            "pt": float(particle.PT),
            "eta": float(particle.Eta),
            "phi": float(particle.Phi),
            "pid": int(particle.PID),
        }
        for particle in particles
        if abs(int(particle.PID)) == 5
        and int(particle.Status) == 23
        and not int(particle.IsPU)
    ]
    b_quarks = [particle for particle in bottoms if particle["pid"] == 5]
    bbar_quarks = [particle for particle in bottoms if particle["pid"] == -5]
    if len(b_quarks) != 1 or len(bbar_quarks) != 1 or len(reco_jets) < 2:
        return None

    best = None
    for b_index in range(len(reco_jets)):
        for bbar_index in range(len(reco_jets)):
            if b_index == bbar_index:
                continue
            b_distance = delta_r(b_quarks[0], reco_jets[b_index])
            bbar_distance = delta_r(bbar_quarks[0], reco_jets[bbar_index])
            candidate = (
                b_distance + bbar_distance,
                b_index,
                bbar_index,
                b_distance,
                bbar_distance,
            )
            if best is None or candidate < best:
                best = candidate
    if best[3] >= BOTTOM_MATCH_DR_MAX or best[4] >= BOTTOM_MATCH_DR_MAX:
        return None
    return best[1], best[2], best[3], best[4]


def analyze_dataset(ROOT, dataset, files, args):
    rows = []
    dijet_rows = []
    events_read = 0
    gen_jets_considered = 0
    eta_max = args.eta_bins[-1]
    for path in files:
        if events_read >= args.max_events:
            break
        root_file = ROOT.TFile.Open(str(path))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not open ROOT file: {path}")
        tree = root_file.Get("Delphes")
        if (
            not tree
            or not tree.GetBranch("JetPUPPI")
            or not tree.GetBranch("GenJet")
            or not tree.GetBranch("Particle")
        ):
            root_file.Close()
            raise RuntimeError(f"JetPUPPI, GenJet, or Particle branch missing from {path}")

        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("JetPUPPI*", 1)
        tree.SetBranchStatus("GenJet*", 1)
        tree.SetBranchStatus("Particle*", 1)
        entries = min(int(tree.GetEntries()), args.max_events - events_read)
        print(f"Reading {dataset}: {path.name} ({entries} events)")
        for entry in range(entries):
            tree.GetEntry(entry)
            events_read += 1
            reco_jets = [jet for jet in read_jets(tree.JetPUPPI) if abs(jet["eta"]) < eta_max]
            gen_jets = [
                jet
                for jet in read_jets(tree.GenJet)
                if jet["pt"] >= args.gen_pt_min and abs(jet["eta"]) < eta_max
            ]
            gen_jets_considered += len(gen_jets)
            for gen_index, reco_index, distance in match_jets(
                gen_jets, reco_jets, args.match_dr_max
            ):
                gen_jet = gen_jets[gen_index]
                reco_jet = reco_jets[reco_index]
                response = reco_jet["pt"] / gen_jet["pt"]
                rows.append(
                    {
                        "dataset": dataset,
                        "file": path.name,
                        "event": events_read,
                        "gen_pt": gen_jet["pt"],
                        "gen_eta": gen_jet["eta"],
                        "reco_pt": reco_jet["pt"],
                        "reco_eta": reco_jet["eta"],
                        "match_dr": distance,
                        "raw_response": response,
                    }
                )

            particles = [tree.Particle.At(index) for index in range(tree.Particle.GetEntriesFast())]
            bottom_match = match_truth_bottoms(particles, reco_jets)
            if bottom_match is not None:
                b_index, bbar_index, b_distance, bbar_distance = bottom_match
                b_jet = reco_jets[b_index]
                bbar_jet = reco_jets[bbar_index]
                ordered_pts = sorted((b_jet["pt"], bbar_jet["pt"]), reverse=True)
                dijet_rows.append(
                    {
                        "dataset": dataset,
                        "file": path.name,
                        "event": events_read,
                        "match_dr_b": b_distance,
                        "match_dr_bbar": bbar_distance,
                        "b_pt_raw": b_jet["pt"],
                        "b_eta": b_jet["eta"],
                        "b_phi": b_jet["phi"],
                        "b_mass_raw": b_jet["mass"],
                        "bbar_pt_raw": bbar_jet["pt"],
                        "bbar_eta": bbar_jet["eta"],
                        "bbar_phi": bbar_jet["phi"],
                        "bbar_mass_raw": bbar_jet["mass"],
                        "leading_pt_raw": ordered_pts[0],
                        "subleading_pt_raw": ordered_pts[1],
                        "passes_pt20_15_raw": int(
                            ordered_pts[0] > LEADING_PT_MIN
                            and ordered_pts[1] > SUBLEADING_PT_MIN
                        ),
                        "dijet_mass_raw": invariant_mass((jet_p4(b_jet), jet_p4(bbar_jet))),
                    }
                )
        root_file.Close()

    if events_read == 0:
        raise RuntimeError(f"No events read for {dataset}")
    return rows, dijet_rows, {
        "dataset": dataset,
        "events_read": events_read,
        "gen_jets_considered": gen_jets_considered,
        "matched_jets": len(rows),
        "match_efficiency": len(rows) / gen_jets_considered if gen_jets_considered else math.nan,
        "truth_b_dijets": len(dijet_rows),
    }


def find_bin(value, edges):
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if low <= value < high:
            return index
    return None


def percentile(np, values, quantile):
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else math.nan


def derive_table(np, rows, args):
    table = []
    for dataset in ("noFSR", "FSR"):
        for eta_index, (eta_min, eta_max) in enumerate(
            zip(args.eta_bins[:-1], args.eta_bins[1:])
        ):
            for pt_index, (pt_min, pt_max) in enumerate(
                zip(args.pt_bins[:-1], args.pt_bins[1:])
            ):
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and pt_min <= row["reco_pt"] < pt_max
                    and eta_min <= abs(row["reco_eta"]) < eta_max
                ]
                responses = [row["raw_response"] for row in selected]
                valid = len(responses) >= args.min_bin_entries
                median_response = percentile(np, responses, 0.5)
                table.append(
                    {
                        "dataset": dataset,
                        "eta_bin": eta_index,
                        "eta_min": eta_min,
                        "eta_max": eta_max,
                        "pt_bin": pt_index,
                        "reco_pt_min": pt_min,
                        "reco_pt_max": pt_max,
                        "entries": len(responses),
                        "valid": int(valid),
                        "mean_response": float(np.mean(responses)) if responses else math.nan,
                        "median_response": median_response,
                        "response_p16": percentile(np, responses, 0.16),
                        "response_p84": percentile(np, responses, 0.84),
                        "correction": (
                            1.0 / median_response
                            if valid and median_response > 0.0
                            else 1.0
                        ),
                    }
                )
    return table


def correction_for(row, table, args):
    pt_bin = find_bin(row["reco_pt"], args.pt_bins)
    eta_bin = find_bin(abs(row["reco_eta"]), args.eta_bins)
    if pt_bin is None or eta_bin is None:
        return 1.0, False
    table_row = next(
        item
        for item in table
        if item["dataset"] == row["dataset"]
        and item["pt_bin"] == pt_bin
        and item["eta_bin"] == eta_bin
    )
    return table_row["correction"], bool(table_row["valid"])


def add_corrected_responses(rows, table, args):
    for row in rows:
        correction, calibrated = correction_for(row, table, args)
        row["correction"] = correction
        row["corrected_pt"] = correction * row["reco_pt"]
        row["corrected_response"] = correction * row["raw_response"]
        row["calibrated_bin"] = int(calibrated)


def add_corrected_dijets(dijet_rows, table, args):
    for row in dijet_rows:
        b_lookup = {
            "dataset": row["dataset"],
            "reco_pt": row["b_pt_raw"],
            "reco_eta": row["b_eta"],
        }
        bbar_lookup = {
            "dataset": row["dataset"],
            "reco_pt": row["bbar_pt_raw"],
            "reco_eta": row["bbar_eta"],
        }
        b_correction, b_calibrated = correction_for(b_lookup, table, args)
        bbar_correction, bbar_calibrated = correction_for(bbar_lookup, table, args)
        row["b_correction"] = b_correction
        row["bbar_correction"] = bbar_correction
        row["both_jets_calibrated"] = int(b_calibrated and bbar_calibrated)
        row["b_pt_corrected"] = b_correction * row["b_pt_raw"]
        row["bbar_pt_corrected"] = bbar_correction * row["bbar_pt_raw"]
        corrected_pts = sorted(
            (row["b_pt_corrected"], row["bbar_pt_corrected"]), reverse=True
        )
        row["leading_pt_corrected"] = corrected_pts[0]
        row["subleading_pt_corrected"] = corrected_pts[1]
        row["passes_pt20_15_corrected"] = int(
            corrected_pts[0] > LEADING_PT_MIN
            and corrected_pts[1] > SUBLEADING_PT_MIN
        )
        b_jet = {
            "pt": row["b_pt_raw"],
            "eta": row["b_eta"],
            "phi": row["b_phi"],
            "mass": row["b_mass_raw"],
        }
        bbar_jet = {
            "pt": row["bbar_pt_raw"],
            "eta": row["bbar_eta"],
            "phi": row["bbar_phi"],
            "mass": row["bbar_mass_raw"],
        }
        row["dijet_mass_corrected"] = invariant_mass(
            (jet_p4(b_jet, b_correction), jet_p4(bbar_jet, bbar_correction))
        )


def summarize_closure(np, rows, input_summaries):
    summaries = []
    for input_summary in input_summaries:
        dataset = input_summary["dataset"]
        sample = [row for row in rows if row["dataset"] == dataset]
        calibrated = [row for row in sample if row["calibrated_bin"]]
        result = dict(input_summary)
        result["jets_in_calibrated_bins"] = len(calibrated)
        for label, key in (("raw", "raw_response"), ("corrected", "corrected_response")):
            values = [row[key] for row in calibrated]
            result[f"mean_{label}_response"] = float(np.mean(values)) if values else math.nan
            result[f"median_{label}_response"] = percentile(np, values, 0.5)
            result[f"{label}_response_p16"] = percentile(np, values, 0.16)
            result[f"{label}_response_p84"] = percentile(np, values, 0.84)
        summaries.append(result)
    return summaries


def summarize_dijets(np, dijet_rows):
    summaries = []
    for dataset in ("noFSR", "FSR"):
        sample = [row for row in dijet_rows if row["dataset"] == dataset]
        result = {
            "dataset": dataset,
            "truth_b_dijets": len(sample),
            "both_jets_calibrated": sum(row["both_jets_calibrated"] for row in sample),
            "passes_pt20_15_raw": sum(row["passes_pt20_15_raw"] for row in sample),
            "passes_pt20_15_corrected": sum(
                row["passes_pt20_15_corrected"] for row in sample
            ),
        }
        for label, key in (
            ("raw", "dijet_mass_raw"),
            ("corrected", "dijet_mass_corrected"),
        ):
            values = [row[key] for row in sample]
            result[f"mean_dijet_mass_{label}"] = float(np.mean(values)) if values else math.nan
            result[f"median_dijet_mass_{label}"] = percentile(np, values, 0.5)
        summaries.append(result)
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tcl_formula(path, table, dataset):
    terms = []
    for row in table:
        if not row["valid"]:
            continue
        condition = (
            f"abs(eta) >= {row['eta_min']:g} && abs(eta) < {row['eta_max']:g} && "
            f"pt >= {row['reco_pt_min']:g} && pt < {row['reco_pt_max']:g}"
        )
        terms.append(f"    + ({condition}) * ({row['correction'] - 1.0:+.8f})")
    text = "\n".join(
        (
            f"# Correction derived from the {dataset} sample only.",
            "# Replace ScaleFormula in JetEnergyScalePUPPI with this expression.",
            "# Uncalibrated and out-of-range bins retain scale factor 1.0.",
            "set ScaleFormula {",
            "    1.0 \\",
            " \\\n".join(terms),
            "}",
            "",
        )
    )
    path.write_text(text, encoding="utf-8")


def print_table(table, summaries, dijet_summaries):
    print()
    print("Separate corrections C = 1 / median(pT_reco / pT_gen), binned in raw reco pT")
    print(f"{'sample':<6} {'|eta|':>11} {'raw pT':>11} {'N':>6} {'median R':>10} {'C':>8}")
    for row in table:
        print(
            f"{row['dataset']:<6} {row['eta_min']:>4.1f}-{row['eta_max']:<4.1f} "
            f"{row['reco_pt_min']:>4.0f}-{row['reco_pt_max']:<4.0f} "
            f"{row['entries']:>6} {row['median_response']:>10.3f} "
            f"{row['correction']:>8.3f}"
        )
    print()
    print(f"{'sample':<6} {'matched':>13} {'used':>8} {'raw mean':>10} {'raw med':>9} {'corr mean':>10} {'corr med':>9}")
    for row in summaries:
        print(
            f"{row['dataset']:<6} {row['matched_jets']:>5}/{row['gen_jets_considered']:<7} "
            f"{row['jets_in_calibrated_bins']:>8} "
            f"{row['mean_raw_response']:>10.3f} {row['median_raw_response']:>9.3f} "
            f"{row['mean_corrected_response']:>10.3f} {row['median_corrected_response']:>9.3f}"
        )
    print()
    print("Truth-matched b-jet kinematics before and after the sample-specific correction")
    print(
        f"{'sample':<6} {'pairs':>6} {'both cal':>9} {'pass raw':>9} {'pass corr':>10} "
        f"{'mass mean':>11} {'mass med':>10} {'corr mean':>11} {'corr med':>10}"
    )
    for row in dijet_summaries:
        print(
            f"{row['dataset']:<6} {row['truth_b_dijets']:>6} "
            f"{row['both_jets_calibrated']:>9} {row['passes_pt20_15_raw']:>9} "
            f"{row['passes_pt20_15_corrected']:>10} "
            f"{row['mean_dijet_mass_raw']:>11.2f} {row['median_dijet_mass_raw']:>10.2f} "
            f"{row['mean_dijet_mass_corrected']:>11.2f} "
            f"{row['median_dijet_mass_corrected']:>10.2f}"
        )


def make_plots(np, plt, rows, table, output_dir):
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    line_styles = ("-", "--", "-.", ":")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for dataset in ("noFSR", "FSR"):
        for eta_bin in sorted({row["eta_bin"] for row in table}):
            selected = [
                row
                for row in table
                if row["dataset"] == dataset
                and row["eta_bin"] == eta_bin
                and row["valid"]
            ]
            if not selected:
                continue
            centers = [
                (row["reco_pt_min"] + row["reco_pt_max"]) / 2.0 for row in selected
            ]
            half_widths = [
                (row["reco_pt_max"] - row["reco_pt_min"]) / 2.0 for row in selected
            ]
            label = (
                f"{dataset}, {selected[0]['eta_min']:g} <= |eta| < "
                f"{selected[0]['eta_max']:g}"
            )
            style = line_styles[eta_bin % len(line_styles)]
            axes[0].errorbar(
                centers,
                [row["median_response"] for row in selected],
                xerr=half_widths,
                fmt="o",
                linestyle=style,
                color=colors[dataset],
                label=label,
            )
            axes[1].errorbar(
                centers,
                [row["correction"] for row in selected],
                xerr=half_widths,
                fmt="o",
                linestyle=style,
                color=colors[dataset],
                label=label,
            )
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Median raw response")
    axes[1].set_ylabel("Correction factor")
    for axis in axes:
        axis.set_xlabel("Raw reconstructed jet pT [GeV]")
        axis.set_xscale("log")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    figure.suptitle("Delphes R=0.4 JetPUPPI energy response and correction")
    figure.tight_layout()
    figure.savefig(output_dir / "jet_energy_correction.png", dpi=160)
    plt.close(figure)

    bins = np.linspace(0.0, 2.0, 81)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (key, title) in zip(
        axes,
        (("raw_response", "Before correction"), ("corrected_response", "After correction")),
    ):
        for dataset, color in (("noFSR", "#0072B2"), ("FSR", "#D55E00")):
            values = [
                row[key]
                for row in rows
                if row["dataset"] == dataset and row["calibrated_bin"]
            ]
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=color,
                label=dataset,
            )
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("pT(reco) / pT(gen)")
        axis.set_ylabel("Normalized matched jets")
        axis.legend(frameon=False)
    figure.suptitle("Closure of the sample-specific JetPUPPI corrections")
    figure.tight_layout()
    figure.savefig(output_dir / "jet_energy_correction_closure.png", dpi=160)
    plt.close(figure)


def make_kinematic_plots(np, plt, dijet_rows, output_dir):
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    pt_bins = np.linspace(0.0, 100.0, 81)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (suffix, title) in zip(
        axes, (("raw", "Before correction"), ("corrected", "After correction"))
    ):
        for dataset in ("noFSR", "FSR"):
            sample = [row for row in dijet_rows if row["dataset"] == dataset]
            values = [
                value
                for row in sample
                for value in (row[f"b_pt_{suffix}"], row[f"bbar_pt_{suffix}"])
            ]
            axis.hist(
                values,
                bins=pt_bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[dataset],
                label=dataset,
            )
        axis.axvline(SUBLEADING_PT_MIN, color="0.4", linestyle="--", linewidth=1.0)
        axis.axvline(LEADING_PT_MIN, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Truth-matched reconstructed jet pT [GeV]")
        axis.set_ylabel("Normalized jets")
        axis.legend(frameon=False)
    figure.suptitle("Actual pT of the two truth-matched R=0.4 PUPPI b jets")
    figure.tight_layout()
    figure.savefig(output_dir / "truth_b_jet_pt_before_after.png", dpi=160)
    plt.close(figure)

    all_masses = [
        row[key]
        for row in dijet_rows
        for key in ("dijet_mass_raw", "dijet_mass_corrected")
    ]
    upper = max(175.0, 25.0 * math.ceil(float(np.quantile(all_masses, 0.995)) / 25.0))
    mass_bins = np.linspace(0.0, upper, 81)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (key, title) in zip(
        axes,
        (("dijet_mass_raw", "Before correction"), ("dijet_mass_corrected", "After correction")),
    ):
        for dataset in ("noFSR", "FSR"):
            values = [row[key] for row in dijet_rows if row["dataset"] == dataset]
            axis.hist(
                values,
                bins=mass_bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[dataset],
                label=dataset,
            )
        axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Truth-matched dijet mass [GeV]")
        axis.set_ylabel("Normalized events")
        axis.set_xlim(0.0, upper)
        axis.legend(frameon=False)
    figure.suptitle("Actual mass of the two truth-matched R=0.4 PUPPI b jets")
    figure.tight_layout()
    figure.savefig(output_dir / "truth_b_dijet_mass_before_after.png", dpi=160)
    plt.close(figure)


def validate_args(args):
    if args.max_files <= 0 or args.max_events <= 0 or args.min_bin_entries <= 0:
        raise RuntimeError("--max-files, --max-events, and --min-bin-entries must be positive")
    if args.match_dr_max <= 0.0 or args.gen_pt_min < 0.0:
        raise RuntimeError("--match-dr-max must be positive and --gen-pt-min non-negative")
    for name, edges in (("--pt-bins", args.pt_bins), ("--eta-bins", args.eta_bins)):
        if len(edges) < 2 or any(high <= low for low, high in zip(edges[:-1], edges[1:])):
            raise RuntimeError(f"{name} must contain at least two strictly increasing edges")
    if args.eta_bins[0] != 0.0:
        raise RuntimeError("--eta-bins must start at zero")


def main():
    args = parse_args()
    validate_args(args)

    import ROOT
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ROOT.gSystem.Load("libDelphes") < 0:
        raise RuntimeError("Could not load libDelphes")

    selected_files = {
        "noFSR": input_files(args.no_fsr_dir, args.max_files),
        "FSR": input_files(args.fsr_dir, args.max_files),
    }
    rows = []
    dijet_rows = []
    input_summaries = []
    for dataset, files in selected_files.items():
        sample_rows, sample_dijets, input_summary = analyze_dataset(
            ROOT, dataset, files, args
        )
        rows.extend(sample_rows)
        dijet_rows.extend(sample_dijets)
        input_summaries.append(input_summary)
    if not rows:
        raise RuntimeError("No matched jets were found")
    if not dijet_rows:
        raise RuntimeError("No truth-matched b-jet pairs were found")

    table = derive_table(np, rows, args)
    add_corrected_responses(rows, table, args)
    add_corrected_dijets(dijet_rows, table, args)
    summaries = summarize_closure(np, rows, input_summaries)
    dijet_summaries = summarize_dijets(np, dijet_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "jet_energy_correction.csv", table)
    write_csv(output_dir / "matched_jets.csv", rows)
    write_csv(output_dir / "closure_summary.csv", summaries)
    write_csv(output_dir / "truth_b_dijets.csv", dijet_rows)
    write_csv(output_dir / "dijet_summary.csv", dijet_summaries)
    for dataset in ("noFSR", "FSR"):
        sample_table = [row for row in table if row["dataset"] == dataset]
        write_tcl_formula(
            output_dir / f"jet_energy_scale_formula_{dataset}.tcl",
            sample_table,
            dataset,
        )
    (output_dir / "jet_energy_scale_formula.tcl").write_text(
        "# Corrections are now sample-specific. Use "
        "jet_energy_scale_formula_noFSR.tcl or jet_energy_scale_formula_FSR.tcl.\n",
        encoding="utf-8",
    )
    make_plots(np, plt, rows, table, output_dir)
    make_kinematic_plots(np, plt, dijet_rows, output_dir)
    print_table(table, summaries, dijet_summaries)
    print(
        f"Wrote {len(rows)} matched jets, {len(dijet_rows)} truth-matched dijets, "
        f"and calibration outputs to {output_dir}"
    )


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
