#!/usr/bin/env python3
"""Stage-1 comparison of Delphes GenJets and PUPPI jets."""

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


REPO = Path(__file__).resolve().parents[1]
HBB_CAMPAIGN = REPO / "output-superchic/Hbb/Hbb__v01/sim-Delphes"
QCD_CAMPAIGN = REPO / "output-madgraph/QCDbb/QCDbb__v01/sim-Delphes"
DEFAULT_INPUTS = {
    ("FSR", "Hbb"): HBB_CAMPAIGN / "Hbb_FSR_200PU__v01/root",
    ("noFSR", "Hbb"): HBB_CAMPAIGN / "Hbb_noFSR_200PU__v01/root",
    ("FSR", "QCDbb"): QCD_CAMPAIGN / "QCDbb_DPy8_FSR_200PU__v01/root",
    ("noFSR", "QCDbb"): QCD_CAMPAIGN / "QCDbb_DPy8_noFSR_200PU__v01/root",
}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/stage1"
RANKS = ("leading", "subleading")
OBJECTS = (*RANKS, "dijet")
COLORS = {"gen": "#0072B2", "reco": "#D55E00", "Hbb": "#009E73", "QCDbb": "#CC79A7"}
HISTOGRAM_BINS = 60
RATIO_RANGE = (0.0, 2.5)
ANGULAR_RESIDUAL_RANGE = (-0.2, 0.2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the two leading Delphes GenJets with uniquely matched "
            "JetPUPPI jets in H(bb) and MadGraph QCDbb samples."
        )
    )
    parser.add_argument("--hbb-fsr-dir", type=Path, default=DEFAULT_INPUTS[("FSR", "Hbb")])
    parser.add_argument(
        "--hbb-no-fsr-dir", type=Path, default=DEFAULT_INPUTS[("noFSR", "Hbb")]
    )
    parser.add_argument("--qcd-fsr-dir", type=Path, default=DEFAULT_INPUTS[("FSR", "QCDbb")])
    parser.add_argument(
        "--qcd-no-fsr-dir", type=Path, default=DEFAULT_INPUTS[("noFSR", "QCDbb")]
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional event limit per dataset, accumulated across selected files",
    )
    parser.add_argument("--eta-max", type=float, default=3.0)
    parser.add_argument("--match-dr-max", type=float, default=0.2)
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_JET_ENERGY_STAGE1_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_JET_ENERGY_STAGE1_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def validate_args(args):
    if args.max_files <= 0:
        raise RuntimeError("--max-files must be positive")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be positive")
    if args.eta_max <= 0.0 or args.match_dr_max <= 0.0:
        raise RuntimeError("--eta-max and --match-dr-max must be positive")


def natural_key(path):
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", path.name)
    ]


def input_files(directory, max_files):
    directory = directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Input directory does not exist: {directory}")
    files = sorted(directory.glob("*.root"), key=natural_key)
    if not files:
        raise RuntimeError(f"No ROOT files found in {directory}")
    return files[:max_files]


def delta_phi(first, second):
    """Return first - second wrapped to [-pi, pi]."""
    return math.atan2(math.sin(first - second), math.cos(first - second))


def delta_r(first, second):
    return math.hypot(
        first["eta"] - second["eta"],
        delta_phi(first["phi"], second["phi"]),
    )


def match_jets(gen_jets, reco_jets, max_delta_r):
    """Greedily make unique matches, starting from the smallest delta-R."""
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
    return sorted(matches, key=lambda match: match[0])


def jet_four_vector(jet):
    pt = jet["pt"]
    px = pt * math.cos(jet["phi"])
    py = pt * math.sin(jet["phi"])
    pz = pt * math.sinh(jet["eta"])
    energy = math.sqrt(px * px + py * py + pz * pz + jet["mass"] ** 2)
    return px, py, pz, energy


def add_four_vectors(*vectors):
    return tuple(sum(vector[index] for vector in vectors) for index in range(4))


def four_vector_kinematics(vector):
    px, py, pz, energy = vector
    pt = math.hypot(px, py)
    momentum_squared = px * px + py * py + pz * pz
    eta = math.asinh(pz / pt) if pt > 0.0 else math.nan
    phi = math.atan2(py, px) if pt > 0.0 else math.nan
    mass = math.sqrt(max(energy * energy - momentum_squared, 0.0))
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass}


def dijet_kinematics(first, second):
    return four_vector_kinematics(
        add_four_vectors(jet_four_vector(first), jet_four_vector(second))
    )


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0.0 else math.nan


def jet_observation(gen_jet, reco_jet, match_distance):
    return {
        "gen": dict(gen_jet),
        "reco": dict(reco_jet),
        "response": {
            "pt_ratio": safe_ratio(reco_jet["pt"], gen_jet["pt"]),
            "delta_eta": reco_jet["eta"] - gen_jet["eta"],
            "delta_phi": delta_phi(reco_jet["phi"], gen_jet["phi"]),
        },
        "match_dr": match_distance,
    }


def dijet_observation(gen_jets, reco_jets):
    gen_dijet = dijet_kinematics(*gen_jets)
    reco_dijet = dijet_kinematics(*reco_jets)
    return {
        "gen": gen_dijet,
        "reco": reco_dijet,
        "response": {
            "pt_ratio": safe_ratio(reco_dijet["pt"], gen_dijet["pt"]),
            "delta_eta": reco_dijet["eta"] - gen_dijet["eta"],
            "delta_phi": delta_phi(reco_dijet["phi"], gen_dijet["phi"]),
            "mass_ratio": safe_ratio(reco_dijet["mass"], gen_dijet["mass"]),
        },
    }


def analyze_event(gen_jets, reco_jets, max_delta_r):
    leading_gen = sorted(gen_jets, key=lambda jet: jet["pt"], reverse=True)[:2]
    ordered_reco = sorted(reco_jets, key=lambda jet: jet["pt"], reverse=True)
    matches = {
        gen_index: (reco_index, distance)
        for gen_index, reco_index, distance in match_jets(
            leading_gen, ordered_reco, max_delta_r
        )
    }
    result = {
        "eligible": {
            "leading": len(leading_gen) >= 1,
            "subleading": len(leading_gen) >= 2,
            "dijet": len(leading_gen) >= 2,
        },
        "leading": None,
        "subleading": None,
        "dijet": None,
    }
    for gen_index, rank in enumerate(RANKS):
        if gen_index not in matches:
            continue
        reco_index, distance = matches[gen_index]
        result[rank] = jet_observation(
            leading_gen[gen_index], ordered_reco[reco_index], distance
        )
    if result["leading"] is not None and result["subleading"] is not None:
        result["dijet"] = dijet_observation(
            [result[rank]["gen"] for rank in RANKS],
            [result[rank]["reco"] for rank in RANKS],
        )
    return result


def read_jets(collection, eta_max):
    jets = []
    for index in range(collection.GetEntriesFast()):
        jet = collection.At(index)
        eta = float(jet.Eta)
        if abs(eta) >= eta_max:
            continue
        jets.append(
            {
                "pt": float(jet.PT),
                "eta": eta,
                "phi": float(jet.Phi),
                "mass": float(jet.Mass),
            }
        )
    return jets


def analyze_dataset(ROOT, fsr_state, sample, files, args):
    result = {
        "fsr_state": fsr_state,
        "sample": sample,
        "files": [str(path.resolve()) for path in files],
        "events_read": 0,
        "eligible": {name: 0 for name in OBJECTS},
        "matched": {name: 0 for name in OBJECTS},
        "observations": {name: [] for name in OBJECTS},
    }
    for path in files:
        if args.max_events is not None and result["events_read"] >= args.max_events:
            break
        root_file = ROOT.TFile.Open(str(path))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not open ROOT file: {path}")
        tree = root_file.Get("Delphes")
        missing = [name for name in ("GenJet", "JetPUPPI") if not tree or not tree.GetBranch(name)]
        if missing:
            root_file.Close()
            raise RuntimeError(f"Missing {', '.join(missing)} branch(es) in {path}")
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("GenJet*", 1)
        tree.SetBranchStatus("JetPUPPI*", 1)
        entries = int(tree.GetEntries())
        if args.max_events is not None:
            entries = min(entries, args.max_events - result["events_read"])
        print(f"Reading {fsr_state}/{sample}: {path.name} ({entries} events)", flush=True)
        for entry in range(entries):
            tree.GetEntry(entry)
            event = analyze_event(
                read_jets(tree.GenJet, args.eta_max),
                read_jets(tree.JetPUPPI, args.eta_max),
                args.match_dr_max,
            )
            result["events_read"] += 1
            for name in OBJECTS:
                result["eligible"][name] += int(event["eligible"][name])
                if event[name] is not None:
                    result["matched"][name] += 1
                    result["observations"][name].append(event[name])
        root_file.Close()
    if result["events_read"] == 0:
        raise RuntimeError(f"No events read for {fsr_state}/{sample}")
    return result


def finite_values(np, observations, group, field):
    return np.asarray(
        [
            observation[group][field]
            for observation in observations
            if math.isfinite(observation[group][field])
        ],
        dtype=np.float64,
    )


def kinematic_edges(np, observations, field, eta_max):
    if field == "eta":
        return np.linspace(-eta_max, eta_max, HISTOGRAM_BINS + 1)
    if field == "phi":
        return np.linspace(-math.pi, math.pi, HISTOGRAM_BINS + 1)
    values = [finite_values(np, observations, group, field) for group in ("gen", "reco")]
    populated = [array for array in values if array.size]
    upper = 10.0
    if populated:
        upper = max(upper, 1.05 * float(np.quantile(np.concatenate(populated), 0.995)))
    return np.linspace(0.0, upper, HISTOGRAM_BINS + 1)


def draw_histogram(axis, values, edges, label, color):
    if values.size:
        axis.hist(
            values,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{label} (N={values.size})",
        )


def save_kinematic_plot(np, plt, observations, object_name, title, output_path, eta_max):
    fields = [
        ("pt", r"$p_T$ [GeV]"),
        ("eta", r"$\eta$"),
        ("phi", r"$\phi$"),
    ]
    if object_name == "dijet":
        fields.append(("mass", r"$m_{jj}$ [GeV]"))
    rows, columns = ((2, 2) if len(fields) == 4 else (1, 3))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 7 if rows == 2 else 3.8))
    flat_axes = np.asarray(axes).reshape(-1)
    for axis, (field, xlabel) in zip(flat_axes, fields):
        edges = kinematic_edges(np, observations, field, eta_max)
        for group, label in (("gen", "GenJet"), ("reco", "JetPUPPI")):
            draw_histogram(
                axis,
                finite_values(np, observations, group, field),
                edges,
                label,
                COLORS[group],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Normalized entries")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in flat_axes[len(fields) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def response_fields(object_name):
    fields = [
        ("pt_ratio", r"$p_T^{reco}/p_T^{gen}$", RATIO_RANGE, 1.0),
        ("delta_eta", r"$\eta^{reco}-\eta^{gen}$", ANGULAR_RESIDUAL_RANGE, 0.0),
        ("delta_phi", r"wrapped $\phi^{reco}-\phi^{gen}$", ANGULAR_RESIDUAL_RANGE, 0.0),
    ]
    if object_name == "dijet":
        fields.append(("mass_ratio", r"$m_{jj}^{reco}/m_{jj}^{gen}$", RATIO_RANGE, 1.0))
    return fields


def save_response_plot(np, plt, series, object_name, title, output_path):
    fields = response_fields(object_name)
    rows, columns = ((2, 2) if len(fields) == 4 else (1, 3))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 7 if rows == 2 else 3.8))
    flat_axes = np.asarray(axes).reshape(-1)
    for axis, (field, xlabel, value_range, reference) in zip(flat_axes, fields):
        edges = np.linspace(*value_range, HISTOGRAM_BINS + 1)
        for label, observations, color in series:
            values = finite_values(np, observations, "response", field)
            draw_histogram(axis, values, edges, label, color)
        axis.axvline(reference, color="black", linestyle=":", linewidth=1.0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Normalized entries")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in flat_axes[len(fields) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def make_sample_plots(np, plt, result, output_dir, eta_max):
    sample_dir = output_dir / result["fsr_state"] / result["sample"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    for object_name in OBJECTS:
        observations = result["observations"][object_name]
        object_label = object_name.capitalize()
        title = f"{result['fsr_state']} {result['sample']}: matched {object_label} kinematics"
        save_kinematic_plot(
            np,
            plt,
            observations,
            object_name,
            title,
            sample_dir / f"{object_name}_kinematics.png",
            eta_max,
        )
        save_response_plot(
            np,
            plt,
            [("Matched jets", observations, COLORS["reco"])],
            object_name,
            f"{result['fsr_state']} {result['sample']}: {object_label} response",
            sample_dir / f"{object_name}_response.png",
        )
    return 6


def make_cross_sample_plots(np, plt, fsr_state, by_sample, output_dir):
    comparison_dir = output_dir / fsr_state / "cross_sample"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for object_name in OBJECTS:
        series = [
            (
                sample,
                by_sample[sample]["observations"][object_name],
                COLORS[sample],
            )
            for sample in ("Hbb", "QCDbb")
        ]
        save_response_plot(
            np,
            plt,
            series,
            object_name,
            f"{fsr_state}: H(bb) versus QCDbb {object_name} response",
            comparison_dir / f"{object_name}_response.png",
        )
    return 3


def distribution_summary(np, values, value_range):
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    low, high = value_range
    return {
        "entries": len(values),
        "finite_entries": int(finite.size),
        "nonfinite_entries": len(values) - int(finite.size),
        "underflow": int(np.count_nonzero(finite < low)),
        "overflow": int(np.count_nonzero(finite > high)),
        "p16": float(np.quantile(finite, 0.16)) if finite.size else None,
        "median": float(np.quantile(finite, 0.50)) if finite.size else None,
        "p84": float(np.quantile(finite, 0.84)) if finite.size else None,
    }


def zero_denominator_entries(observations, response_field):
    denominator = {"pt_ratio": "pt", "mass_ratio": "mass"}.get(response_field)
    if denominator is None:
        return 0
    return sum(observation["gen"][denominator] <= 0.0 for observation in observations)


def result_summary(np, result):
    efficiencies = {
        name: (
            result["matched"][name] / result["eligible"][name]
            if result["eligible"][name]
            else None
        )
        for name in OBJECTS
    }
    responses = {}
    for object_name in OBJECTS:
        observations = result["observations"][object_name]
        responses[object_name] = {}
        for field, _label, value_range, _reference in response_fields(object_name):
            summary = distribution_summary(
                np,
                [observation["response"][field] for observation in observations],
                value_range,
            )
            if field in ("pt_ratio", "mass_ratio"):
                summary["zero_denominator_entries"] = zero_denominator_entries(
                    observations, field
                )
            responses[object_name][field] = summary
    return {
        "files": result["files"],
        "events_read": result["events_read"],
        "eligible": result["eligible"],
        "matched": result["matched"],
        "matching_efficiency": efficiencies,
        "response": responses,
    }


def build_summary(np, args, results, plot_count):
    by_key = {(result["fsr_state"], result["sample"]): result for result in results}
    return {
        "configuration": {
            "gen_collection": "GenJet",
            "reco_collection": "JetPUPPI",
            "max_files": args.max_files,
            "max_events_per_dataset": args.max_events,
            "eta_max": args.eta_max,
            "match_dr_max": args.match_dr_max,
            "histogram_bins": HISTOGRAM_BINS,
            "ratio_range": list(RATIO_RANGE),
            "angular_residual_range": list(ANGULAR_RESIDUAL_RANGE),
            "normalization": "unit area per distribution",
            "plot_count": plot_count,
        },
        "samples": {
            fsr_state: {
                sample: result_summary(np, by_key[(fsr_state, sample)])
                for sample in ("Hbb", "QCDbb")
            }
            for fsr_state in ("FSR", "noFSR")
        },
    }


def run(args):
    validate_args(args)
    import ROOT
    import matplotlib
    import numpy as np
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ROOT.gSystem.Load("libDelphes") < 0:
        raise RuntimeError("Could not load libDelphes")

    directories = {
        ("FSR", "Hbb"): args.hbb_fsr_dir,
        ("noFSR", "Hbb"): args.hbb_no_fsr_dir,
        ("FSR", "QCDbb"): args.qcd_fsr_dir,
        ("noFSR", "QCDbb"): args.qcd_no_fsr_dir,
    }
    results = []
    for fsr_state in ("FSR", "noFSR"):
        for sample in ("Hbb", "QCDbb"):
            files = input_files(directories[(fsr_state, sample)], args.max_files)
            results.append(analyze_dataset(ROOT, fsr_state, sample, files, args))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_count = sum(
        make_sample_plots(np, plt, result, output_dir, args.eta_max)
        for result in results
    )
    by_fsr = {
        fsr_state: {
            result["sample"]: result
            for result in results
            if result["fsr_state"] == fsr_state
        }
        for fsr_state in ("FSR", "noFSR")
    }
    plot_count += sum(
        make_cross_sample_plots(np, plt, fsr_state, by_fsr[fsr_state], output_dir)
        for fsr_state in ("FSR", "noFSR")
    )
    summary = build_summary(np, args, results, plot_count)
    summary_path = output_dir / "summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    for result in results:
        efficiencies = {
            name: (
                result["matched"][name] / result["eligible"][name]
                if result["eligible"][name]
                else math.nan
            )
            for name in OBJECTS
        }
        print(
            f"{result['fsr_state']}/{result['sample']}: {result['events_read']} events, "
            f"leading={efficiencies['leading']:.1%}, "
            f"subleading={efficiencies['subleading']:.1%}, "
            f"dijet={efficiencies['dijet']:.1%}"
        )
    print(f"Wrote {plot_count} plots and {summary_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
