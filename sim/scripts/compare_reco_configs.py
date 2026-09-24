#!/usr/bin/env python3
"""Compare H(bb) jet outcomes across FSR states and Delphes reconstruction cards.

Every card gets the same jet selection (pT > 20 GeV, |eta| < 2.5), matching the
Phase-I card threshold and tracker coverage.  Histograms are normalized per
generated event, so differences in the two-jet efficiency stay visible.

With --correct, a simple per-card jet energy correction is derived and applied
before the selection.  Every card processes the same generator events in the
same order, so the common truth is the R=0.4 GenJet collection of the 0PU file.
Reco jets are matched to GenJets (dR < 0.2); in bins of GenJet pT and |eta| the
median response gives a (median raw pT -> median raw pT / response) node, and
common.jet_calibration.correction_factors interpolates between nodes.  The
correction is derived separately for each card and FSR state on even events and
applied to the same sample; the summary then uses odd events only, raw and corrected.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import awkward as ak
import matplotlib
import numpy as np
import uproot
import vector

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

vector.register_awkward()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.jet_calibration import correction_factors  # noqa: E402

SIM_DIR = ROOT / "output-superchic" / "Hbb" / "Hbb__v01" / "sim-Delphes"

JET_MIN_PT = 20.0
JET_MAX_ABS_ETA = 2.5
MASS_WINDOW = (110.0, 140.0)

# Correction binning follows jet-energy/output/stage2_3/corrections.yaml.
CALIBRATION_GEN_PT_EDGES = (12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 50.0, 65.0, 90.0, 130.0)
CALIBRATION_ABS_ETA_EDGES = (0.0, 0.8, 1.5, 2.0, 2.5)
MATCH_DR_MAX = 0.2
MIN_BIN_ENTRIES = 30
# Skip bins where reco thresholds lose GenJets, which would bias the median upward.
MIN_MATCH_EFFICIENCY = 0.8
RAW_PT_SUPPORT = (5.0, 1000.0)

TRUTH_TAG = "Hbb_{fsr}__v01"
TRUTH_COLLECTION = "GenJet"

# (label, sim-Delphes tag pattern, jet collection); {fsr} is FSR or noFSR.
CELLS = (
    ("Phase-II v02 0PU", "Hbb_{fsr}__v01", "JetPUPPI"),
    # Non-PUPPI jets: PUPPI drops neutral energy when there is no pileup.
    ("Phase-II v02 0PU no-PUPPI Jet", "Hbb_{fsr}__v01", "Jet"),
    ("Phase-II v04 0PU", "Hbb_{fsr}_0PU_v04__v01", "JetPUPPI"),
    ("Phase-II v04 200PU", "Hbb_{fsr}_200PU__v01", "JetPUPPI"),
    ("Phase-II v04 200PU+MTD", "Hbb_{fsr}_200PU_MTD35ps__v01", "JetPUPPI"),
    ("Phase-I 50PU", "Hbb_{fsr}_PhaseI_50PU__v01", "Jet"),
    ("GenJet (0PU)", TRUTH_TAG, TRUTH_COLLECTION),
)
FSR_STATES = ("FSR", "noFSR")
COLORS = ("#2f6fb0", "#56b4e9", "#e69f00", "#c93c3c", "#3f8f45", "#8a5cc2", "black")

# (key, x label, bins, range)
OBSERVABLES = (
    ("jet1_pt", "Leading jet pT [GeV]", 45, (20.0, 200.0)),
    ("jet2_pt", "Subleading jet pT [GeV]", 40, (20.0, 150.0)),
    ("jet1_eta", "Leading jet eta", 40, (-2.5, 2.5)),
    ("jet2_eta", "Subleading jet eta", 40, (-2.5, 2.5)),
    ("jet1_phi", "Leading jet phi", 32, (-np.pi, np.pi)),
    ("jet2_phi", "Subleading jet phi", 32, (-np.pi, np.pi)),
    ("dijet_mass", "Dijet mass [GeV]", 50, (50.0, 150.0)),
    ("dijet_pt", "Dijet pT [GeV]", 40, (0.0, 60.0)),
    ("dijet_rapidity", "Dijet rapidity", 40, (-2.5, 2.5)),
    ("dijet_phi", "Dijet phi", 32, (-np.pi, np.pi)),
    ("n_jets", f"Jet multiplicity (pT > {JET_MIN_PT:.0f} GeV, |eta| < {JET_MAX_ABS_ETA})", 10, (-0.5, 9.5)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-files", type=int, default=5, help="Use files _1.._N of each sample")
    parser.add_argument("--correct", action="store_true", help="Derive and apply a per-card jet energy correction")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to output/reco_test_hbb, or output/reco_test_hbb_corrected with --correct")
    return parser.parse_args()


def read_jets(tag, fsr, collection, max_files):
    """Return all jets of one sample as float64 Momentum4D records, or None if files are missing."""
    paths = [SIM_DIR / tag / "root" / f"Hbb_Hbb_{fsr}__v01_{index}.root" for index in range(1, max_files + 1)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        print(f"Skipping {tag} {collection}: missing {len(missing)} of {len(paths)} files")
        return None

    fields = [f"{collection}/{collection}.{name}" for name in ("PT", "Eta", "Phi", "Mass")]
    arrays = ak.concatenate([uproot.open(path)["Delphes"].arrays(fields, library="ak") for path in paths])
    # float32 -> float64 before E^2 - p^2 in the dijet mass.
    pt, eta, phi, mass = (ak.values_astype(arrays[field], "float64") for field in fields)
    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass}, with_name="Momentum4D")


def jet_observables(jets):
    jets = jets[(jets.pt > JET_MIN_PT) & (abs(jets.eta) < JET_MAX_ABS_ETA)]
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]

    n_jets = ak.to_numpy(ak.num(jets))
    two = jets[n_jets >= 2]
    jet1, jet2 = two[:, 0], two[:, 1]
    dijet = jet1 + jet2
    observables = {
        "jet1_pt": jet1.pt, "jet2_pt": jet2.pt,
        "jet1_eta": jet1.eta, "jet2_eta": jet2.eta,
        "jet1_phi": jet1.phi, "jet2_phi": jet2.phi,
        "dijet_mass": dijet.mass, "dijet_pt": dijet.pt,
        "dijet_rapidity": dijet.rapidity, "dijet_phi": dijet.phi,
    }
    observables = {key: ak.to_numpy(value) for key, value in observables.items()}
    observables["n_jets"] = n_jets.astype(np.float64)
    observables["event_index"] = np.flatnonzero(n_jets >= 2)
    return observables, len(n_jets)


def matched_pairs(reco, truth):
    """Flat arrays of (gen pT, gen eta, reco pT or NaN) for every truth jet in |eta| < 2.5."""
    truth = truth[(truth.pt >= CALIBRATION_GEN_PT_EDGES[0]) & (abs(truth.eta) < JET_MAX_ABS_ETA)]
    pairs = ak.cartesian({"gen": truth, "reco": reco}, axis=1, nested=True)
    delta_r = pairs.gen.deltaR(pairs.reco)
    best = ak.argmin(delta_r, axis=2, keepdims=True)
    best_dr = ak.fill_none(ak.firsts(delta_r[best], axis=2), np.inf)
    best_pt = ak.fill_none(ak.firsts(pairs.reco.pt[best], axis=2), np.nan)
    reco_pt = ak.where(best_dr < MATCH_DR_MAX, best_pt, np.nan)
    return (
        ak.to_numpy(ak.flatten(truth.pt)),
        ak.to_numpy(ak.flatten(truth.eta)),
        ak.to_numpy(ak.flatten(reco_pt)),
    )


def derive_correction(reco, truth):
    """Build a correction map readable by common.jet_calibration.correction_factors."""
    gen_pt, gen_eta, reco_pt = matched_pairs(reco, truth)
    abs_eta = np.abs(gen_eta)
    rows = []
    eta_bins = []
    for eta_min, eta_max in zip(CALIBRATION_ABS_ETA_EDGES[:-1], CALIBRATION_ABS_ETA_EDGES[1:]):
        nodes = []
        for pt_min, pt_max in zip(CALIBRATION_GEN_PT_EDGES[:-1], CALIBRATION_GEN_PT_EDGES[1:]):
            in_bin = (abs_eta >= eta_min) & (abs_eta < eta_max) & (gen_pt >= pt_min) & (gen_pt < pt_max)
            matched = in_bin & np.isfinite(reco_pt)
            efficiency = matched.sum() / in_bin.sum() if in_bin.any() else 0.0
            node = {"usable": False}
            if matched.sum() >= MIN_BIN_ENTRIES and efficiency >= MIN_MATCH_EFFICIENCY:
                response = float(np.median(reco_pt[matched] / gen_pt[matched]))
                raw_median = float(np.median(reco_pt[matched]))
                node = {"usable": True, "raw_reco_pt_median": raw_median, "target_pt": raw_median / response}
            # Keep only strictly increasing nodes, as correction_factors requires.
            previous = [item for item in nodes if item["usable"]]
            if node["usable"] and previous and (
                node["raw_reco_pt_median"] <= previous[-1]["raw_reco_pt_median"]
                or node["target_pt"] <= previous[-1]["target_pt"]
            ):
                node = {"usable": False}
            nodes.append(node)
            rows.append({
                "eta_min": eta_min, "eta_max": eta_max, "gen_pt_min": pt_min, "gen_pt_max": pt_max,
                "entries": int(matched.sum()), "match_efficiency": round(float(efficiency), 3),
                "usable": node["usable"],
                "raw_reco_pt_median": round(node.get("raw_reco_pt_median", np.nan), 2),
                "correction_factor": round(node["target_pt"] / node["raw_reco_pt_median"], 3) if node["usable"] else np.nan,
            })
        usable = sum(item["usable"] for item in nodes)
        eta_bins.append({"eta_min": eta_min, "eta_max": eta_max, "supported": usable >= 2, "nodes": nodes})
    correction_map = {"raw_pt_support": list(RAW_PT_SUPPORT), "eta_bins": eta_bins}
    return correction_map, rows


def apply_correction(jets, correction_map):
    """Scale pT and mass by the correction; jets outside the map support are dropped."""
    counts = ak.num(jets)
    flat = ak.flatten(jets)
    factors, valid = correction_factors(ak.to_numpy(flat.pt), ak.to_numpy(flat.eta), correction_map)
    factors = ak.unflatten(np.where(valid, factors, 1.0), counts)
    valid = ak.unflatten(valid, counts)
    corrected = ak.zip(
        {"pt": jets.pt * factors, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass * factors},
        with_name="Momentum4D",
    )
    return corrected[valid]


def median_and_relative_width(values):
    """Median and half inter-quartile range divided by the median."""
    if values.size == 0:
        return np.nan, np.nan
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return float(q50), float(0.5 * (q75 - q25) / q50)


def jet_response(reco, truth):
    """Median and relative width of reco/gen pT for matched jets with GenJet pT > 30 GeV."""
    gen_pt, _gen_eta, reco_pt = matched_pairs(reco, truth)
    selected = np.isfinite(reco_pt) & (gen_pt > 30.0)
    return median_and_relative_width(reco_pt[selected] / gen_pt[selected])


def paired_mass_ratio(observables, truth_observables):
    """Event-by-event reco/GenJet dijet mass ratio, for events with two selected jets in both."""
    _common, reco_rows, truth_rows = np.intersect1d(
        observables["event_index"], truth_observables["event_index"], return_indices=True
    )
    return observables["dijet_mass"][reco_rows] / truth_observables["dijet_mass"][truth_rows]


def summary_row(label, fsr, calibration, observables, n_events, truth_observables=None):
    mass = observables["dijet_mass"]
    ratio = paired_mass_ratio(observables, truth_observables) if truth_observables else np.array([])
    ratio_median, ratio_width = median_and_relative_width(ratio)
    q25, q50, q75 = np.percentile(mass, [25, 50, 75]) if mass.size else (np.nan,) * 3
    in_window = (mass > MASS_WINDOW[0]) & (mass < MASS_WINDOW[1])
    return {
        "card": label,
        "fsr": fsr,
        "calibration": calibration,
        "events": n_events,
        "frac_two_jets": round(mass.size / n_events, 4),
        "mean_n_jets": round(float(np.mean(observables["n_jets"])), 3),
        "median_jet1_pt": round(float(np.median(observables["jet1_pt"])), 2),
        "median_jet2_pt": round(float(np.median(observables["jet2_pt"])), 2),
        "median_dijet_pt": round(float(np.median(observables["dijet_pt"])), 2),
        "median_dijet_mass": round(float(q50), 2),
        "half_iqr_dijet_mass": round(float(0.5 * (q75 - q25)), 2),
        "relative_mass_width": round(float(0.5 * (q75 - q25) / q50), 4),
        "frac_mass_110_140": round(float(np.sum(in_window)) / n_events, 4),
        "paired_mass_ratio_median": round(ratio_median, 4),
        "paired_mass_ratio_rel_width": round(ratio_width, 4),
    }


def plot_observable(results, key, xlabel, bins, value_range, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, fsr in zip(axes, FSR_STATES):
        for (label, _tag, _collection), color in zip(CELLS, COLORS):
            if (label, fsr) not in results:
                continue
            observables, n_events = results[(label, fsr)]
            values = observables[key]
            ax.hist(
                values, bins=bins, range=value_range,
                weights=np.full(values.size, 1.0 / n_events),
                histtype="step", linewidth=1.5, color=color,
                linestyle="--" if label.startswith("GenJet") else "-",
                label=label,
            )
        ax.set_title(fsr)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Fraction of generated events")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{key}.png", dpi=150)
    plt.close(fig)


def plot_corrections(correction_rows, output_dir):
    fig, axes = plt.subplots(1, len(CALIBRATION_ABS_ETA_EDGES) - 1, figsize=(15, 4), sharey=True)
    for ax, (eta_min, eta_max) in zip(axes, zip(CALIBRATION_ABS_ETA_EDGES[:-1], CALIBRATION_ABS_ETA_EDGES[1:])):
        for (label, _tag, _collection), color in zip(CELLS, COLORS):
            for fsr, style in zip(FSR_STATES, ("-", "--")):
                rows = [row for row in correction_rows.get((label, fsr), [])
                        if row["eta_min"] == eta_min and row["usable"]]
                if rows:
                    ax.plot([row["raw_reco_pt_median"] for row in rows], [row["correction_factor"] for row in rows],
                            marker="o", color=color, linestyle=style, label=f"{label} {fsr}")
        ax.set_title(f"{eta_min} <= |eta| < {eta_max}")
        ax.set_xlabel("Median raw jet pT [GeV]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Correction factor (1 / median response)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "correction_factors.png", dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_raw(args, output_dir):
    results = {}
    for fsr in FSR_STATES:
        for label, tag_pattern, collection in CELLS:
            jets = read_jets(tag_pattern.format(fsr=fsr), fsr, collection, args.max_files)
            if jets is not None:
                results[(label, fsr)] = jet_observables(jets)
                print(f"Loaded {label:20s} {fsr:6s} events={len(jets)}", flush=True)
    truth_label = CELLS[-1][0]
    rows = [
        summary_row(label, fsr, "raw", *result, truth_observables=results.get((truth_label, fsr), (None,))[0])
        for (label, fsr), result in results.items()
    ]
    return results, rows


def run_corrected(args, output_dir):
    truth = {fsr: read_jets(TRUTH_TAG.format(fsr=fsr), fsr, TRUTH_COLLECTION, args.max_files) for fsr in FSR_STATES}
    if any(value is None for value in truth.values()):
        raise RuntimeError("The 0PU GenJet truth samples are required for --correct")
    n_events = len(truth["FSR"])
    derive_events = np.arange(n_events) % 2 == 0
    evaluate_events = ~derive_events

    truth_observables = {fsr: jet_observables(truth[fsr][evaluate_events])[0] for fsr in FSR_STATES}
    results, rows, correction_rows, closure_rows = {}, [], {}, []
    for label, tag_pattern, collection in CELLS:
        reco = {fsr: read_jets(tag_pattern.format(fsr=fsr), fsr, collection, args.max_files) for fsr in FSR_STATES}
        if any(value is None for value in reco.values()):
            continue
        if collection == TRUTH_COLLECTION:
            for fsr in FSR_STATES:
                results[(label, fsr)] = jet_observables(reco[fsr][evaluate_events])
                rows.append(summary_row(label, fsr, "truth", *results[(label, fsr)]))
            continue

        for fsr in FSR_STATES:
            correction_map, table = derive_correction(reco[fsr][derive_events], truth[fsr][derive_events])
            correction_rows[(label, fsr)] = table
            raw = reco[fsr][evaluate_events]
            corrected = apply_correction(raw, correction_map)
            rows.append(summary_row(label, fsr, "raw", *jet_observables(raw),
                                    truth_observables=truth_observables[fsr]))
            results[(label, fsr)] = jet_observables(corrected)
            rows.append(summary_row(label, fsr, "corrected", *results[(label, fsr)],
                                    truth_observables=truth_observables[fsr]))
            gen = truth[fsr][evaluate_events]
            raw_median, raw_width = jet_response(raw, gen)
            corrected_median, corrected_width = jet_response(corrected, gen)
            closure_rows.append({
                "card": label, "fsr": fsr,
                "median_response_raw": round(raw_median, 4),
                "median_response_corrected": round(corrected_median, 4),
                "relative_resolution_raw": round(raw_width, 4),
                "relative_resolution_corrected": round(corrected_width, 4),
            })
        print(f"Corrected {label}", flush=True)

    write_csv(output_dir / "correction_factors.csv",
              [{"card": label, "fsr": fsr, **row} for (label, fsr), table in correction_rows.items() for row in table])
    write_csv(output_dir / "closure.csv", closure_rows)
    plot_corrections(correction_rows, output_dir)
    for row in closure_rows:
        print(row)
    return results, rows


def main():
    args = parse_args()
    if args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    default_dir = "reco_test_hbb_corrected" if args.correct else "reco_test_hbb"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results, rows = (run_corrected if args.correct else run_raw)(args, output_dir)
    if not results:
        raise RuntimeError("No samples found")
    for key, xlabel, bins, value_range in OBSERVABLES:
        plot_observable(results, key, xlabel, bins, value_range, output_dir)
    write_csv(output_dir / "summary.csv", rows)
    for row in rows:
        print(row)
    print(f"Wrote plots and CSV files to {output_dir}")


if __name__ == "__main__":
    main()
