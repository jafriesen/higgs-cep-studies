#!/usr/bin/env python3
"""Derive jet corrections for the FSR-recovered jet pair.

Two targets are supported:

* ``genjet`` (production default): the reco pair is calibrated to the *recovered*
  GenJet pair, i.e. the same recovery definition applied at generator level. This
  is a pure detector correction -- it removes the 5-7% reco/GenJet difference and
  leaves the out-of-cone FSR alone, which is physics (FSR costs the exclusive
  signal 12-16% and the inclusive background 4-5%, a colour-flow effect that
  separates them). Calibrating that away would delete signal/background
  information, which is why the parton target is not the production choice.
* ``parton`` (study only): the hard-process quark, which also absorbs the average
  FSR loss and is therefore strongly process dependent (34% between samples).

Note the reco side adds in-jet muons and the gen side does not: Delphes jets
exclude muons while GenJets already contain them, so this makes the two sides
describe the same physical object.

The calibrated object is the one the MVA uses: the two selected jets, each with
nearby softer jets and in-jet muons added back (mva.common.features.recovered_pair).

In bins of target pT and |eta| the median response gives one node,
(median raw pT -> median raw pT / median response), and
common.jet_calibration.correction_factors interpolates between nodes at use time.
Maps are keyed by (card, fsr_state, sample): the card sets the pileup and timing
model, and the response depends on the initiating parton and event activity.
"""
import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

vector.register_awkward()

from common.jet_calibration import correction_factors  # noqa: E402
from mva.common.features import (  # noqa: E402
    GENJET_PUPPI_DR as GENJET_MATCH_DR,
    MAX_MATCH_ABS_ETA,
    PARTON_MATCH_DR,
    RECOVERY_JET_MAX_DR,
    RECOVERY_JET_PT_MIN,
    RECOVERY_MUON_MAX_DR,
    recovered_pair,
)

SCHEMA_VERSION = 5
PARTON_PT_EDGES = (12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 50.0, 65.0, 90.0, 130.0)
ABS_ETA_EDGES = (0.0, 0.8, 1.5, 2.0, 2.5, 3.0)
RAW_PT_SUPPORT = (5.0, 1000.0)
MIN_BIN_ENTRIES = 50
SELECTION_PT_MIN = 5.0  # loose: the pair is chosen before calibration, donors go down to this
JET_FIELDS = ("PT", "Eta", "Phi", "Mass")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", required=True, help="Delphes ROOT files")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--fsr-state", required=True, choices=("FSR", "noFSR"))
    parser.add_argument("--card", required=True, help="Delphes card name the files were made with")
    parser.add_argument("--truth-pid-abs", type=int, required=True, choices=(4, 5, 21))
    parser.add_argument("--target", default="genjet", choices=("genjet", "parton"),
                        help="Calibrate to the recovered GenJet pair (production) or to the hard parton (study)")
    parser.add_argument("--collection", default="JetPUPPI")
    parser.add_argument("--output", required=True, help="corrections YAML to write or update")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--no-recovery", action="store_true",
                        help="Calibrate the bare leading pair instead, for A/B comparison")
    return parser.parse_args()


def read_event_arrays(
    path, collection, truth_pid_abs, max_events, *, include_partons=True
):
    fields = [f"{collection}/{collection}.{name}" for name in JET_FIELDS]
    fields += [f"MuonLoose/MuonLoose.{name}" for name in ("PT", "Eta", "Phi")]
    if include_partons:
        fields += [
            f"Particle/Particle.{name}"
            for name in ("PID", "Status", "IsPU", "PT", "Eta", "Phi", "Mass")
        ]
    fields += [f"GenJet/GenJet.{name}" for name in JET_FIELDS]
    arrays = uproot.open(path)["Delphes"].arrays(fields, entry_stop=max_events, library="ak")

    def zipped(prefix, names, massless=False):
        record = {
            key: ak.values_astype(arrays[f"{prefix}/{prefix}.{name}"], "float64")
            for key, name in zip(("pt", "eta", "phi", "mass"), names)
        }
        if massless:
            record["mass"] = 0.0 * record["pt"]
        return ak.zip(record, with_name="Momentum4D")

    jets = zipped(collection, JET_FIELDS)
    muons = zipped("MuonLoose", ("PT", "Eta", "Phi"), massless=True)
    gen_jets = zipped("GenJet", JET_FIELDS)

    if include_partons:
        status = arrays["Particle/Particle.Status"]
        is_pu = arrays["Particle/Particle.IsPU"]
        pid = arrays["Particle/Particle.PID"]
        hard = (status == 23) & (is_pu == 0) & (abs(pid) == truth_pid_abs)
        partons = ak.zip(
            {
                key: ak.values_astype(
                    arrays[f"Particle/Particle.{name}"][hard], "float64"
                )
                for key, name in zip(
                    ("pt", "eta", "phi", "mass"),
                    ("PT", "Eta", "Phi", "Mass"),
                )
            },
            with_name="Momentum4D",
        )
    else:
        partons = gen_jets[:, :0]
    return jets, muons, partons, gen_jets


def matched_pairs(jets, muons, partons, gen_jets, recovery=True, target="genjet"):
    """Flat (target pT, target eta, reco pT) for the selected pair.

    The reco pair is recovered with nearby jets and in-jet muons. For the GenJet
    target the gen side is recovered the same way (minus muons, already inside
    GenJets) so both sides describe the same object; for the parton target the
    hard quark is used as is.
    """
    candidates = jets[(jets.pt > SELECTION_PT_MIN) & (abs(jets.eta) < MAX_MATCH_ABS_ETA)]
    candidates = candidates[ak.argsort(candidates.pt, axis=1, ascending=False)]
    keep = ak.num(candidates) >= 2
    if target == "parton":
        keep = keep & (ak.num(partons) == 2)
    else:
        keep = keep & (ak.num(gen_jets) >= 2)
    candidates, muons, partons, gen_jets = (
        candidates[keep], muons[keep], partons[keep], gen_jets[keep]
    )
    pair, others = candidates[:, :2], candidates[:, 2:]
    if recovery:
        first, second, _n1, _n2 = recovered_pair(pair, others, muons)
    else:
        first, second = pair[:, 0], pair[:, 1]

    rows = np.arange(len(pair))
    gen_pt, gen_eta, reco_pt = [], [], []
    for index, jet in enumerate((first, second)):
        core = pair[:, index]
        if target == "parton":
            delta = np.stack([ak.to_numpy(core.deltaR(partons[:, i])) for i in (0, 1)], axis=1)
            best = np.argmin(delta, axis=1)
            close = delta[rows, best] < PARTON_MATCH_DR
            target_pt = np.stack([ak.to_numpy(partons[:, i].pt) for i in (0, 1)], axis=1)[rows, best]
            target_eta = np.stack([ak.to_numpy(partons[:, i].eta) for i in (0, 1)], axis=1)[rows, best]
        else:
            # match the core reco jet to its GenJet, then recover the gen side too
            best = ak.to_numpy(ak.fill_none(ak.firsts(ak.argsort(gen_jets.deltaR(core), axis=1)), -1))
            matched = gen_jets[rows, best]
            close = ak.to_numpy(matched.deltaR(core)) < GENJET_MATCH_DR
            if recovery:
                near = (gen_jets.deltaR(matched) < RECOVERY_JET_MAX_DR) & (gen_jets.pt > RECOVERY_JET_PT_MIN)
                near = near & (ak.local_index(gen_jets, axis=1) != best[:, np.newaxis])
                extra = gen_jets[near]
                components = {
                    axis: getattr(matched, axis) + ak.sum(getattr(extra, axis), axis=1)
                    for axis in ("px", "py", "pz", "E")
                }
                matched = ak.zip(components, with_name="Momentum4D")
            target_pt = ak.to_numpy(matched.pt)
            target_eta = ak.to_numpy(matched.eta)
        gen_pt.append(target_pt[close])
        gen_eta.append(target_eta[close])
        reco_pt.append(ak.to_numpy(jet.pt)[close])
    return np.concatenate(gen_pt), np.concatenate(gen_eta), np.concatenate(reco_pt)


def derive(gen_pt, gen_eta, reco_pt):
    abs_eta = np.abs(gen_eta)
    eta_bins = []
    for eta_min, eta_max in zip(ABS_ETA_EDGES[:-1], ABS_ETA_EDGES[1:]):
        nodes = []
        for pt_min, pt_max in zip(PARTON_PT_EDGES[:-1], PARTON_PT_EDGES[1:]):
            selected = (abs_eta >= eta_min) & (abs_eta < eta_max) & (gen_pt >= pt_min) & (gen_pt < pt_max)
            node = {"pt_bin": [pt_min, pt_max], "entries": int(selected.sum()), "usable": False}
            if selected.sum() >= MIN_BIN_ENTRIES:
                response = float(np.median(reco_pt[selected] / gen_pt[selected]))
                raw_median = float(np.median(reco_pt[selected]))
                node.update(
                    {
                        "usable": True,
                        "raw_reco_pt_median": raw_median,
                        "response_median": response,
                        "target_pt": raw_median / response,
                        "correction_factor": 1.0 / response,
                    }
                )
            kept = [item for item in nodes if item["usable"]]
            if node["usable"] and kept and (
                node["raw_reco_pt_median"] <= kept[-1]["raw_reco_pt_median"]
                or node["target_pt"] <= kept[-1]["target_pt"]
            ):
                # correction_factors needs strictly increasing nodes
                node["usable"] = False
                node["invalid_reason"] = "not monotonic"
            nodes.append(node)
        usable = sum(node["usable"] for node in nodes)
        eta_bins.append(
            {"eta_min": eta_min, "eta_max": eta_max, "supported": usable >= 2, "nodes": nodes}
        )
    return {"raw_pt_support": list(RAW_PT_SUPPORT), "eta_bins": eta_bins}


def closure(correction_map, gen_pt, gen_eta, reco_pt):
    factors, valid = correction_factors(reco_pt, gen_eta, correction_map)
    ratio = np.where(valid, reco_pt * factors, np.nan) / gen_pt
    ratio = ratio[np.isfinite(ratio)]
    raw = reco_pt / gen_pt
    q25, q50, q75 = np.percentile(ratio, [25, 50, 75])
    return {
        "matched_jets": int(gen_pt.size),
        "raw_response_median": round(float(np.median(raw)), 4),
        "corrected_response_median": round(float(q50), 4),
        "corrected_relative_width": round(float(0.5 * (q75 - q25) / q50), 4),
    }


def main():
    import yaml

    args = parse_args()
    gen_pt, gen_eta, reco_pt = [], [], []
    for index, path in enumerate(args.files, start=1):
        pieces = matched_pairs(
            *read_event_arrays(
                path,
                args.collection,
                args.truth_pid_abs,
                args.max_events,
                include_partons=args.target == "parton",
            ),
            recovery=not args.no_recovery,
            target=args.target,
        )
        for target, piece in zip((gen_pt, gen_eta, reco_pt), pieces):
            target.append(piece)
        print(f"  [{index}/{len(args.files)}] {Path(path).name}: {pieces[0].size} matched jets", flush=True)
    gen_pt, gen_eta, reco_pt = (np.concatenate(values) for values in (gen_pt, gen_eta, reco_pt))

    correction_map = derive(gen_pt, gen_eta, reco_pt)
    correction_map.update(
        {
            "collection": args.collection,
            "sample": args.sample,
            "fsr_state": args.fsr_state,
            "card": args.card,
            "target": ("recovered GenJet pair" if args.target == "genjet"
                        else "hard-process parton (status 23)"),
            "truth_pid_abs": args.truth_pid_abs,
            "recovery": None if args.no_recovery else {
                "jet_pt_min_gev": RECOVERY_JET_PT_MIN,
                "jet_max_dr": RECOVERY_JET_MAX_DR,
                "muon_max_dr": RECOVERY_MUON_MAX_DR,
            },
            "parton_pt_edges": list(PARTON_PT_EDGES),
            "abs_eta_edges": list(ABS_ETA_EDGES),
            "parton_match_dr": PARTON_MATCH_DR,
            "derivation_files": [str(path) for path in args.files],
            "closure": closure(correction_map, gen_pt, gen_eta, reco_pt),
        }
    )
    print(f"closure: {correction_map['closure']}")

    output = Path(args.output)
    document = {"schema_version": SCHEMA_VERSION, "maps": {}}
    if output.exists():
        document = yaml.safe_load(output.read_text(encoding="utf-8")) or document
        if document.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"{output} has schema {document.get('schema_version')}, expected {SCHEMA_VERSION}")
    # layout matches common.jet_calibration.load_correction_map: maps[fsr_state][sample].
    # One file per card; the card is recorded inside each entry.
    document.setdefault("maps", {}).setdefault(args.fsr_state, {})[args.sample] = correction_map
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"wrote {output}: maps[{args.fsr_state}][{args.sample}]  (card {args.card})")


if __name__ == "__main__":
    main()
