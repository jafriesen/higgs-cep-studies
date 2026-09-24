#!/usr/bin/env python3
"""Pre-MVA cutflow and b-jet pair choice for the FSR+MTD H(bb) samples.

For each event the pair is chosen four ways, then recovered and calibrated exactly as
mva/common/features.py does (per-sample map, donor jets pT>5 within dR<1.5, muons):

  leading   two highest raw-pT pool jets (|eta|<3, raw pT>=5): the current MVA choice
  truth     the reco jets matched to the status-23 b quarks (oracle upper bound)
  btag      emulated tagger: a jet within dR<0.4 of a status-23 b is tagged with
            eff_b, any other jet with the light mistag rate (parameters.yaml); the two
            highest raw-pT tagged jets are used, filled with leading jets if needed
  backtoback among the three leading pool jets, the pair closest to delta-phi = pi

The cutflow counts every generated event. The flat tag factor eff_b^2 the MVA applies
is NOT in the leading/truth/backtoback rows; the btag row already contains its own.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import vector
import yaml
from uproot.source.file import MemmapSource

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mva.common import features as F  # noqa: E402
from mva.common.dataset import (  # noqa: E402
    _madgraph_inputs,
    _superchic_inputs,
    load_component_correction_map,
)
from mva.common.protons import (  # noqa: E402
    load_pps_config,
    parse_hepmc_protons,
    parse_lhe_protons,
    real_proton_pass,
)

STRATEGIES = ("leading", "truth", "btag", "backtoback")
TAG_DR = 0.4
OUTPUT = REPO / "analysis/MVA/output/hbb_preselection_study"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs="+",
                        default=["Hbb_fsr", "QCDbb_fsr", "QEDbb_fsr", "QCDbb_madgraph_fsr"])
    parser.add_argument("--madgraph-campaign", default="QCDbb__v05")
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def pair_dphi(phi, indices):
    first = F._gather_jagged(phi, indices[:, [0]])[:, 0]
    second = F._gather_jagged(phi, indices[:, [1]])[:, 0]
    return np.abs(np.asarray(F.wrap_phi(first - second)))


def choose_pairs(raw_pt, raw_eta, raw_phi, pool_mask, original_index, parton_eta,
                 parton_phi, hard_valid, tagging, rng):
    order = ak.argsort(raw_pt[pool_mask], axis=1, ascending=False)
    pool_index = original_index[pool_mask][order]
    leading = F._padded_numpy(pool_index, width=2, fill=-1).astype(np.int64)

    # emulated tagger on the pT-ordered pool
    pool_eta = raw_eta[pool_mask][order]
    pool_phi = raw_phi[pool_mask][order]
    dr = []
    for column in (0, 1):
        deta = pool_eta - parton_eta[:, column]
        dphi = F.wrap_phi(pool_phi - parton_phi[:, column])
        dr.append(np.hypot(deta, dphi))
    is_b = ((dr[0] < TAG_DR) | (dr[1] < TAG_DR)) & ak.Array(hard_valid)
    counts = ak.to_numpy(ak.num(pool_index))
    draws = ak.unflatten(rng.random(int(counts.sum())), counts)
    tagged = ak.where(is_b, draws < tagging["eff_b"], draws < tagging["mistag_light_to_b"])
    # tagged first, each group in pT order; stable sort keeps pT order inside groups
    rank = ak.argsort(ak.values_astype(~tagged, np.int8), axis=1, stable=True)
    btag = F._padded_numpy(pool_index[rank], width=2, fill=-1).astype(np.int64)
    n_tagged = ak.to_numpy(ak.sum(tagged, axis=1))

    # among the three leading jets, the most back-to-back pair
    top3 = F._padded_numpy(pool_index, width=3, fill=-1).astype(np.int64)
    candidates = [(0, 1), (0, 2), (1, 2)]
    scores = []
    for a, b in candidates:
        idx = top3[:, [a, b]]
        score = pair_dphi(raw_phi, idx)
        score[np.any(idx < 0, axis=1)] = -1.0
        scores.append(score)
    best = np.argmax(np.stack(scores, axis=1), axis=1)
    backtoback = np.stack([top3[np.arange(len(best)), [candidates[k][0] for k in best]],
                           top3[np.arange(len(best)), [candidates[k][1] for k in best]]],
                          axis=1)
    return {"leading": leading, "btag": btag, "backtoback": backtoback}, n_tagged


def process_file(task):
    (path, correction_map, tagging, proton_kind, proton_file, proton_offset, pps, seed) = task
    recovery = correction_map["recovery"]
    names = [F.branch_name("JetPUPPI", f) for f in ("PT", "Eta", "Phi", "Mass", "NCharged")]
    names += [F.branch_name("MuonLoose", f) for f in ("PT", "Eta", "Phi")]
    names += [F.branch_name("Particle", f) for f in ("PID", "Status", "IsPU", "PT", "Eta", "Phi")]
    names += [F.branch_name("GenJet", f) for f in ("PT", "Eta", "Phi")]
    with uproot.open(path, handler=MemmapSource) as handle:
        arrays = handle["Delphes"].arrays(names, library="ak")

    def field(coll, name):
        return ak.values_astype(arrays[F.branch_name(coll, name)], "float64")

    raw_pt, raw_eta = field("JetPUPPI", "PT"), field("JetPUPPI", "Eta")
    raw_phi, raw_mass = field("JetPUPPI", "Phi"), field("JetPUPPI", "Mass")
    n_events = len(raw_pt)
    original_index = ak.local_index(raw_pt)
    acceptance = np.abs(raw_eta) < 3.0
    pool_mask = acceptance & (raw_pt >= float(recovery["jet_pt_min_gev"]))

    pid, status, is_pu = (arrays[F.branch_name("Particle", k)] for k in ("PID", "Status", "IsPU"))
    hard = (status == 23) & (is_pu == 0)
    quark, antiquark = hard & (pid == 5), hard & (pid == -5)
    hard_valid = ak.to_numpy((ak.sum(quark, axis=1) == 1) & (ak.sum(antiquark, axis=1) == 1))

    def hard_values(name):
        values = field("Particle", name)
        return np.column_stack([ak.to_numpy(ak.fill_none(ak.firsts(values[m]), np.nan))
                                for m in (quark, antiquark)])

    parton_pt, parton_eta, parton_phi = hard_values("PT"), hard_values("Eta"), hard_values("Phi")

    # truth pair: exactly the features.py definition (parton -> GenJet -> reco jet)
    gen_mask = np.abs(field("GenJet", "Eta")) < F.MAX_MATCH_ABS_ETA
    gen_eta, gen_phi = field("GenJet", "Eta")[gen_mask], field("GenJet", "Phi")[gen_mask]
    gen_index, _, complete_gen = F._match_two_to_many(
        parton_eta, parton_phi, gen_eta, gen_phi, F.PARTON_GENJET_DR)
    complete_gen &= hard_valid
    reco_mask = np.abs(raw_eta) < F.MAX_MATCH_ABS_ETA
    reco_local, _, complete_reco = F._match_two_to_many(
        F._gather_jagged(gen_eta, gen_index), F._gather_jagged(gen_phi, gen_index),
        raw_eta[reco_mask], raw_phi[reco_mask], F.GENJET_PUPPI_DR)
    truth_indices = F._gather_jagged(original_index[reco_mask], reco_local, fill=-1).astype(np.int64)
    truth_matched = complete_gen & complete_reco

    rng = np.random.default_rng(seed)
    pairs, n_tagged = choose_pairs(raw_pt, raw_eta, raw_phi, pool_mask, original_index,
                                   parton_eta, parton_phi, hard_valid, tagging, rng)
    pairs["truth"] = np.where(truth_matched[:, None], truth_indices, pairs["leading"])

    out = {
        "n_events": n_events,
        "hard_valid": hard_valid,
        "truth_matched": truth_matched,
        "n_tagged": n_tagged,
        "gen_matched": complete_gen,
        "parton_min_pt": np.min(parton_pt, axis=1),
        "parton_max_abs_eta": np.max(np.abs(parton_eta), axis=1),
        "event_index": np.arange(n_events),
    }
    muon = [field("MuonLoose", k) for k in ("PT", "Eta", "Phi")]
    for name in STRATEGIES:
        chosen = pairs[name]
        recovered, _ = F._recovered_chosen_pair(
            chosen, pool_mask, original_index, raw_pt, raw_eta, raw_phi, raw_mass,
            field("JetPUPPI", "NCharged"), *muon, correction_map)
        pt = recovered["pt"]
        has_two = (np.all(chosen >= 0, axis=1)
                   & np.all(recovered["valid"].astype(bool), axis=1)
                   & np.all(np.isfinite(pt) & (pt >= F.MIN_JET_PT), axis=1))
        j1 = vector.array({k: np.nan_to_num(recovered[k][:, 0]) for k in ("pt", "eta", "phi", "mass")})
        j2 = vector.array({k: np.nan_to_num(recovered[k][:, 1]) for k in ("pt", "eta", "phi", "mass")})
        dijet = j1 + j2
        same = truth_matched & np.all(np.sort(chosen, axis=1) == np.sort(truth_indices, axis=1), axis=1)
        n_true = sum(((chosen[:, c] == truth_indices[:, 0]) | (chosen[:, c] == truth_indices[:, 1]))
                     .astype(int) for c in (0, 1))
        one = truth_matched & (n_true == 1)
        out[f"{name}_has_two"] = has_two
        out[f"{name}_core_dphi"] = pair_dphi(raw_phi, chosen)
        out[f"{name}_dphi"] = np.abs(np.asarray(F.wrap_phi(recovered["phi"][:, 0] - recovered["phi"][:, 1])))
        out[f"{name}_mjj"] = np.where(has_two, dijet.mass, np.nan)
        out[f"{name}_y"] = np.where(has_two, dijet.rapidity, np.nan)
        out[f"{name}_is_truth"] = same
        out[f"{name}_one_truth"] = one
        out[f"{name}_pt1"] = pt[:, 0]
        out[f"{name}_pt2"] = pt[:, 1]

    if proton_kind is not None:
        parser = parse_hepmc_protons if proton_kind == "hepmc" else parse_lhe_protons
        protons = parser(proton_file, np.arange(n_events) + proton_offset, pps["sqrt_s"])
        passed, _l, _r, mx, yx = real_proton_pass(
            protons["xi_left"], protons["xi_right"], pps, np.random.default_rng(seed + 1))
        out["proton_pass"] = passed
        out["proton_mx"] = mx
        out["proton_yx"] = yx
    return out


def merge(pieces):
    merged = {"n_events": sum(p["n_events"] for p in pieces)}
    for key in pieces[0]:
        if key != "n_events":
            merged[key] = np.concatenate([p[key] for p in pieces])
    return merged


def cutflow(data, name, mass_window, dphi_cut=3.0, mjj_range=(50.0, 150.0), dy_cut=0.2):
    n = data["n_events"]
    steps = [("generated", np.ones(len(data["hard_valid"]), bool))]
    keep = data[f"{name}_has_two"].copy()
    steps.append(("two jets pT>15 after recovery", keep.copy()))
    if dphi_cut is not None:
        keep &= data[f"{name}_dphi"] > dphi_cut
    steps.append((f"dphi > {dphi_cut}", keep.copy()))
    if mjj_range is not None:
        mjj = data[f"{name}_mjj"]
        keep &= (mjj >= mjj_range[0]) & (mjj <= mjj_range[1])
    steps.append((f"mjj in {mjj_range}", keep.copy()))
    if "proton_pass" in data:
        keep &= data["proton_pass"]
        steps.append(("protons in PPS acceptance", keep.copy()))
        keep &= (data["proton_mx"] >= mass_window[0]) & (data["proton_mx"] <= mass_window[1])
        steps.append((f"mX in {mass_window}", keep.copy()))
        keep &= np.abs(data["proton_yx"] - data[f"{name}_y"]) < dy_cut
        steps.append((f"|yX - y_jj| < {dy_cut}", keep.copy()))
    return [(label, mask.sum() / n) for label, mask in steps], keep


def main():
    args = parse_args()
    config = yaml.safe_load((REPO / "mva/Hbb/config.yaml").read_text())
    parameters = yaml.safe_load((REPO / config["parameters"]).read_text())
    tagging = parameters["tagging"]
    pps = load_pps_config(REPO / config["pps_config"])
    specs = {spec["name"]: spec for spec in config["components"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mass_window = tuple(config["mass_window_gev"])
    report = {}

    for component in args.components:
        spec = specs[component]
        if spec["generator"] == "superchic":
            campaign, _sub, _dir, files, kind, proton_files, offsets = _superchic_inputs(
                spec, args.max_files)
            correction_map = load_component_correction_map(REPO, spec, campaign)
            tasks = [(p, correction_map, tagging, kind, pf, off, pps, args.seed + i)
                     for i, (p, pf, off) in enumerate(zip(files, proton_files, offsets))]
        else:
            campaign = args.madgraph_campaign
            _cfg, _dir, files = _madgraph_inputs(spec, campaign, args.max_files)
            correction_map = load_component_correction_map(REPO, spec, campaign)
            tasks = [(p, correction_map, tagging, None, None, 0, pps, args.seed + i)
                     for i, p in enumerate(files)]
        print(f"{component}: {len(tasks)} files", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            data = merge(list(pool.map(process_file, tasks)))
        np.savez_compressed(args.output_dir / f"{component}.npz",
                            **{k: v for k, v in data.items() if k != "n_events"},
                            n_events=data["n_events"])

        entry = {"events": int(data["n_events"]),
                 "truth_matched_fraction": float(data["truth_matched"].mean())}
        print(f"\n=== {component} ({data['n_events']} events, truth pair reconstructable "
              f"{data['truth_matched'].mean():.3f}) ===")
        for name in STRATEGIES:
            flow, final = cutflow(data, name, mass_window)
            has_two = data[f"{name}_has_two"]
            matched = data["truth_matched"] & has_two
            purity = data[f"{name}_is_truth"][matched].mean() if matched.any() else np.nan
            one = data[f"{name}_one_truth"][matched].mean() if matched.any() else np.nan
            entry[name] = {"cutflow": {k: float(v) for k, v in flow},
                           "pair_is_truth_given_matched": float(purity),
                           "pair_has_one_truth_given_matched": float(one)}
            print(f"  [{name}] pair==truth {purity:.3f}  one-of-two {one:.3f}")
            for label, value in flow:
                print(f"      {label:34s} {value:.4f}")
        report[component] = entry


    (args.output_dir / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"\nWrote {args.output_dir / 'report.yaml'}")


if __name__ == "__main__":
    main()
