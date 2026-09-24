#!/usr/bin/env python3
"""Prototype color-flow observables for exclusive-Hbb vs QCD-bb discrimination.

Standalone study script: reads Delphes ROOT files directly, applies a central
dijet selection (two jets with pT >= 15 GeV, |delta_phi_jj| > 3,
50 <= mjj <= 150 GeV; no outer-track or proton/PPS requirements), and computes
per-event color-flow observables from EFlowTrack:
  - track activity outside the two b-jet cones, split into the true interjet
    corridor, two beam-side outer regions, and the remaining side region
  - track mini-jets (FastJet anti-kt R = 0.2, pT >= 3 GeV) built from the
    out-of-jet tracks
  - beam-side pseudorapidity gap sizes and the location of the hardest extra
    activity
  - ordinary extra-jet activity and its location relative to the dijet system
  - jet pull vectors and a projection-based interjet bridge asymmetry
Plots normalized comparisons for Hbb signal vs SuperChic QCDbb vs MadGraph
inclusive QCDbb, with separation scores per observable.

Requires the scikit-hep `fastjet` package (pip install --user fastjet) on top
of the LCG view used by the analysis, e.g.:
  source setup_env.sh
  python3 analysis/MVA/prototype_color_flow.py
"""
import argparse
import os
import site
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import awkward as ak
import matplotlib
import numpy as np
import uproot
import vector

try:
    import fastjet
except ImportError:  # LCG views may suppress the pip --user site-packages
    sys.path.append(site.getusersitepackages())
    import fastjet

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

vector.register_awkward()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.MVA.color_flow_geometry import (  # noqa: E402
    activity_region_masks,
    projected_bridge_asymmetry,
    wrap_phi,
)

MIN_JET_PT = 15.0
JET_CONE_R = 0.4
TRACK_MIN_PT = 0.5
GAP_TRACK_MIN_PT = 0.5
TRACK_MAX_ABS_ETA = 2.5
TRACKER_ETA = TRACK_MAX_ABS_ETA
MIN_DIJET_DELTA_PHI = 3.0
PRE_MVA_DIJET_MASS_RANGE_GEV = (50.0, 150.0)
TRACK_JET_R = 0.2
TRACK_JET_MIN_PT = 3.0
EXTRA_JET_MIN_PT = 15.0
REGION_LABELS = ("interjet", "outer+", "outer-", "side")

SAMPLES = (
    {
        "name": "Hbb",
        "label": "H->bb (SuperChic)",
        "input_dir": REPO / "output-superchic/Hbb/Hbb__v01/sim-Delphes/Hbb_noFSR_200PU__v01/root",
        "max_files": 15,
        "color": "#0072B2",
        "linestyle": "-",
    },
    {
        "name": "QCDbb",
        "label": "SuperChic QCDbb",
        "input_dir": REPO / "output-superchic/QCDbb/QCDbb__v01/sim-Delphes/QCDbb_noFSR_200PU__v01/root",
        "max_files": 15,
        "color": "#E69F00",
        "linestyle": "--",
    },
    {
        "name": "QCDbb_madgraph",
        "label": "MadGraph QCDbb (inclusive)",
        "input_dir": REPO / "output-madgraph/QCDbb/QCDbb__v02/sim-Delphes/QCDbb_DPy8_noFSR_200PU__v02/root",
        "max_files": 5,
        "color": "#D55E00",
        "linestyle": "-.",
    },
)

OBSERVABLE_GROUPS = (
    (
        "color_flow_track_activity",
        "Out-of-jet track activity",
        (4, 3),
        (
            ("n_tracks_outside_jets", "N tracks outside jets", np.arange(-0.5, 40.5, 1.0)),
            ("sum_track_pt_outside_jets", "Sum track pT outside jets [GeV]", np.linspace(0.0, 60.0, 41)),
            ("max_track_pt_outside_jets", "Max track pT outside jets [GeV]", np.linspace(0.0, 20.0, 41)),
            ("n_tracks_projected_bridge", "N tracks in projected bridge", np.arange(-0.5, 30.5, 1.0)),
            ("sum_track_pt_projected_bridge", "Sum track pT in projected bridge [GeV]", np.linspace(0.0, 60.0, 41)),
            ("n_tracks_outer_positive", "N tracks outer positive", np.arange(-0.5, 20.5, 1.0)),
            ("n_tracks_outer_negative", "N tracks outer negative", np.arange(-0.5, 20.5, 1.0)),
            ("sum_track_pt_outer_positive", "Sum track pT outer positive [GeV]", np.linspace(0.0, 30.0, 41)),
            ("sum_track_pt_outer_negative", "Sum track pT outer negative [GeV]", np.linspace(0.0, 30.0, 41)),
            ("n_tracks_projected_side", "N tracks outside the projected bridge/outer regions", np.arange(-0.5, 30.5, 1.0)),
            ("sum_track_pt_projected_side", "Projected side-region track pT [GeV]", np.linspace(0.0, 60.0, 41)),
        ),
    ),
    (
        "color_flow_track_jets",
        "Extra track jets (anti-kt R = 0.2, pT >= 3 GeV)",
        (4, 3),
        (
            ("n_extra_track_jets", "N extra track jets", np.arange(-0.5, 8.5, 1.0)),
            ("leading_extra_track_jet_pt", "Leading extra track jet pT [GeV]", np.linspace(0.0, 40.0, 41)),
            ("extra_track_jet_ht", "Extra track jet HT [GeV]", np.linspace(0.0, 60.0, 41)),
            ("n_interjet_track_jets", "N interjet track jets", np.arange(-0.5, 6.5, 1.0)),
            ("leading_interjet_track_jet_pt", "Leading interjet track jet pT [GeV]", np.linspace(0.0, 40.0, 41)),
            ("n_outer_track_jets", "N outer track jets", np.arange(-0.5, 6.5, 1.0)),
            ("leading_outer_track_jet_pt", "Leading outer track jet pT [GeV]", np.linspace(0.0, 40.0, 41)),
            ("n_side_track_jets", "N side-region track jets", np.arange(-0.5, 6.5, 1.0)),
            ("leading_side_track_jet_pt", "Leading side-region track jet pT [GeV]", np.linspace(0.0, 40.0, 41)),
            ("track_clumping_fraction", "Track-jet pT / out-of-jet track pT", np.linspace(0.0, 1.05, 43)),
            ("leading_track_jet_fraction", "Leading track-jet pT / out-of-jet track pT", np.linspace(0.0, 1.05, 43)),
        ),
    ),
    (
        "color_flow_extra_jets",
        "Ordinary extra PUPPI jets (pT >= 15 GeV)",
        (3, 3),
        (
            ("n_extra_jets", "N extra PUPPI jets", np.arange(-0.5, 8.5, 1.0)),
            ("leading_extra_jet_pt", "Leading extra PUPPI jet pT [GeV]", np.linspace(0.0, 100.0, 51)),
            ("extra_jet_ht", "Extra PUPPI jet HT [GeV]", np.linspace(0.0, 160.0, 51)),
            ("n_interjet_extra_jets", "N interjet extra PUPPI jets", np.arange(-0.5, 6.5, 1.0)),
            ("leading_interjet_extra_jet_pt", "Leading interjet extra-jet pT [GeV]", np.linspace(0.0, 100.0, 51)),
            ("n_outer_extra_jets", "N beam-side outer extra PUPPI jets", np.arange(-0.5, 6.5, 1.0)),
            ("leading_outer_extra_jet_pt", "Leading beam-side extra-jet pT [GeV]", np.linspace(0.0, 100.0, 51)),
            ("n_side_extra_jets", "N side-region extra PUPPI jets", np.arange(-0.5, 6.5, 1.0)),
            ("leading_side_extra_jet_pt", "Leading side-region extra-jet pT [GeV]", np.linspace(0.0, 100.0, 51)),
        ),
    ),
    (
        "color_flow_gaps",
        "Beam-side gaps and hardest extra activity",
        (2, 3),
        (
            ("positive_gap_size", "Positive-side gap size", np.linspace(0.0, 5.0, 41)),
            ("negative_gap_size", "Negative-side gap size", np.linspace(0.0, 5.0, 41)),
            ("minimum_gap_size", "Minimum gap size", np.linspace(0.0, 5.0, 41)),
            ("sum_gap_size", "Sum of gap sizes", np.linspace(0.0, 8.0, 41)),
            ("leading_extra_activity_eta", "Leading extra activity eta", np.linspace(-2.5, 2.5, 41)),
            ("leading_extra_activity_region", "Leading extra activity region", np.arange(-0.5, 4.5, 1.0)),
        ),
    ),
    (
        "color_flow_pull",
        "Jet pull and interjet bridge",
        (2, 3),
        (
            ("jet1_pull_angle", "Jet1 pull angle w.r.t. jet2 [rad]", np.linspace(0.0, np.pi, 41)),
            ("jet2_pull_angle", "Jet2 pull angle w.r.t. jet1 [rad]", np.linspace(0.0, np.pi, 41)),
            ("jet1_pull_magnitude", "Jet1 pull magnitude", np.linspace(0.0, 0.02, 41)),
            ("jet2_pull_magnitude", "Jet2 pull magnitude", np.linspace(0.0, 0.02, 41)),
            ("combined_pull_alignment", "Combined pull alignment", np.linspace(-1.0, 1.0, 41)),
            ("interjet_bridge_asymmetry_projected", "Projected interjet bridge asymmetry", np.linspace(-1.0, 1.0, 41)),
        ),
    ),
)
OBSERVABLE_NAMES = tuple(
    name for _f, _t, _s, obs_list in OBSERVABLE_GROUPS for name, _x, _b in obs_list
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="JetPUPPI", help="Jet collection branch")
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "analysis/MVA/output/color_flow_proto"),
        help="Output directory for plots and the separation summary.",
    )
    return parser.parse_args()


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def scalar_max(values, mask, empty=0.0):
    """Per-event max of values[mask]; `empty` where no entry passes."""
    out = ak.to_numpy(ak.max(values[mask], axis=1, mask_identity=False))
    return np.where(np.isfinite(out), out, empty)


def jet_pull(track_pt, track_eta, track_phi, jet, other_jet):
    """Pull magnitude and cos(pull angle w.r.t. other jet), per event."""
    deta = track_eta - jet.eta[:, np.newaxis]
    dphi = wrap_phi(track_phi - jet.phi[:, np.newaxis])
    dr = np.hypot(deta, dphi)
    in_jet = (dr < JET_CONE_R) & (track_pt >= TRACK_MIN_PT)
    jet_pt = ak.to_numpy(jet.pt)
    pull_eta = ak.to_numpy(ak.sum((track_pt * deta * dr)[in_jet], axis=1)) / jet_pt
    pull_phi = ak.to_numpy(ak.sum((track_pt * dphi * dr)[in_jet], axis=1)) / jet_pt
    pull_mag = np.hypot(pull_eta, pull_phi)
    link_eta = ak.to_numpy(other_jet.eta - jet.eta)
    link_phi = ak.to_numpy(wrap_phi(other_jet.phi - jet.phi))
    link_mag = np.hypot(link_eta, link_phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_angle = (pull_eta * link_eta + pull_phi * link_phi) / (pull_mag * link_mag)
    return pull_mag, np.clip(cos_angle, -1.0, 1.0)


def cluster_track_jets(track_pt, track_eta, track_phi, mask):
    """Anti-kt R=0.2 track jets (pT >= 3 GeV) from masked tracks, pT-sorted."""
    particles = ak.zip(
        {
            "pt": track_pt[mask],
            "eta": track_eta[mask],
            "phi": track_phi[mask],
            "mass": ak.zeros_like(track_pt[mask]),
        },
        with_name="Momentum4D",
    )
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, TRACK_JET_R)
    raw = fastjet.ClusterSequence(particles, jet_def).inclusive_jets(min_pt=TRACK_JET_MIN_PT)
    jets = ak.zip(
        {"px": raw.px, "py": raw.py, "pz": raw.pz, "E": raw.E}, with_name="Momentum4D"
    )
    return jets[ak.argsort(jets.pt, axis=1, ascending=False)]


def process_file(path, tree_name, collection):
    jet_fields = ("PT", "Eta", "Phi", "Mass")
    required = [branch_name(collection, field) for field in jet_fields]
    required.extend(
        branch_name("EFlowTrack", field) for field in ("PT", "Eta", "Phi", "IsRecoPU")
    )
    with uproot.open(path) as root_file:
        tree = root_file[tree_name]
        arrays = tree.arrays(required, library="ak")

    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    phi = ak.values_astype(arrays[branch_name(collection, "Phi")], "float64")
    mass = ak.values_astype(arrays[branch_name(collection, "Mass")], "float64")
    jet_mask = pt >= MIN_JET_PT
    pt, eta, phi, mass = pt[jet_mask], eta[jet_mask], phi[jet_mask], mass[jet_mask]
    has_two = ak.to_numpy(ak.num(pt) >= 2)
    if not np.any(has_two):
        return None

    jets = ak.zip(
        {"pt": pt[has_two], "eta": eta[has_two], "phi": phi[has_two], "mass": mass[has_two]},
        with_name="Momentum4D",
    )
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    leading, subleading = jets[:, 0], jets[:, 1]
    mjj = ak.to_numpy((leading + subleading).mass)
    abs_dphi_jj = np.abs(ak.to_numpy(wrap_phi(leading.phi - subleading.phi)))
    keep = (
        (abs_dphi_jj > MIN_DIJET_DELTA_PHI)
        & (mjj >= PRE_MVA_DIJET_MASS_RANGE_GEV[0])
        & (mjj <= PRE_MVA_DIJET_MASS_RANGE_GEV[1])
    )
    if not np.any(keep):
        return None
    keep_ak = ak.Array(keep)
    jets = jets[keep_ak]
    leading, subleading = jets[:, 0], jets[:, 1]

    track_pt = ak.values_astype(arrays[branch_name("EFlowTrack", "PT")], "float64")[has_two][keep_ak]
    track_eta = ak.values_astype(arrays[branch_name("EFlowTrack", "Eta")], "float64")[has_two][keep_ak]
    track_phi = ak.values_astype(arrays[branch_name("EFlowTrack", "Phi")], "float64")[has_two][keep_ak]
    track_is_pu = ak.values_astype(
        arrays[branch_name("EFlowTrack", "IsRecoPU")], "float64"
    )[has_two][keep_ak]
    pv_track = (
        (track_is_pu == 0)
        & (track_pt >= TRACK_MIN_PT)
        & (np.abs(track_eta) < TRACK_MAX_ABS_ETA)
    )
    track_pt, track_eta, track_phi = (
        track_pt[pv_track],
        track_eta[pv_track],
        track_phi[pv_track],
    )

    dr1 = np.hypot(
        track_eta - leading.eta[:, np.newaxis],
        wrap_phi(track_phi - leading.phi[:, np.newaxis]),
    )
    dr2 = np.hypot(
        track_eta - subleading.eta[:, np.newaxis],
        wrap_phi(track_phi - subleading.phi[:, np.newaxis]),
    )
    outside = (dr1 > JET_CONE_R) & (dr2 > JET_CONE_R) & (track_pt >= TRACK_MIN_PT)

    eta_pos = np.maximum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    eta_neg = np.minimum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    track_interjet, track_outer_pos, track_outer_neg, track_side = activity_region_masks(
        track_eta, track_phi, leading, subleading
    )
    interjet = outside & track_interjet
    outer_pos = outside & track_outer_pos
    outer_neg = outside & track_outer_neg
    side = outside & track_side

    result = {}
    result["n_tracks_outside_jets"] = ak.to_numpy(ak.sum(outside, axis=1)).astype(np.float64)
    sum_pt_outside = ak.to_numpy(ak.sum(track_pt[outside], axis=1))
    result["sum_track_pt_outside_jets"] = sum_pt_outside
    result["max_track_pt_outside_jets"] = scalar_max(track_pt, outside)
    result["n_tracks_projected_bridge"] = ak.to_numpy(ak.sum(interjet, axis=1)).astype(np.float64)
    result["sum_track_pt_projected_bridge"] = ak.to_numpy(ak.sum(track_pt[interjet], axis=1))
    result["n_tracks_outer_positive"] = ak.to_numpy(ak.sum(outer_pos, axis=1)).astype(np.float64)
    result["n_tracks_outer_negative"] = ak.to_numpy(ak.sum(outer_neg, axis=1)).astype(np.float64)
    result["sum_track_pt_outer_positive"] = ak.to_numpy(ak.sum(track_pt[outer_pos], axis=1))
    result["sum_track_pt_outer_negative"] = ak.to_numpy(ak.sum(track_pt[outer_neg], axis=1))
    result["n_tracks_projected_side"] = ak.to_numpy(ak.sum(side, axis=1)).astype(np.float64)
    result["sum_track_pt_projected_side"] = ak.to_numpy(ak.sum(track_pt[side], axis=1))

    track_jets = cluster_track_jets(track_pt, track_eta, track_phi, outside)
    tj_pt = track_jets.pt
    tj_eta = track_jets.eta
    tj_phi = track_jets.phi
    tj_interjet, tj_outer_pos, tj_outer_neg, tj_side = activity_region_masks(
        tj_eta, tj_phi, leading, subleading
    )
    tj_outer = tj_outer_pos | tj_outer_neg
    n_tj = ak.to_numpy(ak.num(tj_pt)).astype(np.float64)
    tj_ht = ak.to_numpy(ak.sum(tj_pt, axis=1))
    leading_tj_pt = scalar_max(tj_pt, tj_pt > 0)
    result["n_extra_track_jets"] = n_tj
    result["leading_extra_track_jet_pt"] = leading_tj_pt
    result["extra_track_jet_ht"] = tj_ht
    result["n_interjet_track_jets"] = ak.to_numpy(ak.sum(tj_interjet, axis=1)).astype(np.float64)
    result["leading_interjet_track_jet_pt"] = scalar_max(tj_pt, tj_interjet)
    result["n_outer_track_jets"] = ak.to_numpy(ak.sum(tj_outer, axis=1)).astype(np.float64)
    result["leading_outer_track_jet_pt"] = scalar_max(tj_pt, tj_outer)
    result["n_side_track_jets"] = ak.to_numpy(ak.sum(tj_side, axis=1)).astype(np.float64)
    result["leading_side_track_jet_pt"] = scalar_max(tj_pt, tj_side)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["track_clumping_fraction"] = np.where(
            sum_pt_outside > 0, tj_ht / sum_pt_outside, np.nan
        )
        result["leading_track_jet_fraction"] = np.where(
            sum_pt_outside > 0, np.nan_to_num(leading_tj_pt) / sum_pt_outside, np.nan
        )

    gap_quality = outside & (track_pt >= GAP_TRACK_MIN_PT)
    pos_edge = eta_pos + JET_CONE_R
    neg_edge = eta_neg - JET_CONE_R
    nearest_pos = ak.to_numpy(
        ak.min(track_eta[gap_quality & (track_eta > pos_edge[:, np.newaxis])], axis=1, mask_identity=False)
    )
    nearest_pos = np.where(np.isfinite(nearest_pos), nearest_pos, TRACKER_ETA)
    nearest_neg = ak.to_numpy(
        ak.max(track_eta[gap_quality & (track_eta < neg_edge[:, np.newaxis])], axis=1, mask_identity=False)
    )
    nearest_neg = np.where(np.isfinite(nearest_neg), nearest_neg, -TRACKER_ETA)
    pos_gap = np.clip(np.minimum(nearest_pos, TRACKER_ETA) - pos_edge, 0.0, None)
    neg_gap = np.clip(neg_edge - np.maximum(nearest_neg, -TRACKER_ETA), 0.0, None)
    result["positive_gap_size"] = pos_gap
    result["negative_gap_size"] = neg_gap
    result["minimum_gap_size"] = np.minimum(pos_gap, neg_gap)
    result["sum_gap_size"] = pos_gap + neg_gap

    extra_jets = jets[:, 2:]
    extra_jets = extra_jets[extra_jets.pt >= EXTRA_JET_MIN_PT]
    extra_pt = extra_jets.pt
    extra_interjet, extra_outer_pos, extra_outer_neg, extra_side = activity_region_masks(
        extra_jets.eta, extra_jets.phi, leading, subleading
    )
    extra_outer = extra_outer_pos | extra_outer_neg
    result["n_extra_jets"] = ak.to_numpy(ak.num(extra_pt)).astype(np.float64)
    result["leading_extra_jet_pt"] = scalar_max(extra_pt, extra_pt > 0)
    result["extra_jet_ht"] = ak.to_numpy(ak.sum(extra_pt, axis=1))
    result["n_interjet_extra_jets"] = ak.to_numpy(
        ak.sum(extra_interjet, axis=1)
    ).astype(np.float64)
    result["leading_interjet_extra_jet_pt"] = scalar_max(extra_pt, extra_interjet)
    result["n_outer_extra_jets"] = ak.to_numpy(
        ak.sum(extra_outer, axis=1)
    ).astype(np.float64)
    result["leading_outer_extra_jet_pt"] = scalar_max(extra_pt, extra_outer)
    result["n_side_extra_jets"] = ak.to_numpy(
        ak.sum(extra_side, axis=1)
    ).astype(np.float64)
    result["leading_side_extra_jet_pt"] = scalar_max(extra_pt, extra_side)

    cand_pt = ak.concatenate([tj_pt, extra_jets.pt], axis=1)
    cand_eta = ak.concatenate([tj_eta, extra_jets.eta], axis=1)
    tj_region = ak.where(
        tj_interjet, 0.0,
        ak.where(tj_outer_pos, 1.0, ak.where(tj_outer_neg, 2.0, 3.0)),
    )
    extra_region = ak.where(
        extra_interjet, 0.0,
        ak.where(extra_outer_pos, 1.0, ak.where(extra_outer_neg, 2.0, 3.0)),
    )
    cand_region = ak.concatenate([tj_region, extra_region], axis=1)
    lead_idx = ak.argmax(cand_pt, axis=1, keepdims=True)
    lead_eta = ak.to_numpy(ak.fill_none(ak.firsts(cand_eta[lead_idx]), np.nan))
    result["leading_extra_activity_eta"] = lead_eta
    result["leading_extra_activity_region"] = ak.to_numpy(
        ak.fill_none(ak.firsts(cand_region[lead_idx]), np.nan)
    )

    pull1_mag, cos1 = jet_pull(track_pt, track_eta, track_phi, leading, subleading)
    pull2_mag, cos2 = jet_pull(track_pt, track_eta, track_phi, subleading, leading)
    result["jet1_pull_angle"] = np.arccos(cos1)
    result["jet2_pull_angle"] = np.arccos(cos2)
    result["jet1_pull_magnitude"] = pull1_mag
    result["jet2_pull_magnitude"] = pull2_mag
    result["combined_pull_alignment"] = 0.5 * (cos1 + cos2)
    result["interjet_bridge_asymmetry_projected"] = projected_bridge_asymmetry(
        track_pt, track_eta, track_phi, outside, leading, subleading
    )

    result["n_events"] = int(np.sum(keep))
    return result


def load_sample(spec, tree_name, collection):
    files = sorted(spec["input_dir"].glob("*.root"))[: spec["max_files"]]
    if not files:
        raise RuntimeError(f"No ROOT files in {spec['input_dir']}")
    pieces = []
    n_events = 0
    for path in files:
        piece = process_file(path, tree_name, collection)
        if piece is not None:
            n_events += piece.pop("n_events")
            pieces.append(piece)
    merged = {
        key: np.concatenate([piece[key] for piece in pieces])
        for key in list(pieces[0])
    }
    print(f"{spec['name']}: files={len(files)} selected_events={n_events}", flush=True)
    return merged


def separation(values_a, values_b, bins):
    hist_a, _ = np.histogram(values_a[np.isfinite(values_a)], bins=bins, density=True)
    hist_b, _ = np.histogram(values_b[np.isfinite(values_b)], bins=bins, density=True)
    width = np.diff(bins)
    return 1.0 - float(np.sum(np.minimum(hist_a, hist_b) * width))


def plot_group(samples, data, group, output_dir):
    figname, title, (nrows, ncols), obs_list = group
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.25 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for (name, xlabel, bins), ax in zip(obs_list, axes):
        for spec in samples:
            values = data[spec["name"]][name]
            values = values[np.isfinite(values)]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=spec["color"],
                linestyle=spec["linestyle"],
                label=spec["label"],
            )
        sep_sc = separation(data["Hbb"][name], data["QCDbb"][name], bins)
        sep_mg = separation(data["Hbb"][name], data["QCDbb_madgraph"][name], bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized events / bin")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"sep vs SC = {sep_sc:.2f}, vs MG = {sep_mg:.2f}", fontsize=10)
        if name == "leading_extra_activity_region":
            ax.set_xticks(range(len(REGION_LABELS)), REGION_LABELS)
    for ax in axes[len(obs_list):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=1.04)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_dir / f"{figname}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {spec["name"]: load_sample(spec, args.tree, args.collection) for spec in SAMPLES}

    summary_lines = [
        "separation = 1 - histogram overlap (normalized, finite values only), per event",
        "selection: two jets pT >= 15, |dphi_jj| > 3, 50 <= mjj <= 150 (no outer-track cut)",
        "",
    ]
    for _figname, title, _shape, obs_list in OBSERVABLE_GROUPS:
        summary_lines.append(f"[{title}]")
        summary_lines.append(f"{'observable':32s} {'Hbb vs SC QCDbb':>16s} {'Hbb vs MG comb':>16s}")
        for name, _xlabel, bins in obs_list:
            sep_sc = separation(data["Hbb"][name], data["QCDbb"][name], bins)
            sep_mg = separation(data["Hbb"][name], data["QCDbb_madgraph"][name], bins)
            summary_lines.append(f"{name:32s} {sep_sc:16.3f} {sep_mg:16.3f}")
        summary_lines.append("")

    for group in OBSERVABLE_GROUPS:
        plot_group(SAMPLES, data, group, output_dir)
    summary = "\n".join(summary_lines)
    (output_dir / "separations.txt").write_text(summary + "\n")
    print(summary)
    print(f"Wrote plots and separations to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
