#!/usr/bin/env python3
"""Central-only H->bb dijet MVA (no proton/PPS information).

Trains an XGBoost classifier for exclusive H->bb signal vs SuperChic QCDbb and
MadGraph inclusive QCDbb using only central-detector information, to measure
how much discrimination the central event alone provides. All three samples use
the FSR + 200PU Delphes campaigns so pileup conditions are identical, and every
track observable uses primary-vertex-associated tracks (EFlowTrack.IsRecoPU == 0).

Features: dijet/single-jet kinematics (incl. |cos theta*| and pT asymmetry),
per-jet substructure (NCharged, PTD, width, charged fraction, track pull),
color-flow/exclusivity observables (out-of-jet PV-track activity by region,
rapidity gaps, anti-kt R=0.2 track mini-jet fractions, bridge asymmetry,
eta spread), soft in-jet leptons, Puppi MET, and vertex multiplicity.

Pre-MVA selection: two jets pT >= 15 GeV, |delta_phi_jj| > 3,
50 <= mjj <= 150 GeV.

A small hyperparameter grid is scored on a stratified holdout, then the best
configuration is retrained with 5-fold cross-validation for out-of-fold scores.

Run under the analysis environment (needs the pip-installed fastjet package):
  source setup_env.sh
  python3 analysis/MVA/run_dijet_mva_no_protons.py
"""
import argparse
import os
import site
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import awkward as ak
import matplotlib
import numpy as np
import uproot
import vector
import yaml

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

from analysis.cross_sections import generator_cross_section_fb, generator_weight  # noqa: E402
from common.config_utils import load_yaml  # noqa: E402

LUMI_FB = 3000.0
MIN_JET_PT = 15.0
JET_CONE_R = 0.4
TRACK_MIN_PT = 0.5
TRACK_MAX_ABS_ETA = 2.5
TRACKER_ETA = 2.5
MIN_DIJET_DELTA_PHI = 3.0
PRE_MVA_DIJET_MASS_RANGE_GEV = (50.0, 150.0)
TRACK_JET_R = 0.2
TRACK_JET_MIN_PT = 3.0
BRIDGE_HALF_WIDTH = 0.4
SIDE_BAND_OUTER = 1.2
SOFT_LEPTON_MIN_PT = 2.0
CV_FOLDS = 5
ROOT_READ_WORKERS = 8
SEARCH_MAX_ROWS = 300000

SAMPLES = (
    {
        "name": "Hbb",
        "label": "H->bb (SuperChic)",
        "generator": "superchic",
        "process": "Hbb",
        "role": "signal",
        "input_dir": REPO / "output-superchic/Hbb/Hbb__v01/sim-Delphes/Hbb_FSR_200PU__v01/root",
        "max_files": 100,
    },
    {
        "name": "QCDbb",
        "label": "SuperChic QCDbb",
        "generator": "superchic",
        "process": "QCDbb",
        "role": "background",
        "input_dir": REPO / "output-superchic/QCDbb/QCDbb__v01/sim-Delphes/QCDbb_FSR_200PU__v01/root",
        "max_files": 100,
    },
    {
        "name": "QCDbb_madgraph",
        "label": "MadGraph QCDbb (inclusive)",
        "generator": "madgraph",
        "process": "QCDbb",
        "role": "background",
        "input_dir": REPO / "output-madgraph/QCDbb/QCDbb__v01/sim-Delphes/QCDbb_DPy8_FSR_200PU__v01/root",
        "max_files": 35,
    },
)

FEATURE_NAMES = (
    # dijet / single-jet kinematics
    "jet1_pt_over_mjj",
    "jet2_pt_over_mjj",
    "jet1_eta",
    "jet2_eta",
    "delta_eta_jj",
    "delta_phi_jj",
    "dijet_pt",
    "dijet_rapidity",
    "dijet_mass",
    "jet_multiplicity",
    "cos_theta_star",
    "pt_asymmetry",
    # per-jet substructure
    "jet1_ncharged",
    "jet2_ncharged",
    "jet1_ptd",
    "jet2_ptd",
    "jet1_width",
    "jet2_width",
    "jet1_charged_fraction",
    "jet2_charged_fraction",
    "jet1_pull_magnitude",
    "jet2_pull_magnitude",
    # color flow / exclusivity (PV tracks)
    "n_tracks_outside_jets",
    "sum_track_pt_outside_jets",
    "max_track_pt_outside_jets",
    "n_tracks_interjet",
    "sum_track_pt_interjet",
    "n_tracks_outer_positive",
    "n_tracks_outer_negative",
    "sum_track_pt_outer_positive",
    "sum_track_pt_outer_negative",
    "sum_gap_size",
    "minimum_gap_size",
    "track_clumping_fraction",
    "leading_track_jet_fraction",
    "interjet_bridge_asymmetry",
    "eta_rms_outside",
    # leptons / MET / vertices
    "n_soft_leptons_in_jets",
    "max_in_jet_lepton_pt",
    "puppi_met",
    "met_min_dphi_jet",
    "n_vertices",
)

PARAM_GRID = tuple(
    {"max_depth": depth, "learning_rate": rate, "min_child_weight": child}
    for depth in (4, 6, 8)
    for rate in (0.02, 0.05)
    for child in (1, 20)
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="JetPUPPI", help="Jet collection branch")
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Cap ROOT files per sample (overrides per-sample defaults).",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip the hyperparameter grid search and use the default configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "analysis/MVA/output/mva_bb_no_protons"),
        help="Output directory for dataset, model, plots, and summary.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    return parser.parse_args()


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def wrap_phi(np_like):
    return (np_like + np.pi) % (2.0 * np.pi) - np.pi


def scalar_max(values, mask, empty=0.0):
    """Per-event max of values[mask]; `empty` where no entry passes."""
    out = ak.to_numpy(ak.max(values[mask], axis=1, mask_identity=False))
    return np.where(np.isfinite(out), out, empty)


def jet_pull_magnitude(track_pt, track_eta, track_phi, jet):
    deta = track_eta - jet.eta[:, np.newaxis]
    dphi = wrap_phi(track_phi - jet.phi[:, np.newaxis])
    dr = np.hypot(deta, dphi)
    in_jet = (dr < JET_CONE_R) & (track_pt >= TRACK_MIN_PT)
    jet_pt = ak.to_numpy(jet.pt)
    pull_eta = ak.to_numpy(ak.sum((track_pt * deta * dr)[in_jet], axis=1)) / jet_pt
    pull_phi = ak.to_numpy(ak.sum((track_pt * dphi * dr)[in_jet], axis=1)) / jet_pt
    return np.hypot(pull_eta, pull_phi)


def cluster_track_jets(track_pt, track_eta, track_phi, mask):
    """Anti-kt R=0.2 track jets (pT > TRACK_JET_MIN_PT) from masked tracks."""
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
    return ak.zip({"px": raw.px, "py": raw.py, "pz": raw.pz, "E": raw.E}, with_name="Momentum4D")


def bridge_asymmetry(track_pt, track_eta, track_phi, interjet, jet1, jet2):
    eta1 = ak.to_numpy(jet1.eta)
    eta2 = ak.to_numpy(jet2.eta)
    phi1 = ak.to_numpy(jet1.phi)
    phi2_unwrapped = phi1 + ak.to_numpy(wrap_phi(jet2.phi - jet1.phi))
    deta_jj = eta2 - eta1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (track_eta - eta1[:, np.newaxis]) / deta_jj[:, np.newaxis]
    line_phi = phi1[:, np.newaxis] + frac * (phi2_unwrapped - phi1)[:, np.newaxis]
    dphi_line = wrap_phi(track_phi - line_phi)
    in_bridge = interjet & (np.abs(dphi_line) < BRIDGE_HALF_WIDTH)
    in_side = interjet & (np.abs(dphi_line) >= BRIDGE_HALF_WIDTH) & (
        np.abs(dphi_line) < SIDE_BAND_OUTER
    )
    pt_bridge = ak.to_numpy(ak.sum(track_pt[in_bridge], axis=1))
    pt_up = ak.to_numpy(ak.sum(track_pt[in_side & (dphi_line > 0)], axis=1))
    pt_down = ak.to_numpy(ak.sum(track_pt[in_side & (dphi_line < 0)], axis=1))
    mean_side = 0.5 * (pt_up + pt_down)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (pt_bridge - mean_side) / (pt_bridge + mean_side)


def process_file(path, tree_name, collection):
    jet_fields = (
        "PT", "Eta", "Phi", "Mass", "NCharged", "PTD", "MeanSqDeltaR",
        "ChargedEnergyFraction",
    )
    required = [branch_name(collection, field) for field in jet_fields]
    required.extend(
        branch_name("EFlowTrack", field) for field in ("PT", "Eta", "Phi", "IsRecoPU")
    )
    required.extend(branch_name("MuonLoose", field) for field in ("PT", "Eta", "Phi"))
    required.extend(branch_name("Electron", field) for field in ("PT", "Eta", "Phi"))
    required.extend(branch_name("PuppiMissingET", field) for field in ("MET", "Phi"))
    required.append(branch_name("Vertex", "Z"))
    try:
        with uproot.open(path) as root_file:
            tree = root_file[tree_name]
            arrays = tree.arrays(required, library="ak")
            n_generated = int(tree.num_entries)
    except Exception as exc:  # skip corrupt/empty ROOT files
        print(f"Warning: skipping unreadable file {path}: {exc}", flush=True)
        return {"n_generated": 0}

    def jet_field(field):
        return ak.values_astype(arrays[branch_name(collection, field)], "float64")

    pt = jet_field("PT")
    jet_mask = pt >= MIN_JET_PT
    jets = ak.zip(
        {
            "pt": pt[jet_mask],
            "eta": jet_field("Eta")[jet_mask],
            "phi": jet_field("Phi")[jet_mask],
            "mass": jet_field("Mass")[jet_mask],
            "ncharged": jet_field("NCharged")[jet_mask],
            "ptd": jet_field("PTD")[jet_mask],
            "mean_sq_dr": jet_field("MeanSqDeltaR")[jet_mask],
            "charged_fraction": jet_field("ChargedEnergyFraction")[jet_mask],
        },
        with_name="Momentum4D",
    )
    has_two = ak.to_numpy(ak.num(jets.pt) >= 2)
    if not np.any(has_two):
        return {"n_generated": n_generated}
    jets = jets[has_two]
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    leading, subleading = jets[:, 0], jets[:, 1]
    dijet = leading + subleading
    mjj = ak.to_numpy(dijet.mass)
    abs_dphi_jj = np.abs(ak.to_numpy(wrap_phi(leading.phi - subleading.phi)))
    keep = (
        (abs_dphi_jj > MIN_DIJET_DELTA_PHI)
        & (mjj >= PRE_MVA_DIJET_MASS_RANGE_GEV[0])
        & (mjj <= PRE_MVA_DIJET_MASS_RANGE_GEV[1])
    )
    if not np.any(keep):
        return {"n_generated": n_generated}
    keep_ak = ak.Array(keep)
    jets = jets[keep_ak]
    leading, subleading = jets[:, 0], jets[:, 1]
    dijet = leading + subleading
    mjj = mjj[keep]
    abs_dphi_jj = abs_dphi_jj[keep]

    def event_slice(coll, field):
        values = ak.values_astype(arrays[branch_name(coll, field)], "float64")
        return values[has_two][keep_ak]

    track_pt = event_slice("EFlowTrack", "PT")
    track_eta = event_slice("EFlowTrack", "Eta")
    track_phi = event_slice("EFlowTrack", "Phi")
    track_is_pu = event_slice("EFlowTrack", "IsRecoPU")
    pv_track = (
        (track_is_pu == 0)
        & (track_pt >= TRACK_MIN_PT)
        & (np.abs(track_eta) < TRACK_MAX_ABS_ETA)
    )
    track_pt, track_eta, track_phi = track_pt[pv_track], track_eta[pv_track], track_phi[pv_track]

    features = {}
    jet1_pt = ak.to_numpy(leading.pt)
    jet2_pt = ak.to_numpy(subleading.pt)
    y1 = ak.to_numpy(leading.rapidity)
    y2 = ak.to_numpy(subleading.rapidity)
    features["jet1_pt_over_mjj"] = jet1_pt / mjj
    features["jet2_pt_over_mjj"] = jet2_pt / mjj
    features["jet1_eta"] = ak.to_numpy(leading.eta)
    features["jet2_eta"] = ak.to_numpy(subleading.eta)
    features["delta_eta_jj"] = np.abs(ak.to_numpy(leading.eta - subleading.eta))
    features["delta_phi_jj"] = abs_dphi_jj
    features["dijet_pt"] = ak.to_numpy(dijet.pt)
    features["dijet_rapidity"] = ak.to_numpy(dijet.rapidity)
    features["dijet_mass"] = mjj
    features["jet_multiplicity"] = ak.to_numpy(ak.num(jets.pt)).astype(np.float64)
    features["cos_theta_star"] = np.abs(np.tanh(0.5 * (y1 - y2)))
    features["pt_asymmetry"] = (jet1_pt - jet2_pt) / (jet1_pt + jet2_pt)

    for tag, jet in (("jet1", leading), ("jet2", subleading)):
        features[f"{tag}_ncharged"] = ak.to_numpy(jet.ncharged)
        features[f"{tag}_ptd"] = ak.to_numpy(jet.ptd)
        features[f"{tag}_width"] = np.sqrt(np.clip(ak.to_numpy(jet.mean_sq_dr), 0.0, None))
        features[f"{tag}_charged_fraction"] = ak.to_numpy(jet.charged_fraction)
        features[f"{tag}_pull_magnitude"] = jet_pull_magnitude(
            track_pt, track_eta, track_phi, jet
        )

    dr1 = np.hypot(
        track_eta - leading.eta[:, np.newaxis],
        wrap_phi(track_phi - leading.phi[:, np.newaxis]),
    )
    dr2 = np.hypot(
        track_eta - subleading.eta[:, np.newaxis],
        wrap_phi(track_phi - subleading.phi[:, np.newaxis]),
    )
    outside = (dr1 > JET_CONE_R) & (dr2 > JET_CONE_R)
    eta_pos = np.maximum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    eta_neg = np.minimum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    interjet = outside & (track_eta > eta_neg[:, np.newaxis]) & (track_eta < eta_pos[:, np.newaxis])
    outer_pos = outside & (track_eta > eta_pos[:, np.newaxis])
    outer_neg = outside & (track_eta < eta_neg[:, np.newaxis])

    sum_pt_outside = ak.to_numpy(ak.sum(track_pt[outside], axis=1))
    features["n_tracks_outside_jets"] = ak.to_numpy(ak.sum(outside, axis=1)).astype(np.float64)
    features["sum_track_pt_outside_jets"] = sum_pt_outside
    features["max_track_pt_outside_jets"] = scalar_max(track_pt, outside)
    features["n_tracks_interjet"] = ak.to_numpy(ak.sum(interjet, axis=1)).astype(np.float64)
    features["sum_track_pt_interjet"] = ak.to_numpy(ak.sum(track_pt[interjet], axis=1))
    features["n_tracks_outer_positive"] = ak.to_numpy(ak.sum(outer_pos, axis=1)).astype(np.float64)
    features["n_tracks_outer_negative"] = ak.to_numpy(ak.sum(outer_neg, axis=1)).astype(np.float64)
    features["sum_track_pt_outer_positive"] = ak.to_numpy(ak.sum(track_pt[outer_pos], axis=1))
    features["sum_track_pt_outer_negative"] = ak.to_numpy(ak.sum(track_pt[outer_neg], axis=1))

    pos_edge = eta_pos + JET_CONE_R
    neg_edge = eta_neg - JET_CONE_R
    nearest_pos = ak.to_numpy(
        ak.min(track_eta[outside & (track_eta > pos_edge[:, np.newaxis])], axis=1, mask_identity=False)
    )
    nearest_pos = np.where(np.isfinite(nearest_pos), nearest_pos, TRACKER_ETA)
    nearest_neg = ak.to_numpy(
        ak.max(track_eta[outside & (track_eta < neg_edge[:, np.newaxis])], axis=1, mask_identity=False)
    )
    nearest_neg = np.where(np.isfinite(nearest_neg), nearest_neg, -TRACKER_ETA)
    pos_gap = np.clip(np.minimum(nearest_pos, TRACKER_ETA) - pos_edge, 0.0, None)
    neg_gap = np.clip(neg_edge - np.maximum(nearest_neg, -TRACKER_ETA), 0.0, None)
    features["sum_gap_size"] = pos_gap + neg_gap
    features["minimum_gap_size"] = np.minimum(pos_gap, neg_gap)

    track_jets = cluster_track_jets(track_pt, track_eta, track_phi, outside)
    tj_ht = ak.to_numpy(ak.sum(track_jets.pt, axis=1))
    leading_tj_pt = scalar_max(track_jets.pt, track_jets.pt > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        features["track_clumping_fraction"] = np.where(
            sum_pt_outside > 0, tj_ht / sum_pt_outside, np.nan
        )
        features["leading_track_jet_fraction"] = np.where(
            sum_pt_outside > 0, leading_tj_pt / sum_pt_outside, np.nan
        )
    features["interjet_bridge_asymmetry"] = bridge_asymmetry(
        track_pt, track_eta, track_phi, interjet, leading, subleading
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        eta_centroid = ak.to_numpy(ak.sum((track_pt * track_eta)[outside], axis=1)) / sum_pt_outside
        eta_var = (
            ak.to_numpy(
                ak.sum((track_pt * (track_eta - eta_centroid[:, np.newaxis]) ** 2)[outside], axis=1)
            )
            / sum_pt_outside
        )
    features["eta_rms_outside"] = np.sqrt(np.clip(eta_var, 0.0, None))

    lepton_pt = ak.concatenate(
        [event_slice("MuonLoose", "PT"), event_slice("Electron", "PT")], axis=1
    )
    lepton_eta = ak.concatenate(
        [event_slice("MuonLoose", "Eta"), event_slice("Electron", "Eta")], axis=1
    )
    lepton_phi = ak.concatenate(
        [event_slice("MuonLoose", "Phi"), event_slice("Electron", "Phi")], axis=1
    )
    lep_dr1 = np.hypot(
        lepton_eta - leading.eta[:, np.newaxis],
        wrap_phi(lepton_phi - leading.phi[:, np.newaxis]),
    )
    lep_dr2 = np.hypot(
        lepton_eta - subleading.eta[:, np.newaxis],
        wrap_phi(lepton_phi - subleading.phi[:, np.newaxis]),
    )
    in_jet_lepton = (
        (np.minimum(lep_dr1, lep_dr2) < JET_CONE_R) & (lepton_pt >= SOFT_LEPTON_MIN_PT)
    )
    features["n_soft_leptons_in_jets"] = ak.to_numpy(ak.sum(in_jet_lepton, axis=1)).astype(np.float64)
    features["max_in_jet_lepton_pt"] = scalar_max(lepton_pt, in_jet_lepton)

    met = ak.to_numpy(ak.firsts(event_slice("PuppiMissingET", "MET")))
    met_phi = ak.to_numpy(ak.firsts(event_slice("PuppiMissingET", "Phi")))
    features["puppi_met"] = met
    features["met_min_dphi_jet"] = np.minimum(
        np.abs(wrap_phi(met_phi - ak.to_numpy(leading.phi))),
        np.abs(wrap_phi(met_phi - ak.to_numpy(subleading.phi))),
    )
    features["n_vertices"] = ak.to_numpy(
        ak.num(arrays[branch_name("Vertex", "Z")][has_two][keep_ak])
    ).astype(np.float64)

    matrix = np.column_stack([features[name] for name in FEATURE_NAMES])
    return {"features": matrix, "n_generated": n_generated}


def process_file_worker(arguments):
    return process_file(*arguments)


def load_sample(spec, tree_name, collection, max_files_override):
    max_files = max_files_override or spec["max_files"]
    files = sorted(spec["input_dir"].glob("*.root"))[:max_files]
    if not files:
        raise RuntimeError(f"No ROOT files in {spec['input_dir']}")
    tasks = [(path, tree_name, collection) for path in files]
    with ProcessPoolExecutor(max_workers=min(ROOT_READ_WORKERS, len(tasks))) as executor:
        pieces = list(executor.map(process_file_worker, tasks))
    n_generated = sum(piece["n_generated"] for piece in pieces)
    matrices = [piece["features"] for piece in pieces if "features" in piece]
    features = (
        np.vstack(matrices) if matrices else np.empty((0, len(FEATURE_NAMES)))
    )
    xsec_fb, xsec_source = generator_cross_section_fb(spec["generator"], spec["process"])
    parameters = load_yaml(REPO / "parameters.yaml")
    tag = float((parameters.get("tagging") or {})["eff_b"]) ** 2
    event_weight = (
        xsec_fb * LUMI_FB * generator_weight(spec["generator"], spec["process"]) * tag / n_generated
    )
    print(
        f"{spec['name']}: files={len(files)} generated={n_generated} "
        f"selected={features.shape[0]} xsec_fb={xsec_fb:.6g} ({xsec_source}) "
        f"event_weight={event_weight:.6g}",
        flush=True,
    )
    return {
        "name": spec["name"],
        "label": 1 if spec["role"] == "signal" else 0,
        "features": features,
        "event_weight": event_weight,
        "n_files": len(files),
        "n_generated": n_generated,
        "xsec_fb": xsec_fb,
        "xsec_source": xsec_source,
    }


def build_dataset(samples):
    return {
        "x": np.vstack([sample["features"] for sample in samples]),
        "y": np.concatenate(
            [np.full(sample["features"].shape[0], sample["label"], dtype=np.int8) for sample in samples]
        ),
        "physical_weight": np.concatenate(
            [np.full(sample["features"].shape[0], sample["event_weight"]) for sample in samples]
        ),
        "process": np.concatenate(
            [np.full(sample["features"].shape[0], sample["name"]) for sample in samples]
        ),
        "feature_names": np.asarray(FEATURE_NAMES),
    }


def balanced_weights(labels, physical_weights):
    weights = np.asarray(physical_weights, dtype=np.float64).copy()
    totals = {}
    for label in (0, 1):
        total = float(np.sum(weights[labels == label]))
        if total <= 0.0:
            raise RuntimeError(f"Class {label} has non-positive total training weight")
        totals[label] = total
    target = 0.5 * (totals[0] + totals[1])
    for label in (0, 1):
        weights[labels == label] *= target / totals[label]
    return weights


def make_model(XGBClassifier, params, seed):
    return XGBClassifier(
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=8,
        tree_method="hist",
        early_stopping_rounds=30,
        **params,
    )


def fit_model(XGBClassifier, train_test_split, x, y, w, params, seed):
    fit_idx, stop_idx = train_test_split(
        np.arange(y.shape[0]), train_size=0.9, random_state=seed, stratify=y
    )
    model = make_model(XGBClassifier, params, seed)
    model.fit(
        x[fit_idx],
        y[fit_idx],
        sample_weight=balanced_weights(y[fit_idx], w[fit_idx]),
        eval_set=[(x[stop_idx], y[stop_idx])],
        sample_weight_eval_set=[balanced_weights(y[stop_idx], w[stop_idx])],
        verbose=False,
    )
    return model


def grid_search(XGBClassifier, train_test_split, roc_auc_score, dataset, seed, rng):
    x, y, w = dataset["x"], dataset["y"], dataset["physical_weight"]
    rows = np.arange(y.shape[0])
    if rows.size > SEARCH_MAX_ROWS:
        rows = rng.choice(rows, size=SEARCH_MAX_ROWS, replace=False)
    train_idx, test_idx = train_test_split(
        rows, train_size=0.75, random_state=seed, stratify=y[rows]
    )
    results = []
    for params in PARAM_GRID:
        model = fit_model(
            XGBClassifier, train_test_split, x[train_idx], y[train_idx], w[train_idx],
            params, seed,
        )
        scores = model.predict_proba(x[test_idx])[:, 1]
        auc_value = roc_auc_score(
            y[test_idx], scores, sample_weight=balanced_weights(y[test_idx], w[test_idx])
        )
        results.append({**params, "auc": float(auc_value), "best_iteration": int(model.best_iteration)})
        print(
            f"search: depth={params['max_depth']} lr={params['learning_rate']} "
            f"min_child_weight={params['min_child_weight']} "
            f"auc={auc_value:.5f} best_iter={model.best_iteration}",
            flush=True,
        )
    best = max(results, key=lambda item: item["auc"])
    print(f"best configuration: {best}", flush=True)
    return {k: best[k] for k in ("max_depth", "learning_rate", "min_child_weight")}, results


def train_oof(XGBClassifier, StratifiedKFold, train_test_split, dataset, params, seed):
    x, y, w = dataset["x"], dataset["y"], dataset["physical_weight"]
    oof_scores = np.full(y.shape[0], np.nan)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    best_iterations = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        model = fit_model(
            XGBClassifier, train_test_split, x[train_idx], y[train_idx], w[train_idx],
            params, seed,
        )
        oof_scores[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        best_iterations.append(int(model.best_iteration))
        print(f"fold_{fold}: train={train_idx.size} test={test_idx.size} best_iter={model.best_iteration}", flush=True)
    if np.any(np.isnan(oof_scores)):
        raise RuntimeError("Cross-validation left events without an out-of-fold score")
    final_model = fit_model(XGBClassifier, train_test_split, x, y, w, params, seed)
    return oof_scores, best_iterations, final_model


def pairwise_auc(roc_auc_score, dataset, scores, background):
    mask = (dataset["process"] == "Hbb") | (dataset["process"] == background)
    return float(
        roc_auc_score(
            dataset["y"][mask],
            scores[mask],
            sample_weight=balanced_weights(dataset["y"][mask], dataset["physical_weight"][mask]),
        )
    )


def plot_outputs(roc_curve, dataset, scores, model, output_dir):
    styles = {
        "Hbb": ("#0072B2", "-", "H->bb (SuperChic)"),
        "QCDbb": ("#E69F00", "--", "SuperChic QCDbb"),
        "QCDbb_madgraph": ("#D55E00", "-.", "MadGraph QCDbb (inclusive)"),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0.0, 1.0, 51)
    for name, (color, linestyle, label) in styles.items():
        mask = dataset["process"] == name
        ax.hist(
            scores[mask], bins=bins, density=True, histtype="step",
            color=color, linestyle=linestyle, linewidth=1.6, label=label,
        )
    ax.set_xlabel("Out-of-fold MVA score")
    ax.set_ylabel("Normalized events / bin")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oof_score.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for background in ("QCDbb", "QCDbb_madgraph"):
        color, linestyle, label = styles[background]
        mask = (dataset["process"] == "Hbb") | (dataset["process"] == background)
        fpr, tpr, _ = roc_curve(
            dataset["y"][mask],
            scores[mask],
            sample_weight=balanced_weights(dataset["y"][mask], dataset["physical_weight"][mask]),
        )
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, label=f"Hbb vs {label}")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc.png", dpi=160)
    plt.close(fig)

    importance = model.get_booster().get_score(importance_type="gain")
    names = list(FEATURE_NAMES)
    gains = np.array([importance.get(f"f{i}", 0.0) for i in range(len(names))])
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1.5))
    ax.barh(np.arange(len(names)), gains[order], color="#0072B2")
    ax.set_yticks(np.arange(len(names)), [names[i] for i in order], fontsize=8)
    ax.set_xlabel("XGBoost gain")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from xgboost import XGBClassifier

    samples = [
        load_sample(spec, args.tree, args.collection, args.max_files) for spec in SAMPLES
    ]
    dataset = build_dataset(samples)
    print(f"dataset: {dataset['x'].shape[0]} events, {dataset['x'].shape[1]} features", flush=True)

    rng = np.random.default_rng(args.seed)
    if args.skip_search:
        best_params = {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 20}
        search_results = []
    else:
        best_params, search_results = grid_search(
            XGBClassifier, train_test_split, roc_auc_score, dataset, args.seed, rng
        )

    oof_scores, best_iterations, model = train_oof(
        XGBClassifier, StratifiedKFold, train_test_split, dataset, best_params, args.seed
    )
    auc_all = float(
        roc_auc_score(
            dataset["y"], oof_scores,
            sample_weight=balanced_weights(dataset["y"], dataset["physical_weight"]),
        )
    )
    auc_sc = pairwise_auc(roc_auc_score, dataset, oof_scores, "QCDbb")
    auc_mg = pairwise_auc(roc_auc_score, dataset, oof_scores, "QCDbb_madgraph")
    print(f"OOF class-balanced AUC (both backgrounds): {auc_all:.5f}")
    print(f"OOF AUC Hbb vs SuperChic QCDbb:            {auc_sc:.5f}")
    print(f"OOF AUC Hbb vs MadGraph QCDbb:             {auc_mg:.5f}")

    np.savez_compressed(output_dir / "dataset.npz", **dataset)
    np.savez_compressed(output_dir / "scores.npz", oof=oof_scores)
    model._estimator_type = "classifier"
    model.save_model(output_dir / "model.json")
    plot_outputs(roc_curve, dataset, oof_scores, model, output_dir)

    summary = {
        "selection": {
            "min_jet_pt_gev": MIN_JET_PT,
            "min_delta_phi_jj": MIN_DIJET_DELTA_PHI,
            "dijet_mass_gev": list(PRE_MVA_DIJET_MASS_RANGE_GEV),
        },
        "pv_tracks": "EFlowTrack.IsRecoPU == 0, pT >= 0.5 GeV, |eta| < 2.5",
        "track_jets": {"algorithm": "anti-kt", "R": TRACK_JET_R, "min_pt_gev": TRACK_JET_MIN_PT},
        "features": list(FEATURE_NAMES),
        "samples": [
            {k: sample[k] for k in ("name", "label", "n_files", "n_generated", "xsec_fb", "xsec_source", "event_weight")}
            | {"n_selected": int(sample["features"].shape[0])}
            for sample in samples
        ],
        "hyperparameter_search": search_results,
        "best_params": best_params,
        "cv_best_iterations": best_iterations,
        "auc_oof_class_balanced": auc_all,
        "auc_oof_hbb_vs_superchic_qcdbb": auc_sc,
        "auc_oof_hbb_vs_madgraph_qcdbb": auc_mg,
        "seed": args.seed,
    }
    with open(output_dir / "summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"Wrote dataset, model, plots, and summary to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
