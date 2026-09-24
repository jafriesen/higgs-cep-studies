#!/usr/bin/env python3
"""Prepare the multiclass central+proton H->bb dijet MVA dataset.

Extends run_dijet_mva_multiclass.py with one proton/PPS observable on top of
the central features from run_dijet_mva_exclusive_central.py:
  yx_minus_dijet_rapidity ("deltaY").

mx is the eventual fit/significance variable (see scan_pp_mass_significance.py),
and mx = sqrt(xi_left * xi_right) * sqrt_s, yx = 0.5*ln(xi_right/xi_left). Those
two coordinates are independent: mx uses only the product xi_left*xi_right, yx
only the ratio xi_right/xi_left. So the safe proton observables are the ones
living in the ratio (rapidity) direction, and the dangerous ones live in the
product (mass) direction:
  - raw xi_left, xi_right, mx, yx: reparametrizations of the fit variable,
    excluded (an earlier version included them and xi_left alone carried more
    than half the total gain -- confirmed leakage, not signal).
  - jet-side/proton-side xi ratios: combined with the jet-based xi estimate
    they reconstruct xi_left/xi_right exactly, excluded.
  - deltaM = mx_minus_dijet_mass: EXCLUDED even though run_dijet_mva.py stores
    it, because dijet_mass is a feature and deltaM + dijet_mass = mx, so the
    pair reconstructs the fit variable (feature-set closure leak). deltaM lives
    in the product/mass direction.
  - deltaY = yx_minus_dijet_rapidity: KEPT. It lives in the ratio direction;
    deltaY + dijet_rapidity = yx, which is orthogonal to mx, so it cannot
    rebuild the fit variable. It is a jet-vs-proton matching variable (~0 for
    genuine exclusive pairs, broad for random min-bias pairs).
mx and yx are still computed internally to select events within the
signal-region mass window / rapidity-difference band (a physical requirement,
not a feature), and mx is retained as a NON-feature bookkeeping column so the
score-vs-mx sculpting can be measured (see below).

Sculpting diagnostic: dijet_mass is itself correlated with mx (both are the
central-system mass), so even after closing the exact-reconstruction leak the
score can still shape the background mx spectrum. The script therefore reports
the distance correlation dCorr(pairwise score, mx) on the SuperChic QCDbb
background (the smooth-mx background the peak fit sits on). dCorr = 0 means the
score is statistically independent of mx (a clean bump hunt); a large value
means a categorized / 2D (score x mx) fit is needed instead. This is a
measurement, not a constraint -- no decorrelation penalty is applied here.

Proton pairs, same convention as run_dijet_mva.py ("from before"):
  - Hbb, QCDbb (SuperChic): one real proton pair per event from the matching
    HepMC file (hadr-Pythia stage), smeared by the PPS xi resolution. Events
    without both protons in the PPS xi acceptance, or whose (mx, yx) falls
    outside the signal-region mass window / rapidity-difference band, are
    dropped (no proton candidate for that event).
  - QCDbb_madgraph: fake proton pairs sampled from the global weighted min-bias
    pair pool (analysis/minbias_analyzer_bootstrap.py, proton_pairs.parquet),
    which pools every PPS-accepted forward proton across all interactions and
    weights each cross-arm window pair by its per-BX probability. Each central
    event is sampled directly from the weighted pairs satisfying the rapidity
    band. The global-pool probability of entering that band is folded into the
    physical event weight. Four candidates per hard event are cached so any
    later held-out fold has four independent candidates; trainers use one
    candidate per training group. The pool's Poisson-sampled
    bx_pair_acceptance normalizes the yield. Reads
    output/minbias/<campaign>/pairs/proton_pairs.parquet with <campaign> from
    config.yaml minbias.default_campaign unless --minbias-campaign overrides it.
    If the pool is absent, MadGraph is still cached as a class but with its
    proton-block feature left as NaN (XGBoost's hist method treats NaN as a
    learned missing-value split direction) rather than fabricating substitute
    pairs; that leaks "has protons or not" into the vs-MG comparison, so treat
    the vs-MG number as valid only when the pool is present.

The configured MadGraph campaigns are stitched in their generator-level
hard-b phase space. Their exclusive regions retain the usual sigma/N weight;
in their overlap, both are retained with the combined effective-MC-luminosity
weight. The fit gives equal total weight to each class while preserving these
physical relative weights within the MadGraph class.

Use prepare_dijet_mva_multiclass_protons.py for cache-only preparation and
optimize_dijet_mva_multiclass_protons.py for significance-driven training.

Run under the analysis environment (needs the pip-installed fastjet package):
  source setup_env.sh
  python3 analysis/MVA/run_dijet_mva_multiclass_protons.py
"""
import argparse
import json
import os
import re
import resource
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import awkward as ak
import matplotlib
import numpy as np
import pyarrow.parquet as pq
import uproot
import vector
import yaml
from uproot.source.file import MemmapSource

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

from analysis.cross_sections import (  # noqa: E402
    event_record_files,
    generator_cross_section_fb,
    generator_weight,
    parse_lhe_xsec_pb,
)
from analysis.MVA.color_flow_geometry import (  # noqa: E402
    activity_region_masks,
    projected_bridge_asymmetry,
)
from common.config_utils import load_yaml, natural_key, resolve_minbias_campaign, resolve_path  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_root,
    generation_config,
    generation_process_config,
    generation_stage_root,
)

# generic PPS/min-bias helpers reused from run_dijet_mva.py (pure functions,
# no module-level side effects beyond lightweight imports)
from analysis.MVA import run_dijet_mva as pps_lib  # noqa: E402
from analysis.MVA.run_dijet_mva_exclusive_central import (  # noqa: E402
    FEATURE_NAMES as CENTRAL_FEATURE_NAMES,
)

MIN_JET_PT = 15.0
JET_CONE_R = 0.4
TRACK_MIN_PT = 0.5
HARD_TRACK_MIN_PT = 2.0
TRACK_MAX_ABS_ETA = 2.5
TRACKER_ETA = 2.5
MIN_DIJET_DELTA_PHI = 3.0
PRE_MVA_DIJET_MASS_RANGE_GEV = (50.0, 150.0)
TRACK_JET_R = 0.2
TRACK_JET_MIN_PT = 3.0
BRIDGE_HALF_WIDTH = 0.4
SIDE_BAND_OUTER = 1.2
SOFT_LEPTON_MIN_PT = 2.0
FSR_RECOVERY_MAX_R = 1.0

# proton-selection and normalization constants, matching run_dijet_mva.py ("from before")
MASS_WINDOW_GEV = pps_lib.MASS_WINDOW_GEV
MAX_ABS_RAPIDITY_DIFFERENCE = pps_lib.MAX_ABS_RAPIDITY_DIFFERENCE
LUMI_FB = pps_lib.LUMI_FB
COMBINATORIAL_ACCEPTANCE_FACTOR = float(
    load_yaml(REPO / "parameters.yaml")["normalization"][
        "combinatorial_acceptance_factor"
    ]
)
SPLIT_NAMES = ("train", "calibration", "optimization", "test")
SPLIT_FRACTIONS = (0.60, 0.10, 0.15, 0.15)

def configured_madgraph_campaigns():
    process = generation_process_config("madgraph", "QCDbb")
    result = {}
    for campaign, campaign_cfg in (process.get("campaigns") or {}).items():
        phase_space = campaign_cfg.get("phase_space") or {}
        subcampaign = campaign_cfg.get("mva_subcampaign")
        if not subcampaign:
            continue
        required = ("parton_pt_gev", "max_abs_eta", "dijet_mass_gev")
        missing = [name for name in required if name not in phase_space]
        if missing:
            raise RuntimeError(
                f"MadGraph campaign {campaign} is missing phase-space fields: "
                f"{', '.join(missing)}"
            )
        pt = phase_space["parton_pt_gev"]
        mass = phase_space["dijet_mass_gev"]
        if len(pt) != 2 or len(mass) != 2:
            raise RuntimeError(
                f"MadGraph campaign {campaign} pT/mass bounds must have length two"
            )
        result[campaign] = {
            "subcampaign": subcampaign,
            "pt": tuple(pt),
            "max_abs_eta": float(phase_space["max_abs_eta"]),
            "mass": tuple(mass),
        }
    if not result:
        raise RuntimeError("No MadGraph QCDbb campaigns are configured for the MVA")
    return result


MADGRAPH_CAMPAIGNS = configured_madgraph_campaigns()


def parse_run_card_settings(path):
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("!", 1)[0]
        if "=" not in content:
            continue
        value, name = content.split("=", 1)
        if value.strip() and name.strip():
            settings[name.strip()] = value.strip().strip("'\"")
    return settings


def validate_madgraph_campaign_cards(campaigns):
    manifest = {}
    for campaign in campaigns:
        region = MADGRAPH_CAMPAIGNS[campaign]
        run_card = (
            generation_campaign_root("madgraph", "QCDbb", campaign)
            / generation_config("madgraph")["generation_dir"]
            / "init"
            / "run_card.dat"
        )
        if not run_card.is_file():
            raise RuntimeError(f"Missing initialized MadGraph run card: {run_card}")
        settings = parse_run_card_settings(run_card)
        names = ("ptb", "ptbmax", "etab", "mmbb", "mmbbmax")
        missing = [name for name in names if name not in settings]
        if missing:
            raise RuntimeError(
                f"MadGraph run card {run_card} is missing: {', '.join(missing)}"
            )
        actual = {
            "pt": (float(settings["ptb"]), float(settings["ptbmax"])),
            "max_abs_eta": float(settings["etab"]),
            "mass": (float(settings["mmbb"]), float(settings["mmbbmax"])),
        }
        expected = {
            "pt": tuple(region["pt"]),
            "max_abs_eta": float(region["max_abs_eta"]),
            "mass": tuple(region["mass"]),
        }
        if actual != expected:
            raise RuntimeError(
                f"Configured phase space for {campaign} does not match {run_card}: "
                f"configured={expected}, card={actual}"
            )
        manifest[campaign] = {
            "run_card": str(run_card),
            "phase_space": expected,
        }
    return manifest


def physical_event_weight(spec, n_generated, n_draws, bx_pair_acceptance, tag, campaign=None):
    """Per-generated-event physical weight, identical to run_dijet_mva.py:
    xsec * LUMI * process_weight * acceptance * bx_pair_acceptance * tag
    / (n_generated * n_draws). The MadGraph combinatorial acceptance factor and
    bx_pair_acceptance make its inclusive cross section into the fake-double-tag
    yield; SuperChic samples use acceptance = bx_pair_acceptance = 1."""
    if n_generated <= 0 or not np.isfinite(bx_pair_acceptance):
        return 0.0
    xsec_fb, _source = generator_cross_section_fb(
        spec["generator"], spec["process"], campaign
    )
    process_weight = generator_weight(spec["generator"], spec["process"])
    acceptance = COMBINATORIAL_ACCEPTANCE_FACTOR if spec["generator"] == "madgraph" else 1.0
    return (
        xsec_fb * LUMI_FB * process_weight * acceptance * bx_pair_acceptance * tag
        / (n_generated * n_draws)
    )

CV_FOLDS = 5
ROOT_READ_WORKERS = 8
SEARCH_MAX_ROWS = 300000
PPS_CONFIG_DEFAULT = "analysis/scripts/new/config.yaml"

CLASS_ORDER = ("Hbb", "QCDbb", "QCDbb_madgraph")
CLASS_LABELS = {
    "Hbb": "H->bb (SuperChic)",
    "QCDbb": "SuperChic QCDbb",
    "QCDbb_madgraph": "MadGraph QCDbb (inclusive)",
}

# Input ROOT/HepMC paths are resolved from the process default campaigns in
# processes-<generator>.yaml via run_dijet_mva.resolve_sample_inputs (the shared
# repo convention), not hardcoded. All three default sim-delphes campaigns are
# the FSR + 200PU ones, so pileup conditions are identical across processes.
SAMPLES = (
    {
        "name": "Hbb",
        "generator": "superchic",
        "process": "Hbb",
        "role": "signal",
        "max_files": 100,
    },
    {
        "name": "QCDbb",
        "generator": "superchic",
        "process": "QCDbb",
        "role": "background",
        "max_files": 100,
    },
    {
        "name": "QCDbb_madgraph",
        "generator": "madgraph",
        "process": "QCDbb",
        "role": "background",
        "max_files": None,
    },
)

PROTON_FEATURE_NAMES = (
    "yx_minus_dijet_rapidity",
)
FEATURE_NAMES = tuple(CENTRAL_FEATURE_NAMES) + PROTON_FEATURE_NAMES
DCORR_MAX_ROWS = 4000

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
        "--campaign", default=None,
        help="Hard-process campaign for the SuperChic samples. Defaults to each "
        "process default campaign in processes-superchic.yaml.",
    )
    parser.add_argument(
        "--madgraph-campaigns", nargs="+", choices=tuple(MADGRAPH_CAMPAIGNS),
        default=tuple(MADGRAPH_CAMPAIGNS),
        help="MadGraph campaigns to stitch. Defaults to every campaign carrying "
        "an mva_subcampaign in processes-madgraph.yaml.",
    )
    parser.add_argument(
        "--pps-config", default=PPS_CONFIG_DEFAULT,
        help="YAML file defining beam energy, PPS xi acceptance, and xi resolution.",
    )
    parser.add_argument(
        "--minbias-campaign", default=None,
        help="Min-bias campaign providing pairs/proton_pairs.parquet. "
        "Defaults to config.yaml minbias.default_campaign.",
    )
    parser.add_argument(
        "--train-pairs", type=int, default=1,
        help="Conditional proton-pair copies per MadGraph training group.",
    )
    parser.add_argument(
        "--evaluation-pairs", type=int, default=4,
        help="Conditional proton-pair copies per held-out MadGraph group.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO / "analysis/MVA/output/mva_bb_multiclass_protons/cache"),
        help="Directory for reusable memory-mappable dataset arrays.",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Create the reusable cache and stop before legacy training.",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip the hyperparameter grid search and use the default configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "analysis/MVA/output/mva_bb_multiclass_protons"),
        help="Output directory for dataset, model, plots, and summary.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    return parser.parse_args()


def deterministic_group_split(event_ids, salt, seed):
    """Assign hard-event IDs to fixed 60/10/15/15 splits."""
    values = np.asarray(event_ids, dtype=np.uint64)
    mixed = values ^ np.uint64(seed) ^ np.uint64(salt)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    bucket = mixed % np.uint64(10000)
    split = np.full(values.shape, 3, dtype=np.uint8)
    split[bucket < 8500] = 2
    split[bucket < 7000] = 1
    split[bucket < 6000] = 0
    return split


def conditional_pair_sample(
    central_matrix,
    dijet_rapidity,
    central_group,
    central_campaign,
    central_split,
    stitch_weight_fb,
    coverage_mask,
    pool,
    train_pairs,
    evaluation_pairs,
    seed,
):
    """Sample directly from the weighted proton pool inside the delta-y band.

    The returned stitched weight includes the probability for a global-pool
    draw to land in the band and is divided among copies of the same hard
    event. This is the conditional equivalent of rejection sampling.
    """
    if train_pairs <= 0 or evaluation_pairs <= 0:
        raise RuntimeError("Conditional proton copy counts must be positive")
    probability = np.array(pool["weight"], dtype=np.float64, copy=True)
    probability /= np.sum(probability)
    order = np.argsort(pool["yx"], kind="stable")
    sorted_yx = np.asarray(pool["yx"], dtype=np.float64)[order]
    sorted_mx = np.asarray(pool["mx"], dtype=np.float64)[order]
    sorted_probability = probability[order]
    cdf = np.cumsum(sorted_probability)
    cdf[-1] = 1.0

    low_index = np.searchsorted(
        sorted_yx,
        dijet_rapidity - MAX_ABS_RAPIDITY_DIFFERENCE,
        side="right",
    )
    high_index = np.searchsorted(
        sorted_yx,
        dijet_rapidity + MAX_ABS_RAPIDITY_DIFFERENCE,
        side="left",
    )
    cdf_low = np.zeros(dijet_rapidity.shape[0], dtype=np.float64)
    has_lower = low_index > 0
    cdf_low[has_lower] = cdf[low_index[has_lower] - 1]
    cdf_high = np.zeros(dijet_rapidity.shape[0], dtype=np.float64)
    has_band = high_index > 0
    cdf_high[has_band] = cdf[high_index[has_band] - 1]
    band_probability = np.maximum(cdf_high - cdf_low, 0.0)

    # Cache enough candidates for any later group-safe split. Trainers select
    # `train_pairs` copies per training group and retain all candidates for
    # held-out groups; this is required because final cross-fit folds differ
    # from the fixed development split.
    copies = np.full(central_split.shape, evaluation_pairs, dtype=np.int64)
    valid = band_probability > 0.0
    central_index = np.repeat(np.flatnonzero(valid), copies[valid])
    if central_index.size == 0:
        raise RuntimeError("No MadGraph central event has a proton pair in the rapidity band")
    rng = np.random.default_rng(seed)
    uniform = rng.random(central_index.size)
    quantile = (
        cdf_low[central_index]
        + uniform * band_probability[central_index]
    )
    pair_index = np.searchsorted(cdf, quantile, side="right")
    pair_index = np.minimum(pair_index, cdf.size - 1)
    pair_yx = sorted_yx[pair_index]
    proton_block = delta_block(dijet_rapidity[central_index], pair_yx)

    return {
        "features": np.asarray(
            np.column_stack([central_matrix[central_index], proton_block]),
            dtype=np.float32,
        ),
        "mx": sorted_mx[pair_index],
        "group": central_group[central_index],
        "source_campaign": central_campaign[central_index],
        "split": central_split[central_index],
        "band_probability": band_probability[central_index],
        "central_stitch_weight_fb": stitch_weight_fb[central_index],
        "coverage_mask": coverage_mask[central_index],
        "stitch_weight_fb": (
            stitch_weight_fb[central_index]
            * band_probability[central_index]
            / copies[central_index]
        ),
    }


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def wrap_phi(np_like):
    return (np_like + np.pi) % (2.0 * np.pi) - np.pi


def scalar_max(values, mask, empty=0.0):
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


def central_features(path, tree_name, collection, include_hard_bb=False):
    """Central feature block for one file, plus per-row bookkeeping needed
    for proton matching: original in-file event index, jet1/jet2 pt,
    rapidity, dijet mass/rapidity."""
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
    if include_hard_bb:
        required.extend(
            branch_name("Particle", field)
            for field in ("PID", "Status", "IsPU", "PT", "Eta", "Phi", "Mass")
        )
    try:
        with uproot.open(path, handler=MemmapSource) as root_file:
            tree = root_file[tree_name]
            arrays = tree.arrays(required, library="ak")
            n_generated = int(tree.num_entries)
    except Exception as exc:
        raise RuntimeError(f"Failed to read required data from {path}: {exc}") from exc

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
    event_index_has_two = np.nonzero(has_two)[0]
    if not np.any(has_two):
        return {"n_generated": n_generated, "empty": True}
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
        return {"n_generated": n_generated, "empty": True}
    keep_ak = ak.Array(keep)
    jets = jets[keep_ak]
    leading, subleading = jets[:, 0], jets[:, 1]
    dijet = leading + subleading
    mjj = mjj[keep]
    abs_dphi_jj = abs_dphi_jj[keep]
    event_indices = event_index_has_two[keep]

    hard_bb = None
    if include_hard_bb:
        particle = {
            field.lower(): arrays[branch_name("Particle", field)]
            for field in ("PID", "Status", "IsPU", "PT", "Eta", "Phi", "Mass")
        }
        hard_mask = (
            (abs(particle["pid"]) == 5)
            & (particle["status"] == 23)
            & (particle["ispu"] == 0)
        )
        hard_bb = ak.zip(
            {
                "pt": particle["pt"][hard_mask],
                "eta": particle["eta"][hard_mask],
                "phi": particle["phi"][hard_mask],
                "mass": particle["mass"][hard_mask],
            },
            with_name="Momentum4D",
        )[has_two][keep_ak]

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
    outside_hard = outside & (track_pt >= HARD_TRACK_MIN_PT)
    eta_pos = np.maximum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    eta_neg = np.minimum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    interjet = outside & (track_eta > eta_neg[:, np.newaxis]) & (track_eta < eta_pos[:, np.newaxis])
    outer_pos = outside & (track_eta > eta_pos[:, np.newaxis])
    outer_neg = outside & (track_eta < eta_neg[:, np.newaxis])
    projected_bridge, _projected_outer_pos, _projected_outer_neg, projected_side = (
        activity_region_masks(track_eta, track_phi, leading, subleading)
    )
    projected_bridge = outside & projected_bridge
    projected_side = outside & projected_side

    sum_pt_outside = ak.to_numpy(ak.sum(track_pt[outside], axis=1))
    features["n_tracks_outside_jets"] = ak.to_numpy(ak.sum(outside, axis=1)).astype(np.float64)
    features["sum_track_pt_outside_jets"] = sum_pt_outside
    features["n_tracks_outside_pt2"] = ak.to_numpy(ak.sum(outside_hard, axis=1)).astype(np.float64)
    features["sum_track_pt_outside_pt2"] = ak.to_numpy(ak.sum(track_pt[outside_hard], axis=1))
    features["max_track_pt_outside_jets"] = scalar_max(track_pt, outside)
    features["n_tracks_interjet"] = ak.to_numpy(ak.sum(interjet, axis=1)).astype(np.float64)
    features["sum_track_pt_interjet"] = ak.to_numpy(ak.sum(track_pt[interjet], axis=1))
    features["n_tracks_projected_bridge"] = ak.to_numpy(
        ak.sum(projected_bridge, axis=1)
    ).astype(np.float64)
    features["sum_track_pt_projected_bridge"] = ak.to_numpy(
        ak.sum(track_pt[projected_bridge], axis=1)
    )
    features["n_tracks_projected_side"] = ak.to_numpy(
        ak.sum(projected_side, axis=1)
    ).astype(np.float64)
    features["sum_track_pt_projected_side"] = ak.to_numpy(
        ak.sum(track_pt[projected_side], axis=1)
    )
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
    _tj_bridge, tj_outer_pos, tj_outer_neg, _tj_side = activity_region_masks(
        track_jets.eta, track_jets.phi, leading, subleading
    )
    tj_outer = tj_outer_pos | tj_outer_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        features["track_clumping_fraction"] = np.where(
            sum_pt_outside > 0, tj_ht / sum_pt_outside, np.nan
        )
        features["leading_track_jet_fraction"] = np.where(
            sum_pt_outside > 0, leading_tj_pt / sum_pt_outside, np.nan
        )
    features["n_extra_track_jets"] = ak.to_numpy(ak.num(track_jets.pt)).astype(np.float64)
    features["n_outer_track_jets"] = ak.to_numpy(ak.sum(tj_outer, axis=1)).astype(np.float64)
    features["leading_outer_track_jet_pt"] = scalar_max(track_jets.pt, tj_outer)
    features["interjet_bridge_asymmetry"] = bridge_asymmetry(
        track_pt, track_eta, track_phi, interjet, leading, subleading
    )
    features["interjet_bridge_asymmetry_projected"] = projected_bridge_asymmetry(
        track_pt, track_eta, track_phi, outside, leading, subleading
    )

    tj_dr1 = np.hypot(
        track_jets.eta - leading.eta[:, np.newaxis],
        wrap_phi(track_jets.phi - leading.phi[:, np.newaxis]),
    )
    tj_dr2 = np.hypot(
        track_jets.eta - subleading.eta[:, np.newaxis],
        wrap_phi(track_jets.phi - subleading.phi[:, np.newaxis]),
    )
    recovered = track_jets[np.minimum(tj_dr1, tj_dr2) < FSR_RECOVERY_MAX_R]
    total_px = ak.to_numpy(dijet.px) + ak.to_numpy(ak.sum(recovered.px, axis=1))
    total_py = ak.to_numpy(dijet.py) + ak.to_numpy(ak.sum(recovered.py, axis=1))
    total_pz = ak.to_numpy(dijet.pz) + ak.to_numpy(ak.sum(recovered.pz, axis=1))
    total_e = ak.to_numpy(dijet.E) + ak.to_numpy(ak.sum(recovered.E, axis=1))
    mass_sq = total_e**2 - total_px**2 - total_py**2 - total_pz**2
    features["dijet_mass_fsr"] = np.sqrt(np.clip(mass_sq, 0.0, None))
    features["fsr_recovered_pt"] = ak.to_numpy(ak.sum(recovered.pt, axis=1))

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

    matrix = np.asarray(
        np.column_stack([features[name] for name in CENTRAL_FEATURE_NAMES]),
        dtype=np.float32,
    )
    result = {
        "matrix": matrix,
        "event_indices": event_indices,
        "dijet_mass": mjj,
        "dijet_rapidity": features["dijet_rapidity"],
        "n_generated": n_generated,
    }
    if hard_bb is not None:
        valid = ak.to_numpy(ak.num(hard_bb.pt) == 2)
        padded = ak.pad_none(hard_bb, 2, clip=True)
        first, second = padded[:, 0], padded[:, 1]
        hard_dijet = first + second
        result.update(
            {
                "hard_bb_valid": valid,
                "hard_b1_pt": ak.to_numpy(ak.fill_none(first.pt, np.nan)),
                "hard_b2_pt": ak.to_numpy(ak.fill_none(second.pt, np.nan)),
                "hard_b1_eta": ak.to_numpy(ak.fill_none(first.eta, np.nan)),
                "hard_b2_eta": ak.to_numpy(ak.fill_none(second.eta, np.nan)),
                "hard_bb_mass": ak.to_numpy(ak.fill_none(hard_dijet.mass, np.nan)),
            }
        )
    return result


def delta_block(dijet_rapidity, yx):
    """The proton-block columns, in PROTON_FEATURE_NAMES order.

    Only deltaY is stored -- never raw mx/yx/xi_left/xi_right nor deltaM, all of
    which (given dijet_mass is a feature) reconstruct the fit variable mx. See
    the module docstring for the product/ratio-direction argument.
    """
    return np.column_stack([yx - dijet_rapidity])


def real_proton_pass(np_mod, xi_left_truth, xi_right_truth, pps, rng):
    """PPS acceptance + smearing + signal-region band, matching run_dijet_mva.py."""
    valid_pair = (
        np_mod.isfinite(xi_left_truth)
        & np_mod.isfinite(xi_right_truth)
        & (xi_left_truth > 0.0)
        & (xi_right_truth > 0.0)
    )
    pps_mask = valid_pair & pps_lib.passes_pps(np_mod, xi_left_truth, pps["xi_ranges"])
    pps_mask &= pps_lib.passes_pps(np_mod, xi_right_truth, pps["xi_ranges"])
    xi_left, xi_right, mx, yx, reco_valid = pps_lib.smear_pair_observables(
        np_mod, xi_left_truth, xi_right_truth, pps, rng
    )
    return pps_mask & reco_valid, xi_left, xi_right, mx, yx


def process_file_superchic(arguments):
    path, tree_name, collection, hepmc_path, pps, seed = arguments
    piece = central_features(path, tree_name, collection)
    if piece is None or piece.get("empty"):
        return {
            "features": np.empty((0, len(FEATURE_NAMES))),
            "mx": np.empty(0),
            "event_indices": np.empty(0, dtype=np.int64),
            "n_generated": 0 if piece is None else piece["n_generated"],
            "n_central": 0,
        }
    xi_left_truth, xi_right_truth = pps_lib.parse_hepmc_proton_xi(
        np, hepmc_path, piece["event_indices"], pps["sqrt_s"]
    )
    rng = np.random.default_rng(seed)
    passed, xi_left, xi_right, mx, yx = real_proton_pass(np, xi_left_truth, xi_right_truth, pps, rng)
    band = (
        (np.abs((yx - piece["dijet_rapidity"])) < MAX_ABS_RAPIDITY_DIFFERENCE)
        & (mx >= MASS_WINDOW_GEV[0])
        & (mx <= MASS_WINDOW_GEV[1])
    )
    keep = passed & band
    proton_block = delta_block(piece["dijet_rapidity"], yx)
    matrix = np.asarray(
        np.column_stack([piece["matrix"], proton_block])[keep], dtype=np.float32
    )
    return {
        "features": matrix,
        "mx": mx[keep],
        "event_indices": piece["event_indices"][keep],
        "n_generated": piece["n_generated"],
        "n_central": piece["matrix"].shape[0],
    }


def madgraph_phase_space_masks(hard_bb):
    masks = {}
    valid = hard_bb["hard_bb_valid"]
    pt1, pt2 = hard_bb["hard_b1_pt"], hard_bb["hard_b2_pt"]
    eta1, eta2 = hard_bb["hard_b1_eta"], hard_bb["hard_b2_eta"]
    mass = hard_bb["hard_bb_mass"]
    for campaign, region in MADGRAPH_CAMPAIGNS.items():
        pt_min, pt_max = region["pt"]
        mass_min, mass_max = region["mass"]
        masks[campaign] = (
            valid
            & (pt1 >= pt_min)
            & ((pt1 <= pt_max) if pt_max is not None else True)
            & (pt2 >= pt_min)
            & ((pt2 <= pt_max) if pt_max is not None else True)
            & (np.abs(eta1) <= region["max_abs_eta"])
            & (np.abs(eta2) <= region["max_abs_eta"])
            & (mass >= mass_min)
            & ((mass <= mass_max) if mass_max is not None else True)
        )
    return masks


def stitched_cross_section_weights(campaign, phase_masks, mc_luminosities):
    """Cross section represented by each central event, in fb.

    At each phase-space point, samples that cover the point are pooled using
    their summed effective MC luminosity. Events outside their own campaign's
    generator region are rejected rather than assigned an extrapolation weight.
    """
    coverage_luminosity = np.zeros_like(
        next(iter(phase_masks.values())), dtype=np.float64
    )
    for name, luminosity in mc_luminosities.items():
        coverage_luminosity += luminosity * phase_masks[name]
    own_region = phase_masks[campaign]
    weights = np.zeros(own_region.shape[0], dtype=np.float64)
    weights[own_region] = 1.0 / coverage_luminosity[own_region]
    return weights


def weighted_histogram_summary(values, weights, edges):
    weighted, _ = np.histogram(values, bins=edges, weights=weights)
    rows, _ = np.histogram(values, bins=edges)
    return {
        "edges": np.asarray(edges, dtype=float).tolist(),
        "cross_section_fb": weighted.tolist(),
        "rows": rows.tolist(),
    }


def process_file_madgraph(arguments):
    path, tree_name, collection = arguments
    piece = central_features(path, tree_name, collection, include_hard_bb=True)
    if piece is None or piece.get("empty"):
        return {
            "matrix": np.empty((0, len(CENTRAL_FEATURE_NAMES))),
            "event_indices": np.empty(0, dtype=np.int64),
            "dijet_mass": np.empty(0), "dijet_rapidity": np.empty(0),
            "hard_bb_valid": np.empty(0, dtype=bool),
            "hard_b1_pt": np.empty(0), "hard_b2_pt": np.empty(0),
            "hard_b1_eta": np.empty(0), "hard_b2_eta": np.empty(0),
            "hard_bb_mass": np.empty(0),
            "n_generated": 0 if piece is None else piece["n_generated"],
        }
    return piece


def load_superchic_sample(
    spec, input_files, proton_files, tree_name, collection, max_files_override,
    pps, base_seed, split_seed,
):
    max_files = max_files_override or spec["max_files"]
    files = input_files[:max_files]
    hepmc = proton_files[:max_files]
    if not files:
        raise RuntimeError(f"No ROOT files resolved for {spec['name']}")
    tasks = [
        (path, tree_name, collection, hepmc_path, pps, base_seed + index)
        for index, (path, hepmc_path) in enumerate(zip(files, hepmc))
    ]
    with ProcessPoolExecutor(max_workers=min(ROOT_READ_WORKERS, len(tasks))) as executor:
        pieces = list(executor.map(process_file_superchic, tasks))
    n_generated = sum(p["n_generated"] for p in pieces)
    n_central = sum(p["n_central"] for p in pieces)
    generated_offsets = np.cumsum([0] + [p["n_generated"] for p in pieces[:-1]])
    features = np.vstack([p["features"] for p in pieces])
    mx = np.concatenate([p["mx"] for p in pieces]) if pieces else np.empty(0)
    event_ids = np.concatenate(
        [
            p["event_indices"].astype(np.int64) + offset
            for p, offset in zip(pieces, generated_offsets)
        ]
    )
    print(
        f"{spec['name']}: files={len(files)} generated={n_generated} "
        f"central_selected={n_central} with_proton_pair={features.shape[0]} "
        f"(pair_eff={features.shape[0] / max(n_central, 1):.3f})",
        flush=True,
    )
    group = np.char.add(f"{spec['name']}:", event_ids.astype(str))
    split_salt = 1001 if spec["name"] == "Hbb" else 2001
    return {
        "name": spec["name"], "features": features, "mx": mx, "group": group,
        "split": deterministic_group_split(event_ids, split_salt, split_seed),
        "n_generated": n_generated, "n_draws": 1, "bx_pair_acceptance": 1.0,
        "input_stats": {
            "root_files": len(files),
            "root_entries": n_generated,
            "central_selected": n_central,
            "proton_selected": int(features.shape[0]),
        },
    }


def load_bootstrap_pool(minbias_campaign):
    """Global weighted min-bias proton-pair pool from minbias_analyzer_bootstrap.py.

    Returns the per-pair (mx, yx) and sampling weight, plus the Poisson-sampled
    fraction of BX carrying a window pair (the physical MadGraph normalization)."""
    campaign_dir, campaign = resolve_minbias_campaign(minbias_campaign)
    pool_path = campaign_dir / "pairs" / "proton_pairs.parquet"
    meta_path = campaign_dir / "pairs" / "metadata.json"
    if not pool_path.is_file():
        raise RuntimeError(
            f"Missing bootstrap proton-pair pool: {pool_path}; "
            "run analysis/minbias_analyzer_bootstrap.py first"
        )
    table = pq.read_table(pool_path, columns=["mx", "yx", "weight"])
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {
        "mx": np.asarray(table["mx"], dtype=np.float64),
        "yx": np.asarray(table["yx"], dtype=np.float64),
        "weight": np.asarray(table["weight"], dtype=np.float64),
        "acceptance": float(metadata.get("bx_pair_acceptance_poisson", float("nan"))),
        "path": str(pool_path),
        "campaign": campaign,
    }


def load_madgraph_sample(
    spec, campaign_inputs, tree_name, collection, max_files_override,
    minbias_campaign, train_pairs, evaluation_pairs, seed, split_seed,
):
    campaigns = []
    for campaign_input in campaign_inputs:
        campaign = campaign_input["campaign"]
        max_files = max_files_override or spec["max_files"]
        files = campaign_input["input_files"][:max_files]
        if not files:
            raise RuntimeError(f"No ROOT files resolved for {spec['name']}/{campaign}")
        tasks = [(path, tree_name, collection) for path in files]
        with ProcessPoolExecutor(max_workers=min(ROOT_READ_WORKERS, len(tasks))) as executor:
            pieces = list(executor.map(process_file_madgraph, tasks))

        generated_offset = 0
        for piece in pieces:
            piece["global_event_indices"] = piece["event_indices"] + generated_offset
            generated_offset += piece["n_generated"]

        def concat(name, dtype=np.float64):
            values = [p[name] for p in pieces if p[name].size]
            return np.concatenate(values) if values else np.empty(0, dtype=dtype)

        xsec_fb, xsec_source = generator_cross_section_fb(
            spec["generator"], spec["process"], campaign
        )
        lhe_xsecs_fb = np.asarray([
            1000.0 * parse_lhe_xsec_pb(path)
            for path in event_record_files(spec["generator"], spec["process"], campaign)
        ])
        campaigns.append(
            {
                "campaign": campaign,
                "subcampaign": campaign_input["subcampaign"],
                "input_dir": campaign_input["input_dir"],
                "files": files,
                "matrix": np.vstack([p["matrix"] for p in pieces]),
                "dijet_rapidity": concat("dijet_rapidity"),
                "event_index": concat("global_event_indices", np.int64).astype(np.int64),
                "hard_bb_valid": concat("hard_bb_valid", bool).astype(bool),
                "hard_b1_pt": concat("hard_b1_pt"),
                "hard_b2_pt": concat("hard_b2_pt"),
                "hard_b1_eta": concat("hard_b1_eta"),
                "hard_b2_eta": concat("hard_b2_eta"),
                "hard_bb_mass": concat("hard_bb_mass"),
                "n_generated": generated_offset,
                "xsec_fb": xsec_fb,
                "xsec_source": xsec_source,
                "xsec_std_fb": float(np.std(lhe_xsecs_fb)),
            }
        )

    mc_luminosities = {
        sample["campaign"]: sample["n_generated"] / sample["xsec_fb"]
        for sample in campaigns
    }
    campaign_stats = {}
    for sample in campaigns:
        phase_masks = madgraph_phase_space_masks(sample)
        coverage_mask = np.zeros_like(
            next(iter(phase_masks.values())), dtype=np.uint8
        )
        for bit, name in enumerate(mc_luminosities):
            coverage_mask |= np.asarray(phase_masks[name], dtype=np.uint8) << bit
        stitch_weight_fb = stitched_cross_section_weights(
            sample["campaign"], phase_masks, mc_luminosities
        )
        keep = stitch_weight_fb > 0.0
        overlap = np.sum(
            np.column_stack([phase_masks[name] for name in mc_luminosities]), axis=1
        ) > 1
        sample["matrix"] = sample["matrix"][keep]
        sample["dijet_rapidity"] = sample["dijet_rapidity"][keep]
        sample["event_index"] = sample["event_index"][keep]
        sample["stitch_weight_fb"] = stitch_weight_fb[keep]
        sample["coverage_mask"] = coverage_mask[keep]
        sample["group"] = np.char.add(
            f"{spec['name']}:{sample['campaign']}:",
            sample["event_index"].astype(str),
        )
        campaign_stats[sample["campaign"]] = {
            "subcampaign": MADGRAPH_CAMPAIGNS[sample["campaign"]]["subcampaign"],
            "files": len(sample["files"]),
            "n_generated": sample["n_generated"],
            "n_central_before_stitching": int(keep.size),
            "n_central_kept": int(np.sum(keep)),
            "n_central_in_overlap": int(np.sum(keep & overlap)),
            "xsec_fb": sample["xsec_fb"],
            "xsec_source": sample["xsec_source"],
            "xsec_std_fb": sample["xsec_std_fb"],
            "effective_mc_luminosity_fb_inv": mc_luminosities[sample["campaign"]],
            "stitched_central_cross_section_fb": float(np.sum(stitch_weight_fb[keep])),
            "coverage_counts": {
                str(int(mask)): int(np.sum(coverage_mask[keep] == mask))
                for mask in np.unique(coverage_mask[keep])
            },
            "truth_histograms": {
                "minimum_parton_pt_gev": weighted_histogram_summary(
                    np.minimum(sample["hard_b1_pt"], sample["hard_b2_pt"])[keep],
                    stitch_weight_fb[keep],
                    np.arange(0.0, 115.0, 5.0),
                ),
                "maximum_abs_parton_eta": weighted_histogram_summary(
                    np.maximum(
                        np.abs(sample["hard_b1_eta"]),
                        np.abs(sample["hard_b2_eta"]),
                    )[keep],
                    stitch_weight_fb[keep],
                    np.arange(0.0, 3.6, 0.1),
                ),
                "parton_dijet_mass_gev": weighted_histogram_summary(
                    sample["hard_bb_mass"][keep],
                    stitch_weight_fb[keep],
                    np.arange(40.0, 195.0, 5.0),
                ),
            },
        }
        job_indices = []
        for path in sample["files"]:
            match = re.search(r"_(\d+)(?:__part\d+)?\.root$", path.name)
            if match:
                job_indices.append(int(match.group(1)))
        generation_metadata = (
            generation_campaign_root("madgraph", spec["process"], sample["campaign"])
            / generation_config("madgraph")["generation_dir"]
            / "metadata.yaml"
        )
        expected_jobs = None
        if generation_metadata.is_file():
            expected_jobs = (yaml.safe_load(generation_metadata.read_text()) or {}).get("jobs")
        simulation_metadata_path = sample["input_dir"].parent / "metadata.yaml"
        simulation_metadata = (
            yaml.safe_load(simulation_metadata_path.read_text()) or {}
            if simulation_metadata_path.is_file() else {}
        )
        simulation_tag = simulation_metadata.get("sim_delphes_tag", sample["subcampaign"])
        campaign_stats[sample["campaign"]].update({
            "root_files": len(sample["files"]),
            "root_entries": sample["n_generated"],
            "expected_generation_jobs": expected_jobs,
            "missing_job_indices": (
                sorted(set(range(1, int(expected_jobs) + 1)) - set(job_indices))
                if expected_jobs is not None and job_indices else []
            ),
            "simulation_tag": simulation_tag,
            "simulation_metadata": str(simulation_metadata_path),
            "simulation_pipeline": simulation_metadata.get("pipeline"),
            "delphes_card": simulation_metadata.get("delphes_card"),
            "fsr": simulation_metadata.get("fsr"),
            "pileup_condition": (
                "200PU" if "200PU" in str(simulation_tag) else "not encoded in tag"
            ),
        })
        print(
            f"{spec['name']}/{sample['campaign']}: files={len(sample['files'])} "
            f"generated={sample['n_generated']} central_selected={keep.size} "
            f"stitch_kept={np.sum(keep)} overlap={np.sum(keep & overlap)} "
            f"xsec={sample['xsec_fb']:.6g} fb "
            f"mc_lumi={mc_luminosities[sample['campaign']]:.6g} fb^-1",
            flush=True,
        )

    central_matrix = np.vstack([sample["matrix"] for sample in campaigns])
    dijet_rapidity = np.concatenate([sample["dijet_rapidity"] for sample in campaigns])
    central_group = np.concatenate([sample["group"] for sample in campaigns])
    central_campaign = np.concatenate(
        [
            np.full(sample["matrix"].shape[0], sample["campaign"])
            for sample in campaigns
        ]
    )
    stitch_weight_fb = np.concatenate(
        [sample["stitch_weight_fb"] for sample in campaigns]
    )
    central_coverage_mask = np.concatenate(
        [sample["coverage_mask"] for sample in campaigns]
    )
    central_split = np.concatenate(
        [
            deterministic_group_split(
                sample["event_index"],
                3001 + index * 1000,
                split_seed,
            )
            for index, sample in enumerate(campaigns)
        ]
    )
    n_central = central_matrix.shape[0]

    try:
        pool = load_bootstrap_pool(minbias_campaign)
    except RuntimeError as exc:
        print(
            f"Warning: bootstrap proton-pair pool unavailable for {spec['name']} ({exc}); "
            "caching this class with the proton block left as NaN.",
            flush=True,
        )
        proton_block = np.full((n_central, len(PROTON_FEATURE_NAMES)), np.nan)
        return {
            "name": spec["name"],
            "features": np.column_stack([central_matrix, proton_block]),
            "mx": np.full(n_central, np.nan),
            "group": central_group,
            "source_campaign": central_campaign,
            "split": central_split,
            "stitch_weight_fb": stitch_weight_fb,
            "n_draws": 1,
            "bx_pair_acceptance": float("nan"),
            "campaign_stats": campaign_stats,
        }

    sampled = conditional_pair_sample(
        central_matrix,
        dijet_rapidity,
        central_group,
        central_campaign,
        central_split,
        stitch_weight_fb,
        central_coverage_mask,
        pool,
        train_pairs,
        evaluation_pairs,
        seed,
    )
    bx_pair_acceptance = pool["acceptance"]
    print(
        f"{spec['name']}: stitched_central={n_central} pool={pool['path']} "
        f"({pool['mx'].size} pairs) cached_conditional_pairs="
        f"{evaluation_pairs}/group ({train_pairs} used for training) "
        f"rows_with_proton_pair={sampled['features'].shape[0]} "
        f"bx_pair_acceptance={bx_pair_acceptance:.6g}",
        flush=True,
    )
    return {
        "name": spec["name"], **sampled,
        "n_draws": 1,
        "bx_pair_acceptance": bx_pair_acceptance, "campaign_stats": campaign_stats,
    }


def distance_correlation(a, b, rng, max_rows=DCORR_MAX_ROWS):
    """Szekely distance correlation between two 1D samples (0 = independent).

    O(n^2) in memory via the double-centered distance matrices, so the inputs
    are subsampled to max_rows first. Returns NaN if fewer than 2 finite pairs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2:
        return float("nan")
    if a.size > max_rows:
        pick = rng.choice(a.size, size=max_rows, replace=False)
        a, b = a[pick], b[pick]
    A = np.abs(a[:, None] - a[None, :])
    B = np.abs(b[:, None] - b[None, :])
    A -= A.mean(axis=0, keepdims=True) + A.mean(axis=1, keepdims=True) - A.mean()
    B -= B.mean(axis=0, keepdims=True) + B.mean(axis=1, keepdims=True) - B.mean()
    dcov2 = float(np.mean(A * B))
    dvar_a = float(np.mean(A * A))
    dvar_b = float(np.mean(B * B))
    denom = np.sqrt(dvar_a * dvar_b)
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def class_balanced_weights(classes, mixture_weights):
    """Equal total weight per class while retaining its physical composition."""
    weights = np.asarray(mixture_weights, dtype=np.float64).copy()
    n_classes = len(np.unique(classes))
    for value in np.unique(classes):
        mask = classes == value
        total = np.sum(weights[mask])
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"Class {value} has non-positive mixture weight")
        weights[mask] *= classes.shape[0] / (n_classes * total)
    return weights


def pair_balanced_weights(labels, mixture_weights):
    weights = np.asarray(mixture_weights, dtype=np.float64).copy()
    for label in (0, 1):
        mask = labels == label
        weights[mask] /= np.sum(weights[mask])
    return weights


def pairwise_scores(probabilities, background_class):
    with np.errstate(divide="ignore", invalid="ignore"):
        return probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, background_class])


def pairwise_auc(roc_auc_score, dataset, probabilities, background):
    background_class = CLASS_ORDER.index(background)
    mask = (dataset["process"] == "Hbb") | (dataset["process"] == background)
    if not np.any(dataset["process"] == background):
        return float("nan")
    labels = (dataset["process"][mask] == "Hbb").astype(np.int8)
    scores = pairwise_scores(probabilities[mask], background_class)
    return float(
        roc_auc_score(
            labels,
            scores,
            sample_weight=pair_balanced_weights(
                labels, dataset["training_mixture_weight"][mask]
            ),
        )
    )


def make_model(XGBClassifier, params, seed):
    return XGBClassifier(
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=8,
        tree_method="hist",
        early_stopping_rounds=30,
        **params,
    )


def fit_model(XGBClassifier, StratifiedGroupKFold, x, classes, groups, weights, params, seed):
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fit_idx, stop_idx = next(splitter.split(x, classes, groups))
    model = make_model(XGBClassifier, params, seed)
    model.fit(
        x[fit_idx], classes[fit_idx], sample_weight=weights[fit_idx],
        eval_set=[(x[stop_idx], classes[stop_idx])],
        sample_weight_eval_set=[weights[stop_idx]],
        verbose=False,
    )
    return model


def grid_search(XGBClassifier, StratifiedGroupKFold, roc_auc_score, dataset, weights, seed, rng):
    x, classes = dataset["x"], dataset["class"]
    rows = np.arange(classes.shape[0])
    if rows.size > SEARCH_MAX_ROWS:
        rows = rng.choice(rows, size=SEARCH_MAX_ROWS, replace=False)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_local, test_local = next(
        splitter.split(x[rows], classes[rows], dataset["group_id"][rows])
    )
    train_idx, test_idx = rows[train_local], rows[test_local]
    results = []
    for params in PARAM_GRID:
        model = fit_model(
            XGBClassifier, StratifiedGroupKFold, x[train_idx], classes[train_idx],
            dataset["group_id"][train_idx], weights[train_idx], params, seed,
        )
        probabilities = model.predict_proba(x[test_idx])
        subset = {
            "process": dataset["process"][test_idx],
            "training_mixture_weight": dataset["training_mixture_weight"][test_idx],
        }
        auc_sc = pairwise_auc(roc_auc_score, subset, probabilities, "QCDbb")
        auc_mg = pairwise_auc(roc_auc_score, subset, probabilities, "QCDbb_madgraph")
        mean_auc = np.nanmean([auc_sc, auc_mg])
        results.append(
            {**params, "mean_pairwise_auc": float(mean_auc), "auc_sc": float(auc_sc),
             "auc_mg": float(auc_mg), "best_iteration": int(model.best_iteration)}
        )
        print(
            f"search: depth={params['max_depth']} lr={params['learning_rate']} "
            f"min_child_weight={params['min_child_weight']} mean_auc={mean_auc:.5f} "
            f"(sc={auc_sc:.5f}, mg={auc_mg:.5f}) best_iter={model.best_iteration}",
            flush=True,
        )
    best = max(results, key=lambda item: item["mean_pairwise_auc"])
    print(f"best configuration: {best}", flush=True)
    return {k: best[k] for k in ("max_depth", "learning_rate", "min_child_weight")}, results


def train_oof(XGBClassifier, StratifiedGroupKFold, dataset, weights, params, seed):
    x, classes, groups = dataset["x"], dataset["class"], dataset["group_id"]
    oof = np.full((classes.shape[0], len(CLASS_ORDER)), np.nan)
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    best_iterations = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, classes, groups)):
        model = fit_model(
            XGBClassifier, StratifiedGroupKFold, x[train_idx], classes[train_idx],
            groups[train_idx], weights[train_idx], params, seed,
        )
        oof[test_idx] = model.predict_proba(x[test_idx])
        best_iterations.append(int(model.best_iteration))
        print(f"fold_{fold}: train={train_idx.size} test={test_idx.size} best_iter={model.best_iteration}", flush=True)
    if np.any(np.isnan(oof)):
        raise RuntimeError("Cross-validation left events without an out-of-fold score")
    final_model = fit_model(
        XGBClassifier, StratifiedGroupKFold, x, classes, groups, weights, params, seed
    )
    return oof, best_iterations, final_model


def plot_outputs(roc_curve, dataset, oof, model, output_dir):
    styles = {"Hbb": ("#0072B2", "-"), "QCDbb": ("#E69F00", "--"), "QCDbb_madgraph": ("#D55E00", "-.")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    bins = np.linspace(0.0, 1.0, 51)
    for ax, background in zip(axes, ("QCDbb", "QCDbb_madgraph")):
        background_class = CLASS_ORDER.index(background)
        scores = pairwise_scores(oof, background_class)
        for name, (color, linestyle) in styles.items():
            mask = dataset["process"] == name
            if not np.any(mask):
                continue
            ax.hist(
                scores[mask], bins=bins, density=True, histtype="step",
                color=color, linestyle=linestyle, linewidth=1.6, label=CLASS_LABELS[name],
                weights=dataset["training_mixture_weight"][mask],
            )
        ax.set_xlabel(f"p(sig) / (p(sig) + p({background}))")
        ax.set_ylabel("Normalized events / bin")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_pairwise_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for background in ("QCDbb", "QCDbb_madgraph"):
        if not np.any(dataset["process"] == background):
            continue
        color, linestyle = styles[background]
        background_class = CLASS_ORDER.index(background)
        mask = (dataset["process"] == "Hbb") | (dataset["process"] == background)
        labels = (dataset["process"][mask] == "Hbb").astype(np.int8)
        fpr, tpr, _ = roc_curve(
            labels, pairwise_scores(oof[mask], background_class),
            sample_weight=pair_balanced_weights(
                labels, dataset["training_mixture_weight"][mask]
            ),
        )
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, label=f"Hbb vs {CLASS_LABELS[background]}")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc.png", dpi=160)
    plt.close(fig)

    # sculpting check: SC-QCDbb background mx spectrum in slices of the vs-SC
    # score. Overlapping shapes => the score does not sculpt the fit variable.
    sc_class = CLASS_ORDER.index("QCDbb")
    sc_mask = dataset["process"] == "QCDbb"
    sc_score = pairwise_scores(oof[sc_mask], sc_class)
    sc_mx = dataset["mx"][sc_mask]
    finite = np.isfinite(sc_mx)
    sc_score, sc_mx = sc_score[finite], sc_mx[finite]
    fig, ax = plt.subplots(figsize=(8, 6))
    if sc_mx.size:
        mx_bins = np.linspace(MASS_WINDOW_GEV[0], MASS_WINDOW_GEV[1], 33)
        cuts = [(0.0, 0.5, "score < 0.5", "#56B4E9"), (0.5, 1.01, "score >= 0.5", "#D55E00")]
        for low, high, label, color in cuts:
            sel = (sc_score >= low) & (sc_score < high)
            if np.any(sel):
                ax.hist(
                    sc_mx[sel], bins=mx_bins, density=True, histtype="step",
                    linewidth=1.6, color=color, label=f"{label} (n={int(np.sum(sel))})",
                )
    ax.set_xlabel("mx [GeV] (SuperChic QCDbb background)")
    ax.set_ylabel("Normalized events / bin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mx_sculpting_check.png", dpi=160)
    plt.close(fig)

    names = list(FEATURE_NAMES)
    importance = model.get_booster().get_score(importance_type="gain")
    gains = np.array([importance.get(f"f{i}", 0.0) for i in range(len(names))])
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1.5))
    ax.barh(np.arange(len(names)), gains[order], color="#0072B2")
    ax.set_yticks(np.arange(len(names)), [names[i] for i in order], fontsize=8)
    ax.set_xlabel("XGBoost gain")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def write_cache(cache_dir, dataset, metadata):
    """Write one .npy per column so training can memory-map without ROOT I/O."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": np.asarray(dataset["x"], dtype=np.float32),
        "class": np.asarray(dataset["class"], dtype=np.int8),
        "physical_weight": np.asarray(dataset["physical_weight"], dtype=np.float64),
        "training_mixture_weight": np.asarray(
            dataset["training_mixture_weight"], dtype=np.float64
        ),
        "group_id": np.asarray(dataset["group_id"], dtype=np.int64),
        "split": np.asarray(dataset["split"], dtype=np.uint8),
        "mx": np.asarray(dataset["mx"], dtype=np.float32),
        "source_campaign": np.asarray(dataset["source_campaign"]),
        "central_stitch_weight_fb": np.asarray(
            dataset["central_stitch_weight_fb"], dtype=np.float64
        ),
        "band_probability": np.asarray(
            dataset["band_probability"], dtype=np.float32
        ),
        "coverage_mask": np.asarray(dataset["coverage_mask"], dtype=np.uint8),
    }
    for name, values in arrays.items():
        np.save(cache_dir / f"{name}.npy", values, allow_pickle=False)
    with open(cache_dir / "metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    with open(cache_dir / "input_manifest.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata["input_manifest"], handle, sort_keys=False)


def main():
    started = time.perf_counter()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    card_manifest = validate_madgraph_campaign_cards(args.madgraph_campaigns)
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    pps_path = resolve_path(args.pps_config, base=REPO)
    pps = pps_lib.load_pps_config(pps_path)
    print(f"PPS config: {pps_path} (sqrt_s={pps['sqrt_s']:.0f} GeV, xi_res={pps['xi_res']:.4g})", flush=True)

    seed_seq = np.random.SeedSequence(args.seed)
    sample_seeds = seed_seq.spawn(len(SAMPLES))
    samples = {}
    for spec, sample_seed in zip(SAMPLES, sample_seeds):
        if spec["generator"] == "madgraph":
            campaign_inputs = []
            for campaign in args.madgraph_campaigns:
                subcampaign = MADGRAPH_CAMPAIGNS[campaign]["subcampaign"]
                input_dir = generation_stage_root(
                    spec["generator"], spec["process"], campaign, "sim-delphes",
                    subcampaign=subcampaign,
                ) / "root"
                input_files = sorted(input_dir.glob("*.root"), key=natural_key)
                if not input_files:
                    raise RuntimeError(
                        f"No ROOT files for {spec['name']} under {input_dir}"
                    )
                print(
                    f"{spec['name']}: campaign={campaign}/{subcampaign}, "
                    f"{len(input_files)} ROOT files in {input_dir}",
                    flush=True,
                )
                campaign_inputs.append(
                    {
                        "campaign": campaign,
                        "subcampaign": subcampaign,
                        "input_dir": input_dir,
                        "input_files": input_files,
                    }
                )
            samples[spec["name"]] = load_madgraph_sample(
                spec, campaign_inputs, args.tree, args.collection, args.max_files,
                args.minbias_campaign, args.train_pairs, args.evaluation_pairs,
                int(sample_seed.generate_state(1)[0]), args.seed,
            )
            continue

        campaign, subcampaign, input_dir, input_files, proton_files = pps_lib.resolve_sample_inputs(
            spec, args.campaign
        )
        if not input_files:
            raise RuntimeError(f"No ROOT files for {spec['name']} under {input_dir}")
        print(
            f"{spec['name']}: campaign={campaign}/{subcampaign}, {len(input_files)} ROOT files in {input_dir}",
            flush=True,
        )
        samples[spec["name"]] = load_superchic_sample(
            spec, input_files, proton_files, args.tree, args.collection, args.max_files, pps,
            int(sample_seed.generate_state(1)[0]), args.seed,
        )
        simulation_metadata_path = input_dir.parent / "metadata.yaml"
        simulation_metadata = (
            yaml.safe_load(simulation_metadata_path.read_text()) or {}
            if simulation_metadata_path.is_file() else {}
        )
        simulation_tag = simulation_metadata.get("sim_delphes_tag", subcampaign)
        samples[spec["name"]]["input_stats"].update({
            "campaign": campaign,
            "subcampaign": subcampaign,
            "simulation_tag": simulation_tag,
            "simulation_metadata": str(simulation_metadata_path),
            "simulation_pipeline": simulation_metadata.get("pipeline"),
            "delphes_card": simulation_metadata.get("delphes_card"),
            "fsr": simulation_metadata.get("fsr"),
            "pileup_condition": (
                "200PU" if "200PU" in str(simulation_tag) else "not encoded in tag"
            ),
        })

    spec_by_name = {spec["name"]: spec for spec in SAMPLES}
    tag = pps_lib.tag_weight(load_yaml(REPO / "parameters.yaml"))
    event_weights = {}
    mixture_weights = {}
    for name in CLASS_ORDER:
        sample = samples[name]
        n_rows = sample["features"].shape[0]
        if name == "QCDbb_madgraph":
            mixture_weights[name] = sample["stitch_weight_fb"]
            physical_scale = (
                LUMI_FB
                * generator_weight(spec_by_name[name]["generator"], spec_by_name[name]["process"])
                * COMBINATORIAL_ACCEPTANCE_FACTOR
                * sample["bx_pair_acceptance"]
                * tag
            )
            event_weights[name] = sample["stitch_weight_fb"] * physical_scale
            print(
                f"{name}: stitched event_weight range="
                f"[{np.min(event_weights[name]):.6g}, {np.max(event_weights[name]):.6g}], "
                f"expected yield={np.sum(event_weights[name]):.6g}",
                flush=True,
            )
        else:
            event_weight = physical_event_weight(
                spec_by_name[name], sample["n_generated"], sample["n_draws"],
                sample["bx_pair_acceptance"], tag,
            )
            event_weights[name] = np.full(n_rows, event_weight)
            mixture_weights[name] = np.full(
                n_rows,
                generator_cross_section_fb(
                    spec_by_name[name]["generator"], spec_by_name[name]["process"]
                )[0] / (sample["n_generated"] * sample["n_draws"]),
            )
            print(
                f"{name}: physical event_weight={event_weight:.6g} "
                f"(n_generated={sample['n_generated']}, n_draws={sample['n_draws']}, "
                f"bx_pair_acceptance={sample['bx_pair_acceptance']:.6g}), "
                f"expected yield={np.sum(event_weights[name]):.6g}",
                flush=True,
            )

    dataset = {
        "x": np.vstack([samples[name]["features"] for name in CLASS_ORDER]),
        "process": np.concatenate(
            [np.full(samples[name]["features"].shape[0], name) for name in CLASS_ORDER]
        ),
        "physical_weight": np.concatenate(
            [event_weights[name] for name in CLASS_ORDER]
        ),
        "training_mixture_weight": np.concatenate(
            [mixture_weights[name] for name in CLASS_ORDER]
        ),
        "group": np.concatenate([samples[name]["group"] for name in CLASS_ORDER]),
        "source_campaign": np.concatenate(
            [
                samples[name].get(
                    "source_campaign",
                    np.full(samples[name]["features"].shape[0], ""),
                )
                for name in CLASS_ORDER
            ]
        ),
        "central_stitch_weight_fb": np.concatenate([
            np.asarray(
                samples[name].get(
                    "central_stitch_weight_fb",
                    np.full(samples[name]["features"].shape[0], np.nan),
                )
            )
            for name in CLASS_ORDER
        ]),
        "band_probability": np.concatenate([
            np.asarray(
                samples[name].get(
                    "band_probability",
                    np.full(samples[name]["features"].shape[0], np.nan),
                )
            )
            for name in CLASS_ORDER
        ]),
        "coverage_mask": np.concatenate([
            np.asarray(
                samples[name].get(
                    "coverage_mask",
                    np.zeros(samples[name]["features"].shape[0], dtype=np.uint8),
                ),
                dtype=np.uint8,
            )
            for name in CLASS_ORDER
        ]),
        "mx": np.concatenate([samples[name]["mx"] for name in CLASS_ORDER]),
        "split": np.concatenate([samples[name]["split"] for name in CLASS_ORDER]),
        "feature_names": np.asarray(FEATURE_NAMES),
    }
    class_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    dataset["class"] = np.array([class_index[p] for p in dataset["process"]], dtype=np.int8)
    _group_names, dataset["group_id"] = np.unique(dataset["group"], return_inverse=True)
    print(f"dataset: {dataset['x'].shape[0]} events, {dataset['x'].shape[1]} features", flush=True)

    cache_metadata = {
        "format_version": 1,
        "feature_schema_version": 2,
        "created_by": str(Path(__file__).relative_to(REPO)),
        "features": list(FEATURE_NAMES),
        "track_selection": (
            "EFlowTrack.IsRecoPU == 0, pT >= 0.5 GeV, |eta| < 2.5"
        ),
        "activity_regions": {
            "legacy_interjet": "outside both jet cones and eta between the jet axes",
            "projected_bridge": (
                "outside both jet cones, 0 < u < 1 and |v| < 0.4 in the "
                "finite wrapped (eta, phi) jet--jet frame"
            ),
            "projected_side": (
                "outside both jet cones and outside the projected bridge and "
                "beam-side outer regions"
            ),
        },
        "track_jets": {
            "algorithm": "anti-kt",
            "R": TRACK_JET_R,
            "min_pt_gev": TRACK_JET_MIN_PT,
        },
        "classes": list(CLASS_ORDER),
        "split_names": list(SPLIT_NAMES),
        "split_fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS)),
        "rows": int(dataset["x"].shape[0]),
        "groups": int(np.unique(dataset["group_id"]).size),
        "rows_per_split": {
            name: int(np.sum(dataset["split"] == index))
            for index, name in enumerate(SPLIT_NAMES)
        },
        "groups_per_split": {
            name: int(np.unique(dataset["group_id"][dataset["split"] == index]).size)
            for index, name in enumerate(SPLIT_NAMES)
        },
        "rows_per_class": {
            name: int(np.sum(dataset["class"] == index))
            for index, name in enumerate(CLASS_ORDER)
        },
        "madgraph_campaigns": list(args.madgraph_campaigns),
        "madgraph_coverage_bits": {
            name: bit for bit, name in enumerate(args.madgraph_campaigns)
        },
        "madgraph_stitching": samples["QCDbb_madgraph"]["campaign_stats"],
        "input_manifest": {
            "madgraph_cards": card_manifest,
            "samples": {
                name: samples[name].get(
                    "input_stats",
                    samples[name].get("campaign_stats", {}),
                )
                for name in CLASS_ORDER
            },
        },
        "conditional_pairs": {
            "cached_per_madgraph_group": args.evaluation_pairs,
            "used_per_training_group": args.train_pairs,
            "used_per_held_out_group": args.evaluation_pairs,
        },
        "nominal_combinatorial_acceptance": COMBINATORIAL_ACCEPTANCE_FACTOR,
        "bx_pair_acceptance": float(
            samples["QCDbb_madgraph"]["bx_pair_acceptance"]
        ),
        "mass_window_gev": list(MASS_WINDOW_GEV),
        "max_abs_rapidity_difference": MAX_ABS_RAPIDITY_DIFFERENCE,
        "seed": args.seed,
        "preparation_runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": {
            "main_process": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "worker_process_max": float(
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
            ),
        },
    }
    write_cache(Path(args.cache_dir), dataset, cache_metadata)
    print(f"Wrote memory-mappable cache to {args.cache_dir}", flush=True)
    if args.prepare_only:
        return

    weights = class_balanced_weights(
        dataset["class"], dataset["training_mixture_weight"]
    )
    rng = np.random.default_rng(args.seed)
    if args.skip_search:
        best_params = {"max_depth": 6, "learning_rate": 0.02, "min_child_weight": 20}
        search_results = []
    else:
        best_params, search_results = grid_search(
            XGBClassifier, StratifiedGroupKFold, roc_auc_score,
            dataset, weights, args.seed, rng
        )

    oof, best_iterations, model = train_oof(
        XGBClassifier, StratifiedGroupKFold, dataset, weights, best_params, args.seed
    )
    auc_sc = pairwise_auc(roc_auc_score, dataset, oof, "QCDbb")
    auc_mg = pairwise_auc(roc_auc_score, dataset, oof, "QCDbb_madgraph")
    print(f"OOF pairwise AUC Hbb vs SuperChic QCDbb: {auc_sc:.5f}")
    print(f"OOF pairwise AUC Hbb vs MadGraph QCDbb:  {auc_mg:.5f}")

    # sculpting diagnostic: distance correlation between the vs-SC pairwise
    # score and mx on the SuperChic QCDbb background (the smooth-mx background
    # the peak fit sits on). 0 => score independent of the fit variable.
    dcorr_rng = np.random.default_rng(args.seed + 1)
    sc_mask = dataset["process"] == "QCDbb"
    sc_score = pairwise_scores(oof[sc_mask], CLASS_ORDER.index("QCDbb"))
    dcorr_sc = distance_correlation(sc_score, dataset["mx"][sc_mask], dcorr_rng)
    sig_mask = dataset["process"] == "Hbb"
    sig_score = pairwise_scores(oof[sig_mask], CLASS_ORDER.index("QCDbb"))
    dcorr_sig = distance_correlation(sig_score, dataset["mx"][sig_mask], dcorr_rng)
    print(f"dCorr(vs-SC score, mx) on SuperChic QCDbb: {dcorr_sc:.4f} (0 = no sculpting)")
    print(f"dCorr(vs-SC score, mx) on Hbb signal:      {dcorr_sig:.4f}")

    np.savez_compressed(output_dir / "dataset.npz", **dataset)
    np.savez_compressed(output_dir / "scores.npz", oof_probabilities=oof, process=dataset["process"])
    model._estimator_type = "classifier"
    model.save_model(output_dir / "model.json")
    plot_outputs(roc_curve, dataset, oof, model, output_dir)

    mg_has_protons = bool(
        np.any(np.isfinite(dataset["mx"][dataset["process"] == "QCDbb_madgraph"]))
    )
    summary = {
        "pps_config": str(pps_path),
        "sqrt_s_gev": pps["sqrt_s"],
        "xi_resolution": pps["xi_res"],
        "mass_window_gev": list(MASS_WINDOW_GEV),
        "max_abs_rapidity_difference": MAX_ABS_RAPIDITY_DIFFERENCE,
        "conditional_pairs_madgraph": {
            "train": args.train_pairs,
            "held_out": args.evaluation_pairs,
        },
        "madgraph_campaigns": list(args.madgraph_campaigns),
        "madgraph_stitching": samples["QCDbb_madgraph"]["campaign_stats"],
        "madgraph_has_real_proton_pairs": mg_has_protons,
        "features": list(FEATURE_NAMES),
        "events_per_class": {
            name: int(np.sum(dataset["process"] == name)) for name in CLASS_ORDER
        },
        "effective_events_per_class": {
            name: float(
                np.sum(dataset["training_mixture_weight"][dataset["process"] == name]) ** 2
                / np.sum(dataset["training_mixture_weight"][dataset["process"] == name] ** 2)
            )
            for name in CLASS_ORDER
        },
        "expected_yield_per_class": {
            name: float(np.sum(dataset["physical_weight"][dataset["process"] == name]))
            for name in CLASS_ORDER
        },
        "hyperparameter_search": search_results,
        "best_params": best_params,
        "cv_best_iterations": best_iterations,
        "auc_oof_pairwise_hbb_vs_superchic_qcdbb": auc_sc,
        "auc_oof_pairwise_hbb_vs_madgraph_qcdbb": auc_mg,
        "dcorr_vs_sc_score_mx_on_superchic_qcdbb": dcorr_sc,
        "dcorr_vs_sc_score_mx_on_hbb": dcorr_sig,
        "seed": args.seed,
    }
    with open(output_dir / "summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"Wrote dataset, model, plots, and summary to {output_dir}")
    if not mg_has_protons:
        print(
            "NOTE: QCDbb_madgraph has no proton-block features this run (min-bias data "
            "unavailable); its vs-MG AUC reflects central features only.",
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
