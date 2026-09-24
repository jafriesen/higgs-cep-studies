#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "parameters.yaml", encoding="utf-8") as _parameters_handle:
    _parameters = yaml.safe_load(_parameters_handle)

LUMI_FB = 3000.0
MIN_JET_PT = 15.0
OUTER_TRACK_MIN_R = 0.4
OUTER_TRACK_MIN_PT = 1.0
IN_JET_TRACK_MAX_R = 0.4
IN_JET_TRACK_MIN_PT = 1.0
TRACK_MAX_ABS_ETA = 2.5
MAX_OUTER_TRACK_MULTIPLICITY = 0
MASS_WINDOW_GEV = (117.0, 133.0)
PRE_MVA_DIJET_MASS_RANGE_GEV = (50.0, 150.0)
MIN_DIJET_DELTA_PHI = 3.0
MAX_ABS_RAPIDITY_DIFFERENCE = 0.2
COMBINATORIAL_ACCEPTANCE_FACTOR = float(
    _parameters["normalization"]["combinatorial_acceptance_factor"]
)
MINBIAS_PAIR_RESAMPLING = 20
CV_FOLDS = 5
ROOT_READ_WORKERS = 8
BASE_FEATURE_NAMES = (
    "jet1_pt_over_mjj",
    "jet2_pt_over_mjj",
    "jet1_pt",
    "jet2_pt",
    "jet1_eta",
    "jet2_eta",
    "delta_r_jj",
    "delta_phi_jj",
    "delta_eta_jj",
    "dijet_pt",
    "dijet_eta",
    "jet_multiplicity",
    "track_multiplicity_r_gt_0p4_pt1",
    "outer_track_sum_pt",
    "jet1_girth",
    "jet2_girth",
    "dijet_mass",
    "mx_minus_dijet_mass",
    "yx_minus_dijet_rapidity",
)
FEATURE_NAMES = BASE_FEATURE_NAMES
# Each feature set lists the features to EXCLUDE from training. The dataset
# always stores every BASE_FEATURE_NAMES column; exclusions of features absent
# from a reused --dataset-from dataset are ignored.
FEATURE_SETS = {
    "current": ("jet1_pt", "jet2_pt", "jet1_girth", "jet2_girth"),
    "no_tracks": ("jet1_pt", "jet2_pt", "jet1_girth", "jet2_girth", "track_multiplicity_r_gt_0p4_pt1", "outer_track_sum_pt"),
    "no_dijet_mass": ("jet1_pt", "jet2_pt", "dijet_mass", "jet1_girth", "jet2_girth"),
    "raw_pt_with_dijet_mass": ("jet1_pt_over_mjj", "jet2_pt_over_mjj", "jet1_girth", "jet2_girth"),
    "raw_pt_no_dijet_mass": ("jet1_pt_over_mjj", "jet2_pt_over_mjj", "dijet_mass", "jet1_girth", "jet2_girth"),
}
SAMPLE_SPECS = (
    {"name": "Hbb", "generator": "superchic", "process": "Hbb", "role": "signal"},
    {"name": "QCDbb", "generator": "superchic", "process": "QCDbb", "role": "background"},
    {
        "name": "QCDbb_madgraph_comb",
        "generator": "madgraph",
        "process": "QCDbb",
        "role": "background",
        "acceptance_factor": COMBINATORIAL_ACCEPTANCE_FACTOR,
        "pair_resampling": MINBIAS_PAIR_RESAMPLING,
    },
)
# Remove a name to exclude that channel (ignored when --dataset-from reuses a cached dataset).
ENABLED_SAMPLES = (
    "Hbb",
    "QCDbb",
    "QCDbb_madgraph_comb",
)


def repo_root():
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(ROOT))

from analysis.cross_sections import generator_cross_section_fb, generator_weight  # noqa: E402
from common.config_utils import (  # noqa: E402
    load_yaml,
    natural_key,
    resolve_minbias_campaign,
    resolve_path,
)
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_process_config,
    generation_stage_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost H->bb dijet MVA using PPS-selected proton pairs."
    )
    parser.add_argument("--flavor", choices=("bb",), default="bb", help="Signal flavor")
    parser.add_argument(
        "--feature-set",
        choices=tuple(FEATURE_SETS),
        default="current",
        help="Jet-feature variant to train.",
    )
    parser.add_argument(
        "--dataset-from",
        default=None,
        help="Reuse PPS-selected events from another run's dataset.npz.",
    )
    parser.add_argument(
        "--selection",
        choices=("pps_mass_window",),
        default="pps_mass_window",
        help="Required event selection.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Hard-process campaign. Defaults to each process default campaign.",
    )
    parser.add_argument(
        "--minbias-campaign",
        default=None,
        help="Min-bias campaign providing bx/bunch_crossings.parquet.",
    )
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file defining beam energy, PPS xi acceptance, and xi resolution.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to analysis/MVA/output/mva_bb.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Save dataset/model/scores/summary without generating plots.",
    )
    return parser.parse_args()


def ensure_delphes_python_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_ANALYZER_ENV") == "1":
        return

    lcg_view = Path(
        os.environ.get(
            "DELPHES_LCG_VIEW",
            "/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh",
        )
    )
    delphes_dir = Path(os.environ.get("DELPHES_DIR", "/home/jfriesen/Delphes"))
    if not lcg_view.is_file():
        raise RuntimeError(f"Delphes LCG view setup script does not exist: {lcg_view}")

    command = "\n".join(
        [
            f"source {shlex.quote(str(lcg_view))}",
            f"export DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"export LD_LIBRARY_PATH={shlex.quote(str(delphes_dir))}:$LD_LIBRARY_PATH",
            "export HIGGS_CEP_DELPHES_ANALYZER_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        ]
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


def import_libraries():
    import awkward as ak
    import numpy as np
    import uproot
    import vector
    import yaml
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    from xgboost import XGBClassifier

    vector.register_awkward()
    return ak, np, uproot, yaml, XGBClassifier, StratifiedGroupKFold, train_test_split, roc_curve, auc


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def load_pps_config(path):
    config = load_yaml(path)
    sqrt_s = float((config.get("beam") or {}).get("sqrt_s_gev", 0.0))
    if sqrt_s <= 0.0:
        raise RuntimeError(f"{path} must define a positive beam.sqrt_s_gev")

    xi_ranges = []
    for station, bounds in ((config.get("pps") or {}).get("xi_ranges") or {}).items():
        if len(bounds) != 2:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        xi_ranges.append((str(station), low, high))
    if not xi_ranges:
        raise RuntimeError(f"{path} must define at least one pps.xi_ranges entry")
    return {
        "sqrt_s": sqrt_s,
        "xi_ranges": xi_ranges,
        "xi_res": float((config.get("pps") or {}).get("xi_res", 0.0)),
    }


def passes_pps(np, xi, xi_ranges):
    xi = np.asarray(xi, dtype=np.float64)
    passed = np.zeros(xi.shape, dtype=bool)
    for _station, low, high in xi_ranges:
        passed |= (xi >= low) & (xi < high)
    return passed


def pair_observables(np, xi_left, xi_right, sqrt_s):
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    return (
        np.sqrt(xi_left * xi_right) * sqrt_s,
        0.5 * np.log(xi_right / xi_left),
    )


def smear_pair_observables(np, xi_left, xi_right, pps, rng):
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    if pps["xi_res"] > 0.0:
        xi_left = rng.normal(xi_left, pps["xi_res"])
        xi_right = rng.normal(xi_right, pps["xi_res"])
    valid = (xi_left > 0.0) & (xi_right > 0.0)
    mx = np.full(xi_left.shape, np.nan, dtype=np.float64)
    yx = np.full(xi_left.shape, np.nan, dtype=np.float64)
    mx[valid], yx[valid] = pair_observables(
        np, xi_left[valid], xi_right[valid], pps["sqrt_s"]
    )
    return xi_left, xi_right, mx, yx, valid


def shuffled_bx_assignments(np, n_events, n_bx, rng):
    if n_bx <= 0:
        raise RuntimeError("Min-bias input contains no bunch crossings")
    assignments = []
    remaining = n_events
    while remaining:
        permutation = rng.permutation(n_bx)
        take = min(remaining, n_bx)
        assignments.append(permutation[:take])
        remaining -= take
    return np.concatenate(assignments) if assignments else np.empty(0, dtype=np.int64)


def default_subcampaign(process_cfg, stage):
    defaults = process_cfg.get("default_campaign") or {}
    return defaults.get(stage) if isinstance(defaults, dict) else None


def resolve_sample_inputs(spec, campaign_arg):
    campaign, _campaign_cfg = generation_campaign_config(
        spec["generator"], spec["process"], campaign_arg
    )
    process_cfg = generation_process_config(spec["generator"], spec["process"])
    subcampaign = default_subcampaign(process_cfg, "sim-delphes")
    input_dir = generation_stage_root(
        spec["generator"],
        spec["process"],
        campaign,
        "sim-delphes",
        subcampaign=subcampaign,
    ) / "root"
    input_files = sorted(input_dir.glob("*.root"), key=natural_key)
    proton_files = None
    if spec["generator"] == "superchic":
        pythia_subcampaign = default_subcampaign(process_cfg, "hadr-pythia")
        pythia_dir = generation_stage_root(
            spec["generator"],
            spec["process"],
            campaign,
            "hadr-pythia",
            subcampaign=pythia_subcampaign,
        ) / "hepmc"
        hepmc_by_stem = {
            path.stem: path
            for path in sorted(pythia_dir.glob("*.hepmc"), key=natural_key)
        }
        proton_files = []
        for input_file in input_files:
            proton_file = hepmc_by_stem.get(input_file.stem)
            if proton_file is None:
                raise RuntimeError(
                    f"Missing matching HepMC file for {input_file.name} in {pythia_dir}"
                )
            proton_files.append(proton_file)
    return campaign, subcampaign, input_dir, input_files, proton_files


def parse_hepmc_proton_xi(np, input_file, selected_event_indices, sqrt_s):
    xi_left = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    xi_right = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    if selected_event_indices.size == 0:
        return xi_left, xi_right

    selected_positions = {
        int(event_index): output_index
        for output_index, event_index in enumerate(selected_event_indices)
    }
    last_selected = int(selected_event_indices[-1])
    beam_energy = sqrt_s / 2.0
    event_index = -1
    output_index = None
    left_energy = right_energy = None
    left_abs_pz = right_abs_pz = -1.0

    def store_event():
        if output_index is not None and left_energy is not None and right_energy is not None:
            xi_left[output_index] = (beam_energy - left_energy) / beam_energy
            xi_right[output_index] = (beam_energy - right_energy) / beam_energy

    with open(input_file, "r") as handle:
        for line in handle:
            if line.startswith("E "):
                if event_index >= 0:
                    store_event()
                event_index += 1
                if event_index > last_selected:
                    break
                output_index = selected_positions.get(event_index)
                left_energy = right_energy = None
                left_abs_pz = right_abs_pz = -1.0
            elif output_index is not None and line.startswith("P "):
                fields = line.split()
                if int(fields[3]) != 2212 or int(fields[9]) != 1:
                    continue
                pz = float(fields[6])
                energy = float(fields[7])
                abs_pz = abs(pz)
                if pz < 0.0 and abs_pz > left_abs_pz:
                    left_energy, left_abs_pz = energy, abs_pz
                elif pz > 0.0 and abs_pz > right_abs_pz:
                    right_energy, right_abs_pz = energy, abs_pz
        else:
            store_event()
    return xi_left, xi_right


def load_file_central(
    ak, np, uproot, input_file, tree_name, collection, with_protons,
    proton_file=None, sqrt_s=None,
):
    jet_fields = ("PT", "Eta", "Phi", "Mass")
    required = [branch_name(collection, field) for field in jet_fields]
    required.extend(
        branch_name("EFlowTrack", field) for field in ("PT", "Eta", "Phi")
    )
    if with_protons and proton_file is None:
        required.extend(
            branch_name("Particle", field) for field in ("PID", "Status", "Pz", "E")
        )

    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {input_file}: {', '.join(missing)}")
        arrays = tree.arrays(required, library="ak")
        n_generated = int(tree.num_entries)

    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    phi = ak.values_astype(arrays[branch_name(collection, "Phi")], "float64")
    mass = ak.values_astype(arrays[branch_name(collection, "Mass")], "float64")
    track_pt = ak.values_astype(arrays[branch_name("EFlowTrack", "PT")], "float64")
    track_eta = ak.values_astype(arrays[branch_name("EFlowTrack", "Eta")], "float64")
    track_phi = ak.values_astype(arrays[branch_name("EFlowTrack", "Phi")], "float64")
    track_in_acceptance = np.abs(track_eta) < TRACK_MAX_ABS_ETA
    track_pt = track_pt[track_in_acceptance]
    track_eta = track_eta[track_in_acceptance]
    track_phi = track_phi[track_in_acceptance]
    jet_pt_mask = pt >= MIN_JET_PT
    pt, eta, phi, mass = (
        pt[jet_pt_mask],
        eta[jet_pt_mask],
        phi[jet_pt_mask],
        mass[jet_pt_mask],
    )
    has_two_jets = ak.to_numpy(ak.num(pt) >= 2)
    event_indices = np.nonzero(has_two_jets)[0]
    if event_indices.size == 0:
        result = {
            "features": np.empty((0, len(FEATURE_NAMES) - 2)),
            "dijet_mass": np.empty(0),
            "dijet_rapidity": np.empty(0),
            "event_indices": np.empty(0, dtype=np.int64),
            "n_generated": n_generated,
            "n_two_jet": 0,
            "n_finite": 0,
        }
        if with_protons and proton_file is not None:
            result["xi_left_truth"] = np.empty(0)
            result["xi_right_truth"] = np.empty(0)
        elif with_protons:
            result["left_energy"] = np.empty(0)
            result["right_energy"] = np.empty(0)
        return result

    jets = ak.zip(
        {
            "pt": pt[has_two_jets],
            "eta": eta[has_two_jets],
            "phi": phi[has_two_jets],
            "mass": mass[has_two_jets],
        },
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    leading, subleading = jets[order][:, 0], jets[order][:, 1]
    dijet = leading + subleading
    deta = leading.eta - subleading.eta
    dphi = (leading.phi - subleading.phi + np.pi) % (2.0 * np.pi) - np.pi
    abs_deta, abs_dphi = np.abs(deta), np.abs(dphi)
    mjj = ak.to_numpy(dijet.mass)
    dijet_rapidity = ak.to_numpy(dijet.rapidity)
    selected_track_pt = track_pt[has_two_jets]
    selected_track_eta = track_eta[has_two_jets]
    selected_track_phi = track_phi[has_two_jets]
    track_dphi1 = (
        selected_track_phi - leading.phi[:, np.newaxis] + np.pi
    ) % (2.0 * np.pi) - np.pi
    track_dphi2 = (
        selected_track_phi - subleading.phi[:, np.newaxis] + np.pi
    ) % (2.0 * np.pi) - np.pi
    track_delta_r1 = np.hypot(
        selected_track_eta - leading.eta[:, np.newaxis], track_dphi1
    )
    track_delta_r2 = np.hypot(
        selected_track_eta - subleading.eta[:, np.newaxis], track_dphi2
    )
    nearest_track_delta_r = np.minimum(track_delta_r1, track_delta_r2)
    outer_track_mask = (
        np.isfinite(nearest_track_delta_r)
        & np.isfinite(selected_track_pt)
        & (nearest_track_delta_r > OUTER_TRACK_MIN_R)
        & (selected_track_pt >= OUTER_TRACK_MIN_PT)
    )
    outer_track_multiplicity = ak.to_numpy(ak.sum(outer_track_mask, axis=1))
    outer_track_sum_pt = ak.to_numpy(ak.sum(selected_track_pt[outer_track_mask], axis=1))

    def track_girth(track_delta_r):
        in_jet = (
            np.isfinite(track_delta_r)
            & np.isfinite(selected_track_pt)
            & (track_delta_r < IN_JET_TRACK_MAX_R)
            & (selected_track_pt >= IN_JET_TRACK_MIN_PT)
        )
        sum_pt = ak.to_numpy(ak.sum(selected_track_pt[in_jet], axis=1))
        weighted = ak.to_numpy(ak.sum((selected_track_pt * track_delta_r)[in_jet], axis=1))
        return np.divide(weighted, sum_pt, out=np.zeros_like(sum_pt), where=sum_pt > 0.0)

    jet1_girth = track_girth(track_delta_r1)
    jet2_girth = track_girth(track_delta_r2)
    base_features = np.column_stack(
        [
            ak.to_numpy(leading.pt) / mjj,
            ak.to_numpy(subleading.pt) / mjj,
            ak.to_numpy(leading.pt),
            ak.to_numpy(subleading.pt),
            ak.to_numpy(leading.eta),
            ak.to_numpy(subleading.eta),
            ak.to_numpy(np.sqrt(abs_deta * abs_deta + abs_dphi * abs_dphi)),
            ak.to_numpy(abs_dphi),
            ak.to_numpy(abs_deta),
            ak.to_numpy(dijet.pt),
            ak.to_numpy(dijet.eta),
            ak.to_numpy(ak.num(jets.pt)),
            outer_track_multiplicity,
            outer_track_sum_pt,
            jet1_girth,
            jet2_girth,
            mjj,
        ]
    )
    finite = (
        np.all(np.isfinite(base_features), axis=1)
        & np.isfinite(dijet_rapidity)
        & (mjj > 0.0)
    )
    result = {
        "features": base_features[finite],
        "dijet_mass": mjj[finite],
        "dijet_rapidity": dijet_rapidity[finite],
        "event_indices": event_indices[finite],
        "n_generated": n_generated,
        "n_two_jet": int(event_indices.size),
        "n_finite": int(np.sum(finite)),
    }

    if with_protons and proton_file is not None:
        xi_left, xi_right = parse_hepmc_proton_xi(
            np, proton_file, result["event_indices"], sqrt_s
        )
        result["xi_left_truth"] = xi_left
        result["xi_right_truth"] = xi_right
    elif with_protons:
        pid = arrays[branch_name("Particle", "PID")]
        status = arrays[branch_name("Particle", "Status")]
        pz = arrays[branch_name("Particle", "Pz")]
        energy = arrays[branch_name("Particle", "E")]
        protons = ak.zip({"pz": pz, "energy": energy})[(pid == 2212) & (status == 1)]
        left = protons[protons.pz < 0.0]
        right = protons[protons.pz > 0.0]
        left = ak.firsts(left[ak.argsort(abs(left.pz), axis=1, ascending=False)])
        right = ak.firsts(right[ak.argsort(abs(right.pz), axis=1, ascending=False)])
        result["left_energy"] = ak.to_numpy(ak.fill_none(left.energy, np.nan))[result["event_indices"]]
        result["right_energy"] = ak.to_numpy(ak.fill_none(right.energy, np.nan))[result["event_indices"]]
    return result


def load_file_central_worker(arguments):
    input_file, tree_name, collection, with_protons, proton_file, sqrt_s = arguments
    import awkward as ak
    import numpy as np
    import uproot
    import vector

    vector.register_awkward()
    return load_file_central(
        ak, np, uproot, input_file, tree_name, collection, with_protons,
        proton_file, sqrt_s,
    )


def load_sample_central(
    ak, np, uproot, input_files, tree_name, collection, with_protons,
    proton_files=None, sqrt_s=None,
):
    if proton_files is None:
        proton_files = [None] * len(input_files)
    tasks = [
        (input_file, tree_name, collection, with_protons, proton_file, sqrt_s)
        for input_file, proton_file in zip(input_files, proton_files)
    ]
    with ProcessPoolExecutor(max_workers=min(ROOT_READ_WORKERS, len(tasks))) as executor:
        pieces = list(executor.map(load_file_central_worker, tasks))

    generated_offset = 0
    totals = {"n_generated": 0, "n_two_jet": 0, "n_finite": 0}
    for piece in pieces:
        piece["global_event_indices"] = piece["event_indices"] + generated_offset
        generated_offset += piece["n_generated"]
        for name in totals:
            totals[name] += piece[name]

    def concatenate(name, dtype=np.float64):
        values = [piece[name] for piece in pieces if piece[name].size]
        return np.concatenate(values) if values else np.empty(0, dtype=dtype)

    result = {
        "features": concatenate("features").reshape(-1, len(FEATURE_NAMES) - 2),
        "dijet_mass": concatenate("dijet_mass"),
        "dijet_rapidity": concatenate("dijet_rapidity"),
        "global_event_indices": concatenate("global_event_indices", np.int64).astype(np.int64),
        **totals,
    }
    if with_protons:
        if proton_files[0] is not None:
            result["xi_left_truth"] = concatenate("xi_left_truth")
            result["xi_right_truth"] = concatenate("xi_right_truth")
        else:
            result["left_energy"] = concatenate("left_energy")
            result["right_energy"] = concatenate("right_energy")
    return result


def select_superchic_pairs(np, central, pps, rng):
    if "xi_left_truth" in central:
        xi_left_truth = central["xi_left_truth"]
        xi_right_truth = central["xi_right_truth"]
    else:
        beam_energy = pps["sqrt_s"] / 2.0
        xi_left_truth = (beam_energy - central["left_energy"]) / beam_energy
        xi_right_truth = (beam_energy - central["right_energy"]) / beam_energy
    valid_pair = (
        np.isfinite(xi_left_truth)
        & np.isfinite(xi_right_truth)
        & (xi_left_truth > 0.0)
        & (xi_right_truth > 0.0)
    )
    pps_mask = valid_pair & passes_pps(np, xi_left_truth, pps["xi_ranges"])
    pps_mask &= passes_pps(np, xi_right_truth, pps["xi_ranges"])
    xi_left, xi_right, mx, yx, reco_valid = smear_pair_observables(
        np, xi_left_truth, xi_right_truth, pps, rng
    )
    low, high = MASS_WINDOW_GEV
    selected = pps_mask & reco_valid & (mx >= low) & (mx <= high)
    return {
        "selected": selected,
        "n_pair": int(np.sum(valid_pair)),
        "n_pps": int(np.sum(pps_mask)),
        "xi_left": xi_left[selected],
        "xi_right": xi_right[selected],
        "mx": mx[selected],
        "yx": yx[selected],
        "bx_id": np.full(int(np.sum(selected)), -1, dtype=np.int64),
        "proton_source": np.full(int(np.sum(selected)), "superchic"),
    }


def smear_nested_xi(ak, np, protons, xi_res, rng):
    counts = ak.to_numpy(ak.num(protons, axis=1))
    flat = ak.to_numpy(ak.flatten(protons.xi))
    if xi_res > 0.0:
        flat = rng.normal(flat, xi_res)
    return ak.unflatten(flat, counts)


def build_minbias_pair_pool(ak, np, bunch_crossings, pps, rng, chunk_size=50000):
    if len(bunch_crossings) == 0:
        raise RuntimeError("Min-bias input contains no bunch crossings")
    chunks = []
    has_pair_chunks = []
    low, high = MASS_WINDOW_GEV
    for start in range(0, len(bunch_crossings), chunk_size):
        chunk = bunch_crossings[start : start + chunk_size]
        accepted = passes_pps(np, ak.to_numpy(ak.flatten(chunk.protons.xi)), pps["xi_ranges"])
        accepted = ak.unflatten(accepted, ak.to_numpy(ak.num(chunk.protons, axis=1)))
        protons = chunk.protons[accepted]
        xi_reco = smear_nested_xi(ak, np, protons, pps["xi_res"], rng)
        reco = ak.zip({"side": protons.side, "xi": xi_reco})
        left, right = reco[reco.side < 0], reco[reco.side > 0]
        pairs = ak.cartesian({"left": left, "right": right}, axis=1)
        valid = (pairs.left.xi > 0.0) & (pairs.right.xi > 0.0)
        product = ak.where(valid, pairs.left.xi * pairs.right.xi, 1.0)
        ratio = ak.where(valid, pairs.right.xi / pairs.left.xi, 1.0)
        mx = np.sqrt(product) * pps["sqrt_s"]
        yx = 0.5 * np.log(ratio)
        in_window = valid & (mx >= low) & (mx <= high)
        qualifying_pairs = ak.zip(
            {
                "xi_left": pairs.left.xi[in_window],
                "xi_right": pairs.right.xi[in_window],
                "mx": mx[in_window],
                "yx": yx[in_window],
            }
        )
        chunks.append(qualifying_pairs)
        has_pair_chunks.append(ak.to_numpy(ak.num(qualifying_pairs, axis=1) > 0))
    return ak.concatenate(chunks, axis=0), np.concatenate(has_pair_chunks)


def choose_minbias_pairs(ak, np, pair_pool, bx_ids, rng):
    event_pairs = pair_pool[bx_ids]
    counts = ak.to_numpy(ak.num(event_pairs, axis=1))
    selected = counts > 0
    nonempty = event_pairs[selected]
    choices = np.floor(rng.random(int(np.sum(selected))) * counts[selected]).astype(np.int64)
    chosen = ak.firsts(nonempty[ak.local_index(nonempty, axis=1) == choices])
    return {
        "selected": selected,
        "xi_left": ak.to_numpy(chosen.xi_left),
        "xi_right": ak.to_numpy(chosen.xi_right),
        "mx": ak.to_numpy(chosen.mx),
        "yx": ak.to_numpy(chosen.yx),
        "bx_id": bx_ids[selected],
        "proton_source": np.full(int(np.sum(selected)), "minbias_bx"),
    }


def assign_qualifying_minbias_pairs(
    ak, np, pair_pool, has_pair, n_generated, event_indices, rng
):
    qualifying_bx = np.flatnonzero(has_pair)
    if qualifying_bx.size == 0:
        raise RuntimeError("No min-bias BX has a PPS pair in the mass window")
    assignments = shuffled_bx_assignments(
        np, n_generated, qualifying_bx.size, rng
    )
    bx_ids = qualifying_bx[assignments[event_indices]]
    pair_data = choose_minbias_pairs(ak, np, pair_pool, bx_ids, rng)
    if not np.all(pair_data["selected"]):
        raise RuntimeError("A preselected min-bias BX has no qualifying proton pair")
    return pair_data, qualifying_bx.size / len(pair_pool)


def append_proton_features(np, central, pair_data):
    selected = pair_data["selected"]
    return np.column_stack(
        [
            central["features"][selected],
            pair_data["mx"] - central["dijet_mass"][selected],
            pair_data["yx"] - central["dijet_rapidity"][selected],
        ]
    )


def apply_pre_mva_cuts(np, central, pair_data):
    mass_selected = pair_data["selected"]
    delta_phi = central["features"][
        mass_selected, BASE_FEATURE_NAMES.index("delta_phi_jj")
    ]
    track_multiplicity = central["features"][
        mass_selected,
        BASE_FEATURE_NAMES.index("track_multiplicity_r_gt_0p4_pt1"),
    ]
    dijet_mass = central["dijet_mass"][mass_selected]
    rapidity_difference = (
        pair_data["yx"] - central["dijet_rapidity"][mass_selected]
    )
    passes_delta_phi = delta_phi > MIN_DIJET_DELTA_PHI
    passes_rapidity = (
        passes_delta_phi
        & (np.abs(rapidity_difference) < MAX_ABS_RAPIDITY_DIFFERENCE)
    )
    if MAX_OUTER_TRACK_MULTIPLICITY:
        passes_track_multiplicity = (
            passes_rapidity & (track_multiplicity < MAX_OUTER_TRACK_MULTIPLICITY)
        )
    else:
        passes_track_multiplicity = passes_rapidity
    dijet_mass_low, dijet_mass_high = PRE_MVA_DIJET_MASS_RANGE_GEV
    keep = (
        passes_track_multiplicity
        & (dijet_mass >= dijet_mass_low)
        & (dijet_mass <= dijet_mass_high)
    )

    selected = np.zeros_like(mass_selected)
    selected[np.flatnonzero(mass_selected)[keep]] = True
    filtered = dict(pair_data)
    filtered["selected"] = selected
    for name in ("xi_left", "xi_right", "mx", "yx", "bx_id", "proton_source"):
        filtered[name] = pair_data[name][keep]
    return filtered, {
        "n_mass_window": int(np.sum(mass_selected)),
        "n_delta_phi": int(np.sum(passes_delta_phi)),
        "n_rapidity": int(np.sum(passes_rapidity)),
        "n_track_multiplicity": int(np.sum(passes_track_multiplicity)),
        "n_dijet_mass": int(np.sum(keep)),
        "n_pre_mva": int(np.sum(keep)),
    }


def tag_weight(parameters):
    return float((parameters.get("tagging") or {})["eff_b"]) ** 2


def load_minbias_inputs(ak, np, campaign_arg, pps, rng):
    campaign_dir, campaign = resolve_minbias_campaign(campaign_arg)
    input_file = campaign_dir / "bx" / "bunch_crossings.parquet"
    if not input_file.is_file():
        raise RuntimeError(
            f"Missing min-bias BX input: {input_file}; run analysis/minbias_analyzer.py first"
        )
    print(f"Reading min-bias BX proton pairs: {input_file}", flush=True)
    bunch_crossings = ak.from_parquet(input_file)
    pair_pool, has_pair = build_minbias_pair_pool(ak, np, bunch_crossings, pps, rng)
    print(
        "Preselected min-bias BXs with a PPS pair in the mass window: "
        f"{int(np.sum(has_pair))}/{len(has_pair)} "
        f"({np.mean(has_pair):.6g})",
        flush=True,
    )
    return campaign, input_file, bunch_crossings, pair_pool, has_pair


def read_samples(ak, np, uproot, parameters, pps, args):
    samples, skipped = [], []
    seed_sequence = np.random.SeedSequence(args.seed)
    sample_seeds = seed_sequence.spawn(len(SAMPLE_SPECS) + 1)
    minbias = None
    track_cut_text = (
        f"track_multiplicity_r_gt_0p4_pt1 < {MAX_OUTER_TRACK_MULTIPLICITY}"
        if MAX_OUTER_TRACK_MULTIPLICITY
        else "no track-multiplicity cut"
    )
    print(
        f"Pre-MVA cuts: delta_phi_jj > {MIN_DIJET_DELTA_PHI:g}, "
        f"abs(yx_minus_dijet_rapidity) < {MAX_ABS_RAPIDITY_DIFFERENCE:g}, "
        f"{track_cut_text}, "
        f"{PRE_MVA_DIJET_MASS_RANGE_GEV[0]:g} <= dijet_mass <= "
        f"{PRE_MVA_DIJET_MASS_RANGE_GEV[1]:g} GeV",
        flush=True,
    )
    for spec_index, spec in enumerate(SAMPLE_SPECS):
        if spec["name"] not in ENABLED_SAMPLES:
            print(f"Skipping disabled sample: {spec['name']}", flush=True)
            continue
        campaign, subcampaign, input_dir, input_files, proton_files = resolve_sample_inputs(
            spec, args.campaign
        )
        if not input_files:
            skipped.append({"process": spec["name"], "input": str(input_dir), "reason": "missing ROOT files"})
            continue
        print(f"Reading {spec['name']} from {len(input_files)} ROOT file(s): {input_dir}", flush=True)
        try:
            central = load_sample_central(
                ak,
                np,
                uproot,
                input_files,
                args.tree,
                args.collection,
                with_protons=spec["generator"] == "superchic",
                proton_files=proton_files,
                sqrt_s=pps["sqrt_s"],
            )
        except RuntimeError as exc:
            skipped.append({"process": spec["name"], "input": str(input_dir), "reason": str(exc)})
            continue
        if central["n_generated"] <= 0:
            raise RuntimeError(f"{input_dir} has zero generated events")

        rng = np.random.default_rng(sample_seeds[spec_index])
        n_draws = int(spec.get("pair_resampling", 1))
        if spec["generator"] == "superchic":
            minbias_campaign = None
            minbias_input = None
        else:
            if minbias is None:
                minbias_rng = np.random.default_rng(sample_seeds[-1])
                minbias = load_minbias_inputs(ak, np, args.minbias_campaign, pps, minbias_rng)
            minbias_campaign, minbias_input, bunch_crossings, pair_pool, has_pair = minbias

        # Filter each draw immediately: unfiltered pair arrays span every event.
        filtered_draws = []
        pre_mva_cutflow = None
        for _draw in range(n_draws):
            if spec["generator"] == "superchic":
                pair_data = select_superchic_pairs(np, central, pps, rng)
                n_pair = pair_data["n_pair"]
                n_pps = pair_data["n_pps"]
                bx_pair_acceptance = 1.0
            else:
                pair_data, bx_pair_acceptance = assign_qualifying_minbias_pairs(
                    ak,
                    np,
                    pair_pool,
                    has_pair,
                    central["n_generated"],
                    central["global_event_indices"],
                    rng,
                )
                n_pair = central["n_finite"]
                n_pps = n_pair
            pair_data, cutflow = apply_pre_mva_cuts(np, central, pair_data)
            filtered_draws.append(pair_data)
            if pre_mva_cutflow is None:
                pre_mva_cutflow = cutflow
            else:
                for name in pre_mva_cutflow:
                    pre_mva_cutflow[name] += cutflow[name]
        features = np.vstack(
            [append_proton_features(np, central, pair_data) for pair_data in filtered_draws]
        )
        if features.shape[0] == 0:
            skipped.append({
                "process": spec["name"],
                "input": str(input_dir),
                "reason": (
                    "no events after pre-MVA cuts "
                    f"(mass_window={pre_mva_cutflow['n_mass_window']}, "
                    f"delta_phi={pre_mva_cutflow['n_delta_phi']}, "
                    f"rapidity={pre_mva_cutflow['n_rapidity']}, "
                    f"track_multiplicity={pre_mva_cutflow['n_track_multiplicity']}, "
                    f"dijet_mass={pre_mva_cutflow['n_dijet_mass']}, pre_mva=0)"
                ),
            })
            continue

        def gather_pairs(name):
            return np.concatenate([draw[name] for draw in filtered_draws])

        def gather_central(values):
            return np.concatenate([values[draw["selected"]] for draw in filtered_draws])

        xsec_fb, xsec_source = generator_cross_section_fb(
            spec["generator"], spec["process"], campaign
        )
        process_weight = generator_weight(spec["generator"], spec["process"])
        acceptance_factor = float(spec.get("acceptance_factor", 1.0))
        tag = tag_weight(parameters)
        event_weight = (
            xsec_fb
            * LUMI_FB
            * process_weight
            * acceptance_factor
            * bx_pair_acceptance
            * tag
            / (central["n_generated"] * n_draws)
        )
        expected_yield = event_weight * features.shape[0]
        technical_efficiency = features.shape[0] / (central["n_generated"] * n_draws)
        efficiency = technical_efficiency * bx_pair_acceptance
        label = 1 if spec["role"] == "signal" else 0
        sample = {
            "name": spec["name"],
            "generator": spec["generator"],
            "process": spec["process"],
            "campaign": campaign,
            "subcampaign": subcampaign,
            "input_dir": str(input_dir),
            "n_input_files": len(input_files),
            "features": features,
            "label": label,
            "event_weight": event_weight,
            "expected_yield": expected_yield,
            "xsec_fb": xsec_fb,
            "xsec_source": xsec_source,
            "process_weight": process_weight,
            "acceptance_factor": acceptance_factor,
            "bx_pair_acceptance": bx_pair_acceptance,
            "tag_weight": tag,
            "n_generated": central["n_generated"],
            "n_two_jet": central["n_two_jet"],
            "n_finite": central["n_finite"],
            "n_pair": n_pair,
            "n_pps": n_pps,
            "n_mass_window": pre_mva_cutflow["n_mass_window"],
            "n_delta_phi": pre_mva_cutflow["n_delta_phi"],
            "n_rapidity": pre_mva_cutflow["n_rapidity"],
            "n_track_multiplicity": pre_mva_cutflow["n_track_multiplicity"],
            "n_dijet_mass": pre_mva_cutflow["n_dijet_mass"],
            "n_pre_mva": pre_mva_cutflow["n_pre_mva"],
            "technical_selection_efficiency": technical_efficiency,
            "selection_efficiency": efficiency,
            "pair_resampling": n_draws,
            "central_event_index": gather_central(central["global_event_indices"]),
            "xi_left": gather_pairs("xi_left"),
            "xi_right": gather_pairs("xi_right"),
            "mx": gather_pairs("mx"),
            "yx": gather_pairs("yx"),
            "dijet_mass": gather_central(central["dijet_mass"]),
            "dijet_rapidity": gather_central(central["dijet_rapidity"]),
            "delta_phi_jj": gather_central(
                central["features"][:, BASE_FEATURE_NAMES.index("delta_phi_jj")]
            ),
            "bx_id": gather_pairs("bx_id"),
            "proton_source": gather_pairs("proton_source"),
            "minbias_campaign": minbias_campaign,
            "minbias_input": str(minbias_input) if minbias_input else None,
        }
        samples.append(sample)
        print(
            f"{spec['name']}_{campaign}/{subcampaign}: role={spec['role']}, "
            f"generated={central['n_generated']}, two_jet={central['n_two_jet']}, "
            f"finite={central['n_finite']}, pair={n_pair}, pps={n_pps}, "
            f"mass_window={pre_mva_cutflow['n_mass_window']}, "
            f"delta_phi_gt_{MIN_DIJET_DELTA_PHI:g}={pre_mva_cutflow['n_delta_phi']}, "
            f"abs_rapidity_diff_lt_{MAX_ABS_RAPIDITY_DIFFERENCE:g}="
            f"{pre_mva_cutflow['n_rapidity']}, "
            + (
                f"track_multiplicity_lt_{MAX_OUTER_TRACK_MULTIPLICITY}="
                if MAX_OUTER_TRACK_MULTIPLICITY
                else "track_multiplicity_nocut="
            )
            + f"{pre_mva_cutflow['n_track_multiplicity']}, "
            f"dijet_mass_{PRE_MVA_DIJET_MASS_RANGE_GEV[0]:g}_to_"
            f"{PRE_MVA_DIJET_MASS_RANGE_GEV[1]:g}={pre_mva_cutflow['n_dijet_mass']}, "
            f"pair_resampling={n_draws}, "
            f"technical_efficiency={technical_efficiency:.6g}, "
            f"bx_pair_acceptance={bx_pair_acceptance:.6g}, "
            f"effective_efficiency={efficiency:.6g}, "
            f"acceptance_factor={acceptance_factor:.6g}, tag_weight={tag:.6g}, "
            f"event_weight={event_weight:.6g}, "
            f"expected_yield={expected_yield:.6g}"
        )

    for item in skipped:
        print(f"Warning: skipping {item['process']}: {item['reason']} ({item['input']})")
    if not samples:
        raise RuntimeError("No usable PPS-selected Delphes inputs found")
    if not any(sample["label"] == 1 for sample in samples):
        raise RuntimeError("No usable signal sample found")
    if not any(sample["label"] == 0 for sample in samples):
        raise RuntimeError("No usable background samples found")
    return samples, skipped


def build_dataset(np, samples):
    def concatenate(name):
        return np.concatenate([sample[name] for sample in samples])

    return {
        "x": concatenate("features"),
        "y": np.concatenate(
            [np.full(sample["features"].shape[0], sample["label"], dtype=np.int8) for sample in samples]
        ),
        "physical_weight": np.concatenate(
            [np.full(sample["features"].shape[0], sample["event_weight"]) for sample in samples]
        ),
        "process": np.concatenate(
            [np.full(sample["features"].shape[0], sample["name"]) for sample in samples]
        ),
        "campaign": np.concatenate(
            [np.full(sample["features"].shape[0], sample["campaign"]) for sample in samples]
        ),
        "group": np.concatenate(
            [
                np.char.add(
                    f"{sample['name']}:", sample["central_event_index"].astype(str)
                )
                for sample in samples
            ]
        ),
        "xi_left": concatenate("xi_left"),
        "xi_right": concatenate("xi_right"),
        "mx": concatenate("mx"),
        "yx": concatenate("yx"),
        "dijet_mass": concatenate("dijet_mass"),
        "dijet_rapidity": concatenate("dijet_rapidity"),
        "delta_phi_jj": concatenate("delta_phi_jj"),
        "bx_id": concatenate("bx_id"),
        "proton_source": concatenate("proton_source"),
    }


def select_training_features(np, dataset, source_feature_names, feature_set):
    """Keep every stored feature in the dataset; train only on the non-excluded ones."""
    if feature_set not in FEATURE_SETS:
        raise RuntimeError(f"Unknown feature set: {feature_set}")
    names = [str(name) for name in source_feature_names]
    excluded = [name for name in FEATURE_SETS[feature_set] if name in names]
    keep = [index for index, name in enumerate(names) if name not in excluded]
    if not keep:
        raise RuntimeError(f"Feature set {feature_set} excludes every feature")
    dataset["feature_names"] = np.asarray(names)
    dataset["training_feature_names"] = np.asarray([names[index] for index in keep])
    print(
        f"Feature set {feature_set}: training on {len(keep)}/{len(names)} stored features"
        + (f", excluding: {', '.join(excluded)}" if excluded else ""),
        flush=True,
    )
    return np.asarray(dataset["x"])[:, keep]


def load_cached_selection(np, path):
    dataset_path = resolve_path(path, base=ROOT)
    if dataset_path.is_dir():
        dataset_path = dataset_path / "dataset.npz"
    if not dataset_path.is_file():
        raise RuntimeError(f"Missing source selection dataset: {dataset_path}")
    source = np.load(dataset_path, allow_pickle=False)
    if "feature_names" not in source.files:
        raise RuntimeError(f"{dataset_path} does not contain feature_names")
    dataset = {name: source[name] for name in source.files if name != "feature_names"}
    summary_path = dataset_path.with_name("summary.yaml")
    return dataset, source["feature_names"], dataset_path, summary_path


def balanced_weights(np, labels, physical_weights):
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


def derive_groups(np, dataset):
    """Group id per row: all pair-resampled copies of one central event share a group."""
    if "group" in dataset:
        _names, groups = np.unique(dataset["group"], return_inverse=True)
        return groups
    # Older cached datasets lack group ids: copies of one central event share
    # bit-identical central-only feature columns, so recover groups from those.
    names = [str(name) for name in dataset["feature_names"]]
    central = [
        index
        for index, name in enumerate(names)
        if name not in ("mx_minus_dijet_mass", "yx_minus_dijet_rapidity")
    ]
    columns = np.ascontiguousarray(dataset["x"][:, central])
    rows = columns.view([("", columns.dtype)] * len(central)).ravel()
    _values, groups = np.unique(rows, return_inverse=True)
    return groups


def cv_folds(np, StratifiedGroupKFold, dataset, groups, seed):
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return {
        f"fold_{index}": test_idx
        for index, (_train_idx, test_idx) in enumerate(
            splitter.split(dataset["x"], dataset["y"], groups)
        )
    }


def fit_model(np, XGBClassifier, train_test_split, x, y, w, seed):
    fit_idx, stop_idx = train_test_split(
        np.arange(y.shape[0]), train_size=0.9, random_state=seed, stratify=y
    )
    model = XGBClassifier(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=20,
    )
    model.fit(
        x[fit_idx],
        y[fit_idx],
        sample_weight=balanced_weights(np, y[fit_idx], w[fit_idx]),
        eval_set=[(x[stop_idx], y[stop_idx])],
        sample_weight_eval_set=[balanced_weights(np, y[stop_idx], w[stop_idx])],
        verbose=False,
    )
    return model


def train_oof(np, XGBClassifier, train_test_split, x, dataset, folds, args):
    """K-fold CV: every event is scored by the model that never trained on its group."""
    y, w = dataset["y"], dataset["physical_weight"]
    oof_scores = np.full(y.shape[0], np.nan)
    best_iterations = []
    for name in sorted(folds):
        test_idx = folds[name]
        train_mask = np.ones(y.shape[0], dtype=bool)
        train_mask[test_idx] = False
        model = fit_model(
            np, XGBClassifier, train_test_split,
            x[train_mask], y[train_mask], w[train_mask], args.seed,
        )
        oof_scores[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        best_iterations.append(int(model.best_iteration))
        print(
            f"{name}: train={int(np.sum(train_mask))} test={test_idx.size} "
            f"best_iteration={model.best_iteration}",
            flush=True,
        )
    if np.any(np.isnan(oof_scores)):
        raise RuntimeError("Cross-validation left events without an out-of-fold score")
    final_model = fit_model(np, XGBClassifier, train_test_split, x, y, w, args.seed)
    return oof_scores, best_iterations, final_model


def split_summary(np, dataset, splits):
    summary = {}
    for name, indices in splits.items():
        labels, weights = dataset["y"][indices], dataset["physical_weight"][indices]
        summary[name] = {
            "events": int(indices.size),
            "signal_events": int(np.sum(labels == 1)),
            "background_events": int(np.sum(labels == 0)),
            "signal_expected_yield": float(np.sum(weights[labels == 1])),
            "background_expected_yield": float(np.sum(weights[labels == 0])),
        }
    return summary


def sample_summary(samples):
    fields = (
        "generator", "process", "campaign", "subcampaign", "input_dir", "n_input_files",
        "n_generated", "n_two_jet", "n_finite", "n_pair", "n_pps", "n_mass_window",
        "n_delta_phi", "n_rapidity", "n_track_multiplicity", "n_dijet_mass", "n_pre_mva",
        "pair_resampling", "technical_selection_efficiency", "selection_efficiency",
        "xsec_fb", "xsec_source", "process_weight", "acceptance_factor",
        "bx_pair_acceptance", "tag_weight", "event_weight", "expected_yield",
        "minbias_campaign", "minbias_input",
    )
    return [
        {"process": sample["name"], "role": "signal" if sample["label"] else "background"}
        | {name: sample[name] for name in fields}
        for sample in samples
    ]


def save_artifacts(np, output_dir, dataset, splits, scores, model):
    paths = {
        "dataset": output_dir / "dataset.npz",
        "splits": output_dir / "splits.npz",
        "scores": output_dir / "scores.npz",
        "model": output_dir / "model.json",
    }
    np.savez_compressed(paths["dataset"], **dataset)
    np.savez_compressed(paths["splits"], **splits)
    np.savez_compressed(paths["scores"], **scores)
    model._estimator_type = "classifier"
    model.save_model(paths["model"])
    return paths


def write_summary(
    yaml,
    output_path,
    args,
    pps_path,
    pps,
    samples,
    skipped,
    dataset,
    splits,
    roc_auc,
    outputs,
    model,
    best_iterations,
    np,
    selection_source=None,
):
    source_summary = selection_source or {}
    summary = {
        "flavor": args.flavor,
        "feature_set": args.feature_set,
        "selection": args.selection,
        "tree": args.tree,
        "collection": args.collection,
        "min_jet_pt_gev": MIN_JET_PT,
        "track_max_abs_eta": TRACK_MAX_ABS_ETA,
        "outer_track_feature": {
            "names": ["track_multiplicity_r_gt_0p4_pt1", "outer_track_sum_pt"],
            "min_delta_r_exclusive": OUTER_TRACK_MIN_R,
            "min_track_pt_gev_inclusive": OUTER_TRACK_MIN_PT,
        },
        "in_jet_girth_feature": {
            "names": ["jet1_girth", "jet2_girth"],
            "max_delta_r_exclusive": IN_JET_TRACK_MAX_R,
            "min_track_pt_gev_inclusive": IN_JET_TRACK_MIN_PT,
        },
        "mass_window_gev": list(MASS_WINDOW_GEV),
        "pre_mva_cuts": {
            "delta_phi_jj_min_exclusive": MIN_DIJET_DELTA_PHI,
            "abs_yx_minus_dijet_rapidity_max_exclusive": MAX_ABS_RAPIDITY_DIFFERENCE,
            "track_multiplicity_r_gt_0p4_pt1_max_exclusive": MAX_OUTER_TRACK_MULTIPLICITY,
            "dijet_mass_gev_inclusive": list(PRE_MVA_DIJET_MASS_RANGE_GEV),
        },
        "pps_config": str(pps_path),
        "sqrt_s_gev": pps["sqrt_s"],
        "xi_resolution": pps["xi_res"],
        "seed": args.seed,
        "luminosity_fb": LUMI_FB,
        "enabled_samples": list(ENABLED_SAMPLES),
        "features": [str(name) for name in dataset["training_feature_names"]],
        "stored_features": [str(name) for name in dataset["feature_names"]],
        "excluded_features": list(FEATURE_SETS[args.feature_set]),
        "samples": sample_summary(samples) if samples is not None else source_summary.get("samples", []),
        "skipped": skipped if skipped is not None else source_summary.get("skipped", []),
        "selection_dataset": str(args.dataset_from) if args.dataset_from else None,
        "splits": split_summary(np, dataset, splits),
        "auc_oof_class_balanced": float(roc_auc),
        "model": {
            "type": "XGBClassifier",
            "n_estimators": 5000,
            "cv_folds": CV_FOLDS,
            "cv_best_iterations": [int(value) for value in best_iterations],
            "final_best_iteration": int(getattr(model, "best_iteration", -1)),
            "max_depth": 6,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def run_plot_script(output_dir):
    script = Path(__file__).resolve().with_name("plot_dijet_mva.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--input-dir", str(output_dir)], cwd=ROOT, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Plotting failed with exit code {completed.returncode}")


def main():
    args = parse_args()
    ensure_delphes_python_runtime()
    print("Importing analysis libraries", flush=True)
    (
        ak, np, uproot, yaml, XGBClassifier, StratifiedGroupKFold,
        train_test_split, roc_curve, auc,
    ) = import_libraries()
    parameters = load_yaml(ROOT / "parameters.yaml")
    pps_path = resolve_path(args.pps_config, base=ROOT)
    pps = load_pps_config(pps_path)
    output_dir = resolve_path(args.output_dir, base=ROOT) if args.output_dir else ROOT / "analysis/MVA/output/mva_bb"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_source = None
    if args.dataset_from:
        dataset, source_feature_names, dataset_path, summary_path = load_cached_selection(
            np, args.dataset_from
        )
        if summary_path.is_file():
            selection_source = load_yaml(summary_path)
        samples, skipped = None, None
        print(f"Reusing PPS-selected events from {dataset_path}", flush=True)
    else:
        samples, skipped = read_samples(ak, np, uproot, parameters, pps, args)
        dataset = build_dataset(np, samples)
        source_feature_names = BASE_FEATURE_NAMES
    train_x = select_training_features(
        np, dataset, source_feature_names, args.feature_set
    )
    groups = derive_groups(np, dataset)
    splits = cv_folds(np, StratifiedGroupKFold, dataset, groups, args.seed)
    print(
        f"Training XGBoost model ({CV_FOLDS}-fold group-aware CV over "
        f"{int(groups.max()) + 1} central-event groups)",
        flush=True,
    )
    oof_scores, best_iterations, model = train_oof(
        np, XGBClassifier, train_test_split, train_x, dataset, splits, args
    )

    scores = {"all": oof_scores, "train_final": model.predict_proba(train_x)[:, 1]}
    fpr, tpr, _ = roc_curve(
        dataset["y"],
        scores["all"],
        sample_weight=balanced_weights(np, dataset["y"], dataset["physical_weight"]),
    )
    scores["fpr"], scores["tpr"], scores["roc_auc"] = fpr, tpr, np.asarray(auc(fpr, tpr))

    outputs = {
        "dataset": output_dir / "dataset.npz",
        "splits": output_dir / "splits.npz",
        "scores": output_dir / "scores.npz",
        "model": output_dir / "model.json",
        "summary": output_dir / "summary.yaml",
        "oof_score": output_dir / "oof_score.png",
        "oof_score_logit": output_dir / "oof_score_logit.png",
        "roc": output_dir / "roc.png",
        "feature_importance": output_dir / "feature_importance.png",
        "tmva_score": output_dir / "tmva_score.png",
        "tmva_score_log": output_dir / "tmva_score_log.png",
        "tmva_score_logit": output_dir / "tmva_score_logit.png",
        "tmva_score_logit_log": output_dir / "tmva_score_logit_log.png",
    }
    paths = save_artifacts(np, output_dir, dataset, splits, scores, model)
    write_summary(
        yaml, outputs["summary"], args, pps_path, pps, samples, skipped,
        dataset, splits, float(scores["roc_auc"]), outputs, model, best_iterations, np,
        selection_source=selection_source,
    )
    print(f"Out-of-fold class-balanced AUC: {float(scores['roc_auc']):.6g}")
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    print(f"Wrote summary: {outputs['summary']}")
    if args.skip_plots:
        print("Skipped plot generation")
    else:
        sys.stdout.flush()
        run_plot_script(output_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
