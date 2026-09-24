#!/usr/bin/env python3
"""Build the reduced five-class H(cc) central-plus-proton MVA dataset."""
import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml
from uproot.source.file import MemmapSource


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from analysis.MVA import run_dijet_mva as pps_lib  # noqa: E402
from analysis.MVA import run_dijet_mva_multiclass_protons as source  # noqa: E402
from analysis.MVA_hcc.common import (  # noqa: E402
    CLASS_NAMES,
    COMPONENT_SPECS,
    LOCKED_FEATURES,
    component_manifest,
    plain,
    read_parameters,
    stitched_cross_section_weights,
    tag_factor,
    write_yaml,
)
from analysis.cross_sections import (  # noqa: E402
    event_record_files,
    generator_cross_section_fb,
    generator_weight,
)
from common.config_utils import natural_key, resolve_path  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_process_config,
    generation_stage_root,
)


DEFAULT_DATA = SCRIPT_DIR / "data"
DEFAULT_PPS_CONFIG = "analysis/scripts/new/config.yaml"
SC_MAX_FILES = 100
ROOT_WORKERS = 8
LUMI_FB = pps_lib.LUMI_FB
MASS_WINDOW_GEV = pps_lib.MASS_WINDOW_GEV
MAX_ABS_RAPIDITY_DIFFERENCE = pps_lib.MAX_ABS_RAPIDITY_DIFFERENCE
FULL_FEATURES = tuple(source.FEATURE_NAMES)
# Rebound by main() before any worker pool starts, so forked workers inherit them.
SELECTED_FEATURES = LOCKED_FEATURES
FEATURE_INDICES = np.asarray([FULL_FEATURES.index(name) for name in LOCKED_FEATURES])
ACTIVE_SPECS = COMPONENT_SPECS
ACTIVE_CLASSES = CLASS_NAMES
# Parton-level kinematics of the hard pair, stored only with --store-parton.
PARTON_FIELDS = ("pt1", "pt2", "eta1", "eta2", "mass")
STORE_PARTON = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--tree", default="Delphes")
    parser.add_argument("--collection", default="JetPUPPI")
    parser.add_argument("--pps-config", default=DEFAULT_PPS_CONFIG)
    parser.add_argument("--minbias-campaign", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=ROOT_WORKERS)
    parser.add_argument("--evaluation-pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--feature-set", choices=("locked", "full"), default="locked",
        help="locked: the 20 LOCKED_FEATURES. full: every column of FULL_FEATURES.",
    )
    parser.add_argument(
        "--store-parton", action="store_true",
        help="Also store the parton-level hard pair (pt1, pt2, eta1, eta2, mass) "
             "for MadGraph components, NaN elsewhere. Off by default.",
    )
    parser.add_argument(
        "--extra-campaigns", nargs="+", default=None, metavar="NAME=CAMP[,CAMP]",
        help="Override the campaign list of a MadGraph component, e.g. "
             "QCDbb_madgraph=QCDbb__v01,QCDbb__v02,QCDbb__v03.",
    )
    parser.add_argument(
        "--components", nargs="+", default=None,
        help="Physical components to build. Defaults to every COMPONENT_SPECS entry. "
        "A restricted set is renumbered contiguously and keeps only the classes it uses.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.evaluation_pairs <= 0:
        raise ValueError("--evaluation-pairs must be positive")
    if args.components is not None:
        known = [spec["name"] for spec in COMPONENT_SPECS]
        unknown = [name for name in args.components if name not in known]
        if unknown:
            raise ValueError(f"Unknown components {unknown}; choose from {known}")
        if not args.components:
            raise ValueError("--components must name at least one component")
    for item in args.extra_campaigns or []:
        if "=" not in item:
            raise ValueError(f"--extra-campaigns entry {item!r} must be NAME=CAMP[,CAMP]")
        name, campaigns = item.split("=", 1)
        if name not in [spec["name"] for spec in COMPONENT_SPECS]:
            raise ValueError(f"--extra-campaigns names unknown component {name!r}")
        if not campaigns.strip():
            raise ValueError(f"--extra-campaigns entry {item!r} lists no campaigns")


def default_subcampaign(process_config, stage):
    defaults = process_config.get("default_campaign") or {}
    return defaults.get(stage) if isinstance(defaults, dict) else None


def resolve_superchic_files(spec, max_files):
    campaign, _ = generation_campaign_config("superchic", spec["process"], None)
    config = generation_process_config("superchic", spec["process"])
    subcampaign = default_subcampaign(config, "sim-delphes")
    input_dir = generation_stage_root(
        "superchic", spec["process"], campaign, "sim-delphes", subcampaign=subcampaign
    ) / "root"
    files = sorted(input_dir.glob("*.root"), key=natural_key)
    limit = max_files if max_files is not None else SC_MAX_FILES
    files = files[:limit]
    if not files:
        raise RuntimeError(f"No SuperChic ROOT files found for {spec['process']} in {input_dir}")

    pythia_subcampaign = default_subcampaign(config, "hadr-pythia")
    pythia_dir = generation_stage_root(
        "superchic",
        spec["process"],
        campaign,
        "hadr-pythia",
        subcampaign=pythia_subcampaign,
    ) / "hepmc"
    hepmc_by_stem = {
        path.stem: path
        for path in sorted(pythia_dir.glob("*.hepmc"), key=natural_key)
    }
    if all(path.stem in hepmc_by_stem for path in files):
        proton_kind = "hepmc"
        proton_files = [hepmc_by_stem[path.stem] for path in files]
    else:
        lhe_by_stem = {
            path.stem: path
            for path in event_record_files("superchic", spec["process"], campaign)
        }
        missing = [path.name for path in files if path.stem not in lhe_by_stem]
        if missing:
            raise RuntimeError(
                f"Missing matching SuperChic proton records for: {', '.join(missing)}"
            )
        proton_kind = "lhe"
        proton_files = [lhe_by_stem[path.stem] for path in files]
    return campaign, subcampaign, input_dir, files, proton_kind, proton_files


def madgraph_campaign_config(process, campaign):
    process_config = generation_process_config("madgraph", process)
    campaign_config = (process_config.get("campaigns") or {}).get(campaign)
    if campaign_config is None:
        raise RuntimeError(f"MadGraph {process} has no configured campaign {campaign}")
    phase_space = campaign_config.get("phase_space") or {}
    required = ("parton_pt_gev", "max_abs_eta", "dijet_mass_gev")
    missing = [name for name in required if name not in phase_space]
    if missing:
        raise RuntimeError(f"MadGraph {process}/{campaign} lacks: {', '.join(missing)}")
    subcampaign = campaign_config.get("mva_subcampaign")
    if not subcampaign:
        raise RuntimeError(f"MadGraph {process}/{campaign} has no mva_subcampaign")
    return {
        "subcampaign": subcampaign,
        "pt": tuple(phase_space["parton_pt_gev"]),
        "max_abs_eta": float(phase_space["max_abs_eta"]),
        "mass": tuple(phase_space["dijet_mass_gev"]),
    }


def resolve_madgraph_files(process, campaign, max_files):
    config = madgraph_campaign_config(process, campaign)
    input_dir = generation_stage_root(
        "madgraph", process, campaign, "sim-delphes", subcampaign=config["subcampaign"]
    ) / "root"
    files = sorted(input_dir.glob("*.root"), key=natural_key)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No MadGraph ROOT files found for {process}/{campaign} in {input_dir}")
    return config, input_dir, files


def particle_arrays(path, tree_name, fields):
    names = [source.branch_name("Particle", field) for field in fields]
    try:
        with uproot.open(path, handler=MemmapSource) as root_file:
            tree = root_file[tree_name]
            missing = [name for name in names if name not in tree.keys()]
            if missing:
                raise RuntimeError(f"Missing proton/truth branches: {', '.join(missing)}")
            return tree.arrays(names, library="ak")
    except Exception as exc:
        raise RuntimeError(f"Failed to read particle truth from {path}: {exc}") from exc


def extract_nonpileup_proton_energies(arrays, event_indices):
    pid = arrays[source.branch_name("Particle", "PID")]
    status = arrays[source.branch_name("Particle", "Status")]
    is_pu = arrays[source.branch_name("Particle", "IsPU")]
    pz = arrays[source.branch_name("Particle", "Pz")]
    energy = arrays[source.branch_name("Particle", "E")]
    mask = (abs(pid) == 2212) & (status == 1) & (is_pu == 0)
    protons = ak.zip({"pz": pz[mask], "energy": energy[mask]})
    left = protons[protons.pz < 0.0]
    right = protons[protons.pz > 0.0]
    left = ak.firsts(left[ak.argsort(abs(left.pz), axis=1, ascending=False)])
    right = ak.firsts(right[ak.argsort(abs(right.pz), axis=1, ascending=False)])
    left_energy = ak.to_numpy(ak.fill_none(left.energy, np.nan))[event_indices]
    right_energy = ak.to_numpy(ak.fill_none(right.energy, np.nan))[event_indices]
    return left_energy, right_energy


def parse_lhe_proton_xi(input_file, selected_event_indices, sqrt_s):
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

    with open(input_file, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("<event"):
                event_index += 1
                if event_index > last_selected:
                    break
                output_index = selected_positions.get(event_index)
                left_energy = right_energy = None
                left_abs_pz = right_abs_pz = -1.0
            elif stripped.startswith("</event"):
                store_event()
                output_index = None
            elif output_index is not None:
                fields = stripped.split()
                if len(fields) < 10:
                    continue
                try:
                    pid, status = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                if pid != 2212 or status != 1:
                    continue
                pz, energy = float(fields[8]), float(fields[9])
                abs_pz = abs(pz)
                if pz < 0.0 and abs_pz > left_abs_pz:
                    left_energy, left_abs_pz = energy, abs_pz
                elif pz > 0.0 and abs_pz > right_abs_pz:
                    right_energy, right_abs_pz = energy, abs_pz
    return xi_left, xi_right


def extract_hard_pair(arrays, event_indices, final_pdg):
    pid = arrays[source.branch_name("Particle", "PID")]
    status = arrays[source.branch_name("Particle", "Status")]
    is_pu = arrays[source.branch_name("Particle", "IsPU")]
    pt = arrays[source.branch_name("Particle", "PT")]
    eta = arrays[source.branch_name("Particle", "Eta")]
    phi = arrays[source.branch_name("Particle", "Phi")]
    mass = arrays[source.branch_name("Particle", "Mass")]
    mask = (abs(pid) == final_pdg) & (status == 23) & (is_pu == 0)
    selected = ak.zip(
        {"pt": pt[mask], "eta": eta[mask], "phi": phi[mask], "mass": mass[mask]}
    )[event_indices]
    valid = ak.to_numpy(ak.num(selected.pt) == 2)
    padded = ak.pad_none(selected, 2, clip=True)
    first, second = padded[:, 0], padded[:, 1]

    def values(field):
        return ak.to_numpy(ak.fill_none(field, np.nan))

    pt1, pt2 = values(first.pt), values(second.pt)
    eta1, eta2 = values(first.eta), values(second.eta)
    phi1, phi2 = values(first.phi), values(second.phi)
    mass1, mass2 = values(first.mass), values(second.mass)
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
    energy = np.sqrt((pt1 * np.cosh(eta1)) ** 2 + mass1**2)
    energy += np.sqrt((pt2 * np.cosh(eta2)) ** 2 + mass2**2)
    pair_mass = np.sqrt(np.clip(energy**2 - px**2 - py**2 - pz**2, 0.0, None))
    return {
        "valid": valid,
        "pt1": pt1,
        "pt2": pt2,
        "eta1": eta1,
        "eta2": eta2,
        "mass": pair_mass,
    }


def superchic_piece(arguments):
    path, proton_kind, proton_file, tree_name, collection, pps, seed = arguments
    central = source.central_features(path, tree_name, collection)
    if central.get("empty"):
        return {"n_generated": central["n_generated"], "empty": True}
    if proton_kind == "hepmc":
        xi_left, xi_right = pps_lib.parse_hepmc_proton_xi(
            np, proton_file, central["event_indices"], pps["sqrt_s"]
        )
    elif proton_kind == "lhe":
        xi_left, xi_right = parse_lhe_proton_xi(
            proton_file, central["event_indices"], pps["sqrt_s"]
        )
    else:
        raise RuntimeError(f"Unsupported SuperChic proton record type: {proton_kind}")
    passed, _left, _right, mx, yx = source.real_proton_pass(
        np, xi_left, xi_right, pps, np.random.default_rng(seed)
    )
    keep = (
        passed
        & (np.abs(yx - central["dijet_rapidity"]) < MAX_ABS_RAPIDITY_DIFFERENCE)
        & (mx >= MASS_WINDOW_GEV[0])
        & (mx <= MASS_WINDOW_GEV[1])
    )
    full = np.column_stack(
        [central["matrix"], yx - central["dijet_rapidity"]]
    ).astype(np.float32, copy=False)
    return {
        "features": full[keep][:, FEATURE_INDICES],
        "mx": mx[keep],
        "event_indices": central["event_indices"][keep],
        "n_generated": central["n_generated"],
        "n_central": central["matrix"].shape[0],
        "n_proton": int(np.sum(keep)),
    }


def madgraph_piece(arguments):
    path, tree_name, collection, final_pdg = arguments
    central = source.central_features(path, tree_name, collection)
    if central.get("empty"):
        return {"n_generated": central["n_generated"], "empty": True}
    arrays = particle_arrays(
        path, tree_name, ("PID", "Status", "IsPU", "PT", "Eta", "Phi", "Mass")
    )
    truth = extract_hard_pair(arrays, central["event_indices"], final_pdg)
    return {**central, "truth": truth}


def concatenate(pieces, name, dtype=np.float64):
    arrays = [piece[name] for piece in pieces if not piece.get("empty")]
    if not arrays:
        return np.empty(0, dtype=dtype)
    return np.concatenate(arrays)


def load_superchic_component(spec, component_id, seed_id, parameters, args, pps):
    campaign, subcampaign, input_dir, files, proton_kind, proton_files = (
        resolve_superchic_files(spec, args.max_files)
    )
    tasks = [
        (
            path,
            proton_kind,
            proton_file,
            args.tree,
            args.collection,
            pps,
            args.seed + seed_id * 10000 + index,
        )
        for index, (path, proton_file) in enumerate(zip(files, proton_files))
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        pieces = list(executor.map(superchic_piece, tasks))
    n_generated = sum(piece["n_generated"] for piece in pieces)
    features = [piece["features"] for piece in pieces if not piece.get("empty")]
    if not features:
        raise RuntimeError(f"{spec['name']} has no selected events")
    matrix = np.vstack(features)
    mx = concatenate(pieces, "mx")
    offsets = np.cumsum([0] + [piece["n_generated"] for piece in pieces[:-1]])
    groups = np.concatenate(
        [piece["event_indices"] + offset for piece, offset in zip(pieces, offsets) if not piece.get("empty")]
    ).astype(np.int64)
    xsec_fb, xsec_source = generator_cross_section_fb("superchic", spec["process"], campaign)
    process_scale = generator_weight("superchic", spec["process"])
    tag = tag_factor(parameters, spec["source_flavor"])
    physical = xsec_fb * process_scale * LUMI_FB * tag / n_generated
    mixture = xsec_fb * process_scale * tag / n_generated
    if physical <= 0.0 or mixture <= 0.0:
        raise RuntimeError(f"{spec['name']} has a non-positive event weight")
    rows = matrix.shape[0]
    print(
        f"{spec['name']}: files={len(files)} generated={n_generated:,} "
        f"selected={rows:,} yield={physical * rows:.6g}",
        flush=True,
    )
    return {
        "x": matrix,
        "class": np.full(rows, ACTIVE_CLASSES.index(spec["class_name"]), dtype=np.int8),
        "component": np.full(rows, component_id, dtype=np.int8),
        "group_id": groups,
        "mx": mx,
        "physical_weight": np.full(rows, physical, dtype=np.float64),
        "training_mixture_weight": np.full(rows, mixture, dtype=np.float64),
        "central_weight": np.zeros(rows, dtype=np.float64),
        "band_probability": np.full(rows, np.nan, dtype=np.float64),
        "parton": np.full((rows, len(PARTON_FIELDS)), np.nan, dtype=np.float64),
        "input": {
            "campaign": campaign,
            "subcampaign": subcampaign,
            "input_dir": str(input_dir),
            "proton_record_type": proton_kind,
            "files": len(files),
            "generated": n_generated,
            "central_selected": sum(piece.get("n_central", 0) for piece in pieces),
            "proton_selected": rows,
            "xsec_fb": xsec_fb,
            "xsec_source": xsec_source,
        },
    }


def phase_space_mask(truth, config):
    pt_min, pt_max = config["pt"]
    mass_min, mass_max = config["mass"]
    return (
        truth["valid"]
        & (truth["pt1"] >= pt_min)
        & (truth["pt2"] >= pt_min)
        & ((truth["pt1"] <= pt_max) if pt_max is not None else True)
        & ((truth["pt2"] <= pt_max) if pt_max is not None else True)
        & (np.abs(truth["eta1"]) <= config["max_abs_eta"])
        & (np.abs(truth["eta2"]) <= config["max_abs_eta"])
        & (truth["mass"] >= mass_min)
        & ((truth["mass"] <= mass_max) if mass_max is not None else True)
    )


def load_madgraph_component(spec, component_id, seed_id, parameters, args, pool):
    final_pdg = 4 if spec["source_flavor"] == "cc" else 5
    configs = {name: madgraph_campaign_config(spec["process"], name) for name in spec["campaigns"]}
    samples = []
    for campaign in spec["campaigns"]:
        config, input_dir, files = resolve_madgraph_files(spec["process"], campaign, args.max_files)
        tasks = [(path, args.tree, args.collection, final_pdg) for path in files]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            pieces = list(executor.map(madgraph_piece, tasks))
        valid_pieces = [piece for piece in pieces if not piece.get("empty")]
        if not valid_pieces:
            raise RuntimeError(f"{spec['name']}/{campaign} has no central events")
        n_generated = sum(piece["n_generated"] for piece in pieces)
        xsec_fb, xsec_source = generator_cross_section_fb("madgraph", spec["process"], campaign)
        offsets = np.cumsum([0] + [piece["n_generated"] for piece in pieces[:-1]])
        groups = np.concatenate(
            [piece["event_indices"] + offset for piece, offset in zip(pieces, offsets) if not piece.get("empty")]
        ).astype(np.int64)
        truth = {
            key: np.concatenate([piece["truth"][key] for piece in valid_pieces])
            for key in ("valid", "pt1", "pt2", "eta1", "eta2", "mass")
        }
        samples.append(
            {
                "campaign": campaign,
                "config": config,
                "input_dir": input_dir,
                "files": files,
                "matrix": np.vstack([piece["matrix"] for piece in valid_pieces]),
                "rapidity": np.concatenate([piece["dijet_rapidity"] for piece in valid_pieces]),
                "groups": groups,
                "truth": truth,
                "n_generated": n_generated,
                "xsec_fb": xsec_fb,
                "xsec_source": xsec_source,
            }
        )

    luminosities = {sample["campaign"]: sample["n_generated"] / sample["xsec_fb"] for sample in samples}
    campaign_stats = {}
    kept = []
    for sample in samples:
        masks = {campaign: phase_space_mask(sample["truth"], config) for campaign, config in configs.items()}
        stitch = stitched_cross_section_weights(sample["campaign"], masks, luminosities)
        keep = stitch > 0.0
        if not np.any(keep):
            raise RuntimeError(f"{spec['name']}/{sample['campaign']} has no covered events")
        coverage = np.zeros(keep.shape, dtype=np.uint8)
        for bit, campaign in enumerate(spec["campaigns"]):
            coverage |= np.asarray(masks[campaign], dtype=np.uint8) << bit
        kept.append(
            {
                "matrix": sample["matrix"][keep],
                "rapidity": sample["rapidity"][keep],
                "groups": sample["groups"][keep],
                "campaign": np.full(np.sum(keep), sample["campaign"]),
                "stitch": stitch[keep],
                "coverage": coverage[keep],
                "parton": np.column_stack(
                    [sample["truth"][key][keep] for key in PARTON_FIELDS]
                ),
            }
        )
        campaign_stats[sample["campaign"]] = {
            "subcampaign": sample["config"]["subcampaign"],
            "phase_space": {
                "parton_pt_gev": sample["config"]["pt"],
                "max_abs_eta": sample["config"]["max_abs_eta"],
                "dijet_mass_gev": sample["config"]["mass"],
            },
            "input_dir": str(sample["input_dir"]),
            "files": len(sample["files"]),
            "generated": sample["n_generated"],
            "central_before_stitching": int(keep.size),
            "central_kept": int(np.sum(keep)),
            "xsec_fb": sample["xsec_fb"],
            "xsec_source": sample["xsec_source"],
            "effective_mc_luminosity_fb_inv": luminosities[sample["campaign"]],
        }

    matrix = np.vstack([item["matrix"] for item in kept])
    rapidity = np.concatenate([item["rapidity"] for item in kept])
    campaign = np.concatenate([item["campaign"] for item in kept])
    stitch = np.concatenate([item["stitch"] for item in kept])
    coverage = np.concatenate([item["coverage"] for item in kept])
    # Campaign-local event numbers are made unique inside this physical component.
    group_blocks = []
    offset = 0
    for item in kept:
        _, inverse = np.unique(item["groups"], return_inverse=True)
        group_blocks.append(inverse.astype(np.int64) + offset)
        offset += int(inverse.max()) + 1
    groups = np.concatenate(group_blocks)
    sampled = source.conditional_pair_sample(
        matrix,
        rapidity,
        groups,
        campaign,
        np.zeros(groups.size, dtype=np.uint8),
        stitch,
        coverage,
        pool,
        train_pairs=1,
        evaluation_pairs=args.evaluation_pairs,
        seed=args.seed + seed_id * 10000,
    )
    # groups is a dense but permuted index over central events, so invert it
    # before expanding per-event quantities onto the sampled pair rows.
    parton_rows = np.vstack([item["parton"] for item in kept])
    parton_by_group = np.empty((int(groups.max()) + 1, len(PARTON_FIELDS)), dtype=np.float64)
    parton_by_group[groups] = parton_rows
    selected_features = sampled["features"][:, FEATURE_INDICES]
    process_scale = generator_weight("madgraph", spec["process"])
    tag = tag_factor(parameters, spec["source_flavor"])
    combinatorial = float(parameters["normalization"]["combinatorial_acceptance_factor"])
    physical_scale = LUMI_FB * process_scale * combinatorial * pool["acceptance"] * tag
    mixture_scale = process_scale * combinatorial * pool["acceptance"] * tag
    physical = sampled["stitch_weight_fb"] * physical_scale
    mixture = sampled["stitch_weight_fb"] * mixture_scale
    central_weight = sampled["central_stitch_weight_fb"] * physical_scale
    if np.any(physical <= 0.0) or np.any(mixture <= 0.0):
        raise RuntimeError(f"{spec['name']} has non-positive sampled weights")
    rows = selected_features.shape[0]
    print(
        f"{spec['name']}: campaigns={','.join(spec['campaigns'])} central={matrix.shape[0]:,} "
        f"sampled={rows:,} yield={physical.sum():.6g}",
        flush=True,
    )
    return {
        "x": selected_features,
        "parton": parton_by_group[sampled["group"]],
        "class": np.full(rows, ACTIVE_CLASSES.index(spec["class_name"]), dtype=np.int8),
        "component": np.full(rows, component_id, dtype=np.int8),
        "group_id": sampled["group"].astype(np.int64),
        "mx": sampled["mx"].astype(np.float64),
        "physical_weight": physical.astype(np.float64),
        "training_mixture_weight": mixture.astype(np.float64),
        "central_weight": central_weight.astype(np.float64),
        "band_probability": sampled["band_probability"].astype(np.float64),
        "input": {
            "campaigns": campaign_stats,
            "central_selected": int(matrix.shape[0]),
            "sampled_rows": rows,
            "conditional_pairs": args.evaluation_pairs,
            "bx_pair_acceptance": pool["acceptance"],
        },
    }


ARRAY_DTYPES = {
    "x": np.float32,
    "parton": np.float64,
    "class": np.int8,
    "component": np.int8,
    "group_id": np.int64,
    "mx": np.float64,
    "physical_weight": np.float64,
    "training_mixture_weight": np.float64,
    "central_weight": np.float64,
    "band_probability": np.float64,
}


def active_arrays():
    names = list(ARRAY_DTYPES)
    if not STORE_PARTON:
        names.remove("parton")
    return names


def array_shape(name, rows):
    if name == "x":
        return (rows, len(SELECTED_FEATURES))
    if name == "parton":
        return (rows, len(PARTON_FIELDS))
    return (rows,)


def write_dataset(data_dir, samples):
    rows = sum(sample["class"].size for sample in samples)
    names = active_arrays()
    outputs = {
        name: np.lib.format.open_memmap(
            data_dir / f"{name}.npy", mode="w+",
            dtype=ARRAY_DTYPES[name], shape=array_shape(name, rows),
        )
        for name in names
    }
    start = 0
    group_offset = 0
    for sample in samples:
        stop = start + sample["class"].size
        unique_groups, local_groups = np.unique(sample["group_id"], return_inverse=True)
        for name in names:
            values = sample[name]
            if name == "group_id":
                values = local_groups.astype(np.int64) + group_offset
            outputs[name][start:stop] = values
        group_offset += unique_groups.size
        start = stop
    for output in outputs.values():
        output.flush()
    del outputs
    return rows, group_offset


def main():
    global SELECTED_FEATURES, FEATURE_INDICES, ACTIVE_SPECS, ACTIVE_CLASSES, STORE_PARTON
    args = parse_args()
    validate_args(args)
    STORE_PARTON = args.store_parton
    SELECTED_FEATURES = FULL_FEATURES if args.feature_set == "full" else LOCKED_FEATURES
    FEATURE_INDICES = np.asarray([FULL_FEATURES.index(name) for name in SELECTED_FEATURES])
    if args.components is not None:
        chosen = set(args.components)
        ACTIVE_SPECS = tuple(spec for spec in COMPONENT_SPECS if spec["name"] in chosen)
        used = {spec["class_name"] for spec in ACTIVE_SPECS}
        ACTIVE_CLASSES = tuple(name for name in CLASS_NAMES if name in used)
    overrides = dict(item.split("=", 1) for item in (args.extra_campaigns or []))
    if overrides:
        ACTIVE_SPECS = tuple(
            {**spec, "campaigns": tuple(overrides[spec["name"]].split(","))}
            if spec["name"] in overrides else spec
            for spec in ACTIVE_SPECS
        )
    started = time.perf_counter()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    parameters = read_parameters()
    pps_path = resolve_path(args.pps_config, base=REPO)
    pps = pps_lib.load_pps_config(pps_path)
    pool = source.load_bootstrap_pool(args.minbias_campaign)
    if not np.isfinite(pool["acceptance"]) or pool["acceptance"] <= 0.0:
        raise RuntimeError("Min-bias proton-pair pool has invalid BX acceptance")

    samples = []
    inputs = {}
    for component_id, spec in enumerate(ACTIVE_SPECS):
        seed_id = [item["name"] for item in COMPONENT_SPECS].index(spec["name"])
        if spec["generator"] == "superchic":
            sample = load_superchic_component(spec, component_id, seed_id, parameters, args, pps)
        else:
            sample = load_madgraph_component(spec, component_id, seed_id, parameters, args, pool)
        if sample["class"].size == 0:
            raise RuntimeError(f"Physical component {spec['name']} is empty")
        samples.append(sample)
        inputs[spec["name"]] = sample.pop("input")

    rows, groups = write_dataset(data_dir, samples)
    shutil.copyfile(pool["path"], data_dir / "proton_pairs.parquet")
    classes = np.concatenate([sample["class"] for sample in samples])
    components = np.concatenate([sample["component"] for sample in samples])
    physical = np.concatenate([sample["physical_weight"] for sample in samples])
    rows_per_class = np.bincount(classes, minlength=len(ACTIVE_CLASSES))
    yields_per_class = np.bincount(classes, weights=physical, minlength=len(ACTIVE_CLASSES))
    rows_per_component = np.bincount(components, minlength=len(ACTIVE_SPECS))
    yields_per_component = np.bincount(
        components, weights=physical, minlength=len(ACTIVE_SPECS)
    )
    if np.any(rows_per_class == 0) or np.any(yields_per_class <= 0.0):
        raise RuntimeError("Every training class must have rows and positive physical yield")

    metadata = {
        "format_version": 1,
        "description": "Five-class reduced H(cc) central-plus-proton MVA dataset",
        "features": SELECTED_FEATURES,
        "classes": list(ACTIVE_CLASSES),
        "components": component_manifest(parameters, ACTIVE_SPECS, ACTIVE_CLASSES),
        "rows": rows,
        "groups": groups,
        "rows_per_class": dict(zip(ACTIVE_CLASSES, rows_per_class)),
        "physical_yields_per_class": dict(zip(ACTIVE_CLASSES, yields_per_class)),
        "rows_per_component": {
            spec["name"]: rows_per_component[index]
            for index, spec in enumerate(ACTIVE_SPECS)
        },
        "physical_yields_per_component": {
            spec["name"]: yields_per_component[index]
            for index, spec in enumerate(ACTIVE_SPECS)
        },
        "tagging": {
            "eff_c": float(parameters["tagging"]["eff_c"]),
            "mistag_b_to_c": float(parameters["tagging"]["mistag_b_to_c"]),
            "double_tag_convention": "per-event factor is the square of the per-jet probability",
        },
        "selection": {
            "mass_window_gev": MASS_WINDOW_GEV,
            "max_abs_rapidity_difference": MAX_ABS_RAPIDITY_DIFFERENCE,
            "jet_collection": args.collection,
        },
        "madgraph": {
            "qcdcc_campaigns": ["QCDcc__v01"],
            "qcdbb_campaigns": ["QCDbb__v02", "QCDbb__v03"],
            "excluded_qcdbb_campaign": "QCDbb__v01",
            "combinatorial_acceptance_factor": float(
                parameters["normalization"]["combinatorial_acceptance_factor"]
            ),
            "proton_pool": str(pool["path"]),
            "bx_pair_acceptance": pool["acceptance"],
        },
        "inputs": inputs,
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "arrays": {
            name: {
                "dtype": str(np.dtype(ARRAY_DTYPES[name])),
                "shape": list(array_shape(name, rows)),
            }
            for name in active_arrays()
        },
    }
    write_yaml(data_dir / "metadata.yaml", metadata)
    print(
        f"Wrote {rows:,} rows, {groups:,} groups, and {len(SELECTED_FEATURES)} features to {data_dir}\n"
        f"Class rows: {plain(rows_per_class)}\n"
        f"Class yields: {plain(yields_per_class)}\n"
        f"Done in {time.perf_counter() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
