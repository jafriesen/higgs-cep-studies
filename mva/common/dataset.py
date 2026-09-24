"""Central-event dataset construction for the H(bb) and H(cc) channels."""

import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from analysis.cross_sections import event_record_files, generator_cross_section_fb, generator_weight
from common.config_utils import natural_key
from common.jet_calibration import load_correction_map
from common.path_helper import (
    generation_campaign_config,
    generation_process_config,
    generation_stage_root,
)
from mva.common.features import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    central_features,
    stored_central_features,
)
from mva.common.protons import (
    build_pair_density,
    parse_hepmc_protons,
    parse_lhe_protons,
    real_proton_pass,
)
from mva.common.weights import LUMI_FB, stitched_cross_section_weights, tag_factor


PARTON_FIELDS = ("pt1", "pt2", "eta1", "eta2", "mass")
PROTON_TRANSVERSE_FIELDS = (
    "proton_px_left",
    "proton_py_left",
    "proton_px_right",
    "proton_py_right",
)
ARRAY_DTYPES = {
    "x": np.float32,
    "class": np.int8,
    "component": np.int8,
    "campaign": np.int16,
    "group_id": np.int64,
    "dijet_rapidity": np.float64,
    "dijet_mass": np.float64,
    "physical_weight": np.float64,
    "training_mixture_weight": np.float64,
    "pair_band_intensity": np.float64,
    "proton_yx": np.float64,
    "proton_mx": np.float64,
    **{name: np.float64 for name in PROTON_TRANSVERSE_FIELDS},
    "truth_matched": np.bool_,
    "truth_is_leading": np.bool_,
    "match_dr1": np.float64,
    "match_dr2": np.float64,
    "correction_valid": np.bool_,
    "parton": np.float64,
}


def _default_subcampaign(process_config, stage):
    defaults = process_config.get("default_campaign") or {}
    return defaults.get(stage) if isinstance(defaults, dict) else None


def correction_source_sample(spec, campaign=None):
    """Resolve the correction-map sample key for one component campaign."""
    source_sample = str(spec["source_sample"])
    if "{campaign}" not in source_sample:
        return source_sample
    if campaign is None:
        raise ValueError(
            f"Component {spec['name']} needs a campaign to resolve "
            f"source_sample={source_sample!r}"
        )
    return source_sample.format(campaign=campaign.split("__")[-1])


def load_component_correction_map(repo, spec, campaign=None):
    """Load the correction map selected for one component campaign."""
    return load_correction_map(
        repo / spec["correction_map"],
        spec["fsr_state"],
        correction_source_sample(spec, campaign),
    )


def component_truth_pid_abs(spec):
    """Resolve the hard-pair PID used for truth matching."""
    configured = spec.get("truth_pid_abs")
    if configured is not None:
        return int(configured)
    try:
        return {"bb": 5, "cc": 4}[spec["source_flavor"]]
    except KeyError as error:
        raise ValueError(
            f"Component {spec['name']} with source_flavor={spec['source_flavor']!r} "
            "must define truth_pid_abs"
        ) from error


def _part_offset(path):
    """Events skipped before this file, for a sample split into parts.

    A part covers events [skip, skip + max_events) of its event record, so proton
    indices have to be shifted by the same amount. The runner writes the Pythia card
    next to the ROOT output and it carries the offset.
    """
    if "__part" not in path.stem:
        return 0
    card = path.parent.parent / "cmnd" / f"{path.stem}.cmnd"
    if not card.is_file():
        raise RuntimeError(f"Part file without its Pythia card, cannot place protons: {path}")
    for line in card.read_text(encoding="utf-8").splitlines():
        if line.startswith("Beams:nSkipLHEFatInit"):
            return int(line.split("=")[1])
    return 0


def _record_stem(path):
    """File stem with any part suffix removed, which is how proton records are named."""
    return path.stem.split("__part")[0]


def _superchic_inputs(spec, max_files):
    campaign, campaign_config = generation_campaign_config(
        "superchic", spec["process"], spec.get("campaign")
    )
    process = generation_process_config("superchic", spec["process"])
    subcampaign = spec.get("sim_subcampaign") or _default_subcampaign(process, "sim-delphes")
    input_dir = generation_stage_root(
        "superchic", spec["process"], campaign, "sim-delphes", subcampaign=subcampaign
    ) / "root"
    files = sorted(input_dir.glob("*.root"), key=natural_key)
    file_limit = max_files if max_files is not None else spec.get("max_files", 100)
    files = files[:file_limit]
    if not files:
        raise RuntimeError(f"No ROOT files found in {input_dir}")
    explicit_pythia = spec.get("pythia_subcampaign")
    pythia_subcampaign = explicit_pythia or _default_subcampaign(process, "hadr-pythia")
    allowed_pythia = campaign_config.get("hadr-pythia") or []
    hepmc = {}
    if explicit_pythia or pythia_subcampaign in allowed_pythia:
        pythia_dir = generation_stage_root(
            "superchic",
            spec["process"],
            campaign,
            "hadr-pythia",
            subcampaign=pythia_subcampaign,
        ) / "hepmc"
        hepmc = {
            path.stem: path
            for path in sorted(pythia_dir.glob("*.hepmc"), key=natural_key)
        }
    if all(_record_stem(path) in hepmc for path in files):
        kind = "hepmc"
        proton_files = [hepmc[_record_stem(path)] for path in files]
    else:
        records = {
            path.stem: path
            for path in event_record_files("superchic", spec["process"], campaign)
        }
        missing = [path.name for path in files if _record_stem(path) not in records]
        if missing:
            raise RuntimeError(f"Missing matching proton records for: {', '.join(missing)}")
        kind = "lhe"
        proton_files = [records[_record_stem(path)] for path in files]
    offsets = [_part_offset(path) for path in files]
    return campaign, subcampaign, input_dir, files, kind, proton_files, offsets


def _madgraph_campaign_config(process, campaign):
    process_config = generation_process_config("madgraph", process)
    campaign_config = (process_config.get("campaigns") or {}).get(campaign)
    if campaign_config is None:
        raise RuntimeError(f"MadGraph {process} has no campaign {campaign}")
    phase = campaign_config.get("phase_space") or {}
    subcampaign = campaign_config.get("mva_subcampaign")
    if not subcampaign or any(
        key not in phase for key in ("parton_pt_gev", "max_abs_eta", "dijet_mass_gev")
    ):
        raise RuntimeError(f"MadGraph {process}/{campaign} lacks MVA phase-space metadata")
    return {
        "subcampaign": subcampaign,
        "pt": tuple(phase["parton_pt_gev"]),
        "max_abs_eta": float(phase["max_abs_eta"]),
        "mass": tuple(phase["dijet_mass_gev"]),
    }


def _madgraph_inputs(spec, campaign, max_files):
    config = _madgraph_campaign_config(spec["process"], campaign)
    # A component may override the campaign's default sim-Delphes subcampaign, so the
    # same campaigns can be read in a different FSR state or card without editing
    # processes-madgraph.yaml (whose defaults other datasets still depend on).
    # "{campaign}" in the pattern expands to the campaign's version suffix.
    pattern = spec.get("sim_subcampaign")
    subcampaign = (
        pattern.format(campaign=campaign.split("__")[-1]) if pattern else config["subcampaign"]
    )
    input_dir = generation_stage_root(
        "madgraph",
        spec["process"],
        campaign,
        "sim-delphes",
        subcampaign=subcampaign,
    ) / "root"
    files = sorted(input_dir.glob("*.root"), key=natural_key)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No ROOT files found in {input_dir}")
    return config, input_dir, files


def _slice_piece(piece, keep):
    output = dict(piece)
    for name in (
        "matrix",
        "event_indices",
        "dijet_mass",
        "dijet_rapidity",
        "truth_matched",
        "truth_is_leading",
        "match_dr1",
        "match_dr2",
        "correction_valid",
    ):
        output[name] = piece[name][keep]
    output["parton"] = {name: values[keep] for name, values in piece["parton"].items()}
    return output


def _file_piece(task):
    (
        path,
        proton_kind,
        proton_path,
        proton_offset,
        tree,
        collection,
        truth_pid_abs,
        jets,
        correction_map,
        track_min_pt,
        max_abs_jet_eta,
        pps,
        seed,
        mass_window,
        max_delta_y,
    ) = task
    piece = central_features(
        path,
        tree,
        collection,
        truth_pid_abs=truth_pid_abs,
        jet_mode=jets,
        correction_map=correction_map,
        track_min_pt=track_min_pt,
        max_abs_jet_eta=max_abs_jet_eta,
    )
    if piece.get("empty") or proton_kind is None:
        return piece
    parser = parse_hepmc_protons if proton_kind == "hepmc" else parse_lhe_protons
    protons = parser(
        proton_path, np.asarray(piece["event_indices"]) + proton_offset, pps["sqrt_s"]
    )
    passed, _left, _right, mx, yx = real_proton_pass(
        protons["xi_left"], protons["xi_right"], pps, np.random.default_rng(seed)
    )
    keep = (
        passed
        & (mx >= mass_window[0])
        & (mx <= mass_window[1])
        & (np.abs(yx - piece["dijet_rapidity"]) < max_delta_y)
    )
    selected = _slice_piece(piece, keep)
    selected["proton_mx"] = mx[keep]
    selected["proton_yx"] = yx[keep]
    for name in PROTON_TRANSVERSE_FIELDS:
        selected[name] = protons[name][keep]
    selected["n_central_before_protons"] = piece["matrix"].shape[0]
    return selected


def _run_tasks(tasks, workers, label):
    started = time.perf_counter()
    pieces = [None] * len(tasks)
    if workers == 1:
        for index, task in enumerate(tasks):
            pieces[index] = _file_piece(task)
            elapsed = time.perf_counter() - started
            print(
                f"  {label}: file {index + 1}/{len(tasks)} elapsed={elapsed:.1f}s "
                f"eta={elapsed / (index + 1) * (len(tasks) - index - 1):.1f}s",
                flush=True,
            )
        return pieces
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        futures = {executor.submit(_file_piece, task): index for index, task in enumerate(tasks)}
        completed = 0
        for future in as_completed(futures):
            pieces[futures[future]] = future.result()
            completed += 1
            elapsed = time.perf_counter() - started
            print(
                f"  {label}: file {completed}/{len(tasks)} elapsed={elapsed:.1f}s "
                f"eta={elapsed / completed * (len(tasks) - completed):.1f}s",
                flush=True,
            )
    return pieces


def _concatenate(pieces, name, dtype=np.float64):
    values = [piece[name] for piece in pieces if not piece.get("empty")]
    return np.concatenate(values) if values else np.empty(0, dtype=dtype)


def _central_block(pieces, feature_indices):
    valid = [piece for piece in pieces if not piece.get("empty")]
    if not valid:
        raise RuntimeError("No selected central events")
    generated_offsets = np.cumsum([0] + [piece["n_generated"] for piece in pieces[:-1]])
    groups = np.concatenate(
        [
            piece["event_indices"] + offset
            for piece, offset in zip(pieces, generated_offsets)
            if not piece.get("empty")
        ]
    ).astype(np.int64)
    parton = np.column_stack(
        [np.concatenate([piece["parton"][name] for piece in valid]) for name in PARTON_FIELDS]
    )
    return {
        "x": np.vstack([piece["matrix"][:, feature_indices] for piece in valid]),
        "group_id": groups,
        "dijet_rapidity": _concatenate(valid, "dijet_rapidity"),
        "dijet_mass": _concatenate(valid, "dijet_mass"),
        "truth_matched": _concatenate(valid, "truth_matched", bool).astype(bool),
        "truth_is_leading": _concatenate(valid, "truth_is_leading", bool).astype(bool),
        "match_dr1": _concatenate(valid, "match_dr1"),
        "match_dr2": _concatenate(valid, "match_dr2"),
        "correction_valid": _concatenate(valid, "correction_valid", bool).astype(bool),
        "parton": parton,
        "n_generated": sum(piece["n_generated"] for piece in pieces),
        "correction_candidates": sum(piece.get("correction_candidate_events", 0) for piece in pieces),
        "correction_invalid": sum(piece.get("correction_invalid_events", 0) for piece in pieces),
    }


def _phase_space_mask(parton, config):
    pt_min, pt_max = config["pt"]
    mass_min, mass_max = config["mass"]
    return (
        parton[:, 0] >= pt_min
    ) & (
        parton[:, 1] >= pt_min
    ) & (
        (parton[:, 0] <= pt_max) if pt_max is not None else True
    ) & (
        (parton[:, 1] <= pt_max) if pt_max is not None else True
    ) & (
        np.abs(parton[:, 2]) <= config["max_abs_eta"]
    ) & (
        np.abs(parton[:, 3]) <= config["max_abs_eta"]
    ) & (
        parton[:, 4] >= mass_min
    ) & (
        (parton[:, 4] <= mass_max) if mass_max is not None else True
    )


def _finish_component(spec, config, campaign_blocks, input_metadata, parameters, started):
    """Assemble weighted campaign blocks into one component sample."""
    rows = sum(block["x"].shape[0] for block in campaign_blocks)
    if rows == 0:
        raise RuntimeError(f"Component {spec['name']} has no selected events")
    tag = tag_factor(parameters, spec["source_flavor"], config["target_flavor"])
    cross_section_weight = np.concatenate(
        [block["cross_section_weight"] for block in campaign_blocks]
    )
    physical = cross_section_weight * float(config.get("luminosity_fb", LUMI_FB)) * tag
    mixture = cross_section_weight * tag
    real_protons = bool(spec["real_protons"])
    group_blocks = []
    group_offset = 0
    for block in campaign_blocks:
        _unique, local = np.unique(block["group_id"], return_inverse=True)
        group_blocks.append(local.astype(np.int64) + group_offset)
        group_offset += _unique.size
    output = {
        "x": np.vstack([block["x"] for block in campaign_blocks]),
        "class": np.full(rows, spec["class_id"], dtype=np.int8),
        "component": np.full(rows, spec["id"], dtype=np.int8),
        "campaign": np.concatenate([
            np.full(block["x"].shape[0], index, dtype=np.int16)
            for index, block in enumerate(campaign_blocks)
        ]),
        "group_id": np.concatenate(group_blocks),
        "dijet_rapidity": np.concatenate(
            [block["dijet_rapidity"] for block in campaign_blocks]
        ),
        "dijet_mass": np.concatenate([block["dijet_mass"] for block in campaign_blocks]),
        "physical_weight": physical,
        "training_mixture_weight": mixture,
        "proton_mx": (
            np.concatenate([block["proton_mx"] for block in campaign_blocks])
            if real_protons else np.full(rows, np.nan)
        ),
        "proton_yx": (
            np.concatenate([block["proton_yx"] for block in campaign_blocks])
            if real_protons else np.full(rows, np.nan)
        ),
        "truth_matched": np.concatenate(
            [block["truth_matched"] for block in campaign_blocks]
        ),
        "truth_is_leading": np.concatenate(
            [block["truth_is_leading"] for block in campaign_blocks]
        ),
        "match_dr1": np.concatenate([block["match_dr1"] for block in campaign_blocks]),
        "match_dr2": np.concatenate([block["match_dr2"] for block in campaign_blocks]),
        "correction_valid": np.concatenate(
            [block["correction_valid"] for block in campaign_blocks]
        ),
        "parton": np.vstack([block["parton"] for block in campaign_blocks]),
    }
    for name in PROTON_TRANSVERSE_FIELDS:
        output[name] = (
            np.concatenate([block[name] for block in campaign_blocks])
            if real_protons else np.full(rows, np.nan)
        )
    correction_candidates = sum(
        block["correction_candidates"] for block in campaign_blocks
    )
    correction_invalid = sum(block["correction_invalid"] for block in campaign_blocks)
    output["input"] = {
        "campaigns": input_metadata,
        "generated": sum(block["n_generated"] for block in campaign_blocks),
        "selected": rows,
        "correction_candidates": correction_candidates,
        "correction_invalid": correction_invalid,
        "correction_invalid_fraction": (
            correction_invalid / correction_candidates if correction_candidates else None
        ),
        "truth_matched_fraction": float(np.mean(output["truth_matched"])),
        "truth_is_leading_fraction": float(np.mean(output["truth_is_leading"])),
        "runtime_seconds": time.perf_counter() - started,
    }
    print(
        f"{spec['name']}: selected={rows:,} truth_matched={np.mean(output['truth_matched']):.3%} "
        f"truth_is_leading={np.mean(output['truth_is_leading']):.3%} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return output


def build_component(
    spec,
    config,
    feature_indices,
    pps,
    parameters,
    args,
):
    started = time.perf_counter()
    truth_pid_abs = component_truth_pid_abs(spec)
    input_metadata = {}
    campaign_blocks = []
    if spec["generator"] == "superchic":
        campaign, subcampaign, input_dir, files, proton_kind, proton_files, proton_offsets = (
            _superchic_inputs(spec, args.max_files)
        )
        correction_source = correction_source_sample(spec, campaign)
        correction_map = (
            load_component_correction_map(args.repo, spec, campaign)
            if args.corrections else None
        )
        common_task = (
            args.tree,
            args.collection,
            truth_pid_abs,
            args.jets,
            correction_map,
            args.track_min_pt,
            args.max_abs_jet_eta,
            pps,
        )
        tasks = [
            (
                path,
                proton_kind,
                proton_file,
                proton_offset,
                *common_task,
                args.seed + spec["global_id"] * 10000 + index,
                args.mass_window,
                args.max_delta_y,
            )
            for index, (path, proton_file, proton_offset) in enumerate(
                zip(files, proton_files, proton_offsets)
            )
        ]
        pieces = _run_tasks(tasks, args.workers, spec["name"])
        block = _central_block(pieces, feature_indices)
        block["proton_mx"] = _concatenate(pieces, "proton_mx")
        block["proton_yx"] = _concatenate(pieces, "proton_yx")
        for name in PROTON_TRANSVERSE_FIELDS:
            block[name] = _concatenate(pieces, name)
        xsec_fb, xsec_source = generator_cross_section_fb("superchic", spec["process"], campaign)
        block["cross_section_weight"] = np.full(
            block["x"].shape[0],
            xsec_fb * generator_weight("superchic", spec["process"]) / block["n_generated"],
        )
        block["campaign_name"] = campaign
        campaign_blocks.append(block)
        input_metadata[campaign] = {
            "subcampaign": subcampaign,
            "input_dir": str(input_dir),
            "files": len(files),
            "generated": block["n_generated"],
            "central_before_protons": sum(piece.get("n_central_before_protons", 0) for piece in pieces),
            "selected": int(block["x"].shape[0]),
            "xsec_fb": xsec_fb,
            "xsec_source": xsec_source,
            "proton_record_type": proton_kind,
            "correction_source_sample": correction_source if args.corrections else None,
        }
    else:
        campaign_inputs = []
        for campaign in spec["campaigns"]:
            phase, input_dir, files = _madgraph_inputs(spec, campaign, args.max_files)
            correction_source = correction_source_sample(spec, campaign)
            correction_map = (
                load_component_correction_map(args.repo, spec, campaign)
                if args.corrections else None
            )
            campaign_task = (
                args.tree,
                args.collection,
                truth_pid_abs,
                args.jets,
                correction_map,
                args.track_min_pt,
                args.max_abs_jet_eta,
                pps,
            )
            tasks = [
                (
                    path,
                    None,
                    None,
                    0,  # no proton record for pooled MadGraph events
                    *campaign_task,
                    args.seed + spec["global_id"] * 10000 + index,
                    args.mass_window,
                    args.max_delta_y,
                )
                for index, path in enumerate(files)
            ]
            pieces = _run_tasks(tasks, args.workers, f"{spec['name']}/{campaign}")
            block = _central_block(pieces, feature_indices)
            xsec_fb, xsec_source = generator_cross_section_fb("madgraph", spec["process"], campaign)
            campaign_inputs.append(
                (campaign, phase, input_dir, files, block, xsec_fb, xsec_source, correction_source)
            )
        if spec.get("campaign_combination") == "stitch":
            luminosities = {
                campaign: block["n_generated"] / xsec_fb
                for campaign, _phase, _dir, _files, block, xsec_fb, _source, _correction in campaign_inputs
            }
            phases = {campaign: phase for campaign, phase, *_rest in campaign_inputs}
            for (
                campaign, phase, input_dir, files, block, xsec_fb, xsec_source,
                correction_source,
            ) in campaign_inputs:
                masks = {name: _phase_space_mask(block["parton"], item) for name, item in phases.items()}
                weights = stitched_cross_section_weights(campaign, masks, luminosities)
                keep = weights > 0.0
                for name, values in list(block.items()):
                    if isinstance(values, np.ndarray) and values.shape[:1] == keep.shape:
                        block[name] = values[keep]
                block["cross_section_weight"] = weights[keep] * generator_weight(
                    "madgraph", spec["process"]
                )
                input_metadata[campaign] = {
                    "subcampaign": input_dir.parent.name, "input_dir": str(input_dir),
                    "files": len(files), "generated": block["n_generated"],
                    "selected": int(np.sum(keep)), "xsec_fb": xsec_fb,
                    "xsec_source": xsec_source, "combination": "stitched",
                    "correction_source_sample": correction_source if args.corrections else None,
                }
                block["campaign_name"] = campaign
                campaign_blocks.append(block)
        else:
            for (
                campaign, phase, input_dir, files, block, xsec_fb, xsec_source,
                correction_source,
            ) in campaign_inputs:
                block["cross_section_weight"] = np.full(
                    block["x"].shape[0],
                    xsec_fb * generator_weight("madgraph", spec["process"]) / block["n_generated"],
                )
                input_metadata[campaign] = {
                    "subcampaign": input_dir.parent.name, "input_dir": str(input_dir),
                    "files": len(files), "generated": block["n_generated"],
                    "selected": int(block["x"].shape[0]), "xsec_fb": xsec_fb,
                    "xsec_source": xsec_source, "combination": "disjoint",
                    "correction_source_sample": correction_source if args.corrections else None,
                }
                block["campaign_name"] = campaign
                campaign_blocks.append(block)

    return _finish_component(
        spec, config, campaign_blocks, input_metadata, parameters, started
    )


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain(payload), handle, sort_keys=False)


def write_dataset(data_dir, samples, central_feature_names, store_parton, progress_rows=500000):
    data_dir.mkdir(parents=True, exist_ok=True)
    names = list(ARRAY_DTYPES)
    if not store_parton:
        names.remove("parton")
    rows = sum(sample["class"].size for sample in samples)
    shapes = {
        name: ((rows, len(central_feature_names)) if name == "x" else (rows, len(PARTON_FIELDS)) if name == "parton" else (rows,))
        for name in names
    }
    arrays = {
        name: np.lib.format.open_memmap(
            data_dir / f"{name}.npy", mode="w+", dtype=ARRAY_DTYPES[name], shape=shapes[name]
        )
        for name in names
    }
    started = time.perf_counter()
    start = group_offset = 0
    for sample in samples:
        stop = start + sample["class"].size
        _unique, groups = np.unique(sample["group_id"], return_inverse=True)
        for block_start in range(start, stop, progress_rows):
            block_stop = min(block_start + progress_rows, stop)
            local = slice(block_start - start, block_stop - start)
            for name in names:
                values = groups[local] + group_offset if name == "group_id" else sample[name][local]
                arrays[name][block_start:block_stop] = values
            elapsed = time.perf_counter() - started
            print(
                f"  write rows {block_stop:,}/{rows:,} elapsed={elapsed:.1f}s "
                f"eta={elapsed / block_stop * (rows - block_stop):.1f}s",
                flush=True,
            )
        group_offset += np.unique(groups).size
        start = stop
    for array in arrays.values():
        array.flush()
    del arrays
    return rows, group_offset, shapes


def finish_dataset(
    config,
    args,
    samples,
    parameters,
    pps,
    requested_features,
    central_feature_names,
    started,
):
    """Apply proton intensities and write already-built component samples."""
    if config["profile"].get("proton_backend") == "analytic":
        minbias_path = (args.repo / config["minbias_path"]).resolve()
        print(f"Loading analytic min-bias flux: {minbias_path}", flush=True)
        pairs = build_pair_density(minbias_path, pps, seed=args.seed, verify_hash=not args.skip_hash)
        for spec, sample in zip(config["components"], samples):
            if spec["real_protons"]:
                sample["pair_band_intensity"] = np.ones(sample["class"].shape)
                continue
            intensity = np.empty(sample["class"].shape, dtype=np.float64)
            intensity_started = time.perf_counter()
            for start in range(0, intensity.size, args.intensity_chunk):
                stop = min(start + args.intensity_chunk, intensity.size)
                y = sample["dijet_rapidity"][start:stop]
                intensity[start:stop] = config["pileup_mu"] ** 2 * pairs.integrate_yx_ranges(
                    args.mass_window, y - args.max_delta_y, y + args.max_delta_y
                )
                elapsed = time.perf_counter() - intensity_started
                print(
                    f"  {spec['name']} pair intensity {stop:,}/{intensity.size:,} "
                    f"elapsed={elapsed:.1f}s eta={elapsed / stop * (intensity.size - stop):.1f}s",
                    flush=True,
                )
            sample["pair_band_intensity"] = intensity
        pair_metadata = {
            "backend": "analytic",
            "path": str(minbias_path),
            "n_inelastic_generated": pairs.flux.n_inelastic_generated,
            "pileup_mu": config["pileup_mu"],
            "full_mass_window_intensity_per_interaction_pair": pairs.integrate(args.mass_window),
        }
    else:
        from mva.common.protons import LegacyPairDensity, load_bootstrap_pool

        pool = load_bootstrap_pool()
        pairs = LegacyPairDensity(pool)
        legacy_scale = 0.005 * pool["acceptance"]
        for spec, sample in zip(config["components"], samples):
            if spec["real_protons"]:
                sample["pair_band_intensity"] = np.ones(sample["class"].shape)
                continue
            sample["physical_weight"] *= legacy_scale
            sample["training_mixture_weight"] *= legacy_scale
            y = sample["dijet_rapidity"]
            sample["pair_band_intensity"] = pairs.integrate_yx_ranges(
                args.mass_window, y - args.max_delta_y, y + args.max_delta_y
            )
        pair_metadata = {
            "backend": "legacy_pool",
            "path": pool["path"],
            "pileup_mu": 1.0,
            "bx_pair_acceptance": pool["acceptance"],
            "combinatorial_acceptance_factor": 0.005,
            "note": "bounded parity control only",
        }

    rows, groups, shapes = write_dataset(
        args.data_dir, samples, central_feature_names, args.store_parton
    )
    component_yields = {}
    components = []
    inputs = {}
    for spec, sample in zip(config["components"], samples):
        effective = sample["physical_weight"] * sample["pair_band_intensity"]
        component_yields[spec["name"]] = float(np.sum(effective))
        inputs[spec["name"]] = sample["input"]
        item = dict(spec)
        item["campaigns"] = list(spec.get("campaigns") or []) or None
        item["tag_factor"] = tag_factor(
            parameters, spec["source_flavor"], config["target_flavor"]
        )
        # The legacy pool folds the flat vertex acceptance into the weights
        # already (0.005 on the pooled background, 1.0 on real pairs), so the
        # per-event vertex likelihood must not be applied on top of it.
        if pair_metadata["backend"] == "legacy_pool":
            item["vertex_hypothesis"] = "none"
        else:
            item["vertex_hypothesis"] = (
                "matched" if spec["real_protons"] else "unrelated"
            )
        components.append(item)
    metadata = {
        "format_version": 2,
        "channel": config["channel"],
        "profile": config["profile_name"],
        "feature_set": args.feature_set,
        "features": requested_features,
        "central_features": central_feature_names,
        "classes": config["classes"],
        "components": components,
        "tagging": parameters["tagging"],
        "rows": rows,
        "groups": groups,
        "physical_yields_per_component": component_yields,
        "selection": {
            "mass_window_gev": list(args.mass_window),
            "max_abs_rapidity_difference": args.max_delta_y,
            "jets": args.jets,
            "require_truth_matched_at_training": args.jets == "truth",
            "corrections": args.corrections,
            "max_abs_jet_eta": args.max_abs_jet_eta,
            "track_min_pt_gev": args.track_min_pt,
        },
        "protons": {
            **pair_metadata,
            "stored_transverse_components": {
                "arrays": list(PROTON_TRANSVERSE_FIELDS),
                "units": "GeV",
                "real_pairs": "unsmeared truth event-record values",
                "analytic_pairs": "not materialized; stored as NaN",
                "used_as_mva_features": False,
            },
        },
        "inputs": inputs,
        "arrays": {
            name: {
                "dtype": str(np.dtype(ARRAY_DTYPES[name])),
                "shape": list(shape),
            }
            for name, shape in shapes.items()
        },
        "runtime_seconds": time.perf_counter() - started,
        "seed": args.seed,
    }
    write_yaml(args.data_dir / "metadata.yaml", metadata)
    shutil.copyfile(config["path"], args.data_dir / "channel_config.yaml")
    print(
        f"Wrote {rows:,} central events in {groups:,} groups to {args.data_dir} "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return metadata


def prepare_dataset(config, args):
    started = time.perf_counter()
    with open(args.repo / config["parameters"], encoding="utf-8") as handle:
        parameters = yaml.safe_load(handle)
    from mva.common.protons import load_pps_config

    pps = load_pps_config(args.repo / config["pps_config"])
    requested_features = (
        list(ALL_FEATURE_NAMES)
        if args.feature_set == "full"
        else list(config["feature_sets"][args.feature_set])
    )
    central_feature_names = stored_central_features(requested_features)
    feature_indices = np.asarray([FEATURE_NAMES.index(name) for name in central_feature_names])
    print(
        f"Preparing {config['channel']} profile={config['profile_name']} components={len(config['components'])} "
        f"jets={args.jets} corrections={'on' if args.corrections else 'off'}",
        flush=True,
    )
    samples = []
    for index, spec in enumerate(config["components"]):
        print(f"[{index + 1}/{len(config['components'])}] {spec['name']}", flush=True)
        samples.append(build_component(spec, config, feature_indices, pps, parameters, args))
    return finish_dataset(
        config,
        args,
        samples,
        parameters,
        pps,
        requested_features,
        central_feature_names,
        started,
    )
