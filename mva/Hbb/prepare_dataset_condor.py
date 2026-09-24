#!/usr/bin/env python3
"""Build an H(bb) dataset as resumable HTCondor file shards."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from analysis.cross_sections import generator_cross_section_fb, generator_weight  # noqa: E402
from mva.common.config import load_channel_config  # noqa: E402
from mva.common.dataset import (  # noqa: E402
    PARTON_FIELDS,
    PROTON_TRANSVERSE_FIELDS,
    _central_block,
    _file_piece,
    _finish_component,
    _madgraph_inputs,
    _phase_space_mask,
    _superchic_inputs,
    component_truth_pid_abs,
    correction_source_sample,
    finish_dataset,
    load_component_correction_map,
    write_yaml,
)
from mva.common.features import (  # noqa: E402
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    stored_central_features,
)
from mva.common.protons import load_pps_config  # noqa: E402
from mva.common.weights import stitched_cross_section_weights  # noqa: E402
from common.jet_calibration import load_correction_map  # noqa: E402


FORMAT_VERSION = 2
BASE_ARRAYS = (
    "x",
    "group_id",
    "dijet_rapidity",
    "dijet_mass",
    "truth_matched",
    "truth_is_leading",
    "match_dr1",
    "match_dr2",
    "correction_valid",
    "parton",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _requested_features(config, feature_set):
    return (
        list(ALL_FEATURE_NAMES)
        if feature_set == "full"
        else list(config["feature_sets"][feature_set])
    )


def _dependency_paths(config):
    paths = {
        Path(config["path"]),
        REPO / config["parameters"],
        REPO / config["pps_config"],
        Path(__file__).resolve(),
        REPO / "mva/common/dataset.py",
        REPO / "mva/common/features.py",
        REPO / "mva/common/color_flow_geometry.py",
        REPO / "mva/common/protons.py",
        REPO / "mva/common/weights.py",
        REPO / "common/jet_calibration.py",
    }
    for spec in config["components"]:
        if spec.get("correction_map"):
            paths.add(REPO / spec["correction_map"])
    return sorted(path.resolve() for path in paths)


def _file_record(path, seed, proton_kind=None, proton_path=None, proton_offset=0):
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seed": int(seed),
        "proton_kind": proton_kind,
        "proton_path": str(proton_path.resolve()) if proton_path else None,
        "proton_offset": int(proton_offset),
    }
    if proton_path:
        proton_stat = proton_path.stat()
        record["proton_size"] = proton_stat.st_size
        record["proton_mtime_ns"] = proton_stat.st_mtime_ns
    return record


def _build_manifest(args):
    config = load_channel_config(SCRIPT_DIR / "config.yaml", args.profile)
    profile = config["profile"]
    feature_set = args.feature_set or profile.get("feature_set", "locked")
    requested = _requested_features(config, feature_set)
    central = stored_central_features(requested)
    missing = [name for name in central if name not in FEATURE_NAMES]
    if missing:
        raise RuntimeError(f"Configured features are not implemented: {missing}")

    settings = {
        "tree": args.tree,
        "collection": args.collection,
        "jets": args.jets or profile["jets"],
        "corrections": (
            bool(profile["corrections"])
            if args.corrections is None else args.corrections == "on"
        ),
        "track_min_pt": float(
            args.track_min_pt
            if args.track_min_pt is not None else profile["track_min_pt_gev"]
        ),
        "max_abs_jet_eta": profile.get("max_abs_jet_eta"),
        "mass_window": [float(value) for value in config["mass_window_gev"]],
        "max_delta_y": float(config["max_abs_rapidity_difference"]),
        "seed": int(args.seed),
        "skip_hash": bool(args.skip_hash),
        "intensity_chunk": int(args.intensity_chunk),
        "store_parton": bool(args.store_parton),
    }
    groups = []
    for spec in config["components"]:
        seed_base = settings["seed"] + spec["global_id"] * 10000
        if spec["generator"] == "superchic":
            campaign, subcampaign, input_dir, files, kind, proton_files, offsets = (
                _superchic_inputs(spec, args.max_files)
            )
            xsec_fb, xsec_source = generator_cross_section_fb(
                "superchic", spec["process"], campaign
            )
            correction_source = correction_source_sample(spec, campaign)
            if settings["corrections"]:
                load_component_correction_map(REPO, spec, campaign)
            records = [
                _file_record(path, seed_base + index, kind, proton_path, offset)
                for index, (path, proton_path, offset) in enumerate(
                    zip(files, proton_files, offsets)
                )
            ]
            groups.append(
                {
                    "index": len(groups),
                    "component": spec["name"],
                    "generator": "superchic",
                    "campaign": campaign,
                    "subcampaign": subcampaign,
                    "input_dir": str(input_dir.resolve()),
                    "phase": None,
                    "xsec_fb": float(xsec_fb),
                    "xsec_source": xsec_source,
                    "correction_source_sample": (
                        correction_source if settings["corrections"] else None
                    ),
                    "records": records,
                }
            )
        else:
            for campaign in spec["campaigns"]:
                phase, input_dir, files = _madgraph_inputs(spec, campaign, args.max_files)
                xsec_fb, xsec_source = generator_cross_section_fb(
                    "madgraph", spec["process"], campaign
                )
                correction_source = correction_source_sample(spec, campaign)
                if settings["corrections"]:
                    load_component_correction_map(REPO, spec, campaign)
                records = [
                    _file_record(path, seed_base + index)
                    for index, path in enumerate(files)
                ]
                groups.append(
                    {
                        "index": len(groups),
                        "component": spec["name"],
                        "generator": "madgraph",
                        "campaign": campaign,
                        "subcampaign": input_dir.parent.name,
                        "input_dir": str(input_dir.resolve()),
                        "phase": phase,
                        "xsec_fb": float(xsec_fb),
                        "xsec_source": xsec_source,
                        "correction_source_sample": (
                            correction_source if settings["corrections"] else None
                        ),
                        "records": records,
                    }
                )

    shards = []
    for group in groups:
        for start in range(0, len(group["records"]), args.files_per_job):
            shards.append(
                {
                    "index": len(shards),
                    "group": group["index"],
                    "start": start,
                    "stop": min(start + args.files_per_job, len(group["records"])),
                }
            )
    dependencies = {
        str(path): _sha256(path)
        for path in _dependency_paths(config)
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "repo": str(REPO),
        "config_path": str(Path(config["path"]).resolve()),
        "profile": args.profile,
        "feature_set": feature_set,
        "requested_features": requested,
        "central_features": central,
        "data_dir": str(args.data_dir.resolve()),
        "settings": settings,
        "dependencies": dependencies,
        "groups": groups,
        "shards": shards,
    }
    payload["manifest_hash"] = _manifest_digest(payload)
    return payload


def _load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"Unsupported shard manifest format: {manifest.get('format_version')}")
    expected = manifest.get("manifest_hash")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    if expected != _manifest_digest(unsigned):
        raise RuntimeError("Shard manifest hash does not match its contents")
    return manifest


def _verify_dependencies(manifest):
    changed = []
    for name, expected in manifest["dependencies"].items():
        path = Path(name)
        if not path.is_file() or _sha256(path) != expected:
            changed.append(name)
    if changed:
        raise RuntimeError(f"Dependencies changed after manifest creation: {changed}")


def _verify_record(record):
    path = Path(record["path"])
    stat = path.stat()
    if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
        raise RuntimeError(f"Input ROOT file changed after planning: {path}")
    if record["proton_path"]:
        proton = Path(record["proton_path"])
        proton_stat = proton.stat()
        if (
            proton_stat.st_size != record["proton_size"]
            or proton_stat.st_mtime_ns != record["proton_mtime_ns"]
        ):
            raise RuntimeError(f"Proton record changed after planning: {proton}")


def _empty_block(pieces, columns):
    return {
        "x": np.empty((0, columns), dtype=np.float32),
        "group_id": np.empty(0, dtype=np.int64),
        "dijet_rapidity": np.empty(0),
        "dijet_mass": np.empty(0),
        "truth_matched": np.empty(0, dtype=bool),
        "truth_is_leading": np.empty(0, dtype=bool),
        "match_dr1": np.empty(0),
        "match_dr2": np.empty(0),
        "correction_valid": np.empty(0, dtype=bool),
        "parton": np.empty((0, len(PARTON_FIELDS))),
        "n_generated": sum(piece["n_generated"] for piece in pieces),
        "correction_candidates": sum(
            piece.get("correction_candidate_events", 0) for piece in pieces
        ),
        "correction_invalid": sum(
            piece.get("correction_invalid_events", 0) for piece in pieces
        ),
    }


def _shard_path(manifest_path, shard_index):
    return manifest_path.parent / "shards" / f"shard_{shard_index:05d}.npz"


def _read_shard_metadata(path):
    with np.load(path, allow_pickle=False) as payload:
        return json.loads(str(payload["__metadata__"].item()))


def _valid_existing_shard(path, manifest, shard_index):
    if not path.is_file():
        return False
    try:
        metadata = _read_shard_metadata(path)
    except Exception:
        return False
    return (
        metadata.get("manifest_hash") == manifest["manifest_hash"]
        and metadata.get("shard_index") == shard_index
    )


def worker(args):
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    _verify_dependencies(manifest)
    if args.shard_index < 0 or args.shard_index >= len(manifest["shards"]):
        raise ValueError(f"Invalid shard index {args.shard_index}")
    output = _shard_path(manifest_path, args.shard_index)
    if _valid_existing_shard(output, manifest, args.shard_index):
        print(f"Shard {args.shard_index} already complete: {output}")
        return

    shard = manifest["shards"][args.shard_index]
    group = manifest["groups"][shard["group"]]
    records = group["records"][shard["start"]:shard["stop"]]
    for record in records:
        _verify_record(record)
    config = load_channel_config(manifest["config_path"], manifest["profile"])
    spec = next(item for item in config["components"] if item["name"] == group["component"])
    settings = manifest["settings"]
    correction_map = None
    if settings["corrections"]:
        correction_map = load_correction_map(
            REPO / spec["correction_map"],
            spec["fsr_state"],
            group["correction_source_sample"],
        )
    pps = load_pps_config(REPO / config["pps_config"])
    truth_pid_abs = component_truth_pid_abs(spec)
    tasks = [
        (
            Path(record["path"]),
            record["proton_kind"],
            Path(record["proton_path"]) if record["proton_path"] else None,
            record["proton_offset"],
            settings["tree"],
            settings["collection"],
            truth_pid_abs,
            settings["jets"],
            correction_map,
            settings["track_min_pt"],
            settings["max_abs_jet_eta"],
            pps,
            record["seed"],
            tuple(settings["mass_window"]),
            settings["max_delta_y"],
        )
        for record in records
    ]
    started = time.perf_counter()
    pieces = []
    for index, task in enumerate(tasks, start=1):
        pieces.append(_file_piece(task))
        print(
            f"shard {args.shard_index}: file {index}/{len(tasks)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    feature_indices = np.asarray(
        [FEATURE_NAMES.index(name) for name in manifest["central_features"]]
    )
    if any(not piece.get("empty") for piece in pieces):
        block = _central_block(pieces, feature_indices)
    else:
        block = _empty_block(pieces, len(feature_indices))
    if spec["real_protons"]:
        for name in ("proton_mx", "proton_yx", *PROTON_TRANSVERSE_FIELDS):
            values = [piece[name] for piece in pieces if not piece.get("empty")]
            block[name] = np.concatenate(values) if values else np.empty(0)
    metadata = {
        "manifest_hash": manifest["manifest_hash"],
        "shard_index": args.shard_index,
        "group_index": group["index"],
        "component": group["component"],
        "campaign": group["campaign"],
        "start": shard["start"],
        "stop": shard["stop"],
        "files": len(records),
        "generated": int(block["n_generated"]),
        "selected": int(block["x"].shape[0]),
        "correction_candidates": int(block["correction_candidates"]),
        "correction_invalid": int(block["correction_invalid"]),
        "correction_source_sample": group["correction_source_sample"],
        "central_before_protons": int(
            sum(piece.get("n_central_before_protons", 0) for piece in pieces)
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    arrays = {name: block[name] for name in BASE_ARRAYS}
    if spec["real_protons"]:
        arrays.update(
            {name: block[name] for name in ("proton_mx", "proton_yx", *PROTON_TRANSVERSE_FIELDS)}
        )
    arrays["__metadata__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Wrote shard {args.shard_index}: {output}", flush=True)


def _load_group_block(manifest_path, manifest, group):
    shards = [item for item in manifest["shards"] if item["group"] == group["index"]]
    arrays = {name: [] for name in BASE_ARRAYS}
    if group["generator"] == "superchic":
        for name in ("proton_mx", "proton_yx", *PROTON_TRANSVERSE_FIELDS):
            arrays[name] = []
    metadata = []
    group_offset = 0
    for shard in shards:
        path = _shard_path(manifest_path, shard["index"])
        if not _valid_existing_shard(path, manifest, shard["index"]):
            raise RuntimeError(f"Missing or invalid shard: {path}")
        with np.load(path, allow_pickle=False) as payload:
            item = json.loads(str(payload["__metadata__"].item()))
            metadata.append(item)
            for name in arrays:
                values = np.asarray(payload[name])
                if name == "group_id":
                    _unique, values = np.unique(values, return_inverse=True)
                    values = values.astype(np.int64) + group_offset
                    group_offset += _unique.size
                arrays[name].append(values)
    block = {}
    for name, values in arrays.items():
        if name in ("x", "parton"):
            block[name] = np.vstack(values)
        else:
            block[name] = np.concatenate(values)
    block["n_generated"] = sum(item["generated"] for item in metadata)
    block["correction_candidates"] = sum(
        item["correction_candidates"] for item in metadata
    )
    block["correction_invalid"] = sum(item["correction_invalid"] for item in metadata)
    block["central_before_protons"] = sum(
        item["central_before_protons"] for item in metadata
    )
    block["shard_runtime_seconds"] = sum(item["runtime_seconds"] for item in metadata)
    return block


def _build_samples_from_shards(manifest_path, manifest, config, parameters):
    loaded_groups = {
        group["index"]: _load_group_block(manifest_path, manifest, group)
        for group in manifest["groups"]
    }
    samples = []
    for spec in config["components"]:
        started = time.perf_counter()
        groups = [group for group in manifest["groups"] if group["component"] == spec["name"]]
        campaign_blocks = []
        input_metadata = {}
        if spec["generator"] == "superchic":
            if len(groups) != 1:
                raise RuntimeError(f"Expected one SuperChic group for {spec['name']}")
            group = groups[0]
            block = loaded_groups[group["index"]]
            block["cross_section_weight"] = np.full(
                block["x"].shape[0],
                group["xsec_fb"]
                * generator_weight("superchic", spec["process"])
                / block["n_generated"],
            )
            block["campaign_name"] = group["campaign"]
            campaign_blocks.append(block)
            input_metadata[group["campaign"]] = {
                "subcampaign": group["subcampaign"],
                "input_dir": group["input_dir"],
                "files": len(group["records"]),
                "generated": block["n_generated"],
                "central_before_protons": block["central_before_protons"],
                "selected": int(block["x"].shape[0]),
                "xsec_fb": group["xsec_fb"],
                "xsec_source": group["xsec_source"],
                "proton_record_type": group["records"][0]["proton_kind"],
                "correction_source_sample": group["correction_source_sample"],
            }
        else:
            campaign_inputs = [
                (
                    group["campaign"],
                    group["phase"],
                    group,
                    loaded_groups[group["index"]],
                )
                for group in groups
            ]
            if spec.get("campaign_combination") == "stitch":
                luminosities = {
                    campaign: block["n_generated"] / group["xsec_fb"]
                    for campaign, _phase, group, block in campaign_inputs
                }
                phases = {
                    campaign: phase for campaign, phase, _group, _block in campaign_inputs
                }
                for campaign, phase, group, block in campaign_inputs:
                    masks = {
                        name: _phase_space_mask(block["parton"], item)
                        for name, item in phases.items()
                    }
                    weights = stitched_cross_section_weights(campaign, masks, luminosities)
                    keep = weights > 0.0
                    for name, values in list(block.items()):
                        if isinstance(values, np.ndarray) and values.shape[:1] == keep.shape:
                            block[name] = values[keep]
                    block["cross_section_weight"] = weights[keep] * generator_weight(
                        "madgraph", spec["process"]
                    )
                    combination = "stitched"
                    selected = int(np.sum(keep))
                    block["campaign_name"] = campaign
                    campaign_blocks.append(block)
                    input_metadata[campaign] = {
                        "subcampaign": group["subcampaign"],
                        "input_dir": group["input_dir"],
                        "files": len(group["records"]),
                        "generated": block["n_generated"],
                        "selected": selected,
                        "xsec_fb": group["xsec_fb"],
                        "xsec_source": group["xsec_source"],
                        "combination": combination,
                        "correction_source_sample": group["correction_source_sample"],
                    }
            else:
                for campaign, phase, group, block in campaign_inputs:
                    block["cross_section_weight"] = np.full(
                        block["x"].shape[0],
                        group["xsec_fb"]
                        * generator_weight("madgraph", spec["process"])
                        / block["n_generated"],
                    )
                    block["campaign_name"] = campaign
                    campaign_blocks.append(block)
                    input_metadata[campaign] = {
                        "subcampaign": group["subcampaign"],
                        "input_dir": group["input_dir"],
                        "files": len(group["records"]),
                        "generated": block["n_generated"],
                        "selected": int(block["x"].shape[0]),
                        "xsec_fb": group["xsec_fb"],
                        "xsec_source": group["xsec_source"],
                        "combination": "disjoint",
                        "correction_source_sample": group["correction_source_sample"],
                    }
        samples.append(
            _finish_component(
                spec, config, campaign_blocks, input_metadata, parameters, started
            )
        )
    return samples


def merge(args):
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    _verify_dependencies(manifest)
    missing = [
        shard["index"]
        for shard in manifest["shards"]
        if not _valid_existing_shard(
            _shard_path(manifest_path, shard["index"]), manifest, shard["index"]
        )
    ]
    if missing:
        raise RuntimeError(f"Cannot merge; missing or invalid shards: {missing[:20]}")
    data_dir = Path(manifest["data_dir"])
    if (data_dir / "metadata.yaml").is_file():
        existing = yaml.safe_load((data_dir / "metadata.yaml").read_text(encoding="utf-8"))
        if (existing.get("sharding") or {}).get("manifest_hash") == manifest["manifest_hash"]:
            print(f"Dataset already merged: {data_dir}")
            return
    if data_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset directory: {data_dir}")
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    config = load_channel_config(manifest["config_path"], manifest["profile"])
    with open(REPO / config["parameters"], encoding="utf-8") as handle:
        parameters = yaml.safe_load(handle)
    pps = load_pps_config(REPO / config["pps_config"])
    started = time.perf_counter()
    samples = _build_samples_from_shards(manifest_path, manifest, config, parameters)
    settings = manifest["settings"]
    staging = data_dir.parent / f".{data_dir.name}.merge_staging"
    if staging.exists():
        marker = staging / ".manifest_hash"
        if not marker.is_file() or marker.read_text().strip() != manifest["manifest_hash"]:
            raise RuntimeError(f"Unrecognized merge staging directory: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / ".manifest_hash").write_text(manifest["manifest_hash"], encoding="utf-8")
    dataset_args = SimpleNamespace(
        repo=REPO,
        data_dir=staging,
        feature_set=manifest["feature_set"],
        jets=settings["jets"],
        corrections=settings["corrections"],
        max_abs_jet_eta=settings["max_abs_jet_eta"],
        track_min_pt=settings["track_min_pt"],
        mass_window=tuple(settings["mass_window"]),
        max_delta_y=settings["max_delta_y"],
        seed=settings["seed"],
        skip_hash=settings["skip_hash"],
        intensity_chunk=settings["intensity_chunk"],
        store_parton=settings["store_parton"],
    )
    metadata = finish_dataset(
        config,
        dataset_args,
        samples,
        parameters,
        pps,
        manifest["requested_features"],
        manifest["central_features"],
        started,
    )
    metadata["sharding"] = {
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "shards": len(manifest["shards"]),
    }
    write_yaml(staging / "metadata.yaml", metadata)
    (staging / ".manifest_hash").unlink()
    os.replace(staging, data_dir)
    print(f"Merged dataset: {data_dir}")


def status(args):
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    complete = []
    for shard in manifest["shards"]:
        if _valid_existing_shard(
            _shard_path(manifest_path, shard["index"]), manifest, shard["index"]
        ):
            complete.append(shard["index"])
    print(f"Complete shards: {len(complete)}/{len(manifest['shards'])}")
    if len(complete) != len(manifest["shards"]):
        done = set(complete)
        missing = [item["index"] for item in manifest["shards"] if item["index"] not in done]
        print("Missing:", " ".join(map(str, missing[:100])))


def _write_condor_files(shard_dir, manifest_path, manifest, args):
    logs = shard_dir / "logs"
    logs.mkdir(exist_ok=True)
    python_user_base = Path.home().resolve() / ".local"
    queue = shard_dir / "queue_items.txt"
    queue.write_text(
        "".join(f"{item['index']}\n" for item in manifest["shards"]),
        encoding="utf-8",
    )
    runner = shard_dir / "run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO}",
                f"export PYTHONUSERBASE={python_user_base}",
                "source env/setup_superchic.sh",
                "source env/setup_delphes.sh",
                "python3 -c 'import awkward, fastjet, numpy, pyarrow, uproot, vector, yaml'",
                f"exec python3 {Path(__file__).resolve()} \"$@\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    worker_submit = shard_dir / "worker.sub"
    worker_submit.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {runner}",
                f"arguments = worker --manifest {manifest_path} --shard-index $(SHARD_INDEX)",
                "should_transfer_files = NO",
                f"output = {logs}/shard_$(SHARD_INDEX).out",
                f"error = {logs}/shard_$(SHARD_INDEX).err",
                f"log = {logs}/worker.log",
                f"request_memory = {args.request_memory}",
                "request_cpus = 1",
                f"max_materialize = {args.max_idle}",
                f"max_idle = {args.max_idle}",
                f"queue SHARD_INDEX from {queue}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    merge_submit = shard_dir / "merge.sub"
    merge_submit.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {runner}",
                f"arguments = merge --manifest {manifest_path}",
                "should_transfer_files = NO",
                f"output = {logs}/merge.out",
                f"error = {logs}/merge.err",
                f"log = {logs}/merge.log",
                f"request_memory = {args.merge_memory}",
                "request_cpus = 1",
                "queue 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dag = shard_dir / "dataset.dag"
    dag.write_text(
        "\n".join(
            [
                f"JOB SHARDS {worker_submit}",
                f"JOB MERGE {merge_submit}",
                "PARENT SHARDS CHILD MERGE",
                "RETRY SHARDS 2",
                "RETRY MERGE 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dag


def submit(args):
    shard_dir = args.shard_dir.resolve()
    manifest_path = shard_dir / "manifest.yaml"
    if args.resume:
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume without a manifest: {manifest_path}")
        manifest = _load_manifest(manifest_path)
    else:
        if shard_dir.exists() and any(shard_dir.iterdir()):
            raise RuntimeError(
                f"Shard directory is not empty; use --resume or choose another path: {shard_dir}"
            )
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest = _build_manifest(args)
        write_yaml(manifest_path, manifest)
    dag = _write_condor_files(shard_dir, manifest_path, manifest, args)
    print(
        f"Planned {len(manifest['shards'])} shards across {len(manifest['groups'])} groups\n"
        f"Manifest: {manifest_path}\nDAG: {dag}\nDataset: {manifest['data_dir']}"
    )
    if args.dry_run:
        print("Dry run requested; not submitting.")
        return
    if shutil.which("condor_submit_dag") is None:
        raise RuntimeError("condor_submit_dag is not available")
    subprocess.run(["condor_submit_dag", str(dag)], cwd=shard_dir, check=True)


def _parser():
    document = yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit", help="Plan and submit the shard DAG")
    submit_parser.add_argument("--profile", default="fsr_mtd_four_class", choices=document["profiles"])
    submit_parser.add_argument(
        "--feature-set", default="full", choices=(*document["feature_sets"], "full")
    )
    submit_parser.add_argument("--data-dir", type=Path)
    submit_parser.add_argument("--shard-dir", type=Path)
    submit_parser.add_argument("--files-per-job", type=int, default=50)
    submit_parser.add_argument("--max-files", type=int)
    submit_parser.add_argument("--tree", default="Delphes")
    submit_parser.add_argument("--collection", default="JetPUPPI")
    submit_parser.add_argument("--jets", choices=("truth", "leading"))
    submit_parser.add_argument("--corrections", choices=("on", "off"))
    submit_parser.add_argument("--track-min-pt", type=float)
    submit_parser.add_argument("--seed", type=int, default=12345)
    submit_parser.add_argument("--intensity-chunk", type=int, default=500000)
    submit_parser.add_argument("--store-parton", action="store_true")
    submit_parser.add_argument("--skip-hash", action="store_true")
    submit_parser.add_argument("--request-memory", type=int, default=4096)
    submit_parser.add_argument("--merge-memory", type=int, default=32768)
    submit_parser.add_argument("--max-idle", type=int, default=100)
    submit_parser.add_argument("--resume", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.set_defaults(handler=submit)
    for name, handler in (("worker", worker), ("status", status), ("merge", merge)):
        subparser = commands.add_parser(name)
        subparser.add_argument("--manifest", type=Path, required=True)
        if name == "worker":
            subparser.add_argument("--shard-index", type=int, required=True)
        subparser.set_defaults(handler=handler)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.command == "submit":
        if args.files_per_job <= 0:
            parser.error("--files-per-job must be positive")
        if args.max_files is not None and args.max_files <= 0:
            parser.error("--max-files must be positive")
        if args.request_memory <= 0 or args.merge_memory <= 0 or args.max_idle <= 0:
            parser.error("memory and idle limits must be positive")
        args.data_dir = (
            args.data_dir.resolve()
            if args.data_dir
            else SCRIPT_DIR / "data" / f"{args.profile}_{args.feature_set}"
        )
        args.shard_dir = (
            args.shard_dir.resolve()
            if args.shard_dir
            else SCRIPT_DIR / "condor" / f"{args.profile}_{args.feature_set}"
        )
    args.handler(args)


if __name__ == "__main__":
    main()
