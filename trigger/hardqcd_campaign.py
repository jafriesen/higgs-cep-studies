"""Validation helpers for sharded HardQCD campaigns."""

import math
from pathlib import Path

from minbias.artifact import read_json, sha256


MANIFEST_SCHEMA_VERSION = 1
SHARD_ROLES = ("derivation", "rate_validation")


def _normalized_settings(metadata):
    return tuple(
        setting
        for setting in metadata["pythia"]["settings"]
        if not setting.startswith("Random:seed")
    )


def _settings(metadata):
    settings = {}
    raw_settings = metadata.get("pythia", {}).get("settings")
    if not isinstance(raw_settings, list):
        raise RuntimeError("HardQCD Pythia settings are missing")
    for setting in raw_settings:
        if "=" not in setting:
            raise RuntimeError(f"Invalid Pythia setting: {setting!r}")
        name, value = setting.split("=", 1)
        name = name.strip()
        if name in settings:
            raise RuntimeError(f"Duplicate Pythia setting: {name}")
        settings[name] = value.strip()
    return settings


def _validate_pythia(metadata, job, configuration, metadata_path):
    pythia = metadata.get("pythia", {})
    if pythia.get("process") != "HardQCD:all":
        raise RuntimeError(f"Invalid HardQCD process in {metadata_path}")
    settings = _settings(metadata)
    try:
        valid = (
            settings["Beams:idA"] == "2212"
            and settings["Beams:idB"] == "2212"
            and float(settings["Beams:eCM"])
            == float(configuration["sqrt_s_gev"])
            and settings["HardQCD:all"] == "on"
            and float(settings["PhaseSpace:pTHatMin"])
            == float(configuration["pthat_min_gev"])
            and int(settings["Tune:pp"]) == int(configuration["tune_pp"])
            and settings["Random:setSeed"] == "on"
            and int(settings["Random:seed"]) == int(job["seed"])
            and int(pythia["n_tried"]) > 0
            and math.isfinite(float(pythia["sigma_gen_mb"]))
            and float(pythia["sigma_gen_mb"]) > 0.0
            and math.isfinite(float(pythia["sigma_err_mb"]))
            and float(pythia["sigma_err_mb"]) >= 0.0
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError(f"Invalid HardQCD Pythia metadata in {metadata_path}")


def validate_manifest(manifest, path=None):
    label = str(path) if path is not None else "HardQCD manifest"
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported HardQCD manifest schema in {label}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"HardQCD manifest has no jobs: {label}")
    if int(manifest.get("jobs_requested", -1)) != len(jobs):
        raise RuntimeError("HardQCD manifest job count is inconsistent")
    indices = [int(job["index"]) for job in jobs]
    if indices != list(range(len(jobs))):
        raise RuntimeError("HardQCD job indices must be ordered, unique, and contiguous")
    seeds = [int(job["seed"]) for job in jobs]
    if len(seeds) != len(set(seeds)) or any(
        seed < 1 or seed > 900_000_000 for seed in seeds
    ):
        raise RuntimeError("HardQCD job seeds must be unique and valid for Pythia")
    if any(job.get("role") not in SHARD_ROLES for job in jobs):
        raise RuntimeError("HardQCD manifest contains an invalid shard role")
    roles = [job["role"] for job in jobs]
    if roles != sorted(roles, key=SHARD_ROLES.index):
        raise RuntimeError("HardQCD shard roles must be phase-ordered")
    if any(int(job["events"]) <= 0 for job in jobs):
        raise RuntimeError("HardQCD shard event counts must be positive")
    total = sum(int(job["events"]) for job in jobs)
    if total != int(manifest.get("events", -1)):
        raise RuntimeError("HardQCD shard counts do not sum to total events")
    for role, field in (
        ("derivation", "derivation_events"),
        ("rate_validation", "rate_validation_events"),
    ):
        observed = sum(int(job["events"]) for job in jobs if job["role"] == role)
        if observed != int(manifest.get(field, -1)):
            raise RuntimeError(f"HardQCD {role} shard counts are inconsistent")
    directories = [str(Path(job["campaign_dir"]).resolve()) for job in jobs]
    if len(directories) != len(set(directories)):
        raise RuntimeError("HardQCD shard campaign directories are not unique")
    if path is not None:
        campaign = Path(path).resolve().parent
        expected = [
            str((campaign / "shards" / f"job_{index:05d}").resolve())
            for index in indices
        ]
        if directories != expected:
            raise RuntimeError("HardQCD shard paths conflict with the manifest")
    configuration = manifest.get("configuration")
    required_configuration = (
        "sqrt_s_gev",
        "pthat_min_gev",
        "tune_pp",
        "card_sha256",
    )
    if not isinstance(configuration, dict) or any(
        key not in configuration for key in required_configuration
    ):
        raise RuntimeError("HardQCD manifest configuration is incomplete")
    return jobs


def pool_pythia(shards):
    total_tried = sum(int(shard["metadata"]["pythia"]["n_tried"]) for shard in shards)
    if total_tried <= 0:
        raise RuntimeError("HardQCD shards report no attempted Pythia events")
    weighted_sigma = sum(
        int(shard["metadata"]["pythia"]["n_tried"])
        * float(shard["metadata"]["pythia"]["sigma_gen_mb"])
        for shard in shards
    )
    weighted_error_square = sum(
        (
            int(shard["metadata"]["pythia"]["n_tried"])
            * float(shard["metadata"]["pythia"]["sigma_err_mb"])
        )
        ** 2
        for shard in shards
    )
    reference = shards[0]["metadata"]["pythia"]
    return {
        "process": reference["process"],
        "tune_pp": int(reference["tune_pp"]),
        "n_tried": total_tried,
        "sigma_gen_mb": weighted_sigma / total_tried,
        "sigma_err_mb": math.sqrt(weighted_error_square) / total_tried,
        "settings": list(_normalized_settings(shards[0]["metadata"])),
    }


def load_campaign(campaign_dir, *, role="rate_validation", verify_hash=True):
    if role not in SHARD_ROLES:
        raise ValueError(f"Unknown HardQCD shard role: {role}")
    campaign = Path(campaign_dir).resolve()
    manifest_path = campaign / "manifest.json"
    manifest = read_json(manifest_path)
    jobs = validate_manifest(manifest, manifest_path)
    configuration = manifest["configuration"]
    shards = []
    reference = None
    for job in jobs:
        shard_dir = Path(job["campaign_dir"]).resolve()
        metadata_path = shard_dir / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing HardQCD shard metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        if int(metadata.get("schema_version", -1)) != 1:
            raise RuntimeError(f"Unsupported HardQCD shard schema: {metadata_path}")
        if int(metadata.get("events", -1)) != int(job["events"]):
            raise RuntimeError(f"HardQCD event-count mismatch in {metadata_path}")
        if int(metadata.get("seed", -1)) != int(job["seed"]):
            raise RuntimeError(f"HardQCD seed mismatch in {metadata_path}")
        manifest_compatible = (
            float(metadata.get("sqrt_s_gev", -1.0))
            == float(configuration["sqrt_s_gev"])
            and float(metadata.get("pthat_min_gev", -1.0))
            == float(configuration["pthat_min_gev"])
            and int(metadata["pythia"].get("tune_pp", -1))
            == int(configuration["tune_pp"])
            and metadata["delphes"].get("card_sha256")
            == configuration["card_sha256"]
        )
        if not manifest_compatible:
            raise RuntimeError(f"HardQCD shard conflicts with manifest: {metadata_path}")
        _validate_pythia(metadata, job, configuration, metadata_path)
        if reference is None:
            reference = metadata
        else:
            compatible = (
                metadata.get("sqrt_s_gev") == reference.get("sqrt_s_gev")
                and metadata.get("pthat_min_gev") == reference.get("pthat_min_gev")
                and metadata["pythia"].get("process")
                == reference["pythia"].get("process")
                and metadata["pythia"].get("tune_pp")
                == reference["pythia"].get("tune_pp")
                and _normalized_settings(metadata) == _normalized_settings(reference)
                and metadata["delphes"].get("card_sha256")
                == reference["delphes"].get("card_sha256")
            )
            if not compatible:
                raise RuntimeError(f"Incompatible HardQCD shard: {metadata_path}")
        event_path = Path(metadata["files"]["events"]["path"])
        root_path = Path(metadata["delphes"]["path"])
        expected_events = shard_dir / "events.parquet"
        expected_root = shard_dir / "delphes.root"
        if (
            event_path.resolve() != expected_events
            or root_path.resolve() != expected_root
        ):
            raise RuntimeError(f"HardQCD output paths conflict with {metadata_path}")
        if not event_path.is_file() or not root_path.is_file():
            raise RuntimeError(f"Missing HardQCD shard output in {shard_dir}")
        if verify_hash and sha256(event_path) != metadata["files"]["events"]["sha256"]:
            raise RuntimeError(f"HardQCD event sidecar hash mismatch: {event_path}")
        if verify_hash and sha256(root_path) != metadata["delphes"]["sha256"]:
            raise RuntimeError(f"HardQCD Delphes ROOT hash mismatch: {root_path}")
        shards.append(
            {
                "job": job,
                "metadata": metadata,
                "metadata_path": str(metadata_path),
                "event_path": event_path,
                "root_path": root_path,
            }
        )
    selected = [shard for shard in shards if shard["job"]["role"] == role]
    if not selected:
        raise RuntimeError(f"HardQCD campaign has no {role} shards")
    pooled = {
        "schema_version": int(reference["schema_version"]),
        "events": sum(int(shard["job"]["events"]) for shard in selected),
        "sqrt_s_gev": float(reference["sqrt_s_gev"]),
        "pthat_min_gev": float(reference["pthat_min_gev"]),
        "pythia": pool_pythia(shards),
        "manifest": str(manifest_path),
        "shard_role": role,
        "shards": [
            {
                "job": int(shard["job"]["index"]),
                "events": int(shard["job"]["events"]),
                "seed": int(shard["job"]["seed"]),
                "metadata": shard["metadata_path"],
            }
            for shard in selected
        ],
    }
    return {
        "campaign": campaign,
        "manifest": manifest,
        "metadata": pooled,
        "all_shards": shards,
        "shards": selected,
    }
