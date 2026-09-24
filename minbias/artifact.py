#!/usr/bin/env python3
"""Shared artifact schema and metadata helpers for the minbias package."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = 1
PROTON_SCHEMA = pa.schema(
    [
        ("event", pa.int32()),
        ("arm", pa.int8()),
        ("xi", pa.float32()),
        ("px", pa.float32()),
        ("py", pa.float32()),
        ("process", pa.int16()),
    ]
)
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    try:
        sqrt_s = float(config["beam"]["sqrt_s_gev"])
        store_window = tuple(float(value) for value in config["generation"]["store_xi"])
        tune_pp = int(config["generation"]["tune_pp"])
        windows = [tuple(float(value) for value in bounds) for bounds in config["pps"]["xi_windows"]]
        xi_resolution = float(config["pps"]["xi_resolution"])
        bins = int(config["analysis"]["log_xi_bins"])
        seed = int(config["analysis"]["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid minbias configuration {path}: {exc}") from exc
    if sqrt_s <= 0.0 or len(store_window) != 2 or not 0.0 < store_window[0] < store_window[1] < 1.0:
        raise RuntimeError(f"Invalid beam energy or store window in {path}")
    if not windows or any(len(bounds) != 2 or not 0.0 < bounds[0] < bounds[1] < 1.0 for bounds in windows):
        raise RuntimeError(f"Invalid PPS xi windows in {path}")
    if xi_resolution < 0.0 or bins < 2 or seed < 0:
        raise RuntimeError(f"Invalid resolution, bins, or seed in {path}")
    return {
        "path": path.resolve(),
        "sqrt_s_gev": sqrt_s,
        "beam_energy_gev": sqrt_s / 2.0,
        "store_window": store_window,
        "tune_pp": tune_pp,
        "xi_windows": windows,
        "xi_resolution": xi_resolution,
        "log_xi_bins": bins,
        "seed": seed,
    }


def metadata_path(parquet_path):
    parquet_path = Path(parquet_path)
    if parquet_path.name == "protons.parquet":
        return parquet_path.parent / "metadata.json"
    return parquet_path.with_suffix(".metadata.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def schema_with_metadata(metadata):
    essential = {
        key: metadata[key]
        for key in (
            "schema_version",
            "n_inelastic_generated",
            "n_events_written",
            "n_protons_written",
            "store_window",
            "sqrt_s_gev",
            "beam_energy_gev",
            "seed",
            "filter_enabled",
        )
        if key in metadata
    }
    return PROTON_SCHEMA.with_metadata(
        {b"minbias_metadata": json.dumps(essential, sort_keys=True).encode("utf-8")}
    )


def validate_parquet(path, metadata=None, verify_hash=True):
    path = Path(path)
    parquet = pq.ParquetFile(path)
    fields = pa.schema([pa.field(field.name, field.type) for field in parquet.schema_arrow])
    if fields != PROTON_SCHEMA:
        raise RuntimeError(f"Unexpected proton schema in {path}: {fields}")
    metadata = metadata or read_json(metadata_path(path))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported schema version in {metadata_path(path)}")
    if int(metadata.get("n_protons_written", -1)) != parquet.metadata.num_rows:
        raise RuntimeError(f"Row count does not match metadata for {path}")
    if verify_hash and metadata.get("content_sha256") != sha256(path):
        raise RuntimeError(f"SHA-256 does not match metadata for {path}")
    return metadata
