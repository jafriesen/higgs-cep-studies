#!/usr/bin/env python3
"""Verify and merge all proton shards from a minbias campaign manifest."""

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from minbias.artifact import (
    SCHEMA_VERSION,
    git_state,
    metadata_path,
    read_json,
    schema_with_metadata,
    sha256,
    utc_timestamp,
    validate_parquet,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None, help="Defaults to CAMPAIGN/protons.parquet")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compatible(reference, candidate, path):
    keys = ("schema_version", "filter_enabled", "store_window", "sqrt_s_gev", "beam_energy_gev")
    differences = [key for key in keys if candidate.get(key) != reference.get(key)]
    for key in ("version", "tune_pp", "elastic_enabled"):
        if candidate.get("pythia", {}).get(key) != reference.get("pythia", {}).get(key):
            differences.append(f"pythia.{key}")
    reference_settings = [
        setting for setting in reference.get("pythia", {}).get("settings", [])
        if not setting.startswith("Random:seed")
    ]
    candidate_settings = [
        setting for setting in candidate.get("pythia", {}).get("settings", [])
        if not setting.startswith("Random:seed")
    ]
    if candidate_settings != reference_settings:
        differences.append("pythia.physics_settings")
    if differences:
        raise RuntimeError(f"Incompatible shard metadata in {path}: {', '.join(differences)}")


def merge(manifest_path, output, overwrite=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    jobs = manifest.get("jobs") or []
    if not jobs:
        raise RuntimeError(f"Manifest has no jobs: {manifest_path}")
    if int(manifest.get("jobs_requested", -1)) != len(jobs):
        raise RuntimeError("Manifest job count does not match jobs_requested")
    indices = [int(job["index"]) for job in jobs]
    if sorted(indices) != list(range(len(jobs))):
        raise RuntimeError("Manifest job indices are not unique and contiguous")
    paths = [str(Path(job["output"]).resolve()) for job in jobs]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Manifest contains duplicate shard output paths")
    requested = int(manifest["events"])
    if sum(int(job["events"]) for job in jobs) != requested:
        raise RuntimeError("Manifest job event counts do not sum to the requested total")
    seeds = [int(job["seed"]) for job in jobs]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Manifest contains duplicate Pythia seeds")

    output = Path(output).resolve()
    out_meta = metadata_path(output)
    if not overwrite and (output.exists() or out_meta.exists()):
        raise RuntimeError(f"Merged output exists; pass --overwrite to replace it: {output}")

    tables = []
    shard_metadata = []
    event_offset = 0
    reference = None
    process_counts = Counter()
    process_names = {}
    total_generated = 0
    total_tried = 0
    total_written_events = 0
    total_protons = 0
    weighted_sigma = 0.0

    for job in jobs:
        path = Path(job["output"])
        if not path.is_file():
            raise RuntimeError(f"Missing shard for job {job['index']}: {path}")
        metadata = validate_parquet(path)
        if metadata.get("seed") != job["seed"]:
            raise RuntimeError(f"Seed mismatch for job {job['index']}: {path}")
        if metadata.get("n_inelastic_generated") != job["events"]:
            raise RuntimeError(f"Event-count mismatch for job {job['index']}: {path}")
        shard_process_total = sum(
            int(process["events"])
            for process in metadata["cross_sections"]["processes"].values()
        )
        if shard_process_total != metadata["n_inelastic_generated"]:
            raise RuntimeError(f"Process counts do not sum to the denominator in {path}")
        if reference is None:
            reference = metadata
        else:
            compatible(reference, metadata, path)

        table = pq.read_table(path)
        n_events = int(metadata["n_events_written"])
        if table.num_rows:
            events = np.asarray(table["event"], dtype=np.int64)
            unique = np.unique(events)
            if not np.array_equal(unique, np.arange(n_events)):
                raise RuntimeError(f"Shard event IDs are not dense in {path}")
            table = table.set_column(
                table.schema.get_field_index("event"),
                "event",
                pc.add(table["event"], pa.scalar(event_offset, pa.int32())),
            )
        elif n_events:
            raise RuntimeError(f"Shard records written events but has no rows: {path}")
        tables.append(table.replace_schema_metadata(None))
        event_offset += n_events
        total_written_events += n_events
        total_generated += int(metadata["n_inelastic_generated"])
        total_tried += int(metadata.get("n_pythia_tried", metadata["n_inelastic_generated"]))
        total_protons += int(metadata["n_protons_written"])
        sigma = float(metadata["cross_sections"]["sigma_gen_mb"])
        weighted_sigma += sigma * int(metadata["n_inelastic_generated"])
        for code, process in metadata["cross_sections"]["processes"].items():
            process_counts[code] += int(process["events"])
            process_names[code] = process["name"]
        shard_metadata.append(
            {
                "job": int(job["index"]),
                "path": str(path),
                "seed": int(metadata["seed"]),
                "events": int(metadata["n_inelastic_generated"]),
                "sha256": metadata["content_sha256"],
            }
        )

    if total_generated != requested:
        raise RuntimeError(f"Merged denominator {total_generated} does not equal manifest {requested}")
    if sum(process_counts.values()) != total_generated:
        raise RuntimeError("Merged process counts do not equal the merged denominator")
    if total_written_events > np.iinfo(np.int32).max:
        raise RuntimeError("Merged written-event IDs do not fit in int32")
    sigma_gen = weighted_sigma / total_generated
    process_payload = {
        str(code): {
            "name": process_names[str(code)],
            "events": count,
            "fraction": count / total_generated,
            "sigma_mb": sigma_gen * count / total_generated,
        }
        for code, count in sorted(process_counts.items(), key=lambda item: int(item[0]))
    }
    merged_pythia = dict(reference["pythia"])
    merged_pythia["settings"] = [
        setting
        for setting in merged_pythia.get("settings", [])
        if not setting.startswith("Random:seed")
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "campaign": manifest.get("campaign"),
        "n_inelastic_generated": total_generated,
        "n_pythia_tried": total_tried,
        "n_events_written": total_written_events,
        "n_protons_written": total_protons,
        "filter_enabled": reference["filter_enabled"],
        "store_window": reference["store_window"],
        "sqrt_s_gev": reference["sqrt_s_gev"],
        "beam_energy_gev": reference["beam_energy_gev"],
        "seed": None,
        "seeds": seeds,
        "pythia": merged_pythia,
        "cross_sections": {"sigma_gen_mb": sigma_gen, "processes": process_payload},
        "manifest": str(manifest_path),
        "shards": shard_metadata,
        "git": git_state(),
        "timestamp": utc_timestamp(),
    }
    merged = pa.concat_tables(tables).cast(schema_with_metadata(metadata))
    if merged.num_rows != total_protons:
        raise RuntimeError("Merged row count does not equal summed shard metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    pq.write_table(merged, temporary, compression="snappy", row_group_size=500_000)
    metadata["content_sha256"] = sha256(temporary)
    metadata["parquet_compression"] = "snappy"
    os.replace(temporary, output)
    write_json(out_meta, metadata)
    return metadata


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve() if args.output else manifest.parent / "protons.parquet"
    try:
        metadata = merge(manifest, output, args.overwrite)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(
        f"Merged {len(metadata['shards'])} shards: {metadata['n_inelastic_generated']:,} "
        f"interactions, {metadata['n_protons_written']:,} protons"
    )
    print(f"Wrote {output}")
    print(f"Wrote {metadata_path(output)}")


if __name__ == "__main__":
    main()
