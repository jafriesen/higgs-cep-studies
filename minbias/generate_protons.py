#!/usr/bin/env python3
"""Generate inelastic Pythia events and store loose-window forward protons."""

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pythia8

from minbias.artifact import (
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    git_state,
    load_config,
    metadata_path,
    schema_with_metadata,
    sha256,
    utc_timestamp,
    validate_parquet,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, required=True, help="Accepted inelastic interactions")
    parser.add_argument("--seed", type=int, required=True, help="Pythia random seed")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--no-filter", action="store_true", help="Store every final-state proton for a control run"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.events <= 0:
        parser.error("--events must be positive")
    if not 1 <= args.seed <= 900_000_000:
        parser.error("--seed must be in [1, 900000000]")
    return args


def configure_pythia(config, seed, verbose=False):
    pythia = pythia8.Pythia("", verbose)
    settings = (
        "Beams:idA = 2212",
        "Beams:idB = 2212",
        f"Beams:eCM = {config['sqrt_s_gev']}",
        "SoftQCD:nonDiffractive = off",
        "SoftQCD:elastic = off",
        "SoftQCD:singleDiffractive = off",
        "SoftQCD:doubleDiffractive = off",
        "SoftQCD:centralDiffractive = off",
        "SoftQCD:inelastic = on",
        f"Tune:pp = {config['tune_pp']}",
        "Random:setSeed = on",
        f"Random:seed = {seed}",
        "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0",
        "Next:numberShowEvent = 0",
    )
    for setting in settings:
        if not pythia.readString(setting):
            raise RuntimeError(f"Pythia rejected setting: {setting}")
    if not verbose:
        for setting in (
            "Print:quiet = on",
            "Init:showProcesses = off",
            "Init:showChangedSettings = off",
            "Init:showChangedParticleData = off",
        ):
            pythia.readString(setting)
    if not pythia.init():
        raise RuntimeError("Pythia initialization failed")
    if pythia.settings.flag("SoftQCD:elastic"):
        raise RuntimeError("Elastic production is unexpectedly enabled")
    return pythia, settings


def generate(config, events, seed, filter_enabled=True, verbose=False):
    pythia, settings = configure_pythia(config, seed, verbose)
    columns = {name: [] for name in ("event", "arm", "xi", "px", "py", "process")}
    process_counts = Counter()
    process_names = {}
    skipped_zero_pz = 0
    written_events = 0
    failures = 0
    low, high = config["store_window"]

    while pythia.infoPython().nAccepted() < events:
        if not pythia.next():
            failures += 1
            if failures >= 1000:
                raise RuntimeError("Pythia failed 1000 consecutive event attempts")
            continue
        failures = 0
        info = pythia.infoPython()
        process = int(info.codeSub())
        process_counts[process] += 1
        process_names[process] = str(info.name())
        selected = []
        for index in range(1, pythia.event.size()):
            particle = pythia.event[index]
            if particle.id() != 2212 or not particle.isFinal():
                continue
            pz = float(particle.pz())
            if pz == 0.0:
                skipped_zero_pz += 1
                continue
            xi = np.float32(1.0 - float(particle.e()) / config["beam_energy_gev"])
            if filter_enabled and not low < xi < high:
                continue
            selected.append(
                (
                    -1 if pz < 0.0 else 1,
                    xi,
                    float(particle.px()),
                    float(particle.py()),
                    process,
                )
            )
        if selected:
            for arm, xi, px, py, code in selected:
                columns["event"].append(written_events)
                columns["arm"].append(arm)
                columns["xi"].append(xi)
                columns["px"].append(px)
                columns["py"].append(py)
                columns["process"].append(code)
            written_events += 1
        accepted = info.nAccepted()
        if verbose and (accepted <= 5 or accepted % 10000 == 0):
            print(f"accepted={accepted} written_events={written_events} protons={len(columns['xi'])}")

    info = pythia.infoPython()
    sigma_gen = float(info.sigmaGen())
    process_payload = {
        str(code): {
            "name": process_names[code],
            "events": count,
            "fraction": count / events,
            "sigma_mb": sigma_gen * count / events,
        }
        for code, count in sorted(process_counts.items())
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "n_inelastic_generated": int(info.nAccepted()),
        "n_pythia_tried": int(info.nTried()),
        "n_events_written": written_events,
        "n_protons_written": len(columns["xi"]),
        "filter_enabled": bool(filter_enabled),
        "store_window": list(config["store_window"]),
        "sqrt_s_gev": config["sqrt_s_gev"],
        "beam_energy_gev": config["beam_energy_gev"],
        "seed": seed,
        "pythia": {
            "version": float(pythia.settings.parm("Pythia:versionNumber")),
            "tune_pp": int(pythia.settings.mode("Tune:pp")),
            "settings": list(settings),
            "elastic_enabled": bool(pythia.settings.flag("SoftQCD:elastic")),
        },
        "cross_sections": {"sigma_gen_mb": sigma_gen, "processes": process_payload},
        "skipped_zero_pz_protons": skipped_zero_pz,
        "git": git_state(),
        "timestamp": utc_timestamp(),
    }
    table = pa.Table.from_pydict(columns, schema=schema_with_metadata(metadata))
    return table, metadata


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    meta_path = metadata_path(output)
    if not args.overwrite and output.exists() and meta_path.exists():
        try:
            existing = validate_parquet(output)
        except RuntimeError:
            existing = None
        if existing is not None:
            expected_filter = not args.no_filter
            if (
                existing.get("n_inelastic_generated") == args.events
                and existing.get("seed") == args.seed
                and existing.get("filter_enabled") == expected_filter
            ):
                print(f"Validated existing complete artifact for this job: {output}")
                return
            raise SystemExit(f"ERROR: complete output exists with different job settings: {output}")
    if output.exists() or meta_path.exists():
        print(f"Replacing incomplete output from an interrupted attempt: {output}")
    config = load_config(args.config)
    table, metadata = generate(
        config, args.events, args.seed, filter_enabled=not args.no_filter, verbose=args.verbose
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="snappy")
    metadata["content_sha256"] = sha256(temporary)
    metadata["parquet_compression"] = "snappy"
    os.replace(temporary, output)
    write_json(meta_path, metadata)
    print(
        f"Wrote {metadata['n_protons_written']:,} protons from "
        f"{metadata['n_inelastic_generated']:,} inelastic interactions to {output}"
    )
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
