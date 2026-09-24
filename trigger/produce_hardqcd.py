#!/usr/bin/env python3
"""Generate an inclusive HardQCD sample and run the PU200 Delphes card."""

import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

from minbias.artifact import git_state, sha256, utc_timestamp, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "output" / "trigger" / "hardqcd_pthat10_pilot"
DEFAULT_CARD = ROOT / "sim" / "Cards" / "CMS_PhaseII_200PU_v04.tcl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--ecm-gev", type=float, default=14_000.0)
    parser.add_argument("--pthat-min-gev", type=float, default=10.0)
    parser.add_argument("--tune", type=int, default=14)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--skip-delphes", action="store_true")
    parser.add_argument(
        "--discard-hepmc",
        action="store_true",
        help="Remove HepMC only after Delphes output validates successfully",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.events <= 0 or args.ecm_gev <= 0.0 or args.pthat_min_gev < 0.0:
        parser.error(
            "event count and energy must be positive; pTHat minimum cannot be negative"
        )
    if not 1 <= args.seed <= 900_000_000:
        parser.error("--seed must be in [1, 900000000]")
    if args.skip_delphes and args.discard_hepmc:
        parser.error("--discard-hepmc cannot be used with --skip-delphes")
    return args


def build_generator():
    source = Path(__file__).with_name("hardqcd_generator.cc")
    build_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "higgs_cep_hardqcd"
    binary = build_dir / "hardqcd_generator"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    pythia = os.environ.get("PYTHIA8")
    lcg = os.environ.get("LCG_VIEW")
    if not pythia or not lcg:
        raise RuntimeError("PYTHIA8/LCG_VIEW are unset; source setup_env.sh first")
    view = lcg.rsplit("/setup.sh", 1)[0]
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        str(source),
        f"-I{pythia}/include",
        f"-I{view}/include",
        f"-L{pythia}/lib",
        f"-L{view}/lib64",
        f"-Wl,-rpath,{pythia}/lib",
        f"-Wl,-rpath,{view}/lib64",
        "-lpythia8",
        "-lHepMC3",
        "-lHepMC3search",
        "-o",
        str(binary),
    ]
    print("Compiling HardQCD generator helper...", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return binary


def read_summary(path):
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split("=", 1)
            values[key] = value
    return values


def convert_event_table(csv_path, parquet_path):
    columns = {name: [] for name in ("event_id", "pthat_gev", "weight", "process_code")}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = int(row["event_id"])
            if event_id != len(columns["event_id"]):
                raise RuntimeError("Generator event IDs are not dense and ordered")
            columns["event_id"].append(event_id)
            columns["pthat_gev"].append(float(row["pthat_gev"]))
            columns["weight"].append(float(row["weight"]))
            columns["process_code"].append(int(row["process_code"]))
    schema = pa.schema(
        [
            ("event_id", pa.int64()),
            ("pthat_gev", pa.float64()),
            ("weight", pa.float64()),
            ("process_code", pa.int32()),
        ]
    )
    pq.write_table(
        pa.Table.from_pydict(columns, schema=schema), parquet_path, compression="snappy"
    )
    return len(columns["event_id"])


def validate_delphes(path, expected):
    from trigger.jet_response import load_delphes_root

    ROOT = load_delphes_root()
    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("Delphes") if root_file else None
    if not tree or int(tree.GetEntries()) != expected:
        if root_file:
            root_file.Close()
        raise RuntimeError(f"Delphes event count does not match {expected}")
    required = ("Event", "Vertex", "GenJet", "JetPUPPI", "Particle", "EFlowTrack")
    missing = [name for name in required if not tree.GetBranch(name)]
    if not missing:
        for entry in range(expected):
            tree.GetEntry(entry)
            if not tree.Event.GetEntriesFast():
                missing.append("Event entries")
                break
            number = int(tree.Event.At(0).Number)
            if number != entry:
                missing.append("Event.Number matching the generator sidecar")
                break
    root_file.Close()
    if missing:
        raise RuntimeError(f"Missing Delphes branches: {', '.join(missing)}")


def prepare_runtime_card(card, campaign, delphes_dir):
    """Copy the card and its relative Phase-II Tcl includes together."""
    runtime_card = campaign / card.name
    shutil.copy2(card, runtime_card)
    dependency_names = set(
        re.findall(
            r"^\s*source\s+([^\s#]+)", card.read_text(encoding="utf-8"), re.MULTILINE
        )
    )
    for dependency_name in dependency_names:
        local_dependency = card.parent / dependency_name
        delphes_dependency = delphes_dir / "cards" / "CMS_PhaseII" / dependency_name
        dependency = (
            local_dependency if local_dependency.is_file() else delphes_dependency
        )
        if not dependency.is_file():
            raise RuntimeError(f"Delphes card dependency not found: {dependency_name}")
        shutil.copy2(dependency, campaign / dependency_name)
    return runtime_card


def main():
    args = parse_args()
    campaign = args.campaign_dir.resolve()
    hepmc = campaign / "hardqcd.hepmc"
    event_csv = campaign / "events.csv"
    event_parquet = campaign / "events.parquet"
    summary = campaign / "generator.summary"
    metadata_path = campaign / "metadata.json"
    delphes_output = campaign / "delphes.root"
    targets = [hepmc, event_parquet, metadata_path]
    if not args.skip_delphes:
        targets.append(delphes_output)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"ERROR: output exists (use --overwrite): {existing[0]}")
    campaign.mkdir(parents=True, exist_ok=True)
    binary = build_generator()
    print(
        f"Generating {args.events:,} HardQCD events with pTHatMin={args.pthat_min_gev:g} GeV...",
        flush=True,
    )
    subprocess.run(
        [
            str(binary),
            "--events",
            str(args.events),
            "--ecm",
            str(args.ecm_gev),
            "--pthat-min",
            str(args.pthat_min_gev),
            "--tune",
            str(args.tune),
            "--seed",
            str(args.seed),
            "--hepmc",
            str(hepmc),
            "--event-table",
            str(event_csv),
            "--summary",
            str(summary),
        ],
        cwd=ROOT,
        check=True,
    )
    n_events = convert_event_table(event_csv, event_parquet)
    values = read_summary(summary)
    if n_events != args.events or int(values["events"]) != args.events:
        raise RuntimeError("Generator output event counts are inconsistent")
    event_csv.unlink()
    summary.unlink()

    if not args.skip_delphes:
        delphes_dir = Path(
            os.environ.get("DELPHES_DIR", ROOT.parent / "delphes")
        ).resolve()
        executable = delphes_dir / "DelphesHepMC3"
        if not executable.is_file():
            raise RuntimeError(f"DelphesHepMC3 not found: {executable}")
        runtime_card = prepare_runtime_card(args.card.resolve(), campaign, delphes_dir)
        print(f"Running PU200 Delphes on {n_events:,} events...", flush=True)
        with (campaign / "delphes.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run(
                    [
                        str(executable),
                        str(runtime_card),
                        str(delphes_output),
                        str(hepmc),
                    ],
                    cwd=delphes_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Delphes failed; inspect {campaign / 'delphes.log'}"
                ) from exc
        print("Validating Delphes event alignment and required branches...", flush=True)
        validate_delphes(delphes_output, n_events)

    hepmc_sha256 = None if args.discard_hepmc else sha256(hepmc)
    if args.discard_hepmc:
        hepmc.unlink()

    metadata = {
        "schema_version": 1,
        "events": n_events,
        "sqrt_s_gev": float(args.ecm_gev),
        "pthat_min_gev": float(args.pthat_min_gev),
        "seed": int(args.seed),
        "pythia": {
            "process": "HardQCD:all",
            "tune_pp": int(args.tune),
            "n_tried": int(values["n_tried"]),
            "sigma_gen_mb": float(values["sigma_gen_mb"]),
            "sigma_err_mb": float(values["sigma_err_mb"]),
            "settings": [
                "Beams:idA = 2212",
                "Beams:idB = 2212",
                f"Beams:eCM = {args.ecm_gev}",
                "HardQCD:all = on",
                f"PhaseSpace:pTHatMin = {args.pthat_min_gev}",
                f"Tune:pp = {args.tune}",
                "Random:setSeed = on",
                f"Random:seed = {args.seed}",
            ],
        },
        "files": {
            "hepmc": {
                "path": str(hepmc) if not args.discard_hepmc else None,
                "sha256": hepmc_sha256 if not args.discard_hepmc else None,
                "retained": not args.discard_hepmc,
            },
            "events": {"path": str(event_parquet), "sha256": sha256(event_parquet)},
        },
        "delphes": {
            "path": str(delphes_output) if not args.skip_delphes else None,
            "card": str(args.card.resolve()),
            "card_sha256": sha256(args.card.resolve()),
            "runtime_card": str(runtime_card) if not args.skip_delphes else None,
            "runtime_card_sha256": sha256(runtime_card)
            if not args.skip_delphes
            else None,
            "sha256": sha256(delphes_output) if not args.skip_delphes else None,
        },
        "git": git_state(),
        "timestamp": utc_timestamp(),
    }
    write_json(metadata_path, metadata)
    print(f"Wrote HardQCD campaign to {campaign}")


if __name__ == "__main__":
    main()
