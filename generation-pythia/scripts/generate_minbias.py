#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pythia8mc as pythia8

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.config_utils import resolve_minbias_campaign  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Pythia8 minbias events and write a flat particle parquet file."
    )
    parser.add_argument("--events", type=int, default=1000, help="Number of accepted events")
    parser.add_argument("--e-cm", type=float, default=14000.0, help="Center-of-mass energy in GeV")
    parser.add_argument(
        "--processes",
        default="SoftQCD:all",
        help='Pythia8 process switch to enable, e.g. "SoftQCD:all"',
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional Pythia random seed")
    parser.add_argument(
        "--campaign",
        "--out-tag",
        dest="campaign",
        default="minbias",
        help="Campaign name for the default output path",
    )
    parser.add_argument("-o", "--output", default=None, help="Output parquet file")
    parser.add_argument("--verbose", action="store_true", help="Print event progress")
    args = parser.parse_args()

    if args.events <= 0:
        raise ValueError("--events must be > 0")
    if args.e_cm <= 0.0:
        raise ValueError("--e-cm must be > 0")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not args.campaign:
        raise ValueError("--campaign must not be empty")
    if args.output is None:
        campaign_dir, campaign = resolve_minbias_campaign(args.campaign)
        args.output = campaign_dir / f"{campaign}.parquet"
    else:
        args.output = Path(args.output)
    return args


def configure_pythia(e_cm, processes, seed=None, verbose=False):
    os.environ.pop("PYTHIA8DATA", None)

    pythia = pythia8.Pythia("", verbose)
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString(f"Beams:eCM = {e_cm}")

    pythia.readString("SoftQCD:nonDiffractive      = off")
    pythia.readString("SoftQCD:elastic             = off")
    pythia.readString("SoftQCD:singleDiffractive   = off")
    pythia.readString("SoftQCD:doubleDiffractive   = off")
    pythia.readString("SoftQCD:centralDiffractive  = off")
    pythia.readString(f"{processes} = on")

    pythia.readString("Next:numberShowInfo = 0")
    pythia.readString("Next:numberShowProcess = 0")
    pythia.readString("Next:numberShowEvent = 0")
    if not verbose:
        pythia.readString("Print:quiet = on")
        pythia.readString("Init:showProcesses = off")
        pythia.readString("Init:showChangedSettings = off")
        pythia.readString("Init:showChangedParticleData = off")
    if seed is not None:
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")

    if not pythia.init():
        raise RuntimeError("Pythia initialization failed")
    return pythia


def empty_columns():
    return {
        "event_id": [],
        "particle_index": [],
        "pdg_id": [],
        "status": [],
        "is_final": [],
        "charge": [],
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
        "m": [],
        "pt": [],
        "eta": [],
        "phi": [],
    }


def append_event(columns, event_id, event):
    for particle_index in range(1, event.size()):
        particle = event[particle_index]
        columns["event_id"].append(event_id)
        columns["particle_index"].append(particle_index)
        columns["pdg_id"].append(int(particle.id()))
        columns["status"].append(int(particle.status()))
        columns["is_final"].append(bool(particle.isFinal()))
        columns["charge"].append(float(particle.charge()))
        columns["px"].append(float(particle.px()))
        columns["py"].append(float(particle.py()))
        columns["pz"].append(float(particle.pz()))
        columns["E"].append(float(particle.e()))
        columns["m"].append(float(particle.m()))
        columns["pt"].append(float(particle.pT()))
        columns["eta"].append(float(particle.eta()))
        columns["phi"].append(float(particle.phi()))


def write_parquet(path, columns):
    schema = pa.schema(
        [
            ("event_id", pa.int64()),
            ("particle_index", pa.int32()),
            ("pdg_id", pa.int32()),
            ("status", pa.int32()),
            ("is_final", pa.bool_()),
            ("charge", pa.float64()),
            ("px", pa.float64()),
            ("py", pa.float64()),
            ("pz", pa.float64()),
            ("E", pa.float64()),
            ("m", pa.float64()),
            ("pt", pa.float64()),
            ("eta", pa.float64()),
            ("phi", pa.float64()),
        ]
    )
    table = pa.Table.from_pydict(columns, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def generate(args):
    pythia = configure_pythia(args.e_cm, args.processes, args.seed, args.verbose)
    columns = empty_columns()

    written = 0
    consecutive_failures = 0
    max_failures = 1000
    while written < args.events:
        if not pythia.next():
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                raise RuntimeError("Too many consecutive Pythia event failures")
            continue

        consecutive_failures = 0
        written += 1
        append_event(columns, written, pythia.event)
        if args.verbose and (written <= 5 or written % 1000 == 0):
            print(f"Event {written}: pythia_entries={pythia.event.size()}")

    write_parquet(args.output, columns)
    return written, len(columns["event_id"])


def main():
    try:
        args = parse_args()
        events, particles = generate(args)
        print(f"Wrote {particles} particles from {events} events to {args.output}")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
