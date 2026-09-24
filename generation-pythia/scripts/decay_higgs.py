#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pylhe
import pythia8


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import discover_event_files  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_env,
    generation_process_config,
    generation_stage_root,
)

JET_TYPE_PDG_IDS = {"b": (5, -5), "c": (4, -4)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Pythia8 to decay the undecayed Higgs in FPMC Hadr-N LHE output, "
        "writing new LHE files with the protons unchanged and the Higgs decayed to its "
        "intended flavor. No showering, hadronization, ISR/FSR, or MPI is performed."
    )
    parser.add_argument("--process", required=True, help="FPMC process name from processes-fpmc.yaml.")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Main FPMC campaign name. Defaults to the process's default_campaign.main.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="parton-Pythia subcampaign tag (destination folder name).",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap per output file")
    parser.add_argument("--max-files", type=int, default=None, help="Optional input FPMC file cap")
    parser.add_argument("--seed", type=int, default=None, help="Optional Pythia seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LHE outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without running Pythia")
    parser.add_argument("--verbose", action="store_true", help="Print Pythia8 initialization/run info")
    return parser.parse_args()


def decay_pdg_pair(process_cfg):
    jet_type = process_cfg.get("jet_type")
    if jet_type not in JET_TYPE_PDG_IDS:
        raise RuntimeError(
            f"Unsupported jet_type '{jet_type}' for Higgs decay; supported: {sorted(JET_TYPE_PDG_IDS)}"
        )
    return JET_TYPE_PDG_IDS[jet_type]


def configure_pythia(pdg1, pdg2, seed, verbose):
    """Standalone-decay setup (mirrors Pythia8's own main21.cc "single-resonance gun"
    example, idGun=25): with ProcessLevel off, appending a status-1 resonance and calling
    next() decays it via the normal decay-table machinery. HadronLevel stays on by default
    though, so string fragmentation and resonance-decay showering of the decay products
    must be switched off explicitly.
    """
    pythia = pythia8.Pythia("", verbose)
    pythia.readString("ProcessLevel:all = off")
    pythia.readString("Check:event = off")
    pythia.readString("HadronLevel:Hadronize = off")
    pythia.readString("PartonLevel:FSRinResonances = off")
    pythia.readString("25:onMode = off")
    pythia.readString(f"25:onIfMatch = {pdg1} {pdg2}")
    if not verbose:
        pythia.readString("Print:quiet = on")
        pythia.readString("Init:showProcesses = off")
        pythia.readString("Init:showChangedSettings = off")
        pythia.readString("Init:showChangedParticleData = off")
    pythia.readString("Next:numberShowInfo = 0")
    pythia.readString("Next:numberShowProcess = 0")
    pythia.readString("Next:numberShowEvent = 0")
    if seed is not None:
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")
    if not pythia.init():
        raise RuntimeError("Pythia initialization failed")
    return pythia


def decay_higgs(pythia, higgs):
    pythia.event.reset()
    pythia.event.append(25, 1, 0, 0, higgs.px, higgs.py, higgs.pz, higgs.e, higgs.m)
    if not pythia.next():
        raise RuntimeError("Pythia failed to decay the Higgs")
    return [pythia.event[i] for i in range(pythia.event.size()) if pythia.event[i].isFinal()]


def decay_event(pythia, event, input_file, event_index):
    higgs = next((p for p in event.particles if p.id == 25), None)
    if higgs is None:
        raise RuntimeError(f"No Higgs (PDG 25) found in event {event_index} of {input_file}")
    others = [p for p in event.particles if p.id != 25]

    decay_particles = [
        pylhe.LHEParticle(
            id=d.id(), status=1, mother1=0, mother2=0, color1=d.col(), color2=d.acol(),
            px=d.px(), py=d.py(), pz=d.pz(), e=d.e(), m=d.m(), lifetime=0.0, spin=9.0,
        )
        for d in decay_higgs(pythia, higgs)
    ]
    particles = others + decay_particles

    eventinfo = event.eventinfo
    new_eventinfo = pylhe.LHEEventInfo(
        nparticles=len(particles), pid=eventinfo.pid, weight=eventinfo.weight,
        scale=eventinfo.scale, aqed=eventinfo.aqed, aqcd=eventinfo.aqcd,
    )
    return pylhe.LHEEvent(eventinfo=new_eventinfo, particles=particles, weights={}, attributes=event.attributes)


def decay_file(pythia, input_file, output_file, max_events):
    lhefile = pylhe.LHEFile.fromfile(str(input_file))
    events = []
    for index, event in enumerate(lhefile.events, start=1):
        if max_events is not None and len(events) >= max_events:
            break
        events.append(decay_event(pythia, event, input_file, index))

    if not events:
        raise RuntimeError(f"No events were written from {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pylhe.LHEFile(init=lhefile.init, events=events).tofile(str(output_file))
    print(f"Wrote {len(events)} events from {input_file} to {output_file}")


def write_metadata(root, args, campaign_name, process_cfg, pdg1, pdg2, input_files, output_dir):
    metadata_writer = root / "common" / "write_metadata.py"
    command = [
        "python3", str(metadata_writer),
        "--output", str(output_dir.parent / "metadata.yaml"),
        "--string-field", "generator=fpmc",
        "--string-field", f"process={args.process}",
        "--string-field", f"campaign={campaign_name}",
        "--string-field", f"tag={args.tag}",
        "--string-field", "stage=parton-pythia",
        "--string-field", f"jet_type={process_cfg['jet_type']}",
        "--field", f"decay_pdg_ids={json.dumps([pdg1, pdg2])}",
        "--field", f"n_files={len(input_files)}",
        "--field", f"source_files={json.dumps([str(f) for f in input_files])}",
        "--string-field", f"command={shlex.join(sys.argv)}",
        "--string-field", f"created_at={datetime.now().isoformat()}",
    ]
    if args.seed is not None:
        command.extend(["--field", f"seed={args.seed}"])
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    root = ROOT

    process_cfg = generation_process_config("fpmc", args.process)
    pdg1, pdg2 = decay_pdg_pair(process_cfg)
    campaign_name, _ = generation_campaign_config("fpmc", args.process, args.campaign)

    env = generation_env("fpmc", args.process, campaign_name)
    input_dir = env["EVENT_RECORDS_DIR"]
    output_dir = (
        generation_stage_root("fpmc", args.process, campaign_name, "parton-pythia", subcampaign=args.tag)
        / "lhe"
    )

    input_files = discover_event_files(input_dir, max_files=args.max_files)

    jobs = []
    for index, input_file in enumerate(input_files, start=1):
        output_file = output_dir / f"{args.process}_{args.tag}_{index}.lhe"
        if output_file.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_file}")
            continue
        jobs.append((input_file, output_file))

    if not jobs:
        return

    if args.dry_run:
        for input_file, output_file in jobs:
            print(f"{input_file} -> {output_file}")
        return

    pythia = configure_pythia(pdg1, pdg2, args.seed, args.verbose)
    for input_file, output_file in jobs:
        decay_file(pythia, input_file, output_file, args.max_events)

    write_metadata(root, args, campaign_name, process_cfg, pdg1, pdg2, input_files, output_dir)


if __name__ == "__main__":
    main()
