#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_process_config,
)


SETTINGS = {
    "ebeam1": 7000,
    "ebeam2": 7000,
    "pdlabel": "lhapdf",
    "lhaid": 315000,
    "maxjetflavor": 4,
    "use_syst": False,
    "gridpack": True,
    "init_events": 1,
    "init_seed": 1101,
}


def parse_run_card(path):
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("!", 1)[0]
        if "=" not in content:
            continue
        value, name = content.split("=", 1)
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and value:
            settings[name] = value
    return settings


def metadata_fields(process, campaign, run_card):
    cfg, _settings = process_settings(process, campaign)
    settings = parse_run_card(run_card)
    cut_names = {
        "b": ("ptb", "ptbmax", "etab", "mmbb", "mmbbmax"),
        "c": ("ptj", "ptjmax", "etaj", "mmjj", "mmjjmax"),
    }[cfg["jet_type"]]
    required = (
        "ebeam1",
        "ebeam2",
        "pdlabel",
        "lhaid",
        "maxjetflavor",
        "use_syst",
        "gridpack",
        *cut_names,
    )
    missing = [name for name in required if name not in settings]
    if missing:
        raise RuntimeError(
            f"MadGraph run card '{run_card}' is missing: {', '.join(missing)}"
        )

    pt_min, pt_max, eta_max, mass_min, mass_max = cut_names
    return (
        ("field", "beam_energy_1_gev", settings["ebeam1"]),
        ("field", "beam_energy_2_gev", settings["ebeam2"]),
        ("string-field", "pdlabel", settings["pdlabel"]),
        ("field", "lhaid", settings["lhaid"]),
        ("string-field", "parton_cut_type", cfg["jet_type"]),
        ("field", "parton_pt_min_gev", settings[pt_min]),
        ("field", "parton_pt_max_gev", settings[pt_max]),
        ("field", "parton_eta_max", settings[eta_max]),
        ("field", "dijet_mass_min_gev", settings[mass_min]),
        ("field", "dijet_mass_max_gev", settings[mass_max]),
        ("field", "maxjetflavor", settings["maxjetflavor"]),
        ("field", "use_syst", settings["use_syst"]),
        ("field", "gridpack", settings["gridpack"]),
    )


def extract_lhe_run_card(path):
    card_lines = []
    in_card = False
    found_end = False
    with path.open(encoding="utf-8") as lhe:
        for line in lhe:
            line = line.rstrip("\n")
            stripped = line.strip()
            if stripped == "<MGRunCard>":
                in_card = True
                continue
            if stripped == "</MGRunCard>":
                found_end = in_card
                break
            if in_card and stripped not in ("<![CDATA[", "]]>"):
                card_lines.append(line)
    if not found_end or not any(line.strip() for line in card_lines):
        raise RuntimeError(f"LHE file '{path}' does not contain an MGRunCard")
    return "\n".join(card_lines).strip("\n") + "\n"


def write_output(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def process_settings(process, campaign):
    cfg = generation_process_config("madgraph", process)
    _campaign, campaign_cfg = generation_campaign_config(
        "madgraph", process, campaign
    )
    required = ("process_command", "jet_type", "final_pdg")
    missing = [name for name in required if cfg.get(name) is None]
    if missing:
        raise RuntimeError(
            f"MadGraph process '{process}' is missing: {', '.join(missing)}"
        )
    if cfg["jet_type"] not in ("b", "c"):
        raise RuntimeError(
            f"MadGraph process '{process}' jet_type must be 'b' or 'c'"
        )
    phase_space = campaign_cfg.get("phase_space") or {}
    required_phase_space = (
        "parton_pt_gev", "max_abs_eta", "dijet_mass_gev"
    )
    missing_phase_space = [
        name for name in required_phase_space if name not in phase_space
    ]
    if missing_phase_space:
        raise RuntimeError(
            f"MadGraph campaign '{campaign}' is missing phase_space fields: "
            f"{', '.join(missing_phase_space)}"
        )
    pt = phase_space["parton_pt_gev"]
    mass = phase_space["dijet_mass_gev"]
    if len(pt) != 2 or len(mass) != 2:
        raise RuntimeError("phase-space pT and mass settings must have two bounds")
    settings = {
        **SETTINGS,
        "pt_min": pt[0],
        "pt_max": pt[1],
        "eta_max": phase_space["max_abs_eta"],
        "mass_min": mass[0],
        "mass_max": mass[1],
    }
    return cfg, settings


def render_init_card(process, campaign, process_dir):
    cfg, settings = process_settings(process, campaign)
    if "\n" in cfg["process_command"]:
        raise RuntimeError("process_command must be a single line")

    if cfg["jet_type"] == "b":
        cuts = (
            ("ptb", settings["pt_min"]),
            ("ptbmax", settings["pt_max"]),
            ("etab", settings["eta_max"]),
            ("mmbb", settings["mass_min"]),
            ("mmbbmax", settings["mass_max"]),
        )
    else:
        cuts = (
            ("ptj", settings["pt_min"]),
            ("ptjmax", settings["pt_max"]),
            ("etaj", settings["eta_max"]),
            ("mmjj", settings["mass_min"]),
            ("mmjjmax", settings["mass_max"]),
        )

    lines = [
        "import model sm",
        "define p = g u c d s u~ c~ d~ s~",
        "define j = g u c d s u~ c~ d~ s~",
        f"generate {cfg['process_command']}"
    ]
    if cfg.get("add_process"):
        for add in cfg["add_process"]:
            lines.append(f"add process {add}")
    lines.extend(
        [
            f"output {process_dir} -f",
            f"launch {process_dir}",
            "0",
            f"set nevents {settings['init_events']}",
            f"set iseed {settings['init_seed']}",
            f"set ebeam1 {settings['ebeam1']}",
            f"set ebeam2 {settings['ebeam2']}",
            f"set pdlabel {settings['pdlabel']}",
            f"set lhaid {settings['lhaid']}",
            f"set maxjetflavor {settings['maxjetflavor']}",
        ]
    )
    lines.extend(
        f"set {name} {value}" for name, value in cuts if value is not None
    )
    lines.extend(
        [
            f"set use_syst {settings['use_syst']}",
            f"set gridpack {settings['gridpack']}",
            "done",
        ]
    )
    return "\n".join(lines) + "\n"


def initialization_key(process, campaign, madgraph_version):
    cfg, settings = process_settings(process, campaign)
    payload = {
        "madgraph_version": madgraph_version,
        "process": process,
        "campaign": campaign,
        "process_command": cfg["process_command"],
        "jet_type": cfg["jet_type"],
        "final_pdg": cfg["final_pdg"],
        "settings": settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Render a MadGraph gridpack initialization card or key."
    )
    parser.add_argument("--process", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--mode",
        choices=("card", "key", "process", "metadata", "lhe-card"),
        default="card",
    )
    parser.add_argument("--process-dir", type=Path)
    parser.add_argument("--madgraph-version")
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--lhe", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mode == "process":
        cfg, _settings = process_settings(args.process, args.campaign)
        print(cfg["process_command"])
        return

    if args.mode == "key":
        if not args.madgraph_version:
            parser.error("--madgraph-version is required with --mode key")
        print(initialization_key(args.process, args.campaign, args.madgraph_version))
        return

    if args.mode == "metadata":
        if args.run_card is None:
            parser.error("--run-card is required with --mode metadata")
        for field_type, name, value in metadata_fields(
            args.process, args.campaign, args.run_card
        ):
            print(f"{field_type}\t{name}={value}")
        return

    if args.mode == "lhe-card":
        if args.lhe is None:
            parser.error("--lhe is required with --mode lhe-card")
        card = extract_lhe_run_card(args.lhe)
        if args.output is None:
            print(card, end="")
        else:
            write_output(args.output, card)
        return

    if args.process_dir is None:
        parser.error("--process-dir is required with --mode card")
    card = render_init_card(args.process, args.campaign, args.process_dir)
    if args.output is None:
        print(card, end="")
        return
    write_output(args.output, card)


if __name__ == "__main__":
    main()
