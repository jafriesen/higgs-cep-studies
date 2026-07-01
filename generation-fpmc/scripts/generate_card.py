#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_process_config,
)


def render_card(process, campaign, events, seed, lhefile="FPMC.lhe"):
    if events <= 0:
        raise RuntimeError("events must be a positive integer")
    if seed < 0:
        raise RuntimeError("seed must be a non-negative integer")
    if not lhefile or "'" in lhefile or "\n" in lhefile:
        raise RuntimeError("LHE filename must be nonempty and cannot contain quotes or newlines")

    process_cfg = generation_process_config("fpmc", process)
    _, campaign_cfg = generation_campaign_config("fpmc", process, campaign)
    hadronize = campaign_cfg.get("hadronize", True)
    if not isinstance(hadronize, bool):
        raise RuntimeError(
            f"FPMC campaign '{campaign}' hadronize must be true or false"
        )

    required = ("process_code", "typint", "yjmin", "yjmax", "ptmin")
    missing = [name for name in required if name not in process_cfg]
    if missing:
        raise RuntimeError(
            f"FPMC process '{process}' is missing: {', '.join(missing)}"
        )

    lines = [
        "OUTPUT      0",
        "OUTPUTLHE   1",
        f"LHEFILE     '{lhefile}'",
        f"MAXEV       {events}",
        "TYPEPR      'EXC'",
        f"TYPINT      '{process_cfg['typint']}'",
        "ECMS        14000.",
        f"IPROC       {process_cfg['process_code']}",
        "NFLUX       16",
        f"YJMAX       {process_cfg['yjmax']}",
        f"YJMIN       {process_cfg['yjmin']}",
        f"PTMIN       {process_cfg['ptmin']}",
        "YWWMIN      0.002",
        "YWWMAX      0.2",
        f"NRN1        {seed}",
        f"HADR        '{'Y' if hadronize else 'N'}'",
    ]
    if "hmass" in process_cfg:
        lines.append(f"HMASS       {process_cfg['hmass']}")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Render an FPMC campaign card.")
    parser.add_argument("--process", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--nev", "--events", dest="events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=33799)
    parser.add_argument("--lhefile", default="FPMC.lhe")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    campaign, _ = generation_campaign_config("fpmc", args.process, args.campaign)
    card = render_card(args.process, campaign, args.events, args.seed, args.lhefile)
    if args.output is None:
        print(card, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
