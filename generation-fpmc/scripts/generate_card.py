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


def render_card(process, campaign, events, seed, hadr="Y", lhefile="FPMC.lhe"):
    if events <= 0:
        raise RuntimeError("events must be a positive integer")
    if seed < 0:
        raise RuntimeError("seed must be a non-negative integer")
    if not lhefile or "'" in lhefile or "\n" in lhefile:
        raise RuntimeError("LHE filename must be nonempty and cannot contain quotes or newlines")

    hadr = str(hadr).upper()
    if hadr not in ("Y", "N"):
        raise RuntimeError("hadr must be Y or N")

    process_cfg = generation_process_config("fpmc", process)
    generation_campaign_config("fpmc", process, campaign)

    required = ("process_code", "typint")
    missing = [name for name in required if name not in process_cfg]
    if missing:
        raise RuntimeError(
            f"FPMC process '{process}' is missing: {', '.join(missing)}"
        )

    typepr = process_cfg.get("typepr", "EXC")
    nflux = process_cfg.get("nflux", 16)
    yjmax = process_cfg.get("yjmax", 0.2)
    yjmin = process_cfg.get("yjmin", 0.002)
    ptmin = process_cfg.get("ptmin", 15.0)
    ywwmin = process_cfg.get("ywwmin", 0.002)
    ywwmax = process_cfg.get("ywwmax", 0.2)
    isoftm = process_cfg.get("isoftm", 1)

    lines = [
        "OUTPUT      0",
        "OUTPUTLHE   1",
        f"LHEFILE     '{lhefile}'",
        f"MAXEV       {events}",
        f"TYPEPR      '{typepr}'",
        f"TYPINT      '{process_cfg['typint']}'",
        "ECMS        14000.",
        f"IPROC       {process_cfg['process_code']}",
        f"NFLUX       {nflux}",
        f"YJMAX       {yjmax}",
        f"YJMIN       {yjmin}",
        f"PTMIN       {ptmin}",
        f"YWWMIN      {ywwmin}",
        f"YWWMAX      {ywwmax}",
        f"NRN1        {seed}",
        f"HADR        '{hadr}'",
        f"HMASS       125.0",
        f"ISOFTM      {isoftm}"
    ]
    if "ifit" in process_cfg:
        lines.insert(-1, f"IFIT        {process_cfg['ifit']}")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Render an FPMC campaign card.")
    parser.add_argument("--process", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--nev", "--events", dest="events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=33799)
    parser.add_argument("--hadr", type=str.upper, choices=("Y", "N"), default="Y")
    parser.add_argument("--lhefile", default="FPMC.lhe")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    campaign, _ = generation_campaign_config("fpmc", args.process, args.campaign)
    card = render_card(
        args.process,
        campaign,
        args.events,
        args.seed,
        args.hadr,
        args.lhefile,
    )
    if args.output is None:
        print(card, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
