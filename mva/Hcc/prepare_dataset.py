#!/usr/bin/env python3
"""Build a central-event H(cc) dataset for one configured profile."""

import argparse
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.config import load_channel_config, restrict_components  # noqa: E402
from mva.common.dataset import prepare_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text())["profiles"]),
        default="nominal_full",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tree", default="Delphes")
    parser.add_argument("--collection", default="JetPUPPI")
    parser.add_argument(
        "--feature-set", choices=("locked", "legacy_locked", "full"), default=None
    )
    parser.add_argument("--components", nargs="+")
    parser.add_argument("--extra-campaigns", nargs="+", metavar="NAME=CAMP[,CAMP]")
    parser.add_argument("--jets", choices=("truth", "leading"), default=None)
    parser.add_argument("--corrections", choices=("on", "off"), default=None)
    parser.add_argument("--track-min-pt", type=float, default=None)
    parser.add_argument("--store-parton", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--intensity-chunk", type=int, default=500000)
    parser.add_argument("--skip-hash", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.workers <= 0 or args.intensity_chunk <= 0:
        raise ValueError("--workers and --intensity-chunk must be positive")
    config = load_channel_config(SCRIPT_DIR / "config.yaml", args.profile)
    config = restrict_components(config, args.components)
    overrides = {}
    for value in args.extra_campaigns or []:
        if "=" not in value:
            raise ValueError(f"Invalid --extra-campaigns value: {value!r}")
        name, campaigns = value.split("=", 1)
        overrides[name] = [item for item in campaigns.split(",") if item]
    for component in config["components"]:
        if component["name"] in overrides:
            if component["generator"] != "madgraph":
                raise ValueError("--extra-campaigns applies only to MadGraph components")
            component["campaigns"] = overrides[component["name"]]
    unknown = sorted(set(overrides) - {item["name"] for item in config["components"]})
    if unknown:
        raise ValueError(f"Campaign overrides name inactive components: {unknown}")

    profile = config["profile"]
    args.feature_set = args.feature_set or profile.get("feature_set", "locked")
    args.jets = args.jets or profile["jets"]
    if args.corrections is None:
        args.corrections = bool(profile["corrections"])
    else:
        args.corrections = args.corrections == "on"
    args.track_min_pt = (
        float(args.track_min_pt)
        if args.track_min_pt is not None
        else float(profile["track_min_pt_gev"])
    )
    args.max_abs_jet_eta = profile.get("max_abs_jet_eta")
    args.mass_window = tuple(float(value) for value in config["mass_window_gev"])
    args.max_delta_y = float(config["max_abs_rapidity_difference"])
    args.repo = REPO
    args.data_dir = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else SCRIPT_DIR / "data" / args.profile
    )
    prepare_dataset(config, args)


if __name__ == "__main__":
    main()
