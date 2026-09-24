#!/usr/bin/env python3
"""Build a central-event H(bb) dataset for one configured profile."""

import argparse
import sys

import yaml
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.config import load_channel_config, restrict_components  # noqa: E402
from mva.common.dataset import prepare_dataset  # noqa: E402


def main():
    channel_config = yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="nominal_three_class",
                        choices=tuple(channel_config["profiles"]),
                        help="Profile from config.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--tree", default="Delphes")
    parser.add_argument("--collection", default="JetPUPPI")
    parser.add_argument(
        "--feature-set",
        choices=(*channel_config["feature_sets"], "full"),
        default=None,
    )
    parser.add_argument("--components", nargs="+")
    parser.add_argument("--extra-campaigns", nargs="+", metavar="NAME=CAMP[,CAMP]")
    parser.add_argument("--jets", choices=("truth", "leading"))
    parser.add_argument("--corrections", choices=("on", "off"))
    parser.add_argument("--track-min-pt", type=float)
    parser.add_argument("--store-parton", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--intensity-chunk", type=int, default=500000)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.workers <= 0 or args.intensity_chunk <= 0:
        raise ValueError("Worker and chunk counts must be positive")
    config = restrict_components(
        load_channel_config(SCRIPT_DIR / "config.yaml", args.profile), args.components
    )
    overrides = {}
    for value in args.extra_campaigns or []:
        if "=" not in value:
            raise ValueError(f"Invalid campaign override: {value!r}")
        name, campaigns = value.split("=", 1)
        overrides[name] = [item for item in campaigns.split(",") if item]
    for component in config["components"]:
        if component["name"] in overrides:
            if component["generator"] != "madgraph":
                raise ValueError("Campaign overrides apply only to MadGraph")
            component["campaigns"] = overrides.pop(component["name"])
    if overrides:
        raise ValueError(f"Overrides name inactive components: {sorted(overrides)}")
    profile = config["profile"]
    args.feature_set = args.feature_set or profile.get("feature_set", "locked")
    args.jets = args.jets or profile["jets"]
    args.corrections = bool(profile["corrections"]) if args.corrections is None else args.corrections == "on"
    args.track_min_pt = float(args.track_min_pt if args.track_min_pt is not None else profile["track_min_pt_gev"])
    args.max_abs_jet_eta = profile.get("max_abs_jet_eta")
    args.mass_window = tuple(float(value) for value in config["mass_window_gev"])
    args.max_delta_y = float(config["max_abs_rapidity_difference"])
    args.repo = REPO
    args.data_dir = Path(args.data_dir).resolve() if args.data_dir else SCRIPT_DIR / "data" / args.profile
    prepare_dataset(config, args)


if __name__ == "__main__":
    main()
