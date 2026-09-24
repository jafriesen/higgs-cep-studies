#!/usr/bin/env python3
"""Clone a prepared dataset with rescaled component weights.

Both the c-tag working point and a flat SuperChic-QCD normalisation enter the
stored weights (mva/common/dataset.py builds physical_weight and
training_mixture_weight as cross_section_weight * tag), so changing either one
for a retrain means rewriting those two arrays. Everything else is symlinked.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

WEIGHT_ARRAYS = ("physical_weight", "training_mixture_weight")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--eff-c", type=float, default=None, help="New c-tag efficiency")
    parser.add_argument("--mistag-b-to-c", type=float, default=None)
    parser.add_argument("--mistag-light-to-c", type=float, default=None)
    parser.add_argument(
        "--survival-scale",
        type=float,
        default=1.0,
        help="Flat factor on every survival_scaled component (signal included)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def component_scales(metadata, args):
    """Per-component weight multiplier, printed so the change is auditable."""
    tagging = metadata["tagging"]
    requested = {
        "cc": args.eff_c if args.eff_c is not None else tagging["eff_c"],
        "bb": args.mistag_b_to_c if args.mistag_b_to_c is not None else tagging["mistag_b_to_c"],
        "light": args.mistag_light_to_c
        if args.mistag_light_to_c is not None
        else tagging["mistag_light_to_c"],
    }
    stored = {
        "cc": float(tagging["eff_c"]),
        "bb": float(tagging["mistag_b_to_c"]),
        "light": float(tagging["mistag_light_to_c"]),
    }
    scales = {}
    for component in metadata["components"]:
        flavor = component["source_flavor"]
        scale = (requested[flavor] / stored[flavor]) ** 2
        if component["survival_scaled"]:
            scale *= args.survival_scale
        scales[component["id"]] = scale
    return scales, requested


def main():
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    if dest.exists():
        if not args.force:
            raise SystemExit(f"Destination already exists: {dest}")
        shutil.rmtree(dest)
    metadata = yaml.safe_load((source / "metadata.yaml").read_text(encoding="utf-8"))
    scales, requested = component_scales(metadata, args)

    dest.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name in {"metadata.yaml"} or path.stem in WEIGHT_ARRAYS:
            continue
        (dest / path.name).symlink_to(path.resolve())

    component = np.load(source / "component.npy")
    factor = np.array([scales[index] for index in range(len(metadata["components"]))])
    row_scale = factor[component]
    for name in WEIGHT_ARRAYS:
        values = np.load(source / f"{name}.npy")
        np.save(dest / f"{name}.npy", values * row_scale)

    metadata["tagging"] = dict(metadata["tagging"])
    metadata["tagging"]["eff_c"] = float(requested["cc"])
    metadata["tagging"]["mistag_b_to_c"] = float(requested["bb"])
    metadata["tagging"]["mistag_light_to_c"] = float(requested["light"])
    for item in metadata["components"]:
        flavor = item["source_flavor"]
        item["tag_factor"] = float(requested[flavor]) ** 2
    yields = metadata.get("physical_yields_per_component") or {}
    for item in metadata["components"]:
        if item["name"] in yields:
            yields[item["name"]] = float(yields[item["name"]] * scales[item["id"]])
    metadata["weight_rescale"] = {
        "source_dataset": str(source),
        "survival_scale": float(args.survival_scale),
        "component_scales": {
            item["name"]: float(scales[item["id"]]) for item in metadata["components"]
        },
    }
    (dest / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )

    print(f"Wrote {dest}")
    for item in metadata["components"]:
        print(f"  {item['name']:24s} x{scales[item['id']]:.6g}")


if __name__ == "__main__":
    main()
