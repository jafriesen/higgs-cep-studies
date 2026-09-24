#!/usr/bin/env python3
"""Clone a prepared H(bb) dataset with rescaled survival normalisation.

The survival factor S^2 is common to every central-exclusive process, so a
scan point multiplies the signal and the exclusive backgrounds together and
leaves the accidental-overlap MadGraph component alone.  Both stored weight
arrays carry it (mva/common/training.py reads physical_weight for the class
yields and the plugin score, and training_mixture_weight for the fit), so a
retrain at a different survival factor means rewriting both.  Everything else
is symlinked.

Unlike the H(cc) script this does not rescale the b-tag working point: the
tagging parameters are held at their configured values here.
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
    parser.add_argument(
        "--survival-scale",
        type=float,
        required=True,
        help="Flat factor on every survival_scaled component (signal included)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def component_scales(metadata, survival_scale):
    """Per-component weight multiplier, printed so the change is auditable."""
    ids = [item["id"] for item in metadata["components"]]
    if sorted(ids) != list(range(len(ids))):
        raise SystemExit(f"Component ids must be 0..N-1, got {sorted(ids)}")
    return {
        item["id"]: survival_scale if item["survival_scaled"] else 1.0
        for item in metadata["components"]
    }


def main():
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    if dest.exists():
        if not args.force:
            raise SystemExit(f"Destination already exists: {dest}")
        shutil.rmtree(dest)
    metadata = yaml.safe_load((source / "metadata.yaml").read_text(encoding="utf-8"))
    scales = component_scales(metadata, args.survival_scale)

    dest.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name == "metadata.yaml" or path.stem in WEIGHT_ARRAYS:
            continue
        (dest / path.name).symlink_to(path.resolve())

    component = np.load(source / "component.npy")
    factor = np.array([scales[index] for index in range(len(metadata["components"]))])
    row_scale = factor[component]
    for name in WEIGHT_ARRAYS:
        values = np.load(source / f"{name}.npy")
        np.save(dest / f"{name}.npy", values * row_scale)

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
