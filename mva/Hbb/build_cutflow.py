#!/usr/bin/env python3
"""Recount the actual H(bb) pre-MVA selections from the production ROOT files."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from common.jet_calibration import load_correction_map  # noqa: E402
from mva.common.config import load_channel_config  # noqa: E402
from mva.common.dataset import component_truth_pid_abs  # noqa: E402
from mva.common.features import central_features  # noqa: E402
from mva.common.protons import (  # noqa: E402
    load_pps_config,
    parse_hepmc_protons,
    parse_lhe_protons,
    real_proton_pass,
)


STAGES = ("generated", "two_jets", "delta_phi", "dijet_mass", "pps", "mass_window", "rapidity")


@lru_cache(maxsize=None)
def _correction_map(path, fsr_state, source):
    return load_correction_map(Path(path), fsr_state, source)


def _count_record(task):
    record, group, spec, settings, pps = task
    correction = (
        _correction_map(
            str(REPO / spec["correction_map"]),
            spec["fsr_state"],
            group["correction_source_sample"],
        )
        if settings["corrections"]
        else None
    )
    piece = central_features(
        Path(record["path"]),
        settings["tree"],
        settings["collection"],
        truth_pid_abs=component_truth_pid_abs(spec),
        jet_mode=settings["jets"],
        correction_map=correction,
        track_min_pt=settings["track_min_pt"],
        max_abs_jet_eta=settings["max_abs_jet_eta"],
        cutflow_only=True,
    )
    counts = {stage: 0 for stage in STAGES}
    counts["generated"] = int(piece["n_generated"])
    if piece.get("empty"):
        return group["index"], counts
    counts.update(piece["cutflow"])
    if not spec["real_protons"]:
        return group["index"], counts

    parser = parse_hepmc_protons if record["proton_kind"] == "hepmc" else parse_lhe_protons
    protons = parser(
        Path(record["proton_path"]),
        np.asarray(piece["event_indices"]) + int(record["proton_offset"]),
        pps["sqrt_s"],
    )
    accepted, _left, _right, mx, yx = real_proton_pass(
        protons["xi_left"],
        protons["xi_right"],
        pps,
        np.random.default_rng(record["seed"]),
    )
    in_mass = accepted & (mx >= settings["mass_window"][0]) & (
        mx <= settings["mass_window"][1]
    )
    rapidity = in_mass & (
        np.abs(yx - piece["dijet_rapidity"]) < settings["max_delta_y"]
    )
    counts["pps"] = int(np.sum(accepted))
    counts["mass_window"] = int(np.sum(in_mass))
    counts["rapidity"] = int(np.sum(rapidity))
    return group["index"], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    config = load_channel_config(Path(manifest["config_path"]), manifest["profile"])
    specifications = {item["name"]: item for item in config["components"]}
    pps = load_pps_config(REPO / config["pps_config"])
    tasks = []
    for group in manifest["groups"]:
        spec = specifications[group["component"]]
        tasks.extend(
            (record, group, spec, manifest["settings"], pps)
            for record in group["records"]
        )

    totals = {
        group["index"]: {stage: 0 for stage in STAGES}
        for group in manifest["groups"]
    }
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, (group_index, counts) in enumerate(
            executor.map(_count_record, tasks), start=1
        ):
            for stage in STAGES:
                totals[group_index][stage] += counts[stage]
            if completed % 100 == 0 or completed == len(tasks):
                print(f"processed {completed:,}/{len(tasks):,} ROOT files", flush=True)

    output = {
        "format_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "cuts": {
            "two_jets": "two FSR-recovered, calibrated AK4 jets with pT >= 15 GeV, raw core |eta| < 3, and valid correction",
            "delta_phi": "|delta phi(j1,j2)| > 3",
            "dijet_mass": "50 <= m(jj) <= 150 GeV",
            "pps": "one accepted proton in each PPS arm after xi smearing",
            "mass_window": "117 <= M_X <= 133 GeV",
            "rapidity": "|y_X - y_jj| < 0.2",
        },
        "groups": [],
    }
    for group in manifest["groups"]:
        output["groups"].append(
            {
                "component": group["component"],
                "campaign": group["campaign"],
                "xsec_fb": float(group["xsec_fb"]),
                "real_protons": bool(specifications[group["component"]]["real_protons"]),
                "counts": totals[group["index"]],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
