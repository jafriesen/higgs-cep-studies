#!/usr/bin/env python3
"""Proton--dijet rapidity difference histograms before the |Delta y| < 0.2 cut.

The H(bb) datasets only store events that pass the cut, so:
- the real-proton processes (CEP H->bb, exclusive gg->bb) are recounted from the
  dataset manifest's own file records and smearing seeds, with the dataset's
  central selection, PPS acceptance and mass window but no rapidity cut;
- the inclusive background folds each stored central event's dijet rapidity
  with the analytic accidental-pair density the dataset uses.
Inside |Delta y| < 0.2 both must reproduce the dataset's preselection yields;
the script checks this before writing.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.jet_calibration import load_correction_map  # noqa: E402
from mva.common.config import load_channel_config  # noqa: E402
from mva.common.dataset import component_truth_pid_abs  # noqa: E402
from mva.common.features import central_features  # noqa: E402
from mva.common.protons import (  # noqa: E402
    build_pair_density,
    load_pps_config,
    parse_hepmc_protons,
    parse_lhe_protons,
    proton_cell_intensities,
    real_proton_pass,
)

DATASET = ROOT / "mva/Hbb/data/fsr_mtd_eight_class_gg_cc"
MANIFEST = ROOT / "mva/Hbb/condor/fsr_mtd_eight_class_gg_cc/manifest.yaml"
OUTPUT = ROOT / "paper-plots/data/delta_y_histograms.npz"
EDGES = np.linspace(-0.3, 0.3, 31)
COMPONENTS = ("Hbb_fsr", "QCDbb_fsr", "QCDbb_madgraph_fsr")  # plotted classes, in order
LUMINOSITY_FB = 3000.0


@lru_cache(maxsize=None)
def _correction_map(path, fsr_state, source):
    return load_correction_map(Path(path), fsr_state, source)


def _record_delta_y(task):
    """Delta y of the events in the mass window, as in mva/Hbb/build_cutflow.py."""
    record, group, spec, settings, pps = task
    correction = (
        _correction_map(str(ROOT / spec["correction_map"]), spec["fsr_state"], group["correction_source_sample"])
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
    if piece.get("empty"):
        return np.empty(0), int(piece["n_generated"])
    parser = parse_hepmc_protons if record["proton_kind"] == "hepmc" else parse_lhe_protons
    protons = parser(
        Path(record["proton_path"]),
        np.asarray(piece["event_indices"]) + int(record["proton_offset"]),
        pps["sqrt_s"],
    )
    accepted, _left, _right, mx, yx = real_proton_pass(
        protons["xi_left"], protons["xi_right"], pps, np.random.default_rng(record["seed"])
    )
    window = accepted & (mx >= settings["mass_window"][0]) & (mx <= settings["mass_window"][1])
    return (yx - piece["dijet_rapidity"])[window], int(piece["n_generated"])


def real_histogram(component, manifest, specification, settings, pps, tag):
    groups = [group for group in manifest["groups"] if group["component"] == component]
    histogram = np.zeros(EDGES.size - 1)
    for group in groups:
        tasks = [(record, group, specification, settings, pps) for record in group["records"]]
        with ProcessPoolExecutor(max_workers=min(14, len(tasks))) as executor:
            results = list(executor.map(_record_delta_y, tasks))
        delta_y = np.concatenate([values for values, _n in results])
        generated = sum(n for _values, n in results)
        weight = group["xsec_fb"] * LUMINOSITY_FB / generated * tag
        histogram += np.histogram(delta_y, bins=EDGES)[0] * weight
        print(f"{component} {group['campaign']}: {delta_y.size} events in the mass window", flush=True)
    return histogram


def accidental_histogram(component, metadata, settings, pps):
    component_id = next(item["id"] for item in metadata["components"] if item["name"] == component)
    rows = np.flatnonzero(np.load(DATASET / "component.npy") == component_id)
    rapidity = np.load(DATASET / "dijet_rapidity.npy")[rows]
    weight = np.load(DATASET / "physical_weight.npy")[rows]
    pairs = build_pair_density(metadata["protons"]["path"], pps, seed=settings["seed"])
    histogram = np.zeros(EDGES.size - 1)
    for start in range(0, rows.size, 200_000):
        stop = min(start + 200_000, rows.size)
        intensity = proton_cell_intensities(
            pairs, rapidity[start:stop], EDGES, settings["mass_window"], metadata["protons"]["pileup_mu"]
        )
        histogram += weight[start:stop] @ intensity
    print(f"{component}: {rows.size} central events", flush=True)
    return histogram


def main():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    metadata = yaml.safe_load((DATASET / "metadata.yaml").read_text(encoding="utf-8"))
    config = load_channel_config(Path(manifest["config_path"]), manifest["profile"])
    specifications = {item["name"]: item for item in config["components"]}
    tags = {item["name"]: float(item["tag_factor"]) for item in metadata["components"]}
    real = {item["name"]: bool(item["real_protons"]) for item in metadata["components"]}
    settings = manifest["settings"]
    pps = load_pps_config(ROOT / config["pps_config"])

    histograms = []
    for component in COMPONENTS:
        if real[component]:
            histograms.append(real_histogram(component, manifest, specifications[component], settings, pps, tags[component]))
        else:
            histograms.append(accidental_histogram(component, metadata, settings, pps))
    histograms = np.stack(histograms)

    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    inside = np.abs(centres) < settings["max_delta_y"]
    for component, histogram in zip(COMPONENTS, histograms):
        expected = metadata["physical_yields_per_component"][component]
        closure = histogram[inside].sum() / expected - 1.0
        print(f"closure {component}: inside |dy| < {settings['max_delta_y']} {histogram[inside].sum():.6g} "
              f"vs dataset {expected:.6g} ({closure:+.1e})")
        if abs(closure) > 1e-3:
            raise SystemExit(f"{component} does not reproduce the dataset preselection yield")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, edges=EDGES, histograms=histograms, components=np.array(COMPONENTS))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
