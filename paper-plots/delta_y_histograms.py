#!/usr/bin/env python3
"""Proton--dijet rapidity difference histograms before the |Delta y| < 0.2 cut.

The reduced Hbb dataset only stores events that already pass the cut, so:
- SuperChic Hbb and exclusive QCD bb are recounted with the dataset selection
  (central dijet, PPS acceptance, mass window, same smearing seeds) without it;
- the inclusive background folds each central event's dijet rapidity with the
  weighted min-bias proton-pair pool, which is how accidental pairs are sampled.
  Central events with no pool pair within 0.2 are absent from the dataset; the
  pool ends at |y_X| ~ 1.15, so their contribution inside |Delta y| < 0.3 is
  negligible.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.MVA_hcc import prepare_dataset as pd  # noqa: E402
from common.config_utils import resolve_path  # noqa: E402

OUTPUT = ROOT / "paper-plots/data/delta_y_histograms.npz"
EDGES = np.linspace(-0.3, 0.3, 31)
SEED = 12345


def superchic_delta_y(arguments):
    path, proton_kind, proton_file, pps, seed = arguments
    central = pd.source.central_features(path, "Delphes", "JetPUPPI")
    if central.get("empty"):
        return np.empty(0)
    if proton_kind == "hepmc":
        xi_left, xi_right = pd.pps_lib.parse_hepmc_proton_xi(
            np, proton_file, central["event_indices"], pps["sqrt_s"]
        )
    else:
        xi_left, xi_right = pd.parse_lhe_proton_xi(
            proton_file, central["event_indices"], pps["sqrt_s"]
        )
    passed, _left, _right, mx, yx = pd.source.real_proton_pass(
        np, xi_left, xi_right, pps, np.random.default_rng(seed)
    )
    window = passed & (mx >= pd.MASS_WINDOW_GEV[0]) & (mx <= pd.MASS_WINDOW_GEV[1])
    return (yx - central["dijet_rapidity"])[window]


def superchic_histogram(name, pps):
    names = [spec["name"] for spec in pd.COMPONENT_SPECS]
    spec = pd.COMPONENT_SPECS[names.index(name)]
    seed_id = names.index(name)
    _campaign, _sub, _dir, files, kind, proton_files = pd.resolve_superchic_files(spec, None)
    tasks = [
        (path, kind, proton_file, pps, SEED + seed_id * 10000 + index)
        for index, (path, proton_file) in enumerate(zip(files, proton_files))
    ]
    with ProcessPoolExecutor(max_workers=min(12, len(tasks))) as executor:
        delta_y = np.concatenate(list(executor.map(superchic_delta_y, tasks)))
    print(f"{name}: {delta_y.size} events in the mass window", flush=True)
    return np.histogram(delta_y, bins=EDGES)[0].astype(float)


def inclusive_histogram():
    data_dir = ROOT / "analysis/MVA_new/data"
    with (data_dir / "metadata.yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    classes = np.asarray(np.load(data_dir / "class.npy", mmap_mode="r"))
    rows = np.flatnonzero(classes == 2)
    groups = np.asarray(np.load(data_dir / "group_id.npy", mmap_mode="r")[rows])
    _, first = np.unique(groups, return_index=True)
    rows = rows[first]
    matrix = np.load(data_dir / "x.npy", mmap_mode="r")
    rapidity = np.asarray(
        matrix[rows, list(metadata["features"]).index("dijet_rapidity")], dtype=float
    )
    weight = np.asarray(np.load(data_dir / "central_weight.npy", mmap_mode="r")[rows])

    pool = pq.read_table(metadata["source_proton_pool"], columns=["yx", "weight"])
    order = np.argsort(np.asarray(pool["yx"]))
    pool_yx = np.asarray(pool["yx"], dtype=float)[order]
    pool_cdf = np.concatenate(
        ([0.0], np.cumsum(np.asarray(pool["weight"], dtype=float)[order]))
    )
    pool_cdf /= pool_cdf[-1]
    cdf = np.stack(
        [pool_cdf[np.searchsorted(pool_yx, rapidity + edge, side="left")] for edge in EDGES]
    )
    print(f"QCDbb_madgraph: {rows.size} central events", flush=True)
    return (np.diff(cdf, axis=0) * weight).sum(axis=1)


def main():
    pps = pd.pps_lib.load_pps_config(resolve_path(pd.DEFAULT_PPS_CONFIG, base=pd.REPO))
    histograms = np.stack(
        [
            superchic_histogram("Hbb_superchic", pps),
            superchic_histogram("QCDbb_superchic", pps),
            inclusive_histogram(),
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, edges=EDGES, histograms=histograms)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
