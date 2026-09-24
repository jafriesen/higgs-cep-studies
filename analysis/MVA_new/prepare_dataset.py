#!/usr/bin/env python3
"""Build the reduced, local dataset for the locked 20-feature report."""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_MVA = SCRIPT_DIR.parent / "MVA"
DEFAULT_CACHE = (
    SOURCE_MVA / "output/mva_bb_multiclass_protons/cache_v4_all_campaigns"
)
DEFAULT_RANKING = (
    SOURCE_MVA
    / "output/mva_bb_multiclass_protons/architecture_study_v1/stage1_ranking.yaml"
)
DEFAULT_POOL = (
    SCRIPT_DIR.parents[1]
    / "output/minbias/minbias/pairs/proton_pairs.parquet"
)
DEFAULT_DATA = SCRIPT_DIR / "data"
DROP_CAMPAIGN = "QCDbb__v01"
BATCH_ROWS = 250000
N_FEATURES = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING))
    parser.add_argument("--proton-pool", default=str(DEFAULT_POOL))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    return parser.parse_args()


def read_array(cache_dir, name):
    return np.load(cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def write_filtered(source, keep_rows, target, dtype=None):
    output_dtype = np.dtype(dtype or source.dtype)
    output = np.lib.format.open_memmap(
        target, mode="w+", dtype=output_dtype, shape=(keep_rows.size,)
    )
    for start in range(0, keep_rows.size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, keep_rows.size)
        output[start:stop] = source[keep_rows[start:stop]]
    output.flush()
    del output


def write_features(source, keep_rows, features, target):
    output = np.lib.format.open_memmap(
        target,
        mode="w+",
        dtype=np.float32,
        shape=(keep_rows.size, features.size),
    )
    for start in range(0, keep_rows.size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, keep_rows.size)
        rows = keep_rows[start:stop]
        output[start:stop] = source[rows][:, features]
        print(f"  features: {stop:,} / {keep_rows.size:,}", flush=True)
    output.flush()
    del output


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = parse_args()
    source_cache = Path(args.source_cache).resolve()
    ranking_path = Path(args.ranking).resolve()
    pool_path = Path(args.proton_pool).resolve()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with open(source_cache / "metadata.yaml", encoding="utf-8") as handle:
        source_metadata = yaml.safe_load(handle)
    with open(ranking_path, encoding="utf-8") as handle:
        ranking = yaml.safe_load(handle)

    feature_indices = np.asarray(
        ranking["ranked_feature_indices"][:N_FEATURES], dtype=int
    )
    feature_names = np.asarray(source_metadata["features"])[feature_indices]
    required = {"dijet_rapidity", "yx_minus_dijet_rapidity"}
    missing = required.difference(feature_names)
    if missing:
        raise RuntimeError(
            f"The selected feature set is missing required fields: {sorted(missing)}"
        )

    classes = read_array(source_cache, "class")
    campaigns = read_array(source_cache, "source_campaign")
    coverage = read_array(source_cache, "coverage_mask")
    old_stitch = read_array(source_cache, "central_stitch_weight_fb")
    band = read_array(source_cache, "band_probability")
    source_weight = read_array(source_cache, "physical_weight")
    source_mixture = read_array(source_cache, "training_mixture_weight")
    madgraph = np.asarray(classes == 2)

    luminosity = {
        name: block["effective_mc_luminosity_fb_inv"]
        for name, block in source_metadata["madgraph_stitching"].items()
    }
    coverage_bits = source_metadata["madgraph_coverage_bits"]
    covered = np.zeros(int(madgraph.sum()), dtype=np.float64)
    madgraph_coverage = np.asarray(coverage[madgraph])
    for name, value in luminosity.items():
        covered += value * (
            (madgraph_coverage & (1 << coverage_bits[name])) > 0
        )
    if not np.allclose(1.0 / covered, old_stitch[madgraph], rtol=1e-9):
        raise RuntimeError("Source stitching weights do not match the cache metadata")

    band_madgraph = np.asarray(band[madgraph], dtype=np.float64)
    weight_madgraph = np.asarray(source_weight[madgraph], dtype=np.float64)
    stitch_madgraph = np.asarray(old_stitch[madgraph], dtype=np.float64)
    scale = weight_madgraph / (stitch_madgraph * band_madgraph / 4.0)
    if not np.allclose(scale, scale[0], rtol=1e-6):
        raise RuntimeError("The MadGraph physical scale is not constant")
    physical_scale = float(scale[0])

    keep = ~(madgraph & (campaigns == DROP_CAMPAIGN))
    keep_rows = np.flatnonzero(keep)
    kept_classes = np.asarray(classes[keep_rows])
    kept_madgraph = kept_classes == 2
    retained_campaigns = [name for name in luminosity if name != DROP_CAMPAIGN]
    retained_coverage = np.asarray(coverage[keep_rows[kept_madgraph]])
    recovered = np.zeros(int(kept_madgraph.sum()), dtype=np.float64)
    for name in retained_campaigns:
        recovered += luminosity[name] * (
            (retained_coverage & (1 << coverage_bits[name])) > 0
        )
    if np.any(recovered <= 0.0):
        raise RuntimeError("A retained MadGraph event has no retained campaign coverage")
    new_stitch = 1.0 / recovered

    old_kept_stitch = np.asarray(old_stitch[keep_rows[kept_madgraph]])
    reweight_factor = new_stitch / old_kept_stitch
    physical_weight = np.asarray(source_weight[keep_rows], dtype=np.float64).copy()
    training_weight = np.asarray(source_mixture[keep_rows], dtype=np.float64).copy()
    physical_weight[kept_madgraph] *= reweight_factor
    training_weight[kept_madgraph] *= reweight_factor
    central_weight = np.zeros(keep_rows.size, dtype=np.float64)
    central_weight[kept_madgraph] = new_stitch * physical_scale

    write_features(
        read_array(source_cache, "x"),
        keep_rows,
        feature_indices,
        data_dir / "x.npy",
    )
    write_filtered(classes, keep_rows, data_dir / "class.npy")
    write_filtered(
        read_array(source_cache, "group_id"), keep_rows, data_dir / "group_id.npy"
    )
    write_filtered(
        read_array(source_cache, "mx"), keep_rows, data_dir / "mx.npy", np.float64
    )
    write_filtered(band, keep_rows, data_dir / "band_probability.npy", np.float64)
    np.save(data_dir / "physical_weight.npy", physical_weight, allow_pickle=False)
    np.save(
        data_dir / "training_mixture_weight.npy", training_weight, allow_pickle=False
    )
    np.save(data_dir / "central_weight.npy", central_weight, allow_pickle=False)

    pool_table = pq.read_table(pool_path, columns=["yx", "weight"])
    if pool_table.num_rows == 0:
        raise RuntimeError("The proton-pair pool is empty")
    shutil.copyfile(pool_path, data_dir / "proton_pairs.parquet")

    counts = np.bincount(kept_classes, minlength=3)
    yields = np.bincount(kept_classes, weights=physical_weight, minlength=3)
    metadata = plain({
        "format_version": 1,
        "description": "Reduced locked 20-feature proton MVA report dataset",
        "source_cache": str(source_cache),
        "source_ranking": str(ranking_path),
        "source_proton_pool": str(pool_path),
        "rows": int(keep_rows.size),
        "features": feature_names,
        "original_feature_indices": feature_indices,
        "classes": source_metadata["classes"],
        "rows_per_class": counts,
        "physical_yields": yields,
        "dropped_campaign": DROP_CAMPAIGN,
        "retained_campaigns": retained_campaigns,
        "physical_scale": physical_scale,
        "proton_pairs": int(pool_table.num_rows),
        "arrays": {
            "x": {"dtype": "float32", "shape": [int(keep_rows.size), N_FEATURES]},
            "class": {"dtype": str(classes.dtype), "shape": [int(keep_rows.size)]},
            "group_id": {
                "dtype": str(read_array(source_cache, "group_id").dtype),
                "shape": [int(keep_rows.size)],
            },
            "mx": {"dtype": "float64", "shape": [int(keep_rows.size)]},
            "physical_weight": {"dtype": "float64", "shape": [int(keep_rows.size)]},
            "training_mixture_weight": {
                "dtype": "float64", "shape": [int(keep_rows.size)]
            },
            "band_probability": {"dtype": "float64", "shape": [int(keep_rows.size)]},
            "central_weight": {"dtype": "float64", "shape": [int(keep_rows.size)]},
        },
    })
    with open(data_dir / "metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    print(
        f"Wrote {keep_rows.size:,} rows and {N_FEATURES} features to {data_dir}\n"
        f"Class counts: {counts.tolist()}\n"
        f"Physical yields: {yields.tolist()}\n"
        f"Proton pairs: {pool_table.num_rows:,}\n"
        f"Done in {time.perf_counter() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
