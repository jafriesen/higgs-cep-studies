#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, natural_key, resolve_minbias_campaign, resolve_path  # noqa: E402


OUTPUT_NAME = "bunch_crossings.parquet"
METADATA_NAME = "metadata.json"
PARQUET_COMPRESSION = "snappy"
PROTON_COLUMNS = (
    "event_id",
    "particle_index",
    "pdg_id",
    "is_final",
    "px",
    "py",
    "pz",
    "E",
    "m",
    "pt",
    "eta",
    "phi",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Poisson-distributed bunch crossings from a min-bias campaign."
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Min-bias campaign. Defaults to config.yaml minbias.default_campaign.",
    )
    parser.add_argument("--mu", type=float, default=200.0, help="Mean interactions per BX")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Poisson random seed. Defaults to random.seed from --pps-config.",
    )
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file defining beam.sqrt_s_gev, pps.xi_ranges, and random.seed.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Maximum input Parquet files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to output/minbias/<campaign>/bx.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    return parser.parse_args()


def load_pps_config(path):
    config = load_yaml(path)
    sqrt_s = float((config.get("beam") or {}).get("sqrt_s_gev", 0.0))
    if sqrt_s <= 0.0:
        raise RuntimeError(f"{path} must define a positive beam.sqrt_s_gev")

    ranges = []
    for station, bounds in ((config.get("pps") or {}).get("xi_ranges") or {}).items():
        if len(bounds) != 2:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        ranges.append((str(station), low, high))
    if not ranges:
        raise RuntimeError(f"{path} must define at least one pps.xi_ranges entry")

    return {
        "sqrt_s": sqrt_s,
        "xi_ranges": ranges,
        "seed": int((config.get("random") or {}).get("seed", 12345)),
    }


def discover_inputs(campaign_dir, max_files=None):
    files = sorted((campaign_dir / "parquet").glob("*.parquet"), key=natural_key)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No min-bias Parquet files found in {campaign_dir / 'parquet'}")
    return files


def parquet_event_info(path):
    parquet_file = pq.ParquetFile(path)
    missing = [name for name in PROTON_COLUMNS if name not in parquet_file.schema_arrow.names]
    if missing:
        raise RuntimeError(f"Missing required column(s) in {path}: {', '.join(missing)}")

    event_column = parquet_file.schema_arrow.get_field_index("event_id")
    minima = []
    maxima = []
    for row_group in range(parquet_file.num_row_groups):
        statistics = parquet_file.metadata.row_group(row_group).column(event_column).statistics
        if statistics is None or not statistics.has_min_max:
            event_ids = np.asarray(pq.read_table(path, columns=["event_id"])["event_id"])
            minima = [int(np.min(event_ids))]
            maxima = [int(np.max(event_ids))]
            break
        minima.append(int(statistics.min))
        maxima.append(int(statistics.max))

    if not minima:
        raise RuntimeError(f"No events found in {path}")
    event_min = min(minima)
    event_max = max(maxima)
    return {
        "path": path,
        "event_min": event_min,
        "event_max": event_max,
        "n_events": event_max - event_min + 1,
    }


def sample_bx_counts(total_interactions, mu, rng):
    counts = []
    used = 0
    while used < total_interactions:
        count = int(rng.poisson(mu))
        if used + count > total_interactions:
            break
        counts.append(count)
        used += count
    return np.asarray(counts, dtype=np.int32), total_interactions - used


def station_memberships(xi, xi_ranges):
    passing = np.zeros(xi.shape, dtype=bool)
    masks = []
    for station, low, high in xi_ranges:
        mask = (xi >= low) & (xi < high)
        passing |= mask
        masks.append((station, mask))

    stations = []
    for index in np.nonzero(passing)[0]:
        stations.append([station for station, mask in masks if mask[index]])
    return passing, stations


def empty_flat_protons():
    return {
        "bx_id": [],
        "interaction_id": [],
        "source_file_index": [],
        "source_event_id": [],
        "particle_index": [],
        "side": [],
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
        "m": [],
        "pt": [],
        "eta": [],
        "phi": [],
        "xi": [],
        "pps_stations": [],
    }


def append_file_protons(flat, info, file_index, global_offset, used_interactions, bx_starts, bx_stops, pps):
    usable_events = min(info["n_events"], used_interactions - global_offset)
    if usable_events <= 0:
        return

    event_max = info["event_min"] + usable_events - 1
    filters = [
        ("pdg_id", "=", 2212),
        ("is_final", "=", True),
        ("pz", "!=", 0.0),
        ("event_id", "<=", event_max),
    ]
    table = pq.read_table(info["path"], columns=list(PROTON_COLUMNS), filters=filters)
    if table.num_rows == 0:
        return

    event_ids = np.asarray(table["event_id"], dtype=np.int64)
    pz = np.asarray(table["pz"], dtype=np.float64)
    energy = np.asarray(table["E"], dtype=np.float64)
    beam_energy = pps["sqrt_s"] / 2.0
    xi = (beam_energy - energy) / beam_energy
    valid = np.isfinite(xi) & (xi > 0.0)
    accepted, stations = station_memberships(xi, pps["xi_ranges"])
    selected = np.nonzero(valid & accepted)[0]
    if selected.size == 0:
        return

    global_interactions = global_offset + event_ids[selected] - info["event_min"]
    bx_ids = np.searchsorted(bx_stops, global_interactions, side="right")
    interaction_ids = global_interactions - bx_starts[bx_ids]

    flat["bx_id"].append(bx_ids.astype(np.int64, copy=False))
    flat["interaction_id"].append(interaction_ids.astype(np.int32, copy=False))
    flat["source_file_index"].append(np.full(selected.size, file_index, dtype=np.int32))
    flat["source_event_id"].append(event_ids[selected])
    flat["particle_index"].append(np.asarray(table["particle_index"], dtype=np.int32)[selected])
    flat["side"].append(np.where(pz[selected] < 0.0, -1, 1).astype(np.int8))
    for name in ("px", "py", "pz", "E", "m", "pt", "eta", "phi"):
        flat[name].append(np.asarray(table[name], dtype=np.float64)[selected])
    flat["xi"].append(xi[selected])
    selected_positions = {int(value): pos for pos, value in enumerate(np.nonzero(accepted)[0])}
    flat["pps_stations"].extend(stations[selected_positions[int(index)]] for index in selected)


def concatenate(values, dtype):
    return np.concatenate(values).astype(dtype, copy=False) if values else np.empty(0, dtype=dtype)


def nested_bx_array(flat, bx_counts):
    arrays = {
        "bx_id": concatenate(flat["bx_id"], np.int64),
        "interaction_id": concatenate(flat["interaction_id"], np.int32),
        "source_file_index": concatenate(flat["source_file_index"], np.int32),
        "source_event_id": concatenate(flat["source_event_id"], np.int64),
        "particle_index": concatenate(flat["particle_index"], np.int32),
        "side": concatenate(flat["side"], np.int8),
    }
    for name in ("px", "py", "pz", "E", "m", "pt", "eta", "phi", "xi"):
        arrays[name] = concatenate(flat[name], np.float64)

    stations = ak.Array(flat["pps_stations"]) if flat["pps_stations"] else ak.Array([[""]])[:0]
    protons = ak.zip(
        {
            name: values
            for name, values in arrays.items()
            if name != "bx_id"
        }
        | {"pps_stations": stations},
        depth_limit=1,
    )
    proton_counts = np.bincount(arrays["bx_id"], minlength=len(bx_counts))
    nested_protons = ak.unflatten(protons, proton_counts)
    return ak.zip(
        {
            "bx_id": np.arange(len(bx_counts), dtype=np.int64),
            "n_interactions": bx_counts,
            "protons": nested_protons,
        },
        depth_limit=1,
    )


def build_bunch_crossings(file_infos, bx_counts, pps):
    bx_stops = np.cumsum(bx_counts, dtype=np.int64)
    bx_starts = bx_stops - bx_counts
    used_interactions = int(bx_stops[-1]) if bx_stops.size else 0
    flat = empty_flat_protons()
    global_offset = 0

    for file_index, info in enumerate(file_infos):
        if global_offset >= used_interactions:
            break
        print(f"Reading PPS protons: {info['path']}")
        append_file_protons(
            flat,
            info,
            file_index,
            global_offset,
            used_interactions,
            bx_starts,
            bx_stops,
            pps,
        )
        global_offset += info["n_events"]

    return nested_bx_array(flat, bx_counts)


def write_outputs(array, metadata, output_dir, overwrite=False):
    output_file = output_dir / OUTPUT_NAME
    metadata_file = output_dir / METADATA_NAME
    existing = [path for path in (output_file, metadata_file) if path.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Output already exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(array, output_file, compression=PARQUET_COMPRESSION)
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output_file, metadata_file


def main():
    args = parse_args()
    if args.mu <= 0.0:
        raise RuntimeError("--mu must be > 0")
    if args.seed is not None and args.seed < 0:
        raise RuntimeError("--seed must be >= 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")

    campaign_dir, campaign = resolve_minbias_campaign(args.campaign)
    pps_path = resolve_path(args.pps_config, base=ROOT)
    pps = load_pps_config(pps_path)
    seed = args.seed if args.seed is not None else pps["seed"]
    files = discover_inputs(campaign_dir, args.max_files)
    file_infos = [parquet_event_info(path) for path in files]
    total_interactions = sum(info["n_events"] for info in file_infos)
    bx_counts, unused = sample_bx_counts(
        total_interactions,
        args.mu,
        np.random.default_rng(seed),
    )
    array = build_bunch_crossings(file_infos, bx_counts, pps)

    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else campaign_dir / "bx"
    )
    metadata = {
        "schema_version": 1,
        "campaign": campaign,
        "input_files": [str(path) for path in files],
        "mu": float(args.mu),
        "seed": int(seed),
        "sqrt_s_gev": float(pps["sqrt_s"]),
        "pps_xi_ranges": {
            station: [low, high] for station, low, high in pps["xi_ranges"]
        },
        "total_interactions": int(total_interactions),
        "used_interactions": int(np.sum(bx_counts, dtype=np.int64)),
        "unused_tail_interactions": int(unused),
        "bunch_crossings": int(len(bx_counts)),
        "pps_protons": int(ak.sum(ak.num(array.protons, axis=1))),
        "parquet_compression": PARQUET_COMPRESSION,
    }
    output_file, metadata_file = write_outputs(array, metadata, output_dir, args.overwrite)
    print(
        f"Built {len(array)} BX from {metadata['used_interactions']}/{total_interactions} interactions; "
        f"unused_tail={unused}, PPS_protons={metadata['pps_protons']}"
    )
    print(f"Wrote {output_file}")
    print(f"Wrote {metadata_file}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
