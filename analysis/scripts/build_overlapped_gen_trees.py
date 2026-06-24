#!/usr/bin/env python3
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config_utils import (  # noqa: E402
    discover_npz_files,
    load_yaml,
    resolve_minbias_campaign,
    resolve_path,
    resolve_process_campaign,
)


STATION_XI = {
    "192": (0.0800, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.0140, 0.0263),
    "420": (0.00325, 0.0116),
}
SOURCE_SIGNAL = 0
SOURCE_MINBIAS = 1
DEFAULT_SQRT_S = 14000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build overlapped GEN-level ROOT trees from processed SuperChic and minbias outputs."
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name from processes.yaml; may be repeated. Defaults to all processes.",
    )
    parser.add_argument("--campaign", help="Signal campaign key to use for all selected processes")
    parser.add_argument("--minbias-input", help="Minbias campaign directory or .npz file override")
    parser.add_argument("--max-processes", type=int, help="Optional process count limit")
    parser.add_argument("--max-overlays", type=int, help="Optional overlay count limit per process")
    parser.add_argument("--max-minbias-files", type=int, help="Optional minbias .npz file count limit")
    parser.add_argument("--output-dir", default="analysis/output", help="Output directory")
    args = parser.parse_args()

    for name in ("max_processes", "max_overlays", "max_minbias_files"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    return args


def station_hits(xi):
    hits = {
        station: int(xi >= xi_min and xi < xi_max)
        for station, (xi_min, xi_max) in STATION_XI.items()
    }
    return hits, sum(hits.values())


def eta_from_components(px, py, pz):
    pt = math.hypot(px, py)
    if pt <= 0.0:
        return math.copysign(float("inf"), pz) if pz else 0.0
    return math.asinh(pz / pt)


def phi_from_components(px, py):
    return math.atan2(py, px)


def root_arrays(rows):
    if not rows:
        return {}
    arrays = {}
    keys = rows[0].keys()
    for key in keys:
        values = [row[key] for row in rows]
        if all(isinstance(value, (int, np.integer)) for value in values):
            arrays[key] = np.asarray(values, dtype=np.int32)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    return arrays


def merge_tree_arrays(*parts):
    merged = {}
    for arrays in parts:
        for key, value in arrays.items():
            merged.setdefault(key, []).append(value)
    return {
        key: np.concatenate(values) if values else np.empty(0)
        for key, values in merged.items()
    }


def load_signal(path, max_events=None):
    import ROOT

    if not path.exists():
        raise RuntimeError(f"Processed signal ROOT file does not exist: {path}")

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open processed signal ROOT file: {path}")
    trees = {name: root_file.Get(name) for name in ("Interaction", "Protons", "ProtonPairs")}
    missing = [name for name, tree in trees.items() if not tree]
    if missing:
        root_file.Close()
        raise RuntimeError(f"{path} is missing required trees: {', '.join(missing)}")

    def rows_from_tree(tree, entry_stop=None):
        branch_names = [branch.GetName() for branch in tree.GetListOfBranches()]
        n_entries = tree.GetEntries()
        if entry_stop is not None:
            n_entries = min(n_entries, entry_stop)
        rows = []
        for entry in range(n_entries):
            tree.GetEntry(entry)
            rows.append({name: getattr(tree, name) for name in branch_names})
        return rows

    interactions = rows_from_tree(trees["Interaction"], entry_stop=max_events)
    proton_stop = max_events * 2 if max_events is not None else None
    protons = rows_from_tree(trees["Protons"], entry_stop=proton_stop)
    pairs = rows_from_tree(trees["ProtonPairs"], entry_stop=max_events)
    root_file.Close()

    interaction_by_event = {}
    for row in interactions:
        interaction_by_event[int(row["event_id"])] = row

    protons_by_event = defaultdict(list)
    for row in protons:
        protons_by_event[int(row["event_id"])].append(row)

    pair_by_event = {}
    for row in pairs:
        pair_by_event[int(row["event_id"])] = row

    event_ids = sorted(set(interaction_by_event).intersection(protons_by_event))
    if not event_ids:
        raise RuntimeError(f"No usable signal events found in {path}")
    return event_ids, interaction_by_event, protons_by_event, pair_by_event


def interaction_universe_from_npz(data, filename):
    if all(name in data.files for name in ("mu_per_bx", "bx_offset", "n_bx")):
        mu_per_bx = np.asarray(data["mu_per_bx"], dtype=np.int64)
        bx_offset = int(np.asarray(data["bx_offset"]).item())
        n_bx = int(np.asarray(data["n_bx"]).item())
        if mu_per_bx.size != n_bx:
            raise RuntimeError(
                f"mu_per_bx length ({mu_per_bx.size}) does not match n_bx ({n_bx}) in {filename}"
            )
        bx_parts = []
        interaction_parts = []
        for local_bx, n_interactions in enumerate(mu_per_bx):
            if n_interactions <= 0:
                continue
            bx = bx_offset + local_bx
            bx_parts.append(np.full(int(n_interactions), bx, dtype=np.int64))
            interaction_parts.append(np.arange(int(n_interactions), dtype=np.int64))
        if bx_parts:
            return np.column_stack((np.concatenate(bx_parts), np.concatenate(interaction_parts)))

    keys = [
        np.column_stack(
            (
                np.asarray(data["bx_id"], dtype=np.int64),
                np.asarray(data["interaction_id"], dtype=np.int64),
            )
        )
    ]
    if all(name in data.files for name in ("trk_bx_id", "trk_interaction_id")):
        keys.append(
            np.column_stack(
                (
                    np.asarray(data["trk_bx_id"], dtype=np.int64),
                    np.asarray(data["trk_interaction_id"], dtype=np.int64),
                )
            )
        )
    all_keys = np.concatenate(keys, axis=0)
    return np.unique(all_keys, axis=0) if all_keys.size else np.empty((0, 2), dtype=np.int64)


def append_npz(parts, data, names, mask=None):
    for name in names:
        if name in data.files:
            array = np.asarray(data[name])
            if mask is not None:
                array = array[mask]
            parts[name].append(array)


def load_minbias(path, max_files=None, max_bx=None):
    proton_fields = ("bx_id", "interaction_id", "proton_idx", "side", "px", "py", "pz", "E", "m", "pt", "xi")
    track_fields = ("trk_bx_id", "trk_interaction_id", "trk_pt")
    files = discover_npz_files(path, max_files=max_files)
    proton_parts = {name: [] for name in proton_fields}
    track_parts = {name: [] for name in track_fields}
    universe_parts = []

    for filename in files:
        with np.load(filename) as data:
            missing = [name for name in proton_fields if name not in data.files]
            if missing:
                raise RuntimeError(f"Required arrays missing in {filename}: {', '.join(missing)}")
            universe = interaction_universe_from_npz(data, filename)
            if max_bx is not None:
                universe = universe[universe[:, 0] < max_bx]
                proton_mask = np.asarray(data["bx_id"]) < max_bx
                track_mask = np.asarray(data["trk_bx_id"]) < max_bx if "trk_bx_id" in data.files else None
            else:
                proton_mask = None
                track_mask = None
            append_npz(proton_parts, data, proton_fields, mask=proton_mask)
            append_npz(track_parts, data, track_fields, mask=track_mask)
            universe_parts.append(universe)

    protons = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in proton_parts.items()
    }
    tracks = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in track_parts.items()
    }
    universe = (
        np.unique(np.concatenate(universe_parts, axis=0), axis=0)
        if universe_parts
        else np.empty((0, 2), dtype=np.int64)
    )
    bx_ids = sorted(int(bx) for bx in np.unique(universe[:, 0]))
    return files, protons, tracks, universe, bx_ids


def station_hit_counts(xi):
    counts = np.zeros(xi.shape, dtype=np.int32)
    for xi_min, xi_max in STATION_XI.values():
        counts += ((xi >= xi_min) & (xi < xi_max)).astype(np.int32)
    return counts


def encoded_keys(bx, interaction, key_base):
    return np.asarray(bx, dtype=np.int64) * key_base + np.asarray(interaction, dtype=np.int64)


def fill_counts(target, encoded_universe, codes, weights=None):
    if codes.size == 0:
        return
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    if weights is None:
        values = np.bincount(inverse)
    else:
        values = np.bincount(inverse, weights=weights)
    pos = np.searchsorted(encoded_universe, unique_codes)
    valid = (pos < encoded_universe.size) & (encoded_universe[pos] == unique_codes)
    target[pos[valid]] += values[valid]


def build_interaction_arrays(universe, protons, tracks):
    bx = universe[:, 0].astype(np.int32, copy=False)
    interaction = universe[:, 1].astype(np.int32, copy=False)
    max_interaction = int(np.max(interaction)) if interaction.size else 0
    if protons["interaction_id"].size:
        max_interaction = max(max_interaction, int(np.max(protons["interaction_id"])))
    if tracks["trk_interaction_id"].size:
        max_interaction = max(max_interaction, int(np.max(tracks["trk_interaction_id"])))
    key_base = max_interaction + 1
    encoded_universe = encoded_keys(bx, interaction, key_base)
    order = np.argsort(encoded_universe, kind="mergesort")
    encoded_universe = encoded_universe[order]

    n = encoded_universe.size
    n_protons = np.zeros(n, dtype=np.int32)
    n_pps_protons = np.zeros(n, dtype=np.int32)
    n_l1t_tracks = np.zeros(n, dtype=np.int32)
    sum_l1t_track_pt = np.zeros(n, dtype=np.float64)
    sum_l1t_track_pt2 = np.zeros(n, dtype=np.float64)

    proton_codes = encoded_keys(protons["bx_id"], protons["interaction_id"], key_base)
    fill_counts(n_protons, encoded_universe, proton_codes)
    pps_mask = station_hit_counts(np.asarray(protons["xi"], dtype=np.float64)) > 0
    fill_counts(n_pps_protons, encoded_universe, proton_codes[pps_mask])

    if tracks["trk_pt"].size:
        track_codes = encoded_keys(tracks["trk_bx_id"], tracks["trk_interaction_id"], key_base)
        track_pt = np.asarray(tracks["trk_pt"], dtype=np.float64)
        fill_counts(n_l1t_tracks, encoded_universe, track_codes)
        fill_counts(sum_l1t_track_pt, encoded_universe, track_codes, weights=track_pt)
        fill_counts(sum_l1t_track_pt2, encoded_universe, track_codes, weights=track_pt * track_pt)

    return {
        "bx_id": bx[order],
        "interaction_id": interaction[order] + 1,
        "n_protons": n_protons,
        "n_pps_protons": n_pps_protons,
        "n_l1t_tracks": n_l1t_tracks,
        "sum_l1t_track_pt": sum_l1t_track_pt.astype(np.float32),
        "sum_l1t_track_pt2": sum_l1t_track_pt2.astype(np.float32),
    }


def group_pps_proton_indices_by_bx(protons):
    grouped = defaultdict(list)
    pps_mask = station_hit_counts(np.asarray(protons["xi"], dtype=np.float64)) > 0
    pps_idx = np.flatnonzero(pps_mask)
    if pps_idx.size == 0:
        return grouped
    order = np.argsort(protons["bx_id"][pps_idx], kind="mergesort")
    pps_idx = pps_idx[order]
    bx_values = protons["bx_id"][pps_idx]
    unique_bx, starts = np.unique(bx_values, return_index=True)
    stops = np.r_[starts[1:], pps_idx.size]
    for bx, start, stop in zip(unique_bx, starts, stops):
        grouped[int(bx)] = pps_idx[start:stop].tolist()
    return grouped


def signal_sqrt_s(signal_pair, signal_protons):
    if signal_pair is None:
        return DEFAULT_SQRT_S
    xis = [float(proton["xi"]) for proton in signal_protons]
    if len(xis) != 2 or xis[0] <= 0.0 or xis[1] <= 0.0:
        return DEFAULT_SQRT_S
    mass = float(signal_pair["M"])
    if mass <= 0.0:
        return DEFAULT_SQRT_S
    return mass / math.sqrt(xis[0] * xis[1])


def signal_proton_row(proton, bx_id):
    return {
        "bx_id": int(bx_id),
        "interaction_id": 0,
        "proton_id": int(proton["proton_id"]),
        "source": SOURCE_SIGNAL,
        "side": int(proton["side"]),
        "xi": float(proton["xi"]),
        "pt": float(proton["pt"]),
        "eta": float(proton["eta"]),
        "phi": float(proton["phi"]),
        "E": float(proton["E"]),
        "m": float(proton["m"]),
        "hit_192": int(proton["hit_192"]),
        "hit_213": int(proton["hit_213"]),
        "hit_220": int(proton["hit_220"]),
        "hit_420": int(proton["hit_420"]),
        "n_station_hits": int(proton["n_station_hits"]),
    }


def minbias_proton_row(protons, idx):
    xi = float(protons["xi"][idx])
    hits, n_hits = station_hits(xi)
    px = float(protons["px"][idx])
    py = float(protons["py"][idx])
    pz = float(protons["pz"][idx])
    return {
        "bx_id": int(protons["bx_id"][idx]),
        "interaction_id": int(protons["interaction_id"][idx]) + 1,
        "proton_id": int(protons["proton_idx"][idx]),
        "source": SOURCE_MINBIAS,
        "side": int(protons["side"][idx]),
        "xi": xi,
        "pt": float(protons["pt"][idx]),
        "eta": eta_from_components(px, py, pz),
        "phi": phi_from_components(px, py),
        "E": float(protons["E"][idx]),
        "m": float(protons["m"][idx]),
        "hit_192": hits["192"],
        "hit_213": hits["213"],
        "hit_220": hits["220"],
        "hit_420": hits["420"],
        "n_station_hits": n_hits,
    }


def pair_from_protons(bx_id, pair_id, left, right, sqrt_s):
    pass_pps = int(left["n_station_hits"] > 0 and right["n_station_hits"] > 0)
    xi_left = float(left["xi"])
    xi_right = float(right["xi"])
    mass = math.sqrt(xi_left * xi_right) * sqrt_s if xi_left > 0.0 and xi_right > 0.0 else 0.0
    rapidity = 0.5 * math.log(xi_right / xi_left) if xi_left > 0.0 and xi_right > 0.0 else 0.0
    n_signal = int(left["source"] == SOURCE_SIGNAL) + int(right["source"] == SOURCE_SIGNAL)
    return {
        "bx_id": int(bx_id),
        "pair_id": int(pair_id),
        "interaction_id_L": int(left["interaction_id"]),
        "interaction_id_R": int(right["interaction_id"]),
        "proton_id_L": int(left["proton_id"]),
        "proton_id_R": int(right["proton_id"]),
        "source_L": int(left["source"]),
        "source_R": int(right["source"]),
        "M": float(mass),
        "y": float(rapidity),
        "pass_pps": pass_pps,
        "n_signal_protons": n_signal,
    }


def append_central(rows, bx_id, event_id, interaction):
    row = {"bx_id": int(bx_id), "interaction_id": 0, "event_id": int(event_id)}
    for name in (
        "j1_pid",
        "j2_pid",
        "j1_pt",
        "j1_eta",
        "j1_phi",
        "j1_E",
        "j1_m",
        "j2_pt",
        "j2_eta",
        "j2_phi",
        "j2_E",
        "j2_m",
        "jj_pt",
        "jj_eta",
        "jj_phi",
        "jj_E",
        "jj_m",
        "jj_y",
    ):
        row[name] = interaction[name]
    rows.append(row)


def build_process_output(process, campaign, minbias, output_dir, args):
    campaign_dir, campaign_key = resolve_process_campaign(process, campaign)
    signal_path = campaign_dir / "processed" / "processed.root"
    out_path = output_dir / f"overlapped_gen_{process}.root"

    event_ids, interactions, signal_protons, signal_pairs = load_signal(
        signal_path,
        max_events=args.max_overlays,
    )
    n_overlays = min(len(event_ids), len(minbias["bx_ids"]))
    if args.max_overlays is not None:
        n_overlays = min(n_overlays, args.max_overlays)
    if n_overlays <= 0:
        raise RuntimeError(f"No overlays available for process {process}")

    rows = {
        "Central": [],
        "Protons": [],
        "ProtonPairs": [],
    }

    for overlay_idx in range(n_overlays):
        event_id = event_ids[overlay_idx]
        bx_id = minbias["bx_ids"][overlay_idx]
        sig_protons = sorted(signal_protons[event_id], key=lambda proton: int(proton["side"]))
        sig_rows = [signal_proton_row(proton, bx_id) for proton in sig_protons]
        sqrt_s = signal_sqrt_s(signal_pairs.get(event_id), sig_protons)

        append_central(rows["Central"], bx_id, event_id, interactions[event_id])
        rows["Protons"].extend(sig_rows)

        mb_rows = []
        for idx in minbias["pps_proton_by_bx"].get(bx_id, []):
            row = minbias_proton_row(minbias["protons"], idx)
            mb_rows.append(row)
        rows["Protons"].extend(mb_rows)

        left_signal = next((row for row in sig_rows if row["side"] < 0), None)
        right_signal = next((row for row in sig_rows if row["side"] > 0), None)
        if left_signal is not None and right_signal is not None:
            signal_pair = pair_from_protons(bx_id, 0, left_signal, right_signal, sqrt_s)
            existing_pair = signal_pairs.get(event_id)
            if existing_pair is not None:
                signal_pair["M"] = float(existing_pair["M"])
                signal_pair["y"] = float(existing_pair["y"])
                signal_pair["pass_pps"] = int(existing_pair["pass_pps"])
            rows["ProtonPairs"].append(signal_pair)

        candidates = sig_rows + mb_rows
        left = [row for row in candidates if row["side"] < 0]
        right = [row for row in candidates if row["side"] > 0]
        pair_id = 1
        for left_row in left:
            for right_row in right:
                if left_row["source"] == SOURCE_SIGNAL and right_row["source"] == SOURCE_SIGNAL:
                    continue
                pair = pair_from_protons(bx_id, pair_id, left_row, right_row, sqrt_s)
                if pair["pass_pps"] <= 0:
                    continue
                rows["ProtonPairs"].append(pair)
                pair_id += 1

    interaction_mask = np.isin(minbias["interaction_arrays"]["bx_id"], minbias["bx_ids"][:n_overlays])
    interaction_arrays = {
        name: values[interaction_mask]
        for name, values in minbias["interaction_arrays"].items()
    }
    write_output(out_path, rows, interaction_arrays)
    print(
        f"{process}: wrote {n_overlays} overlays using campaign {campaign_key} to {out_path}",
        flush=True,
    )


def write_output(out_path, rows, interaction_arrays):
    import uproot

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(out_path) as root_file:
        root_file["MinbiasInteractions"] = interaction_arrays
        for tree_name, tree_rows in rows.items():
            arrays = root_arrays(tree_rows)
            if arrays:
                root_file[tree_name] = arrays


def selected_processes(args):
    processes = load_yaml(REPO_ROOT / "processes.yaml")
    names = args.processes or sorted(processes)
    missing = [name for name in names if name not in processes]
    if missing:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(missing)}. Known processes: {known}")
    if args.max_processes is not None:
        names = names[: args.max_processes]
    return names


def minbias_input_path(args):
    if args.minbias_input:
        return resolve_path(args.minbias_input, base=REPO_ROOT)
    path, campaign = resolve_minbias_campaign()
    print(f"Minbias campaign: {campaign}", flush=True)
    return path


def main():
    args = parse_args()
    try:
        minbias_path = minbias_input_path(args)
        process_names = selected_processes(args)
        files, protons, tracks, universe, bx_ids = load_minbias(
            minbias_path,
            max_files=args.max_minbias_files,
            max_bx=args.max_overlays,
        )
        print(
            f"Loaded {len(files)} minbias file(s), {len(bx_ids)} BX, "
            f"{len(protons['xi'])} protons",
            flush=True,
        )
        print("Building minbias interaction summaries...", flush=True)
        interaction_arrays = build_interaction_arrays(universe, protons, tracks)
        print("Indexing PPS minbias protons by BX...", flush=True)
        pps_proton_by_bx = group_pps_proton_indices_by_bx(protons)
        minbias = {
            "protons": protons,
            "tracks": tracks,
            "bx_ids": bx_ids,
            "interaction_arrays": interaction_arrays,
            "pps_proton_by_bx": pps_proton_by_bx,
        }
        output_dir = resolve_path(args.output_dir, base=REPO_ROOT)
        for process in process_names:
            build_process_output(process, args.campaign, minbias, output_dir, args)
    except RuntimeError as err:
        raise SystemExit(f"ERROR: {err}") from None


if __name__ == "__main__":
    main()
