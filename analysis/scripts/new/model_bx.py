#!/usr/bin/env python3
import argparse
import importlib.util
from collections import Counter

import awkward as ak
import numpy as np

try:
    from common import (
        CENTRAL_SOURCE,
        MINBIAS_SOURCE,
        build_central_record,
        build_interaction_summaries,
        build_pairs,
        counter_summary,
        load_hard_events,
        load_minbias,
        load_yaml,
        outgoing_protons,
        repo_root,
        resolve_path,
        selected_bx_values,
        select_hard_jets,
        smear_xi,
        station_tag_counts,
    )
except ImportError:
    from .common import (
        CENTRAL_SOURCE,
        MINBIAS_SOURCE,
        build_central_record,
        build_interaction_summaries,
        build_pairs,
        counter_summary,
        load_hard_events,
        load_minbias,
        load_yaml,
        outgoing_protons,
        repo_root,
        resolve_path,
        selected_bx_values,
        select_hard_jets,
        smear_xi,
        station_tag_counts,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a nested Awkward BX overlay dataset from hard LHE/DAT events and minbias NPZ files."
    )
    parser.add_argument(
        "--config",
        default="analysis/scripts/new/config.yaml",
        help="YAML analysis config.",
    )
    parser.add_argument(
        "--processes",
        default="analysis/scripts/new/process.yaml",
        help="YAML process/sample config.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Sample names to process. Defaults to all samples with existing paths.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Maximum hard events per sample.")
    parser.add_argument("--max-bx", type=int, default=None, help="Maximum BX values to overlay.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum hard-sample files per sample.")
    parser.add_argument("--max-minbias-files", type=int, default=None, help="Maximum minbias NPZ files.")
    parser.add_argument("--output", default=None, help="Override output ROOT path.")
    return parser.parse_args()


def require_root_dependencies():
    missing = [
        name
        for name in ("awkward", "uproot")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "ROOT output requires missing Python package(s): "
            + ", ".join(missing)
            + ". Run after the project environment is set up."
        )


def output_path(config, override=None):
    if override is not None:
        return resolve_path(override)
    output_cfg = config.get("output", {})
    out_dir = resolve_path(output_cfg.get("dir", "analysis/output"))
    name = output_cfg.get("name", "bx_overlay")
    if not str(name).endswith(".root"):
        name = f"{name}.root"
    return out_dir / name


def selected_samples(processes, sample_names):
    if sample_names is None:
        out = []
        for name, cfg in processes.items():
            path = cfg.get("path")
            if path is not None and resolve_path(path).exists():
                out.append(name)
        if not out:
            raise RuntimeError("No process paths exist. Pass --samples or update process.yaml.")
        return out
    missing = [name for name in sample_names if name not in processes]
    if missing:
        raise RuntimeError(f"Unknown sample(s): {', '.join(missing)}")
    return sample_names


def group_indices_by_bx(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    unique, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    return {int(v): order[start : start + count] for v, start, count in zip(unique, starts, counts)}


def hard_event_record(
    overlay_id,
    hard_event_id,
    sample_name,
    sample_cfg,
    event,
    config,
    counters,
):
    header, particles = event
    jets = select_hard_jets(sample_name, sample_cfg, particles)
    if jets is None:
        counters["skip_bad_jets"] += 1
        return None
    sqrt_s_gev = float(config["beam"]["sqrt_s_gev"])
    central_protons = outgoing_protons(particles, sqrt_s_gev)
    if central_protons is None:
        counters["skip_bad_central_protons"] += 1
        return None

    seed = int(config.get("random", {}).get("seed", 12345))
    rng = np.random.default_rng(seed + overlay_id)
    central = build_central_record(
        hard_event_id,
        header,
        jets,
        config.get("central_smearing", {}),
        rng,
    )
    return central, central_protons


def add_proton_record(protons, proton, xi_ranges, xi_res, rng, keep_any_pps=True):
    xi_reco = smear_xi(float(proton["xi_truth"]), xi_res, rng)
    reco_pass, reco_n_400 = station_tag_counts(xi_reco, xi_ranges)
    if keep_any_pps and not reco_pass:
        return

    record = {
        "proton_id": len(protons),
        "source": proton["source"],
        "interaction_id": int(proton["interaction_id"]),
        "input_proton_idx": int(proton["input_proton_idx"]),
        "side": int(proton["side"]),
        "px": float(proton["px"]),
        "py": float(proton["py"]),
        "pz": float(proton["pz"]),
        "E": float(proton["E"]),
        "pt": float(proton["pt"]),
        "xi_truth": float(proton["xi_truth"]),
        "xi_reco": xi_reco,
        "passes_pps": bool(reco_pass),
        "n_400_tags": int(reco_n_400),
    }
    protons.append(record)


def build_minbias_protons_for_bx(bx, protons, proton_group_by_bx):
    idx = proton_group_by_bx.get(bx)
    if idx is None:
        return []
    out = []
    for i in idx:
        out.append(
            {
                "input_proton_idx": int(protons["proton_idx"][i]),
                "source": MINBIAS_SOURCE,
                "interaction_id": int(protons["interaction_id"][i]),
                "side": int(protons["side"][i]),
                "px": float(protons["px"][i]),
                "py": float(protons["py"][i]),
                "pz": float(protons["pz"][i]),
                "E": float(protons["E"][i]),
                "pt": float(protons["pt"][i]),
                "xi_truth": float(protons["xi"][i]),
            }
        )
    return out


def build_overlays(config, processes, sample_names, args):
    input_cfg = config.get("inputs", {})
    minbias_path = input_cfg.get("minbias_path")
    if minbias_path is None:
        raise RuntimeError("config.yaml must set inputs.minbias_path")

    minbias_files, mb_protons, mb_tracks, universe, minbias_metadata = load_minbias(
        minbias_path,
        max_files=args.max_minbias_files,
    )
    bx_values = selected_bx_values(universe, max_bx=args.max_bx)
    if not bx_values:
        raise RuntimeError("No minbias BX values available.")

    interactions_by_bx = build_interaction_summaries(
        universe,
        mb_tracks,
        float(config.get("tracks", {}).get("pt_min", 2.0)),
        float(config.get("tracks", {}).get("eta_max", 2.4)),
    )
    proton_group_by_bx = group_indices_by_bx(mb_protons["bx_id"])

    xi_ranges = config["pps"]["xi_ranges"]
    xi_res = float(config["pps"].get("xi_res", 0.0))
    sqrt_s_gev = float(config["beam"]["sqrt_s_gev"])
    seed = int(config.get("random", {}).get("seed", 12345))

    overlays = []
    total_counters = Counter()
    overlay_id = 0
    for sample_name in sample_names:
        sample_cfg = processes[sample_name]
        sample_path = sample_cfg.get("path")
        if sample_path is None:
            raise RuntimeError(f"Sample {sample_name} has no path in process.yaml")

        files, hard_events, xsec_from_file = load_hard_events(sample_path, max_files=args.max_files)
        if args.max_events is not None:
            hard_events = hard_events[: args.max_events]
        n_to_process = min(len(hard_events), len(bx_values))
        sample_counters = Counter()

        for hard_event_id in range(n_to_process):
            bx = bx_values[hard_event_id]
            hard = hard_event_record(
                overlay_id,
                hard_event_id,
                sample_name,
                sample_cfg,
                hard_events[hard_event_id],
                config,
                sample_counters,
            )
            if hard is None:
                continue
            central, central_protons = hard

            rng = np.random.default_rng(seed + overlay_id * 1000 + 2)
            proton_rows = []
            for proton in central_protons:
                add_proton_record(proton_rows, proton, xi_ranges, xi_res, rng, keep_any_pps=False)
            for proton in build_minbias_protons_for_bx(bx, mb_protons, proton_group_by_bx):
                add_proton_record(proton_rows, proton, xi_ranges, xi_res, rng, keep_any_pps=True)

            pair_rows = build_pairs(proton_rows, sqrt_s_gev)
            interactions = interactions_by_bx.get(bx, [])
            overlays.append(
                {
                    "overlay_id": int(overlay_id),
                    "hard_event_id": int(hard_event_id),
                    "mb_bx_id": int(bx),
                    "sample": sample_name,
                    "sample_type": sample_cfg.get("sample_type", "unknown"),
                    "n_interactions": int(len(interactions)),
                    "n_protons": int(len(proton_rows)),
                    "n_pairs": int(len(pair_rows)),
                    "sqrt_s_gev": sqrt_s_gev,
                    "central": central,
                    "interactions": interactions,
                    "protons": proton_rows,
                    "pairs": pair_rows,
                }
            )
            overlay_id += 1
            sample_counters["overlays_written"] += 1
            sample_counters["protons_written"] += len(proton_rows)
            sample_counters["pairs_written"] += len(pair_rows)

        total_counters.update(sample_counters)
        xsec = sample_cfg.get("xsec_pb", xsec_from_file)
        xsec_text = "none" if xsec is None else f"{float(xsec):.6e} pb"
        print(
            f"[{sample_name}] files={len(files)}, hard_events={len(hard_events)}, "
            f"processed={n_to_process}, xsec={xsec_text}, {counter_summary(sample_counters)}"
        )

    print(f"[minbias] files={len(minbias_files)}, bx_available={len(bx_values)}, metadata={minbias_metadata}")
    print(f"[total] {counter_summary(total_counters)}")
    return ak.Array(overlays)


def append_prefixed_record(columns, prefix, record):
    for key, value in record.items():
        if isinstance(value, dict):
            append_prefixed_record(columns, f"{prefix}{key}_", value)
        else:
            columns.setdefault(f"{prefix}{key}", []).append(value)


def tree_arrays(columns):
    arrays = {}
    for name, values in columns.items():
        if not values:
            continue
        if isinstance(values[0], str):
            arrays[name] = np.asarray(values, dtype=str)
        elif isinstance(values[0], bool):
            arrays[name] = np.asarray(values, dtype=np.bool_)
        elif isinstance(values[0], int):
            arrays[name] = np.asarray(values, dtype=np.int32)
        else:
            arrays[name] = np.asarray(values, dtype=np.float64)
    return arrays


def flatten_for_root(array):
    events = {}
    central = {}
    interactions = {}
    protons = {}
    pairs = {}

    for overlay in array.tolist():
        event_columns = (
            "overlay_id",
            "hard_event_id",
            "mb_bx_id",
            "sample",
            "sample_type",
            "n_interactions",
            "n_protons",
            "n_pairs",
            "sqrt_s_gev",
        )
        for key in event_columns:
            events.setdefault(key, []).append(overlay[key])

        central.setdefault("overlay_id", []).append(overlay["overlay_id"])
        append_prefixed_record(central, "", overlay["central"])

        for interaction in overlay["interactions"]:
            interactions.setdefault("overlay_id", []).append(overlay["overlay_id"])
            append_prefixed_record(interactions, "", interaction)

        for proton in overlay["protons"]:
            protons.setdefault("overlay_id", []).append(overlay["overlay_id"])
            append_prefixed_record(protons, "", proton)

        for pair in overlay["pairs"]:
            pairs.setdefault("overlay_id", []).append(overlay["overlay_id"])
            append_prefixed_record(pairs, "", pair)

    return {
        "Events": tree_arrays(events),
        "Central": tree_arrays(central),
        "Interactions": tree_arrays(interactions),
        "Protons": tree_arrays(protons),
        "Pairs": tree_arrays(pairs),
    }


def write_root(array, out_path):
    import uproot

    trees = flatten_for_root(array)
    with uproot.recreate(out_path) as fout:
        for tree_name, arrays in trees.items():
            if arrays:
                fout[tree_name] = arrays


def main():
    args = parse_args()
    require_root_dependencies()
    root = repo_root()
    config_path = resolve_path(args.config, base=root)
    process_path = resolve_path(args.processes, base=root)
    config = load_yaml(config_path)
    processes = load_yaml(process_path)
    sample_names = selected_samples(processes, args.samples)
    array = build_overlays(config, processes, sample_names, args)

    out_path = output_path(config, override=args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_root(array, out_path)
    print(f"Wrote {len(array)} overlays to {out_path}")


if __name__ == "__main__":
    main()
