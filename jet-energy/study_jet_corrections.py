#!/usr/bin/env python3
"""Derive and validate JetPUPPI corrections from reusable jet caches."""

import argparse
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import jet_cache  # noqa: E402
from common import jet_calibration as calibration  # noqa: E402


# Validation bins are finer than, and share no interior edge with, the
# calibration bins, so closure is not evaluated in the binning it was fitted in.
PT_EDGES = (8.0, 13.0, 17.0, 21.0, 26.0, 31.0, 37.0, 45.0, 55.0, 70.0, 95.0, 130.0)
CALIBRATION_GEN_PT_EDGES = (12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 50.0, 65.0, 90.0)
RAW_PT_SUPPORT = (8.0, 250.0)
# Derivation eta bins follow the PUPPI eta regions in the Delphes card, which
# change behaviour at 1.5; validation uses finer bins so closure is not
# evaluated in the binning the map was derived in.
ETA_EDGES = (0.0, 0.8, 1.5, 2.0, 2.5, 3.0)
VALIDATION_ETA_EDGES = (0.0, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0)
# Dijet residuals are evaluated on a generator-level selection. Selecting on
# reconstructed pT keeps upward-fluctuated jets and biases the closure by an
# amount that depends on the pT spectrum, so it differs between samples.
CLOSURE_GEN_PT_MIN = 30.0
DEFAULT_OUTPUT = HERE / "output/stage2_3"
DEFAULT_RESOLUTION = HERE / "L1_jet_energy_resolution.yaml"
PAIR_FIELDS = (
    "gen_pt",
    "gen_eta",
    "gen_phi",
    "gen_mass",
    "reco_pt",
    "reco_eta",
    "reco_phi",
    "reco_mass",
    "match_dr",
)
PROFILE_FIELDS = ("leading_pt", "subleading_pt", "dijet_mass")
RESIDUAL_FIELDS = ("mass_ratio", "delta_pt", "delta_px", "delta_py")
JET_ACTIVITY_PT_MIN = 20.0
JET_ACTIVITY_ETA_MAX = 2.4
JET_ACTIVITY_SELECTIONS = ("truth_matched", "two_hardest")
JET_ACTIVITY_FIELDS = (
    "leading_pt",
    "subleading_pt",
    "all_pt",
    "dijet_ht_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-yaml", type=Path, default=jet_cache.DEFAULT_DATASETS
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="SAMPLE:FSR_STATE",
        help="Only study the named dataset; repeat to select several",
    )
    parser.add_argument("--cache-dir", type=Path, default=jet_cache.DEFAULT_CACHE)
    parser.add_argument("--resolution-yaml", type=Path, default=DEFAULT_RESOLUTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional event cap per sample and phase",
    )
    parser.add_argument("--min-bin-entries", type=int, default=30)
    parser.add_argument("--bootstrap-replicas", type=int, default=30)
    parser.add_argument("--smear-replicas", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_JET_CALIBRATION_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_JET_CALIBRATION_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def validate_args(args):
    positive = {
        "--min-bin-entries": args.min_bin_entries,
        "--bootstrap-replicas": args.bootstrap_replicas,
        "--smear-replicas": args.smear_replicas,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise RuntimeError(f"{', '.join(invalid)} must be positive")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be positive")


def selected_cache_sets(args, datasets):
    selected = {}
    for dataset in datasets:
        fsr_state, sample = jet_cache.dataset_key(dataset)
        derivation_files = dataset["derivation_files"]
        validation_files = dataset["validation_files"]
        selected[(fsr_state, sample)] = {
            "derivation": jet_cache.cache_files(
                args.cache_dir, fsr_state, sample, 0, derivation_files
            ),
            "validation": jet_cache.cache_files(
                args.cache_dir,
                fsr_state,
                sample,
                derivation_files,
                validation_files,
            ),
        }
    return selected


def cache_settings(records):
    keys = ("eta_max", "match_dr_max", "truth_genjet_dr_max")
    settings = {key: records[0]["metadata"][key] for key in keys}
    for record in records[1:]:
        different = [
            key for key in keys if record["metadata"].get(key) != settings[key]
        ]
        if different:
            raise RuntimeError(
                f"Inconsistent {', '.join(different)} in {record['path']}; rebuild "
                "the selected caches with one matching configuration"
            )
    return settings


def jet_arrays(record, prefix, event_index):
    start, stop = record[f"{prefix}_offsets"][event_index : event_index + 2]
    return {
        field: record[f"{prefix}_{field}"][start:stop]
        for field in jet_cache.JET_FIELDS
    }


def jets_from_arrays(arrays):
    return [
        {field: float(arrays[field][index]) for field in jet_cache.JET_FIELDS}
        for index in range(arrays["pt"].size)
    ]


def event_matches(record, event_index, prefix="match"):
    start, stop = record[f"{prefix}_offsets"][event_index : event_index + 2]
    return [
        (
            int(record[f"{prefix}_gen_index"][index]),
            int(record[f"{prefix}_reco_index"][index]),
            float(record[f"{prefix}_dr"][index]),
        )
        for index in range(start, stop)
    ]


def iter_events(dataset):
    for record, event_count in dataset["sources"]:
        for event_index in range(event_count):
            yield {
                "gen": jets_from_arrays(jet_arrays(record, "gen", event_index)),
                "reco": jets_from_arrays(jet_arrays(record, "reco", event_index)),
                "hard_flavor_gen_indices": [
                    int(index)
                    for index in record["hard_flavor_gen_index"][
                        slice(
                            *record["hard_flavor_gen_offsets"][
                                event_index : event_index + 2
                            ]
                        )
                    ]
                ],
                "hard_flavor_matches": event_matches(
                    record, event_index, "hard_flavor_match"
                ),
            }


def load_dataset(np, dataset_config, paths, max_events):
    fsr_state, sample = jet_cache.dataset_key(dataset_config)
    started = time.perf_counter()
    records = []
    for index, path in enumerate(paths, start=1):
        print(
            f"  loading cache {fsr_state}/{sample} {index}/{len(paths)}: "
            f"{Path(path).name}",
            flush=True,
        )
        records.append(jet_cache.load_cache(path))
    settings = cache_settings(records)
    for record in records:
        metadata = record["metadata"]
        if metadata["fsr_state"] != fsr_state or metadata["sample"] != sample:
            raise RuntimeError(
                f"Cache identity mismatch in {record['path']}: expected "
                f"{fsr_state}/{sample}"
            )
        expected_identity = {
            "match_selection": dataset_config["match_selection"],
            "truth_pid_abs": dataset_config["truth_pid_abs"],
            "closure_mode": dataset_config["closure_mode"],
        }
        if any(metadata.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError(
                f"Cache selection mismatch in {record['path']}; rebuild from "
                f"{dataset_config['manifest_path']}"
            )

    columns = {field: [] for field in PAIR_FIELDS}
    sources = []
    events_read = 0
    selection_pair_events = 0
    selected_genjets = 0
    matched_puppijets = 0
    complete_genjet_pair_events = 0
    complete_puppi_pair_events = 0
    hard_flavor = dataset_config["match_selection"] == "hard_flavor"
    match_prefix = "hard_flavor_match" if hard_flavor else "match"
    for record_index, record in enumerate(records, start=1):
        available = int(record["metadata"]["events_read"])
        if max_events is not None:
            available = min(available, max_events - events_read)
        if available <= 0:
            break
        sources.append((record, available))
        for event_index in range(available):
            gen = jet_arrays(record, "gen", event_index)
            reco = jet_arrays(record, "reco", event_index)
            start, stop = record[f"{match_prefix}_offsets"][
                event_index : event_index + 2
            ]
            gen_indices = record[f"{match_prefix}_gen_index"][start:stop]
            reco_indices = record[f"{match_prefix}_reco_index"][start:stop]
            for field in jet_cache.JET_FIELDS:
                columns[f"gen_{field}"].extend(gen[field][gen_indices])
                columns[f"reco_{field}"].extend(reco[field][reco_indices])
            columns["match_dr"].extend(record[f"{match_prefix}_dr"][start:stop])
        gen_prefix = "hard_flavor_gen" if hard_flavor else "gen"
        gen_counts = np.diff(record[f"{gen_prefix}_offsets"][: available + 1])
        match_counts = np.diff(record[f"{match_prefix}_offsets"][: available + 1])
        if hard_flavor:
            selection_pair_events += int(
                np.count_nonzero(record["hard_flavor_pair_event"][:available])
            )
        else:
            selection_pair_events += int(np.count_nonzero(gen_counts >= 2))
        selected_genjets += int(np.sum(gen_counts))
        matched_puppijets += int(np.sum(match_counts))
        complete_genjet_pair_events += int(np.count_nonzero(gen_counts == 2))
        complete_puppi_pair_events += int(np.count_nonzero(match_counts == 2))
        events_read += available
        elapsed = time.perf_counter() - started
        rate = events_read / elapsed if elapsed > 0.0 else 0.0
        print(
            f"  processed cache {fsr_state}/{sample} {record_index}/{len(records)}: "
            f"events={events_read:,} matched_jets={len(columns['match_dr']):,} "
            f"elapsed={elapsed:.1f}s rate={rate:.1f} events/s",
            flush=True,
        )

    if events_read == 0:
        raise RuntimeError(f"No cached events available for {fsr_state}/{sample}")
    return {
        "fsr_state": fsr_state,
        "sample": sample,
        "files": [record["metadata"]["source_path"] for record, _count in sources],
        "cache_files": [record["path"] for record, _count in sources],
        "events_read": events_read,
        "pair_selection": (
            f"hard_process_pid{dataset_config['truth_pid_abs']}_matched_genjets"
            if hard_flavor
            else "inclusive_hard_interaction_genjets"
        ),
        "match_selection": dataset_config["match_selection"],
        "truth_pid_abs": dataset_config["truth_pid_abs"],
        "closure_mode": dataset_config["closure_mode"],
        "selection_pair_events": selection_pair_events,
        "selected_genjets": selected_genjets,
        "matched_puppijets": matched_puppijets,
        "complete_genjet_pair_events": complete_genjet_pair_events,
        "complete_puppi_pair_events": complete_puppi_pair_events,
        "pairs": {
            field: np.asarray(values, dtype=np.float64)
            for field, values in columns.items()
        },
        "sources": sources,
        **settings,
    }


def bootstrap_node(np, gen_pt, reco_pt, rng, replicas):
    response = reco_pt / gen_pt
    estimates = np.empty(replicas, dtype=np.float64)
    for index in range(replicas):
        choice = rng.integers(0, response.size, size=response.size)
        estimates[index] = 1.0 / np.median(response[choice])
    return [float(value) for value in np.quantile(estimates, [0.16, 0.84])]


def discard_nonmonotonic_nodes(nodes):
    """Discard low-statistics nodes until raw and target medians both increase."""
    while True:
        usable = [index for index, node in enumerate(nodes) if node["usable"]]
        violation = next(
            (
                (left, right)
                for left, right in zip(usable[:-1], usable[1:])
                if nodes[right]["raw_reco_pt_median"]
                <= nodes[left]["raw_reco_pt_median"]
                or nodes[right]["target_pt"] <= nodes[left]["target_pt"]
            ),
            None,
        )
        if violation is None:
            break
        left, right = violation
        remove = left if nodes[left]["entries"] < nodes[right]["entries"] else right
        nodes[remove]["usable"] = False
        nodes[remove]["invalid_reason"] = "nonmonotonic_raw_median"

    usable = [node for node in nodes if node["usable"]]
    if len(usable) < 2:
        for node in usable:
            node["usable"] = False
            node["invalid_reason"] = "insufficient_monotonic_nodes"


def derive_correction_map(np, dataset, args, rng):
    pairs = dataset["pairs"]
    abs_eta = np.abs(pairs["reco_eta"])
    eta_tables = []
    for eta_index, (eta_min, eta_max) in enumerate(zip(ETA_EDGES[:-1], ETA_EDGES[1:])):
        nodes = []
        for pt_index, (pt_min, pt_max) in enumerate(
            zip(CALIBRATION_GEN_PT_EDGES[:-1], CALIBRATION_GEN_PT_EDGES[1:])
        ):
            mask = (
                (abs_eta >= eta_min)
                & (abs_eta < eta_max)
                & (pairs["gen_pt"] >= pt_min)
                & (pairs["gen_pt"] < pt_max)
            )
            gen_pt = pairs["gen_pt"][mask]
            reco_pt = pairs["reco_pt"][mask]
            node = {
                "pt_bin": pt_index,
                "gen_pt_min": pt_min,
                "gen_pt_max": pt_max,
                "entries": int(gen_pt.size),
                "gen_pt_median": None,
                "raw_reco_pt_median": None,
                "raw_response_median": None,
                "target_pt": None,
                "correction_factor": None,
                "correction_ci68": [None, None],
                "usable": False,
                "invalid_reason": "insufficient_entries",
            }
            if gen_pt.size >= args.min_bin_entries:
                response_median = float(np.median(reco_pt / gen_pt))
                gen_pt_median = float(np.median(gen_pt))
                raw_reco_pt_median = float(np.median(reco_pt))
                node.update(
                    {
                        "gen_pt_median": gen_pt_median,
                        "raw_reco_pt_median": raw_reco_pt_median,
                        "raw_response_median": response_median,
                        "target_pt": raw_reco_pt_median / response_median,
                        "correction_factor": 1.0 / response_median,
                        "correction_ci68": bootstrap_node(
                            np, gen_pt, reco_pt, rng, args.bootstrap_replicas
                        ),
                        "usable": True,
                        "invalid_reason": None,
                    }
                )
            nodes.append(node)

        discard_nonmonotonic_nodes(nodes)
        usable = [node for node in nodes if node["usable"]]
        eta_tables.append(
            {
                "eta_bin": eta_index,
                "eta_min": eta_min,
                "eta_max": eta_max,
                "supported": bool(usable),
                "measured_raw_pt_support": (
                    [
                        usable[0]["raw_reco_pt_median"],
                        usable[-1]["raw_reco_pt_median"],
                    ]
                    if usable
                    else [None, None]
                ),
                "nodes": nodes,
            }
        )
    return {
        "collection": "JetPUPPI",
        "fsr_state": dataset["fsr_state"],
        "source_sample": dataset["sample"],
        "derivation_files": dataset["files"],
        "derivation_cache_files": dataset.get("cache_files", []),
        "events_read": dataset["events_read"],
        "matched_jets": int(pairs["gen_pt"].size),
        "pair_selection": dataset["pair_selection"],
        "match_selection": dataset["match_selection"],
        "truth_pid_abs": dataset["truth_pid_abs"],
        "closure_mode": dataset["closure_mode"],
        "selection_pair_events": dataset["selection_pair_events"],
        "selected_genjets": dataset["selected_genjets"],
        "matched_puppijets": dataset["matched_puppijets"],
        "complete_genjet_pair_events": dataset["complete_genjet_pair_events"],
        "complete_puppi_pair_events": dataset["complete_puppi_pair_events"],
        "truth_genjet_dr_max": dataset.get("truth_genjet_dr_max"),
        "match_dr_max": dataset.get("match_dr_max", getattr(args, "match_dr_max", 0.2)),
        "calibration_gen_pt_edges": list(CALIBRATION_GEN_PT_EDGES),
        "raw_pt_support": list(RAW_PT_SUPPORT),
        "abs_eta_edges": list(ETA_EDGES),
        "validation_abs_eta_edges": list(VALIDATION_ETA_EDGES),
        "min_bin_entries": args.min_bin_entries,
        "eta_bins": eta_tables,
    }


def evaluate_response_bins(
    np, pairs, correction_map, resolution_config, args, bootstrap_rng, smear_rng
):
    factors, valid = calibration.correction_factors(
        pairs["reco_pt"], pairs["reco_eta"], correction_map
    )
    raw_response = pairs["reco_pt"] / pairs["gen_pt"]
    corrected_response = factors * raw_response
    abs_eta = np.abs(pairs["gen_eta"])
    rows = []
    for eta_index, (eta_min, eta_max) in enumerate(
        zip(VALIDATION_ETA_EDGES[:-1], VALIDATION_ETA_EDGES[1:])
    ):
        for pt_index, (pt_min, pt_max) in enumerate(zip(PT_EDGES[:-1], PT_EDGES[1:])):
            selected = (
                (abs_eta >= eta_min)
                & (abs_eta < eta_max)
                & (pairs["gen_pt"] >= pt_min)
                & (pairs["gen_pt"] < pt_max)
            )
            corrected_selected = selected & valid
            gen_pt = pairs["gen_pt"][selected]
            gen_eta = pairs["gen_eta"][selected]
            raw_values = raw_response[corrected_selected]
            corrected_values = corrected_response[corrected_selected]
            smeared_values = []
            expected_sigma = None
            gen_pt_median = float(np.median(gen_pt)) if gen_pt.size else None
            if gen_pt.size:
                smeared = calibration.smear_gen_jets(
                    gen_pt,
                    pairs["gen_mass"][selected],
                    gen_eta,
                    smear_rng,
                    resolution_config,
                    args.smear_replicas,
                )
                smeared_values = smeared["factor"].reshape(-1)
                sigma, sigma_valid = calibration.l1_relative_sigma(
                    np.asarray([gen_pt_median]),
                    np.asarray([float(np.median(np.abs(gen_eta)))]),
                    resolution_config,
                )
                expected_sigma = float(sigma[0]) if sigma_valid[0] else None
            matched = int(np.count_nonzero(selected))
            corrected = int(np.count_nonzero(corrected_selected))
            rows.append(
                {
                    "eta_bin": eta_index,
                    "eta_min": eta_min,
                    "eta_max": eta_max,
                    "pt_bin": pt_index,
                    "gen_pt_min": pt_min,
                    "gen_pt_max": pt_max,
                    "gen_pt_median": gen_pt_median,
                    "matched_jets": matched,
                    "corrected_jets": corrected,
                    "correction_coverage": corrected / matched if matched else None,
                    "raw": calibration.response_metrics_with_ci(
                        raw_values, bootstrap_rng, args.bootstrap_replicas, True
                    ),
                    "corrected": calibration.response_metrics_with_ci(
                        corrected_values,
                        bootstrap_rng,
                        args.bootstrap_replicas,
                        True,
                    ),
                    "l1_expected_relative_sigma": expected_sigma,
                    "l1_smeared": calibration.response_metrics_with_ci(
                        smeared_values,
                        bootstrap_rng,
                        args.bootstrap_replicas,
                        False,
                    ),
                }
            )
    return rows, factors, valid


def evaluate_raw_pt_response_bins(np, pairs, correction_map, args, bootstrap_rng):
    """Measure closure in common raw-pT diagnostic bins."""
    factors, valid = calibration.correction_factors(
        pairs["reco_pt"], pairs["reco_eta"], correction_map
    )
    raw_response = pairs["reco_pt"] / pairs["gen_pt"]
    corrected_response = factors * raw_response
    abs_eta = np.abs(pairs["reco_eta"])
    rows = []
    for eta_index, eta_bin in enumerate(correction_map["eta_bins"]):
        eta_selected = (
            (abs_eta >= eta_bin["eta_min"])
            & (abs_eta < eta_bin["eta_max"])
        )
        for pt_index, (pt_min, pt_max) in enumerate(
            zip(CALIBRATION_GEN_PT_EDGES[:-1], CALIBRATION_GEN_PT_EDGES[1:])
        ):
            selected = (
                eta_selected
                & (pairs["reco_pt"] >= pt_min)
                & (pairs["reco_pt"] < pt_max)
                & valid
            )
            raw_pt = pairs["reco_pt"][selected]
            rows.append(
                {
                    "eta_bin": eta_bin.get("eta_bin", eta_index),
                    "eta_min": eta_bin["eta_min"],
                    "eta_max": eta_bin["eta_max"],
                    "pt_bin": pt_index,
                    "raw_pt_min": pt_min,
                    "raw_pt_max": pt_max,
                    "raw_pt_median": (
                        float(np.median(raw_pt)) if raw_pt.size else None
                    ),
                    "matched_jets": int(raw_pt.size),
                    "raw": calibration.response_metrics_with_ci(
                        raw_response[selected],
                        bootstrap_rng,
                        args.bootstrap_replicas,
                        True,
                    ),
                    "corrected": calibration.response_metrics_with_ci(
                        corrected_response[selected],
                        bootstrap_rng,
                        args.bootstrap_replicas,
                        True,
                    ),
                }
            )
    return rows


def jet_four_vector(jet):
    pt = jet["pt"]
    px = pt * math.cos(jet["phi"])
    py = pt * math.sin(jet["phi"])
    pz = pt * math.sinh(jet["eta"])
    energy = math.sqrt(px * px + py * py + pz * pz + jet["mass"] ** 2)
    return px, py, pz, energy


def add_four_vectors(*vectors):
    return tuple(sum(vector[index] for vector in vectors) for index in range(4))


def four_vector_kinematics(vector):
    px, py, pz, energy = vector
    pt = math.hypot(px, py)
    momentum_squared = px * px + py * py + pz * pz
    return {
        "pt": pt,
        "eta": math.asinh(pz / pt) if pt > 0.0 else math.nan,
        "phi": math.atan2(py, px) if pt > 0.0 else math.nan,
        "mass": math.sqrt(max(energy * energy - momentum_squared, 0.0)),
    }


def dijet_kinematics(first, second):
    return four_vector_kinematics(
        add_four_vectors(jet_four_vector(first), jet_four_vector(second))
    )


def scaled_jet(jet, factor):
    return {
        "pt": jet["pt"] * factor,
        "eta": jet["eta"],
        "phi": jet["phi"],
        "mass": jet["mass"] * factor,
    }


def corrected_event_jets(np, reco_jets, correction_map):
    if not reco_jets:
        return [], [], 0
    corrected = calibration.correct_jet_kinematics(
        np.asarray([jet["pt"] for jet in reco_jets]),
        np.asarray([jet["eta"] for jet in reco_jets]),
        np.asarray([jet["phi"] for jet in reco_jets]),
        np.asarray([jet["mass"] for jet in reco_jets]),
        correction_map,
    )
    corrected_jets = [
        {
            "pt": float(corrected["pt"][index]),
            "eta": reco_jets[index]["eta"],
            "phi": reco_jets[index]["phi"],
            "mass": float(corrected["mass"][index]),
        }
        for index in range(len(reco_jets))
        if corrected["valid"][index]
    ]
    supported_raw_jets = [
        jet for index, jet in enumerate(reco_jets) if corrected["valid"][index]
    ]
    return corrected_jets, supported_raw_jets, len(reco_jets) - len(corrected_jets)


def selected_pair(jets, eta_max, pt_min=RAW_PT_SUPPORT[0]):
    accepted = [
        jet for jet in jets if abs(jet["eta"]) < eta_max and jet["pt"] > pt_min
    ]
    accepted.sort(key=lambda jet: jet["pt"], reverse=True)
    return accepted[:2] if len(accepted) >= 2 else None


def append_pair_observables(target, pair):
    if pair is None:
        return
    dijet = dijet_kinematics(*pair)
    target["leading_pt"].append(pair[0]["pt"])
    target["subleading_pt"].append(pair[1]["pt"])
    target["dijet_mass"].append(dijet["mass"])


def empty_profile():
    return {"leading_pt": [], "subleading_pt": [], "dijet_mass": [], "trials": 0}


def empty_jet_activity_profiles():
    return {
        selection: {
            variant: {field: [] for field in JET_ACTIVITY_FIELDS}
            for variant in ("gen", "raw", "corrected", "smeared")
        }
        for selection in JET_ACTIVITY_SELECTIONS
    }


def selected_jets(jets, eta_max, pt_min):
    return sorted(
        (
            jet
            for jet in jets
            if abs(jet["eta"]) < eta_max and jet["pt"] > pt_min
        ),
        key=lambda jet: jet["pt"],
        reverse=True,
    )


def append_jet_activity_observables(profiles, variant, jets, truth_pair):
    accepted = selected_jets(jets, JET_ACTIVITY_ETA_MAX, JET_ACTIVITY_PT_MIN)
    ht = sum(jet["pt"] for jet in accepted)

    hardest = profiles["two_hardest"][variant]
    hardest["all_pt"].extend(jet["pt"] for jet in accepted)
    if len(accepted) >= 2:
        hardest["leading_pt"].append(accepted[0]["pt"])
        hardest["subleading_pt"].append(accepted[1]["pt"])
        hardest["dijet_ht_fraction"].append(
            (accepted[0]["pt"] + accepted[1]["pt"]) / ht
        )

    matched = profiles["truth_matched"][variant]
    accepted_truth = (
        selected_jets(truth_pair, JET_ACTIVITY_ETA_MAX, JET_ACTIVITY_PT_MIN)
        if truth_pair is not None
        else []
    )
    matched["all_pt"].extend(jet["pt"] for jet in accepted_truth)
    if len(accepted_truth) == 2:
        matched["leading_pt"].append(accepted_truth[0]["pt"])
        matched["subleading_pt"].append(accepted_truth[1]["pt"])
        matched["dijet_ht_fraction"].append(
            (accepted_truth[0]["pt"] + accepted_truth[1]["pt"]) / ht
        )


def event_profile_summary(np, profile):
    result = {
        "trials": profile["trials"],
        "selected_events": len(profile["leading_pt"]),
        "selection_efficiency": (
            len(profile["leading_pt"]) / profile["trials"]
            if profile["trials"]
            else None
        ),
    }
    for field in PROFILE_FIELDS:
        values = np.asarray(profile[field], dtype=np.float64)
        result[field] = {
            "entries": int(values.size),
            "q16": float(np.quantile(values, 0.16)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "q84": float(np.quantile(values, 0.84)) if values.size else None,
        }
    return result


def jet_activity_profiles(np, dataset, correction_map, resolution_config, args, rng):
    profiles = empty_jet_activity_profiles()
    for event in iter_events(dataset):
        truth_indices = event["hard_flavor_gen_indices"]
        if len(truth_indices) != 2:
            continue

        gen_truth_pair = [event["gen"][index] for index in truth_indices]
        append_jet_activity_observables(
            profiles, "gen", event["gen"], gen_truth_pair
        )

        by_gen = {
            gen_index: reco_index
            for gen_index, reco_index, _distance in event["hard_flavor_matches"]
        }
        reco_truth_indices = (
            [by_gen[index] for index in truth_indices]
            if all(index in by_gen for index in truth_indices)
            else None
        )
        raw_truth_pair = (
            [event["reco"][index] for index in reco_truth_indices]
            if reco_truth_indices is not None
            else None
        )
        append_jet_activity_observables(
            profiles, "raw", event["reco"], raw_truth_pair
        )

        corrected_by_index = {}
        if event["reco"]:
            factors, valid = calibration.correction_factors(
                np.asarray([jet["pt"] for jet in event["reco"]]),
                np.asarray([jet["eta"] for jet in event["reco"]]),
                correction_map,
            )
            corrected_by_index = {
                index: scaled_jet(event["reco"][index], factors[index])
                for index in range(len(event["reco"]))
                if valid[index]
            }
        corrected_truth_pair = (
            [corrected_by_index[index] for index in reco_truth_indices]
            if reco_truth_indices is not None
            and all(index in corrected_by_index for index in reco_truth_indices)
            else None
        )
        append_jet_activity_observables(
            profiles,
            "corrected",
            list(corrected_by_index.values()),
            corrected_truth_pair,
        )

        smeared = calibration.smear_gen_jets(
            np.asarray([jet["pt"] for jet in event["gen"]]),
            np.asarray([jet["mass"] for jet in event["gen"]]),
            np.asarray([jet["eta"] for jet in event["gen"]]),
            rng,
            resolution_config,
            args.smear_replicas,
        )
        for replica in range(args.smear_replicas):
            smeared_by_index = {
                index: {
                    "pt": float(smeared["pt"][replica, index]),
                    "eta": jet["eta"],
                    "phi": jet["phi"],
                    "mass": float(smeared["mass"][replica, index]),
                }
                for index, jet in enumerate(event["gen"])
                if smeared["valid"][replica, index]
            }
            smeared_truth_pair = (
                [smeared_by_index[index] for index in truth_indices]
                if all(index in smeared_by_index for index in truth_indices)
                else None
            )
            append_jet_activity_observables(
                profiles,
                "smeared",
                list(smeared_by_index.values()),
                smeared_truth_pair,
            )

    summary = {
        "jet_pt_min": JET_ACTIVITY_PT_MIN,
        "abs_eta_max": JET_ACTIVITY_ETA_MAX,
        "ht_definition": "sum of jet pt for jets passing the pt and eta selection",
        "selections": {
            selection: {
                variant: {
                    field: distribution_metrics(np, values)
                    for field, values in profile.items()
                }
                for variant, profile in variants.items()
            }
            for selection, variants in profiles.items()
        },
    }
    return summary, profiles


def pair_components(pair):
    vector = add_four_vectors(*(jet_four_vector(jet) for jet in pair))
    return vector, four_vector_kinematics(vector)


def append_dijet_residual(target, gen_pair, reco_pair):
    gen_vector, gen = pair_components(gen_pair)
    reco_vector, reco = pair_components(reco_pair)
    target["mass_ratio"].append(
        reco["mass"] / gen["mass"] if gen["mass"] > 0.0 else math.nan
    )
    target["delta_pt"].append(reco["pt"] - gen["pt"])
    target["delta_px"].append(reco_vector[0] - gen_vector[0])
    target["delta_py"].append(reco_vector[1] - gen_vector[1])


def empty_residuals():
    return {field: [] for field in RESIDUAL_FIELDS}


def finite_array(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def distribution_metrics(np, values):
    values = finite_array(np, values)
    if values.size == 0:
        return {
            "entries": 0,
            "mean": None,
            "rms": None,
            "q16": None,
            "median": None,
            "q84": None,
        }
    q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "entries": int(values.size),
        "mean": float(np.mean(values)),
        "rms": float(np.std(values)),
        "q16": float(q16),
        "median": float(median),
        "q84": float(q84),
    }


def event_closure(np, dataset, correction_map, resolution_config, args, rng):
    # One selection window, taken from the correction itself: the eta range it
    # covers and the raw-pT floor it is defined down to. Not an analysis cut.
    profiles = {
        variant: empty_profile()
        for variant in ("gen", "raw", "corrected", "smeared")
    }
    residuals = {
        variant: empty_residuals() for variant in ("raw", "corrected", "smeared")
    }
    eligible_hard_flavor_genjet_pair_events = 0
    closure_selected_pair_events = 0
    closure_uncovered_pair_events = 0
    complete_hard_flavor_puppi_pair_events = 0
    fully_correctable_pair_events = 0
    unsupported_hard_flavor_puppijets = 0
    for event in iter_events(dataset):
        if len(event["hard_flavor_gen_indices"]) != 2:
            continue
        eligible_hard_flavor_genjet_pair_events += 1
        gen_pair = [
            event["gen"][index] for index in event["hard_flavor_gen_indices"]
        ]
        closure_selected = min(jet["pt"] for jet in gen_pair) >= CLOSURE_GEN_PT_MIN
        if closure_selected:
            closure_selected_pair_events += 1
        by_gen = {
            gen_index: reco_index
            for gen_index, reco_index, _distance in event["hard_flavor_matches"]
        }
        raw_pair = None
        corrected_pair = None
        if all(index in by_gen for index in event["hard_flavor_gen_indices"]):
            complete_hard_flavor_puppi_pair_events += 1
            raw_pair = [
                event["reco"][by_gen[index]]
                for index in event["hard_flavor_gen_indices"]
            ]
            factors, factor_valid = calibration.correction_factors(
                np.asarray([jet["pt"] for jet in raw_pair]),
                np.asarray([jet["eta"] for jet in raw_pair]),
                correction_map,
            )
            unsupported_hard_flavor_puppijets += int(
                np.count_nonzero(~factor_valid)
            )
            if np.all(factor_valid):
                fully_correctable_pair_events += 1
                corrected_pair = [
                    scaled_jet(jet, factor)
                    for jet, factor in zip(raw_pair, factors)
                ]

        smeared = calibration.smear_gen_jets(
            np.asarray([jet["pt"] for jet in gen_pair]),
            np.asarray([jet["mass"] for jet in gen_pair]),
            np.asarray([jet["eta"] for jet in gen_pair]),
            rng,
            resolution_config,
            args.smear_replicas,
        )

        eta_max = ETA_EDGES[-1]
        for variant, jets in (
            ("gen", gen_pair),
            ("raw", raw_pair),
            ("corrected", corrected_pair),
        ):
            profiles[variant]["trials"] += 1
            if jets is not None:
                append_pair_observables(profiles[variant], selected_pair(jets, eta_max))
        for replica in range(args.smear_replicas):
            smeared_jets = [
                {
                    "pt": float(smeared["pt"][replica, index]),
                    "eta": jet["eta"],
                    "phi": jet["phi"],
                    "mass": float(smeared["mass"][replica, index]),
                }
                for index, jet in enumerate(gen_pair)
                if smeared["valid"][replica, index]
            ]
            profiles["smeared"]["trials"] += 1
            append_pair_observables(
                profiles["smeared"], selected_pair(smeared_jets, eta_max)
            )

        # Compare raw and corrected on the same events: the corrected pair only
        # exists where both jets are inside the raw-pT support, so filling raw
        # on the wider population would not be a like-for-like closure.
        if closure_selected and corrected_pair is not None:
            append_dijet_residual(residuals["raw"], gen_pair, raw_pair)
            append_dijet_residual(
                residuals["corrected"], gen_pair, corrected_pair
            )
        elif closure_selected:
            closure_uncovered_pair_events += 1
        if closure_selected:
            for replica in range(args.smear_replicas):
                smeared_pair = [
                    scaled_jet(gen_pair[index], smeared["factor"][replica, index])
                    for index in range(2)
                ]
                append_dijet_residual(residuals["smeared"], gen_pair, smeared_pair)

    summary = {
        "eligible_hard_flavor_genjet_pair_events": (
            eligible_hard_flavor_genjet_pair_events
        ),
        "closure_gen_pt_min": CLOSURE_GEN_PT_MIN,
        "closure_selected_pair_events": closure_selected_pair_events,
        "closure_uncovered_pair_events": closure_uncovered_pair_events,
        "complete_hard_flavor_puppi_pair_events": (
            complete_hard_flavor_puppi_pair_events
        ),
        "fully_correctable_pair_events": fully_correctable_pair_events,
        "unsupported_hard_flavor_puppijets": unsupported_hard_flavor_puppijets,
        "profiles": {
            variant: event_profile_summary(np, data)
            for variant, data in profiles.items()
        },
        "matched_dijet_residuals": {
            variant: {
                field: distribution_metrics(np, values)
                for field, values in fields.items()
            }
            for variant, fields in residuals.items()
        },
    }
    return summary, profiles, residuals


def plot_key(fsr_state, source_sample, target_sample, kind, *parts):
    return "__".join(
        (fsr_state, f"{source_sample}_map_on_{target_sample}", kind, *parts)
    )


def save_plot_arrays(
    np, target, fsr_state, source_sample, target_sample, profiles, residuals
):
    for variant, data in profiles.items():
        for field in PROFILE_FIELDS:
            key = plot_key(
                fsr_state, source_sample, target_sample, "profile", variant, field
            )
            target[key] = np.asarray(data[field], dtype=np.float64)
    for variant, data in residuals.items():
        for field in RESIDUAL_FIELDS:
            key = plot_key(
                fsr_state,
                source_sample,
                target_sample,
                "residual",
                variant,
                field,
            )
            target[key] = np.asarray(data[field], dtype=np.float64)


def save_jet_activity_arrays(
    np, target, fsr_state, source_sample, target_sample, profiles
):
    for selection, variants in profiles.items():
        for variant, data in variants.items():
            for field in JET_ACTIVITY_FIELDS:
                key = plot_key(
                    fsr_state,
                    source_sample,
                    target_sample,
                    "jet_activity",
                    selection,
                    variant,
                    field,
                )
                target[key] = np.asarray(data[field], dtype=np.float64)


def should_run_dijet_closure(source_config, target_dataset):
    return (
        source_config["closure_mode"] == "dijet"
        and target_dataset["closure_mode"] == "dijet"
    )


def run(args):
    validate_args(args)
    import numpy as np
    import yaml

    started = time.perf_counter()
    resolution_config = yaml.safe_load(args.resolution_yaml.read_text(encoding="utf-8"))
    dataset_configs = jet_cache.load_datasets(
        args.datasets_yaml, args.dataset, require_files=False
    )
    file_sets = selected_cache_sets(args, dataset_configs)
    configs_by_key = {
        jet_cache.dataset_key(dataset): dataset for dataset in dataset_configs
    }
    dataset_keys = list(configs_by_key)
    fsr_states = list(dict.fromkeys(state for state, _sample in dataset_keys))
    print(f"Using dataset manifest {args.datasets_yaml.resolve()}", flush=True)
    datasets = {phase: {} for phase in ("derivation", "validation")}
    all_records = []
    load_total = len(datasets) * len(dataset_keys)
    load_index = 0
    for phase in datasets:
        for key in dataset_keys:
            load_index += 1
            fsr_state, sample = key
            print(
                f"Loading {phase} dataset {fsr_state}/{sample} "
                f"[{load_index}/{load_total}; elapsed={time.perf_counter() - started:.1f}s]",
                flush=True,
            )
            dataset = load_dataset(
                np,
                configs_by_key[key],
                file_sets[key][phase],
                args.max_events,
            )
            datasets[phase][key] = dataset
            all_records.extend(record for record, _count in dataset["sources"])
            print(
                f"Loaded {phase} {fsr_state}/{sample}: "
                f"{dataset['events_read']} events, "
                f"{dataset['pairs']['gen_pt'].size} matched jets",
                flush=True,
            )
    settings = cache_settings(all_records)

    maps = {fsr_state: {} for fsr_state in fsr_states}
    for map_index, (fsr_state, sample) in enumerate(dataset_keys):
        print(
            f"Deriving map {fsr_state}/{sample} "
            f"[{map_index + 1}/{len(dataset_keys)}; "
            f"elapsed={time.perf_counter() - started:.1f}s]",
            flush=True,
        )
        maps[fsr_state][sample] = derive_correction_map(
            np,
            datasets["derivation"][(fsr_state, sample)],
            args,
            np.random.default_rng(args.seed + map_index),
        )
        usable_nodes = sum(
            node["usable"]
            for eta_bin in maps[fsr_state][sample]["eta_bins"]
            for node in eta_bin["nodes"]
        )
        unsupported_trigger_eta = [
            eta_bin
            for eta_bin in maps[fsr_state][sample]["eta_bins"]
            if eta_bin["eta_min"] < 2.4 and not eta_bin["supported"]
        ]
        if unsupported_trigger_eta:
            ranges = ", ".join(
                f"[{eta_bin['eta_min']}, {eta_bin['eta_max']})"
                for eta_bin in unsupported_trigger_eta
            )
            raise RuntimeError(
                f"{fsr_state}/{sample} has fewer than two usable correction "
                f"nodes in trigger eta bin(s): {ranges}"
            )
        print(
            f"Derived {fsr_state}/{sample} map: {usable_nodes} usable nodes",
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    correction_document = {
        "schema_version": calibration.SCHEMA_VERSION,
        "configuration": {
            "collection": "JetPUPPI",
            "method": "monotonic linear target pT versus raw JetPUPPI pT",
            "calibration_gen_pt_edges": list(CALIBRATION_GEN_PT_EDGES),
            "raw_pt_support": list(RAW_PT_SUPPORT),
            "validation_gen_pt_edges": list(PT_EDGES),
            "abs_eta_edges": list(ETA_EDGES),
            "validation_abs_eta_edges": list(VALIDATION_ETA_EDGES),
            "closure_gen_pt_min": CLOSURE_GEN_PT_MIN,
            "match_dr_max": settings["match_dr_max"],
            "truth_genjet_dr_max": settings["truth_genjet_dr_max"],
            "sample_pair_selection": {
                f"{state}/{sample}": datasets["derivation"][(state, sample)][
                    "pair_selection"
                ]
                for state, sample in dataset_keys
            },
            "min_bin_entries": args.min_bin_entries,
            "bootstrap_replicas": args.bootstrap_replicas,
            "seed": args.seed,
            "endpoint_behavior": "constant endpoint factor",
            "outside_support": "reject raw pT or eta outside configured support",
            "dijet_closure_rule": (
                "run only when source and target datasets use dijet closure"
            ),
            "cache_schema_version": jet_cache.CACHE_SCHEMA_VERSION,
            "dataset_manifest": str(args.datasets_yaml.resolve()),
        },
        "maps": maps,
    }
    corrections_path = output_dir / "corrections.yaml"
    corrections_path.write_text(
        yaml.safe_dump(correction_document, sort_keys=False), encoding="utf-8"
    )
    print(f"Saved corrections: {corrections_path}", flush=True)

    validation_summary = {fsr_state: {} for fsr_state in fsr_states}
    plot_data = {}
    validation_total = sum(
        sum(source_state == target_state for source_state, _source in dataset_keys)
        for target_state, _target in dataset_keys
    )
    validation_index = 0
    for target_index, (fsr_state, target_sample) in enumerate(dataset_keys):
        dataset = datasets["validation"][(fsr_state, target_sample)]
        validation_summary[fsr_state][target_sample] = {
            "files": dataset["files"],
            "cache_files": dataset["cache_files"],
            "events_read": dataset["events_read"],
            "matched_jets": int(dataset["pairs"]["gen_pt"].size),
            "pair_selection": dataset["pair_selection"],
            "match_selection": dataset["match_selection"],
            "truth_pid_abs": dataset["truth_pid_abs"],
            "closure_mode": dataset["closure_mode"],
            "selection_pair_events": dataset["selection_pair_events"],
            "selected_genjets": dataset["selected_genjets"],
            "matched_puppijets": dataset["matched_puppijets"],
            "complete_genjet_pair_events": dataset["complete_genjet_pair_events"],
            "complete_puppi_pair_events": dataset[
                "complete_puppi_pair_events"
            ],
            "source_maps": {},
        }
        source_keys = [key for key in dataset_keys if key[0] == fsr_state]
        for source_index, (_state, source_sample) in enumerate(source_keys):
            validation_index += 1
            print(
                f"Validating {fsr_state}: {source_sample} map on {target_sample} "
                f"[{validation_index}/{validation_total}; "
                f"elapsed={time.perf_counter() - started:.1f}s]",
                flush=True,
            )
            map_data = maps[fsr_state][source_sample]
            base_seed = args.seed + 1000 + 10 * target_index + source_index
            rows, _factors, valid = evaluate_response_bins(
                np,
                dataset["pairs"],
                map_data,
                resolution_config,
                args,
                np.random.default_rng(base_seed),
                np.random.default_rng(args.seed + 5000 + 10 * target_index),
            )
            raw_pt_rows = evaluate_raw_pt_response_bins(
                np,
                dataset["pairs"],
                map_data,
                args,
                np.random.default_rng(base_seed + 20000),
            )
            source_config = configs_by_key[(fsr_state, source_sample)]
            run_dijet_closure = should_run_dijet_closure(source_config, dataset)
            event_summary = None
            jet_activity_summary = None
            if run_dijet_closure:
                event_summary, profiles, residuals = event_closure(
                    np,
                    dataset,
                    map_data,
                    resolution_config,
                    args,
                    np.random.default_rng(args.seed + 10000 + 10 * target_index),
                )
                save_plot_arrays(
                    np,
                    plot_data,
                    fsr_state,
                    source_sample,
                    target_sample,
                    profiles,
                    residuals,
                )
                jet_activity_summary, activity_profiles = jet_activity_profiles(
                    np,
                    dataset,
                    map_data,
                    resolution_config,
                    args,
                    np.random.default_rng(args.seed + 15000 + 10 * target_index),
                )
                save_jet_activity_arrays(
                    np,
                    plot_data,
                    fsr_state,
                    source_sample,
                    target_sample,
                    activity_profiles,
                )
            validation_summary[fsr_state][target_sample]["source_maps"][
                source_sample
            ] = {
                "valid_corrected_jets": int(np.count_nonzero(valid)),
                "correction_coverage": (
                    float(np.count_nonzero(valid) / valid.size)
                    if valid.size
                    else None
                ),
                "response_bins": rows,
                "raw_pt_response_bins": raw_pt_rows,
                "event_closure": event_summary,
                "jet_activity": jet_activity_summary,
            }
            coverage = float(np.mean(valid)) if valid.size else math.nan
            print(
                f"Validated {fsr_state}: {source_sample} map on {target_sample}; "
                f"coverage={coverage:.1%}",
                flush=True,
            )

    plot_data_path = output_dir / "plot_data.npz"
    with plot_data_path.open("wb") as output:
        np.savez(output, **plot_data)
    print(f"Saved plot data: {plot_data_path}", flush=True)
    summary = {
        "configuration": {
            **correction_document["configuration"],
            "cache_dir": str(args.cache_dir.resolve()),
            "dataset_file_counts": {
                f"{state}/{sample}": {
                    "derivation": configs_by_key[(state, sample)][
                        "derivation_files"
                    ],
                    "validation": configs_by_key[(state, sample)][
                        "validation_files"
                    ],
                }
                for state, sample in dataset_keys
            },
            "max_events_per_sample_phase": args.max_events,
            "smear_model": "unit-mean log-normal",
            "l1_relative_sigma_formula": "a + b / pt",
            "smear_replicas": args.smear_replicas,
            "resolution_yaml": str(args.resolution_yaml.resolve()),
            "resolution_primary": "(q84-q16)/(2*median)",
            "closure_pair_window": {
                "eta_max": ETA_EDGES[-1],
                "pt_min": RAW_PT_SUPPORT[0],
            },
            "plot_data": str(plot_data_path),
        },
        "validation": validation_summary,
    }
    summary_path = output_dir / "summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}", flush=True)
    print(
        f"Correction study complete in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
