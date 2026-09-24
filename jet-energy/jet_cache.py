#!/usr/bin/env python3
"""Cache the small Delphes jet record needed by the calibration study."""

import argparse
import glob
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATASETS = HERE / "datasets.yaml"
DEFAULT_CACHE = HERE / "cache"
CACHE_SCHEMA_VERSION = 3
JET_FIELDS = ("pt", "eta", "phi", "mass")
TRACK_SUM_DR = 0.4
MATCH_SELECTIONS = ("inclusive", "hard_flavor")
CLOSURE_MODES = ("single_jet", "dijet")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-yaml", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="SAMPLE:FSR_STATE",
        help="Only cache the named dataset; repeat to select several",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional event cap in each cached ROOT file",
    )
    parser.add_argument("--eta-max", type=float, default=3.0)
    parser.add_argument("--match-dr-max", type=float, default=0.2)
    parser.add_argument("--truth-genjet-dr-max", type=float, default=0.4)
    parser.add_argument(
        "--progress-events",
        type=int,
        default=2000,
        help="Report progress this often while processing each ROOT file",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_JET_CACHE_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_JET_CACHE_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def validate_args(args):
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be positive")
    if min(
        args.eta_max,
        args.match_dr_max,
        args.truth_genjet_dr_max,
        args.progress_events,
    ) <= 0.0:
        raise RuntimeError("eta, matching limits, and progress interval must be positive")


def natural_key(path):
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", str(path))
    ]


def dataset_key(dataset):
    return dataset["fsr_state"], dataset["sample"]


def parse_dataset_filter(value):
    pieces = value.split(":", 1)
    if len(pieces) != 2 or not all(pieces):
        raise RuntimeError(f"Invalid dataset selector {value!r}; use SAMPLE:FSR_STATE")
    return pieces[1], pieces[0]


def round_robin_paths(groups):
    """Interleave naturally sorted file groups without duplicating a source."""
    selected = []
    seen = set()
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index >= len(group):
                continue
            path = group[index]
            if path in seen:
                raise RuntimeError(f"Input globs select the same file more than once: {path}")
            seen.add(path)
            selected.append(path)
    return selected


def load_datasets(path=DEFAULT_DATASETS, selected=(), require_files=True):
    import yaml

    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = document.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"Dataset manifest has no datasets: {path}")
    requested = {parse_dataset_filter(value) for value in selected}
    datasets = []
    seen = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid dataset entry in {path}")
        try:
            has_single_glob = "input_glob" in raw
            has_multiple_globs = "input_globs" in raw
            if has_single_glob == has_multiple_globs:
                raise ValueError("define exactly one of input_glob or input_globs")
            input_globs = (
                [str(raw["input_glob"])]
                if has_single_glob
                else [str(value) for value in raw["input_globs"]]
            )
            if not input_globs:
                raise ValueError("input_globs must not be empty")
            dataset = {
                "sample": str(raw["sample"]),
                "fsr_state": str(raw["fsr_state"]),
                "input_globs": input_globs,
                "match_selection": str(raw["match_selection"]),
                "derivation_files": int(raw["derivation_files"]),
                "validation_files": int(raw["validation_files"]),
                "closure_mode": str(raw["closure_mode"]),
                "truth_pid_abs": (
                    int(raw["truth_pid_abs"])
                    if raw.get("truth_pid_abs") is not None
                    else None
                ),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid dataset entry in {path}: {error}") from error
        key = dataset_key(dataset)
        if key in seen:
            raise RuntimeError(
                f"Duplicate dataset {dataset['sample']}/{dataset['fsr_state']} in {path}"
            )
        seen.add(key)
        if requested and key not in requested:
            continue
        if dataset["match_selection"] not in MATCH_SELECTIONS:
            raise RuntimeError(
                f"Invalid match_selection for {dataset['sample']}/{dataset['fsr_state']}"
            )
        if dataset["closure_mode"] not in CLOSURE_MODES:
            raise RuntimeError(
                f"Invalid closure_mode for {dataset['sample']}/{dataset['fsr_state']}"
            )
        if min(dataset["derivation_files"], dataset["validation_files"]) <= 0:
            raise RuntimeError("derivation_files and validation_files must be positive")
        if dataset["match_selection"] == "hard_flavor":
            if dataset["truth_pid_abs"] not in (4, 5):
                raise RuntimeError("hard_flavor datasets require truth_pid_abs 4 or 5")
        elif dataset["truth_pid_abs"] is not None:
            raise RuntimeError("inclusive datasets must not define truth_pid_abs")
        patterns = []
        file_groups = []
        for value in dataset["input_globs"]:
            pattern = Path(value)
            if not pattern.is_absolute():
                pattern = REPO / pattern
            patterns.append(pattern)
            file_groups.append(
                sorted(
                    (Path(match).resolve() for match in glob.glob(str(pattern))),
                    key=natural_key,
                )
            )
        if require_files:
            empty = [str(pattern) for pattern, files in zip(patterns, file_groups) if not files]
            if empty:
                raise RuntimeError(
                    f"No ROOT files for {dataset['sample']}/{dataset['fsr_state']} "
                    f"matching {', '.join(empty)}"
                )
        files = round_robin_paths(file_groups)
        required = dataset["derivation_files"] + dataset["validation_files"]
        if require_files and len(files) < required:
            raise RuntimeError(
                f"Need {required} ROOT files for {dataset['sample']}/"
                f"{dataset['fsr_state']}; found {len(files)} matching "
                f"{', '.join(str(pattern) for pattern in patterns)}"
            )
        dataset["files"] = files[:required]
        dataset["round_robin_sources"] = len(patterns) > 1
        dataset["manifest_path"] = str(path)
        datasets.append(dataset)
    missing = requested - {dataset_key(dataset) for dataset in datasets}
    if missing:
        names = ", ".join(f"{sample}:{state}" for state, sample in sorted(missing))
        raise RuntimeError(f"Requested dataset(s) not found in {path}: {names}")
    return datasets


def cache_files(cache_dir, fsr_state, sample, start, count):
    directory = Path(cache_dir).resolve() / fsr_state / sample
    paths = sorted(directory.glob("*.npz"), key=natural_key)
    selected = paths[start : start + count]
    if len(selected) != count:
        required = start + count
        raise RuntimeError(
            f"Need {required} cached files for {fsr_state}/{sample}; found "
            f"{len(paths)} in {directory}. Run jet_cache.py --max-files {required}."
        )
    return selected


def delta_phi(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


def delta_r(first, second):
    return math.hypot(
        first["eta"] - second["eta"],
        delta_phi(first["phi"], second["phi"]),
    )


def match_jets(first_jets, second_jets, max_delta_r):
    """Greedily make unique matches, starting from the smallest delta-R."""
    candidates = []
    for first_index, first in enumerate(first_jets):
        for second_index, second in enumerate(second_jets):
            distance = delta_r(first, second)
            if distance < max_delta_r:
                candidates.append((distance, first_index, second_index))

    matches = []
    used_first = set()
    used_second = set()
    for distance, first_index, second_index in sorted(candidates):
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append((first_index, second_index, distance))
    return sorted(matches, key=lambda match: match[0])


def read_jets(collection, eta_max):
    jets = []
    for index in range(collection.GetEntriesFast()):
        jet = collection.At(index)
        eta = float(jet.Eta)
        if abs(eta) >= eta_max:
            continue
        jets.append(
            {
                "pt": float(jet.PT),
                "eta": eta,
                "phi": float(jet.Phi),
                "mass": float(jet.Mass),
            }
        )
    return jets


def read_tracks(collection):
    tracks = []
    for index in range(collection.GetEntriesFast()):
        track = collection.At(index)
        tracks.append(
            {
                "pt": float(track.PT),
                "eta": float(track.Eta),
                "phi": float(track.Phi),
            }
        )
    return tracks


def track_sums(jets, tracks, max_delta_r):
    """Sum primary-vertex track pT inside max_delta_r of each jet axis."""
    sums = []
    counts = []
    for jet in jets:
        total = 0.0
        count = 0
        for track in tracks:
            if delta_r(jet, track) < max_delta_r:
                total += track["pt"]
                count += 1
        sums.append(total)
        counts.append(count)
    return sums, counts


def hard_process_partons(particles, pid_abs):
    selected = [
        {
            "pt": float(particle.PT),
            "eta": float(particle.Eta),
            "phi": float(particle.Phi),
            "pid": int(particle.PID),
        }
        for particle in particles
        if abs(int(particle.PID)) == pid_abs
        and int(particle.Status) == 23
        and not int(particle.IsPU)
    ]
    quarks = [particle for particle in selected if particle["pid"] == pid_abs]
    antiquarks = [particle for particle in selected if particle["pid"] == -pid_abs]
    if len(quarks) != 1 or len(antiquarks) != 1:
        return []
    return [quarks[0], antiquarks[0]]


def hard_process_bottoms(particles):
    """Compatibility wrapper for the former b-only selector."""
    return hard_process_partons(particles, 5)


def source_metadata(path, dataset, args, events_read=None):
    stat = path.stat()
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_path": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "fsr_state": dataset["fsr_state"],
        "sample": dataset["sample"],
        "match_selection": dataset["match_selection"],
        "truth_pid_abs": dataset["truth_pid_abs"],
        "closure_mode": dataset["closure_mode"],
        "dataset_manifest": dataset["manifest_path"],
        "eta_max": args.eta_max,
        "match_dr_max": args.match_dr_max,
        "truth_genjet_dr_max": args.truth_genjet_dr_max,
        "track_sum_dr": TRACK_SUM_DR,
        "max_events": args.max_events,
        "events_read": events_read,
    }


def metadata_matches(actual, expected):
    keys = (
        "schema_version",
        "source_path",
        "source_size",
        "source_mtime_ns",
        "fsr_state",
        "sample",
        "match_selection",
        "truth_pid_abs",
        "closure_mode",
        "dataset_manifest",
        "eta_max",
        "match_dr_max",
        "truth_genjet_dr_max",
        "track_sum_dr",
        "max_events",
    )
    return all(actual.get(key) == expected.get(key) for key in keys)


def write_cache(path, metadata, arrays):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(values) for name, values in arrays.items()}
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as output:
        np.savez(output, **payload)


def read_cache_metadata(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise RuntimeError(f"Invalid jet cache {path}: missing metadata")
        metadata = json.loads(str(data["metadata"].item()))
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported cache schema in {path}: {metadata.get('schema_version')}"
        )
    return metadata


TRACK_ARRAYS = (
    "gen_track_pt",
    "gen_track_n",
    "reco_track_pt",
    "reco_track_n",
)


def load_cache(path):
    import numpy as np

    required = {
        *(f"gen_{field}" for field in JET_FIELDS),
        *(f"reco_{field}" for field in JET_FIELDS),
        "gen_offsets",
        "reco_offsets",
        "match_offsets",
        "match_gen_index",
        "match_reco_index",
        "match_dr",
        "hard_flavor_gen_offsets",
        "hard_flavor_gen_index",
        "hard_flavor_gen_dr",
        "hard_flavor_match_offsets",
        "hard_flavor_match_gen_index",
        "hard_flavor_match_reco_index",
        "hard_flavor_match_dr",
        "hard_flavor_pair_event",
    }
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise RuntimeError(f"Invalid jet cache {path}: missing metadata")
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported cache schema in {path}: "
                f"{metadata.get('schema_version')}"
            )
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"Invalid jet cache {path}: missing {', '.join(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required}
        for name in TRACK_ARRAYS:
            if name in data.files:
                arrays[name] = np.asarray(data[name])
    arrays["metadata"] = metadata
    arrays["path"] = str(Path(path).resolve())
    return arrays


def append_jets(target, jets):
    for field in JET_FIELDS:
        target[field].extend(jet[field] for jet in jets)


def append_matches(target, matches):
    target["gen_index"].extend(match[0] for match in matches)
    target["reco_index"].extend(match[1] for match in matches)
    target["dr"].extend(match[2] for match in matches)


def matching_counters(pair_events, gen_offsets, match_offsets):
    gen_counts = [stop - start for start, stop in zip(gen_offsets[:-1], gen_offsets[1:])]
    match_counts = [
        stop - start for start, stop in zip(match_offsets[:-1], match_offsets[1:])
    ]
    return {
        "hard_flavor_pair_events": int(sum(pair_events)),
        "hard_flavor_matched_genjets": int(sum(gen_counts)),
        "hard_flavor_matched_puppijets": int(sum(match_counts)),
        "hard_flavor_complete_genjet_pair_events": sum(
            count == 2 for count in gen_counts
        ),
        "hard_flavor_complete_puppi_pair_events": sum(
            count == 2 for count in match_counts
        ),
    }


def build_cache_file(ROOT, np, root_path, output_path, dataset, args):
    root_file = ROOT.TFile.Open(str(root_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
    tree = root_file.Get("Delphes")
    required = ["GenJet", "JetPUPPI", "EFlowTrack"]
    if dataset["match_selection"] == "hard_flavor":
        required.append("Particle")
    missing = [name for name in required if not tree or not tree.GetBranch(name)]
    if missing:
        root_file.Close()
        raise RuntimeError(f"Missing {', '.join(missing)} branch(es) in {root_path}")

    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("GenJet*", 1)
    tree.SetBranchStatus("JetPUPPI*", 1)
    if dataset["match_selection"] == "hard_flavor":
        tree.SetBranchStatus("Particle*", 1)
    tree.SetBranchStatus("EFlowTrack*", 1)

    entries = int(tree.GetEntries())
    if args.max_events is not None:
        entries = min(entries, args.max_events)
    jets = {
        "gen": {field: [] for field in JET_FIELDS},
        "reco": {field: [] for field in JET_FIELDS},
    }
    offsets = {"gen": [0], "reco": [0]}
    track_pt = {"gen": [], "reco": []}
    track_n = {"gen": [], "reco": []}
    matches = {field: [] for field in ("gen_index", "reco_index", "dr")}
    hard_flavor_gen = {field: [] for field in ("gen_index", "dr")}
    hard_flavor_matches = {
        field: [] for field in ("gen_index", "reco_index", "dr")
    }
    match_offsets = [0]
    hard_flavor_gen_offsets = [0]
    hard_flavor_match_offsets = [0]
    hard_flavor_pair_event = []

    started = time.perf_counter()
    for entry in range(entries):
        tree.GetEntry(entry)
        gen_jets = read_jets(tree.GenJet, args.eta_max)
        reco_jets = read_jets(tree.JetPUPPI, args.eta_max)
        tracks = read_tracks(tree.EFlowTrack)
        inclusive = match_jets(gen_jets, reco_jets, args.match_dr_max)
        partons = []
        if dataset["match_selection"] == "hard_flavor":
            particles = [
                tree.Particle.At(index)
                for index in range(tree.Particle.GetEntriesFast())
            ]
            partons = hard_process_partons(particles, dataset["truth_pid_abs"])
        parton_gen_matches = match_jets(
            partons, gen_jets, args.truth_genjet_dr_max
        )
        selected_indices = [match[1] for match in parton_gen_matches]
        selected_genjets = [gen_jets[index] for index in selected_indices]
        hard_flavor_matches_in_event = [
            (selected_indices[gen_index], reco_index, distance)
            for gen_index, reco_index, distance in match_jets(
                selected_genjets, reco_jets, args.match_dr_max
            )
        ]

        append_jets(jets["gen"], gen_jets)
        append_jets(jets["reco"], reco_jets)
        for prefix, collection in (("gen", gen_jets), ("reco", reco_jets)):
            sums, counts = track_sums(collection, tracks, TRACK_SUM_DR)
            track_pt[prefix].extend(sums)
            track_n[prefix].extend(counts)
        offsets["gen"].append(len(jets["gen"]["pt"]))
        offsets["reco"].append(len(jets["reco"]["pt"]))
        append_matches(matches, inclusive)
        hard_flavor_gen["gen_index"].extend(
            match[1] for match in parton_gen_matches
        )
        hard_flavor_gen["dr"].extend(match[2] for match in parton_gen_matches)
        append_matches(hard_flavor_matches, hard_flavor_matches_in_event)
        match_offsets.append(len(matches["dr"]))
        hard_flavor_gen_offsets.append(len(hard_flavor_gen["dr"]))
        hard_flavor_match_offsets.append(len(hard_flavor_matches["dr"]))
        hard_flavor_pair_event.append(bool(partons))
        completed = entry + 1
        if completed % args.progress_events == 0 or completed == entries:
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = (entries - completed) / rate if rate > 0.0 else math.inf
            print(
                f"  events {completed:,}/{entries:,} "
                f"({completed / entries:.1%}) elapsed={elapsed:.1f}s "
                f"rate={rate:.1f}/s eta={remaining:.1f}s",
                flush=True,
            )

    root_file.Close()
    metadata = source_metadata(root_path, dataset, args, entries)
    metadata.update(
        {
            "inclusive_matched_jets": len(matches["dr"]),
            **matching_counters(
                hard_flavor_pair_event,
                hard_flavor_gen_offsets,
                hard_flavor_match_offsets,
            ),
        }
    )
    arrays = {
        **{
            f"{prefix}_{field}": np.asarray(values, dtype=np.float64)
            for prefix in ("gen", "reco")
            for field, values in jets[prefix].items()
        },
        **{
            f"{prefix}_track_pt": np.asarray(track_pt[prefix], dtype=np.float64)
            for prefix in ("gen", "reco")
        },
        **{
            f"{prefix}_track_n": np.asarray(track_n[prefix], dtype=np.int64)
            for prefix in ("gen", "reco")
        },
        "gen_offsets": np.asarray(offsets["gen"], dtype=np.int64),
        "reco_offsets": np.asarray(offsets["reco"], dtype=np.int64),
        "match_offsets": np.asarray(match_offsets, dtype=np.int64),
        "match_gen_index": np.asarray(matches["gen_index"], dtype=np.int64),
        "match_reco_index": np.asarray(matches["reco_index"], dtype=np.int64),
        "match_dr": np.asarray(matches["dr"], dtype=np.float64),
        "hard_flavor_gen_offsets": np.asarray(
            hard_flavor_gen_offsets, dtype=np.int64
        ),
        "hard_flavor_gen_index": np.asarray(
            hard_flavor_gen["gen_index"], dtype=np.int64
        ),
        "hard_flavor_gen_dr": np.asarray(hard_flavor_gen["dr"], dtype=np.float64),
        "hard_flavor_match_offsets": np.asarray(
            hard_flavor_match_offsets, dtype=np.int64
        ),
        "hard_flavor_match_gen_index": np.asarray(
            hard_flavor_matches["gen_index"], dtype=np.int64
        ),
        "hard_flavor_match_reco_index": np.asarray(
            hard_flavor_matches["reco_index"], dtype=np.int64
        ),
        "hard_flavor_match_dr": np.asarray(
            hard_flavor_matches["dr"], dtype=np.float64
        ),
        "hard_flavor_pair_event": np.asarray(hard_flavor_pair_event, dtype=bool),
    }
    write_cache(output_path, metadata, arrays)
    return metadata


def run(args):
    validate_args(args)
    import ROOT
    import numpy as np

    if ROOT.gSystem.Load("libDelphes") < 0:
        raise RuntimeError("Could not load libDelphes")
    datasets = load_datasets(args.datasets_yaml, args.dataset)
    built = 0
    skipped = 0
    total_files = sum(len(dataset["files"]) for dataset in datasets)
    completed_files = 0
    started = time.perf_counter()
    for dataset in datasets:
        fsr_state, sample = dataset_key(dataset)
        stem_counts = {
            stem: sum(path.stem == stem for path in dataset["files"])
            for stem in {path.stem for path in dataset["files"]}
        }
        for file_index, root_path in enumerate(dataset["files"]):
            output_stem = root_path.stem
            if dataset["round_robin_sources"]:
                output_stem = f"{file_index:04d}_{root_path.stem}"
            elif stem_counts[root_path.stem] > 1:
                output_stem = f"{root_path.parent.name}_{root_path.stem}"
            output_path = (
                args.cache_dir.resolve()
                / fsr_state
                / sample
                / f"{output_stem}.npz"
            )
            expected = source_metadata(root_path, dataset, args)
            if output_path.exists() and not args.force:
                try:
                    cached_metadata = read_cache_metadata(output_path)
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                    cached_metadata = None
                if cached_metadata is not None and metadata_matches(
                    cached_metadata, expected
                ):
                    completed_files += 1
                    elapsed = time.perf_counter() - started
                    rate = completed_files / elapsed if elapsed > 0.0 else 0.0
                    remaining = (
                        (total_files - completed_files) / rate if rate > 0.0 else math.inf
                    )
                    print(
                        f"Using {output_path} [files {completed_files}/{total_files}; "
                        f"elapsed={elapsed:.1f}s eta={remaining:.1f}s]",
                        flush=True,
                    )
                    skipped += 1
                    continue
            print(
                f"Caching {fsr_state}/{sample}: {root_path.name} "
                f"[file {completed_files + 1}/{total_files}]",
                flush=True,
            )
            metadata = build_cache_file(
                ROOT, np, root_path, output_path, dataset, args
            )
            completed_files += 1
            elapsed = time.perf_counter() - started
            rate = completed_files / elapsed if elapsed > 0.0 else 0.0
            remaining = (
                (total_files - completed_files) / rate if rate > 0.0 else math.inf
            )
            print(
                f"Wrote {output_path} ({metadata['events_read']} events) "
                f"[files {completed_files}/{total_files}; elapsed={elapsed:.1f}s "
                f"eta={remaining:.1f}s]",
                flush=True,
            )
            built += 1
    print(
        f"Cache complete: built {built}, reused {skipped}, "
        f"elapsed={time.perf_counter() - started:.1f}s",
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
