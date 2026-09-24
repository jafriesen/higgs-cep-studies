#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import natural_key  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_campaign_root,
    generation_config,
    generation_processes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Pythia on MadGraph campaigns listed in processes-madgraph.yaml."
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name to run. May be repeated. Defaults to all MadGraph processes.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Main campaign name to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Pythia campaign name within the hadr-Pythia stage. Defaults to the campaign name.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap per output file")
    parser.add_argument("--max-files", type=int, default=None, help="Optional MadGraph LHE input file cap per process")
    parser.add_argument(
        "--file-index",
        type=int,
        default=None,
        help="Process only this 1-based index in the sorted LHE input file list",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional Pythia seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing HepMC outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Pythia")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose mode to the Pythia bridge")
    return parser.parse_args()


def validate_args(args):
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.file_index is not None and args.file_index <= 0:
        raise RuntimeError("--file-index must be > 0")
    if args.seed is not None and args.seed < 0:
        raise RuntimeError("--seed must be non-negative")


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def binary_path(root):
    user = os.environ.get("USER", "user")
    build_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"higgs_cep_pythia_{user}"
    return build_dir / "process_madgraph"


def build_binary(root):
    """Compile process_madgraph.cc against the LCG view's Pythia8/HepMC3, caching the binary."""
    binary = binary_path(root)
    src = root / "generation-pythia" / "scripts" / "process_madgraph.cc"
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary

    pythia8 = os.environ.get("PYTHIA8")
    lcg_view = os.environ.get("LCG_VIEW")
    if not pythia8 or not lcg_view:
        raise RuntimeError(
            "$PYTHIA8/$LCG_VIEW not set; run `source env/setup_pythia.sh` first"
        )
    view_root = lcg_view.rsplit("/setup.sh", 1)[0]

    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(src),
            f"-I{pythia8}/include", f"-I{view_root}/include",
            f"-L{pythia8}/lib", f"-L{view_root}/lib64",
            f"-Wl,-rpath,{pythia8}/lib", f"-Wl,-rpath,{view_root}/lib64",
            "-lpythia8", "-lHepMC3", "-lHepMC3search",
            "-o", str(binary),
        ],
        check=True,
    )
    return binary


def seed_for_file(base_seed, file_index):
    """Distinct, reproducible seed per input file, offset from a shared base seed."""
    if base_seed is None:
        return None
    max_seed = 900000000
    seed = base_seed + file_index
    if seed > max_seed:
        seed = ((seed - 1) % max_seed) + 1
    return seed


def build_command(binary, manifest_file, args):
    command = [str(binary), "--manifest", str(manifest_file)]
    if args.max_events is not None:
        command.extend(["--max-events", str(args.max_events)])
    if args.verbose:
        command.append("--verbose")
    return command


def manifest_row(input_file, output_file, seed):
    values = (input_file, output_file)
    if any("\t" in str(value) or "\n" in str(value) for value in values):
        raise RuntimeError("Input/output paths cannot contain tabs or newlines")
    return f"{input_file}\t{output_file}\t{seed if seed is not None else -1}\n"


def discover_lhe_files(path, max_files):
    files = sorted(path.glob("*.lhe"), key=natural_key)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No MadGraph LHE files found in {path}")
    return files


def event_records_dir(process_name, campaign_name):
    cfg = generation_config("madgraph")
    return (
        generation_campaign_root("madgraph", process_name, campaign_name)
        / cfg["generation_dir"]
        / "evrecs"
    )


def stage_output_dir(process_name, campaign_name, tag):
    cfg = generation_config("madgraph")
    stage_dir = (cfg.get("stage_dirs") or {}).get("hadr-pythia")
    if not stage_dir:
        raise RuntimeError("config.yaml must define generation.madgraph.stage_dirs.hadr-pythia")
    return generation_campaign_root("madgraph", process_name, campaign_name) / stage_dir / tag


def configured_max_files(process_config):
    max_files = process_config.get("max_files")
    if max_files is None:
        return None
    max_files = int(max_files)
    if max_files <= 0:
        raise RuntimeError("MadGraph process max_files must be > 0")
    return max_files


def main():
    args = parse_args()
    validate_args(args)
    root = ROOT
    processes = generation_processes("madgraph")
    binary = build_binary(root) if not args.dry_run else binary_path(root)

    for process_name in selected_processes(processes, args.processes):
        process_config = processes[process_name] or {}
        campaign_name, _ = generation_campaign_config("madgraph", process_name, args.campaign)
        tag = args.tag or campaign_name
        max_files = args.max_files
        if max_files is None:
            max_files = configured_max_files(process_config)

        input_dir = event_records_dir(process_name, campaign_name)
        input_files = discover_lhe_files(input_dir, max_files=max_files)
        indexed_input_files = list(enumerate(input_files, start=1))
        if args.file_index is not None:
            if args.file_index > len(indexed_input_files):
                raise RuntimeError(
                    f"--file-index {args.file_index} exceeds the {len(indexed_input_files)} "
                    f"MadGraph LHE files found for {process_name}"
                )
            indexed_input_files = [indexed_input_files[args.file_index - 1]]

        output_dir = stage_output_dir(process_name, campaign_name, tag)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for index, input_file in indexed_input_files:
            output_file = output_dir / f"{process_name}_{tag}_{index}.hepmc"
            if output_file.exists() and not args.overwrite:
                print(f"Skipping existing output: {output_file}")
                continue

            seed = seed_for_file(args.seed, index)
            rows.append(manifest_row(input_file, output_file, seed))

        if not rows:
            continue

        if args.dry_run:
            command = build_command(binary, "<manifest>", args)
            print(" ".join(command), f"# {len(rows)} files", flush=True)
            continue

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tsv") as manifest:
            manifest.writelines(rows)
            manifest.flush()
            command = build_command(binary, manifest.name, args)
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
