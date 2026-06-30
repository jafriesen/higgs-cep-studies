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

from common.config_utils import discover_event_files, load_yaml, process_max_files  # noqa: E402
from common.path_helper import campaign_config, pythia_root, superchic_evrecs_dir  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Pythia on SuperChic campaigns listed in processes.yaml."
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name to run. May be repeated. Defaults to all processes.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Main campaign name to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Pythia campaign name within the main campaign's Pythia folder. Defaults to the campaign name.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap per output file")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional SuperChic input file cap per process. Defaults to process max_files.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional Pythia seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing HepMC outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Pythia")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose mode to the Pythia bridge")
    return parser.parse_args()


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def binary_path(root):
    build_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"higgs_cep_pythia_{os.environ.get('USER', 'user')}"
    return build_dir / "process_superchic"


def build_binary(root):
    """Compile process_superchic.cc against the LCG view's Pythia8/HepMC3, caching the binary."""
    binary = binary_path(root)
    src = root / "generation-pythia" / "scripts" / "process_superchic.cc"
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary

    pythia8 = os.environ.get("PYTHIA8")
    lcg_view = os.environ.get("LCG_VIEW")
    if not pythia8 or not lcg_view:
        raise RuntimeError("$PYTHIA8/$LCG_VIEW not set; run `source setup_env.sh` first")
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


def main():
    args = parse_args()
    root = ROOT
    processes = load_yaml(root / "processes.yaml")
    binary = build_binary(root) if not args.dry_run else binary_path(root)

    for process_name in selected_processes(processes, args.processes):
        campaign_name, _ = campaign_config(process_name, args.campaign, require_known=False)
        tag = args.tag or campaign_name
        max_files = args.max_files
        if max_files is None:
            max_files = process_max_files(process_name)

        input_dir = superchic_evrecs_dir(process_name, campaign_name)
        input_files = discover_event_files(input_dir, max_files=max_files)

        output_dir = pythia_root(process_name, campaign_name, tag=tag)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for index, input_file in enumerate(input_files, start=1):
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
