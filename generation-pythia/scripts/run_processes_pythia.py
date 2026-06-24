#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, process_max_files, resolve_process_campaign  # noqa: E402


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
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
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


def build_command(wrapper, input_dir, output_file, args, max_files):
    command = [
        str(wrapper),
        "--input",
        str(input_dir),
        "--output",
        str(output_file),
    ]
    if args.max_events is not None:
        command.extend(["--max-events", str(args.max_events)])
    if max_files is not None:
        command.extend(["--max-files", str(max_files)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.verbose:
        command.append("--verbose")
    return command


def main():
    args = parse_args()
    root = ROOT
    processes = load_yaml(root / "processes.yaml")
    wrapper = root / "generation-pythia" / "scripts" / "run_superchic_pythia.sh"

    for process_name in selected_processes(processes, args.processes):
        campaign_dir, campaign = resolve_process_campaign(process_name, args.campaign)
        max_files = args.max_files
        if max_files is None:
            max_files = process_max_files(process_name)
        input_dir = campaign_dir / "evrecs"
        output_dir = campaign_dir / "GEN-pythia"
        output_file = output_dir / f"{process_name}_{campaign}.hepmc"

        if not input_dir.exists():
            raise RuntimeError(f"Missing SuperChic evrecs directory for {process_name}: {input_dir}")
        if output_file.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_file}")
            continue

        command = build_command(wrapper, input_dir, output_file, args, max_files)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
