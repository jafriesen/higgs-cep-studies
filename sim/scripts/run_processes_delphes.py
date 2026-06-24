#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_process_campaign, resolve_template_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Delphes on Pythia HepMC outputs listed in processes.yaml."
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Delphes ROOT outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Delphes commands without running them.",
    )
    parser.add_argument(
        "--card",
        default=None,
        help="Delphes card path. Defaults to sim.delphes-card in config.yaml.",
    )
    return parser.parse_args()


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def configured_delphes_dir(config):
    path = config.get("paths", {}).get("delphes-dir")
    if not path:
        raise RuntimeError("config.yaml must define paths.delphes-dir")
    return resolve_template_path(path, config, base=ROOT)


def configured_card(config, card_arg):
    if card_arg:
        return resolve_template_path(card_arg, config, base=ROOT)
    card = config.get("sim", {}).get("delphes-card")
    if not card:
        raise RuntimeError("config.yaml must define sim.delphes-card")
    return resolve_template_path(card, config, base=ROOT)


def delphes_executable(delphes_dir):
    executable = delphes_dir / "DelphesHepMC3"
    if executable.exists():
        return executable

    path_executable = shutil.which("DelphesHepMC3")
    if path_executable:
        return Path(path_executable)

    raise RuntimeError(
        f"Could not find DelphesHepMC3 at {executable} or on PATH. "
        "Run source setup_env.sh first or update paths.delphes-dir in config.yaml."
    )


def build_command(executable, card, output_file, input_file):
    return [str(executable), str(card), str(output_file), str(input_file)]


def run_delphes(command, delphes_dir):
    lcg_view = Path(os.environ.get(
        "DELPHES_LCG_VIEW",
        "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh",
    ))
    if not lcg_view.is_file():
        raise RuntimeError(f"Delphes LCG view setup script does not exist: {lcg_view}")

    script = "\n".join(
        [
            f"source {shlex.quote(str(lcg_view))}",
            f"export LD_LIBRARY_PATH={shlex.quote(str(delphes_dir))}:$LD_LIBRARY_PATH",
            f"exec {shlex.join(command)}",
        ]
    )
    subprocess.run(["bash", "-lc", script], cwd=delphes_dir, check=True)


def main():
    args = parse_args()
    root = ROOT
    config = load_yaml(root / "config.yaml")
    processes = load_yaml(root / "processes.yaml")
    delphes_dir = configured_delphes_dir(config)
    card = configured_card(config, args.card)
    executable = delphes_executable(delphes_dir)

    if not executable.is_file():
        raise RuntimeError(f"Delphes executable is not a file: {executable}")
    if not executable.exists():
        raise RuntimeError(f"Delphes executable does not exist: {executable}")
    if not os.access(executable, os.X_OK):
        raise RuntimeError(f"Delphes executable is not executable: {executable}")
    if not card.is_file():
        raise RuntimeError(f"Delphes card does not exist: {card}")

    for process_name in selected_processes(processes, args.processes):
        campaign_dir, campaign = resolve_process_campaign(process_name, args.campaign)
        input_file = campaign_dir / "GEN-pythia" / f"{process_name}_{campaign}.hepmc"
        output_dir = campaign_dir / "SIM-delphes"
        output_file = output_dir / f"{process_name}_{campaign}.root"

        if not input_file.is_file():
            raise RuntimeError(f"Missing Pythia HepMC input for {process_name}: {input_file}")
        if output_file.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_file}")
            continue

        command = build_command(executable, card, output_file, input_file)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()
        run_delphes(command, delphes_dir)


if __name__ == "__main__":
    main()
