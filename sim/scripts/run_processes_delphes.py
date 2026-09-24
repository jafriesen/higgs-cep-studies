#!/usr/bin/env python3
import argparse
import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import discover_hepmc_files, load_yaml, resolve_template_path  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_processes,
    generation_stage_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Delphes on Pythia HepMC outputs from generator campaigns."
    )
    parser.add_argument(
        "--generator",
        choices=("superchic", "fpmc", "madgraph"),
        default="fpmc",
        help="Generator campaign family to process. Defaults to fpmc.",
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
        "--pythia-tag",
        default=None,
        help="hadr-Pythia subcampaign tag. Defaults to the process campaign config.",
    )
    parser.add_argument(
        "--tag",
        "--delphes-tag",
        dest="tag",
        default=None,
        help="sim-Delphes subcampaign tag. Defaults to the process campaign config.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional Pythia input file cap per process.",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=None,
        help="Process only this 1-based index in the sorted HepMC input file list.",
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


def validate_args(args):
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.file_index is not None and args.file_index <= 0:
        raise RuntimeError("--file-index must be > 0")


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def configured_delphes_dir(config):
    path = os.environ.get("DELPHES_DIR") or os.environ.get("HIGGS_CEP_DELPHES_DIR")
    if path:
        return resolve_template_path(path, config, base=ROOT)

    path = config.get("paths", {}).get("delphes-dir")
    if path:
        return resolve_template_path(path, config, base=ROOT)

    default_path = ROOT.parent / "delphes"
    if default_path.is_dir():
        return default_path

    path_executable = shutil.which("DelphesHepMC3")
    if path_executable:
        return Path(path_executable).resolve().parent

    raise RuntimeError(
        "Could not determine Delphes directory. Run `source setup_env.sh`, "
        "set DELPHES_DIR, or define paths.delphes-dir in config.yaml."
    )


def config_with_delphes_dir(config, delphes_dir):
    merged = dict(config)
    merged["paths"] = dict(config.get("paths", {}))
    merged["paths"].setdefault("delphes-dir", str(delphes_dir))
    return merged


def configured_card(config, delphes_dir, card_arg):
    config = config_with_delphes_dir(config, delphes_dir)
    if card_arg:
        return resolve_template_path(card_arg, config, base=ROOT)
    card = config.get("sim", {}).get("delphes-card")
    if not card:
        raise RuntimeError("config.yaml must define sim.delphes-card")
    return resolve_template_path(card, config, base=ROOT)


def binary_path(root):
    build_dir = Path(os.environ.get("HIGGS_CEP_DELPHES_BUILD_DIR") or os.environ.get("TMPDIR", "/tmp")) / f"higgs_cep_delphes_{os.environ.get('USER', 'user')}"
    return build_dir / "DelphesDepMC3"


def delphes_setup_script(root):
    script = root / "env" / "setup_delphes.sh"
    if not script.is_file():
        raise RuntimeError(f"Delphes setup script does not exist: {script}")
    return script


def root_config(flag, delphes_dir):
    script = "\n".join(
        [
            f"export HIGGS_CEP_DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"source {shlex.quote(str(delphes_setup_script(ROOT)))}",
            f"root-config {shlex.quote(flag)}",
        ]
    )
    try:
        output = subprocess.check_output(["bash", "-lc", script], cwd=ROOT, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "root-config not available; run `source setup_env.sh` first"
        ) from exc
    return shlex.split(output.strip())


def build_binary(root, delphes_dir):
    binary = binary_path(root)
    src = root / "sim" / "scripts" / "DelphesDepMC3.cpp"
    lib = delphes_dir / "libDelphes.so"
    if not lib.is_file():
        raise RuntimeError(f"Delphes library does not exist: {lib}")
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary

    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(src),
        f"-I{delphes_dir}", f"-I{delphes_dir / 'external'}", f"-I{delphes_dir / 'external' / 'tcl'}",
        *root_config("--cflags", delphes_dir),
        f"-L{delphes_dir}", f"-Wl,-rpath,{delphes_dir}",
        "-lDelphes",
        *root_config("--libs", delphes_dir),
        "-o", str(binary),
    ]
    script = "\n".join(
        [
            f"export HIGGS_CEP_DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"source {shlex.quote(str(delphes_setup_script(root)))}",
            f"exec {shlex.join(command)}",
        ]
    )
    subprocess.run(["bash", "-lc", script], cwd=root, check=True)
    return binary


def build_command(binary, card, output_dir, input_files):
    return [str(binary), str(card), str(output_dir), *[str(path) for path in input_files]]


def sourced_files(card):
    pattern = re.compile(r"^\s*source\s+['\"]?([^'\"\s]+)")
    names = []
    for line in card.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            names.append(match.group(1))
    return names


@contextlib.contextmanager
def runtime_card(card, delphes_dir):
    missing = [name for name in sourced_files(card) if not (card.parent / name).exists()]
    if not missing:
        yield card
        return

    include_dirs = [
        delphes_dir / "cards" / "CMS_PhaseII",
        delphes_dir / "cards",
    ]
    with tempfile.TemporaryDirectory(prefix="higgs_cep_delphes_card_") as tmpdir:
        tmpdir = Path(tmpdir)
        staged_card = tmpdir / card.name
        shutil.copy2(card, staged_card)
        for name in sorted(set(missing)):
            source = next((directory / name for directory in include_dirs if (directory / name).is_file()), None)
            if source is None:
                raise RuntimeError(
                    f"Card {card} sources {name}, but it was not found beside the card "
                    f"or under {', '.join(str(path) for path in include_dirs)}"
                )
            (tmpdir / name).symlink_to(source)
        yield staged_card


def run_delphes(command, delphes_dir):
    script = "\n".join(
        [
            f"export HIGGS_CEP_DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"source {shlex.quote(str(delphes_setup_script(ROOT)))}",
            f"exec {shlex.join(command)}",
        ]
    )
    subprocess.run(["bash", "-lc", script], cwd=delphes_dir, check=True)


def default_subcampaign(process_cfg, stage):
    defaults = process_cfg.get("default_campaign") or {}
    if not isinstance(defaults, dict):
        return None
    return defaults.get(stage)


def stage_root(generator, process_name, campaign_name, stage, tag):
    try:
        return generation_stage_root(
            generator, process_name, campaign_name, stage, subcampaign=tag
        )
    except RuntimeError as exc:
        if tag is not None:
            raise
        raise RuntimeError(
            f"No default {stage} subcampaign for {generator} process "
            f"'{process_name}' campaign '{campaign_name}'"
        ) from exc


def input_hepmc_dir(generator, process_name, campaign_name, pythia_tag):
    return (
        stage_root(generator, process_name, campaign_name, "hadr-pythia", pythia_tag)
        / "hepmc"
    )


def delphes_campaign_root(generator, process_name, campaign_name, delphes_tag):
    return stage_root(generator, process_name, campaign_name, "sim-delphes", delphes_tag)


def metadata_text(
    generator,
    process_name,
    campaign_name,
    pythia_tag,
    delphes_tag,
    input_dir,
    output_dir,
    card,
    command,
):
    return "\n".join(
        [
            f"generator: {generator}",
            f"process: {process_name}",
            f"campaign: {campaign_name}",
            f"hadr_pythia_tag: {pythia_tag}",
            f"sim_delphes_tag: {delphes_tag}",
            f"input_dir: {input_dir}",
            f"output_dir: {output_dir}",
            f"delphes_card: {card}",
            f"command: {command}",
            "",
        ]
    )


def write_metadata(
    delphes_dir,
    generator,
    process_name,
    campaign_name,
    pythia_tag,
    delphes_tag,
    input_dir,
    output_dir,
    card,
):
    delphes_dir.mkdir(parents=True, exist_ok=True)
    (delphes_dir / "metadata.yaml").write_text(
        metadata_text(
            generator,
            process_name,
            campaign_name,
            pythia_tag,
            delphes_tag,
            input_dir,
            output_dir,
            card,
            shlex.join(sys.argv),
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    root = ROOT
    config = load_yaml(root / "config.yaml")
    processes = generation_processes(args.generator)
    delphes_dir = configured_delphes_dir(config)
    card = configured_card(config, delphes_dir, args.card)
    binary = binary_path(root) if args.dry_run else build_binary(root, delphes_dir)

    if not args.dry_run and not os.access(binary, os.X_OK):
        raise RuntimeError(f"Delphes runner is not executable: {binary}")
    if not card.is_file():
        raise RuntimeError(f"Delphes card does not exist: {card}")

    card_context = contextlib.nullcontext(card) if args.dry_run else runtime_card(card, delphes_dir)
    with card_context as card_for_run:
        for process_name in selected_processes(processes, args.processes):
            process_cfg = processes[process_name] or {}
            campaign_name, _ = generation_campaign_config(args.generator, process_name, args.campaign)
            pythia_tag = args.pythia_tag or default_subcampaign(process_cfg, "hadr-pythia")
            delphes_tag = args.tag or default_subcampaign(process_cfg, "sim-delphes")
            input_dir = input_hepmc_dir(args.generator, process_name, campaign_name, pythia_tag)
            delphes_output_root = delphes_campaign_root(
                args.generator, process_name, campaign_name, delphes_tag
            )
            output_dir = delphes_output_root / "root"
            logs_dir = delphes_output_root / "logs"
            try:
                input_files = discover_hepmc_files(input_dir, max_files=args.max_files)
            except RuntimeError:
                if not args.dry_run:
                    raise
                print(
                    f"No .hepmc files found in {input_dir}; "
                    f"Delphes output would be written under {output_dir}",
                    flush=True,
                )
                continue
            if args.file_index is not None:
                if args.file_index > len(input_files):
                    raise RuntimeError(
                        f"--file-index {args.file_index} exceeds the {len(input_files)} "
                        f"HepMC files found for {process_name}"
                    )
                input_files = [input_files[args.file_index - 1]]
            pending_inputs = []

            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                logs_dir.mkdir(parents=True, exist_ok=True)
                write_metadata(
                    delphes_output_root,
                    args.generator,
                    process_name,
                    campaign_name,
                    pythia_tag,
                    delphes_tag,
                    input_dir,
                    output_dir,
                    card,
                )

            for input_file in input_files:
                output_file = output_dir / f"{input_file.stem}.root"
                if output_file.exists() and not args.overwrite:
                    print(f"Skipping existing output: {output_file}")
                    continue
                pending_inputs.append(input_file)

            if not pending_inputs:
                continue

            command = build_command(binary, card_for_run, output_dir, pending_inputs)
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue

            if args.overwrite:
                for input_file in pending_inputs:
                    output_file = output_dir / f"{input_file.stem}.root"
                    if output_file.exists():
                        output_file.unlink()
            run_delphes(command, delphes_dir)


if __name__ == "__main__":
    main()
