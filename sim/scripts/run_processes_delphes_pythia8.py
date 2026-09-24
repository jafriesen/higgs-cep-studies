#!/usr/bin/env python3
"""Run DelphesPythia8 directly on generator LHE outputs (no HepMC intermediate).

Pythia hadronization settings come from a template cmnd file in sim/Cards/pythia/;
the per-input Beams:LHEF, seed, and event-count settings are appended here.
Requires the Delphes checkout built with:
    make HAS_PYTHIA8=true PYTHIA8="$PYTHIA8"
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_processes_delphes as base  # noqa: E402

ROOT = base.ROOT

from common.config_utils import discover_event_files  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_campaign_root,
    generation_config,
    generation_processes,
)

CMND_TEMPLATES = {
    "madgraph": ROOT / "sim" / "Cards" / "pythia" / "madgraph_shower.cmnd",
    "superchic": ROOT / "sim" / "Cards" / "pythia" / "superchic_hadronize.cmnd",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DelphesPythia8 on generator LHE outputs, hadronizing in-process."
    )
    parser.add_argument(
        "--generator",
        choices=tuple(CMND_TEMPLATES),
        required=True,
        help="Generator campaign family to process.",
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
        help="Main campaign name. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--tag",
        "--delphes-tag",
        dest="tag",
        default=None,
        help="sim-Delphes subcampaign tag. Defaults to the process campaign config.",
    )
    parser.add_argument(
        "--cmnd-template",
        default=None,
        help="Pythia cmnd template path. Defaults to the per-generator template in sim/Cards/pythia/.",
    )
    parser.add_argument(
        "--fsr",
        choices=("on", "off"),
        default=None,
        help="Override PartonLevel:FSR from the cmnd template.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed; each file or file part gets a distinct seed.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional per-file event cap. Defaults to the whole input file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional generator input file cap per process.",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=None,
        help="Process only this 1-based index in the sorted generator input file list.",
    )
    parser.add_argument(
        "--part-index",
        type=int,
        default=None,
        help="Optional 1-based output part index used when splitting an input file.",
    )
    parser.add_argument(
        "--skip-events",
        type=int,
        default=0,
        help="Number of LHE events to skip before processing this part.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Delphes ROOT outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DelphesPythia8 commands without running them.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Use the particle-level filtered driver: events without two jets "
             "above 12 GeV separated by dphi > 2.8 skip the Delphes chain "
             "entirely, but are still written so n_generated is preserved.",
    )
    parser.add_argument(
        "--filter-activity-max",
        type=float,
        default=None,
        help="With --filter, also veto events whose summed charged pT outside the two "
             "leading particle-level jets exceeds this (GeV). Exclusive signal sits near "
             "3 GeV and inclusive QCD near 26 GeV, so 25 costs 0.3%% of signal and removes "
             "about half the QCD. Omit to disable the activity veto.",
    )
    parser.add_argument(
        "--card",
        default=None,
        help="Delphes card path. Defaults to sim.delphes-card in config.yaml.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.file_index is not None and args.file_index <= 0:
        raise RuntimeError("--file-index must be > 0")
    if args.part_index is not None and args.part_index <= 0:
        raise RuntimeError("--part-index must be > 0")
    if args.skip_events < 0:
        raise RuntimeError("--skip-events must be >= 0")
    if args.skip_events and args.part_index is None:
        raise RuntimeError("--skip-events requires --part-index to avoid output collisions")
    if args.part_index is not None and args.max_events is None:
        raise RuntimeError("--part-index requires --max-events")


def event_chunks(max_events, events_per_job):
    """Return (part index, event offset, event count) tuples for one input file."""
    if max_events <= 0 or events_per_job <= 0:
        raise RuntimeError("event counts must be > 0")
    return [
        (part_index + 1, offset, min(events_per_job, max_events - offset))
        for part_index, offset in enumerate(range(0, max_events, events_per_job))
    ]


def part_suffix(part_index):
    return f"__part{part_index:04d}" if part_index is not None else ""


def seed_for_file(base_seed, file_index, part_index=None):
    """Distinct, reproducible seed per input file and optional file part."""
    if base_seed is None:
        return None
    if part_index is None:
        offset = file_index
    else:
        file_offset = file_index - 1
        part_offset = part_index - 1
        offset = (
            (file_offset + part_offset) * (file_offset + part_offset + 1) // 2
            + part_offset
            + 1
        )
    max_seed = 900000000
    return ((base_seed + offset - 1) % max_seed) + 1


def gen_input_dir(generator, process_name, campaign_name):
    cfg = generation_config(generator)
    return (
        generation_campaign_root(generator, process_name, campaign_name)
        / cfg["generation_dir"]
        / "evrecs"
    )


FILTER_SOURCE = ROOT / "sim" / "scripts" / "DelphesPythia8Filter.cpp"


def pythia8_binary(delphes_dir):
    binary = delphes_dir / "DelphesPythia8"
    if not binary.is_file():
        raise RuntimeError(
            f"DelphesPythia8 does not exist: {binary}. Build Delphes with "
            '`make HAS_PYTHIA8=true PYTHIA8="$PYTHIA8"` after sourcing setup_env.sh.'
        )
    return binary


def filtered_binary(delphes_dir):
    """Build (or reuse) the particle-level filtered driver.

    Same on-demand pattern as run_processes_delphes.build_binary, with Pythia8
    added. Nothing in the Delphes installation is modified.
    """
    import os

    build_dir = Path(os.environ.get("HIGGS_CEP_DELPHES_BUILD_DIR")
        or os.environ.get("TMPDIR", "/tmp")) / (
        f"higgs_cep_delphes_{os.environ.get('USER', 'user')}"
    )
    binary = build_dir / "DelphesPythia8Filter"
    if not FILTER_SOURCE.is_file():
        raise RuntimeError(f"filter source does not exist: {FILTER_SOURCE}")
    lib = delphes_dir / "libDelphes.so"
    if not lib.is_file():
        raise RuntimeError(f"Delphes library does not exist: {lib}")
    if binary.exists() and binary.stat().st_mtime >= FILTER_SOURCE.stat().st_mtime:
        return binary

    pythia8 = os.environ.get("PYTHIA8")
    if not pythia8:
        raise RuntimeError("PYTHIA8 is not set; source setup_env.sh first")
    build_dir.mkdir(parents=True, exist_ok=True)
    # DelphesPythia8Reader and its ROOT dictionary are not in libDelphes.so --
    # they are built only into the stock executable -- so link the objects the
    # Delphes build already produced rather than recompiling them.
    objects = [
        delphes_dir / "tmp" / "classes" / "DelphesPythia8Reader.o",
        delphes_dir / "tmp" / "classes" / "ClassesPythia8Dict.o",
    ]
    missing = [str(item) for item in objects if not item.is_file()]
    if missing:
        raise RuntimeError(
            f"Delphes Pythia8 objects are missing: {missing}. Build Delphes with "
            '`make HAS_PYTHIA8=true PYTHIA8="$PYTHIA8"` first.'
        )
    command = [
        "g++", "-std=c++17", "-O2", str(FILTER_SOURCE), *[str(item) for item in objects],
        f"-I{delphes_dir}", f"-I{delphes_dir / 'external'}",
        f"-I{delphes_dir / 'external' / 'tcl'}",
        f"-I{Path(pythia8) / 'include'}", f"-I{Path(pythia8) / 'include' / 'Pythia8'}",
        *base.root_config("--cflags", delphes_dir),
        f"-L{delphes_dir}", f"-Wl,-rpath,{delphes_dir}", "-lDelphes",
        f"-L{Path(pythia8) / 'lib'}", f"-Wl,-rpath,{Path(pythia8) / 'lib'}", "-lpythia8",
        *base.root_config("--libs", delphes_dir), "-lEG",
        "-o", str(binary),
    ]
    script = "\n".join([
        f"export HIGGS_CEP_DELPHES_DIR={shlex.quote(str(delphes_dir))}",
        f"source {shlex.quote(str(base.delphes_setup_script(ROOT)))}",
        shlex.join(command),
    ])
    print(f"building {binary}", flush=True)
    subprocess.run(["bash", "-lc", script], cwd=ROOT, check=True)
    return binary


def write_cmnd(
    cmnd_file, template_text, input_file, seed, max_events, fsr, skip_events=0
):
    lines = [template_text.rstrip("\n"), ""]
    if fsr is not None:
        lines.append(f"PartonLevel:FSR = {fsr}")
    lines.append(f"Beams:LHEF = {input_file}")
    if skip_events:
        lines.append(f"Beams:nSkipLHEFatInit = {skip_events}")
    if seed is not None:
        lines.append("Random:setSeed = on")
        lines.append(f"Random:seed = {seed}")
    number_of_events = max_events if max_events is not None else 1000000000
    lines.append(f"Main:numberOfEvents = {number_of_events}")
    lines.append("Main:timesAllowErrors = 1000")
    cmnd_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metadata_text(args, process_name, campaign_name, delphes_tag, input_dir, output_dir, cmnd_template, card):
    return "\n".join(
        [
            f"generator: {args.generator}",
            f"process: {process_name}",
            f"campaign: {campaign_name}",
            f"sim_delphes_tag: {delphes_tag}",
            "pipeline: delphes-pythia8-direct",
            f"input_dir: {input_dir}",
            f"cmnd_template: {cmnd_template}",
            f"fsr: {args.fsr if args.fsr is not None else 'template'}",
            f"seed_base: {args.seed if args.seed is not None else 'none'}",
            f"max_events: {args.max_events if args.max_events is not None else 'all'}",
            f"part_index: {args.part_index if args.part_index is not None else 'none'}",
            f"skip_events: {args.skip_events}",
            f"delphes_card: {card}",
            f"particle_filter: {'on' if args.filter else 'off'}",
            (
                "particle_filter_cuts: two anti-kt R=0.4 jets, pT > 12 GeV, dphi > 2.8"
                + (
                    f", charged pT outside jets < {args.filter_activity_max} GeV"
                    if args.filter_activity_max
                    else ""
                )
            )
            if args.filter else "particle_filter_cuts: none",
            f"command: {shlex.join(sys.argv)}",
            "",
        ]
    )


def run_delphes_pythia8(command, delphes_dir, log_file, environment=None):
    """Run the driver. `environment` sets the filter thresholds and the Delphes seed,
    which the driver reads from the environment rather than from the card."""
    exports = [
        f"export HIGGS_CEP_DELPHES_DIR={shlex.quote(str(delphes_dir))}",
    ] + [
        f"export {name}={shlex.quote(str(value))}"
        for name, value in sorted((environment or {}).items())
    ]
    script = "\n".join(
        exports
        + [
            f"source {shlex.quote(str(base.delphes_setup_script(ROOT)))}",
            f"exec {shlex.join(command)}",
        ]
    )
    with open(log_file, "w", encoding="utf-8") as log:
        subprocess.run(
            ["bash", "-lc", script],
            cwd=delphes_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def main():
    args = parse_args()
    validate_args(args)
    config = base.load_yaml(ROOT / "config.yaml")
    processes = generation_processes(args.generator)
    delphes_dir = base.configured_delphes_dir(config)
    card = base.configured_card(config, delphes_dir, args.card)
    binary = filtered_binary(delphes_dir) if args.filter else pythia8_binary(delphes_dir)
    cmnd_template = (
        Path(args.cmnd_template).resolve()
        if args.cmnd_template
        else CMND_TEMPLATES[args.generator]
    )
    if not cmnd_template.is_file():
        raise RuntimeError(f"cmnd template does not exist: {cmnd_template}")
    template_text = cmnd_template.read_text(encoding="utf-8")
    if not card.is_file():
        raise RuntimeError(f"Delphes card does not exist: {card}")

    with base.runtime_card(card, delphes_dir) as card_for_run:
        for process_name in base.selected_processes(processes, args.processes):
            process_cfg = processes[process_name] or {}
            campaign_name, _ = generation_campaign_config(
                args.generator, process_name, args.campaign
            )
            delphes_tag = args.tag or base.default_subcampaign(process_cfg, "sim-delphes")
            delphes_output_root = base.delphes_campaign_root(
                args.generator, process_name, campaign_name, delphes_tag
            )
            input_dir = gen_input_dir(args.generator, process_name, campaign_name)
            input_files = discover_event_files(input_dir, max_files=args.max_files)
            if args.file_index is not None:
                if args.file_index > len(input_files):
                    raise RuntimeError(
                        f"--file-index {args.file_index} exceeds the {len(input_files)} "
                        f"generator input files found for {process_name}"
                    )
                input_files = [input_files[args.file_index - 1]]

            output_dir = delphes_output_root / "root"
            logs_dir = delphes_output_root / "logs"
            cmnd_dir = delphes_output_root / "cmnd"
            if not args.dry_run:
                for directory in (output_dir, logs_dir, cmnd_dir):
                    directory.mkdir(parents=True, exist_ok=True)
                (delphes_output_root / "metadata.yaml").write_text(
                    metadata_text(
                        args, process_name, campaign_name, delphes_tag,
                        input_dir, output_dir, cmnd_template, card,
                    ),
                    encoding="utf-8",
                )

            all_files = discover_event_files(input_dir)
            for input_file in input_files:
                file_index = all_files.index(input_file) + 1
                output_stem = f"{input_file.stem}{part_suffix(args.part_index)}"
                output_file = output_dir / f"{output_stem}.root"
                if output_file.exists() and not args.overwrite:
                    print(f"Skipping existing output: {output_file}")
                    continue
                seed = seed_for_file(args.seed, file_index, args.part_index)
                cmnd_file = cmnd_dir / f"{output_stem}.cmnd"
                command = [str(binary), str(card_for_run), str(cmnd_file), str(output_file)]
                print(" ".join(command), flush=True)
                if args.dry_run:
                    continue
                write_cmnd(
                    cmnd_file,
                    template_text,
                    input_file,
                    seed,
                    args.max_events,
                    args.fsr,
                    args.skip_events,
                )
                if output_file.exists():
                    output_file.unlink()
                log_file = logs_dir / f"{output_stem}.log"
                # Delphes seeds gRandom from the card (default 0 -> clock based).
                # Reuse the per-file/part seed so pileup and smearing are distinct
                # per job and reproducible.
                environment = {}
                if seed is not None:
                    environment["DELPHES_RANDOM_SEED"] = seed
                if args.filter and args.filter_activity_max:
                    environment["DELPHES_FILTER_ACTIVITY_MAX"] = args.filter_activity_max
                try:
                    run_delphes_pythia8(command, delphes_dir, log_file, environment)
                except subprocess.CalledProcessError:
                    if output_file.exists():
                        output_file.unlink()
                    raise


if __name__ == "__main__":
    main()
