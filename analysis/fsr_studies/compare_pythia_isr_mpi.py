#!/usr/bin/env python3
"""Shower the SuperChic H->bb LHE with Pythia 8 for every ISR/MPI/Remnants combination.

FSR and hadronization stay on in all eight runs. Each run is analyzed by the same
HepMC+FastJet code as compare_herwig_pythia.py and the median m(jj)/125 is compared.
Combinations where Pythia cannot generate a single event are reported as failed.
"""

import argparse
import itertools
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import compare_herwig_pythia as base  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_DIR / "output/pythia_isr_mpi_remnants"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evrec", type=Path, default=base.DEFAULT_EVREC)
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31122001)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild", action="store_true", help="Recompile the C++ helpers")
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_ISR_MPI_ENV") == "1":
        return
    command = "\n".join((
        f"source {shlex.quote(str(base.REPO / 'setup_env.sh'))}",
        "export HIGGS_CEP_ISR_MPI_ENV=1",
        f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
    ))
    raise SystemExit(subprocess.run(["bash", "-lc", command], cwd=base.REPO).returncode)


def label_for(isr, mpi, remnants):
    return f"isr_{isr}_mpi_{mpi}_rem_{remnants}"


def shower(binary, args, hepmc, isr, mpi, remnants):
    command = [str(binary), "--lhe", str(args.evrec), "--output", str(hepmc),
               "--events", str(args.events), "--seed", str(args.seed),
               "--isr", isr, "--mpi", mpi, "--remnants", remnants]
    print("Running:", shlex.join(command))
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    match = re.search(r"wrote (\d+) events .* \((\d+) failed\)", completed.stdout)
    return int(match.group(1)), int(match.group(2))


def make_plot(events_by_label, summary, labels, output_dir):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0.0, 1.4, 71)
    for label in labels:
        values = [float(r["m_leading"]) / 125.0 for r in events_by_label[label]
                  if math.isclose(float(r["radius"]), 0.4)]
        axes[0].hist(values, bins=bins, density=True, histtype="step", linewidth=1.4,
                     label=label)
        medians = [row["median_m_over_mh"] for radius in base.RADII for row in summary
                   if row["label"] == label and math.isclose(row["radius"], radius)]
        axes[1].plot(base.RADII, medians, marker="o", label=label)
    axes[0].axvline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("m(jj)/125 GeV, two leading anti-kt R=0.4 jets")
    axes[0].set_ylabel("Normalized events")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Jet radius R")
    axes[1].set_ylabel("median m(jj) / 125 GeV")
    axes[1].legend(frameon=False, fontsize=7)
    figure.suptitle("SuperChic H->bb, Pythia 8 FSR on: ISR/MPI/Remnants combinations "
                    "(Remnants=on fails for every event)")
    figure.tight_layout()
    figure.savefig(output_dir / "pythia_isr_mpi_remnants.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shower_binary, analyze_binary = base.build_helpers(args.rebuild)

    events_by_label = {}
    counts = {}
    for isr, mpi, remnants in itertools.product(("off", "on"), repeat=3):
        label = label_for(isr, mpi, remnants)
        hepmc = output_dir / f"{label}.hepmc"
        counts[label] = shower(shower_binary, args, hepmc, isr, mpi, remnants)
        if counts[label][0] == 0:
            continue
        csv_path = output_dir / f"events_{label}.csv"
        base.analyze(analyze_binary, hepmc, csv_path, label)
        events_by_label[label] = base.read_rows(csv_path)

    print(f"\n{'combination':28s} {'written':>8s} {'failed':>8s}")
    for label, (written, failed) in counts.items():
        print(f"{label:28s} {written:8d} {failed:8d}")

    labels = list(events_by_label)
    summary = base.summarize(events_by_label)
    base.write_csv(output_dir / "pythia_isr_mpi_remnants_summary.csv", summary)
    base.print_table(summary, labels)
    make_plot(events_by_label, summary, labels, output_dir)
    print(f"\nWrote summary CSV and plot to {output_dir}")


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
