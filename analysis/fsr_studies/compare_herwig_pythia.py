#!/usr/bin/env python3
"""Shower the same SuperChic H->bb events with Pythia 8 and Herwig 7 and compare the FSR loss.

Four configurations, all analyzed by the same HepMC+FastJet code:
  pythia_superchic  Pythia 8 on the SuperChic LHE exactly as the production campaign runs it
  pythia_rest       Pythia 8 on the rest-frame LHE (checks that the rewrite changes nothing)
  herwig_angular    Herwig 7 default angular-ordered shower + cluster hadronization
  herwig_dipole     Herwig 7 dipole shower + cluster hadronization
"""

import argparse
import csv
import math
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVREC = (
    REPO / "output-superchic/Hbb/Hbb__v01/gen-SuperChic/evrecs/evrecHbb__v01_1.dat"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "output/herwig_comparison"
RADII = (0.4, 0.6, 0.8, 1.0)

# Herwig's default angular-ordered shower.
ANGULAR_CASCADE = """set LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler
set /Herwig/Shower/ShowerHandler:MPIHandler NULL"""

# Herwig's dipole shower. SuperChic writes m_b = 4.75 GeV while Herwig's nominal b mass
# is 4.18 GeV, and the dipole shower refuses off-shell coloured legs unless told otherwise.
DIPOLE_CASCADE = """read snippets/Dipole_AutoTunes_gss.in
set LHEHandler:CascadeHandler /Herwig/DipoleShower/DipoleShowerHandler
set /Herwig/DipoleShower/DipoleShowerHandler:MPIHandler NULL
insert /Herwig/DipoleShower/DipoleShowerHandler:OffShellInShower 0 5
insert /Herwig/DipoleShower/DipoleShowerHandler:OffShellInShower 0 -5"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evrec", type=Path, default=DEFAULT_EVREC,
                        help="SuperChic evrec LHE to shower")
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31122001)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild", action="store_true", help="Recompile the C++ helpers")
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_HERWIG_COMPARE_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_HERWIG_COMPARE_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def config_flags(program, *options):
    flags = []
    for option in options:
        completed = subprocess.run([program, option], check=True, text=True,
                                   stdout=subprocess.PIPE)
        flags += shlex.split(completed.stdout.strip())
    return flags


def compile_helper(source, rebuild, flags):
    binary = Path(tempfile.gettempdir()) / f"higgs_cep_{source.stem}_{os.getuid()}"
    if not rebuild and binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    command = ["g++", "-O2", str(source), "-o", str(binary)] + flags
    print("Compiling:", shlex.join(command))
    subprocess.run(command, cwd=REPO, check=True)
    return binary


def build_helpers(rebuild):
    hepmc = config_flags("HepMC3-config", "--cppflags")
    hepmc_link = config_flags("HepMC3-config", "--ldflags")
    fastjet = config_flags("fastjet-config", "--cxxflags")
    fastjet_link = config_flags("fastjet-config", "--libs")
    pythia = config_flags("pythia8-config", "--cxxflags")
    pythia_link = config_flags("pythia8-config", "--libs")
    shower = compile_helper(SCRIPT_DIR / "shower_pythia_lhe.cc", rebuild,
                            pythia + hepmc + pythia_link + hepmc_link)
    analyze = compile_helper(SCRIPT_DIR / "analyze_hepmc_fsr.cc", rebuild,
                             hepmc + fastjet + hepmc_link + fastjet_link)
    return shower, analyze


def run(command, **kwargs):
    print("Running:", shlex.join(str(part) for part in command))
    subprocess.run(command, check=True, **kwargs)


def make_rest_lhe(evrec, events, output):
    run([sys.executable, str(SCRIPT_DIR / "make_hbb_rest_lhe.py"), str(evrec),
         str(output), "--max-events", str(events)])


def shower_pythia(binary, lhe, hepmc, events, seed):
    run([str(binary), "--lhe", str(lhe), "--output", str(hepmc),
         "--events", str(events), "--seed", str(seed)])


def herwig_path(option):
    # herwig-config prints the path but exits non-zero, so the return code is ignored.
    completed = subprocess.run(["herwig-config", option], text=True, stdout=subprocess.PIPE)
    path = Path(completed.stdout.strip())
    if not path.is_dir():
        raise RuntimeError(f"herwig-config {option} did not give a directory: {path}")
    return path


def shower_herwig(cascade, lhe, hepmc, events, seed, run_name, work_dir):
    datadir = herwig_path("--datadir")
    libdir = herwig_path("--libdir")
    template = (SCRIPT_DIR / "herwig_hbb.in").read_text()
    card = work_dir / f"{run_name}.in"
    card.write_text(
        template.replace("@LHE@", str(lhe))
        .replace("@CASCADE@", cascade)
        .replace("@EVENTS@", str(events))
        .replace("@SEED@", str(seed))
        .replace("@HEPMC@", str(hepmc))
        .replace("@RUNNAME@", run_name)
    )
    repo_flags = [f"--repo={datadir / 'HerwigDefaults.rpo'}", "-L", str(libdir / "Herwig")]
    run(["Herwig", "read", *repo_flags, "-i", str(datadir), str(card)], cwd=work_dir)
    run(["Herwig", "run", *repo_flags, "-N", str(events), "-q",
         str(work_dir / f"{run_name}.run")], cwd=work_dir)


def analyze(binary, hepmc, csv_path, label):
    run([str(binary), "--input", str(hepmc), "--output", str(csv_path), "--label", label])


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows_by_label):
    summary = []
    for label, rows in rows_by_label.items():
        for radius in RADII:
            selected = [r for r in rows if math.isclose(float(r["radius"]), radius)]
            leading = [float(r["m_leading"]) for r in selected]
            outside = [1.0 - float(r["e_leading"]) / float(r["e_visible"]) for r in selected]
            summary.append(
                {
                    "label": label,
                    "radius": radius,
                    "events": len(selected),
                    "median_m_leading": statistics.median(leading),
                    "median_m_over_mh": statistics.median(leading) / 125.0,
                    "mean_m_leading": statistics.mean(leading),
                    "median_out_of_cone": statistics.median(outside),
                    "frac_below_100": sum(1 for v in leading if v < 100.0) / len(leading),
                    "median_m_visible": statistics.median(
                        float(r["m_visible"]) for r in selected),
                }
            )
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary, labels):
    def value(label, radius, key):
        return next(row[key] for row in summary
                    if row["label"] == label and math.isclose(row["radius"], radius))

    for key, title in (
        ("median_m_over_mh", "median m(jj) of the two leading jets, divided by 125 GeV"),
        ("median_out_of_cone", "median energy fraction outside the two leading jets"),
        ("frac_below_100", "fraction of events with m(jj) < 100 GeV"),
    ):
        print(f"\n{title}")
        print(f"{'':20s}" + "".join(f"  R={r:.1f}" for r in RADII))
        for label in labels:
            print(f"{label:20s}" + "".join(f"  {value(label, r, key):6.3f}" for r in RADII))


def make_plot(events_by_label, summary, labels, output_dir):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "pythia_superchic": "#000000",
        "pythia_rest": "#0072B2",
        "herwig_angular": "#D55E00",
        "herwig_dipole": "#009E73",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bins = np.linspace(0.0, 160.0, 65)
    for label in labels:
        values = [float(r["m_leading"]) for r in events_by_label[label]
                  if math.isclose(float(r["radius"]), 0.4)]
        axes[0].hist(values, bins=bins, density=True, histtype="step", linewidth=1.6,
                     color=colors[label], label=label)
    axes[0].axvline(125.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("m(jj), two leading anti-kt R=0.4 jets [GeV]")
    axes[0].set_ylabel("Normalized events")
    axes[0].legend(frameon=False, fontsize=8)

    for label in labels:
        values = [row["median_m_over_mh"] for radius in RADII
                  for row in summary
                  if row["label"] == label and math.isclose(row["radius"], radius)]
        axes[1].plot(RADII, values, marker="o", color=colors[label], label=label)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Jet radius R")
    axes[1].set_ylabel("median m(jj) / 125 GeV")
    axes[1].legend(frameon=False, fontsize=8)

    figure.suptitle("SuperChic H->bb showered by Pythia 8 and Herwig 7")
    figure.tight_layout()
    figure.savefig(output_dir / "herwig_vs_pythia_fsr.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.events <= 0:
        raise RuntimeError("--events must be positive")
    if not args.evrec.is_file():
        raise RuntimeError(f"SuperChic evrec not found: {args.evrec}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shower_binary, analyze_binary = build_helpers(args.rebuild)

    rest_lhe = output_dir / "hbb_rest_frame.lhe"
    make_rest_lhe(args.evrec, args.events, rest_lhe)

    shower_pythia(shower_binary, args.evrec, output_dir / "pythia_superchic.hepmc",
                  args.events, args.seed)
    shower_pythia(shower_binary, rest_lhe, output_dir / "pythia_rest.hepmc",
                  args.events, args.seed)
    shower_herwig(ANGULAR_CASCADE, rest_lhe, output_dir / "herwig_angular.hepmc",
                  args.events, args.seed, "herwig_angular", output_dir)
    shower_herwig(DIPOLE_CASCADE, rest_lhe, output_dir / "herwig_dipole.hepmc",
                  args.events, args.seed, "herwig_dipole", output_dir)

    labels = ["pythia_superchic", "pythia_rest", "herwig_angular", "herwig_dipole"]
    events_by_label = {}
    for label in labels:
        csv_path = output_dir / f"events_{label}.csv"
        analyze(analyze_binary, output_dir / f"{label}.hepmc", csv_path, label)
        events_by_label[label] = read_rows(csv_path)

    summary = summarize(events_by_label)
    write_csv(output_dir / "herwig_vs_pythia_summary.csv", summary)
    print_table(summary, labels)
    make_plot(events_by_label, summary, labels, output_dir)
    print(f"\nWrote summary CSV and plot to {output_dir}")


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
