#!/usr/bin/env python3
"""Does underlying activity hide the FSR loss of the CEP H->bb jets?

The SuperChic H->bb events are rewritten as g g -> H -> b bbar (make_hbb_gg_lhe.py),
so the b kinematics are unchanged but Pythia can add ISR, MPI and beam remnants.
All eight ISR/MPI/Remnants combinations are showered with FSR on. For the two jets
carrying the Higgs decay products the analysis reports, per event:
  m_jj               mass of the jets from all particles (what a calibration sees)
  m_jj_higgs_only    mass of the same jets from Higgs-descended particles only
  pickup = m_jj - m_jj_higgs_only, the mass added by non-Higgs activity in the cones
"""

import argparse
import itertools
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import compare_herwig_pythia as base  # noqa: E402

EVREC_DIR = base.REPO / "output-superchic/Hbb/Hbb__v01/gen-SuperChic/evrecs"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/activity_hbb"
MH = 125.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=3, help="Number of evrec files to use")
    parser.add_argument("--events", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=31122001)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild", action="store_true", help="Recompile the C++ helper")
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_ACTIVITY_ENV") == "1":
        return
    command = "\n".join((
        f"source {shlex.quote(str(base.REPO / 'setup_env.sh'))}",
        "export HIGGS_CEP_ACTIVITY_ENV=1",
        f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
    ))
    raise SystemExit(subprocess.run(["bash", "-lc", command], cwd=base.REPO).returncode)


def summarize(events_path, label, written, failed):
    import pandas as pd

    rows = []
    frame = pd.read_csv(events_path) if written else None
    for radius in base.RADII:
        row = {"label": label, "radius": radius, "written": written, "failed": failed}
        if frame is not None:
            d = frame[(frame.radius - radius).abs() < 1e-6]
            row.update({
                "median_m_jj_over_mh": d.m_jj.median() / MH,
                "median_m_jj_higgs_only_over_mh": d.m_jj_higgs_only.median() / MH,
                "median_pickup_over_mh": (d.m_jj - d.m_jj_higgs_only).median() / MH,
                "mean_pickup_over_mh": (d.m_jj - d.m_jj_higgs_only).mean() / MH,
                "median_e_jj_other": d.e_jj_other.median(),
                "median_m_leading_over_mh": d.m_leading.median() / MH,
                "median_pt_higgs": d.pt_higgs.median(),
                "mean_n_mpi": d.n_mpi.mean(),
                "min_m_higgs_system": d.m_higgs_system.min(),
                "mean_e_mixed": d.e_mixed.mean(),
            })
        rows.append(row)
    return rows


def make_plot(frames, output_dir):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bins = np.linspace(0.3, 1.3, 81)
    for label, frame in frames.items():
        d = frame[(frame.radius - 0.4).abs() < 1e-6]
        axes[0].hist(d.m_jj / MH, bins=bins, density=True, histtype="step", linewidth=1.3,
                     label=label)
        axes[1].hist((d.m_jj - d.m_jj_higgs_only) / MH, bins=np.linspace(-0.02, 0.2, 56),
                     density=True, histtype="step", linewidth=1.3, label=label)
        medians = [frame[(frame.radius - r).abs() < 1e-6].m_jj.median() / MH for r in base.RADII]
        axes[2].plot(base.RADII, medians, marker="o", label=label)
    axes[0].axvline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("m(jj)/125, Higgs jets, anti-kt R=0.4, all particles")
    axes[0].set_ylabel("Normalized events")
    axes[1].set_xlabel("pickup: [m(jj) - m(jj, Higgs particles only)]/125, R=0.4")
    axes[1].set_yscale("log")
    axes[2].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("Jet radius R")
    axes[2].set_ylabel("median m(jj)/125, all particles")
    for axis in axes:
        axis.legend(frameon=False, fontsize=7)
    figure.suptitle("SuperChic H->bb kinematics as gg->H, Pythia 8 FSR on: ISR/MPI/Remnants")
    figure.tight_layout()
    figure.savefig(output_dir / "activity_hbb.png", dpi=160)
    plt.close(figure)


def main():
    import pandas as pd

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flags = base.config_flags("pythia8-config", "--cxxflags")
    flags += base.config_flags("fastjet-config", "--cxxflags")
    flags += base.config_flags("pythia8-config", "--libs")
    flags += base.config_flags("fastjet-config", "--libs")
    binary = base.compile_helper(SCRIPT_DIR / "shower_activity_hbb.cc", args.rebuild, flags)

    lhe = output_dir / "hbb_gg.lhe"
    sources = [EVREC_DIR / f"evrecHbb__v01_{i}.dat" for i in range(1, args.files + 1)]
    base.run([sys.executable, str(SCRIPT_DIR / "make_hbb_gg_lhe.py"), *map(str, sources),
              str(lhe), "--max-events", str(args.events)])

    jobs = {}
    for isr, mpi, remnants in itertools.product(("off", "on"), repeat=3):
        label = f"isr_{isr}_mpi_{mpi}_rem_{remnants}"
        command = [str(binary), "--lhe", str(lhe), "--output",
                   str(output_dir / f"events_{label}.csv"), "--label", label,
                   "--events", str(args.events), "--seed", str(args.seed),
                   "--isr", isr, "--mpi", mpi, "--remnants", remnants]
        print("Running:", shlex.join(command))
        jobs[label] = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL)

    summary, frames = [], {}
    for label, job in jobs.items():
        stdout, _ = job.communicate()
        if job.returncode != 0:
            raise RuntimeError(f"{label} exited with {job.returncode}")
        line = next(l for l in stdout.splitlines() if l.startswith("Pythia wrote"))
        written = int(line.split()[2])
        failed = int(line.split("(")[1].split()[0])
        events_path = output_dir / f"events_{label}.csv"
        summary += summarize(events_path, label, written, failed)
        # Configurations where Pythia fails almost every event are not usable.
        if written >= 0.9 * args.events:
            frames[label] = pd.read_csv(events_path)

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "activity_hbb_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(table[table.radius == 0.4].to_string(index=False))
    print("\nmedian m(jj)/125 (all particles) vs R")
    print(table.pivot(index="label", columns="radius", values="median_m_jj_over_mh"))
    print("\nmedian pickup/125 vs R")
    print(table.pivot(index="label", columns="radius", values="median_pickup_over_mh"))
    make_plot(frames, output_dir)
    print(f"\nWrote summary CSV and plot to {output_dir}")


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
