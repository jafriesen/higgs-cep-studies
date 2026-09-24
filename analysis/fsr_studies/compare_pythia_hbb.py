#!/usr/bin/env python3
"""Generate a standalone Pythia H->bb control sample and compare its FSR loss."""

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR_SOURCE = SCRIPT_DIR / "generate_pythia_hbb.cc"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/pythia_control"
DEFAULT_SUPERCHIC_SUMMARY = SCRIPT_DIR / "output/summary.csv"
RADII = (0.4, 0.6, 0.8, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate color-singlet Higgs bosons at rest with Pythia, force H->bb, "
            "compare resonance FSR on/off, and overlay the result with analyze_fsr.py."
        )
    )
    parser.add_argument("--events", type=int, default=2000, help="Accepted events per FSR mode")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--higgs-mass", type=float, default=125.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--superchic-summary", type=Path, default=DEFAULT_SUPERCHIC_SUMMARY)
    parser.add_argument("--rebuild", action="store_true", help="Recompile the C++ generator")
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_PYTHIA_CONTROL_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_PYTHIA_CONTROL_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def config_flags(program, option):
    completed = subprocess.run(
        [program, option], check=True, text=True, stdout=subprocess.PIPE
    )
    return shlex.split(completed.stdout.strip())


def compile_generator(rebuild):
    binary = Path(tempfile.gettempdir()) / f"higgs_cep_generate_pythia_hbb_{os.getuid()}"
    if not rebuild and binary.is_file() and binary.stat().st_mtime >= GENERATOR_SOURCE.stat().st_mtime:
        return binary
    command = (
        ["g++", str(GENERATOR_SOURCE), "-o", str(binary)]
        + config_flags("pythia8-config", "--cxxflags")
        + config_flags("fastjet-config", "--cxxflags")
        + config_flags("pythia8-config", "--libs")
        + config_flags("fastjet-config", "--libs")
    )
    print("Compiling:", shlex.join(command))
    subprocess.run(command, cwd=REPO, check=True)
    return binary


def run_generator(binary, args, event_csv):
    command = [
        str(binary),
        "--events", str(args.events),
        "--seed", str(args.seed),
        "--higgs-mass", str(args.higgs_mass),
        "--output", str(event_csv),
    ]
    print("Running:", shlex.join(command))
    subprocess.run(command, cwd=REPO, check=True)


def read_numeric_csv(path, string_fields):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: value if key in string_fields else float(value)
                    for key, value in row.items()
                }
            )
    return rows


def finite(values):
    return [value for value in values if math.isfinite(value)]


def mean(values):
    selected = finite(values)
    return sum(selected) / len(selected) if selected else math.nan


def summarize(events):
    summary = []
    for mode in ("noFSR", "FSR"):
        for radius in RADII:
            rows = [
                row for row in events
                if row["mode"] == mode and math.isclose(row["radius"], radius)
            ]
            if not rows:
                raise RuntimeError(f"No generated rows for {mode}, R={radius}")
            summary.append(
                {
                    "mode": mode,
                    "radius": radius,
                    "events": len(rows),
                    "hard_bb_ratio": mean([row["m_bb"] / row["m_h"] for row in rows]),
                    "visible_ratio": mean([row["m_visible"] / row["m_h"] for row in rows]),
                    "all_jets_ratio": mean([row["m_all_jets"] / row["m_h"] for row in rows]),
                    "leading_ratio": mean([row["m_leading"] / row["m_h"] for row in rows]),
                    "bmatched_ratio": mean([row["m_bmatched"] / row["m_h"] for row in rows]),
                    "out_of_cone_fraction": mean([row["outside_fraction"] for row in rows]),
                    "leading_mismatch_fraction": mean([row["leading_not_bmatched"] for row in rows]),
                    "mean_bottom_hadrons": mean([row["n_bottom_hadrons"] for row in rows]),
                }
            )
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_superchic_summary(path):
    if not path.is_file():
        print(f"WARNING: SuperChic summary not found; skipping overlay: {path}")
        return []
    return read_numeric_csv(path, {"dataset", "first_substantial_drop", "attribution"})


def row_for(rows, mode, radius, mode_field):
    return next(
        row for row in rows
        if row[mode_field] == mode and math.isclose(row["radius"], radius)
    )


def comparison_rows(pythia_summary, superchic_summary):
    if not superchic_summary:
        return []
    result = []
    for radius in RADII:
        pythia_no = row_for(pythia_summary, "noFSR", radius, "mode")["bmatched_ratio"]
        pythia_yes = row_for(pythia_summary, "FSR", radius, "mode")["bmatched_ratio"]
        superchic_no = row_for(superchic_summary, "noFSR", radius, "dataset")["bmatched_ratio"]
        superchic_yes = row_for(superchic_summary, "FSR", radius, "dataset")["bmatched_ratio"]
        pythia_shift = pythia_yes - pythia_no
        superchic_shift = superchic_yes - superchic_no
        result.append(
            {
                "radius": radius,
                "pythia_noFSR": pythia_no,
                "pythia_FSR": pythia_yes,
                "pythia_FSR_shift": pythia_shift,
                "superchic_noFSR": superchic_no,
                "superchic_FSR": superchic_yes,
                "superchic_FSR_shift": superchic_shift,
                "shift_difference": pythia_shift - superchic_shift,
                "same_direction": int(pythia_shift < 0.0 and superchic_shift < 0.0),
            }
        )
    return result


def print_tables(summary, comparison):
    print("\nStandalone Pythia H->bb control")
    print(f"{'mode':<6} {'R':>3} {'visible':>8} {'bmatch':>8} {'out cone':>9} {'mismatch':>9}")
    for row in summary:
        print(
            f"{row['mode']:<6} {row['radius']:>3.1f} {row['visible_ratio']:>8.3f} "
            f"{row['bmatched_ratio']:>8.3f} {row['out_of_cone_fraction']:>9.3f} "
            f"{row['leading_mismatch_fraction']:>9.3f}"
        )
    if comparison:
        print("\nFSR shift in mean b-matched m/mH (FSR minus noFSR)")
        print(f"{'R':>3} {'Pythia':>9} {'SuperChic':>10} {'difference':>11}")
        for row in comparison:
            print(
                f"{row['radius']:>3.1f} {row['pythia_FSR_shift']:>9.3f} "
                f"{row['superchic_FSR_shift']:>10.3f} {row['shift_difference']:>11.3f}"
            )
        same_direction = all(row["same_direction"] for row in comparison)
        pythia_recovers = all(
            row_for(summary, "FSR", later, "mode")["bmatched_ratio"]
            >= row_for(summary, "FSR", earlier, "mode")["bmatched_ratio"]
            for earlier, later in zip(RADII, RADII[1:])
        )
        superchic_values = [row["superchic_FSR"] for row in comparison]
        superchic_recovers = all(b >= a for a, b in zip(superchic_values, superchic_values[1:]))
        print(
            "Qualitative agreement: "
            f"FSR lowers the mass in both={'yes' if same_direction else 'no'}, "
            f"larger R recovers mass in both={'yes' if pythia_recovers and superchic_recovers else 'no'}"
        )


def make_plots(np, plt, events, summary, superchic_summary, output_dir):
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    bins = np.linspace(0.0, 160.0, 65)
    for axis, radius in zip(axes.flat, RADII):
        for mode in ("noFSR", "FSR"):
            values = [
                row["m_bmatched"] for row in events
                if row["mode"] == mode and math.isclose(row["radius"], radius)
                and math.isfinite(row["m_bmatched"])
            ]
            axis.hist(
                values, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=colors[mode], label=mode,
            )
        axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"anti-kt R={radius:.1f}")
        axis.set_xlabel("b-matched dijet mass [GeV]")
        axis.set_ylabel("Normalized events")
        axis.legend(frameon=False)
    figure.suptitle("Standalone Pythia H->bb resonance gun")
    figure.tight_layout()
    figure.savefig(output_dir / "pythia_hbb_dijet_mass.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode, marker in (("noFSR", "o"), ("FSR", "s")):
        values = [row_for(summary, mode, radius, "mode")["bmatched_ratio"] for radius in RADII]
        axes[0].plot(RADII, values, marker=marker, color=colors[mode], label=f"Pythia {mode}")
    if superchic_summary:
        for mode, marker in (("noFSR", "o"), ("FSR", "s")):
            values = [
                row_for(superchic_summary, mode, radius, "dataset")["bmatched_ratio"]
                for radius in RADII
            ]
            axes[0].plot(
                RADII, values, marker=marker, linestyle="--", color=colors[mode],
                label=f"SuperChic+Pythia {mode}",
            )
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Jet radius R")
    axes[0].set_ylabel("Mean b-matched m / mH")
    axes[0].legend(frameon=False, fontsize=9)

    pythia_shift = [
        row_for(summary, "FSR", radius, "mode")["bmatched_ratio"]
        - row_for(summary, "noFSR", radius, "mode")["bmatched_ratio"]
        for radius in RADII
    ]
    axes[1].plot(RADII, pythia_shift, marker="o", label="Standalone Pythia")
    if superchic_summary:
        superchic_shift = [
            row_for(superchic_summary, "FSR", radius, "dataset")["bmatched_ratio"]
            - row_for(superchic_summary, "noFSR", radius, "dataset")["bmatched_ratio"]
            for radius in RADII
        ]
        axes[1].plot(RADII, superchic_shift, marker="s", linestyle="--", label="SuperChic+Pythia")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Jet radius R")
    axes[1].set_ylabel("FSR shift in mean b-matched m / mH")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "pythia_vs_superchic_radius.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.events <= 0:
        raise RuntimeError("--events must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_csv = output_dir / "pythia_hbb_events.csv"

    binary = compile_generator(args.rebuild)
    run_generator(binary, args, event_csv)
    events = read_numeric_csv(event_csv, {"mode"})
    summary = summarize(events)
    write_csv(output_dir / "pythia_hbb_summary.csv", summary)

    superchic_summary = load_superchic_summary(args.superchic_summary.resolve())
    comparison = comparison_rows(summary, superchic_summary)
    if comparison:
        write_csv(output_dir / "pythia_vs_superchic.csv", comparison)
    print_tables(summary, comparison)

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    make_plots(np, plt, events, summary, superchic_summary, output_dir)
    print(f"\nWrote control-sample CSVs and plots to {output_dir}")


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
