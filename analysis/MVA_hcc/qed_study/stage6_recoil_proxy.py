#!/usr/bin/env python3
"""Stage 6: can the proton-pT discrimination be reached without the protons?

Exclusivity forces the central system to balance the outgoing proton pair, so
the dijet transverse momentum is in principle the same observable measured on
the central side. This stage checks how much of the separation survives that
route, and what resolution on the central-system pT would be required.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA_hcc.qed_study.common import (  # noqa: E402
    DEFAULT_DATA, DEFAULT_OUTPUT, component_rows, conditional_separation, load_dataset,
    separation, write_yaml,
)

COMPONENTS = ("Hcc", "QCDcc_superchic", "QEDcc_superchic")
BACKGROUNDS = ("QEDcc_superchic", "QCDcc_superchic")
PROXIES = ("dijet_pt", "puppi_met", "fsr_recovered_pt")
RESOLUTION_GEV = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA / "exclusive_wide"))
    parser.add_argument("--stage2-dir", default=str(DEFAULT_OUTPUT / "stage2"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage6"))
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def central_system_pt(arrays, name):
    """|p_T1 + p_T2| of the outgoing protons: the recoil the central system carries."""
    px = (arrays[f"{name}_pt_left"] * np.cos(arrays[f"{name}_phi_left"])
          + arrays[f"{name}_pt_right"] * np.cos(arrays[f"{name}_phi_right"]))
    py = (arrays[f"{name}_pt_left"] * np.sin(arrays[f"{name}_phi_left"])
          + arrays[f"{name}_pt_right"] * np.sin(arrays[f"{name}_phi_right"]))
    return np.hypot(px, py), px, py


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(Path(args.stage2_dir) / "stage2_arrays.npz")
    data = load_dataset(args.data_dir, mmap=False)
    features = data["features"]
    matrix = np.asarray(data["x"])
    mx = np.asarray(data["mx"])

    rows = {name: component_rows(data, name) for name in COMPONENTS}
    truth, weight, angle = {}, {}, {}
    components = {}
    for name in COMPONENTS:
        # The two blocks come from the same selection with the same seeds, so they
        # are row-aligned; mx is checked rather than assumed.
        if not np.allclose(arrays[f"{name}_mx"], mx[rows[name]], rtol=1e-9):
            raise RuntimeError(f"{name}: the truth block and the feature rows are not aligned")
        magnitude, px, py = central_system_pt(arrays, name)
        truth[name] = magnitude
        components[name] = (px, py)
        weight[name] = arrays[f"{name}_weight"]
        angle[name] = matrix[rows[name], features.index("delta_eta_jj")].astype(float)

    report = {"mean_central_system_pt_gev": {n: float(truth[n].mean()) for n in COMPONENTS},
              "median_central_system_pt_gev": {n: float(np.median(truth[n])) for n in COMPONENTS}}
    print("central-system pT implied by the protons [GeV]:")
    for name in COMPONENTS:
        print(f"  {name:18s} mean {truth[name].mean():.4f}  median {np.median(truth[name]):.4f}")

    report["separation_truth_central_pt"] = {
        name: separation(truth["Hcc"], weight["Hcc"], truth[name], weight[name])
        for name in BACKGROUNDS
    }
    print("\nseparation from H(cc):")
    for name in BACKGROUNDS:
        print(f"  truth central-system pT vs {name:18s} "
              f"{report['separation_truth_central_pt'][name]:.4f}")

    # what the central detector actually delivers
    report["reconstructed_proxies"] = {}
    for column in PROXIES:
        if column not in features:
            continue
        index = features.index(column)
        values = {name: matrix[rows[name], index].astype(float) for name in COMPONENTS}
        entry = {
            "median_signal_gev": float(np.median(values["Hcc"])),
            "correlation_with_truth_central_pt":
                float(np.corrcoef(values["Hcc"], truth["Hcc"])[0, 1]),
            "separation": {
                name: separation(values["Hcc"], weight["Hcc"], values[name], weight[name])
                for name in BACKGROUNDS
            },
            "separation_conditioned_on_angle": {
                name: conditional_separation(
                    values["Hcc"], weight["Hcc"], values[name], weight[name],
                    angle["Hcc"], angle[name],
                )
                for name in BACKGROUNDS
            },
        }
        report["reconstructed_proxies"][column] = entry
        print(f"\n  {column}: median {entry['median_signal_gev']:.2f} GeV, "
              f"correlation with the truth recoil {entry['correlation_with_truth_central_pt']:+.4f}")
        for name in BACKGROUNDS:
            print(f"    vs {name:18s} separation {entry['separation'][name]:.4f}, "
                  f"conditioned on the angle {entry['separation_conditioned_on_angle'][name]:.4f}")

    # how well the central system would have to be measured
    rng = np.random.default_rng(args.seed)
    scan = {}
    for sigma in RESOLUTION_GEV:
        smeared = {}
        for name in COMPONENTS:
            px, py = components[name]
            if sigma > 0.0:
                px = px + rng.normal(0.0, sigma, px.size)
                py = py + rng.normal(0.0, sigma, py.size)
            smeared[name] = np.hypot(px, py)
        scan[sigma] = {
            name: separation(smeared["Hcc"], weight["Hcc"], smeared[name], weight[name])
            for name in BACKGROUNDS
        }
    report["central_pt_resolution_scan"] = scan
    print("\nseparation against a resolution on each central-system pT component:")
    print(f"  {'sigma [GeV]':>12s} {'vs QEDcc':>10s} {'vs QCDcc_SC':>12s}")
    for sigma in RESOLUTION_GEV:
        print(f"  {sigma:12.1f} {scan[sigma]['QEDcc_superchic']:10.4f} "
              f"{scan[sigma]['QCDcc_superchic']:12.4f}")

    report["angle_separation_for_reference"] = {
        name: separation(angle["Hcc"], weight["Hcc"], angle[name], weight[name])
        for name in BACKGROUNDS
    }
    report["runtime_seconds"] = time.perf_counter() - started
    report["note"] = (
        "The recoil is a central-detector observable in principle. The scan gives the "
        "resolution per transverse component of the central-system momentum that would "
        "be needed; jet energy resolution on a 125 GeV system is far coarser."
    )
    write_yaml(output_dir / "stage6_report.yaml", report)
    np.savez_compressed(
        output_dir / "stage6_arrays.npz",
        **{f"{name}_truth_central_pt": truth[name] for name in COMPONENTS},
        **{f"{name}_weight": weight[name] for name in COMPONENTS},
        **{f"{name}_dijet_pt": matrix[rows[name], features.index("dijet_pt")].astype(float)
           for name in COMPONENTS},
    )
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
