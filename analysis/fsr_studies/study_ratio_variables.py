#!/usr/bin/env python3
"""Proton-normalised jet ratios for FSR H(bb), from the full Pythia record.

  R_jj = m(j1 j2) / M_X
  R_j  = 2 E_T(j1) cosh(y_j1 - y_X) / M_X
  M_j  = 2 E_T(j1) cosh(y_j1 - y_X)          (R_j without the division, in GeV)

M_X = sqrt(s xi_L xi_R) and y_X = 0.5 ln(xi_R / xi_L) come from the two outgoing
protons (generator truth, no PPS smearing). E_T is the leading jet transverse mass
sqrt(pT^2 + m^2) and y its rapidity, so E_T cosh(y_j - y_X) is the jet energy in the
frame where X has no longitudinal momentum: R_j = 1 means the jet carries half of M_X.

Jets: anti-kt on all stable visible particles (neutrinos and the two outgoing protons
removed), no eta cut. Selection: the two leading jets have pT > 15 GeV and both protons
are inside the PPS xi acceptance.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import awkward as ak
import fastjet
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mva.common.protons import load_pps_config, pair_observables, passes_pps  # noqa: E402

DEFAULT_INPUT = REPO / "output-superchic/Hbb/Hbb__v01/hadr-Pythia/Hbb_FSR__v01/hepmc"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/ratio_variables"
RADII = (0.4, 0.7, 1.0)
JET_PT_MIN = 15.0
NEUTRINOS = (12, 14, 16)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def read_hepmc(path, beam_energy):
    """Visible stable particles and proton xi per event."""
    events, xi_left, xi_right = [], [], []
    current = None

    def close():
        if current is not None:
            events.append(current["particles"])
            xi_left.append(current["xi_left"])
            xi_right.append(current["xi_right"])

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("E "):
                close()
                current = {"particles": [], "xi_left": np.nan, "xi_right": np.nan,
                           "left_pz": 0.0, "right_pz": 0.0}
            elif line.startswith("P ") and current is not None:
                fields = line.split()
                if int(fields[9]) != 1:
                    continue
                pid = int(fields[3])
                px, py, pz, energy = (float(value) for value in fields[4:8])
                if pid == 2212 and abs(pz) > 0.5 * beam_energy:
                    # outgoing intact proton: keep the hardest one on each side
                    if pz < 0.0 and -pz > current["left_pz"]:
                        current["left_pz"] = -pz
                        current["xi_left"] = (beam_energy - energy) / beam_energy
                    elif pz > 0.0 and pz > current["right_pz"]:
                        current["right_pz"] = pz
                        current["xi_right"] = (beam_energy - energy) / beam_energy
                    continue
                if abs(pid) in NEUTRINOS:
                    continue
                current["particles"].append((px, py, pz, energy))
    close()
    return events, np.asarray(xi_left), np.asarray(xi_right)


def process_file(task):
    path, pps = task
    beam_energy = pps["sqrt_s"] / 2.0
    events, xi_left, xi_right = read_hepmc(path, beam_energy)
    counts = [len(particles) for particles in events]
    flat = np.array([p for particles in events for p in particles], dtype=np.float64)
    particles = ak.unflatten(
        ak.zip({"px": flat[:, 0], "py": flat[:, 1], "pz": flat[:, 2], "E": flat[:, 3]}),
        counts,
    )

    valid = np.isfinite(xi_left) & np.isfinite(xi_right) & (xi_left > 0) & (xi_right > 0)
    protons_in_pps = valid & passes_pps(xi_left, pps["xi_ranges"]) & passes_pps(
        xi_right, pps["xi_ranges"])
    mx = np.full(len(events), np.nan)
    yx = np.full(len(events), np.nan)
    mx[valid], yx[valid] = pair_observables(xi_left[valid], xi_right[valid], pps["sqrt_s"])

    out = {"mx": mx, "yx": yx, "protons_in_pps": protons_in_pps}
    for radius in RADII:
        sequence = fastjet.ClusterSequence(
            particles, fastjet.JetDefinition(fastjet.antikt_algorithm, radius))
        jets = sequence.inclusive_jets(min_pt=1.0)
        pt = np.hypot(jets.px, jets.py)
        jets = ak.pad_none(jets[ak.argsort(pt, axis=1, ascending=False)], 2)
        j1, j2 = jets[:, 0], jets[:, 1]

        def fill(values):
            return ak.to_numpy(ak.fill_none(values, np.nan)).astype(np.float64)

        px1, py1, pz1, e1 = (fill(getattr(j1, k)) for k in ("px", "py", "pz", "E"))
        px2, py2, pz2, e2 = (fill(getattr(j2, k)) for k in ("px", "py", "pz", "E"))
        pt1, pt2 = np.hypot(px1, py1), np.hypot(px2, py2)
        mjj = np.sqrt(np.maximum((e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2
                                 - (pz1 + pz2) ** 2, 0.0))
        et1 = np.sqrt(np.maximum(e1 ** 2 - pz1 ** 2, 0.0))  # transverse mass
        with np.errstate(divide="ignore", invalid="ignore"):
            y1 = 0.5 * np.log((e1 + pz1) / (e1 - pz1))
        tag = f"r{int(round(radius * 10)):02d}"
        out[f"{tag}_selected"] = protons_in_pps & (pt1 > JET_PT_MIN) & (pt2 > JET_PT_MIN)
        out[f"{tag}_R_jj"] = mjj / mx
        out[f"{tag}_M_j"] = 2.0 * et1 * np.cosh(y1 - yx)
        out[f"{tag}_R_j"] = out[f"{tag}_M_j"] / mx
    return out


def file_index(path):
    match = re.search(r"_(\d+)\.hepmc$", path.name)
    return int(match.group(1)) if match else 0


def make_plots(data, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0.4: "#0072B2", 0.7: "#D55E00", 1.0: "#009E73"}
    specs = (
        ("R_jj", r"$R_{jj} = m_{jj} / M_X$", np.linspace(0.0, 1.3, 131)),
        ("R_j", r"$R_j = 2 E_T^{j1} \cosh(y_{j1} - y_X) / M_X$", np.linspace(0.0, 1.3, 131)),
        ("M_j", r"$M_j = 2 E_T^{j1} \cosh(y_{j1} - y_X)$ [GeV]", np.linspace(0.0, 162.5, 131)),
    )
    for name, label, bins in specs:
        reference = 125.0 if name == "M_j" else 1.0
        precision = ".1f" if name == "M_j" else ".3f"
        for log in (False, True):
            figure, axis = plt.subplots(figsize=(6.4, 4.6))
            for radius in RADII:
                tag = f"r{int(round(radius * 10)):02d}"
                values = data[f"{tag}_{name}"][data[f"{tag}_selected"]]
                median = np.median(values)
                axis.hist(np.clip(values, bins[0], bins[-1] - 1e-9), bins=bins, density=True,
                          histtype="step", linewidth=1.7, color=colors[radius],
                          label=f"anti-kt R={radius:.1f} (median {median:{precision}}, N={values.size})")
            axis.axvline(reference, color="black", linestyle=":", linewidth=1.0)
            axis.set_xlabel(label)
            axis.set_ylabel("Normalized events")
            if log:
                axis.set_yscale("log")
            axis.set_title("FSR H(bb), Pythia record: 2 jets pT>15 GeV, protons in PPS")
            axis.legend(frameon=False, fontsize=8, loc="upper left")
            figure.tight_layout()
            figure.savefig(output_dir / f"{name}{'_log' if log else ''}.png", dpi=160)
            plt.close(figure)


def main():
    args = parse_args()
    pps = load_pps_config(REPO / "analysis/scripts/new/config.yaml")
    files = sorted(args.input_dir.glob("*.hepmc"), key=file_index)[: args.max_files]
    if not files:
        raise SystemExit(f"No HepMC files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files from {args.input_dir}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        pieces = list(pool.map(process_file, [(path, pps) for path in files]))
    data = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    np.savez_compressed(args.output_dir / "ratio_variables.npz", **data)

    n = data["mx"].size
    print(f"events {n}, both protons in PPS {data['protons_in_pps'].mean():.3f}")
    print(f"{'R':>4s} {'selected':>9s} {'R_jj median':>12s} {'R_jj mean':>10s} "
          f"{'R_j median':>11s} {'R_j mean':>9s} {'M_j median':>11s} {'M_j mean':>9s}")
    for radius in RADII:
        tag = f"r{int(round(radius * 10)):02d}"
        sel = data[f"{tag}_selected"]
        rjj, rj, mj = (data[f"{tag}_{k}"][sel] for k in ("R_jj", "R_j", "M_j"))
        print(f"{radius:4.1f} {sel.mean():9.3f} {np.median(rjj):12.3f} {rjj.mean():10.3f} "
              f"{np.median(rj):11.3f} {rj.mean():9.3f} {np.median(mj):11.1f} {mj.mean():9.1f}")
    make_plots(data, args.output_dir)
    print(f"Wrote plots and ratio_variables.npz to {args.output_dir}")


if __name__ == "__main__":
    main()
