#!/usr/bin/env python3
"""Closure of the FSR-recovered GenJet-target corrections on independent samples.

Applies maps[FSR][bottom|charm] from the production corrections file to files that
were not used in the derivation, and reports the median corrected response
(reco / recovered GenJet) overall and in bins of GenJet pT and |eta|. An overall
median of 1.00 can hide pT- or eta-dependent non-closure, so the binned view is the
real test. Bins are in the *target* pT, which avoids the bias of binning in the
quantity being corrected.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jet-energy"))

from common.jet_calibration import correction_factors, load_correction_map  # noqa: E402
import derive_parton_corrections as derive  # noqa: E402

PT_BINS = ((12, 22), (22, 33), (33, 50), (50, 90))
ETA_BINS = ((0.0, 1.5), (1.5, 2.5), (2.5, 3.0))
FLAG = 0.03  # report bins more than 3% from unity


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", default="jet-energy/output/fsr_mtd_recovered/corrections.yaml")
    parser.add_argument("--max-files", type=int, default=30, help="held-out files per sample")
    parser.add_argument("--only", default=None, help="only check samples whose label contains this")
    return parser.parse_args()


def held_out(pattern, used):
    """Files after the first `used` in lexical order, which is how the derivation took them."""
    files = sorted(ROOT.glob(pattern), key=lambda path: str(path))
    return files[used:]


# (label, map key, truth PID, glob, files consumed by the derivation). Each SuperChic
# sample is checked against its own map. Each MadGraph slice is checked twice: against
# the component's combined map (all slices) and against its own per-slice map.
SC = "output-superchic/{p}/{p}__v01/sim-Delphes/{p}_DPy8_FSR_200PU_MTD__v01/root/*.root"
MG = "output-madgraph/{p}/{p}__v{v}/sim-Delphes/{p}_DPy8_FSR_200PU_MTD__v{v}/root/*.root"
SAMPLES = [
    ("Hbb signal", "Hbb", 5, SC.format(p="Hbb"), 100),
    ("QCDbb exclusive", "QCDbb_superchic", 5, SC.format(p="QCDbb"), 100),
    ("QEDbb exclusive", "QEDbb", 5, SC.format(p="QEDbb"), 100),
    ("Hcc signal", "Hcc", 4, SC.format(p="Hcc"), 100),
    ("QCDcc exclusive", "QCDcc_superchic", 4, SC.format(p="QCDcc"), 100),
    ("QEDcc exclusive", "QEDcc", 4, SC.format(p="QEDcc"), 100),
    (
        "QCDgg exclusive",
        "QCDgg",
        21,
        "output-superchic/QCDgg/QCDgg__v03/sim-Delphes/"
        "QCDgg_DPy8_FSR_200PU_MTD__v03/root/*.root",
        100,
    ),
]
for process, pid, versions in (("QCDbb", 5, ("04", "05", "06", "07", "08", "09")),
                               ("QCDcc", 4, ("02", "03", "04", "05", "06", "07"))):
    for version in versions:
        pattern = MG.format(p=process, v=version)
        SAMPLES.append((f"{process} MG v{version} combined", f"{process}_madgraph", pid, pattern, 40))
        SAMPLES.append((f"{process} MG v{version} slice", f"{process}_madgraph_v{version}", pid, pattern, 40))


def median_or_nan(values):
    return float(np.median(values)) if values.size >= 50 else float("nan")


def main():
    args = parse_args()
    samples = [row for row in SAMPLES if args.only is None or args.only in row[0]]
    keys = sorted({key for _label, key, *_rest in samples})
    maps = {key: load_correction_map(ROOT / args.corrections, "FSR", key) for key in keys}
    pt_header = " ".join(f"{f'pT {lo}-{hi}':>10s}" for lo, hi in PT_BINS)
    eta_header = " ".join(f"{f'|eta| {lo}-{hi}':>13s}" for lo, hi in ETA_BINS)
    print(f"{'sample':26s} {'map':20s} {'jets':>7s} {'raw':>6s} {'corr':>6s} {'width':>6s} | {pt_header} | {eta_header}")
    flagged = []
    for label, flavour, pid, pattern, used in samples:  # flavour here is the map key
        files = held_out(pattern, used)[: args.max_files]
        if not files:
            print(f"{label:26s} no held-out files")
            continue
        pieces = [
            derive.matched_pairs(
                *derive.read_event_arrays(
                    path, "JetPUPPI", pid, None, include_partons=False
                ),
                recovery=True,
                target="genjet",
            )
            for path in files
        ]
        gen_pt, gen_eta, reco_pt = (np.concatenate([piece[i] for piece in pieces]) for i in range(3))
        factors, valid = correction_factors(reco_pt, gen_eta, maps[flavour])
        corrected = np.where(valid, reco_pt * factors, np.nan) / gen_pt
        ok = np.isfinite(corrected)
        response, gen_pt, abs_eta = corrected[ok], gen_pt[ok], np.abs(gen_eta[ok])
        q25, q50, q75 = np.percentile(response, [25, 50, 75])
        pt_values = [median_or_nan(response[(gen_pt >= lo) & (gen_pt < hi)]) for lo, hi in PT_BINS]
        eta_values = [median_or_nan(response[(abs_eta >= lo) & (abs_eta < hi)]) for lo, hi in ETA_BINS]
        for (lo, hi), value in zip(PT_BINS, pt_values):
            if np.isfinite(value) and abs(value - 1.0) > FLAG:
                flagged.append(f"{label}: pT {lo}-{hi} -> {value:.3f}")
        for (lo, hi), value in zip(ETA_BINS, eta_values):
            if np.isfinite(value) and abs(value - 1.0) > FLAG:
                flagged.append(f"{label}: |eta| {lo}-{hi} -> {value:.3f}")
        print(
            f"{label:26s} {flavour:20s} {response.size:7d} {np.median(reco_pt[ok] / gen_pt):6.3f} {q50:6.3f} "
            f"{0.5 * (q75 - q25) / q50:6.3f} | "
            + " ".join(f"{value:10.3f}" for value in pt_values)
            + " | "
            + " ".join(f"{value:13.3f}" for value in eta_values)
        )
    print(f"\nBins more than {FLAG:.0%} from unity:")
    for line in flagged or ["  none"]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
