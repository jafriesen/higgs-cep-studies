#!/usr/bin/env python3
"""Compare old vs new SuperChic QCDbb LHE samples to explain a cross-section discrepancy.

Old sample card cuts: ptamin/ptbmin=2 GeV, |eta_a/b|<3.0, mmin/mmax=90/200 GeV, |y|<2.5
New sample card cuts: ptamin/ptbmin=1 GeV, |eta_a/b|<3.5, mmin/mmax=90/160 GeV, |y|<5.0
"""
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD_DIR = "/home/jfriesen/CMSSW_15_0_0/src/higgs-cep-studies/generation-superchic/output/qcd_bb/superchic_qcd_bb_nev1000_j100_20260619_051145/evrecs"
NEW_DIR = "/home/jfriesen/higgs-cep-studies/higgs-cep-studies/output/QCDbb/QCDbb__test/SuperChic/evrecs"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

OLD_CUTS = dict(pt_min=2.0, eta_max=3.0, mass_min=90.0, mass_max=200.0, y_max=2.5)


def pt(px, py):
    return math.hypot(px, py)


def eta(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return math.copysign(float("inf"), pz)
    return 0.5 * math.log((p + pz) / (p - pz))


def mass(px, py, pz, E):
    m2 = E * E - px * px - py * py - pz * pz
    return math.sqrt(m2) if m2 > 0.0 else 0.0


def rapidity(pz, E):
    return 0.5 * math.log((E + pz) / (E - pz))


def iter_events(path):
    """Yield (b_4vec, bbar_4vec) for each event, and the file's xsec_pb (from <init>).

    Events in these files are unweighted (XWGTUP == 1 for all of them); the
    actual cross section for the file is given on the second line of <init>.
    """
    in_event = False
    expect_header = False
    in_init = False
    init_line_count = 0
    xsec_pb = None
    b = None
    bbar = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("<init>"):
                in_init, init_line_count = True, 0
                continue
            if line.startswith("</init>"):
                in_init = False
                continue
            if in_init:
                init_line_count += 1
                if init_line_count == 2:
                    xsec_pb = float(line.split()[0])
                continue

            if line.startswith("<event"):
                in_event, expect_header = True, True
                b, bbar = None, None
                continue

            if line.startswith("</event"):
                if b is not None and bbar is not None:
                    yield b, bbar, xsec_pb
                in_event = False
                continue

            if not in_event:
                continue

            if expect_header:
                expect_header = False
                continue

            parts = line.split()
            if len(parts) < 11:
                continue

            pid, status = int(parts[0]), int(parts[1])
            if status != 1:
                continue
            px, py, pz, E = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            if pid == 5:
                b = (px, py, pz, E)
            elif pid == -5:
                bbar = (px, py, pz, E)


def load_sample(directory):
    """Combine many independent SuperChic jobs into one weighted event list.

    Each job (file) is its own independent integration over the *same* phase
    space, with its own statistical estimate of the total cross section. So
    a per-event weight of (job_xsec / nev_in_job) over-counts when combined
    across N jobs -- it must also be divided by the number of jobs so that
    summing weights over the combined sample reproduces the average of the
    per-job cross-section estimates, not their sum.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.dat")))
    rows = []
    for path in paths:
        file_rows = []
        xsec_pb = None
        for b, bbar, xsec_pb in iter_events(path):
            b_pt, bbar_pt = pt(b[0], b[1]), pt(bbar[0], bbar[1])
            b_eta, bbar_eta = eta(*b[:3]), eta(*bbar[:3])
            bb = tuple(b[i] + bbar[i] for i in range(4))
            file_rows.append(dict(
                b_pt=b_pt, bbar_pt=bbar_pt,
                b_eta=b_eta, bbar_eta=bbar_eta,
                bb_mass=mass(*bb), bb_y=rapidity(bb[2], bb[3]),
            ))
        if not file_rows:
            continue
        per_event_weight = xsec_pb / (len(file_rows) * len(paths))
        for row in file_rows:
            row["weight"] = per_event_weight
        rows.extend(file_rows)
    return rows


def passes(row, cuts, which=("pt", "eta", "mass", "y")):
    ok = True
    if "pt" in which:
        ok = ok and row["b_pt"] > cuts["pt_min"] and row["bbar_pt"] > cuts["pt_min"]
    if "eta" in which:
        ok = ok and abs(row["b_eta"]) < cuts["eta_max"] and abs(row["bbar_eta"]) < cuts["eta_max"]
    if "mass" in which:
        ok = ok and cuts["mass_min"] < row["bb_mass"] < cuts["mass_max"]
    if "y" in which:
        ok = ok and abs(row["bb_y"]) < cuts["y_max"]
    return ok


def xsec(rows, cuts=None, which=("pt", "eta", "mass", "y")):
    if cuts is None:
        return sum(r["weight"] for r in rows)
    return sum(r["weight"] for r in rows if passes(r, cuts, which))


def make_plot(rows, value_key, xlabel, lines, fname, bins=50, xrange=None):
    values = [r[value_key] for r in rows]
    weights = [r["weight"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.hist(values, bins=bins, range=xrange, weights=weights, histtype="step", color="#0072B2")
    for x, label in lines:
        if label is None:
            ax.axvline(x, color="#D55E00", linestyle="--")
        else:
            ax.axvline(x, color="#D55E00", linestyle="--", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cross section / bin [pb]")
    ax.set_title("New QCDbb sample (loosened cuts) vs. old card cut boundaries")
    if lines:
        ax.legend()
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, fname)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    print(f"loading old sample from {OLD_DIR} ...")
    old_rows = load_sample(OLD_DIR)
    print(f"loading new sample from {NEW_DIR} ...")
    new_rows = load_sample(NEW_DIR)

    old_total = xsec(old_rows)
    new_total = xsec(new_rows)
    new_restricted = xsec(new_rows, OLD_CUTS)

    print()
    print(f"{'sample':35s} {'cross section [pb]':>20s}")
    print(f"{'old (as generated)':35s} {old_total:20.5f}")
    print(f"{'new (as generated)':35s} {new_total:20.5f}")
    print(f"{'new restricted to old cuts':35s} {new_restricted:20.5f}")
    print()
    print(f"observed ratio new/old:                 {new_total / old_total:8.2f}x")
    print(f"new-restricted / old:                    {new_restricted / old_total:8.2f}x")
    print()

    print("single-cut breakdown (new sample, one cut applied at a time):")
    for which in (("pt",), ("eta",), ("mass",), ("y",)):
        x = xsec(new_rows, OLD_CUTS, which)
        print(f"  only {which[0]:5s} cut -> {x:10.5f} pb  (factor {new_total / x:6.2f}x tighter)")
    print()

    make_plot(
        [r for pair in new_rows for r in (
            dict(weight=pair["weight"], pt=pair["b_pt"]),
            dict(weight=pair["weight"], pt=pair["bbar_pt"]),
        )],
        "pt", "parton p_T [GeV]", [(OLD_CUTS["pt_min"], "old cut: p_T > 2 GeV")],
        "parton_pt.png", xrange=(0, 30),
    )
    make_plot(
        [r for pair in new_rows for r in (
            dict(weight=pair["weight"], abseta=abs(pair["b_eta"])),
            dict(weight=pair["weight"], abseta=abs(pair["bbar_eta"])),
        )],
        "abseta", "|parton eta|", [(OLD_CUTS["eta_max"], "old cut: |eta| < 3.0")],
        "parton_eta.png", xrange=(0, 3.5),
    )
    make_plot(
        new_rows, "bb_mass", "bb invariant mass [GeV]",
        [(OLD_CUTS["mass_max"], "old cut: m < 200 GeV")],
        "bb_mass.png", xrange=(80, 170),
    )
    make_plot(
        new_rows, "bb_y", "bb system rapidity y",
        [(-OLD_CUTS["y_max"], "old cut: |y| < 2.5"), (OLD_CUTS["y_max"], None)],
        "bb_rapidity.png", xrange=(-5, 5),
    )


if __name__ == "__main__":
    main()
