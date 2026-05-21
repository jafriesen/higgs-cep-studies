#!/usr/bin/env python3
import argparse
import csv
import math
import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Timing <-> z conversion
C_CM_PER_PS = 2.99792458e-2
M_H_GEV = 125.0

# HL-LHC clocks used in the analyzer
F_BX_PEAK_HZ = 40.0e6
F_COLL_AVG_HZ = 31.6e6


def parse_timing_grid(text):
    vals = []
    for tok in text.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError as exc:
            raise ValueError(f"Could not parse timing token '{t}'.") from exc
        if v <= 0.0:
            raise ValueError(f"Timing resolution must be > 0, got {v}.")
        if v not in vals:
            vals.append(v)

    if not vals:
        raise ValueError("timing-res-grid is empty.")
    return vals


def parse_optional_scalar(text, name, allow_zero=True):
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None
    if s.lower() in {"none", "all", "inf", "-"}:
        return None

    try:
        v = float(s)
    except ValueError as exc:
        raise ValueError(f"Could not parse --{name}='{text}'.") from exc

    if allow_zero:
        if v < 0.0:
            raise ValueError(f"--{name} must be >= 0 or 'none', got {v}.")
    else:
        if v <= 0.0:
            raise ValueError(f"--{name} must be > 0, got {v}.")
    return v


def as_bool_int(arr):
    return np.asarray(arr, dtype=np.int32) != 0


def load_pair_arrays(root_file, tree_name):
    df = ROOT.RDataFrame(tree_name, root_file)
    cols = {str(c) for c in df.GetColumnNames()}

    required = [
        "bx",
        "xi_L",
        "xi_R",
        "dz",
        "M",
        "tag200_L",
        "tag200_R",
        "tag400_L",
        "tag400_R",
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(
            f"Missing required branches in {root_file}: {', '.join(missing)}"
        )

    optional = [c for c in ["yX", "pt_bal"] if c in cols]
    arr = df.AsNumpy(required + optional)

    xi_L = np.asarray(arr["xi_L"], dtype=np.float64)
    xi_R = np.asarray(arr["xi_R"], dtype=np.float64)

    if "yX" in arr:
        y_pp = np.asarray(arr["yX"], dtype=np.float64)
    else:
        eps = 1e-12
        y_pp = 0.5 * np.log(np.clip(xi_L, eps, None) / np.clip(xi_R, eps, None))

    pt_bal = (
        np.asarray(arr["pt_bal"], dtype=np.float64)
        if "pt_bal" in arr
        else np.full_like(xi_L, np.nan)
    )

    return {
        "bx": np.asarray(arr["bx"], dtype=np.int64),
        "xi_L": xi_L,
        "xi_R": xi_R,
        "dz": np.asarray(arr["dz"], dtype=np.float64),
        "M": np.asarray(arr["M"], dtype=np.float64),
        "y_pp": y_pp,
        "pt_bal": pt_bal,
        "tag200_L": as_bool_int(arr["tag200_L"]),
        "tag200_R": as_bool_int(arr["tag200_R"]),
        "tag400_L": as_bool_int(arr["tag400_L"]),
        "tag400_R": as_bool_int(arr["tag400_R"]),
    }


def pair_base_mask(data, require_double420=False):
    left_tag = data["tag200_L"] | data["tag400_L"]
    right_tag = data["tag200_R"] | data["tag400_R"]
    has_420 = data["tag400_L"] | data["tag400_R"]
    mask = left_tag & right_tag & has_420
    if require_double420:
        mask = mask & data["tag400_L"] & data["tag400_R"]
    return mask


def bx_any_fraction(bx, pair_pass, n_bx_total):
    bx = np.asarray(bx, dtype=np.int64)
    pair_pass = np.asarray(pair_pass, dtype=bool)
    if bx.size == 0:
        return 0.0, 0

    order = np.argsort(bx, kind="mergesort")
    bx_s = bx[order]
    pass_s = pair_pass[order]

    _, idx = np.unique(bx_s, return_index=True)
    any_pass = np.logical_or.reduceat(pass_s, idx)
    n_pass = int(any_pass.sum())

    if n_bx_total <= 0:
        return 0.0, n_pass

    frac = n_pass / float(n_bx_total)
    return frac, n_pass


def sigma_dz_cm(single_arm_time_res_ps):
    # sigma(dt) = sqrt(2) * sigma_t(arm), and dz = (c/2) * dt
    return 0.5 * C_CM_PER_PS * math.sqrt(2.0) * single_arm_time_res_ps


def format_cut(v):
    if v is None:
        return "none"
    return f"{v:g}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan single-arm timing resolution for one fixed working point and "
            "report combinatorial trigger rates."
        )
    )
    parser.add_argument(
        "--sig-root",
        default=None,
        help="Signal ROOT file with ProtonPairs tree.",
    )
    parser.add_argument(
        "--bkg-root",
        default=None,
        help="Combinatorial background ROOT file with ProtonPairs tree.",
    )
    parser.add_argument(
        "--tree",
        default="ProtonPairs",
        help="Tree name in both files (default: ProtonPairs).",
    )

    parser.add_argument(
        "--timing-res-grid",
        default="0.5,1,2,3,4,6,7,8,9,10,15,20,30",
        help="Comma-separated single-arm timing resolutions in ps.",
    )

    # Fixed cuts (single WP)
    parser.add_argument(
        "--nsigma",
        default="2.0",
        help="Fixed timing-match Nsigma for |dz_obs| <= nsigma*sigma_dz.",
    )
    parser.add_argument(
        "--mass-halfwidth",
        default="none",
        help="Fixed half-width for |M-125| in GeV; use 'none' to disable.",
    )
    parser.add_argument(
        "--ymax",
        default="none",
        help="Fixed max |y_pp|; use 'none' to disable.",
    )
    parser.add_argument(
        "--ptbal-max",
        default="none",
        help="Fixed max pt_bal in GeV; use 'none' to disable.",
    )

    parser.add_argument(
        "--require-double420",
        action="store_true",
        help="Require both protons in the 420 m xi window.",
    )
    parser.add_argument(
        "--no-dz-smear",
        action="store_true",
        help=(
            "Disable dz smearing. By default, Gaussian dz smearing is applied "
            "to both samples."
        ),
    )
    parser.add_argument(
        "--smear-seed",
        type=int,
        default=12345,
        help="Seed for dz smearing RNG (default: 12345).",
    )
    parser.add_argument(
        "--sig-total-bx",
        type=int,
        default=None,
        help=(
            "Optional denominator for absolute signal efficiency. "
            "Default: unique signal BX in the tree."
        ),
    )
    parser.add_argument(
        "--bkg-total-bx",
        type=int,
        default=None,
        help=(
            "Optional denominator for background BX fraction. "
            "Default: max(bx)+1 in background tree."
        ),
    )

    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output with one row per timing point.",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help=(
            "Output PNG path for rate31.6 vs timing. "
            "Default: analysis/output/rate31p6_vs_timing.png"
        ),
    )

    args = parser.parse_args()

    study_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_sig_root = os.path.join(
        os.environ.get("HIGGS_ANALYSIS_DIR", os.path.join(study_dir, "analysis")),
        "output",
        "hbb_001_pairs.root",
    )
    default_bkg_root = os.path.join(
        os.environ.get("HIGGS_BKG_DIR", os.path.join(study_dir, "bkg-generation")),
        "output",
        "minbias_mu200",
        "minbias_mu200_pairs.root",
    )

    if args.sig_root is None:
        args.sig_root = default_sig_root
    if args.bkg_root is None:
        args.bkg_root = default_bkg_root

    if not os.path.exists(args.sig_root):
        raise FileNotFoundError(f"Signal ROOT not found: {args.sig_root}")
    if not os.path.exists(args.bkg_root):
        raise FileNotFoundError(f"Background ROOT not found: {args.bkg_root}")

    nsigma = parse_optional_scalar(args.nsigma, "nsigma", allow_zero=False)
    mass_halfwidth = parse_optional_scalar(args.mass_halfwidth, "mass-halfwidth")
    ymax = parse_optional_scalar(args.ymax, "ymax")
    ptbal_max = parse_optional_scalar(args.ptbal_max, "ptbal-max")
    timing_res_vals = parse_timing_grid(args.timing_res_grid)

    sig = load_pair_arrays(args.sig_root, args.tree)
    bkg = load_pair_arrays(args.bkg_root, args.tree)

    sig_bx_total = (
        int(args.sig_total_bx)
        if args.sig_total_bx is not None
        else int(np.unique(sig["bx"]).size)
    )
    bkg_bx_total = (
        int(args.bkg_total_bx)
        if args.bkg_total_bx is not None
        else int(np.max(bkg["bx"])) + 1
    )

    base_sig = pair_base_mask(sig, require_double420=args.require_double420)
    base_bkg = pair_base_mask(bkg, require_double420=args.require_double420)

    sig_base_frac, sig_base_pass = bx_any_fraction(sig["bx"], base_sig, sig_bx_total)
    bkg_base_frac, bkg_base_pass = bx_any_fraction(bkg["bx"], base_bkg, bkg_bx_total)

    do_smear = not args.no_dz_smear

    has_sig_pt = np.isfinite(sig["pt_bal"]).any()
    has_bkg_pt = np.isfinite(bkg["pt_bal"]).any()
    use_pt = has_sig_pt and has_bkg_pt and (ptbal_max is not None)

    if ptbal_max is not None and not use_pt:
        print("Warning: pt_bal cut requested but pt_bal branch unavailable; disabling pt_bal cut.")

    print("=== Inputs ===")
    print(f"Signal file:         {args.sig_root}")
    print(f"Background file:     {args.bkg_root}")
    print(f"Tree:                {args.tree}")
    print(f"Signal entries:      {sig['bx'].size}")
    print(f"Background entries:  {bkg['bx'].size}")
    print(f"Signal BX denom:     {sig_bx_total}")
    print(f"Background BX denom: {bkg_bx_total}")
    print(f"Require double420:   {args.require_double420}")
    print(
        "Fixed cuts: "
        f"nsigma={nsigma:g}, mass-halfwidth={format_cut(mass_halfwidth)}, "
        f"ymax={format_cut(ymax)}, ptbal-max={format_cut(ptbal_max)}"
    )
    print(
        f"dz smearing: {do_smear} "
        f"(single-arm sigma_t grid in ps: {', '.join(str(v) for v in timing_res_vals)})"
    )
    print("==============")

    print("Baseline (tag selection only):")
    print(
        f"  Signal baseline pass: {sig_base_pass}/{sig_bx_total} "
        f"(frac={sig_base_frac:.6f})"
    )
    print(
        f"  Bkg baseline pass:    {bkg_base_pass}/{bkg_bx_total} "
        f"(frac={bkg_base_frac:.6f}, rate40={bkg_base_frac*F_BX_PEAK_HZ/1e3:.1f} kHz)"
    )

    rows = []
    for i, tres_ps in enumerate(timing_res_vals):
        sigma_dz = sigma_dz_cm(tres_ps)
        zcut = nsigma * sigma_dz

        if do_smear:
            # Make each timing point deterministic and reproducible.
            rng_sig = np.random.default_rng(args.smear_seed + 1000 * i + 1)
            rng_bkg = np.random.default_rng(args.smear_seed + 1000 * i + 2)
            sig_dz_obs = sig["dz"] + rng_sig.normal(0.0, sigma_dz, size=sig["dz"].size)
            bkg_dz_obs = bkg["dz"] + rng_bkg.normal(0.0, sigma_dz, size=bkg["dz"].size)
        else:
            sig_dz_obs = sig["dz"]
            bkg_dz_obs = bkg["dz"]

        sig_mask = base_sig & (np.abs(sig_dz_obs) <= zcut)
        bkg_mask = base_bkg & (np.abs(bkg_dz_obs) <= zcut)

        if mass_halfwidth is not None:
            sig_mask = sig_mask & (np.abs(sig["M"] - M_H_GEV) <= mass_halfwidth)
            bkg_mask = bkg_mask & (np.abs(bkg["M"] - M_H_GEV) <= mass_halfwidth)

        if ymax is not None:
            sig_mask = sig_mask & (np.abs(sig["y_pp"]) <= ymax)
            bkg_mask = bkg_mask & (np.abs(bkg["y_pp"]) <= ymax)

        if ptbal_max is not None and use_pt:
            sig_mask = sig_mask & (sig["pt_bal"] <= ptbal_max)
            bkg_mask = bkg_mask & (bkg["pt_bal"] <= ptbal_max)

        sig_frac, sig_pass = bx_any_fraction(sig["bx"], sig_mask, sig_bx_total)
        bkg_frac, bkg_pass = bx_any_fraction(bkg["bx"], bkg_mask, bkg_bx_total)

        sig_eff_rel = sig_frac / sig_base_frac if sig_base_frac > 0 else 0.0
        rate40_khz = bkg_frac * F_BX_PEAK_HZ / 1.0e3
        rate31_khz = bkg_frac * F_COLL_AVG_HZ / 1.0e3

        rows.append(
            {
                "timing_res_ps": tres_ps,
                "sigma_dz_cm": sigma_dz,
                "zcut_cm": zcut,
                "sig_pass": sig_pass,
                "sig_frac": sig_frac,
                "sig_eff_rel": sig_eff_rel,
                "bkg_pass": bkg_pass,
                "bkg_frac": bkg_frac,
                "rate40_khz": rate40_khz,
                "rate31_khz": rate31_khz,
            }
        )

    print("\n=== Timing Scan Table ===")
    print(
        "timing_res_ps  sigma_dz_cm  zcut_cm    bkg_frac    rate40_khz  rate31.6_khz  sig_eff_rel"
    )
    for r in rows:
        print(
            f"{r['timing_res_ps']:>13.1f}  "
            f"{r['sigma_dz_cm']:>11.5f}  "
            f"{r['zcut_cm']:>8.5f}  "
            f"{r['bkg_pass']:>10d}  "
            f"{r['rate40_khz']:>10.2f}  "
            f"{r['rate31_khz']:>12.2f}  "
            f"{r['sig_eff_rel']:>11.5f}"
        )

    if args.csv_out:
        outdir = os.path.dirname(args.csv_out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timing_res_ps",
                    "sigma_dz_cm",
                    "zcut_cm",
                    "nsigma",
                    "mass_halfwidth",
                    "ymax",
                    "ptbal_max",
                    "sig_pass",
                    "sig_frac",
                    "sig_eff_rel",
                    "bkg_pass",
                    "bkg_frac",
                    "rate40_khz",
                    "rate31_khz",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["timing_res_ps"],
                        r["sigma_dz_cm"],
                        r["zcut_cm"],
                        nsigma,
                        "" if mass_halfwidth is None else mass_halfwidth,
                        "" if ymax is None else ymax,
                        "" if ptbal_max is None else ptbal_max,
                        r["sig_pass"],
                        r["sig_frac"],
                        r["sig_eff_rel"],
                        r["bkg_pass"],
                        r["bkg_frac"],
                        r["rate40_khz"],
                        r["rate31_khz"],
                    ]
                )
        print(f"\nWrote CSV: {args.csv_out}")

    if args.plot_out:
        plot_out = args.plot_out
    else:
        plot_out = os.path.join(
            os.environ.get("HIGGS_ANALYSIS_DIR", os.path.join(study_dir, "analysis")),
            "output",
            "rate31p6_vs_timing.png",
        )

    plot_dir = os.path.dirname(plot_out)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    x = np.array([r["timing_res_ps"] for r in rows], dtype=float)
    y = np.array([r["rate31_khz"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_xlabel("Single-arm timing resolution [ps]")
    ax.set_ylabel("Background rate at 31.6 MHz [kHz]")
    #ax.set_title("Rate31.6 vs Timing Resolution")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(plot_out, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {plot_out}")


if __name__ == "__main__":
    main()
