#!/usr/bin/env python3
import argparse
import itertools
import math
import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

# Timing <-> z conversion
C_CM_PER_PS = 2.99792458e-2
M_H_GEV = 125.0

# HL-LHC clocks used in the analyzer
F_BX_PEAK_HZ = 40.0e6
F_COLL_AVG_HZ = 31.6e6


def parse_grid(text, name):
    vals = []
    for tok in text.split(","):
        t = tok.strip()
        if not t:
            continue
        tl = t.lower()
        if tl in {"none", "all", "inf", "-"}:
            vals.append(None)
            continue
        try:
            vals.append(float(t))
        except ValueError as exc:
            raise ValueError(f"Could not parse token '{t}' in {name}.") from exc

    if not vals:
        raise ValueError(f"Grid '{name}' is empty.")

    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


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

    out = {
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
    return out


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

    uniq_bx, idx = np.unique(bx_s, return_index=True)
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
            "Scan proton-pair cut working points and report signal efficiency "
            "vs combinatorial BX trigger rate."
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
        "--single-arm-time-res-ps",
        type=float,
        default=10.0,
        help="Single-arm timing resolution in ps (default: 10).",
    )
    parser.add_argument(
        "--nsigma-grid",
        default="2.0,1.5,1.0,0.7,0.5,0.3",
        help="Comma-separated timing match Nsigma values for |dz| cut.",
    )
    parser.add_argument(
        "--mass-halfwidth-grid",
        default="none,20,15,10,7,5,3,2",
        help="Comma-separated half-widths for |M-125| cut in GeV.",
    )
    parser.add_argument(
        "--ymax-grid",
        default="none,2.0,1.5,1.0,0.7,0.5,0.3",
        help="Comma-separated maxima for |y_pp| cut.",
    )
    parser.add_argument(
        "--ptbal-max-grid",
        default="none",
        help="Comma-separated maxima for pt_bal cut in GeV.",
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
            "Disable measurement smearing on dz. "
            "By default, Gaussian dz smearing is applied to both samples."
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
        "--target-sig-eff",
        type=float,
        default=0.85,
        help="Target relative signal efficiency for selecting best WP (default: 0.85).",
    )
    parser.add_argument(
        "--target-rate-khz",
        type=float,
        default=None,
        help="If set, also report best WP with rate@40MHz <= target (kHz).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Print top N working points sorted by lowest rate (default: 15).",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output with all scanned working points.",
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
        "minbias",
        "minbias_pairs.root",
    )

    if args.sig_root is None:
        args.sig_root = default_sig_root
    if args.bkg_root is None:
        args.bkg_root = default_bkg_root

    if not os.path.exists(args.sig_root):
        raise FileNotFoundError(f"Signal ROOT not found: {args.sig_root}")
    if not os.path.exists(args.bkg_root):
        raise FileNotFoundError(f"Background ROOT not found: {args.bkg_root}")

    nsigma_vals = parse_grid(args.nsigma_grid, "nsigma-grid")
    mass_hw_vals = parse_grid(args.mass_halfwidth_grid, "mass-halfwidth-grid")
    ymax_vals = parse_grid(args.ymax_grid, "ymax-grid")
    ptbal_vals = parse_grid(args.ptbal_max_grid, "ptbal-max-grid")

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

    sigma_dz = sigma_dz_cm(args.single_arm_time_res_ps)
    do_smear = not args.no_dz_smear
    if do_smear:
        rng_sig = np.random.default_rng(args.smear_seed)
        rng_bkg = np.random.default_rng(args.smear_seed + 1)
        sig_dz_obs = sig["dz"] + rng_sig.normal(0.0, sigma_dz, size=sig["dz"].size)
        bkg_dz_obs = bkg["dz"] + rng_bkg.normal(0.0, sigma_dz, size=bkg["dz"].size)
    else:
        sig_dz_obs = sig["dz"].copy()
        bkg_dz_obs = bkg["dz"].copy()

    has_sig_pt = np.isfinite(sig["pt_bal"]).any()
    has_bkg_pt = np.isfinite(bkg["pt_bal"]).any()
    use_pt = has_sig_pt and has_bkg_pt and any(v is not None for v in ptbal_vals)

    print("=== Inputs ===")
    print(f"Signal file:      {args.sig_root}")
    print(f"Background file:  {args.bkg_root}")
    print(f"Tree:             {args.tree}")
    print(f"Signal entries:   {sig['bx'].size}")
    print(f"Background entries:{bkg['bx'].size}")
    print(f"Signal BX denom:  {sig_bx_total}")
    print(f"Background BX denom: {bkg_bx_total}")
    print(f"Require double420: {args.require_double420}")
    print(
        f"Timing model: sigma_t(arm)={args.single_arm_time_res_ps:.2f} ps, "
        f"sigma_dz={sigma_dz:.4f} cm, smear_dz={do_smear}"
    )
    print("================")
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
    for nsig, m_hw, y_max, pt_max in itertools.product(
        nsigma_vals, mass_hw_vals, ymax_vals, ptbal_vals
    ):
        if nsig is None or nsig <= 0:
            continue

        zcut = nsig * sigma_dz
        sig_mask = base_sig & (np.abs(sig_dz_obs) <= zcut)
        bkg_mask = base_bkg & (np.abs(bkg_dz_obs) <= zcut)

        if m_hw is not None:
            sig_mask = sig_mask & (np.abs(sig["M"] - M_H_GEV) <= m_hw)
            bkg_mask = bkg_mask & (np.abs(bkg["M"] - M_H_GEV) <= m_hw)

        if y_max is not None:
            sig_mask = sig_mask & (np.abs(sig["y_pp"]) <= y_max)
            bkg_mask = bkg_mask & (np.abs(bkg["y_pp"]) <= y_max)

        if pt_max is not None and use_pt:
            sig_mask = sig_mask & (sig["pt_bal"] <= pt_max)
            bkg_mask = bkg_mask & (bkg["pt_bal"] <= pt_max)

        sig_frac, sig_pass = bx_any_fraction(sig["bx"], sig_mask, sig_bx_total)
        bkg_frac, bkg_pass = bx_any_fraction(bkg["bx"], bkg_mask, bkg_bx_total)

        sig_eff_rel = sig_frac / sig_base_frac if sig_base_frac > 0 else 0.0
        rate40_khz = bkg_frac * F_BX_PEAK_HZ / 1.0e3
        rate31_khz = bkg_frac * F_COLL_AVG_HZ / 1.0e3

        rows.append(
            {
                "nsig": nsig,
                "zcut": zcut,
                "m_hw": m_hw,
                "y_max": y_max,
                "pt_max": pt_max,
                "sig_pass": sig_pass,
                "sig_frac": sig_frac,
                "sig_eff_rel": sig_eff_rel,
                "bkg_pass": bkg_pass,
                "bkg_frac": bkg_frac,
                "rate40_khz": rate40_khz,
                "rate31_khz": rate31_khz,
            }
        )

    if not rows:
        raise RuntimeError("No working points were scanned. Check your grid options.")

    rows_sorted = sorted(rows, key=lambda r: (r["rate40_khz"], -r["sig_eff_rel"]))

    print("\n=== Best WP at target signal efficiency ===")
    candidates = [r for r in rows_sorted if r["sig_eff_rel"] >= args.target_sig_eff]
    if candidates:
        r = candidates[0]
        print(
            f"Target eff >= {args.target_sig_eff:.3f}: "
            f"nsig={r['nsig']:.3g}, zcut={r['zcut']:.4f} cm, "
            f"|M-125|<={format_cut(r['m_hw'])}, |y|<={format_cut(r['y_max'])}, "
            f"pt_bal<={format_cut(r['pt_max'])}"
        )
        print(
            f"  Signal rel eff={r['sig_eff_rel']:.4f} "
            f"(sig frac={r['sig_frac']:.6f})"
        )
        print(
            f"  Bkg frac={r['bkg_frac']:.6f} "
            f"-> rate40={r['rate40_khz']:.1f} kHz, "
            f"rate31.6={r['rate31_khz']:.1f} kHz"
        )
    else:
        print(f"No WP reached target signal efficiency {args.target_sig_eff:.3f}.")

    if args.target_rate_khz is not None:
        print("\n=== Best WP at target rate (40 MHz clock) ===")
        candidates = [r for r in rows_sorted if r["rate40_khz"] <= args.target_rate_khz]
        if candidates:
            r = max(candidates, key=lambda x: x["sig_eff_rel"])
            print(
                f"Target rate <= {args.target_rate_khz:.1f} kHz: "
                f"nsig={r['nsig']:.3g}, zcut={r['zcut']:.4f} cm, "
                f"|M-125|<={format_cut(r['m_hw'])}, |y|<={format_cut(r['y_max'])}, "
                f"pt_bal<={format_cut(r['pt_max'])}"
            )
            print(
                f"  Signal rel eff={r['sig_eff_rel']:.4f} "
                f"(sig frac={r['sig_frac']:.6f})"
            )
            print(
                f"  Bkg frac={r['bkg_frac']:.6f} "
                f"-> rate40={r['rate40_khz']:.1f} kHz, "
                f"rate31.6={r['rate31_khz']:.1f} kHz"
            )
        else:
            print(f"No WP reached target rate <= {args.target_rate_khz:.1f} kHz.")

    print(f"\n=== Top {min(args.top_n, len(rows_sorted))} WPs by lowest rate ===")
    print(
        "rank  nsig   zcut[cm]  Mhw[GeV]  ymax   ptmax[GeV]  "
        "sigEff(rel)  bkgFrac     rate40[kHz]"
    )
    for i, r in enumerate(rows_sorted[: args.top_n], start=1):
        print(
            f"{i:>4d}  {r['nsig']:>4.2f}   {r['zcut']:>8.4f}  "
            f"{format_cut(r['m_hw']):>8}  {format_cut(r['y_max']):>5}  "
            f"{format_cut(r['pt_max']):>10}  "
            f"{r['sig_eff_rel']:>10.4f}  {r['bkg_frac']:>8.6f}  "
            f"{r['rate40_khz']:>11.1f}"
        )

    if args.csv_out:
        outdir = os.path.dirname(args.csv_out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.csv_out, "w") as f:
            f.write(
                "nsig,zcut_cm,mass_halfwidth,ymax,ptbal_max,"
                "sig_pass,sig_frac,sig_eff_rel,bkg_pass,bkg_frac,rate40_khz,rate31_khz\n"
            )
            for r in rows_sorted:
                f.write(
                    f"{r['nsig']},{r['zcut']},"
                    f"{'' if r['m_hw'] is None else r['m_hw']},"
                    f"{'' if r['y_max'] is None else r['y_max']},"
                    f"{'' if r['pt_max'] is None else r['pt_max']},"
                    f"{r['sig_pass']},{r['sig_frac']},{r['sig_eff_rel']},"
                    f"{r['bkg_pass']},{r['bkg_frac']},"
                    f"{r['rate40_khz']},{r['rate31_khz']}\n"
                )
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
