#!/usr/bin/env python3
import argparse
import os

import ROOT

ROOT.gROOT.SetBatch(True)

ROOT.gInterpreter.Declare(r"""
#include <TRandom3.h>

// Smear xi with Gaussian resolution and keep physical non-negative values.
static TRandom3 gXiRand(24680);
float smear_xi(float xi, float sigma) {
    if (sigma <= 0.f) return xi;
    float out = xi + gXiRand.Gaus(0.0, sigma);
    return out < 0.f ? 0.f : out;
}
""")

XI_CUTS = {
    "420": (0.00325, 0.0116),
    "192": (0.08, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.014, 0.0263),
}


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _complement_intervals(full_lo, full_hi, allowed_intervals):
    """Return disallowed gaps inside [full_lo, full_hi]."""
    clipped = []
    for lo, hi in allowed_intervals:
        c_lo = max(full_lo, lo)
        c_hi = min(full_hi, hi)
        if c_hi > c_lo:
            clipped.append((c_lo, c_hi))

    merged = _merge_intervals(clipped)
    gaps = []
    cursor = full_lo
    for lo, hi in merged:
        if lo > cursor:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < full_hi:
        gaps.append((cursor, full_hi))
    return gaps


def _allowed_intervals(full_lo, full_hi, allowed_intervals):
    """Return merged allowed intervals clipped to [full_lo, full_hi]."""
    clipped = []
    for lo, hi in allowed_intervals:
        c_lo = max(full_lo, lo)
        c_hi = min(full_hi, hi)
        if c_hi > c_lo:
            clipped.append((c_lo, c_hi))
    return _merge_intervals(clipped)


def _draw_cut_gap_shading(xmin, xmax, ymin, ymax):
    """Draw semi-transparent shaded bands where xi cuts do not accept values."""
    allowed = list(XI_CUTS.values())
    x_allowed = _allowed_intervals(xmin, xmax, allowed)
    x_gaps = _complement_intervals(xmin, xmax, allowed)
    y_gaps = _complement_intervals(ymin, ymax, allowed)

    boxes = []

    # Vertical bands for xi1 disallowed regions.
    for xlo, xhi in x_gaps:
        bx = ROOT.TBox(xlo, ymin, xhi, ymax)
        bx.SetFillColorAlpha(ROOT.kBlack, 0.3)
        bx.SetLineColor(0)
        bx.Draw()
        boxes.append(bx)

    # Horizontal bands for xi2 disallowed regions, drawn only on xi1-allowed
    # intervals so there is no double-shaded overlap region.
    for ylo, yhi in y_gaps:
        for xlo, xhi in x_allowed:
            by = ROOT.TBox(xlo, ylo, xhi, yhi)
            by.SetFillColorAlpha(ROOT.kBlack, 0.3)
            by.SetLineColor(0)
            by.Draw()
            boxes.append(by)

    return boxes


def _draw_unshaded_boundaries(xmin, xmax, ymin, ymax):
    """Draw dotted contours around the xi regions accepted by PPS cuts."""
    allowed = list(XI_CUTS.values())
    x_allowed = _allowed_intervals(xmin, xmax, allowed)
    y_allowed = _allowed_intervals(ymin, ymax, allowed)

    lines = []
    for xlo, xhi in x_allowed:
        for ylo, yhi in y_allowed:
            lx1 = ROOT.TLine(xlo, ylo, xlo, yhi)
            lx2 = ROOT.TLine(xhi, ylo, xhi, yhi)
            ly1 = ROOT.TLine(xlo, ylo, xhi, ylo)
            ly2 = ROOT.TLine(xlo, yhi, xhi, yhi)
            for ln in (lx1, lx2, ly1, ly2):
                ln.SetLineColor(ROOT.kRed + 1)
                ln.SetLineStyle(2)
                ln.SetLineWidth(2)
                ln.Draw()
                lines.append(ln)

    return lines


def build_df(filename, xsec_pb, lumi_ab, smear_xi_sigma=0.0):
    """Build weighted RDF and define PPS-side selection booleans."""
    df0 = ROOT.RDataFrame("Events", filename)

    sumw = df0.Sum("XWGTUP").GetValue()
    if sumw == 0:
        raise RuntimeError(f"Sum of XWGTUP is zero in {filename}")

    lumi_pb = lumi_ab * 1e6  # 1 ab^-1 = 1e6 pb^-1
    global_factor = (xsec_pb * lumi_pb) / float(sumw)

    cols = [str(c) for c in df0.GetColumnNames()]
    if "w" in cols:
        df_w = df0.Redefine("w", f"{global_factor:.8e}f * XWGTUP")
    else:
        df_w = df0.Define("w", f"{global_factor:.8e}f * XWGTUP")

    # PPS station xi ranges (same as compare_sig_bkg_pps.py)
    xi_420_min, xi_420_max = XI_CUTS["420"]
    xi_192_min, xi_192_max = XI_CUTS["192"]
    xi_213_min, xi_213_max = XI_CUTS["213"]
    xi_220_min, xi_220_max = XI_CUTS["220"]

    if smear_xi_sigma > 0.0:
        df_w = (
            df_w
            .Define("xi1_plot", f"smear_xi(xi1, {smear_xi_sigma:.8e}f)")
            .Define("xi2_plot", f"smear_xi(xi2, {smear_xi_sigma:.8e}f)")
        )
    else:
        df_w = (
            df_w
            .Define("xi1_plot", "xi1")
            .Define("xi2_plot", "xi2")
        )

    df = (
        df_w
        .Define("in420_1", f"(xi1_plot>={xi_420_min}f && xi1_plot<{xi_420_max}f)")
        .Define("in192_1", f"(xi1_plot>={xi_192_min}f && xi1_plot<{xi_192_max}f)")
        .Define("in213_1", f"(xi1_plot>={xi_213_min}f && xi1_plot<{xi_213_max}f)")
        .Define("in220_1", f"(xi1_plot>={xi_220_min}f && xi1_plot<{xi_220_max}f)")
        .Define("in420_2", f"(xi2_plot>={xi_420_min}f && xi2_plot<{xi_420_max}f)")
        .Define("in192_2", f"(xi2_plot>={xi_192_min}f && xi2_plot<{xi_192_max}f)")
        .Define("in213_2", f"(xi2_plot>={xi_213_min}f && xi2_plot<{xi_213_max}f)")
        .Define("in220_2", f"(xi2_plot>={xi_220_min}f && xi2_plot<{xi_220_max}f)")
        .Define("pass_side1", "in420_1 || in192_1 || in213_1 || in220_1")
        .Define("pass_side2", "in420_2 || in192_2 || in213_2 || in220_2")
        .Define("pass_PPS", "pass_side1 && pass_side2")
    )

    return df, global_factor


def draw_xi_hist2d(
    df,
    sample_label,
    selected,
    out_path,
    bins,
    xmin,
    xmax,
    ymin,
    ymax,
    use_weights,
    draw_cut_lines=False,
    shade_cut_gaps=False,
    logz=False,
):
    title_sel = "PPS selected" if selected else "No PPS selection"
    hist_name = f"h2_xi_{sample_label}_{'sel' if selected else 'all'}"
    hist_title = (
        ";"
        "#xi_{1};#xi_{2}"
    )

    df_use = df.Filter("pass_PPS") if selected else df

    model = (hist_name, hist_title, bins, xmin, xmax, bins, ymin, ymax)
    if use_weights:
        h2_r = df_use.Histo2D(model, "xi1_plot", "xi2_plot", "w")
    else:
        h2_r = df_use.Histo2D(model, "xi1_plot", "xi2_plot")

    h2 = h2_r.GetValue()

    c = ROOT.TCanvas(f"c_{hist_name}", "", 900, 750)
    c.SetRightMargin(0.14)
    c.SetLeftMargin(0.14)
    c.SetBottomMargin(0.11)
    c.SetTopMargin(0.04)
    c.SetLogz(1 if logz else 0)

    ROOT.gStyle.SetPalette(ROOT.kViridis)

    h2.SetStats(0)
    if logz:
        h2.SetMinimum(1e-12)
    h2.Draw("COLZ")

    # Keep Python references to drawn overlays alive until canvas is saved.
    overlays = []

    # Optional overlay for the unselected plot: shade xi regions that PPS cuts reject.
    if shade_cut_gaps and not selected:
        overlays.extend(_draw_cut_gap_shading(xmin, xmax, ymin, ymax))

    if shade_cut_gaps and not selected:
        overlays.extend(_draw_unshaded_boundaries(xmin, xmax, ymin, ymax))
    elif draw_cut_lines and selected:
        cut_vals = sorted({v for rng in XI_CUTS.values() for v in rng})
        for xcut in cut_vals:
            if xmin <= xcut <= xmax:
                lx = ROOT.TLine(xcut, ymin, xcut, ymax)
                lx.SetLineColor(ROOT.kRed + 1)
                lx.SetLineStyle(2)
                lx.SetLineWidth(2)
                lx.Draw()
                overlays.append(lx)
            if ymin <= xcut <= ymax:
                ly = ROOT.TLine(xmin, xcut, xmax, xcut)
                ly.SetLineColor(ROOT.kRed + 1)
                ly.SetLineStyle(2)
                ly.SetLineWidth(2)
                ly.Draw()
                overlays.append(ly)

    c.SaveAs(out_path)
    c.Close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2D histograms in xi1-xi2 for signal/background, with and without "
            "the PPS xi selection used in compare_sig_bkg_pps.py."
        )
    )
    parser.add_argument("--sig-root", default=None, help="Signal ROOT file")
    parser.add_argument("--bkg-root", default=None, help="Background ROOT file")
    parser.add_argument("--sig-xsec-pb", type=float, default=5.708103e-03,
                        help="Signal cross section in pb")
    parser.add_argument("--bkg-xsec-pb", type=float, default=8.997422e-01,
                        help="Background cross section in pb")
    parser.add_argument("--lumi-ab", type=float, default=3.0,
                        help="Integrated luminosity in ab^-1")
    parser.add_argument("--bins", type=int, default=15*15,
                        help="Number of bins in xi1 and xi2")
    parser.add_argument("--xmin", type=float, default=0.0,
                        help="xi1 min")
    parser.add_argument("--xmax", type=float, default=0.03,
                        help="xi1 max")
    parser.add_argument("--ymin", type=float, default=0.0,
                        help="xi2 min")
    parser.add_argument("--ymax", type=float, default=0.03,
                        help="xi2 max")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for PNGs (default: analysis/output)")
    parser.add_argument("--out-prefix", default="xi2d",
                        help="Output file prefix")
    parser.add_argument("--unweighted", action="store_true",
                        help="If set, do not apply event weights")
    parser.add_argument("--draw-cut-lines", action="store_true",
                        help="Draw dashed red xi-cut guide lines on PPS-selected plots")
    parser.add_argument("--shade-cut-gaps", action="store_true",
                        help="Shade xi regions excluded by PPS cuts on unselected plots")
    parser.add_argument("--smear-xi-sigma", type=float, default=0.0,
                        help="Gaussian sigma for xi smearing (0 disables smearing)")
    parser.add_argument("--logz", action="store_true",
                        help="Draw 2D histograms with logarithmic z-axis")
    args = parser.parse_args()

    study_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_sig_root = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output", "h_bb", "hbb_001", "evrecs", "evrec_hbb_001.root"
    )
    default_bkg_root = os.path.join(
        os.environ.get("HIGGS_BKG_DIR", os.path.join(study_dir, "bkg-generation")),
        "output", "qcd_bb", "qcd_001", "evrecs", "evrec_qcd_001.root"
    )

    sig_root = args.sig_root or default_sig_root
    bkg_root = args.bkg_root or default_bkg_root

    if not os.path.exists(sig_root):
        raise FileNotFoundError(f"Signal ROOT not found: {sig_root}")
    if not os.path.exists(bkg_root):
        raise FileNotFoundError(f"Background ROOT not found: {bkg_root}")

    out_dir = args.out_dir or os.path.join(study_dir, "analysis", "output")
    os.makedirs(out_dir, exist_ok=True)

    if args.smear_xi_sigma < 0.0:
        raise ValueError("--smear-xi-sigma must be >= 0")

    df_sig, w_sig = build_df(
        sig_root,
        args.sig_xsec_pb,
        args.lumi_ab,
        smear_xi_sigma=args.smear_xi_sigma,
    )
    df_bkg, w_bkg = build_df(
        bkg_root,
        args.bkg_xsec_pb,
        args.lumi_ab,
        smear_xi_sigma=args.smear_xi_sigma,
    )

    print("Signal global factor:", w_sig)
    print("Background global factor:", w_bkg)
    print("Weight mode:", "unweighted" if args.unweighted else "weighted")
    print("xi smearing sigma:", args.smear_xi_sigma)
    print("z-axis scale:", "log" if args.logz else "linear")

    use_weights = not args.unweighted

    draw_xi_hist2d(
        df_sig,
        "signal",
        selected=False,
        out_path=os.path.join(out_dir, f"{args.out_prefix}_signal_all.png"),
        bins=args.bins,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        use_weights=use_weights,
        draw_cut_lines=args.draw_cut_lines,
        shade_cut_gaps=args.shade_cut_gaps,
        logz=args.logz,
    )
    # draw_xi_hist2d(
    #     df_sig,
    #     "signal",
    #     selected=True,
    #     out_path=os.path.join(out_dir, f"{args.out_prefix}_signal_selected.png"),
    #     bins=args.bins,
    #     xmin=args.xmin,
    #     xmax=args.xmax,
    #     ymin=args.ymin,
    #     ymax=args.ymax,
    #     use_weights=use_weights,
    #     draw_cut_lines=args.draw_cut_lines,
    #     shade_cut_gaps=args.shade_cut_gaps,
    #     logz=args.logz,
    # )
    draw_xi_hist2d(
        df_bkg,
        "background",
        selected=False,
        out_path=os.path.join(out_dir, f"{args.out_prefix}_background_all.png"),
        bins=args.bins,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        use_weights=use_weights,
        draw_cut_lines=args.draw_cut_lines,
        shade_cut_gaps=args.shade_cut_gaps,
        logz=args.logz,
    )
    # draw_xi_hist2d(
    #     df_bkg,
    #     "background",
    #     selected=True,
    #     out_path=os.path.join(out_dir, f"{args.out_prefix}_background_selected.png"),
    #     bins=args.bins,
    #     xmin=args.xmin,
    #     xmax=args.xmax,
    #     ymin=args.ymin,
    #     ymax=args.ymax,
    #     use_weights=use_weights,
    #     draw_cut_lines=args.draw_cut_lines,
    #     shade_cut_gaps=args.shade_cut_gaps,
    #     logz=args.logz,
    # )

    print(f"Wrote xi1-xi2 2D plots to: {out_dir}")


if __name__ == "__main__":
    main()
