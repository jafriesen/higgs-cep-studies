#!/usr/bin/env python3
import argparse
import ROOT
from array import array
import os

ROOT.gROOT.SetBatch(True)

# ----------- C++ helpers (kinematics + smearing) -------------------

ROOT.gInterpreter.Declare(r"""
#include <cmath>
#include <TRandom3.h>

float my_eta(float px, float py, float pz) {
    float p = std::sqrt(px*px + py*py + pz*pz);
    if (p == std::fabs(pz)) return 0.0f;
    float arg = (p + pz) / (p - pz);
    if (arg <= 0.0f) return 0.0f;
    return 0.5f * std::log(arg);
}

float my_rapidity(float E, float pz) {
    if (E == std::fabs(pz)) return 0.0f;
    float arg = (E + pz) / (E - pz);
    if (arg <= 0.0f) return 0.0f;
    return 0.5f * std::log(arg);
}

float my_deltaR(float eta1, float phi1, float eta2, float phi2) {
    float dphi = phi1 - phi2;
    while (dphi >  M_PI) dphi -= 2.f*M_PI;
    while (dphi < -M_PI) dphi += 2.f*M_PI;
    float deta = eta1 - eta2;
    return std::sqrt(deta*deta + dphi*dphi);
}

// global RNG for smearing (ok as long as we don't use implicit MT)
static TRandom3 gRand(12345);

// smear Mx with a Gaussian of width 'sigma' (GeV)
float smear_Mx(float mx, float sigma) {
    if (mx <= 0.f) return mx;
    return mx + gRand.Gaus(0.0, sigma);
}

float smear_xi(float xi, float sigma) {
    if (sigma <= 0.f) return xi;
    float out = xi + gRand.Gaus(0.0, sigma);
    return out < 0.f ? 0.f : out;
}

float smear(float val, float sigma) {
    return val + gRand.Gaus(0.0, sigma);
}

float mx_from_xi(float xi1, float xi2, float sqrt_s=13600.f) {
    return std::sqrt(xi1 * xi2) * sqrt_s;
}
""")

# ----------- helper: build RDataFrame with weights & PPS selection -----------

def build_df(filename, xsec_pb, lumi_ab):
    """
    Build a RDataFrame for a given ROOT file, define:
      - per-event weight 'w' normalized to sigma * lumi
      - kinematics for bb system
      - PPS selection pass_PPS based on xi1, xi2 and stations 192/213/220/420
      - smeared Mx: Mx_smear (Gaussian sigma = 5 GeV)
    """
    df0 = ROOT.RDataFrame("Events", filename)

    # Use only 100000 events for quick testing (remove or increase for full processing)
    df0 = df0.Range(10000000)

    # Sum of generator weights
    sumw = df0.Sum("XWGTUP").GetValue()
    if sumw == 0:
        raise RuntimeError(f"Sum of XWGTUP is zero in {filename}")

    lumi_pb = lumi_ab * 1e6  # 1 ab^-1 = 1e6 pb^-1
    global_factor = (xsec_pb * lumi_pb) / float(sumw)

    print(f"\nFile: {filename}")
    print(f"  sum(XWGTUP)          = {sumw:.6e}")
    print(f"  lumi                 = {lumi_ab:.3f} ab^-1 = {lumi_pb:.3e} pb^-1")
    print(f"  xsec                 = {xsec_pb:.6e} pb")
    print(f"  global weight factor = {global_factor:.6e}")

    # --- handle 'w' possibly already existing in the tree ---
    cols = [str(c) for c in df0.GetColumnNames()]
    if "w" in cols:
        df_w = df0.Redefine("w", f"{global_factor:.8e}f * XWGTUP")
    else:
        df_w = df0.Define("w", f"{global_factor:.8e}f * XWGTUP")

    # PPS station xi ranges
    xi_420_min, xi_420_max = 0.00325, 0.0116
    xi_192_min, xi_192_max = 0.08, 0.1967
    xi_213_min, xi_213_max = 0.0375, 0.0688
    xi_220_min, xi_220_max = 0.014, 0.0263

    # Build reusable station masks once, then compose selections from them.
    df = (
        df_w
        .Define("pt_leading", "b_pt > bbar_pt ? b_pt : bbar_pt")
        .Define("pt_leading_normalized", "pt_leading / bb_mass")
        .Define("pt_subleading", "b_pt > bbar_pt ? bbar_pt : b_pt")
        .Define("pt_subleading_normalized", "pt_subleading / bb_mass")
        .Define("deltaPhi", "abs(atan2(sin(b_phi - bbar_phi), cos(b_phi - bbar_phi)))")
        .Define("deltaEta", "abs(b_eta - bbar_eta)")
        .Define("in420_1", f"(xi1>={xi_420_min}f && xi1<{xi_420_max}f)")
        .Define("in192_1", f"(xi1>={xi_192_min}f && xi1<{xi_192_max}f)")
        .Define("in213_1", f"(xi1>={xi_213_min}f && xi1<{xi_213_max}f)")
        .Define("in220_1", f"(xi1>={xi_220_min}f && xi1<{xi_220_max}f)")
        .Define("in420_2", f"(xi2>={xi_420_min}f && xi2<{xi_420_max}f)")
        .Define("in192_2", f"(xi2>={xi_192_min}f && xi2<{xi_192_max}f)")
        .Define("in213_2", f"(xi2>={xi_213_min}f && xi2<{xi_213_max}f)")
        .Define("in220_2", f"(xi2>={xi_220_min}f && xi2<{xi_220_max}f)")
        .Define("pass_side1", "in420_1 || in192_1 || in213_1 || in220_1")
        .Define("pass_side2", "in420_2 || in192_2 || in213_2 || in220_2")
        .Define("pass_PPS", "pass_side1 && pass_side2")
        .Define("pass_420_1", "(in420_1 || in420_2) && pass_PPS")
        .Define("pass_420_2", "in420_1 && in420_2")
        .Define("pt_cut", "pt_leading > 45.f && pt_leading < 55.f")
        .Define("dR_cut", "bb_dR < 3.3f")
    )

    # Apply PPS selection
    df_sel = df.Filter("pass_PPS", "PPS selection (192+213+220+420 on both sides)")

    df_sel_420_1 = df.Filter("pass_420_1", "PPS selection (420 on at least one side)")
    df_sel_420_2 = df.Filter("pass_420_2", "PPS selection (420 on both sides)")

    total_count = df0.Count()
    pass_count = df_sel.Count()
    pass_420_1_count = df_sel_420_1.Count()
    pass_420_2_count = df_sel_420_2.Count()

    n_total = total_count.GetValue()
    n_pass = pass_count.GetValue()
    print(f"  Events total: {n_total}, {n_total * global_factor:.3f} weighted")
    print(f"  Events passing PPS: {n_pass}, {n_pass * global_factor:.3f} weighted")
    print(f"  PPS acceptance: {n_pass/n_total:.3e}")

    n_pass_420_1 = pass_420_1_count.GetValue()
    n_pass_420_2 = pass_420_2_count.GetValue()
    print(f"  Events passing 420 on at least one side: {n_pass_420_1}, {n_pass_420_1 * global_factor:.3f} weighted")
    print(f"  420 acceptance (at least one side): {n_pass_420_1/n_total:.3e}")
    print(f"  Events passing 420 on both sides: {n_pass_420_2}, {n_pass_420_2 * global_factor:.3f} weighted")
    print(f"  420 acceptance (both sides): {n_pass_420_2/n_total:.3e}")

    return df_sel_420_1, global_factor


# ----------- helper: comparison histogram -------------------------------------

def make_comp_hist(df_sig, df_bkg_qcd, df_bkg_gamgam, var, nbins, xmin, xmax, title, outname,norm = False,same_max = False):
    """
    Make comparison histogram for 'var' after PPS selection:
      - signal in red with error bars
      - background in blue with error bars
    """
    print(f"\nMaking comparison plot for {var} -> {outname}")

    h_sig_r = df_sig.Histo1D(
        (f"h_sig_{var}", title, nbins, xmin, xmax),
        var, "w"
    )
    h_bkg_qcd_r = df_bkg_qcd.Histo1D(
        (f"h_bkg_qcd_{var}", title, nbins, xmin, xmax),
        var, "w"
    )
    h_bkg_gamgam_r = df_bkg_gamgam.Histo1D(
        (f"h_bkg_gamgam_{var}", title, nbins, xmin, xmax),
        var, "w"
    )

    h_sig = h_sig_r.GetValue()
    h_bkg_qcd = h_bkg_qcd_r.GetValue()
    h_bkg_gamgam = h_bkg_gamgam_r.GetValue()

    h_sig.SetLineColor(ROOT.kRed)
    h_sig.SetMarkerColor(ROOT.kRed)
    h_sig.SetMarkerStyle(20)

    h_bkg_qcd.SetLineColor(ROOT.kBlue)
    h_bkg_qcd.SetMarkerColor(ROOT.kBlue)
    h_bkg_qcd.SetMarkerStyle(24)

    h_bkg_gamgam.SetLineColor(ROOT.kGreen)
    h_bkg_gamgam.SetMarkerColor(ROOT.kGreen)
    h_bkg_gamgam.SetMarkerStyle(25)

    c = ROOT.TCanvas("c_"+var, "", 800, 600)
    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.05)
    c.cd()

    if norm:
        b_qcd_int = h_bkg_qcd.Integral()
        b_gamgam_int = h_bkg_gamgam.Integral()
        s_int = h_sig.Integral()
        if b_qcd_int > 0.0:
            h_bkg_qcd.Scale(1.0 / b_qcd_int)
        if b_gamgam_int > 0.0:
            h_bkg_gamgam.Scale(1.0 / b_gamgam_int)
        if s_int > 0.0:
            h_sig.Scale(1.0 / s_int)

    if same_max:
        sig_max = h_sig.GetMaximum()
        bkg_qcd_max = h_bkg_qcd.GetMaximum()
        bkg_gamgam_max = h_bkg_gamgam.GetMaximum()
        bkg_max = max(bkg_qcd_max, bkg_gamgam_max)
        if sig_max > 0.0 and bkg_max > 0.0:
            h_sig.Scale(bkg_max / sig_max)

    h_bkg_qcd.SetStats(0)

    # draw background first (usually larger)
    h_bkg_qcd.Draw(" hist E")
    h_bkg_gamgam.Draw(" hist E SAME")
    h_sig.Draw("hist e SAME")

    # adjust y-axis to show both clearly
    max_y = max(h_sig.GetMaximum(), h_bkg_qcd.GetMaximum())
    h_bkg_qcd.SetMaximum(1.4 * max_y if max_y > 0 else 1.0)

    min_y = min(h_sig.GetMinimum(0), h_bkg_qcd.GetMinimum(0))
    h_bkg_qcd.SetMinimum(0.5 * min_y if min_y > 0 else 0.0)

    leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_sig, "Signal", "lep")
    leg.AddEntry(h_bkg_qcd, "Background (QCD)", "lep")
    leg.AddEntry(h_bkg_gamgam, "Background (Gamma-Gamma)", "lep")
    leg.Draw()

    c.SaveAs(outname)
    c.Close()

    # Print integrals with uncertainties
    errS = array('d', [0.0])
    errB_qcd = array('d', [0.0])
    errB_gamgam = array('d', [0.0])
    start_bin = 1
    end_bin = nbins

    if 'Mx' in var:
        start_bin = h_sig.FindBin(100.0)
        end_bin = h_sig.FindBin(150.0)

    S = h_sig.IntegralAndError(start_bin, end_bin, errS)
    B_qcd = h_bkg_qcd.IntegralAndError(start_bin, end_bin, errB_qcd)
    B_gamgam = h_bkg_gamgam.IntegralAndError(start_bin, end_bin, errB_gamgam)
    B = B_qcd + B_gamgam
    errS_val = errS[0]
    errB_qcd_val = errB_qcd[0]
    errB_gamgam_val = errB_gamgam[0]

    total_S = h_sig.Integral()
    total_B_qcd = h_bkg_qcd.Integral()
    total_B_gamgam = h_bkg_gamgam.Integral()
    total_B = total_B_qcd + total_B_gamgam

    print(f"  Yield (signal)     = {S:.3f} ± {errS_val:.3f}")
    print(f"  Yield (QCD background)        = {B_qcd:.3f} ± {errB_qcd_val:.3f}")
    print(f"  Yield (Gamma-Gamma) = {B_gamgam:.3f} ± {errB_gamgam_val:.3f}")
    print(f"  Yield (total background)   = {B:.3f} ± {((errB_qcd_val**2 + errB_gamgam_val**2)**0.5):.3f}")
    print(f"  Total (signal)     = {total_S:.3f}")
    print(f"  Total (QCD background)        = {total_B_qcd:.3f}")
    print(f"  Total (Gamma-Gamma) = {total_B_gamgam:.3f}")
    print(f"  Total background   = {total_B:.3f} ± {((errB_qcd_val**2 + errB_gamgam_val**2)**0.5):.3f}")
    if B > 0:
        print(f"  S/B                = {S/B:.3f}")
        if S > 0:
            print(f"  S/sqrt(B)          = {S/(B**0.5):.3f}")

# ----------- main -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare H->bb signal and QCD bb background after PPS selection (192,213,220,420)."
    )
    parser.add_argument("--sig-root", default=None,
                        help="Signal ROOT file (from make_hbb_plots.py-style script)")
    parser.add_argument("--bkg-qcd-root", default=None,
                        help="Background ROOT file (from equivalent script on QCD cc)")
    parser.add_argument("--bkg-gamgam-root", default=None,
                        help="Background ROOT file for gamma-gamma -> cc (from equivalent script on gamgam cc)")
    parser.add_argument("--sig-xsec-pb", type=float, default=0.28e-04,
                        help="Signal cross section in pb")
    parser.add_argument("--bkg-qcd-xsec-pb", type=float, default=0.135851672e-01,
                        help="Background cross section in pb")
    parser.add_argument("--bkg-gamgam-xsec-pb", type=float, default=0.4e-01,
                        help="Gamma-gamma background cross section in pb")
    parser.add_argument("--lumi-ab", type=float, default=3.0,
                        help="Integrated luminosity in ab^-1 (default: 3.0)")
    parser.add_argument("--out-prefix", default="comp_pps_cc",
                        help="Prefix for output PNG files (default: comp_pps)")
    parser.add_argument("--norm", action="store_true",
                        help="Normalize histograms to unit area")
    parser.add_argument("--same-max", action="store_true",
                        help="Set same maximum for signal and background histograms")
    parser.add_argument("--out-dir", default="comp_plots_cc",
                        help="Output directory for plots (default: current directory)")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.norm:
        args.out_prefix = "norm_" + args.out_prefix
    if args.same_max:
        args.out_prefix = "sameMax_" + args.out_prefix

    study_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_sig_root = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "hcc_tree.root"
    )
    default_bkg_qcd_root = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "qcdcc_tree.root"
    )
    default_bkg_gamgam_root = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "gamgam_cc_tree.root"
    )

    if args.sig_root is None:
        args.sig_root = default_sig_root
    if args.bkg_qcd_root is None:
        args.bkg_qcd_root = default_bkg_qcd_root
    if args.bkg_gamgam_root is None:
        args.bkg_gamgam_root = default_bkg_gamgam_root

    if not os.path.exists(args.sig_root):
        raise FileNotFoundError(f"Signal ROOT not found: {args.sig_root}")
    if not os.path.exists(args.bkg_qcd_root):
        raise FileNotFoundError(f"Background ROOT not found: {args.bkg_qcd_root}")
    if not os.path.exists(args.bkg_gamgam_root):
        raise FileNotFoundError(f"Gamma-gamma Background ROOT not found: {args.bkg_gamgam_root}")

    # Build dataframes with weights & PPS selection
    df_sig, w_sig = build_df(args.sig_root, args.sig_xsec_pb, args.lumi_ab)
    df_bkg_qcd, w_bkg_qcd = build_df(args.bkg_qcd_root, args.bkg_qcd_xsec_pb, args.lumi_ab)
    df_bkg_gamgam, w_bkg_gamgam = build_df(args.bkg_gamgam_root, args.bkg_gamgam_xsec_pb, args.lumi_ab)

    print("\nSignal global factor:", w_sig)
    print("Background global factor:", w_bkg_qcd)
    print("Gamma-gamma Background global factor:", w_bkg_gamgam)

    # ---- Comparison plots ----
    # You can add more variables here as needed.

    # # 1) m_bb
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bb_mass", 60, 90, 200,
        "c#bar{c} mass; m_{c#bar{c}} [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_bb_mass.png", same_max=args.same_max, norm=args.norm
    )

    # # 2) pT_bb
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bb_pt", 50, 0, 200,
        "b#bar{b} p_{T}; p_{T}(b#bar{b}) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_bb_pt.png", same_max=args.same_max, norm=args.norm
    )

    # # 3) #DeltaR(b,bbar)
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bb_dR", 50, 0, 6,
        "#Delta R(b,#bar{b}); #Delta R; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_bb_dR.png", same_max=args.same_max, norm=args.norm
    )

    # 4) M_X from protons
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "Mx", 60, 90, 150,
        "M_{X} from protons; M_{X} [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_Mx.png", same_max=args.same_max, norm=args.norm
    )

    # 5) xi1 (just to see acceptance)
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "xi1", 200, 0, 0.03,
        "#xi_{1}; #xi_{1}; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_xi1.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "b_pt", 50, 0, 100,
        "p_{T}(b); p_{T}(b) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_pt_b.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "b_eta", 50, -5, 5,
        "#eta(c); #eta(c) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_eta_b.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bbar_pt", 50, 0, 100,
        "p_{T}(#bar{b}); p_{T}(#bar{b}) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_pt_bbar.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bbar_eta", 50, -5, 5,
        "#eta(#bar{b}); #eta(#bar{b}) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_eta_bbar.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bb_phi", 60, -3.5, 3.5,
        "#phi(c#bar{c}); #phi; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_bb_phi.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "bb_eta", 50, -5, 5,
        "#eta(c#bar{c}); #eta; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_bb_eta.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "deltaEta", 60, 0, 7,
        "#Delta#eta(c,#bar{c}); |#Delta#eta|; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_deltaEta.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "yx", 60,-1.5,1.5,
        "yx; yx; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_yx.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "pt_leading", 60, 30, 60,
        "p_{T}(leading c); p_{T}(leading c) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_pt_leading.png", norm=args.norm, same_max=args.same_max
    )
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "pt_subleading", 60, 30, 60,
        "p_{T}(subleading c); p_{T}(subleading c) [GeV]; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_pt_subleading.png", norm=args.norm, same_max=args.same_max
    )

    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "pt_leading_normalized", 60, 0.2, 0.5,
        "p_{T}(leading c) / m_{cc}; p_{T}(leading c) / m_{cc}; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_pt_leading_normalized.png", norm=args.norm, same_max=args.same_max
    )
    make_comp_hist(
        df_sig, df_bkg_qcd, df_bkg_gamgam, "pt_subleading_normalized", 60, 0.2, 0.5,
        "p_{T}(subleading c) / m_{cc}; p_{T}(subleading c) / m_{cc}; events (PPS-selected)",
        f"{args.out_dir}/{args.out_prefix}_pt_subleading_normalized.png", norm=args.norm, same_max=args.same_max
    )

if __name__ == "__main__":
    main()
