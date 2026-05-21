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
        #.Define("bb_px",  "b_px + bbar_px")
        #.Define("bb_py",  "b_py + bbar_py")
        #.Define("bb_pz",  "b_pz + bbar_pz")
        #.Define("bb_E",   "b_E + bbar_E")
        #.Define("bb_mass", "sqrt(bb_E*bb_E - (bb_px*bb_px + bb_py*bb_py + bb_pz*bb_pz))")
        #.Define("bb_pt",   "sqrt(bb_px*bb_px + bb_py*bb_py)")
        #.Define("bb_eta",  "my_eta(bb_px, bb_py, bb_pz)")
        #.Define("b_eta",   "my_eta(b_px, b_py, b_pz)")
        #.Define("bbar_eta","my_eta(bbar_px, bbar_py, bbar_pz)")
        #.Define("b_phi",     "atan2(b_py, b_px)")
        #.Define("bbar_phi",  "atan2(bbar_py, bbar_px)")
        #.Define("bb_dR",   "my_deltaR(b_eta, b_phi, bbar_eta, bbar_phi)")
        # smeared Mx
        #.Define("xi1_smear", "smear_xi(xi1, 0.0003f)")
        #.Define("xi2_smear", "smear_xi(xi2, 0.0003f)")
        #.Define("Mx_smear", "smear_Mx(Mx, 3.0f)")
        #.Define("pass_Mx_smear_window", "(Mx_smear > 100.f && Mx_smear < 150.f)")
        #.Define("mx_from_xi", "mx_from_xi(xi1_smear, xi2_smear)")
        #.Define("pass_mx_from_xi_window", "(mx_from_xi > 100.f && mx_from_xi < 150.f)")
        .Define("pt_leading", "b_pt > bbar_pt ? b_pt : bbar_pt")
        .Define("pt_leading_normalized", "pt_leading / bb_mass")
        #.Define("pt_leading_smeared", "smear(pt_leading, 0.1f*pt_leading)")
        .Define("pt_subleading", "b_pt > bbar_pt ? bbar_pt : b_pt")
        .Define("pt_subleading_normalized", "pt_subleading / bb_mass")
        #.Define("pt_subleading_smeared", "smear(pt_subleading, 0.1f*pt_subleading)")
        #.Define("deltaPhi", "abs(atan2(sin(b_phi - bbar_phi), cos(b_phi - bbar_phi)))")
        #.Define("deltaEta", "abs(b_eta - bbar_eta)")
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
        #.Define("mass_cut", "bb_mass > 100.f")
        #.Define("pt_normalized_cut", "pt_leading_normalized > 0.45")
    )

    # Apply PPS selection
    df_sel = df.Filter("pass_PPS", "PPS selection (192+213+220+420 on both sides)")
    #df_sel = df_sel.Filter("pass_mx_from_xi_window", "Mx from xi window (100-150 GeV)")
    #df_sel = df_sel.Filter("pt_cut", "Leading b pT cut (40-50 GeV)")
    df_sel = df_sel.Filter("dR_cut", "Delta R cut (3.3)")
    #df_sel = df_sel.Filter("mass_cut", "m_bb > 100 GeV")
    #df_sel = df_sel.Filter("pt_normalized_cut", "Normalized leading b pT cut (0.45-0.55)")

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

    return df_sel, global_factor


# ----------- helper: comparison histogram -------------------------------------

def make_comp_hist(df_sig, df_bkg, var, nbins, xmin, xmax, title, outname,norm = False,same_max = False):
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
    h_bkg_r = df_bkg.Histo1D(
        (f"h_bkg_{var}", title, nbins, xmin, xmax),
        var, "w"
    )

    h_sig = h_sig_r.GetValue()
    h_bkg = h_bkg_r.GetValue()

    h_sig.SetLineColor(ROOT.kRed)
    h_sig.SetMarkerColor(ROOT.kRed)
    h_sig.SetMarkerStyle(20)

    h_bkg.SetLineColor(ROOT.kBlue)
    h_bkg.SetMarkerColor(ROOT.kBlue)
    h_bkg.SetMarkerStyle(24)


    c = ROOT.TCanvas("c_"+var, "", 800, 600)
    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.05)
    c.cd()

    if norm:
        b_int = h_bkg.Integral()
        s_int = h_sig.Integral()
        if b_int > 0.0:
            h_bkg.Scale(1.0 / b_int)
        if s_int > 0.0:
            h_sig.Scale(1.0 / s_int)

    if same_max:
        sig_max = h_sig.GetMaximum()
        bkg_max = h_bkg.GetMaximum()
        if sig_max > 0.0 and bkg_max > 0.0:
            h_sig.Scale(bkg_max / sig_max)

    h_bkg.SetStats(0)

    # draw background first (usually larger)
    h_bkg.Draw(" hist E")
    h_sig.Draw("hist e SAME")

    # adjust y-axis to show both clearly
    max_y = max(h_sig.GetMaximum(), h_bkg.GetMaximum())
    h_bkg.SetMaximum(1.4 * max_y if max_y > 0 else 1.0)

    min_y = min(h_sig.GetMinimum(0), h_bkg.GetMinimum(0))
    h_bkg.SetMinimum(0.5 * min_y if min_y > 0 else 0.0)

    leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_sig, "Signal", "lep")
    leg.AddEntry(h_bkg, "Background", "lep")
    leg.Draw()
    if norm:
        c.SaveAs('norm_'+outname)
    if same_max:
        c.SaveAs('sameMax_'+outname)
    else:
        c.SaveAs(outname)
    c.Close()

    # Print integrals with uncertainties
    errS = array('d', [0.0])
    errB = array('d', [0.0])
    start_bin = 1
    end_bin = nbins

    if 'Mx' in var:
        start_bin = h_sig.FindBin(100.0)
        end_bin = h_sig.FindBin(150.0)

    S = h_sig.IntegralAndError(start_bin, end_bin, errS)
    B = h_bkg.IntegralAndError(start_bin, end_bin, errB)
    errS_val = errS[0]
    errB_val = errB[0]

    total_S = h_sig.Integral()
    total_B = h_bkg.Integral()

    print(f"  Yield (signal)     = {S:.3f} ± {errS_val:.3f}")
    print(f"  Yield (background) = {B:.3f} ± {errB_val:.3f}")
    print(f"  Total (signal)     = {total_S:.3f}")
    print(f"  Total (background) = {total_B:.3f}")
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
    parser.add_argument("--bkg-root", default=None,
                        help="Background ROOT file (from equivalent script on QCD bb)")
    parser.add_argument("--sig-xsec-pb", type=float, default=5.708103e-03,
                        help="Signal cross section in pb")
    parser.add_argument("--bkg-xsec-pb", type=float, default=8.997422e-01,
                        help="Background cross section in pb")
    parser.add_argument("--lumi-ab", type=float, default=3.0,
                        help="Integrated luminosity in ab^-1 (default: 3.0)")
    parser.add_argument("--out-prefix", default="comp_pps",
                        help="Prefix for output PNG files (default: comp_pps)")
    args = parser.parse_args()

    study_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_sig_root = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "hbb_tree.root"
    )
    default_bkg_root = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "qcdbb_tree.root"
    )

    if args.sig_root is None:
        args.sig_root = default_sig_root
    if args.bkg_root is None:
        args.bkg_root = default_bkg_root

    if not os.path.exists(args.sig_root):
        raise FileNotFoundError(f"Signal ROOT not found: {args.sig_root}")
    if not os.path.exists(args.bkg_root):
        raise FileNotFoundError(f"Background ROOT not found: {args.bkg_root}")

    # Build dataframes with weights & PPS selection
    df_sig, w_sig = build_df(args.sig_root, args.sig_xsec_pb, args.lumi_ab)
    df_bkg, w_bkg = build_df(args.bkg_root, args.bkg_xsec_pb, args.lumi_ab)

    print("\nSignal global factor:", w_sig)
    print("Background global factor:", w_bkg)

    # ---- Comparison plots ----
    # You can add more variables here as needed.

    # # 1) m_bb
    # make_comp_hist(
    #     df_sig, df_bkg, "bb_mass", 60, 90, 200,
    #     "b#bar{b} mass; m_{b#bar{b}} [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_mass.png"
    # )

    # # 2) pT_bb
    # make_comp_hist(
    #     df_sig, df_bkg, "bb_pt", 50, 0, 200,
    #     "b#bar{b} p_{T}; p_{T}(b#bar{b}) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_pt.png"
    # )

    # # 3) #DeltaR(b,bbar)
    # make_comp_hist(
    #     df_sig, df_bkg, "bb_dR", 50, 0, 6,
    #     "#Delta R(b,#bar{b}); #Delta R; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_dR.png"
    # )

    # 4) M_X from protons
    # make_comp_hist(
    #     df_sig, df_bkg, "Mx", 60, 90, 150,
    #     "M_{X} from protons; M_{X} [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_Mx.png", same_max=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "bb_mass", 60, 90, 200,
    #     "m_{bb} from jets; m_{bb} [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_mbb.png", same_max=True
    # )

    # 5) xi1 (just to see acceptance)
    # make_comp_hist(
    #     df_sig, df_bkg, "xi1", 200, 0, 0.03,
    #     "#xi_{1}; #xi_{1}; events (PPS-selected)",
    #     f"{args.out_prefix}_xi1.png", norm=True
    # )

    # # 5) M_X smeared with 5 GeV resolution
    # make_comp_hist(
    #     df_sig, df_bkg, "Mx_smear", 60, 90, 200,
    #     "M_{X}^{smeared} (#sigma_{res}= 3 GeV); M_{X}^{smeared} [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_Mx_smear.png", same_max=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "xi1_smear", 200, 0, 0.03,
    #     "#xi_{1}^{smeared}; #xi_{1}^{smeared}; events (PPS-selected)",
    #     f"{args.out_prefix}_xi1_smear.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "b_pt", 50, 0, 100,
    #     "p_{T}(b); p_{T}(b) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_b.png"
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "b_eta", 50, -5, 5,
    #     "#eta(b); #eta(b) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_eta_b.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "bbar_pt", 50, 0, 100,
    #     "p_{T}(#bar{b}); p_{T}(#bar{b}) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_bbar.png"
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "bbar_eta", 50, -5, 5,
    #     "#eta(#bar{b}); #eta(#bar{b}) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_eta_bbar.png", norm=True
    # )

    # # 5) M_X smeared with 5 GeV resolution
    # make_comp_hist(
    #     df_sig, df_bkg, "Mx_smear", 60, 90, 200,
    #     "M_{X}^{smeared} (#sigma_{res} = 5 GeV); M_{X}^{smeared} [GeV]; Normalized (PPS-selected)",
    #     f"{args.out_prefix}_Mx_smear.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "b_pt", 60, 0, 100,
    #     "p_{T}(b); p_{T}(b) [GeV]; Normalized (PPS-selected)",
    #     f"{args.out_prefix}_pt_b.png",  norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "b_eta", 60, -3.5, 3.5,
    #     "#eta(b); #eta(b) [GeV]; Normalized (PPS-selected)",
    #     f"{args.out_prefix}_eta_b.png",  norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "bbar_pt", 50, 0, 100,
    #     "p_{T}(#bar{b}); p_{T}(#bar{b}) [GeV]; Normalized (PPS-selected)",
    #     f"{args.out_prefix}_pt_bbar.png",  norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "bbar_eta", 50, -5, 5,
    #     "#eta(#bar{b}); #eta(#bar{b}) [GeV]; Normalized (PPS-selected)",
    #     f"{args.out_prefix}_eta_bbar.png",  norm=True
    # )


    #DeltaR(b,bbar)
    make_comp_hist(
        df_sig, df_bkg, "bb_dR", 60, 3, 5,
        "#Delta R(b,#bar{b}); #Delta R; events (PPS-selected)",
        f"{args.out_prefix}_bb_dR.png", norm=True
    )

    # make_comp_hist(
    #     df_sig, df_bkg, "bb_phi", 60, -3.5, 3.5,
    #     "#phi(b#bar{b}); #phi; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_phi.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "bb_eta", 50, -5, 5,
    #     "#eta(b#bar{b}); #eta; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_eta.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "deltaPhi", 60, 2.5, 3.15,
    #     "#Delta#phi(b,#bar{b}); |#Delta#phi|; events (PPS-selected)",
    #     f"{args.out_prefix}_deltaPhi.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "deltaEta", 60, 0, 7,
    #     "#Delta#eta(b,#bar{b}); |#Delta#eta|; events (PPS-selected)",
    #     f"{args.out_prefix}_deltaEta.png", norm=True
    # )

    # # 5) M_X smeared with 5 GeV resolution
    # make_comp_hist(
    #     df_sig, df_bkg, "Mx_smear", 80, 100, 140,
    #     "M_{X}^{smeared} (#sigma_{res}= 3 GeV); M_{X}^{smeared} [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_Mx_smear.png", same_max=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "mx_from_xi", 50, 100, 150,
    #     "M_{X} from xi (smeared); M_{X} from xi [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_Mx_from_xi.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "Mx_smear", 50, 100, 150,
    #     "M_{X}^{smeared} (#sigma_{res}= 3 GeV); M_{X}^{smeared} [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_Mx_smear.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "yx", 60,-1.5,1.5,
    #     "yx; yx; events (PPS-selected)",
    #     f"{args.out_prefix}_yx.png", norm=True
    # )
    # make_comp_hist(
    #     df_sig, df_bkg, "bb_pt", 60,0,2,
    #     "bb_pt; bb_pt; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_pt.png", norm=True
    # )
    # make_comp_hist(
    #     df_sig, df_bkg, "bb_eta", 60,-12,12,
    #     "bb_eta; bb_eta; events (PPS-selected)",
    #     f"{args.out_prefix}_bb_eta.png", norm=True
    # )

    make_comp_hist(
        df_sig, df_bkg, "pt_leading", 60, 30, 60,
        "p_{T}(leading b); p_{T}(leading b) [GeV]; events (PPS-selected)",
        f"{args.out_prefix}_pt_leading.png", norm=True
    )
    # make_comp_hist(
    #     df_sig, df_bkg, "pt_subleading", 60, 30, 60,
    #     "p_{T}(subleading b); p_{T}(subleading b) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_subleading.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "pt_leading_smeared", 60, 0, 90,
    #     "p_{T}(leading b) smeared; p_{T}(leading b) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_leading_smeared.png", norm=True
    # )
    # make_comp_hist(
    #     df_sig, df_bkg, "pt_subleading_smeared", 60, 0, 90,
    #     "p_{T}(subleading b) smeared; p_{T}(subleading b) [GeV]; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_subleading_smeared.png", norm=True
    # )

    # make_comp_hist(
    #     df_sig, df_bkg, "pt_leading_normalized", 60, 0.2, 0.5,
    #     "p_{T}(leading b) / m_{bb}; p_{T}(leading b) / m_{bb}; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_leading_normalized.png", norm=True
    # )
    # make_comp_hist(
    #     df_sig, df_bkg, "pt_subleading_normalized", 60, 0.2, 0.5,
    #     "p_{T}(subleading b) / m_{bb}; p_{T}(subleading b) / m_{bb}; events (PPS-selected)",
    #     f"{args.out_prefix}_pt_subleading_normalized.png", norm=True
    # )

if __name__ == "__main__":
    main()
