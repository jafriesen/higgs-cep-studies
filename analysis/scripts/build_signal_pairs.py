#!/usr/bin/env python3
import argparse
import os
import math
import glob

import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

# ----------------------------------------------------------------------
# Global / physics constants (match the minbias analyzer)
# ----------------------------------------------------------------------
E_BEAM_GEV   = 7000.0      # beam energy (should match the LHE)
SQRTS_GEV    = 2.0 * E_BEAM_GEV
S_GEV2       = SQRTS_GEV**2

# Higgs mass window (for in_Hwin flag)
M_H_GEV      = 125.0
M_WIN_LOW    = 115.0
M_WIN_HIGH   = 135.0

# PPS / pot xi windows (same as before)
STATION_XI = {
    "192": (0.0140, 0.0250),   # ~196 m
    "213": (0.0390, 0.0680),   # ~220 m
    "220": (0.0390, 0.0680),   # ~234 m
    "420": (0.00325, 0.0120),  # 420 m, low-xi region
}
STATION_NAMES = list(STATION_XI.keys())


# ----------------------------------------------------------------------
# LHE parsing (SuperChic-style, with <init> block)
# ----------------------------------------------------------------------
def parse_lhe_with_xsec(filename):
    """
    Parse SuperChic-style LHE file.

    Returns:
      events: list of (header_dict, particle_list)
        header_dict has: NUP, XWGTUP, ...
        each particle: dict with pid, status, px, py, pz, E, m
      xsec_pb: cross section in pb (float or None if not found)
    """
    events = []
    xsec_pb = None

    in_event = False
    expect_event_header = False
    current_particles = []
    current_header = None

    in_init = False
    init_line_count = 0

    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # <init> block: contains cross section line for SuperChic
            if line.startswith("<init>"):
                in_init = True
                init_line_count = 0
                continue
            if line.startswith("</init>"):
                in_init = False
                continue

            if in_init:
                init_line_count += 1
                # In SuperChic, the 2nd line often has: sigma(pb)  error(pb)  ...
                if init_line_count == 2:
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            xsec_pb = float(parts[0])
                        except ValueError:
                            pass
                continue

            # <event> block
            if line.startswith("<event"):
                in_event = True
                expect_event_header = True
                current_particles = []
                current_header = None
                continue

            if line.startswith("</event"):
                if current_header is not None:
                    events.append((current_header, current_particles))
                in_event = False
                expect_event_header = False
                continue

            if not in_event:
                continue

            if expect_event_header:
                parts = line.split()
                if len(parts) < 6:
                    raise RuntimeError(f"Unexpected event header: {line}")
                current_header = {
                    "NUP":    int(parts[0]),
                    "IDPRUP": int(parts[1]),
                    "XWGTUP": float(parts[2]),
                    "SCALUP": float(parts[3]),
                    "AQEDUP": float(parts[4]),
                    "AQCDUP": float(parts[5]),
                }
                expect_event_header = False
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            p = {
                "pid":    int(parts[0]),
                "status": int(parts[1]),
                "moth1":  int(parts[2]),
                "moth2":  int(parts[3]),
                "col1":   int(parts[4]),
                "col2":   int(parts[5]),
                "px":     float(parts[6]),
                "py":     float(parts[7]),
                "pz":     float(parts[8]),
                "E":      float(parts[9]),
                "m":      float(parts[10]),
            }
            current_particles.append(p)

    return events, xsec_pb


# ----------------------------------------------------------------------
# Helper: Δφ wrapping
# ----------------------------------------------------------------------
def wrap_dphi(dphi):
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi < -math.pi:
        dphi += 2.0 * math.pi
    return dphi


# ----------------------------------------------------------------------
# Build ProtonPairs tree (same structure as minbias analyzer)
# ----------------------------------------------------------------------
def build_proton_pairs_tree(events, output_root):
    """
    Build a ROOT file with a TTree "ProtonPairs" that has the same branches
    and variable definitions as the ProtonPairs tree from the minbias analyzer.
    """

    # Open ROOT file and define tree
    fout = ROOT.TFile(output_root, "RECREATE")
    tree = ROOT.TTree("ProtonPairs", "Left-right proton pairs (SuperChic pHp)")

    branches = {}

    # Integer branches
    for name in [
        "bx",
        "interaction_L", "interaction_R",
        "p_idx_L", "p_idx_R",
        "side_L", "side_R",
        "tag200_L", "tag200_R",
        "tag400_L", "tag400_R",
        "double_tag_420",
        "vtx_ok", "in_Hwin",
    ]:
        branches[name] = np.zeros(1, dtype=np.int32)

    # Float branches
    for name in [
        "xi_L", "xi_R",
        "z_L", "z_R", "dz",
        "M",
        "ln_xi_ratio", "yX", "abs_yX",
        "pT_L", "pT_R",
        "t1_abs", "t2_abs", "t_sum", "t_diff_abs",
        "pt_bal", "pt_bal_ratio",
        "abs_dphi", "dphi_from_pi",
    ]:
        branches[name] = np.zeros(1, dtype=np.float32)

    # Create branches in the ROOT tree
    for name, arr in branches.items():
        if arr.dtype.kind == "i":
            leaf_type = "I"
        else:
            leaf_type = "F"
        tree.Branch(name, arr, f"{name}/{leaf_type}")

    # Beam energy / s from the incoming protons in the first event
    first_parts = events[0][1]
    incoming = [p for p in first_parts if p["pid"] == 2212 and p["status"] == -1]
    if len(incoming) != 2:
        raise RuntimeError("Could not find 2 incoming protons in first SuperChic event.")

    beam_E = incoming[0]["E"]
    s = (2.0 * beam_E) ** 2
    print(f"SuperChic beam energy from LHE: {beam_E:.3f} GeV -> sqrt(s) = {math.sqrt(s):.3f} GeV")

    # xi ranges
    xi_192_min, xi_192_max = STATION_XI["192"]
    xi_213_min, xi_213_max = STATION_XI["213"]
    xi_220_min, xi_220_max = STATION_XI["220"]
    xi_420_min, xi_420_max = STATION_XI["420"]

    n_pairs_filled = 0

    for evt_idx, (header, parts) in enumerate(events):
        # Find outgoing intact protons
        protons_out = [p for p in parts if p["pid"] == 2212 and p["status"] == 1]
        if len(protons_out) != 2:
            continue

        p1, p2 = protons_out

        # Define which is "left" vs "right" by sign of pz
        # Convention: side_R = +1 (pz >= 0), side_L = -1 (pz < 0)
        if p1["pz"] >= 0.0:
            pR = p1
            pL = p2
        else:
            pR = p2
            pL = p1

        # Compute xi (fractional momentum loss) for each
        xi_L = (beam_E - pL["E"]) / beam_E
        xi_R = (beam_E - pR["E"]) / beam_E

        # Basic consistency: ignore completely crazy xi
        if xi_L <= 0.0 or xi_R <= 0.0:
            continue

        # PPS-like tags per proton
        def station_tags(xi_val):
            return {
                "192": (xi_val >= xi_192_min and xi_val < xi_192_max),
                "213": (xi_val >= xi_213_min and xi_val < xi_213_max),
                "220": (xi_val >= xi_220_min and xi_val < xi_220_max),
                "420": (xi_val >= xi_420_min and xi_val < xi_420_max),
            }

        tags_L = station_tags(xi_L)
        tags_R = station_tags(xi_R)

        tag200_L = tags_L["192"] or tags_L["213"] or tags_L["220"]
        tag200_R = tags_R["192"] or tags_R["213"] or tags_R["220"]
        tag400_L = tags_L["420"]
        tag400_R = tags_R["420"]

        tag_any_L = tag200_L or tag400_L
        tag_any_R = tag200_R or tag400_R
        has_400   = tag400_L or tag400_R

        # Base selection for inclusion in the tree:
        # - both protons tagged in some PPS station
        # - at least one proton in the 420 m window
        if not (tag_any_L and tag_any_R and has_400):
            continue

        # Kinematics
        # Four-vectors (just to get pT and φ; no need for full TLorentzVector)
        px_L, py_L, pz_L = pL["px"], pL["py"], pL["pz"]
        px_R, py_R, pz_R = pR["px"], pR["py"], pR["pz"]

        pT_L = math.hypot(px_L, py_L)
        pT_R = math.hypot(px_R, py_R)

        # Mass from proton xi's
        M2 = xi_L * xi_R * s
        M  = math.sqrt(M2) if M2 > 0.0 else 0.0

        # "Vertex" info: SuperChic has no pileup, so just set z = 0
        z_L = 0.0
        z_R = 0.0
        dz  = 0.0
        vtx_ok = True  # always true here

        # Higgs-window flag (from proton mass)
        in_Hwin = (M >= M_WIN_LOW) and (M <= M_WIN_HIGH)

        # xi-based pair variables
        ln_xi_ratio = math.log(xi_L / xi_R)
        yX = 0.5 * ln_xi_ratio
        abs_yX = abs(yX)

        # t-like variables: |t| ~ pT^2
        t1_abs = pT_L * pT_L
        t2_abs = pT_R * pT_R
        t_sum = t1_abs + t2_abs
        t_diff_abs = abs(t1_abs - t2_abs)

        # Δφ and pT-balance
        phi_L = math.atan2(py_L, px_L)
        phi_R = math.atan2(py_R, px_R)
        dphi = wrap_dphi(phi_L - phi_R)
        abs_dphi = abs(dphi)
        dphi_from_pi = math.pi - abs_dphi

        pt_bal_x = px_L + px_R
        pt_bal_y = py_L + py_R
        pt_bal = math.hypot(pt_bal_x, pt_bal_y)
        denom_pt = pT_L + pT_R
        pt_bal_ratio = pt_bal / denom_pt if denom_pt > 0.0 else 0.0

        # Fill branches
        branches["bx"][0] = int(evt_idx)  # fake BX = event index

        # For lack of a more detailed structure, treat "interaction" as event index
        branches["interaction_L"][0] = int(evt_idx)
        branches["interaction_R"][0] = int(evt_idx)

        # Proton indices within the event (0 and 1)
        branches["p_idx_L"][0] = 0
        branches["p_idx_R"][0] = 1

        # side: by convention L = -1, R = +1 (matches minbias scheme)
        branches["side_L"][0] = -1
        branches["side_R"][0] = +1

        # tags & flags
        branches["tag200_L"][0] = int(tag200_L)
        branches["tag200_R"][0] = int(tag200_R)
        branches["tag400_L"][0] = int(tag400_L)
        branches["tag400_R"][0] = int(tag400_R)
        branches["double_tag_420"][0] = int(tag400_L and tag400_R)

        branches["vtx_ok"][0]  = int(vtx_ok)
        branches["in_Hwin"][0] = int(in_Hwin)

        # floats
        branches["xi_L"][0] = float(xi_L)
        branches["xi_R"][0] = float(xi_R)
        branches["z_L"][0]  = float(z_L)
        branches["z_R"][0]  = float(z_R)
        branches["dz"][0]   = float(dz)
        branches["M"][0]    = float(M)

        branches["ln_xi_ratio"][0] = float(ln_xi_ratio)
        branches["yX"][0]          = float(yX)
        branches["abs_yX"][0]      = float(abs_yX)

        branches["pT_L"][0] = float(pT_L)
        branches["pT_R"][0] = float(pT_R)
        branches["t1_abs"][0] = float(t1_abs)
        branches["t2_abs"][0] = float(t2_abs)
        branches["t_sum"][0]  = float(t_sum)
        branches["t_diff_abs"][0] = float(t_diff_abs)

        branches["pt_bal"][0]       = float(pt_bal)
        branches["pt_bal_ratio"][0] = float(pt_bal_ratio)
        branches["abs_dphi"][0]     = float(abs_dphi)
        branches["dphi_from_pi"][0] = float(dphi_from_pi)

        tree.Fill()
        n_pairs_filled += 1

    fout.Write()
    fout.Close()
    print(f"Wrote ROOT file with SuperChic proton pairs: {output_root}")
    print(f"Total pairs stored in ProtonPairs tree: {n_pairs_filled}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    study_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_in = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "h_bb",
        "hbb_001",
        "outputs",
    )
    default_root_out = os.path.join(study_dir, "analysis", "output", "hbb_001_pairs.root")

    parser = argparse.ArgumentParser(
        description=(
            "Build ProtonPairs ROOT tree from SuperChic signal outputs "
            "(for comparison with combinatorial min-bias background)."
        )
    )
    parser.add_argument(
        "-i", "--input", default=default_in,
        help=f"Input LHE-style file OR directory (default: {default_in})"
    )
    parser.add_argument(
        "--xsec-pb", type=float, default=None,
        help="Override cross section in pb. If not set, try to read from <init> block."
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Optional maximum number of events to process (across all files)."
    )
    parser.add_argument(
        "--root-out", default=default_root_out,
        help=f"Output ROOT file for ProtonPairs tree (default: {default_root_out})."
    )
    args = parser.parse_args()

    input_path = args.input

    # Derive ROOT output name
    base_no_ext, ext = os.path.splitext(input_path)
    if args.root_out is None:
        output_root = base_no_ext + "_pairs.root"
    else:
        output_root = args.root_out
    outdir = os.path.dirname(output_root)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Read one or more LHE files
    # ------------------------------------------------------------------
    if os.path.isdir(input_path):
        lhe_files = []
        for pat in ("*.lhe", "*.dat"):
            lhe_files.extend(glob.glob(os.path.join(input_path, pat)))

        lhe_files = [
            f for f in lhe_files
            if not os.path.basename(f).endswith("_summary.dat")
        ]
        lhe_files = sorted(lhe_files)

        if not lhe_files:
            raise RuntimeError(f"No .lhe or .dat event files found in directory {input_path}")

        print(f"Reading LHE from directory: {input_path}")
        print(f"  Found {len(lhe_files)} candidate event files")

        events = []
        xsec_from_file = None

        for f in lhe_files:
            evs, xs = parse_lhe_with_xsec(f)

            if xsec_from_file is None and xs is not None:
                xsec_from_file = xs

            if args.max_events is not None:
                remaining = args.max_events - len(events)
                if remaining <= 0:
                    break
                if len(evs) > remaining:
                    evs = evs[:remaining]

            events.extend(evs)

            if args.max_events is not None and len(events) >= args.max_events:
                break

        print(f"  Total events loaded: {len(events)}")

    else:
        print(f"Reading LHE from: {input_path}")
        events, xsec_from_file = parse_lhe_with_xsec(input_path)
        if args.max_events is not None:
            events = events[:args.max_events]
            print(f"Using only first {len(events)} events due to --max-events")

    if not events:
        print("No events found, exiting.")
        return

    # Cross section (not used for the tree, but useful info)
    if args.xsec_pb is not None:
        xsec_pb = args.xsec_pb
        print(f"Using cross section from command line: {xsec_pb:.6e} pb")
    else:
        xsec_pb = xsec_from_file
        if xsec_pb is None:
            print("Warning: could not determine cross section from files; tree will be unweighted.")
        else:
            print(f"Using cross section from file(s): {xsec_pb:.6e} pb")

    # Build the ProtonPairs tree
    build_proton_pairs_tree(events, output_root)


if __name__ == "__main__":
    main()
