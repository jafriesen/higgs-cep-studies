#!/usr/bin/env python3
import argparse
import glob
import math
import os
from array import array
from collections import Counter

import ROOT as rt

rt.gROOT.SetBatch(True)


def parse_lhe_with_xsec(filename):
    """Parse a SuperChic-style LHE/dat file and return events + optional xsec."""
    events = []
    xsec_pb = None

    in_event = False
    expect_event_header = False
    current_particles = []
    current_header = None

    in_init = False
    init_line_count = 0

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("<init>"):
                in_init = True
                init_line_count = 0
                continue
            if line.startswith("</init>"):
                in_init = False
                continue

            if in_init:
                init_line_count += 1
                # In SuperChic, the second line usually begins with sigma(pb).
                if init_line_count == 2:
                    parts = line.split()
                    if parts:
                        try:
                            xsec_pb = float(parts[0])
                        except ValueError:
                            pass
                continue

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
                    raise RuntimeError(f"Unexpected event header in {filename}: {line}")
                current_header = {
                    "NUP": int(parts[0]),
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
            p = (
                int(parts[0]),   # pid
                int(parts[1]),   # status
                int(parts[2]),   # moth1
                int(parts[3]),   # moth2
                int(parts[4]),   # col1
                int(parts[5]),   # col2
                float(parts[6]), # px
                float(parts[7]), # py
                float(parts[8]), # pz
                float(parts[9]), # E
                float(parts[10]),# m
            )
            current_particles.append(p)

    return events, xsec_pb


def iter_lhe_events_with_xsec(filename):
    xsec_pb = None

    in_event = False
    expect_event_header = False
    current_particles = []
    current_header = None

    in_init = False
    init_line_count = 0

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("<init>"):
                in_init = True
                init_line_count = 0
                continue
            if line.startswith("</init>"):
                in_init = False
                continue

            if in_init:
                init_line_count += 1
                if init_line_count == 2:
                    parts = line.split()
                    if parts:
                        try:
                            xsec_pb = float(parts[0])
                        except ValueError:
                            pass
                continue

            if line.startswith("<event"):
                in_event = True
                expect_event_header = True
                current_particles = []
                current_header = None
                continue

            if line.startswith("</event"):
                if current_header is not None:
                    yield current_header, current_particles, xsec_pb
                in_event = False
                expect_event_header = False
                continue

            if not in_event:
                continue

            parts = line.split()

            if expect_event_header:
                if len(parts) < 6:
                    raise RuntimeError(f"Unexpected event header in {filename}: {line}")
                current_header = (
                    float(parts[2]),  # XWGTUP
                )
                expect_event_header = False
                continue

            if len(parts) < 13:
                continue

            current_particles.append((
                int(parts[0]),   # pid
                int(parts[1]),   # status
                float(parts[6]), # px
                float(parts[7]), # py
                float(parts[8]), # pz
                float(parts[9]), # E
            ))

def discover_lhe_files(input_path):
    if os.path.isdir(input_path):
        files = []
        for pat in ("*.lhe", "*.dat"):
            files.extend(glob.glob(os.path.join(input_path, pat)))
        files = [f for f in sorted(files) if not os.path.basename(f).endswith("_summary.dat")]
        if not files:
            raise RuntimeError(f"No .lhe or .dat files found in directory: {input_path}")
        return files

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return [input_path]


def pt(px, py, pz=None, E=None):
    return math.hypot(px, py)

def phi(px, py, pz=None, E=None):
    return math.atan2(py, px)

def eta(px, py, pz, E=None):
    p = math.hypot(px, py, pz)
    if p == abs(pz):
        return math.copysign(float("inf"), pz)
    return 0.5 * math.log((p + pz) / (p - pz))

def mass(px, py, pz, E):
    m2 = E*E - px*px - py*py - pz*pz
    return math.sqrt(m2) if m2 > 0.0 else 0.0

def delta_phi(phi1, phi2):
    d = abs(phi1 - phi2)
    return 2 * math.pi - d if d > math.pi else d

def delta_r(eta1, phi1, eta2, phi2):
    return math.hypot(eta1 - eta2, delta_phi(phi1, phi2))


def initialize_events_tree(output_root):
    out_dir = os.path.dirname(output_root)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = rt.TFile(output_root, "RECREATE")
    tree = rt.TTree("Events", "Compare-ready events tree from LHE/dat")

    branches = {
        "XWGTUP": array("f", [0.0]),
        "xi1": array("f", [0.0]),
        "xi2": array("f", [0.0]),
        "Mx": array("f", [0.0]),
        "yx": array("f", [0.0]),
        "bb_mass": array("f", [0.0]),
        "bb_pt": array("f", [0.0]),
        "bb_eta": array("f", [0.0]),
        "bb_phi": array("f", [0.0]),
        "bb_dR": array("f", [0.0]),
        "b_pt": array("f", [0.0]),
        "b_eta": array("f", [0.0]),
        "b_phi": array("f", [0.0]),
        "b_mass": array("f", [0.0]),
        "b_E": array("f", [0.0]),
        "bbar_pt": array("f", [0.0]),
        "bbar_eta": array("f", [0.0]),
        "bbar_phi": array("f", [0.0]),
        "bbar_mass": array("f", [0.0]),
        "bbar_E": array("f", [0.0]),
    }

    for name, arr in branches.items():
        tree.Branch(name, arr, f"{name}/F")

    return fout, tree, branches


def fill_event(branches, row):
    for name, value in row.items():
        branches[name][0] = float(value)


def build_events_tree_from_lhe(input_path, output_root, max_events=None, xsec_override=None, label="sample"):
    files = discover_lhe_files(input_path)
    print(f"\n[{label}] input path: {input_path}")
    print(f"[{label}] files discovered: {len(files)}")

    fout, tree, branches = initialize_events_tree(output_root)

    counters = Counter()
    xsec_from_files = None

    done = False
    for fpath in files:
        print(f"[{label}] processing file: {fpath}")
        if done:
            break

        for header, parts, xsec_file in iter_lhe_events_with_xsec(fpath):
            counters["events_parsed"] += 1
            if xsec_from_files is None and xsec_file is not None:
                    xsec_from_files = xsec_file

            if max_events is not None and counters["events_written"] >= max_events:
                done = True
                break

            incoming = []
            protons_out = []
            b = None
            bbar = None
            for p in parts:
                pid, status, px, py, pz, E = p

                if pid == 2212:
                    if status == -1:
                        incoming.append(p)
                    elif status == 1:
                        protons_out.append(p)
                elif status == 1:
                    if pid == 5:
                        if b is not None:
                            b = False
                            counters["skip_bad_b_content"] += 1
                            continue
                        else:
                            b = p
                    elif pid == -5:
                        if bbar is not None:
                            bbar = False
                            counters["skip_bad_b_content"] += 1
                            continue
                        else:
                            bbar = p

            if len(incoming) == 2 :
                beam_E1 = float(incoming[0][5])
                beam_E2 = float(incoming[1][5])
                if abs(beam_E1 - beam_E2) > 1e-3:
                    print(f"[{label}] warning: incoming protons have different energies: {beam_E1:.6f} GeV vs {beam_E2:.6f} GeV.")
                s_gev2 = (beam_E1 + beam_E2) ** 2

            if beam_E1 is None or beam_E2 is None:
                counters["skip_no_incoming_beam"] += 1
                continue

            if len(protons_out) != 2:
                counters["skip_bad_outgoing_protons"] += 1
                continue

            xi1 = (beam_E1 - protons_out[0][5]) / beam_E1
            xi2 = (beam_E2 - protons_out[1][5]) / beam_E2
            if xi1 <= 0.0 or xi2 <= 0.0:
                counters["skip_bad_xi"] += 1
                continue

            b_p = (b[2], b[3], b[4], b[5])
            b_pt = pt(*b_p)
            b_eta = eta(*b_p)
            b_phi = phi(*b_p)

            bbar_p = (bbar[2], bbar[3], bbar[4], bbar[5])
            bbar_pt = pt(*bbar_p)
            bbar_eta = eta(*bbar_p)
            bbar_phi = phi(*bbar_p)

            bb_p = (b_p[0] + bbar_p[0], b_p[1] + bbar_p[1], b_p[2] + bbar_p[2], b_p[3] + bbar_p[3])
            bb_mass = mass(*bb_p)

            branches["XWGTUP"][0] = header[0]
            branches["xi1"][0] = xi1
            branches["xi2"][0] = xi2
            branches["Mx"][0] = math.sqrt(xi1 * xi2 * s_gev2) if s_gev2 is not None else 0.0
            branches["yx"][0] = 0.5 * math.log(xi1 / xi2) if xi2 > 0 else 0.0
            branches["b_pt"][0] = b_pt
            branches["b_eta"][0] = b_eta
            branches["b_phi"][0] = b_phi
            branches["b_mass"][0] = mass(*b_p)
            branches["b_E"][0] = b_p[3]
            branches["bbar_pt"][0] = bbar_pt
            branches["bbar_eta"][0] = bbar_eta
            branches["bbar_phi"][0] = bbar_phi
            branches["bbar_mass"][0] = mass(*bbar_p)
            branches["bbar_E"][0] = bbar_p[3]
            branches["bb_mass"][0] = bb_mass
            branches["bb_pt"][0] = pt(*bb_p)
            branches["bb_eta"][0] = eta(*bb_p)
            branches["bb_phi"][0] = phi(*bb_p)
            branches["bb_dR"][0] = delta_r(b_eta, b_phi, bbar_eta, bbar_phi)
            tree.Fill()
            counters["events_written"] += 1

        print(f"[{label}] finished file: {fpath} (events parsed: {counters['events_parsed']}, events written: {counters['events_written']})")

    fout.Write()
    fout.Close()

    xsec_used = xsec_override if xsec_override is not None else xsec_from_files
    if beam_E1 is None or beam_E2 is None:
        print(f"[{label}] warning: no valid incoming beam protons were found.")
    else:
        print(f"[{label}] beam energies: {beam_E1:.6f} GeV, {beam_E2:.6f} GeV (s = {s_gev2:.6e} GeV^2)")

    if xsec_used is not None:
        src = "override" if xsec_override is not None else "from input files"
        print(f"[{label}] cross section ({src}): {xsec_used:.6e} pb")
    else:
        print(f"[{label}] cross section: not found")

    print(f"[{label}] output ROOT: {output_root}")
    print(f"[{label}] events parsed:  {counters['events_parsed']}")
    print(f"[{label}] events written: {counters['events_written']}")
    print(f"[{label}] skipped (bad outgoing protons): {counters['skip_bad_outgoing_protons']}")
    print(f"[{label}] skipped (bad xi):               {counters['skip_bad_xi']}")
    print(f"[{label}] skipped (bad b/bbar content):  {counters['skip_bad_b_content']}")
    print(f"[{label}] skipped (no incoming beam):    {counters['skip_no_incoming_beam']}")


def main():
    study_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    default_sig_in = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "h_bb"
    )
    default_bkg_in = os.path.join(
        os.environ.get("HIGGS_BKG_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "qcd_bb"
    )

    default_sig_out = os.path.join(
        os.environ.get("HIGGS_SIGNAL_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "hbb_tree.root",
    )
    default_bkg_out = os.path.join(
        os.environ.get("HIGGS_BKG_DIR", os.path.join(study_dir, "signal-generation")),
        "output",
        "qcdbb_tree.root",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build compare_sig_bkg_pps.py-compatible ROOT files (Events tree) "
            "from SuperChic-style LHE/dat inputs for both signal and background."
        )
    )
    parser.add_argument("--sig-input", default=default_sig_in,
                        help=f"Signal input LHE/dat file or directory (default: {default_sig_in})")
    parser.add_argument("--bkg-input", default=default_bkg_in,
                        help=f"Background input LHE/dat file or directory (default: {default_bkg_in})")
    parser.add_argument("--sig-root-out", default=default_sig_out,
                        help=f"Signal output ROOT path (default: {default_sig_out})")
    parser.add_argument("--bkg-root-out", default=default_bkg_out,
                        help=f"Background output ROOT path (default: {default_bkg_out})")
    parser.add_argument("--sig-max-events", type=int, default=None,
                        help="Optional max events to write for signal")
    parser.add_argument("--bkg-max-events", type=int, default=None,
                        help="Optional max events to write for background")
    parser.add_argument("--sig-xsec-pb", type=float, default=None,
                        help="Optional signal cross section override (pb), for logging")
    parser.add_argument("--bkg-xsec-pb", type=float, default=None,
                        help="Optional background cross section override (pb), for logging")
    args = parser.parse_args()

    build_events_tree_from_lhe(
        input_path=args.sig_input,
        output_root=args.sig_root_out,
        max_events=args.sig_max_events,
        xsec_override=args.sig_xsec_pb,
        label="signal",
    )

    build_events_tree_from_lhe(
        input_path=args.bkg_input,
        output_root=args.bkg_root_out,
        max_events=args.bkg_max_events,
        xsec_override=args.bkg_xsec_pb,
        label="background",
    )


if __name__ == "__main__":
    main()
