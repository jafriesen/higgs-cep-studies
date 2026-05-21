#!/usr/bin/env python3
"""
Build a ROOT file with one row per (signal event, minbias BX) combination.

For each combination we store proton-level jagged arrays separately for the
signal outgoing protons and the minbias protons (filtered to PPS xi-tagged
protons). The branches contain only source-level proton fields and tag
flags so pair-level quantities can be reconstructed later.

Usage examples in the repository use similar patterns to the existing
`build_signal_pairs.py` and `build_minbias_pairs.py` scripts.
"""
import argparse
import os
import glob
import math

import numpy as np


def parse_lhe_with_xsec(filename):
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
                    if len(parts) >= 1:
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


def group_indices_by_bx(bx_id):
    order = np.argsort(bx_id, kind="mergesort")
    bx_sorted = bx_id[order]
    unique_bx, starts, counts = np.unique(bx_sorted, return_index=True, return_counts=True)
    groups = [order[s:s + c] for s, c in zip(starts, counts)]
    return unique_bx, groups


# Shared station xi windows (copied from project scripts)
STATION_XI = {
    "192": (0.08, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.014, 0.0263),
    "420": (0.00325, 0.0116),
}


def station_tags_from_xi(xi_val):
    xi_192_min, xi_192_max = STATION_XI["192"]
    xi_213_min, xi_213_max = STATION_XI["213"]
    xi_220_min, xi_220_max = STATION_XI["220"]
    xi_420_min, xi_420_max = STATION_XI["420"]

    return {
        "192": (xi_val >= xi_192_min and xi_val < xi_192_max),
        "213": (xi_val >= xi_213_min and xi_val < xi_213_max),
        "220": (xi_val >= xi_220_min and xi_val < xi_220_max),
        "420": (xi_val >= xi_420_min and xi_val < xi_420_max),
    }


def build_combos(signal_events, bx_unique, bx_groups, npz_data, root_out, max_combos=None, print_each=False):
    try:
        import uproot
        import awkward as ak
    except ImportError as exc:
        raise RuntimeError("This script requires uproot and awkward. Install with: python3 -m pip install --user uproot awkward") from exc

    # Prepare NPZ arrays (npz_data may be a dict from load_minbias_npz)
    bx_id = npz_data["bx_id"]
    interaction_id = npz_data["interaction_id"]
    proton_idx = npz_data["proton_idx"]
    side = npz_data["side"]
    pz = npz_data["pz"]
    pt = npz_data["pt"]
    xi = npz_data["xi"]
    px = npz_data["px"] if "px" in npz_data else None
    py = npz_data["py"] if "py" in npz_data else None

    n_signal = len(signal_events)
    n_bx = int(bx_unique.size)
    n_max = min(n_signal, n_bx)
    if max_combos is not None:
        n_max = min(n_max, int(max_combos))

    # Containers for branches
    combo_id_list = []
    signal_evt_idx_list = []
    minbias_bx_id_list = []

    # jagged proton collections: lists of lists
    signal_side = []
    signal_xi = []
    signal_pt = []
    signal_px = []
    signal_py = []
    signal_pz = []
    signal_E = []
    signal_interaction = []
    signal_pidx = []
    signal_tag200 = []
    signal_tag400 = []

    minbias_side = []
    minbias_xi = []
    minbias_pt = []
    minbias_px = []
    minbias_py = []
    minbias_pz = []
    minbias_E = []
    minbias_interaction = []
    minbias_pidx = []
    minbias_tag200 = []
    minbias_tag400 = []

    skipped_signal = 0
    combos_written = 0

    for i in range(n_max):
        sig_idx = i
        bx = bx_unique[i]
        idx_bx = bx_groups[i]

        header, parts = signal_events[sig_idx]

        # Extract outgoing intact protons from the signal event
        protons_out = [p for p in parts if p["pid"] == 2212 and p["status"] == 1]
        if len(protons_out) != 2:
            skipped_signal += 1
            continue

        # Assign R/L by pz sign
        if protons_out[0]["pz"] >= 0.0:
            pR = protons_out[0]
            pL = protons_out[1]
        else:
            pR = protons_out[1]
            pL = protons_out[0]

        # Signal proton arrays (always store both)
        sig_sides = [-1, +1]
        sig_px = [pL["px"], pR["px"]]
        sig_py = [pL["py"], pR["py"]]
        sig_pz = [pL["pz"], pR["pz"]]
        sig_E = [pL["E"], pR["E"]]
        beam_E = None
        # xi for signal protons from E: try to read beam energy from incoming particles
        incoming = [p for p in parts if p["pid"] == 2212 and p["status"] == -1]
        if incoming and len(incoming) >= 1:
            beam_E = incoming[0]["E"]

        sig_xi_vals = []
        for pE in sig_E:
            if beam_E is not None and beam_E > 0.0:
                sig_xi_vals.append((beam_E - pE) / beam_E)
            else:
                sig_xi_vals.append(0.0)

        # signal tags
        sig_tag200 = []
        sig_tag400 = []
        for xi_val in sig_xi_vals:
            tags = station_tags_from_xi(xi_val)
            tag200 = tags["192"] or tags["213"] or tags["220"]
            tag400 = tags["420"]
            sig_tag200.append(int(tag200))
            sig_tag400.append(int(tag400))

        # minbias: filter protons in this bx by PPS tag (tag200 OR tag400)
        mb_sides = []
        mb_xi = []
        mb_pt = []
        mb_px = []
        mb_py = []
        mb_pz = []
        mb_E = []
        mb_inter = []
        mb_pidx = []
        mb_tag200 = []
        mb_tag400 = []

        for local_idx in idx_bx:
            xi_val = float(xi[local_idx])
            tags = station_tags_from_xi(xi_val)
            tag200 = tags["192"] or tags["213"] or tags["220"]
            tag400 = tags["420"]
            #if not (tag200 or tag400):
            #    continue

            mb_sides.append(int(side[local_idx]))
            mb_xi.append(float(xi_val))
            mb_pt.append(float(pt[local_idx]))
            mb_px.append(float(px[local_idx]) if px is not None else float("nan"))
            mb_py.append(float(py[local_idx]) if py is not None else float("nan"))
            mb_pz.append(float(pz[local_idx]))
            mb_E.append(float("nan"))
            mb_inter.append(int(interaction_id[local_idx]))
            mb_pidx.append(int(proton_idx[local_idx]))
            mb_tag200.append(int(tag200))
            mb_tag400.append(int(tag400))

        # Append into containers
        combo_id_list.append(int(combos_written))
        signal_evt_idx_list.append(int(sig_idx))
        minbias_bx_id_list.append(int(bx))

        signal_side.append(sig_sides)
        signal_xi.append(sig_xi_vals)
        signal_pt.append([math.hypot(sig_px[0], sig_py[0]), math.hypot(sig_px[1], sig_py[1])])
        signal_px.append(sig_px)
        signal_py.append(sig_py)
        signal_pz.append(sig_pz)
        signal_E.append(sig_E)
        signal_interaction.append([int(sig_idx), int(sig_idx)])
        signal_pidx.append([0, 1])
        signal_tag200.append(sig_tag200)
        signal_tag400.append(sig_tag400)

        minbias_side.append(mb_sides)
        minbias_xi.append(mb_xi)
        minbias_pt.append(mb_pt)
        minbias_px.append(mb_px)
        minbias_py.append(mb_py)
        minbias_pz.append(mb_pz)
        minbias_E.append(mb_E)
        minbias_interaction.append(mb_inter)
        minbias_pidx.append(mb_pidx)
        minbias_tag200.append(mb_tag200)
        minbias_tag400.append(mb_tag400)

        # Optionally print a per-combo summary as we go
        if print_each:
            print(
                f"Combo {combos_written}: signal_evt={sig_idx}, bx={bx}, "
                f"n_signal_protons=2, n_minbias_protons={len(mb_xi)}"
            )

        combos_written += 1

    # Convert to awkward arrays / numpy
    import awkward as ak  # already imported above; reimport for IDEs

    arrays = {
        "combo_id": np.asarray(combo_id_list, dtype=np.int32),
        "signal_event_idx": np.asarray(signal_evt_idx_list, dtype=np.int32),
        "minbias_bx_id": np.asarray(minbias_bx_id_list, dtype=np.int32),

        "signal_side": ak.Array(signal_side, behavior=None),
        "signal_xi": ak.Array(signal_xi, behavior=None),
        "signal_pt": ak.Array(signal_pt, behavior=None),
        "signal_px": ak.Array(signal_px, behavior=None),
        "signal_py": ak.Array(signal_py, behavior=None),
        "signal_pz": ak.Array(signal_pz, behavior=None),
        "signal_E": ak.Array(signal_E, behavior=None),
        "signal_interaction": ak.Array(signal_interaction, behavior=None),
        "signal_pidx": ak.Array(signal_pidx, behavior=None),
        "signal_tag200": ak.Array(signal_tag200, behavior=None),
        "signal_tag400": ak.Array(signal_tag400, behavior=None),

        "minbias_side": ak.Array(minbias_side, behavior=None),
        "minbias_xi": ak.Array(minbias_xi, behavior=None),
        "minbias_pt": ak.Array(minbias_pt, behavior=None),
        "minbias_px": ak.Array(minbias_px, behavior=None),
        "minbias_py": ak.Array(minbias_py, behavior=None),
        "minbias_pz": ak.Array(minbias_pz, behavior=None),
        "minbias_E": ak.Array(minbias_E, behavior=None),
        "minbias_interaction": ak.Array(minbias_interaction, behavior=None),
        "minbias_pidx": ak.Array(minbias_pidx, behavior=None),
        "minbias_tag200": ak.Array(minbias_tag200, behavior=None),
        "minbias_tag400": ak.Array(minbias_tag400, behavior=None),
    }

    # Write ROOT file
    print(f"[uproot] Writing output file: {root_out}")
    with uproot.recreate(root_out) as fout:
        fout["SignalMinbiasCombos"] = arrays

    print(f"Wrote ROOT file: {root_out}")
    print(f"Signal events loaded: {len(signal_events)}")
    print(f"Unique BX loaded: {len(bx_unique)}")
    print(f"Combos written: {combos_written}")
    print(f"Signal events skipped (malformed): {skipped_signal}")


def main():
    parser = argparse.ArgumentParser(description="Build signal+minbias combo ROOT file with proton-level jagged branches.")
    parser.add_argument("-s", "--signal-in", required=True, help="Signal LHE file or directory of .lhe/.dat files")
    parser.add_argument("-b", "--minbias-npz", required=True, help="Minbias .npz file (as produced by generate_minbias_protons.py)")
    parser.add_argument("-o", "--root-out", default=None, help="Output ROOT file (default: <signalbase>_plus_<npzbase>_combos.root)")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional max number of combos to write")
    parser.add_argument("--print-combo", action="store_true", help="Print a short summary for each combo as it is created")
    args = parser.parse_args()

    # Load signal events (file or directory)
    input_path = args.signal_in
    files = []
    if os.path.isdir(input_path):
        for pat in ("*.lhe", "*.dat"):
            files.extend(sorted(glob.glob(os.path.join(input_path, pat))))
        if not files:
            raise RuntimeError(f"No .lhe/.dat files found in directory {input_path}")
    else:
        files = [input_path]

    signal_events = []
    xsec_from_file = None
    for f in files:
        evs, xs = parse_lhe_with_xsec(f)
        if xs is not None and xsec_from_file is None:
            xsec_from_file = xs
        signal_events.extend(evs)

    if not signal_events:
        raise RuntimeError("No signal events parsed from input files.")

    # Load NPZ (support single .npz file or directory of .npz files)
    def load_minbias_npz(path_or_dir):
        files = []
        if os.path.isdir(path_or_dir):
            files = sorted(glob.glob(os.path.join(path_or_dir, "*.npz")))
            if not files:
                raise RuntimeError(f"No .npz files found in directory {path_or_dir}")
        else:
            files = [path_or_dir]

        arrs = {
            "bx_id": [],
            "interaction_id": [],
            "proton_idx": [],
            "side": [],
            "pz": [],
            "pt": [],
            "xi": [],
            "px": [],
            "py": [],
        }

        have_px_any = False
        have_py_any = False

        for f in files:
            d = np.load(f)
            for req in ("bx_id", "interaction_id", "proton_idx", "side", "pz", "pt", "xi"):
                if req not in d.files:
                    raise RuntimeError(f"Required array '{req}' missing in {f}")
            arrs["bx_id"].append(d["bx_id"])
            arrs["interaction_id"].append(d["interaction_id"])
            arrs["proton_idx"].append(d["proton_idx"])
            arrs["side"].append(d["side"])
            arrs["pz"].append(d["pz"])
            arrs["pt"].append(d["pt"])
            arrs["xi"].append(d["xi"])

            if "px" in d.files:
                arrs["px"].append(d["px"])
                have_px_any = True
            else:
                arrs["px"].append(np.full(d["bx_id"].shape, np.nan))

            if "py" in d.files:
                arrs["py"].append(d["py"])
                have_py_any = True
            else:
                arrs["py"].append(np.full(d["bx_id"].shape, np.nan))

        # concatenate
        out = {}
        for k, parts in arrs.items():
            out[k] = np.concatenate(parts) if parts else np.empty(0)
        # expose list of files
        out["_files"] = files
        # if no px/py present in any, remove keys to signal absense
        if not have_px_any:
            out.pop("px", None)
        if not have_py_any:
            out.pop("py", None)
        return out

    npz = load_minbias_npz(args.minbias_npz)

    unique_bx, bx_groups = group_indices_by_bx(npz["bx_id"])

    # Derive output name
    if args.root_out is None:
        sigbase = os.path.splitext(os.path.basename(files[0]))[0]
        npzbase = os.path.splitext(os.path.basename(args.minbias_npz))[0]
        outname = f"{sigbase}_plus_{npzbase}_combos.root"
        root_out = os.path.join(os.getcwd(), outname)
    else:
        root_out = args.root_out

    build_combos(
        signal_events,
        unique_bx,
        bx_groups,
        npz,
        root_out,
        max_combos=args.max_combos,
        print_each=getattr(args, "print_combo", False),
    )


if __name__ == "__main__":
    main()
