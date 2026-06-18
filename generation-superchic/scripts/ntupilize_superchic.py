#!/usr/bin/env python3
import argparse
import math
from array import array
from collections import Counter
from pathlib import Path

import ROOT
import vector


STATION_XI = {
    "192": (0.0800, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.0140, 0.0263),
    "420": (0.00325, 0.0116),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SuperChic LHE/dat event records into signal TTrees."
    )
    parser.add_argument("input", help="SuperChic evrec .dat/.lhe file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output ROOT file")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event limit")
    parser.add_argument("--verbose", action="store_true", help="Print parsed inputs and computed rows for each written event")
    return parser.parse_args()


def discover_event_files(path):
    path = Path(path)
    if path.is_file():
        return [path]

    patterns = ("*.lhe", "evrec*.dat", "*.dat")
    files = []
    for pattern in patterns:
        files.extend(path.rglob(pattern))
    files = sorted(
        {
            f
            for f in files
            if not f.name.startswith("output") and not f.name.endswith("_summary.dat")
        }
    )
    if not files:
        raise RuntimeError(f"No SuperChic event files found in {path}")
    return files


def iter_lhe_events(filename):
    in_event = False
    in_init = False
    expect_header = False
    init_line_count = 0
    xsec_pb = None
    header = None
    particles = []

    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
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
                    fields = line.split()
                    if fields:
                        try:
                            xsec_pb = float(fields[0])
                        except ValueError:
                            pass
                continue

            if line.startswith("<event"):
                in_event = True
                expect_header = True
                header = None
                particles = []
                continue
            if line.startswith("</event"):
                if header is not None:
                    yield header, particles, xsec_pb
                in_event = False
                expect_header = False
                continue
            if not in_event:
                continue

            fields = line.split()
            if expect_header:
                if len(fields) < 6:
                    raise RuntimeError(f"Unexpected event header in {filename}: {line}")
                header = {
                    "nup": int(fields[0]),
                    "idprup": int(fields[1]),
                    "xwgtup": float(fields[2]),
                    "scalup": float(fields[3]),
                    "aqedup": float(fields[4]),
                    "aqcdup": float(fields[5]),
                }
                expect_header = False
                continue

            if len(fields) < 11:
                continue
            particles.append(
                {
                    "pid": int(fields[0]),
                    "status": int(fields[1]),
                    "moth1": int(fields[2]),
                    "moth2": int(fields[3]),
                    "col1": int(fields[4]),
                    "col2": int(fields[5]),
                    "px": float(fields[6]),
                    "py": float(fields[7]),
                    "pz": float(fields[8]),
                    "E": float(fields[9]),
                    "m": float(fields[10]),
                }
            )


def p4(p):
    return vector.obj(px=p["px"], py=p["py"], pz=p["pz"], m=p["m"])


def p4_values(prefix, vec, energy=None, mass=None):
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}pt": vec.pt,
        f"{prefix}eta": vec.eta,
        f"{prefix}phi": vec.phi,
        f"{prefix}E": vec.E if energy is None else energy,
        f"{prefix}m": vec.mass if mass is None else mass,
    }


def station_hits(xi):
    hits = {
        station: int(xi >= xi_min and xi < xi_max)
        for station, (xi_min, xi_max) in STATION_XI.items()
    }
    return hits, sum(hits.values())


def make_tree(name, title, branch_specs):
    tree = ROOT.TTree(name, title)
    branches = {}
    leaf_types = {"i": "I", "f": "F", "d": "D"}
    for branch, code in branch_specs:
        leaf = leaf_types[code]
        branches[branch] = array(code, [0])
        tree.Branch(branch, branches[branch], f"{branch}/{leaf}")
    return tree, branches


def set_branch_values(branches, values):
    for name, value in values.items():
        branches[name][0] = value


def format_value(value):
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def print_mapping(title, values, indent="  "):
    print(f"{indent}{title}:")
    for key, value in values.items():
        print(f"{indent}  {key}: {format_value(value)}")


def print_particle(title, particle, indent="  "):
    fields = ("pid", "status", "moth1", "moth2", "col1", "col2", "px", "py", "pz", "E", "m")
    print(f"{indent}{title}:")
    print(f"{indent}  " + ", ".join(f"{field}={format_value(particle[field])}" for field in fields))


def print_event_debug(event_id, filename, header, incoming, protons, jets, beam_pos, beam_neg, proton_rows, interaction, pair):
    print(f"\n=== event {event_id} ===")
    print(f"file: {filename}")
    print_mapping("header", header)
    print(f"  beam_pos_E: {format_value(beam_pos)}")
    print(f"  beam_neg_E: {format_value(beam_neg)}")
    for idx, particle in enumerate(incoming):
        print_particle(f"input incoming[{idx}]", particle)
    for idx, particle in enumerate(protons):
        print_particle(f"input proton[{idx}]", particle)
    for idx, particle in enumerate(jets):
        print_particle(f"input jet[{idx}]", particle)
    for idx, row in enumerate(proton_rows):
        print_mapping(f"output Protons[{idx}]", row)
    print_mapping("output Interaction", interaction)
    if pair is None:
        print("  output ProtonPairs: <none>")
    else:
        print_mapping("output ProtonPairs", pair)


def build_trees():
    proton_tree, proton_br = make_tree(
        "Protons",
        "Signal protons from SuperChic events",
        [
            ("event_id", "i"),
            ("proton_id", "i"),
            ("side", "i"),
            ("xi", "d"),
            ("pt", "d"),
            ("eta", "d"),
            ("phi", "d"),
            ("E", "d"),
            ("m", "d"),
            ("hit_192", "i"),
            ("hit_213", "i"),
            ("hit_220", "i"),
            ("hit_420", "i"),
            ("n_station_hits", "i"),
        ],
    )

    interaction_tree, interaction_br = make_tree(
        "Interaction",
        "Signal hard interaction from SuperChic events",
        [
            ("event_id", "i"),
            ("j1_pid", "i"),
            ("j2_pid", "i"),
            ("j1_pt", "d"),
            ("j1_eta", "d"),
            ("j1_phi", "d"),
            ("j1_E", "d"),
            ("j1_m", "d"),
            ("j2_pt", "d"),
            ("j2_eta", "d"),
            ("j2_phi", "d"),
            ("j2_E", "d"),
            ("j2_m", "d"),
            ("jj_pt", "d"),
            ("jj_eta", "d"),
            ("jj_phi", "d"),
            ("jj_E", "d"),
            ("jj_m", "d"),
            ("jj_y", "d"),
        ],
    )

    pair_tree, pair_br = make_tree(
        "ProtonPairs",
        "Signal proton-pair variables from SuperChic events",
        [
            ("event_id", "i"),
            ("M", "d"),
            ("y", "d"),
            ("pass_pps", "i"),
        ],
    )

    return (proton_tree, proton_br), (interaction_tree, interaction_br), (pair_tree, pair_br)


def beam_energies(incoming):
    pos = next((p["E"] for p in incoming if p["pz"] > 0.0), None)
    neg = next((p["E"] for p in incoming if p["pz"] < 0.0), None)
    if pos is None and incoming:
        pos = incoming[0]["E"]
    if neg is None and len(incoming) > 1:
        neg = incoming[1]["E"]
    return pos, neg


def proton_row(event_id, proton_id, proton, beam_pos, beam_neg):
    side = 1 if proton["pz"] >= 0.0 else -1
    beam_e = beam_pos if side > 0 else beam_neg
    xi = (beam_e - proton["E"]) / beam_e if beam_e and beam_e > 0.0 else 0.0
    hits, n_hits = station_hits(xi)
    row = {
        "event_id": event_id,
        "proton_id": proton_id,
        "side": side,
        "xi": xi,
        "hit_192": hits["192"],
        "hit_213": hits["213"],
        "hit_220": hits["220"],
        "hit_420": hits["420"],
        "n_station_hits": n_hits,
    }
    row.update(p4_values("", p4(proton), energy=proton["E"], mass=proton["m"]))
    return row


def interaction_row(event_id, jets):
    j1, j2 = jets
    j1_p4 = p4(j1)
    j2_p4 = p4(j2)
    jj_p4 = j1_p4 + j2_p4
    row = {
        "event_id": event_id,
        "j1_pid": j1["pid"],
        "j2_pid": j2["pid"],
        "jj_y": jj_p4.rapidity,
    }
    row.update(p4_values("j1", j1_p4, energy=j1["E"], mass=j1["m"]))
    row.update(p4_values("j2", j2_p4, energy=j2["E"], mass=j2["m"]))
    row.update(p4_values("jj", jj_p4))
    return row


def pair_row(event_id, proton_rows, beam_pos, beam_neg):
    left = next((p for p in proton_rows if p["side"] < 0), None)
    right = next((p for p in proton_rows if p["side"] > 0), None)
    if left is None or right is None or left["xi"] <= 0.0 or right["xi"] <= 0.0:
        return None
    sqrt_s = (beam_pos + beam_neg) if beam_pos and beam_neg else 0.0
    pass_pps = int(left["n_station_hits"] > 0 and right["n_station_hits"] > 0)
    return {
        "event_id": event_id,
        "M": math.sqrt(left["xi"] * right["xi"]) * sqrt_s if sqrt_s > 0.0 else 0.0,
        "y": 0.5 * math.log(right["xi"] / left["xi"]),
        "pass_pps": pass_pps,
    }


def convert(input_path, output_path, max_events=None, verbose=False):
    files = discover_event_files(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ROOT.gROOT.SetBatch(True)
    root_file = ROOT.TFile(str(output_path), "RECREATE")
    (proton_tree, proton_br), (interaction_tree, interaction_br), (pair_tree, pair_br) = build_trees()

    counters = Counter()
    event_id = 0
    stop = False

    for filename in files:
        if stop:
            break
        for header, particles, _xsec_pb in iter_lhe_events(filename):
            if max_events is not None and counters["events_written"] >= max_events:
                stop = True
                break

            incoming = [p for p in particles if p["pid"] == 2212 and p["status"] == -1]
            protons = [p for p in particles if p["pid"] == 2212 and p["status"] == 1]
            jets = [p for p in particles if p["status"] == 1 and p["pid"] != 2212]

            counters["events_seen"] += 1
            if len(protons) != 2:
                counters["skip_bad_protons"] += 1
                continue
            if len(jets) != 2:
                counters["skip_bad_jets"] += 1
                continue

            beam_pos, beam_neg = beam_energies(incoming)
            if beam_pos is None or beam_neg is None:
                counters["skip_bad_beams"] += 1
                continue

            rows = [proton_row(event_id, i, proton, beam_pos, beam_neg) for i, proton in enumerate(protons)]
            interaction = interaction_row(event_id, jets)
            pp_row = pair_row(event_id, rows, beam_pos, beam_neg)

            if verbose:
                print_event_debug(
                    event_id,
                    filename,
                    header,
                    incoming,
                    protons,
                    jets,
                    beam_pos,
                    beam_neg,
                    rows,
                    interaction,
                    pp_row,
                )

            for row in rows:
                set_branch_values(proton_br, row)
                proton_tree.Fill()

            set_branch_values(interaction_br, interaction)
            interaction_tree.Fill()

            if pp_row is not None:
                set_branch_values(pair_br, pp_row)
                pair_tree.Fill()

            counters["events_written"] += 1
            event_id += 1

    root_file.Write()
    root_file.Close()
    return files, counters


def main():
    args = parse_args()
    files, counters = convert(args.input, args.output, max_events=args.max_events, verbose=args.verbose)
    print(f"Input files: {len(files)}")
    print(f"Events seen: {counters['events_seen']}")
    print(f"Events written: {counters['events_written']}")
    for key in ("skip_bad_beams", "skip_bad_protons", "skip_bad_jets"):
        if counters[key]:
            print(f"{key}: {counters[key]}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
