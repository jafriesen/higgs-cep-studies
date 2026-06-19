#!/usr/bin/env python3
import argparse
import math
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import ROOT
import pythia8mc as pythia8
import vector
import yaml


STATIONS = ("192", "213", "220", "420")


def repo_root():
    return Path(__file__).resolve().parents[2]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def parse_args():
    default_parameters = repo_root() / "parameters.yaml"
    parser = argparse.ArgumentParser(
        description="Generate Pythia8 minbias events and write PPS-level ROOT TTrees."
    )
    parser.add_argument("-o", "--output", required=True, help="Output ROOT file")
    parser.add_argument(
        "--parameters",
        default=str(default_parameters),
        help=f"Top-level parameter YAML (default: {default_parameters})",
    )
    parser.add_argument("--n-bx", type=int, default=1000, help="Number of bunch crossings")
    parser.add_argument(
        "--mu",
        type=float,
        default=200.0,
        help="Interactions per BX in fixed mode, or Poisson mean in poisson mode",
    )
    parser.add_argument(
        "--mu-mode",
        choices=("fixed", "poisson"),
        default="fixed",
        help="How to choose interactions per BX",
    )
    parser.add_argument(
        "--processes",
        default="SoftQCD:all",
        help='Pythia8 process switch to enable, e.g. "SoftQCD:all"',
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional Pythia random seed")
    parser.add_argument("--bx-offset", type=int, default=0, help="Global offset for BX IDs")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print parsed inputs and computed rows for each generated interaction",
    )
    return parser.parse_args()


def require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def parameters_from_yaml(path):
    params = load_yaml(path)
    beam = params.get("beam", {})
    pps = params.get("PPS", {})
    cms = params.get("CMS", {})
    tracks = cms.get("tracks", {})

    return {
        "sqrt_s_gev": float(beam.get("sqrt_s_gev", 14000.0)),
        "xi_ranges": pps.get("xi_ranges", {}),
        "track_pt_min": float(tracks.get("pt_min", 2.0)),
        "track_eta_max": float(tracks.get("eta_max", 2.4)),
    }


def configure_pythia(sqrt_s_gev, processes, seed=None):
    pythia = pythia8.Pythia()
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString(f"Beams:eCM = {sqrt_s_gev}")

    pythia.readString("SoftQCD:nonDiffractive      = off")
    pythia.readString("SoftQCD:elastic             = off")
    pythia.readString("SoftQCD:singleDiffractive   = off")
    pythia.readString("SoftQCD:doubleDiffractive   = off")
    pythia.readString("SoftQCD:centralDiffractive  = off")
    pythia.readString(f"{processes} = on")

    if seed is not None:
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")

    pythia.init()
    return pythia


def p4(particle):
    return vector.obj(
        px=particle["px"],
        py=particle["py"],
        pz=particle["pz"],
        E=particle["E"],
    )


def p4_values(prefix, vec, energy=None, mass=None):
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}pt": vec.pt,
        f"{prefix}eta": vec.eta,
        f"{prefix}phi": vec.phi,
        f"{prefix}E": vec.E if energy is None else energy,
        f"{prefix}m": vec.mass if mass is None else mass,
    }


def station_hits(xi, xi_ranges):
    hits = {}
    for station in STATIONS:
        bounds = xi_ranges.get(station)
        if bounds is None:
            hits[station] = 0
            continue
        xi_min, xi_max = bounds
        hits[station] = int(float(xi_min) <= xi < float(xi_max))
    return hits, sum(hits.values())


def make_tree(name, title, branch_specs):
    tree = ROOT.TTree(name, title)
    branches = {}
    leaf_types = {"i": "I", "d": "D"}
    for branch, code in branch_specs:
        branches[branch] = array(code, [0])
        tree.Branch(branch, branches[branch], f"{branch}/{leaf_types[code]}")
    return tree, branches


def set_branch_values(branches, values):
    for name, value in values.items():
        branches[name][0] = value


def build_trees():
    proton_tree, proton_br = make_tree(
        "Protons",
        "PPS-passing minbias protons",
        [
            ("bx_id", "i"),
            ("interaction_id", "i"),
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
        "Interactions",
        "Minbias interaction summaries",
        [
            ("bx_id", "i"),
            ("interaction_id", "i"),
            ("n_protons", "i"),
            ("n_pps_protons", "i"),
            ("n_l1t_tracks", "i"),
            ("sum_l1t_pt", "d"),
            ("sum_l1t_pt2", "d"),
        ],
    )

    pair_tree, pair_br = make_tree(
        "ProtonPairs",
        "Left-right PPS-passing minbias proton pairs",
        [
            ("bx_id", "i"),
            ("pair_id", "i"),
            ("interaction_id_L", "i"),
            ("interaction_id_R", "i"),
            ("proton_id_L", "i"),
            ("proton_id_R", "i"),
            ("M", "d"),
            ("y", "d"),
            ("pass_pps", "i"),
        ],
    )

    return (proton_tree, proton_br), (interaction_tree, interaction_br), (pair_tree, pair_br)


def particle_record(particle):
    return {
        "pid": int(particle.id()),
        "status": int(particle.status()),
        "px": float(particle.px()),
        "py": float(particle.py()),
        "pz": float(particle.pz()),
        "E": float(particle.e()),
        "m": float(particle.m()),
    }


def proton_row(bx_id, interaction_id, proton_id, proton, e_beam, xi_ranges):
    side = 1 if proton["pz"] >= 0.0 else -1
    xi = 1.0 - abs(proton["pz"]) / e_beam
    hits, n_hits = station_hits(xi, xi_ranges)
    row = {
        "bx_id": bx_id,
        "interaction_id": interaction_id,
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


def interaction_row(bx_id, interaction_id, n_protons, proton_rows, track_summary):
    return {
        "bx_id": bx_id,
        "interaction_id": interaction_id,
        "n_protons": n_protons,
        "n_pps_protons": len(proton_rows),
        "n_l1t_tracks": track_summary["n_tracks"],
        "sum_l1t_pt": track_summary["sum_pt"],
        "sum_l1t_pt2": track_summary["sum_pt2"],
    }


def pair_row(bx_id, pair_id, left, right, sqrt_s_gev):
    return {
        "bx_id": bx_id,
        "pair_id": pair_id,
        "interaction_id_L": left["interaction_id"],
        "interaction_id_R": right["interaction_id"],
        "proton_id_L": left["proton_id"],
        "proton_id_R": right["proton_id"],
        "M": math.sqrt(left["xi"] * right["xi"]) * sqrt_s_gev,
        "y": 0.5 * math.log(right["xi"] / left["xi"]),
        "pass_pps": 1,
    }


def selected_track(particle, track_pt_min, track_eta_max):
    if not particle.isFinal():
        return False
    if particle.charge() == 0:
        return False
    if abs(particle.id()) == 2212:
        return False
    if abs(particle.eta()) >= track_eta_max:
        return False
    return particle.pT() > track_pt_min


def track_summary(event, track_pt_min, track_eta_max):
    n_tracks = 0
    sum_pt = 0.0
    sum_pt2 = 0.0
    for idx in range(event.size()):
        particle = event[idx]
        if not selected_track(particle, track_pt_min, track_eta_max):
            continue
        pt = float(particle.pT())
        n_tracks += 1
        sum_pt += pt
        sum_pt2 += pt * pt
    return {"n_tracks": n_tracks, "sum_pt": sum_pt, "sum_pt2": sum_pt2}


def format_value(value):
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def print_mapping(title, values, indent="  "):
    print(f"{indent}{title}:")
    for key, value in values.items():
        print(f"{indent}  {key}: {format_value(value)}")


def print_particle(title, particle, indent="  "):
    fields = ("pid", "status", "px", "py", "pz", "E", "m")
    print(f"{indent}{title}:")
    print(f"{indent}  " + ", ".join(f"{field}={format_value(particle[field])}" for field in fields))


def print_generation_debug(
    bx_id,
    interaction_id,
    n_interactions,
    protons,
    proton_rows,
    interaction,
    pairs=None,
):
    print(f"\n=== bx {bx_id}, interaction {interaction_id}/{n_interactions} ===")
    for idx, proton in enumerate(protons):
        print_particle(f"input proton[{idx}]", proton)
    for idx, row in enumerate(proton_rows):
        print_mapping(f"output Protons[{idx}]", row)
    print_mapping("output Interactions", interaction)
    if pairs is not None:
        if not pairs:
            print("  output ProtonPairs: <none>")
        for idx, row in enumerate(pairs):
            print_mapping(f"output ProtonPairs[{idx}]", row)


def pairs_for_bx(bx_id, proton_rows, sqrt_s_gev):
    left = [row for row in proton_rows if row["side"] < 0 and row["xi"] > 0.0]
    right = [row for row in proton_rows if row["side"] > 0 and row["xi"] > 0.0]
    pairs = []
    pair_id = 0
    for left_row in left:
        for right_row in right:
            pairs.append(pair_row(bx_id, pair_id, left_row, right_row, sqrt_s_gev))
            pair_id += 1
    return pairs


def generate(args, params):
    require_positive("--n-bx", args.n_bx)
    require_positive("--mu", args.mu)
    if args.mu_mode == "fixed" and not float(args.mu).is_integer():
        raise ValueError("--mu must be an integer in fixed mode")
    if args.bx_offset < 0:
        raise ValueError("--bx-offset must be >= 0")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be >= 0")

    sqrt_s_gev = params["sqrt_s_gev"]
    e_beam = sqrt_s_gev / 2.0
    xi_ranges = params["xi_ranges"]
    track_pt_min = params["track_pt_min"]
    track_eta_max = params["track_eta_max"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print_mapping(
            "parameters",
            {
                "sqrt_s_gev": sqrt_s_gev,
                "track_pt_min": track_pt_min,
                "track_eta_max": track_eta_max,
                "xi_ranges": xi_ranges,
            },
        )
        print_mapping(
            "generation",
            {
                "n_bx": args.n_bx,
                "mu": args.mu,
                "mu_mode": args.mu_mode,
                "processes": args.processes,
                "seed": args.seed if args.seed is not None else "<none>",
                "bx_offset": args.bx_offset,
            },
        )

    pythia = configure_pythia(sqrt_s_gev, args.processes, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    ROOT.gROOT.SetBatch(True)
    root_file = ROOT.TFile(str(output_path), "RECREATE")
    (proton_tree, proton_br), (interaction_tree, interaction_br), (pair_tree, pair_br) = build_trees()

    counters = Counter()

    for bx_local in range(args.n_bx):
        bx_id = args.bx_offset + bx_local
        if args.mu_mode == "fixed":
            n_interactions = int(args.mu)
        else:
            n_interactions = int(rng.poisson(args.mu))

        bx_proton_rows = []
        interaction_debug = []

        for interaction_id in range(n_interactions):
            while not pythia.next():
                counters["pythia_retries"] += 1

            event = pythia.event
            protons = []
            rows = []
            proton_id = 0

            for idx in range(event.size()):
                particle = event[idx]
                if not particle.isFinal() or particle.id() != 2212:
                    continue
                proton = particle_record(particle)
                protons.append(proton)

                row = proton_row(bx_id, interaction_id, proton_id, proton, e_beam, xi_ranges)
                if row["n_station_hits"] > 0:
                    rows.append(row)
                    set_branch_values(proton_br, row)
                    proton_tree.Fill()
                    counters["protons_written"] += 1

                proton_id += 1

            tracks = track_summary(event, track_pt_min, track_eta_max)
            interaction = interaction_row(bx_id, interaction_id, len(protons), rows, tracks)
            set_branch_values(interaction_br, interaction)
            interaction_tree.Fill()

            bx_proton_rows.extend(rows)
            counters["interactions_written"] += 1
            counters["protons_seen"] += len(protons)
            counters["l1t_tracks"] += tracks["n_tracks"]
            interaction_debug.append((interaction_id, protons, rows, interaction))

        pairs = pairs_for_bx(bx_id, bx_proton_rows, sqrt_s_gev)
        for row in pairs:
            set_branch_values(pair_br, row)
            pair_tree.Fill()
            counters["pairs_written"] += 1

        if args.verbose:
            pairs_by_interaction = {}
            for row in pairs:
                pairs_by_interaction.setdefault(row["interaction_id_L"], []).append(row)
                if row["interaction_id_R"] != row["interaction_id_L"]:
                    pairs_by_interaction.setdefault(row["interaction_id_R"], []).append(row)
            for interaction_id, protons, rows, interaction in interaction_debug:
                print_generation_debug(
                    bx_id,
                    interaction_id,
                    n_interactions,
                    protons,
                    rows,
                    interaction,
                    pairs=pairs_by_interaction.get(interaction_id, []),
                )

        counters["bx_written"] += 1

    root_file.Write()
    root_file.Close()
    return counters


def main():
    args = parse_args()
    params = parameters_from_yaml(args.parameters)
    counters = generate(args, params)
    print(f"BX written: {counters['bx_written']}")
    print(f"Interactions written: {counters['interactions_written']}")
    print(f"Final-state protons seen: {counters['protons_seen']}")
    print(f"PPS protons written: {counters['protons_written']}")
    print(f"Proton pairs written: {counters['pairs_written']}")
    print(f"L1T tracks counted: {counters['l1t_tracks']}")
    if counters["pythia_retries"]:
        print(f"Pythia retries: {counters['pythia_retries']}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
