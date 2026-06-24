#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


# Per-event quantities to report, in print order. Masses and energies are in GeV,
# counts are plain multiplicities. All are derived from the Pythia HepMC record:
# the hard-process quark pair (status 23) and its descendants traced through the
# shower/hadronization tree, plus GenJets clustered from the visible descendants.
QUANTITIES = (
    ("M_cc_status23", "m_cc_status23"),
    ("M_all_terminal_descendants", "m_all_terminal"),
    ("M_visible_descendants", "m_visible"),
    ("M_all_genjets_pt1", "m_genjets_pt1"),
    ("M_all_genjets_pt15", "m_genjets_pt15"),
    ("M_two_leading_genjets", "m_two_leading_genjets"),
    ("E_neutrinos", "e_neutrinos"),
    ("N_genjets_pt1", "n_genjets_pt1"),
    ("N_genjets_pt15", "n_genjets_pt15"),
)

NEUTRINO_PDG_IDS = (12, 14, 16)
VERTEX_LIST_RE = re.compile(r"\[([^\]]*)\]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Pythia HepMC record for a process: compare the hard-process "
        "quark pair (status 23), its terminal descendants, and GenJets clustered from them."
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name to validate. May be repeated. Defaults to all processes.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Explicit Pythia HepMC file override. Requires exactly one --process.",
    )
    parser.add_argument(
        "--hard-pdg-id",
        type=int,
        default=4,
        help="Absolute PDG id of the hard-process quark pair (default 4 = charm)",
    )
    parser.add_argument("--jet-radius", type=float, default=0.4, help="anti-kt jet radius for GenJets")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
    return parser.parse_args()


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def resolve_input(process_name, campaign_name, input_arg):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    default_input = campaign_dir / "GEN-pythia" / f"{process_name}_{campaign}.hepmc"
    input_file = resolve_path(input_arg, base=ROOT) if input_arg else default_input
    return input_file, campaign


def read_hepmc_events(input_file, max_events):
    # Stream HepMC3 Asciiv3 events. Each event yields:
    #   particles: dict pid_index -> (pdg, status, px, py, pz, e)
    #   parent_vertex: dict pid_index -> production-vertex field (>0 = mother particle id,
    #                  <0 = a V vertex, 0 = beam particle with no production vertex)
    #   vertex_incoming: dict vertex_id (<0) -> list of incoming particle ids
    particles = {}
    parent_vertex = {}
    vertex_incoming = {}
    started = False
    emitted = 0

    def finished_event():
        return particles, parent_vertex, vertex_incoming

    with open(input_file, "r") as handle:
        for line in handle:
            tag = line[:2]
            if tag == "E ":
                if started:
                    yield finished_event()
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        return
                particles = {}
                parent_vertex = {}
                vertex_incoming = {}
                started = True
            elif tag == "P ":
                f = line.split()
                pid_index = int(f[1])
                particles[pid_index] = (int(f[3]), int(f[9]), float(f[4]), float(f[5]), float(f[6]), float(f[7]))
                parent_vertex[pid_index] = int(f[2])
            elif tag == "V ":
                f = line.split()
                vertex_id = int(f[1])
                match = VERTEX_LIST_RE.search(line)
                incoming = [int(x) for x in match.group(1).split(",") if x] if match else []
                vertex_incoming[vertex_id] = incoming

    if started:
        yield finished_event()


def build_children(particles, parent_vertex, vertex_incoming):
    children = {pid_index: [] for pid_index in particles}
    for pid_index in particles:
        vertex = parent_vertex[pid_index]
        if vertex > 0:
            mothers = (vertex,)
        elif vertex < 0:
            mothers = vertex_incoming.get(vertex, ())
        else:
            mothers = ()
        for mother in mothers:
            if mother in children:
                children[mother].append(pid_index)
    return children


def descendants(seeds, children):
    seen = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        for child in children[node]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def invariant_mass(np, four_vectors):
    if not four_vectors:
        return None
    arr = np.asarray(four_vectors, dtype=np.float64)
    px, py, pz, e = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum(), arr[:, 3].sum()
    return float(np.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0)))


def cluster_genjets(fastjet, jet_def, particles):
    if not particles:
        return []
    pseudojets = [fastjet.PseudoJet(px, py, pz, e) for px, py, pz, e in particles]
    sequence = fastjet.ClusterSequence(pseudojets, jet_def)
    return fastjet.sorted_by_pt(sequence.inclusive_jets(1.0))


def event_quantities(np, fastjet, jet_def, event, hard_pdg_id):
    particles, parent_vertex, vertex_incoming = event
    children = build_children(particles, parent_vertex, vertex_incoming)

    hard_seeds = [i for i, (pdg, status, *_) in particles.items() if status == 23 and abs(pdg) == hard_pdg_id]
    terminal = [
        i for i in descendants(hard_seeds, children)
        if particles[i][1] == 1
    ]

    def momenta(indices):
        return [particles[i][2:6] for i in indices]

    is_neutrino = {i for i in terminal if abs(particles[i][0]) in NEUTRINO_PDG_IDS}
    visible = [i for i in terminal if i not in is_neutrino]

    jets = cluster_genjets(fastjet, jet_def, momenta(visible))
    jets_pt1 = [j for j in jets if j.pt() > 1.0]
    jets_pt15 = [j for j in jets if j.pt() > 15.0]

    def jet_momenta(jet_list):
        return [(j.px(), j.py(), j.pz(), j.e()) for j in jet_list]

    m_cc = invariant_mass(np, momenta(hard_seeds)) if len(hard_seeds) >= 2 else None
    m_two_leading = invariant_mass(np, jet_momenta(jets_pt1[:2])) if len(jets_pt1) >= 2 else None

    return {
        "m_cc_status23": m_cc,
        "m_all_terminal": invariant_mass(np, momenta(terminal)),
        "m_visible": invariant_mass(np, momenta(visible)),
        "m_genjets_pt1": invariant_mass(np, jet_momenta(jets_pt1)),
        "m_genjets_pt15": invariant_mass(np, jet_momenta(jets_pt15)),
        "m_two_leading_genjets": m_two_leading,
        "e_neutrinos": float(sum(particles[i][5] for i in is_neutrino)),
        "n_genjets_pt1": float(len(jets_pt1)),
        "n_genjets_pt15": float(len(jets_pt15)),
    }


def summarize(np, values):
    finite = np.asarray([v for v in values if v is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite)), float(np.median(finite)), finite.size


def report(np, process_name, campaign, total_events, per_event):
    print(f"=== {process_name} (campaign {campaign}) : {total_events} event(s) ===")
    print(f"  {'quantity':<28} {'mean':>10} {'median':>10} {'events':>9}")
    for label, key in QUANTITIES:
        stats = summarize(np, [event[key] for event in per_event])
        if stats is None:
            print(f"  {label:<28} {'-':>10} {'-':>10} {0:>9}")
            continue
        mean, median, count = stats
        print(f"  {label:<28} {mean:>10.2f} {median:>10.2f} {count:>9}")
    print()


def main():
    args = parse_args()
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")
    if args.input and (not args.processes or len(args.processes) != 1):
        raise RuntimeError("--input requires exactly one --process so the input is unambiguous")

    import fastjet
    import numpy as np

    fastjet.ClusterSequence.print_banner()
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, args.jet_radius)

    processes = load_yaml(ROOT / "processes.yaml")

    for process_name in selected_processes(processes, args.processes):
        input_file, campaign = resolve_input(process_name, args.campaign, args.input)
        if not input_file.is_file():
            raise RuntimeError(f"Missing Pythia HepMC input for {process_name}: {input_file}")

        print(f"Reading {input_file}")
        per_event = [
            event_quantities(np, fastjet, jet_def, event, args.hard_pdg_id)
            for event in read_hepmc_events(input_file, args.max_events)
        ]
        report(np, process_name, campaign, len(per_event), per_event)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
