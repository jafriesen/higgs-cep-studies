#!/usr/bin/env python3
"""Diagnose final-state-radiation losses in the SuperChic H->bb samples."""

import argparse
import csv
import itertools
import math
import os
import re
import site
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "output-superchic/Hbb/Hbb__v01"
DEFAULT_NO_FSR = CAMPAIGN / "hadr-Pythia/Hbb_noFSR__v01/hepmc"
DEFAULT_FSR = CAMPAIGN / "hadr-Pythia/Hbb_FSR__v01/hepmc"
DEFAULT_LHE = CAMPAIGN / "gen-SuperChic/evrecs"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

RADII = (0.4, 0.6, 0.8, 1.0)
ADDITIONAL_JET_RADII = (0.2, 0.3)
ADDITIONAL_JET_PT_THRESHOLDS = (2.0, 5.0, 10.0)
NEUTRINOS = {12, 14, 16}
VERTEX_PARTICLES_RE = re.compile(r"\[([^]]*)\]")
TRAILING_INDEX_RE = re.compile(r"_(\d+)$")
FOUR_MOMENTUM_TOLERANCE_GEV = 1.0e-3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare H->bb Pythia HepMC samples with and without FSR, including "
            "event-record checks, truth b-jet matching, energy budgets, and plots."
        )
    )
    parser.add_argument("--no-fsr-dir", type=Path, default=DEFAULT_NO_FSR)
    parser.add_argument("--fsr-dir", type=Path, default=DEFAULT_FSR)
    parser.add_argument("--lhe-dir", type=Path, default=DEFAULT_LHE)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument(
        "--max-events",
        type=int,
        default=10000,
        help="Maximum events per dataset, accumulated over the selected files",
    )
    parser.add_argument("--constituent-pt-min", type=float, default=0.0)
    parser.add_argument("--jet-pt-min", type=float, default=20.0)
    parser.add_argument(
        "--mass-drop-fraction",
        type=float,
        default=0.10,
        help="A mean mass ratio below 1-this value is classified as substantial",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def natural_key(value):
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", str(value))]


def file_index(path):
    match = TRAILING_INDEX_RE.search(path.stem)
    return int(match.group(1)) if match else path.stem


def indexed_files(directory, patterns):
    if not directory.is_dir():
        raise RuntimeError(f"Input directory does not exist: {directory}")
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    files = sorted(set(files), key=natural_key)
    if not files:
        raise RuntimeError(f"No input files found in {directory}")
    result = {}
    for path in files:
        key = file_index(path)
        if key in result:
            raise RuntimeError(f"Duplicate file index {key!r} in {directory}")
        result[key] = path
    return result


def selected_input_files(args):
    no_fsr = indexed_files(args.no_fsr_dir.resolve(), ("*.hepmc",))
    fsr = indexed_files(args.fsr_dir.resolve(), ("*.hepmc",))
    lhe = indexed_files(args.lhe_dir.resolve(), ("*.dat", "*.lhe", "*.lhef"))
    common = sorted(set(no_fsr) & set(fsr), key=natural_key)
    if not common:
        raise RuntimeError("The FSR and no-FSR directories have no common trailing file indices")
    common = common[: args.max_files]
    missing_lhe = [key for key in common if key not in lhe]
    if missing_lhe:
        print(f"WARNING: exact LHE SCALUP is unavailable for file indices: {missing_lhe}")
    return {
        "noFSR": [(key, no_fsr[key], lhe.get(key)) for key in common],
        "FSR": [(key, fsr[key], lhe.get(key)) for key in common],
    }


def read_hepmc_events(path):
    event = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            tag = line[:2]
            if tag == "E ":
                if event is not None:
                    yield event
                fields = line.split()
                event = {
                    "number": int(fields[1]),
                    "particles": {},
                    "parent_vertex": {},
                    "vertex_incoming": {},
                    "event_attrs": {},
                    "particle_attrs": {},
                }
            elif event is None:
                continue
            elif tag == "U ":
                fields = line.split()
                if fields[1] != "GEV":
                    raise RuntimeError(f"Unsupported HepMC momentum unit {fields[1]} in {path}")
            elif tag == "A ":
                fields = line.split(maxsplit=3)
                if len(fields) < 4:
                    continue
                owner = int(fields[1])
                name = fields[2]
                value = fields[3].strip()
                if owner == 0:
                    event["event_attrs"][name] = value
                else:
                    event["particle_attrs"].setdefault(owner, {})[name] = value
            elif tag == "P ":
                fields = line.split()
                particle_id = int(fields[1])
                event["particles"][particle_id] = {
                    "pdg": int(fields[3]),
                    "p4": tuple(float(value) for value in fields[4:8]),
                    "mass": float(fields[8]),
                    "status": int(fields[9]),
                }
                event["parent_vertex"][particle_id] = int(fields[2])
            elif tag == "V ":
                fields = line.split()
                match = VERTEX_PARTICLES_RE.search(line)
                incoming = []
                if match:
                    incoming = [int(value) for value in match.group(1).split(",") if value]
                event["vertex_incoming"][int(fields[1])] = incoming
    if event is not None:
        yield event


def parse_lhe_event(lines, path):
    records = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not records:
        raise RuntimeError(f"Empty LHE event in {path}")
    header = records[0].split()
    if len(header) < 4:
        raise RuntimeError(f"Malformed LHE event header in {path}: {records[0]}")
    particle_count = int(header[0])
    particles = []
    for record in records[1 : particle_count + 1]:
        fields = record.split()
        if len(fields) < 11:
            raise RuntimeError(f"Malformed LHE particle record in {path}: {record}")
        particles.append(
            {
                "pdg": int(fields[0]),
                "status": int(fields[1]),
                "mothers": (int(fields[2]), int(fields[3])),
                "colors": (int(fields[4]), int(fields[5])),
                "p4": tuple(float(value) for value in fields[6:10]),
                "mass": float(fields[10]),
            }
        )
    if len(particles) != particle_count:
        raise RuntimeError(f"Expected {particle_count} LHE particles in {path}, found {len(particles)}")
    return {"scalup": float(header[3]), "particles": particles}


def read_lhe_events(path):
    block = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("<event"):
                block = []
            elif block is not None and line.lstrip().startswith("</event"):
                yield parse_lhe_event(block, path)
                block = None
            elif block is not None:
                block.append(line)


def mothers_of(particle_id, event):
    parent = event["parent_vertex"].get(particle_id, 0)
    if parent > 0:
        return (parent,)
    if parent < 0:
        return tuple(event["vertex_incoming"].get(parent, ()))
    return ()


def build_children(event):
    children = {particle_id: [] for particle_id in event["particles"]}
    for particle_id in event["particles"]:
        for mother_id in mothers_of(particle_id, event):
            if mother_id in children:
                children[mother_id].append(particle_id)
    return children


def descendants(seeds, children):
    found = set()
    stack = list(seeds)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def add_p4(vectors):
    if not vectors:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(sum(vector[index] for vector in vectors) for index in range(4))


def p4_mass(vector):
    px, py, pz, energy = vector
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def invariant_mass(vectors):
    return p4_mass(add_p4(vectors)) if vectors else math.nan


def p4_pt(vector):
    return math.hypot(vector[0], vector[1])


def p4_eta_phi(vector):
    px, py, pz, _ = vector
    pt = math.hypot(px, py)
    eta = math.asinh(pz / pt) if pt > 0.0 else math.copysign(math.inf, pz)
    return eta, math.atan2(py, px)


def delta_r(first, second):
    eta1, phi1 = p4_eta_phi(first)
    eta2, phi2 = p4_eta_phi(second)
    dphi = (phi1 - phi2 + math.pi) % (2.0 * math.pi) - math.pi
    return math.hypot(eta1 - eta2, dphi)


def max_p4_delta(first, second):
    return max(abs(a - b) for a, b in zip(first, second))


def float_attribute(attributes, names):
    lowered = {key.lower(): value for key, value in attributes.items()}
    for name in names:
        if name.lower() in lowered:
            try:
                return float(lowered[name.lower()].split()[0])
            except ValueError:
                pass
    return math.nan


def particle_scales(event, particle_id):
    attributes = event["particle_attrs"].get(particle_id, {})
    production = float_attribute(attributes, ("production_scale", "productionscale"))
    shower = float_attribute(attributes, ("shower_scale", "showerscale", "scale"))
    return production, shower


def select_higgs_and_bs(event, children):
    particles = event["particles"]
    higgs_candidates = [
        particle_id
        for particle_id, particle in particles.items()
        if particle["pdg"] == 25 and particle["status"] == 22
    ]
    if not higgs_candidates:
        return None, None, None, False

    for higgs_id in higgs_candidates:
        direct = children.get(higgs_id, ())
        b = [particle_id for particle_id in direct if particles[particle_id]["pdg"] == 5 and particles[particle_id]["status"] == 23]
        bbar = [particle_id for particle_id in direct if particles[particle_id]["pdg"] == -5 and particles[particle_id]["status"] == 23]
        if len(b) == 1 and len(bbar) == 1:
            links_valid = mothers_of(b[0], event) == (higgs_id,) and mothers_of(bbar[0], event) == (higgs_id,)
            return higgs_id, b[0], bbar[0], links_valid

    higgs_id = higgs_candidates[0]
    global_b = [particle_id for particle_id, particle in particles.items() if particle["pdg"] == 5 and particle["status"] == 23]
    global_bbar = [particle_id for particle_id, particle in particles.items() if particle["pdg"] == -5 and particle["status"] == 23]
    return higgs_id, global_b[0] if len(global_b) == 1 else None, global_bbar[0] if len(global_bbar) == 1 else None, False


def color_connection_valid(event, b_id, bbar_id):
    if b_id is None or bbar_id is None:
        return False
    b_attrs = event["particle_attrs"].get(b_id, {})
    bbar_attrs = event["particle_attrs"].get(bbar_id, {})
    try:
        b_color = int(b_attrs.get("flow1", "0"))
        b_anticolor = int(b_attrs.get("flow2", "0"))
        bbar_color = int(bbar_attrs.get("flow1", "0"))
        bbar_anticolor = int(bbar_attrs.get("flow2", "0"))
    except ValueError:
        return False
    return b_color != 0 and b_color == bbar_anticolor and b_anticolor == 0 and bbar_color == 0


def is_bottom_hadron(pdg_id):
    value = abs(pdg_id)
    if value < 100:
        return False
    first_quark = (value // 100) % 10
    second_quark = (value // 10) % 10
    third_quark = (value // 1000) % 10
    if first_quark == second_quark == 5:
        return False  # bottomonium is not a flavor tag for either hard b leg
    return first_quark == 5 or third_quark == 5


def weak_bottom_hadrons(event, higgs_descendants, children):
    particles = event["particles"]
    candidates = [particle_id for particle_id in higgs_descendants if is_bottom_hadron(particles[particle_id]["pdg"])]
    result = []
    for particle_id in candidates:
        later = descendants((particle_id,), children)
        if not any(is_bottom_hadron(particles[child]["pdg"]) for child in later):
            result.append(particle_id)
    return result


def assign_bottom_hadrons(event, b_id, bbar_id, candidates):
    if b_id is None or bbar_id is None or len(candidates) < 2:
        return None
    particles = event["particles"]
    best = None
    for first, second in itertools.permutations(candidates, 2):
        score = delta_r(particles[b_id]["p4"], particles[first]["p4"]) + delta_r(
            particles[bbar_id]["p4"], particles[second]["p4"]
        )
        tie_break = -(particles[first]["p4"][3] + particles[second]["p4"][3])
        candidate = (score, tie_break, first, second)
        if best is None or candidate < best:
            best = candidate
    return (best[2], best[3])


def outgoing_intact_protons(event, higgs_descendants):
    particles = event["particles"]
    candidates = [
        particle_id
        for particle_id, particle in particles.items()
        if particle["status"] == 1
        and particle["pdg"] == 2212
        and particle_id not in higgs_descendants
        and particle["p4"][3] > 100.0
        and abs(particle["p4"][2]) > 0.5 * particle["p4"][3]
    ]
    positive = [particle_id for particle_id in candidates if particles[particle_id]["p4"][2] > 0.0]
    negative = [particle_id for particle_id in candidates if particles[particle_id]["p4"][2] < 0.0]
    selected = set()
    if positive:
        selected.add(max(positive, key=lambda particle_id: particles[particle_id]["p4"][2]))
    if negative:
        selected.add(min(negative, key=lambda particle_id: particles[particle_id]["p4"][2]))
    return selected


def make_pseudojet(fastjet, vector, user_index):
    pseudojet = fastjet.PseudoJet(*vector)
    pseudojet.set_user_index(user_index)
    return pseudojet


def make_ghost(fastjet, vector, user_index):
    pt = p4_pt(vector)
    scale = 1.0e-20 / pt if pt > 0.0 else 1.0e-20
    return make_pseudojet(fastjet, tuple(component * scale for component in vector), user_index)


def cluster_jets(fastjet, particles, particle_ids, bottom_hadron_ids, radius):
    inputs = [make_pseudojet(fastjet, particles[particle_id]["p4"], particle_id) for particle_id in particle_ids]
    ghost_tags = (-1000001, -1000002)
    if bottom_hadron_ids is not None:
        for tag, particle_id in zip(ghost_tags, bottom_hadron_ids):
            inputs.append(make_ghost(fastjet, particles[particle_id]["p4"], tag))
    if not inputs:
        return []
    sequence = fastjet.ClusterSequence(inputs, fastjet.JetDefinition(fastjet.antikt_algorithm, radius))
    result = []
    for jet in fastjet.sorted_by_pt(sequence.inclusive_jets(0.0)):
        constituents = jet.constituents()
        real_ids = {constituent.user_index() for constituent in constituents if constituent.user_index() > 0}
        if not real_ids:
            continue
        result.append(
            {
                "p4": (jet.px(), jet.py(), jet.pz(), jet.e()),
                "pt": jet.pt(),
                "real_ids": real_ids,
                "ghosts": {constituent.user_index() for constituent in constituents if constituent.user_index() < 0},
            }
        )
    return result


def add_muons_to_jets(particles, jets, muon_ids, radius):
    corrected = [
        {
            "p4": jet["p4"],
            "pt": jet["pt"],
            "real_ids": set(jet["real_ids"]),
            "ghosts": set(jet["ghosts"]),
        }
        for jet in jets
    ]
    reference_axes = [jet["p4"] for jet in jets]
    added_energy = 0.0
    added_count = 0
    for particle_id in muon_ids:
        if not reference_axes:
            break
        muon_p4 = particles[particle_id]["p4"]
        distances = [delta_r(muon_p4, jet_p4) for jet_p4 in reference_axes]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        if distances[nearest] >= radius:
            continue
        corrected[nearest]["p4"] = add_p4((corrected[nearest]["p4"], muon_p4))
        corrected[nearest]["pt"] = p4_pt(corrected[nearest]["p4"])
        corrected[nearest]["real_ids"].add(particle_id)
        added_energy += muon_p4[3]
        added_count += 1
    corrected.sort(key=lambda jet: jet["pt"], reverse=True)
    return corrected, added_energy, added_count


def matched_jet_indices(jets):
    found = []
    for tag in (-1000001, -1000002):
        indices = [index for index, jet in enumerate(jets) if tag in jet["ghosts"]]
        if len(indices) != 1:
            return None
        found.append(indices[0])
    return tuple(found) if found[0] != found[1] else None


def pair_mass(jets, indices):
    if indices is None or len(indices) != 2:
        return math.nan
    return invariant_mass([jets[index]["p4"] for index in indices])


def all_jet_mass(jets, indices=None):
    selected = jets if indices is None else [jets[index] for index in indices]
    return invariant_mass([jet["p4"] for jet in selected])


def matched_plus_additional_indices(jets, matched, pt_min):
    if matched is None:
        return None
    selected = set(matched)
    selected.update(
        index for index, jet in enumerate(jets)
        if index not in selected and jet["pt"] >= pt_min
    )
    return sorted(selected)


def lhe_alignment_delta(lhe_event, event, higgs_id, b_id, bbar_id):
    if lhe_event is None or higgs_id is None or b_id is None or bbar_id is None:
        return math.nan
    particles = event["particles"]
    deltas = []
    for pdg_id, hepmc_id in ((25, higgs_id), (5, b_id), (-5, bbar_id)):
        candidates = [particle for particle in lhe_event["particles"] if particle["pdg"] == pdg_id]
        if len(candidates) != 1:
            return math.inf
        deltas.append(max_p4_delta(candidates[0]["p4"], particles[hepmc_id]["p4"]))
    return max(deltas)


def analyze_event(fastjet, dataset, file_name, ordinal, event, lhe_event, args):
    particles = event["particles"]
    children = build_children(event)
    higgs_id, b_id, bbar_id, daughter_links_valid = select_higgs_and_bs(event, children)
    higgs_descendants = descendants((higgs_id,), children) if higgs_id is not None else set()
    higgs_terminal = [particle_id for particle_id in higgs_descendants if particles[particle_id]["status"] == 1]
    higgs_visible = [particle_id for particle_id in higgs_terminal if abs(particles[particle_id]["pdg"]) not in NEUTRINOS]

    intact_protons = outgoing_intact_protons(event, higgs_descendants)
    central_final = [
        particle_id
        for particle_id, particle in particles.items()
        if particle["status"] == 1 and particle_id not in intact_protons
    ]
    central_visible = [particle_id for particle_id in central_final if abs(particles[particle_id]["pdg"]) not in NEUTRINOS]
    filtered_visible = [
        particle_id
        for particle_id in central_visible
        if p4_pt(particles[particle_id]["p4"]) >= args.constituent_pt_min
    ]
    central_muons = [particle_id for particle_id in central_visible if abs(particles[particle_id]["pdg"]) == 13]
    central_muon_set = set(central_muons)
    filtered_visible_set = set(filtered_visible)
    central_without_muons = [particle_id for particle_id in central_visible if particle_id not in central_muon_set]
    filtered_without_muons = [particle_id for particle_id in filtered_visible if particle_id not in central_muon_set]
    filtered_higgs_visible = [particle_id for particle_id in higgs_visible if particle_id in filtered_visible_set]

    higgs_p4 = particles[higgs_id]["p4"] if higgs_id is not None else None
    b_p4 = particles[b_id]["p4"] if b_id is not None else None
    bbar_p4 = particles[bbar_id]["p4"] if bbar_id is not None else None
    hard_pair = [vector for vector in (b_p4, bbar_p4) if vector is not None]

    higgs_closure = max_p4_delta(add_p4([particles[particle_id]["p4"] for particle_id in higgs_terminal]), higgs_p4) if higgs_p4 else math.inf
    central_closure = max_p4_delta(add_p4([particles[particle_id]["p4"] for particle_id in central_final]), higgs_p4) if higgs_p4 else math.inf
    closure_delta = max(higgs_closure, central_closure)
    alignment_delta = lhe_alignment_delta(lhe_event, event, higgs_id, b_id, bbar_id)

    bottom_hadrons = weak_bottom_hadrons(event, higgs_descendants, children)
    assigned_hadrons = assign_bottom_hadrons(event, b_id, bbar_id, bottom_hadrons)
    color_valid = color_connection_valid(event, b_id, bbar_id)
    event_scale = float_attribute(event["event_attrs"], ("event_scale",))
    b_production, b_shower = particle_scales(event, b_id) if b_id is not None else (math.nan, math.nan)
    bbar_production, bbar_shower = particle_scales(event, bbar_id) if bbar_id is not None else (math.nan, math.nan)

    metrics = {
        "dataset": dataset,
        "file": file_name,
        "event_number": event["number"],
        "event_ordinal": ordinal,
        "scalup": lhe_event["scalup"] if lhe_event is not None else math.nan,
        "event_scale": event_scale,
        "m_h": p4_mass(higgs_p4) if higgs_p4 else math.nan,
        "m_bb": invariant_mass(hard_pair) if len(hard_pair) == 2 else math.nan,
        "b_production_scale": b_production,
        "b_shower_scale": b_shower,
        "bbar_production_scale": bbar_production,
        "bbar_shower_scale": bbar_shower,
        "m_central_all": invariant_mass([particles[particle_id]["p4"] for particle_id in central_final]),
        "m_central_visible": invariant_mass([particles[particle_id]["p4"] for particle_id in central_visible]),
        "m_higgs_all": invariant_mass([particles[particle_id]["p4"] for particle_id in higgs_terminal]),
        "m_higgs_visible": invariant_mass([particles[particle_id]["p4"] for particle_id in higgs_visible]),
        "m_higgs_visible_filtered": invariant_mass([particles[particle_id]["p4"] for particle_id in filtered_higgs_visible]),
        "e_neutrinos": sum(particles[particle_id]["p4"][3] for particle_id in higgs_terminal if abs(particles[particle_id]["pdg"]) in NEUTRINOS),
        "e_muons": sum(particles[particle_id]["p4"][3] for particle_id in higgs_terminal if abs(particles[particle_id]["pdg"]) == 13),
        "e_higgs_visible": sum(particles[particle_id]["p4"][3] for particle_id in higgs_visible),
        "n_bottom_hadrons": len(bottom_hadrons),
        "n_intact_protons_removed": len(intact_protons),
        "closure_delta": closure_delta,
        "lhe_alignment_delta": alignment_delta,
        "missing_higgs": int(higgs_id is None),
        "invalid_daughter_links": int(not daughter_links_valid),
        "invalid_color": int(not color_valid),
        "momentum_nonclosure": int(not math.isfinite(closure_delta) or closure_delta > FOUR_MOMENTUM_TOLERANCE_GEV),
        "lhe_mismatch": int(math.isfinite(alignment_delta) and alignment_delta > FOUR_MOMENTUM_TOLERANCE_GEV),
        "lhe_unavailable": int(lhe_event is None),
    }

    flags = []
    for flag in ("missing_higgs", "invalid_daughter_links", "invalid_color", "momentum_nonclosure", "lhe_mismatch", "lhe_unavailable"):
        if metrics[flag]:
            flags.append(flag)

    for radius in RADII:
        suffix = f"r{int(round(radius * 10)):02d}"
        raw_jets = cluster_jets(fastjet, particles, central_visible, assigned_hadrons, radius)
        normal_jets = cluster_jets(fastjet, particles, filtered_visible, assigned_hadrons, radius)
        no_muon_jets = cluster_jets(fastjet, particles, central_without_muons, assigned_hadrons, radius)
        filtered_no_muon_jets = cluster_jets(fastjet, particles, filtered_without_muons, assigned_hadrons, radius)
        muon_added_jets, muon_added_energy, muon_added_count = add_muons_to_jets(
            particles, no_muon_jets, central_muons, radius
        )
        filtered_muon_added_jets, _, _ = add_muons_to_jets(
            particles, filtered_no_muon_jets, central_muons, radius
        )
        raw_matched = matched_jet_indices(raw_jets)
        normal_matched = matched_jet_indices(normal_jets)
        no_muon_matched = matched_jet_indices(no_muon_jets)
        muon_added_matched = matched_jet_indices(muon_added_jets)
        filtered_no_muon_matched = matched_jet_indices(filtered_no_muon_jets)
        filtered_muon_added_matched = matched_jet_indices(filtered_muon_added_jets)
        raw_leading = (0, 1) if len(raw_jets) >= 2 else None
        no_muon_leading = (0, 1) if len(no_muon_jets) >= 2 else None
        muon_added_leading = (0, 1) if len(muon_added_jets) >= 2 else None
        thresholded = [index for index, jet in enumerate(normal_jets) if jet["pt"] >= args.jet_pt_min]
        thresholded_no_muon = [
            index for index, jet in enumerate(filtered_no_muon_jets) if jet["pt"] >= args.jet_pt_min
        ]
        thresholded_muon_added = [
            index for index, jet in enumerate(filtered_muon_added_jets) if jet["pt"] >= args.jet_pt_min
        ]
        thresholded_leading = tuple(thresholded[:2]) if len(thresholded) >= 2 else None
        thresholded_matched = normal_matched if normal_matched is not None and all(normal_jets[index]["pt"] >= args.jet_pt_min for index in normal_matched) else None
        thresholded_no_muon_leading = tuple(thresholded_no_muon[:2]) if len(thresholded_no_muon) >= 2 else None
        thresholded_muon_added_leading = tuple(thresholded_muon_added[:2]) if len(thresholded_muon_added) >= 2 else None
        thresholded_no_muon_matched = (
            filtered_no_muon_matched
            if filtered_no_muon_matched is not None
            and all(filtered_no_muon_jets[index]["pt"] >= args.jet_pt_min for index in filtered_no_muon_matched)
            else None
        )
        thresholded_muon_added_matched = (
            filtered_muon_added_matched
            if filtered_muon_added_matched is not None
            and all(
                filtered_muon_added_jets[index]["pt"] >= args.jet_pt_min
                for index in filtered_muon_added_matched
            )
            else None
        )

        matched_real_ids = set()
        if raw_matched is not None:
            for index in raw_matched:
                matched_real_ids.update(raw_jets[index]["real_ids"])
        unmatched_visible = [
            particle_id for particle_id in central_visible
            if particle_id not in matched_real_ids
        ]
        outside_ids = set(higgs_visible) - matched_real_ids if raw_matched is not None else set()
        below_ids = set()
        for jet in normal_jets:
            if jet["pt"] < args.jet_pt_min:
                below_ids.update(jet["real_ids"] & set(higgs_visible))

        metrics[f"n_jets_{suffix}"] = len(raw_jets)
        metrics[f"n_jets_thresholded_{suffix}"] = len(thresholded)
        metrics[f"m_all_jets_{suffix}"] = all_jet_mass(raw_jets)
        metrics[f"m_all_jets_thresholded_{suffix}"] = all_jet_mass(normal_jets, thresholded)
        metrics[f"m_leading_{suffix}"] = pair_mass(raw_jets, raw_leading)
        metrics[f"m_bmatched_{suffix}"] = pair_mass(raw_jets, raw_matched)
        for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
            pt_label = f"pt{int(pt_min)}"
            selected = matched_plus_additional_indices(raw_jets, raw_matched, pt_min)
            metrics[f"m_bmatched_plus_jets_{pt_label}_{suffix}"] = all_jet_mass(raw_jets, selected) if selected is not None else math.nan
            metrics[f"n_additional_jets_{pt_label}_{suffix}"] = len(selected) - 2 if selected is not None else 0
        for additional_radius in ADDITIONAL_JET_RADII:
            additional_radius_label = f"radd{int(round(additional_radius * 10)):02d}"
            additional_jets = (
                cluster_jets(
                    fastjet,
                    particles,
                    unmatched_visible,
                    None,
                    additional_radius,
                )
                if raw_matched is not None
                else []
            )
            for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
                pt_label = f"pt{int(pt_min)}"
                selected_additional = [
                    jet for jet in additional_jets if jet["pt"] >= pt_min
                ]
                vectors = (
                    [raw_jets[index]["p4"] for index in raw_matched]
                    + [jet["p4"] for jet in selected_additional]
                    if raw_matched is not None
                    else []
                )
                key = f"{additional_radius_label}_{pt_label}_{suffix}"
                metrics[f"m_bmatched_plus_jets_{key}"] = (
                    invariant_mass(vectors) if raw_matched is not None else math.nan
                )
                metrics[f"n_additional_jets_{key}"] = (
                    len(selected_additional) if raw_matched is not None else 0
                )
        metrics[f"m_leading_thresholded_{suffix}"] = pair_mass(normal_jets, thresholded_leading)
        metrics[f"m_bmatched_thresholded_{suffix}"] = pair_mass(normal_jets, thresholded_matched)
        metrics[f"m_all_jets_no_muons_{suffix}"] = all_jet_mass(no_muon_jets)
        metrics[f"m_leading_no_muons_{suffix}"] = pair_mass(no_muon_jets, no_muon_leading)
        metrics[f"m_bmatched_no_muons_{suffix}"] = pair_mass(no_muon_jets, no_muon_matched)
        metrics[f"m_all_jets_muon_added_{suffix}"] = all_jet_mass(muon_added_jets)
        metrics[f"m_leading_muon_added_{suffix}"] = pair_mass(muon_added_jets, muon_added_leading)
        metrics[f"m_bmatched_muon_added_{suffix}"] = pair_mass(muon_added_jets, muon_added_matched)
        metrics[f"m_leading_no_muons_thresholded_{suffix}"] = pair_mass(
            filtered_no_muon_jets, thresholded_no_muon_leading
        )
        metrics[f"m_bmatched_no_muons_thresholded_{suffix}"] = pair_mass(
            filtered_no_muon_jets, thresholded_no_muon_matched
        )
        metrics[f"m_leading_muon_added_thresholded_{suffix}"] = pair_mass(
            filtered_muon_added_jets, thresholded_muon_added_leading
        )
        metrics[f"m_bmatched_muon_added_thresholded_{suffix}"] = pair_mass(
            filtered_muon_added_jets, thresholded_muon_added_matched
        )
        metrics[f"e_muons_added_{suffix}"] = muon_added_energy
        metrics[f"n_muons_added_{suffix}"] = muon_added_count
        metrics[f"e_outside_bmatched_{suffix}"] = sum(particles[particle_id]["p4"][3] for particle_id in outside_ids) if raw_matched is not None else math.nan
        metrics[f"out_of_cone_fraction_{suffix}"] = (
            metrics[f"e_outside_bmatched_{suffix}"] / metrics["e_higgs_visible"]
            if metrics["e_higgs_visible"] > 0.0 and raw_matched is not None
            else math.nan
        )
        metrics[f"e_below_jet_threshold_{suffix}"] = sum(particles[particle_id]["p4"][3] for particle_id in below_ids)
        metrics[f"response_b_{suffix}"] = raw_jets[raw_matched[0]]["p4"][3] / b_p4[3] if raw_matched is not None and b_p4 else math.nan
        metrics[f"response_bbar_{suffix}"] = raw_jets[raw_matched[1]]["p4"][3] / bbar_p4[3] if raw_matched is not None and bbar_p4 else math.nan
        metrics[f"bmatch_missing_{suffix}"] = int(raw_matched is None)
        metrics[f"leading_not_bmatched_{suffix}"] = int(raw_matched is not None and set(raw_matched) != {0, 1})
        metrics[f"thresholded_bmatch_missing_{suffix}"] = int(thresholded_matched is None)
        metrics[f"thresholded_leading_not_bmatched_{suffix}"] = int(
            thresholded_matched is not None and thresholded_leading is not None and set(thresholded_matched) != set(thresholded_leading)
        )
        if metrics[f"bmatch_missing_{suffix}"]:
            flags.append(f"bmatch_missing_{suffix}")
        if metrics[f"leading_not_bmatched_{suffix}"]:
            flags.append(f"leading_not_bmatched_{suffix}")

    metrics["malformed_record"] = int(
        metrics["missing_higgs"]
        or metrics["invalid_daughter_links"]
        or metrics["invalid_color"]
        or metrics["momentum_nonclosure"]
        or metrics["lhe_mismatch"]
    )
    metrics["flags"] = "|".join(flags) if flags else "OK"
    return metrics


def format_number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "N/A"


def print_event(metrics):
    print(
        f"{metrics['dataset']:<5} {metrics['file']}:{metrics['event_number']:04d} "
        f"SCALUP={format_number(metrics['scalup'])} mH={format_number(metrics['m_h'])} "
        f"mbb={format_number(metrics['m_bb'])} "
        f"b(prod/shower)={format_number(metrics['b_production_scale'])}/{format_number(metrics['b_shower_scale'])} "
        f"bbar(prod/shower)={format_number(metrics['bbar_production_scale'])}/{format_number(metrics['bbar_shower_scale'])} "
        f"flags={metrics['flags']}"
    )


def analyze_dataset(fastjet, dataset, files, args):
    results = []
    for _, hepmc_path, lhe_path in files:
        if len(results) >= args.max_events:
            break
        print(f"Reading {dataset}: {hepmc_path}")
        lhe_iterator = iter(read_lhe_events(lhe_path)) if lhe_path is not None else None
        for event in read_hepmc_events(hepmc_path):
            if len(results) >= args.max_events:
                break
            lhe_event = next(lhe_iterator, None) if lhe_iterator is not None else None
            metrics = analyze_event(
                fastjet,
                dataset,
                hepmc_path.name,
                len(results) + 1,
                event,
                lhe_event,
                args,
            )
            results.append(metrics)
            print_event(metrics)
    if not results:
        raise RuntimeError(f"No events read for {dataset}")
    return results


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_values(np, rows, key, ratio_to=None):
    values = []
    for row in rows:
        value = row.get(key, math.nan)
        if ratio_to is not None:
            denominator = row.get(ratio_to, math.nan)
            value = value / denominator if math.isfinite(value) and math.isfinite(denominator) and denominator != 0.0 else math.nan
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def mean_ratio(np, rows, key):
    values = finite_values(np, rows, key, ratio_to="m_h")
    return float(np.mean(values)) if values.size else math.nan


def make_summary(np, by_dataset, args):
    threshold = 1.0 - args.mass_drop_fraction
    summaries = []
    for dataset, rows in by_dataset.items():
        malformed_fraction = float(np.mean([row["malformed_record"] for row in rows]))
        for radius in RADII:
            suffix = f"r{int(round(radius * 10)):02d}"
            ratios = {
                "hard_bb_ratio": mean_ratio(np, rows, "m_bb"),
                "all_final_ratio": mean_ratio(np, rows, "m_higgs_all"),
                "visible_ratio": mean_ratio(np, rows, "m_higgs_visible"),
                "filtered_visible_ratio": mean_ratio(np, rows, "m_higgs_visible_filtered"),
                "all_jets_ratio": mean_ratio(np, rows, f"m_all_jets_{suffix}"),
                "thresholded_all_jets_ratio": mean_ratio(np, rows, f"m_all_jets_thresholded_{suffix}"),
                "bmatched_ratio": mean_ratio(np, rows, f"m_bmatched_{suffix}"),
                "leading_ratio": mean_ratio(np, rows, f"m_leading_{suffix}"),
                "no_muon_bmatched_ratio": mean_ratio(np, rows, f"m_bmatched_no_muons_{suffix}"),
                "muon_added_bmatched_ratio": mean_ratio(np, rows, f"m_bmatched_muon_added_{suffix}"),
            }
            ratios["muon_added_mass_gain"] = (
                ratios["muon_added_bmatched_ratio"] - ratios["no_muon_bmatched_ratio"]
            )
            for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
                pt_label = f"pt{int(pt_min)}"
                ratios[f"bmatched_plus_jets_{pt_label}_ratio"] = mean_ratio(
                    np, rows, f"m_bmatched_plus_jets_{pt_label}_{suffix}"
                )
                ratios[f"bmatched_plus_jets_{pt_label}_gain"] = (
                    ratios[f"bmatched_plus_jets_{pt_label}_ratio"] - ratios["bmatched_ratio"]
                )
            for additional_radius in ADDITIONAL_JET_RADII:
                additional_radius_label = f"radd{int(round(additional_radius * 10)):02d}"
                for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
                    pt_label = f"pt{int(pt_min)}"
                    key = f"bmatched_plus_jets_{additional_radius_label}_{pt_label}"
                    ratios[f"{key}_ratio"] = mean_ratio(
                        np, rows, f"m_{key}_{suffix}"
                    )
                    ratios[f"{key}_gain"] = ratios[f"{key}_ratio"] - ratios["bmatched_ratio"]
            first_drop = "none"
            cause = "none"
            if malformed_fraction >= args.mass_drop_fraction:
                first_drop, cause = "event flags", "malformed record"
            else:
                stages = (
                    ("hard bb", "hard_bb_ratio", "malformed record"),
                    ("all Higgs final", "all_final_ratio", "malformed record"),
                    ("visible", "visible_ratio", "neutrinos"),
                    ("constituent-filtered", "filtered_visible_ratio", "constituent filtering"),
                    ("all jets", "all_jets_ratio", "malformed record"),
                    ("b-matched dijet", "bmatched_ratio", "FSR/out-of-cone"),
                    ("leading dijet", "leading_ratio", "incorrect pairing"),
                    ("jets above threshold", "thresholded_all_jets_ratio", "below-threshold jets"),
                )
                for label, key, stage_cause in stages:
                    value = ratios[key]
                    if math.isfinite(value) and value < threshold:
                        first_drop, cause = label, stage_cause
                        break
            summary = {
                "dataset": dataset,
                "radius": radius,
                **ratios,
                "leading_mismatch_fraction": float(np.mean([row[f"leading_not_bmatched_{suffix}"] for row in rows])),
                "malformed_fraction": malformed_fraction,
                "first_substantial_drop": first_drop,
                "attribution": cause,
            }
            summaries.append(summary)
    return summaries


def print_summary(summaries, args):
    print()
    print(f"Mean mass ladder; substantial means m/mH < {1.0 - args.mass_drop_fraction:.2f}")
    print(
        f"{'sample':<6} {'R':>3} {'hard':>6} {'final':>6} {'vis':>6} {'filt':>6} "
        f"{'all':>6} {'all>cut':>7} {'bmatch':>7} {'lead':>6} {'mismatch':>9} {'bad':>6}  "
        "first drop / attribution"
    )
    for row in summaries:
        print(
            f"{row['dataset']:<6} {row['radius']:>3.1f} "
            f"{format_number(row['hard_bb_ratio'], 3):>6} {format_number(row['all_final_ratio'], 3):>6} "
            f"{format_number(row['visible_ratio'], 3):>6} {format_number(row['filtered_visible_ratio'], 3):>6} "
            f"{format_number(row['all_jets_ratio'], 3):>6} {format_number(row['thresholded_all_jets_ratio'], 3):>7} "
            f"{format_number(row['bmatched_ratio'], 3):>7} {format_number(row['leading_ratio'], 3):>6} "
            f"{format_number(row['leading_mismatch_fraction'], 3):>9} {format_number(row['malformed_fraction'], 3):>6}  "
            f"{row['first_substantial_drop']} / {row['attribution']}"
        )
    print()
    print("Post-clustering muon-addition layer")
    print(f"{'sample':<6} {'R':>3} {'no muons':>10} {'+ muons':>10} {'gain':>8}")
    for row in summaries:
        print(
            f"{row['dataset']:<6} {row['radius']:>3.1f} "
            f"{format_number(row['no_muon_bmatched_ratio'], 3):>10} "
            f"{format_number(row['muon_added_bmatched_ratio'], 3):>10} "
            f"{format_number(row['muon_added_mass_gain'], 3):>8}"
        )
    print()
    print("Two b-matched jets plus additional jets clustered with the same R")
    print(f"{'sample':<6} {'R':>3} {'bmatch':>7} {'+jets>2':>8} {'+jets>5':>8} {'+jets>10':>9}")
    for row in summaries:
        print(
            f"{row['dataset']:<6} {row['radius']:>3.1f} "
            f"{format_number(row['bmatched_ratio'], 3):>7} "
            f"{format_number(row['bmatched_plus_jets_pt2_ratio'], 3):>8} "
            f"{format_number(row['bmatched_plus_jets_pt5_ratio'], 3):>8} "
            f"{format_number(row['bmatched_plus_jets_pt10_ratio'], 3):>9}"
        )
    for additional_radius in ADDITIONAL_JET_RADII:
        additional_radius_label = f"radd{int(round(additional_radius * 10)):02d}"
        print()
        print(
            "Two b-matched jets plus disjoint additional jets clustered with "
            f"R={additional_radius:.1f}"
        )
        print(f"{'sample':<6} {'R':>3} {'bmatch':>7} {'+jets>2':>8} {'+jets>5':>8} {'+jets>10':>9}")
        for row in summaries:
            print(
                f"{row['dataset']:<6} {row['radius']:>3.1f} "
                f"{format_number(row['bmatched_ratio'], 3):>7} "
                f"{format_number(row[f'bmatched_plus_jets_{additional_radius_label}_pt2_ratio'], 3):>8} "
                f"{format_number(row[f'bmatched_plus_jets_{additional_radius_label}_pt5_ratio'], 3):>8} "
                f"{format_number(row[f'bmatched_plus_jets_{additional_radius_label}_pt10_ratio'], 3):>9}"
            )


def plot_histograms(np, plt, axes, by_dataset, key, bins, xlabel, ratio_to=None):
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    for dataset, rows in by_dataset.items():
        values = finite_values(np, rows, key, ratio_to=ratio_to)
        if values.size:
            axes.hist(values, bins=bins, histtype="step", density=True, linewidth=1.8, label=dataset, color=colors[dataset])
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Normalized events")
    axes.legend(frameon=False)


def make_dijet_plots(np, plt, by_dataset, output_dir, args):
    configurations = (
        ("m_leading", "raw_leading", "Two leading jets"),
        ("m_bmatched", "raw_bmatched", "Two b-hadron-matched jets"),
        ("m_leading_thresholded", "thresholded_leading", f"Two leading jets, pT >= {args.jet_pt_min:g} GeV"),
        ("m_bmatched_thresholded", "thresholded_bmatched", f"Two b-matched jets, pT >= {args.jet_pt_min:g} GeV"),
        ("m_bmatched_no_muons", "raw_bmatched_no_muons", "Two b-matched jets, muons excluded"),
        ("m_bmatched_muon_added", "raw_bmatched_muon_added", "Two b-matched jets, muons added after clustering"),
        (
            "m_bmatched_no_muons_thresholded",
            "thresholded_bmatched_no_muons",
            f"Two b-matched jets without muons, pT >= {args.jet_pt_min:g} GeV",
        ),
        (
            "m_bmatched_muon_added_thresholded",
            "thresholded_bmatched_muon_added",
            f"Two b-matched jets with added muons, pT >= {args.jet_pt_min:g} GeV",
        ),
    )
    bins = np.linspace(0.0, 160.0, 65)
    for prefix, filename, title in configurations:
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        for axis, radius in zip(axes.flat, RADII):
            suffix = f"r{int(round(radius * 10)):02d}"
            plot_histograms(np, plt, axis, by_dataset, f"{prefix}_{suffix}", bins, "Dijet mass [GeV]")
            axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
            axis.set_title(f"anti-kt R={radius:.1f}")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_dir / f"dijet_mass_{filename}.png", dpi=160)
        plt.close(figure)


def make_additional_jet_plots(np, plt, by_dataset, output_dir):
    bins = np.linspace(0.0, 160.0, 65)
    for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
        pt_label = f"pt{int(pt_min)}"
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        for axis, radius in zip(axes.flat, RADII):
            suffix = f"r{int(round(radius * 10)):02d}"
            plot_histograms(
                np,
                plt,
                axis,
                by_dataset,
                f"m_bmatched_plus_jets_{pt_label}_{suffix}",
                bins,
                "Matched jets + additional-jet mass [GeV]",
            )
            axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
            axis.set_title(f"anti-kt R={radius:.1f}")
        figure.suptitle(
            f"Two b-matched jets + other jets with pT >= {pt_min:g} GeV"
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"mass_bmatched_plus_jets_{pt_label}.png", dpi=160)
        plt.close(figure)

    for additional_radius in ADDITIONAL_JET_RADII:
        additional_radius_label = f"radd{int(round(additional_radius * 10)):02d}"
        for pt_min in ADDITIONAL_JET_PT_THRESHOLDS:
            pt_label = f"pt{int(pt_min)}"
            figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
            for axis, radius in zip(axes.flat, RADII):
                suffix = f"r{int(round(radius * 10)):02d}"
                plot_histograms(
                    np,
                    plt,
                    axis,
                    by_dataset,
                    f"m_bmatched_plus_jets_{additional_radius_label}_{pt_label}_{suffix}",
                    bins,
                    "Matched jets + small-R additional-jet mass [GeV]",
                )
                axis.axvline(125.0, color="black", linestyle=":", linewidth=1.0)
                axis.set_title(f"main anti-kt R={radius:.1f}")
            figure.suptitle(
                f"Two b-matched jets + disjoint R={additional_radius:.1f} jets "
                f"with pT >= {pt_min:g} GeV"
            )
            figure.tight_layout()
            figure.savefig(
                output_dir
                / f"mass_bmatched_plus_jets_{additional_radius_label}_{pt_label}.png",
                dpi=160,
            )
            plt.close(figure)


def make_diagnostic_plots(np, plt, by_dataset, output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    mass_panels = (
        ("m_bb", "Hard b pair"),
        ("m_higgs_visible", "Higgs-origin visible"),
        ("m_bmatched_r04", "b-matched dijet, R=0.4"),
        ("m_leading_r04", "leading dijet, R=0.4"),
    )
    for axis, (key, title) in zip(axes.flat, mass_panels):
        plot_histograms(np, plt, axis, by_dataset, key, np.linspace(0.4, 1.1, 57), "m / mH", ratio_to="m_h")
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_dir / "mass_ratio_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, radius in zip(axes.flat, RADII):
        suffix = f"r{int(round(radius * 10)):02d}"
        colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
        for dataset, rows in by_dataset.items():
            values = np.concatenate(
                (finite_values(np, rows, f"response_b_{suffix}"), finite_values(np, rows, f"response_bbar_{suffix}"))
            )
            if values.size:
                axis.hist(values, bins=np.linspace(0.0, 1.5, 61), histtype="step", density=True, linewidth=1.8, label=dataset, color=colors[dataset])
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"anti-kt R={radius:.1f}")
        axis.set_xlabel("Ejet / E(status-23 b)")
        axis.set_ylabel("Normalized b jets")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "jet_energy_response.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, radius in zip(axes.flat, RADII):
        suffix = f"r{int(round(radius * 10)):02d}"
        plot_histograms(
            np,
            plt,
            axis,
            by_dataset,
            f"out_of_cone_fraction_{suffix}",
            np.linspace(0.0, 1.0, 51),
            "Visible Higgs energy outside matched jets / total",
        )
        axis.set_title(f"anti-kt R={radius:.1f}")
    figure.tight_layout()
    figure.savefig(output_dir / "out_of_cone_energy.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    for axis, radius in zip(axes.flat, RADII):
        suffix = f"r{int(round(radius * 10)):02d}"
        for dataset, rows in by_dataset.items():
            changes = []
            for row in rows:
                before = row[f"m_bmatched_no_muons_{suffix}"]
                after = row[f"m_bmatched_muon_added_{suffix}"]
                higgs_mass = row["m_h"]
                if all(math.isfinite(item) for item in (before, after, higgs_mass)) and higgs_mass > 0.0:
                    changes.append((after - before) / higgs_mass)
            if changes:
                axis.hist(
                    changes,
                    bins=np.linspace(-0.01, 0.35, 55),
                    histtype="step",
                    density=True,
                    linewidth=1.8,
                    label=dataset,
                    color=colors[dataset],
                )
        axis.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"anti-kt R={radius:.1f}")
        axis.set_xlabel("Muon-added change in b-matched m / mH")
        axis.set_ylabel("Normalized events")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "muon_added_mass_recovery.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    colors = {"noFSR": "#0072B2", "FSR": "#D55E00"}
    for axis, radius in zip(axes, RADII[1:]):
        suffix = f"r{int(round(radius * 10)):02d}"
        for dataset, rows in by_dataset.items():
            changes = []
            for row in rows:
                reference = row["m_bmatched_r04"]
                value = row[f"m_bmatched_{suffix}"]
                higgs_mass = row["m_h"]
                if all(math.isfinite(item) for item in (reference, value, higgs_mass)) and higgs_mass > 0.0:
                    changes.append((value - reference) / higgs_mass)
            if changes:
                axis.hist(changes, bins=np.linspace(-0.2, 0.6, 65), histtype="step", density=True, linewidth=1.8, label=dataset, color=colors[dataset])
        axis.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"R={radius:.1f} minus R=0.4")
        axis.set_xlabel("Change in b-matched m / mH")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Normalized events")
    figure.tight_layout()
    figure.savefig(output_dir / "mass_change_vs_radius.png", dpi=160)
    plt.close(figure)


def validate_args(args):
    if args.max_files <= 0 or args.max_events <= 0:
        raise RuntimeError("--max-files and --max-events must be positive")
    if args.constituent_pt_min < 0.0 or args.jet_pt_min < 0.0:
        raise RuntimeError("pT thresholds must be non-negative")
    if not 0.0 < args.mass_drop_fraction < 1.0:
        raise RuntimeError("--mass-drop-fraction must be between 0 and 1")


def import_analysis_libraries():
    try:
        import fastjet
    except ImportError:
        sys.path.append(site.getusersitepackages())
        import fastjet
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return fastjet, np, plt


def main():
    args = parse_args()
    validate_args(args)
    fastjet, np, plt = import_analysis_libraries()
    selected = selected_input_files(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = {
        dataset: analyze_dataset(fastjet, dataset, files, args)
        for dataset, files in selected.items()
    }
    all_events = by_dataset["noFSR"] + by_dataset["FSR"]
    write_csv(output_dir / "event_metrics.csv", all_events)

    summaries = make_summary(np, by_dataset, args)
    write_csv(output_dir / "summary.csv", summaries)
    print_summary(summaries, args)

    make_dijet_plots(np, plt, by_dataset, output_dir, args)
    make_additional_jet_plots(np, plt, by_dataset, output_dir)
    make_diagnostic_plots(np, plt, by_dataset, output_dir)
    print(f"\nWrote {len(all_events)} event rows, {len(summaries)} summary rows, and plots to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"ERROR: {error}") from error
