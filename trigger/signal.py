"""H(bb) Delphes/HepMC loading and dijet-PPS trigger efficiencies."""

import math
from pathlib import Path
import time

import numpy as np

from common.config_utils import natural_key
from minbias.vertex import vertex_overlap_probability
from trigger.dijet_rate import RAPIDITY_CUTS, correct_jets, select_dijet
from trigger.jet_response import load_delphes_root, read_jets
from trigger.pps import CRITERIA, accepted_groups, criterion_masks, sample_pps_pairs


def paired_signal_files(root_dir, hepmc_dir, max_files=None, exclude=()):
    """Return naturally sorted, stem-aligned Delphes and HepMC files."""
    root_dir = Path(root_dir).resolve()
    hepmc_dir = Path(hepmc_dir).resolve()
    excluded = {Path(path).resolve() for path in exclude}
    roots = [
        path
        for path in sorted(root_dir.glob("*.root"), key=natural_key)
        if path.resolve() not in excluded
    ]
    hepmc_by_stem = {
        path.stem: path for path in sorted(hepmc_dir.glob("*.hepmc"), key=natural_key)
    }
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        roots = roots[:max_files]
    if not roots:
        raise RuntimeError(f"No signal ROOT files found in {root_dir}")
    pairs = []
    for root in roots:
        hepmc = hepmc_by_stem.get(root.stem)
        if hepmc is None:
            raise RuntimeError(f"No aligned HepMC file for {root.name} in {hepmc_dir}")
        pairs.append((root, hepmc))
    return pairs


def parse_hepmc_proton_xi(path, n_events, sqrt_s_gev):
    """Read the highest-|pz| final proton in each arm from HepMC3 ASCII."""
    left = np.full(n_events, np.nan, dtype=np.float64)
    right = np.full(n_events, np.nan, dtype=np.float64)
    event_ids = np.full(n_events, -1, dtype=np.int64)
    beam_energy = float(sqrt_s_gev) / 2.0
    event_index = -1
    left_energy = right_energy = None
    left_abs_pz = right_abs_pz = -1.0

    def store_event():
        if 0 <= event_index < n_events:
            if left_energy is not None:
                left[event_index] = (beam_energy - left_energy) / beam_energy
            if right_energy is not None:
                right[event_index] = (beam_energy - right_energy) / beam_energy

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("E "):
                if event_index >= 0:
                    store_event()
                event_index += 1
                if event_index >= n_events:
                    break
                event_ids[event_index] = int(line.split()[1])
                left_energy = right_energy = None
                left_abs_pz = right_abs_pz = -1.0
            elif 0 <= event_index < n_events and line.startswith("P "):
                fields = line.split()
                if len(fields) < 10 or int(fields[3]) != 2212 or int(fields[9]) != 1:
                    continue
                pz = float(fields[6])
                energy = float(fields[7])
                abs_pz = abs(pz)
                if pz < 0.0 and abs_pz > left_abs_pz:
                    left_energy, left_abs_pz = energy, abs_pz
                elif pz > 0.0 and abs_pz > right_abs_pz:
                    right_energy, right_abs_pz = energy, abs_pz
        else:
            store_event()
    observed_events = min(event_index + 1, n_events)
    if observed_events != n_events:
        raise RuntimeError(
            f"HepMC event count {observed_events} does not match Delphes count {n_events}: {path}"
        )
    return event_ids, left, right


def load_hbb_signal(
    root_dir,
    hepmc_dir,
    *,
    correction_map,
    correction_info,
    sqrt_s_gev=14_000.0,
    eta_max=2.4,
    leading_pt_min=20.0,
    subleading_pt_min=20.0,
    max_files=None,
    progress=False,
):
    """Load aligned H(bb) central observables, occupancy, and truth protons."""
    ROOT = load_delphes_root()
    files = paired_signal_files(
        root_dir, hepmc_dir, max_files, correction_map.get("derivation_files", ())
    )
    parts = {
        name: []
        for name in (
            "n_pileup",
            "dijet_rapidity",
            "leading_pt_gev",
            "subleading_pt_gev",
            "xi_left_truth",
            "xi_right_truth",
            "weight",
        )
    }
    jet_counts = {
        "raw_jets": 0,
        "supported_jets": 0,
        "rejected_below_pt_support": 0,
        "rejected_above_pt_support": 0,
        "rejected_eta_or_map_support": 0,
    }
    started = time.monotonic()
    for file_index, (root_path, hepmc_path) in enumerate(files, start=1):
        root_file = ROOT.TFile.Open(str(root_path))
        tree = root_file.Get("Delphes") if root_file else None
        required = ("Event", "Vertex", "JetPUPPI")
        if not tree or any(not tree.GetBranch(name) for name in required):
            if root_file:
                root_file.Close()
            raise RuntimeError(f"Required signal branches missing from {root_path}")
        entries = int(tree.GetEntries())
        event_ids, xi_left, xi_right = parse_hepmc_proton_xi(
            hepmc_path, entries, sqrt_s_gev
        )
        n_pileup = np.empty(entries, dtype=np.int32)
        dijet_y = np.full(entries, np.nan, dtype=np.float64)
        leading_pt = np.zeros(entries, dtype=np.float64)
        subleading_pt = np.zeros(entries, dtype=np.float64)
        weights = np.empty(entries, dtype=np.float64)
        event_number_offset = None
        for entry in range(entries):
            tree.GetEntry(entry)
            if not tree.Event.GetEntriesFast():
                root_file.Close()
                raise RuntimeError(f"Signal Event branch is empty in {root_path}")
            event = tree.Event.At(0)
            number = int(event.Number)
            if number != event_ids[entry]:
                root_file.Close()
                raise RuntimeError(
                    f"Signal Event.Number does not match HepMC in {root_path}"
                )
            if event_number_offset is None:
                event_number_offset = number - entry
            if number - entry != event_number_offset:
                root_file.Close()
                raise RuntimeError(
                    f"Signal Event.Number is not entry-aligned in {root_path}"
                )
            weights[entry] = float(event.Weight)
            n_pileup[entry] = int(tree.Vertex.GetEntriesFast()) - 1
            if n_pileup[entry] < 0:
                root_file.Close()
                raise RuntimeError(f"Signal event has no primary vertex in {root_path}")
            raw_jets = read_jets(tree.JetPUPPI, math.inf)
            jets, counts = correct_jets(raw_jets, correction_map)
            for name, value in counts.items():
                jet_counts[name] += value
            leading_pt[entry], subleading_pt[entry], dijet_y[entry] = select_dijet(
                jets, leading_pt_min, subleading_pt_min, eta_max
            )
        root_file.Close()
        for name, values in (
            ("n_pileup", n_pileup),
            ("dijet_rapidity", dijet_y),
            ("leading_pt_gev", leading_pt),
            ("subleading_pt_gev", subleading_pt),
            ("xi_left_truth", xi_left),
            ("xi_right_truth", xi_right),
            ("weight", weights),
        ):
            parts[name].append(values)
        if progress:
            elapsed = time.monotonic() - started
            print(
                f"  Hbb files {file_index:,}/{len(files):,}; "
                f"events {sum(part.size for part in parts['n_pileup']):,}; "
                f"elapsed {elapsed:.1f} s",
                flush=True,
            )

    signal = {name: np.concatenate(values) for name, values in parts.items()}
    if np.any(signal["weight"] <= 0.0) or not np.allclose(
        signal["weight"], signal["weight"][0]
    ):
        raise RuntimeError(
            "Hbb efficiency study requires positive, equal event weights"
        )
    raw_jets = jet_counts["raw_jets"]
    jet_counts["total_jets"] = raw_jets
    jet_counts["rejected_jets"] = raw_jets - jet_counts["supported_jets"]
    jet_counts["correction_coverage"] = (
        jet_counts["supported_jets"] / raw_jets if raw_jets else None
    )
    signal.update(
        {
            "root_dir": str(Path(root_dir).resolve()),
            "hepmc_dir": str(Path(hepmc_dir).resolve()),
            "files": len(files),
            "excluded_derivation_files": len(correction_map.get("derivation_files", ())),
            "sqrt_s_gev": float(sqrt_s_gev),
            "jet_counts": jet_counts,
            "jet_selection": {
                "collection": "JetPUPPI",
                "correction": correction_info,
                "eta_max": float(eta_max),
                "leading_pt_min_gev": float(leading_pt_min),
                "subleading_pt_min_gev": float(subleading_pt_min),
                "threshold_comparison": "strictly greater than",
                "unsupported_jets": "rejected",
            },
        }
    )
    return signal


def _efficiency(passing, total):
    value = passing / total if total else 0.0
    error = math.sqrt(value * (1.0 - value) / total) if total else 0.0
    return float(value), float(error)


def study_hbb_signal_efficiency(
    signal,
    flux,
    acceptance,
    *,
    mass_range=(117.0, 133.0),
    xi_resolution=0.0003,
    beam_sigma_z_cm=5.7,
    single_arm_time_resolution_ps=10.0,
    pv_z_resolution_cm=0.1,
    pv_time_resolution_ps=30.0,
    nsigma=2.0,
    seed=22345,
    progress=False,
):
    """Evaluate unconditional H(bb) efficiencies with one PPS overlay/event."""
    n_events = int(signal["n_pileup"].size)
    if n_events <= 0:
        raise ValueError("Hbb signal library is empty")
    fields = ("dijet_rapidity", "xi_left_truth", "xi_right_truth")
    if any(np.asarray(signal[name]).size != n_events for name in fields):
        raise ValueError("Hbb signal arrays are not aligned")
    groups = accepted_groups(flux, acceptance)
    keys = [
        f"{criterion}__dy_{cut_name}"
        for criterion, _label in CRITERIA
        for cut_name, _cut in RAPIDITY_CUTS
    ]
    counts = {
        key: {
            "passing": 0,
            "genuine": 0,
            "mixed": 0,
            "pileup": 0,
            "rescued": 0,
        }
        for key in keys
    }
    central_passing = 0
    central_double_tag = 0
    valid_protons = 0
    double_tag = 0
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    progress_every = max(1, n_events // 10)
    for event in range(n_events):
        xi_truth = np.array(
            [signal["xi_left_truth"][event], signal["xi_right_truth"][event]],
            dtype=np.float64,
        )
        arms = np.array([-1, 1], dtype=np.int8)
        valid = np.isfinite(xi_truth) & (xi_truth > 0.0)
        accepted = valid & acceptance.mask(xi_truth)
        if np.all(valid):
            valid_protons += 1
        if np.all(accepted):
            double_tag += 1
        y_jj = float(signal["dijet_rapidity"][event])
        if np.isfinite(y_jj):
            central_passing += 1
            central_double_tag += bool(np.all(accepted))
            pairs = sample_pps_pairs(
                groups,
                int(signal["n_pileup"][event]),
                rng,
                sqrt_s_gev=flux.sqrt_s_gev,
                mass_range=mass_range,
                xi_resolution=xi_resolution,
                beam_sigma_z_cm=beam_sigma_z_cm,
                single_arm_time_resolution_ps=single_arm_time_resolution_ps,
                pv_z_resolution_cm=pv_z_resolution_cm,
                pv_time_resolution_ps=pv_time_resolution_ps,
                nsigma=nsigma,
                matched_xi=xi_truth[accepted],
                matched_arm=arms[accepted],
            )
            masks = criterion_masks(pairs)
            delta_y = np.abs(pairs["rapidity"] - y_jj)
            for criterion, _label in CRITERIA:
                for cut_name, cut in RAPIDITY_CUTS:
                    key = f"{criterion}__dy_{cut_name}"
                    mask = (
                        masks[criterion]
                        if cut is None
                        else masks[criterion] & (delta_y < cut)
                    )
                    genuine = bool(np.any(mask & pairs["signal_signal"]))
                    mixed = bool(np.any(mask & pairs["signal_pileup"]))
                    pileup = bool(np.any(mask & pairs["pileup_pileup"]))
                    passed = genuine or mixed or pileup
                    counts[key]["passing"] += passed
                    counts[key]["genuine"] += genuine
                    counts[key]["mixed"] += mixed
                    counts[key]["pileup"] += pileup
                    counts[key]["rescued"] += passed and not genuine
        if progress and ((event + 1) % progress_every == 0 or event + 1 == n_events):
            elapsed = time.monotonic() - started
            eta = elapsed * (n_events - event - 1) / (event + 1)
            print(
                f"  Hbb events {event + 1:,}/{n_events:,}; "
                f"elapsed {elapsed:.1f} s; ETA {eta:.1f} s",
                flush=True,
            )

    results = {}
    for key, item in counts.items():
        criterion, cut_name = key.split("__dy_", maxsplit=1)
        cut = dict(RAPIDITY_CUTS)[cut_name]
        efficiency, error = _efficiency(item["passing"], n_events)
        genuine_efficiency, genuine_error = _efficiency(item["genuine"], n_events)
        mixed_efficiency, mixed_error = _efficiency(item["mixed"], n_events)
        pileup_efficiency, pileup_error = _efficiency(item["pileup"], n_events)
        rescue_efficiency, rescue_error = _efficiency(item["rescued"], n_events)
        results[key] = {
            "criterion": criterion,
            "rapidity_cut": cut,
            "passing_events": int(item["passing"]),
            "efficiency": efficiency,
            "efficiency_error": error,
            "genuine_pair_passing_events": int(item["genuine"]),
            "genuine_pair_efficiency": genuine_efficiency,
            "genuine_pair_efficiency_error": genuine_error,
            "mixed_pair_passing_events": int(item["mixed"]),
            "mixed_pair_efficiency": mixed_efficiency,
            "mixed_pair_efficiency_error": mixed_error,
            "pileup_pair_passing_events": int(item["pileup"]),
            "pileup_pair_efficiency": pileup_efficiency,
            "pileup_pair_efficiency_error": pileup_error,
            "rescued_events": int(item["rescued"]),
            "rescue_efficiency": rescue_efficiency,
            "rescue_efficiency_error": rescue_error,
        }
    central_efficiency, central_error = _efficiency(central_passing, n_events)
    valid_efficiency, valid_error = _efficiency(valid_protons, n_events)
    tag_efficiency, tag_error = _efficiency(double_tag, n_events)
    central_tag_efficiency, central_tag_error = _efficiency(
        central_double_tag, n_events
    )
    matched_vertex_reference = {
        mode: vertex_overlap_probability(
            truth_status="matched",
            timing_mode=mode,
            beam_sigma_z_cm=beam_sigma_z_cm,
            single_arm_time_resolution_ps=single_arm_time_resolution_ps,
            pv_z_resolution_cm=pv_z_resolution_cm,
            pv_time_resolution_ps=pv_time_resolution_ps,
            nsigma=nsigma,
        )
        for mode in ("z_only", "luminous", "central")
    }
    return {
        "configuration": {
            "root_dir": signal["root_dir"],
            "hepmc_dir": signal["hepmc_dir"],
            "files": int(signal["files"]),
            "events": n_events,
            "jet_selection": signal["jet_selection"],
            "jet_counts": signal.get("jet_counts"),
            "mass_range_gev": list(mass_range),
            "rapidity_cuts": [cut for _name, cut in RAPIDITY_CUTS],
            "xi_resolution": float(xi_resolution),
            "beam_sigma_z_cm": float(beam_sigma_z_cm),
            "single_arm_time_resolution_ps": float(single_arm_time_resolution_ps),
            "pv_z_resolution_cm": float(pv_z_resolution_cm),
            "pv_time_resolution_ps": float(pv_time_resolution_ps),
            "nsigma": float(nsigma),
            "pps_overlays_per_event": 1,
            "seed": int(seed),
        },
        "stages": {
            "valid_truth_proton_pair": {
                "events": int(valid_protons),
                "efficiency": valid_efficiency,
                "efficiency_error": valid_error,
            },
            "truth_double_pps_tag": {
                "events": int(double_tag),
                "efficiency": tag_efficiency,
                "efficiency_error": tag_error,
            },
            "central_dijet": {
                "events": int(central_passing),
                "efficiency": central_efficiency,
                "efficiency_error": central_error,
            },
            "central_dijet_and_truth_double_pps_tag": {
                "events": int(central_double_tag),
                "efficiency": central_tag_efficiency,
                "efficiency_error": central_tag_error,
            },
        },
        "single_pair_matched_vertex_probability": matched_vertex_reference,
        "criteria": results,
    }
