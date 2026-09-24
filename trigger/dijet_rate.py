#!/usr/bin/env python3
"""Estimate a HardQCD dijet plus pileup-PPS trigger rate."""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from common import jet_calibration
from minbias.artifact import DEFAULT_CONFIG, load_config, sha256, write_json
from minbias.flux import Acceptance, ProtonFlux
from minbias.vertex import vertex_overlap_probability
from trigger.jet_response import dijet_rapidity, load_delphes_root, read_jets
from trigger.hardqcd_campaign import load_campaign
from trigger.pps import CRITERIA, accepted_groups, criterion_masks, sample_pps_pairs


DEFAULT_MASS_RANGE = (117.0, 133.0)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HBB_ROOT = (
    ROOT / "output-superchic/Hbb/Hbb__v01/sim-Delphes/Hbb_FSR_200PU__v01/root"
)
DEFAULT_HBB_HEPMC = (
    ROOT / "output-superchic/Hbb/Hbb__v01/hadr-Pythia/Hbb_FSR__v01/hepmc"
)
RAPIDITY_CUTS = (
    ("none", None),
    ("1", 1.0),
    ("0.75", 0.75),
    ("0.5", 0.5),
    ("0.25", 0.25),
    ("0.1", 0.1),
)
PRIMARY_CRITERIA = (
    "pps_mass",
    "pps_mass_z",
    "pps_mass_z_luminous",
    "pps_mass_z_central",
)


def select_dijet(
    jets, leading_pt_min=20.0, subleading_pt_min=20.0, eta_max=2.4
):
    """Return leading/subleading pT and dijet rapidity after strict thresholds."""
    ordered = sorted(jets, key=lambda jet: jet["pt"], reverse=True)
    if len(ordered) < 2:
        return 0.0, 0.0, math.nan
    leading, subleading = ordered[:2]
    rapidity = math.nan
    if (
        leading["pt"] > leading_pt_min
        and subleading["pt"] > subleading_pt_min
        and abs(leading["eta"]) < eta_max
        and abs(subleading["eta"]) < eta_max
    ):
        rapidity = dijet_rapidity(leading, subleading)
    return leading["pt"], subleading["pt"], rapidity


def correct_jets(jets, correction_map):
    """Correct supported jets and report why unsupported jets were removed."""
    if not jets:
        return [], {
            "raw_jets": 0,
            "supported_jets": 0,
            "rejected_below_pt_support": 0,
            "rejected_above_pt_support": 0,
            "rejected_eta_or_map_support": 0,
        }
    pt = np.asarray([jet["pt"] for jet in jets], dtype=np.float64)
    eta = np.asarray([jet["eta"] for jet in jets], dtype=np.float64)
    phi = np.asarray([jet["phi"] for jet in jets], dtype=np.float64)
    mass = np.asarray([jet["mass"] for jet in jets], dtype=np.float64)
    corrected = jet_calibration.correct_jet_kinematics(
        pt, eta, phi, mass, correction_map
    )
    valid = corrected["valid"]
    support_min, support_max = correction_map["raw_pt_support"]
    below = pt < support_min
    above = pt >= support_max
    other = ~valid & ~below & ~above
    output = [
        {
            "pt": float(corrected["pt"][index]),
            "eta": float(corrected["eta"][index]),
            "phi": float(corrected["phi"][index]),
            "mass": float(corrected["mass"][index]),
        }
        for index in range(len(jets))
        if valid[index]
    ]
    return output, {
        "raw_jets": len(jets),
        "supported_jets": int(np.count_nonzero(valid)),
        "rejected_below_pt_support": int(np.count_nonzero(below)),
        "rejected_above_pt_support": int(np.count_nonzero(above)),
        "rejected_eta_or_map_support": int(np.count_nonzero(other)),
    }


def load_jet_correction(path, fsr_state="FSR", sample="HardQCD"):
    path = Path(path).resolve()
    correction_map = jet_calibration.load_correction_map(path, fsr_state, sample)
    if correction_map.get("fsr_state") != fsr_state or correction_map.get(
        "source_sample"
    ) != sample:
        raise ValueError(
            f"Correction-map identity does not match {fsr_state}/{sample}: {path}"
        )
    return correction_map, {
        "path": str(path),
        "sha256": sha256(path),
        "schema_version": jet_calibration.SCHEMA_VERSION,
        "fsr_state": fsr_state,
        "sample": sample,
    }


def load_hardqcd_campaign(
    campaign_dir,
    *,
    correction_map,
    correction_info,
    eta_max=2.4,
    leading_pt_min=20.0,
    subleading_pt_min=20.0,
    verify_hash=True,
    progress=False,
):
    """Load corrected dijets from validated rate/validation campaign shards."""
    ROOT = load_delphes_root()
    campaign_data = load_campaign(
        campaign_dir, role="rate_validation", verify_hash=verify_hash
    )
    total_events = sum(int(shard["job"]["events"]) for shard in campaign_data["shards"])
    progress_every = max(1, total_events // 10)
    completed = 0
    parts = {
        name: []
        for name in (
            "pthat",
            "weights",
            "n_pileup",
            "dijet_y",
            "leading_pt",
            "subleading_pt",
        )
    }
    jet_counts = {
        "raw_jets": 0,
        "supported_jets": 0,
        "rejected_below_pt_support": 0,
        "rejected_above_pt_support": 0,
        "rejected_eta_or_map_support": 0,
    }
    for shard in campaign_data["shards"]:
        table = pq.read_table(shard["event_path"])
        event_id = np.asarray(table["event_id"], dtype=np.int64)
        pthat = np.asarray(table["pthat_gev"], dtype=np.float64)
        weights = np.asarray(table["weight"], dtype=np.float64)
        expected = np.arange(int(shard["job"]["events"]), dtype=np.int64)
        if not np.array_equal(event_id, expected):
            raise RuntimeError(
                f"HardQCD sidecar IDs are not dense in {shard['event_path']}"
            )
        if pthat.size != event_id.size or weights.size != event_id.size:
            raise RuntimeError(f"HardQCD sidecar columns differ in {shard['event_path']}")

        root_path = shard["root_path"]
        root_file = ROOT.TFile.Open(str(root_path))
        tree = root_file.Get("Delphes") if root_file else None
        required = ("Event", "Vertex", "JetPUPPI")
        if not tree or any(not tree.GetBranch(name) for name in required):
            if root_file:
                root_file.Close()
            raise RuntimeError(
                f"Required Event/Vertex/JetPUPPI branches missing from {root_path}"
            )
        if int(tree.GetEntries()) != event_id.size:
            root_file.Close()
            raise RuntimeError(f"Delphes and sidecar counts differ in {root_path}")
        n_pileup = np.empty(event_id.size, dtype=np.int32)
        dijet_y = np.full(event_id.size, np.nan, dtype=np.float64)
        leading_pt = np.zeros(event_id.size, dtype=np.float64)
        subleading_pt = np.zeros(event_id.size, dtype=np.float64)
        for entry in range(event_id.size):
            tree.GetEntry(entry)
            if not tree.Event.GetEntriesFast():
                root_file.Close()
                raise RuntimeError(f"Delphes Event branch is empty in {root_path}")
            number = int(tree.Event.At(0).Number)
            if number != int(event_id[entry]):
                root_file.Close()
                raise RuntimeError(
                    f"Delphes Event.Number does not match the sidecar in {root_path}"
                )
            n_pileup[entry] = int(tree.Vertex.GetEntriesFast()) - 1
            if n_pileup[entry] < 0:
                root_file.Close()
                raise RuntimeError(f"Delphes event has no primary vertex in {root_path}")
            raw_jets = read_jets(tree.JetPUPPI, math.inf)
            jets, counts = correct_jets(raw_jets, correction_map)
            for name, value in counts.items():
                jet_counts[name] += value
            leading_pt[entry], subleading_pt[entry], dijet_y[entry] = select_dijet(
                jets, leading_pt_min, subleading_pt_min, eta_max
            )
            completed += 1
            if progress and (
                completed % progress_every == 0 or completed == total_events
            ):
                print(f"  central events {completed:,}/{total_events:,}", flush=True)
        root_file.Close()
        for name, values in (
            ("pthat", pthat),
            ("weights", weights),
            ("n_pileup", n_pileup),
            ("dijet_y", dijet_y),
            ("leading_pt", leading_pt),
            ("subleading_pt", subleading_pt),
        ):
            parts[name].append(values)

    weights = np.concatenate(parts["weights"])
    if np.any(weights <= 0.0) or not np.allclose(weights, weights[0]):
        raise RuntimeError("HardQCD campaign must contain positive, equal event weights")
    raw_jets = jet_counts["raw_jets"]
    jet_counts["total_jets"] = raw_jets
    jet_counts["rejected_jets"] = raw_jets - jet_counts["supported_jets"]
    jet_counts["correction_coverage"] = (
        jet_counts["supported_jets"] / raw_jets if raw_jets else None
    )
    return {
        "campaign": str(campaign_data["campaign"]),
        "metadata": campaign_data["metadata"],
        "event_id": np.arange(total_events, dtype=np.int64),
        "pthat_gev": np.concatenate(parts["pthat"]),
        "n_pileup": np.concatenate(parts["n_pileup"]),
        "dijet_rapidity": np.concatenate(parts["dijet_y"]),
        "leading_pt_gev": np.concatenate(parts["leading_pt"]),
        "subleading_pt_gev": np.concatenate(parts["subleading_pt"]),
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


def select_hardest_indices(rng, counts, pthat):
    """Sample source records with replacement and return the largest-pTHat index per BX."""
    counts = np.asarray(counts, dtype=np.int64)
    selected = np.full(counts.size, -1, dtype=np.int64)
    total = int(np.sum(counts))
    candidates = rng.integers(0, pthat.size, size=total)
    offset = 0
    for bx, count in enumerate(counts):
        if count:
            group = candidates[offset : offset + count]
            selected[bx] = group[int(np.argmax(pthat[group]))]
            offset += int(count)
    return selected


def _empty_accumulators():
    return {
        f"{criterion}__dy_{cut_name}": {
            "criterion": criterion,
            "rapidity_cut": cut,
            "pair_sum": 0,
            "pair_square_sum": 0,
            "passing_bx": 0,
        }
        for criterion, _label in CRITERIA
        for cut_name, cut in RAPIDITY_CUTS
    }


def _criterion_result(accumulator, n_bx, frequency):
    pair_sum = accumulator["pair_sum"]
    pair_square_sum = accumulator["pair_square_sum"]
    mean = pair_sum / n_bx
    if n_bx > 1:
        variance = max((pair_square_sum - pair_sum * pair_sum / n_bx) / (n_bx - 1), 0.0)
        mean_error = math.sqrt(variance / n_bx)
    else:
        mean_error = 0.0
    probability = accumulator["passing_bx"] / n_bx
    binomial = math.sqrt(probability * (1.0 - probability) / n_bx)
    return {
        "criterion": accumulator["criterion"],
        "rapidity_cut": accumulator["rapidity_cut"],
        "mean_pairs_per_bx": float(mean),
        "mean_pairs_per_bx_error": float(mean_error),
        "passing_bx": int(accumulator["passing_bx"]),
        "bx_probability": float(probability),
        "binomial_error": float(binomial),
        "rate_hz": float(probability * frequency),
        "rate_binomial_error_hz": float(binomial * frequency),
    }


def _bootstrap_errors(
    source_selected, source_passing, hard_probability, n_bootstrap, rng
):
    n_source, n_criteria = source_passing.shape
    samples = np.zeros((n_bootstrap, n_criteria), dtype=np.float64)
    for bootstrap in range(n_bootstrap):
        chosen = rng.integers(0, n_source, size=n_source)
        denominator = int(np.sum(source_selected[chosen]))
        if denominator:
            samples[bootstrap] = (
                hard_probability * np.sum(source_passing[chosen], axis=0) / denominator
            )
    return np.std(samples, axis=0, ddof=1) if n_bootstrap > 1 else np.zeros(n_criteria)


def study_dijet_trigger_rates(
    hard,
    flux,
    acceptance,
    *,
    mu=200.0,
    n_bx=1_000_000,
    mass_range=DEFAULT_MASS_RANGE,
    xi_resolution=0.0003,
    beam_sigma_z_cm=5.7,
    single_arm_time_resolution_ps=10.0,
    pv_z_resolution_cm=0.1,
    pv_time_resolution_ps=30.0,
    nsigma=2.0,
    bx_frequency_hz=31.6e6,
    seed=12345,
    batch_size=10_000,
    n_bootstrap=200,
    progress=False,
):
    """Return unconditional BX probabilities and rates for the dijet-PPS trigger."""
    if mu <= 0.0 or n_bx <= 0 or batch_size <= 0 or n_bootstrap < 0:
        raise ValueError(
            "mu, n_bx, and batch_size must be positive; bootstrap cannot be negative"
        )
    pthat = np.asarray(hard["pthat_gev"], dtype=np.float64)
    if not pthat.size:
        raise ValueError("HardQCD library is empty")
    sigma_hard = float(hard["metadata"]["pythia"]["sigma_gen_mb"])
    sigma_hard_error = float(hard["metadata"]["pythia"].get("sigma_err_mb", 0.0))
    sigma_inelastic = float(flux.metadata["cross_sections"]["sigma_gen_mb"])
    hard_fraction = sigma_hard / sigma_inelastic
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("HardQCD/inelastic cross-section ratio must be in (0, 1]")
    hard_mu = float(mu) * hard_fraction
    hard_probability = -math.expm1(-hard_mu)
    mass_range = tuple(float(value) for value in mass_range)
    groups = accepted_groups(flux, acceptance)
    accumulators = _empty_accumulators()
    accumulator_keys = list(accumulators)
    key_index = {key: index for index, key in enumerate(accumulator_keys)}
    source_selected = np.zeros(pthat.size, dtype=np.int64)
    source_passing = np.zeros((pthat.size, len(accumulator_keys)), dtype=np.int64)
    selected_pthat = []
    n_hard_bx = 0
    n_central_bx = 0
    same_pps_pairs = 0
    same_mass_pairs = 0
    rng = np.random.default_rng(seed)
    start_time = time.monotonic()
    n_batches = math.ceil(n_bx / batch_size)
    if progress:
        print(
            f"Sampling {n_bx:,} BX in {n_batches:,} batches; "
            f"HardQCD mean={hard_mu:.4g}, P(>=1)={hard_probability:.8f}...",
            flush=True,
        )
    for batch_number, batch_start in enumerate(range(0, n_bx, batch_size), start=1):
        size = min(batch_size, n_bx - batch_start)
        hard_counts = rng.poisson(hard_mu, size=size)
        selected = select_hardest_indices(rng, hard_counts, pthat)
        for source in selected[selected >= 0]:
            n_hard_bx += 1
            source_selected[source] += 1
            selected_pthat.append(pthat[source])
            y_jj = hard["dijet_rapidity"][source]
            if not np.isfinite(y_jj):
                continue
            n_central_bx += 1
            pairs = sample_pps_pairs(
                groups,
                int(hard["n_pileup"][source]),
                rng,
                sqrt_s_gev=flux.sqrt_s_gev,
                mass_range=mass_range,
                xi_resolution=xi_resolution,
                beam_sigma_z_cm=beam_sigma_z_cm,
                single_arm_time_resolution_ps=single_arm_time_resolution_ps,
                pv_z_resolution_cm=pv_z_resolution_cm,
                pv_time_resolution_ps=pv_time_resolution_ps,
                nsigma=nsigma,
            )
            masks = criterion_masks(pairs)
            same_pps_pairs += int(np.sum(pairs["same_interaction"]))
            same_mass_pairs += int(np.sum(pairs["same_interaction"] & pairs["mass_ok"]))
            delta_y = np.abs(pairs["rapidity"] - y_jj)
            for criterion, _label in CRITERIA:
                for cut_name, cut in RAPIDITY_CUTS:
                    key = f"{criterion}__dy_{cut_name}"
                    mask = (
                        masks[criterion]
                        if cut is None
                        else masks[criterion] & (delta_y < cut)
                    )
                    count = int(np.sum(mask))
                    accumulator = accumulators[key]
                    accumulator["pair_sum"] += count
                    accumulator["pair_square_sum"] += count * count
                    if count:
                        accumulator["passing_bx"] += 1
                        source_passing[source, key_index[key]] += 1
        if progress and (
            batch_number == n_batches or batch_number % max(1, n_batches // 10) == 0
        ):
            elapsed = time.monotonic() - start_time
            completed = batch_start + size
            eta = elapsed * (n_bx - completed) / completed
            print(
                f"  BX {completed:,}/{n_bx:,}; elapsed {elapsed:.1f} s; ETA {eta:.1f} s",
                flush=True,
            )

    results = {
        key: _criterion_result(value, n_bx, bx_frequency_hz)
        for key, value in accumulators.items()
    }
    if progress:
        print(
            f"Evaluating {n_bootstrap} finite-library bootstrap replicas...", flush=True
        )
    bootstrap_rng = np.random.default_rng(seed + 1)
    bootstrap = _bootstrap_errors(
        source_selected, source_passing, hard_probability, n_bootstrap, bootstrap_rng
    )
    conditional_pass = np.array(
        [
            source_passing[:, index].sum() / n_hard_bx if n_hard_bx else 0.0
            for index in range(len(accumulator_keys))
        ]
    )
    hard_probability_error = (
        math.exp(-hard_mu) * float(mu) * sigma_hard_error / sigma_inelastic
    )
    cross_section_error = conditional_pass * hard_probability_error
    for index, key in enumerate(accumulator_keys):
        result = results[key]
        result["library_bootstrap_error"] = float(bootstrap[index])
        result["cross_section_error"] = float(cross_section_error[index])
        total = math.sqrt(
            result["binomial_error"] ** 2
            + bootstrap[index] ** 2
            + cross_section_error[index] ** 2
        )
        result["total_error"] = float(total)
        result["rate_library_bootstrap_error_hz"] = float(
            bootstrap[index] * bx_frequency_hz
        )
        result["rate_cross_section_error_hz"] = float(
            cross_section_error[index] * bx_frequency_hz
        )
        result["rate_total_error_hz"] = float(total * bx_frequency_hz)

    selected_pthat = np.asarray(selected_pthat, dtype=np.float64)
    top_threshold = float(np.quantile(pthat, 0.99))
    dijet_probability = n_central_bx / n_bx
    dijet_probability_error = math.sqrt(
        dijet_probability * (1.0 - dijet_probability) / n_bx
    )
    dijet_efficiency = n_central_bx / n_hard_bx if n_hard_bx else 0.0
    dijet_efficiency_error = (
        math.sqrt(dijet_efficiency * (1.0 - dijet_efficiency) / n_hard_bx)
        if n_hard_bx
        else 0.0
    )
    labels = dict(CRITERIA)
    vertex_reference = {
        mode: vertex_overlap_probability(
            truth_status="unrelated",
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
            "hardqcd_campaign": hard["campaign"],
            "jet_selection": hard.get("jet_selection"),
            "pps_artifact": str(flux.path) if flux.path else None,
            "mu": float(mu),
            "n_bx": int(n_bx),
            "mass_range_gev": list(mass_range),
            "rapidity_cuts": [cut for _name, cut in RAPIDITY_CUTS],
            "xi_resolution": float(xi_resolution),
            "beam_sigma_z_cm": float(beam_sigma_z_cm),
            "pps_time_resolution_ps": float(single_arm_time_resolution_ps),
            "pv_z_resolution_cm": float(pv_z_resolution_cm),
            "pv_time_resolution_ps": float(pv_time_resolution_ps),
            "nsigma": float(nsigma),
            "bx_frequency_hz": float(bx_frequency_hz),
            "seed": int(seed),
            "bootstrap_replicas": int(n_bootstrap),
        },
        "hardqcd": {
            "library_events": int(pthat.size),
            "jet_correction": hard.get("jet_selection", {}).get("correction"),
            "jet_counts": hard.get("jet_counts"),
            "sigma_mb": sigma_hard,
            "sigma_error_mb": sigma_hard_error,
            "inelastic_sigma_mb": sigma_inelastic,
            "cross_section_ratio": hard_fraction,
            "mean_interactions_per_bx": hard_mu,
            "probability_at_least_one": hard_probability,
            "probability_at_least_one_error": hard_probability_error,
            "rate_at_least_one_hz": float(hard_probability * bx_frequency_hz),
            "rate_at_least_one_error_hz": float(
                hard_probability_error * bx_frequency_hz
            ),
            "sampled_bx_with_hardqcd": int(n_hard_bx),
            "sampled_hardqcd_bx_probability": float(n_hard_bx / n_bx),
            "sampled_bx_with_dijet": int(n_central_bx),
            "dijet_efficiency_given_hardqcd": float(dijet_efficiency),
            "dijet_efficiency_error": float(dijet_efficiency_error),
            "dijet_bx_probability": float(dijet_probability),
            "dijet_bx_probability_error": float(dijet_probability_error),
            "dijet_rate_hz": float(dijet_probability * bx_frequency_hz),
            "dijet_rate_error_hz": float(dijet_probability_error * bx_frequency_hz),
            "selected_pthat_mean_gev": float(np.mean(selected_pthat))
            if selected_pthat.size
            else None,
            "selected_pthat_max_gev": float(np.max(selected_pthat))
            if selected_pthat.size
            else None,
            "library_pthat_99pct_gev": top_threshold,
            "selected_top_1pct_fraction": float(
                np.mean(selected_pthat >= top_threshold)
            )
            if selected_pthat.size
            else None,
        },
        "pps": {
            "exact_pileup_count_source": "Delphes Vertex entries - 1",
            "truth_pps_interaction_probability": groups["probability"],
            "single_pair_unrelated_vertex_probability": vertex_reference,
            "same_interaction_mean_pairs_per_bx": float(same_pps_pairs / n_bx),
            "same_interaction_mass_mean_pairs_per_bx": float(same_mass_pairs / n_bx),
        },
        "criteria_labels": labels,
        "criteria": results,
    }


def print_report(report):
    hard = report["hardqcd"]
    frequency = report["configuration"]["bx_frequency_hz"]
    print(
        f"HardQCD sigma/inelastic = {hard['cross_section_ratio']:.6g}; "
        f"mean/BX = {hard['mean_interactions_per_bx']:.6g}; "
        f"P(>=1) = {hard['probability_at_least_one']:.8g}; "
        f"rate = {hard['rate_at_least_one_hz'] / 1e6:.6g} MHz"
    )
    print(
        f"Selected central dijet efficiency = {hard['dijet_efficiency_given_hardqcd']:.6g} "
        f"+/- {hard['dijet_efficiency_error']:.2g} "
        f"({hard['sampled_bx_with_dijet']:,}/{hard['sampled_bx_with_hardqcd']:,}); "
        f"BX rate = {hard['dijet_rate_hz'] / 1e6:.6g} MHz"
    )
    labels = report["criteria_labels"]
    signal = report.get("signal")
    if signal is None:
        print("Main trigger rates (mass window required):")
        print(
            f"{'criterion':<34} {'|dy|':>7} {'P(BX)':>13} {'stat':>11} "
            f"{'library':>11} {'rate [kHz]':>16} {'total [kHz]':>14}"
        )
        for criterion in PRIMARY_CRITERIA:
            for cut_name, cut in RAPIDITY_CUTS:
                result = report["criteria"][f"{criterion}__dy_{cut_name}"]
                print(
                    f"{labels[criterion]:<34} "
                    f"{('none' if cut is None else f'{cut:g}'):>7} "
                    f"{result['bx_probability']:>13.6g} "
                    f"{result['binomial_error']:>11.2g} "
                    f"{result['library_bootstrap_error']:>11.2g} "
                    f"{result['rate_hz'] / 1e3:>16.6g} "
                    f"{result['rate_total_error_hz'] / 1e3:>14.2g}"
                )
    else:
        stages = signal["stages"]
        print(
            "H(bb) signal stages: "
            f"central dijet={stages['central_dijet']['efficiency']:.6g} +/- "
            f"{stages['central_dijet']['efficiency_error']:.2g}; "
            f"truth double PPS={stages['truth_double_pps_tag']['efficiency']:.6g} +/- "
            f"{stages['truth_double_pps_tag']['efficiency_error']:.2g}; "
            "both="
            f"{stages['central_dijet_and_truth_double_pps_tag']['efficiency']:.6g} "
            "+/- "
            f"{stages['central_dijet_and_truth_double_pps_tag']['efficiency_error']:.2g}"
        )
        print("Background rate and unconditional H(bb) efficiency:")
        print(
            f"{'criterion':<34} {'|dy|':>7} {'rate [kHz]':>16} "
            f"{'rate err':>11} {'signal eff':>13} {'sig err':>10} "
            f"{'genuine':>11} {'PU rescue':>11}"
        )
        for criterion in PRIMARY_CRITERIA:
            for cut_name, cut in RAPIDITY_CUTS:
                key = f"{criterion}__dy_{cut_name}"
                background = report["criteria"][key]
                efficiency = signal["criteria"][key]
                print(
                    f"{labels[criterion]:<34} "
                    f"{('none' if cut is None else f'{cut:g}'):>7} "
                    f"{background['rate_hz'] / 1e3:>16.6g} "
                    f"{background['rate_total_error_hz'] / 1e3:>11.2g} "
                    f"{efficiency['efficiency']:>13.6g} "
                    f"{efficiency['efficiency_error']:>10.2g} "
                    f"{efficiency['genuine_pair_efficiency']:>11.6g} "
                    f"{efficiency['rescue_efficiency']:>11.6g}"
                )
    print(f"Configured BX frequency: {frequency / 1e6:g} MHz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardqcd-campaign", type=Path, required=True)
    parser.add_argument("--jet-corrections", type=Path, required=True)
    parser.add_argument("--pps-artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mu", type=float, default=200.0)
    parser.add_argument("--bunch-crossings", type=int, default=100_000)
    parser.add_argument(
        "--mass-window", type=float, nargs=2, default=DEFAULT_MASS_RANGE
    )
    parser.add_argument("--leading-pt-gev", type=float, default=20.0)
    parser.add_argument("--subleading-pt-gev", type=float, default=20.0)
    parser.add_argument("--eta-max", type=float, default=2.4)
    parser.add_argument("--pps-time-resolution-ps", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bx-frequency-mhz", type=float, default=31.6)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument(
        "--with-hbb-signal",
        action="store_true",
        help="Also evaluate the aligned SuperChic H(bb) FSR+PU200 signal sample.",
    )
    parser.add_argument("--hbb-root-dir", type=Path, default=DEFAULT_HBB_ROOT)
    parser.add_argument("--hbb-hepmc-dir", type=Path, default=DEFAULT_HBB_HEPMC)
    parser.add_argument("--hbb-max-files", type=int, default=5)
    parser.add_argument("--hbb-seed", type=int, default=22345)
    args = parser.parse_args()
    config = load_config(args.config)
    correction_map, correction_info = load_jet_correction(args.jet_corrections)
    print(
        "Loading corrected HardQCD JetPUPPI, pTHat, and exact PU occupancies...",
        flush=True,
    )
    hard = load_hardqcd_campaign(
        args.hardqcd_campaign,
        correction_map=correction_map,
        correction_info=correction_info,
        eta_max=args.eta_max,
        leading_pt_min=args.leading_pt_gev,
        subleading_pt_min=args.subleading_pt_gev,
        verify_hash=not args.skip_hash,
        progress=True,
    )
    print("Loading and validating the minbias proton artifact...", flush=True)
    flux = ProtonFlux.load(args.pps_artifact)
    print(
        "Preparing truth-PPS groups and starting correlated BX sampling...", flush=True
    )
    report = study_dijet_trigger_rates(
        hard,
        flux,
        Acceptance(config["xi_windows"]),
        mu=args.mu,
        n_bx=args.bunch_crossings,
        mass_range=args.mass_window,
        xi_resolution=config["xi_resolution"],
        single_arm_time_resolution_ps=args.pps_time_resolution_ps,
        bx_frequency_hz=args.bx_frequency_mhz * 1e6,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        progress=True,
    )
    if args.with_hbb_signal:
        from trigger.signal import load_hbb_signal, study_hbb_signal_efficiency

        signal_map, signal_info = load_jet_correction(
            args.jet_corrections, sample="Hbb"
        )
        print(
            "Loading corrected, aligned H(bb) PU200 Delphes and HepMC events...",
            flush=True,
        )
        signal = load_hbb_signal(
            args.hbb_root_dir,
            args.hbb_hepmc_dir,
            correction_map=signal_map,
            correction_info=signal_info,
            sqrt_s_gev=flux.sqrt_s_gev,
            eta_max=args.eta_max,
            leading_pt_min=args.leading_pt_gev,
            subleading_pt_min=args.subleading_pt_gev,
            max_files=args.hbb_max_files,
            progress=True,
        )
        print(
            "Sampling matched signal protons plus exact-count PPS pileup...", flush=True
        )
        report["signal"] = study_hbb_signal_efficiency(
            signal,
            flux,
            Acceptance(config["xi_windows"]),
            mass_range=args.mass_window,
            xi_resolution=config["xi_resolution"],
            single_arm_time_resolution_ps=args.pps_time_resolution_ps,
            seed=args.hbb_seed,
            progress=True,
        )
    if args.output:
        print(f"Writing full JSON report to {args.output.resolve()}...", flush=True)
        write_json(args.output, report)
    print_report(report)


if __name__ == "__main__":
    main()
