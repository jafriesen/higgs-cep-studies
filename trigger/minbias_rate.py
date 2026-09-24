#!/usr/bin/env python3
"""Sample Poisson pileup bunch crossings and estimate PPS trigger rates."""

import argparse
import math
from pathlib import Path
import time

import numpy as np

from minbias.artifact import DEFAULT_CONFIG, load_config, write_json
from minbias.flux import Acceptance, ProtonFlux
from minbias.vertex import (
    C_CM_PER_PS,
    pps_vertex_resolution_cm,
    vertex_overlap_probability,
)
from trigger.pps import CRITERIA, accepted_groups as _accepted_groups

DEFAULT_MASS_RANGE = (117.0, 133.0)


def _count_pair_categories(
    xi,
    arm,
    arrival,
    interaction,
    sqrt_s_gev,
    mass_range,
    pv_z,
    pv_u,
    z_cut,
    luminous_u_cut,
    central_u_cut,
):
    left = arm < 0
    right = arm > 0
    if not np.any(left) or not np.any(right):
        return {key: 0 for key, _label in CRITERIA}, {"pps": 0, "pps_mass": 0}

    xi_left = xi[left, None]
    xi_right = xi[None, right]
    mass = np.sqrt(xi_left * xi_right) * sqrt_s_gev
    mass_ok = (mass >= mass_range[0]) & (mass < mass_range[1])

    left_arrival = arrival[left, None]
    right_arrival = arrival[None, right]
    pp_z = 0.5 * (left_arrival - right_arrival)
    pp_u = 0.5 * (left_arrival + right_arrival)
    z_ok = np.abs(pp_z - pv_z) <= z_cut
    luminous_ok = np.abs(pp_u) <= luminous_u_cut
    central_ok = np.abs(pp_u - pv_u) <= central_u_cut

    masks = {
        "pps": np.ones(mass.shape, dtype=bool),
        "pps_mass": mass_ok,
        "pps_z": z_ok,
        "pps_mass_z": mass_ok & z_ok,
        "pps_z_luminous": z_ok & luminous_ok,
        "pps_mass_z_luminous": mass_ok & z_ok & luminous_ok,
        "pps_z_central": z_ok & central_ok,
        "pps_mass_z_central": mass_ok & z_ok & central_ok,
    }
    counts = {key: int(np.sum(mask)) for key, mask in masks.items()}
    same = interaction[left, None] == interaction[None, right]
    same_counts = {
        "pps": int(np.sum(same)),
        "pps_mass": int(np.sum(same & mass_ok)),
    }
    return counts, same_counts


def _validate_study_inputs(
    flux,
    mu,
    n_bx,
    mass_range,
    xi_resolution,
    beam_sigma_z_cm,
    single_arm_time_resolution_ps,
    pv_z_resolution_cm,
    pv_time_resolution_ps,
    nsigma,
    bx_frequency_hz,
    seed,
):
    if not isinstance(flux, ProtonFlux):
        raise TypeError("flux must be a ProtonFlux")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if n_bx <= 0:
        raise ValueError("n_bx must be positive")
    if len(mass_range) != 2 or not 0.0 < mass_range[0] < mass_range[1]:
        raise ValueError("mass_range must satisfy 0 < low < high")
    if xi_resolution < 0.0:
        raise ValueError("xi_resolution must be non-negative")
    if beam_sigma_z_cm < 0.0:
        raise ValueError("beam_sigma_z_cm must be non-negative")
    if pv_z_resolution_cm < 0.0:
        raise ValueError("pv_z_resolution_cm must be non-negative")
    if pv_time_resolution_ps <= 0.0:
        raise ValueError("pv_time_resolution_ps must be positive")
    if nsigma <= 0.0:
        raise ValueError("nsigma must be positive")
    if bx_frequency_hz <= 0.0:
        raise ValueError("bx_frequency_hz must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    pps_vertex_resolution_cm(single_arm_time_resolution_ps)


def _finalize_criterion(pair_sum, pair_square_sum, passing_bx, n_bx, frequency):
    mean = pair_sum / n_bx
    if n_bx > 1:
        variance = max((pair_square_sum - pair_sum * pair_sum / n_bx) / (n_bx - 1), 0.0)
        mean_error = math.sqrt(variance / n_bx)
    else:
        mean_error = 0.0
    probability = passing_bx / n_bx
    probability_error = math.sqrt(probability * (1.0 - probability) / n_bx)
    return {
        "mean_pairs_per_bx": float(mean),
        "mean_pairs_per_bx_error": float(mean_error),
        "passing_bx": int(passing_bx),
        "bx_probability": float(probability),
        "bx_probability_error": float(probability_error),
        "rate_hz": float(probability * frequency),
        "rate_error_hz": float(probability_error * frequency),
    }


def study_trigger_rates(
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
    progress=False,
):
    """Return correlated pair and BX-level rates for a Poisson pileup sample."""
    mass_range = tuple(float(value) for value in mass_range)
    _validate_study_inputs(
        flux,
        float(mu),
        int(n_bx),
        mass_range,
        float(xi_resolution),
        float(beam_sigma_z_cm),
        float(single_arm_time_resolution_ps),
        float(pv_z_resolution_cm),
        float(pv_time_resolution_ps),
        float(nsigma),
        float(bx_frequency_hz),
        int(seed),
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if progress:
        print("Preparing truth-PPS interaction groups...", flush=True)
    preparation_start = time.monotonic()
    groups = _accepted_groups(flux, acceptance)
    n_relevant = int(groups["starts"].size)
    relevant_probability = n_relevant / flux.n_inelastic_generated
    relevant_mu = float(mu) * relevant_probability
    if progress:
        elapsed = time.monotonic() - preparation_start
        print(
            f"Prepared {n_relevant:,} relevant interactions "
            f"({relevant_probability:.3%} of the inelastic sample) in {elapsed:.1f} s.",
            flush=True,
        )
    pps_sigma = pps_vertex_resolution_cm(single_arm_time_resolution_ps)
    z_cut = float(nsigma) * math.sqrt(pps_sigma**2 + float(pv_z_resolution_cm) ** 2)
    central_u_cut = float(nsigma) * math.sqrt(
        pps_sigma**2 + (C_CM_PER_PS * float(pv_time_resolution_ps)) ** 2
    )
    luminous_u_cut = float(nsigma) * float(beam_sigma_z_cm)

    accumulators = {
        key: {"pair_sum": 0, "pair_square_sum": 0, "passing_bx": 0}
        for key, _label in CRITERIA
    }
    same_pair_sum = {"pps": 0, "pps_mass": 0}
    sampled_interactions = 0
    sampled_protons = 0
    rng = np.random.default_rng(seed)
    n_batches = math.ceil(n_bx / batch_size)
    progress_every = max(1, math.ceil(n_batches / 10))
    sampling_start = time.monotonic()
    if progress:
        print(
            f"Sampling {n_bx:,} BX in {n_batches:,} batches "
            f"(expected {relevant_mu:.2f} relevant interactions/BX)...",
            flush=True,
        )

    for batch_index, batch_start in enumerate(range(0, n_bx, batch_size)):
        current_n_bx = min(batch_size, n_bx - batch_start)
        interactions_per_bx = rng.poisson(relevant_mu, size=current_n_bx)
        n_interactions = int(np.sum(interactions_per_bx))
        sampled_interactions += n_interactions
        if n_relevant:
            chosen = rng.integers(0, n_relevant, size=n_interactions)
        else:
            chosen = np.empty(0, dtype=np.int64)
        interaction_bx = np.repeat(np.arange(current_n_bx), interactions_per_bx)
        interaction_z = rng.normal(0.0, beam_sigma_z_cm, size=n_interactions)
        interaction_u = rng.normal(0.0, beam_sigma_z_cm, size=n_interactions)

        central_z = rng.normal(0.0, beam_sigma_z_cm, size=current_n_bx)
        central_u = rng.normal(0.0, beam_sigma_z_cm, size=current_n_bx)
        central_z += rng.normal(0.0, pv_z_resolution_cm, size=current_n_bx)
        central_u += C_CM_PER_PS * rng.normal(
            0.0, pv_time_resolution_ps, size=current_n_bx
        )

        rows_per_interaction = groups["counts"][chosen]
        n_protons = int(np.sum(rows_per_interaction))
        sampled_protons += n_protons
        row_prefix = np.cumsum(rows_per_interaction) - rows_per_interaction
        source_row = (
            np.repeat(groups["starts"][chosen], rows_per_interaction)
            + np.arange(n_protons)
            - np.repeat(row_prefix, rows_per_interaction)
        )
        proton_interaction = np.repeat(np.arange(n_interactions), rows_per_interaction)
        proton_bx = interaction_bx[proton_interaction]
        arm = groups["arm"][source_row]
        xi = groups["xi"][source_row] + float(xi_resolution) * rng.standard_normal(
            n_protons
        )
        arrival = (
            interaction_u[proton_interaction]
            - arm * interaction_z[proton_interaction]
            + C_CM_PER_PS
            * rng.normal(0.0, single_arm_time_resolution_ps, size=n_protons)
        )

        positive = xi > 0.0
        xi = xi[positive]
        arm = arm[positive]
        arrival = arrival[positive]
        proton_interaction = proton_interaction[positive]
        proton_bx = proton_bx[positive]
        proton_counts = np.bincount(proton_bx, minlength=current_n_bx)
        proton_stops = np.cumsum(proton_counts)
        proton_starts = proton_stops - proton_counts
        batch_counts = {
            key: np.zeros(current_n_bx, dtype=np.int64) for key, _label in CRITERIA
        }

        for bx in np.flatnonzero(proton_counts):
            start, stop = proton_starts[bx], proton_stops[bx]
            counts, same_counts = _count_pair_categories(
                xi[start:stop],
                arm[start:stop],
                arrival[start:stop],
                proton_interaction[start:stop],
                flux.sqrt_s_gev,
                mass_range,
                central_z[bx],
                central_u[bx],
                z_cut,
                luminous_u_cut,
                central_u_cut,
            )
            for key, value in counts.items():
                batch_counts[key][bx] = value
            for key, value in same_counts.items():
                same_pair_sum[key] += value

        for key, values in batch_counts.items():
            accumulators[key]["pair_sum"] += int(np.sum(values))
            accumulators[key]["pair_square_sum"] += int(np.dot(values, values))
            accumulators[key]["passing_bx"] += int(np.count_nonzero(values))

        completed_bx = batch_start + current_n_bx
        if progress and (
            (batch_index + 1) % progress_every == 0 or completed_bx == n_bx
        ):
            elapsed = time.monotonic() - sampling_start
            eta = elapsed * (n_bx - completed_bx) / completed_bx
            print(
                f"  BX {completed_bx:,}/{n_bx:,} ({completed_bx / n_bx:.0%}); "
                f"elapsed {elapsed:.1f} s; ETA {eta:.1f} s",
                flush=True,
            )

    if progress:
        print("Finalizing probabilities, uncertainties, and rates...", flush=True)
    criteria = {
        key: {
            "label": label,
            **_finalize_criterion(
                accumulators[key]["pair_sum"],
                accumulators[key]["pair_square_sum"],
                accumulators[key]["passing_bx"],
                n_bx,
                bx_frequency_hz,
            ),
        }
        for key, label in CRITERIA
    }
    vertex_probabilities = {
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
            "artifact": str(flux.path) if flux.path is not None else None,
            "mu": float(mu),
            "n_bx": int(n_bx),
            "mass_range_gev": list(mass_range),
            "xi_resolution": float(xi_resolution),
            "beam_sigma_z_cm": float(beam_sigma_z_cm),
            "single_arm_time_resolution_ps": float(single_arm_time_resolution_ps),
            "pv_z_resolution_cm": float(pv_z_resolution_cm),
            "pv_time_resolution_ps": float(pv_time_resolution_ps),
            "nsigma": float(nsigma),
            "bx_frequency_hz": float(bx_frequency_hz),
            "seed": int(seed),
        },
        "sampling": {
            "n_inelastic_generated": int(flux.n_inelastic_generated),
            "truth_pps_interactions": n_relevant,
            "truth_pps_interaction_probability": float(relevant_probability),
            "sampled_truth_pps_interactions": int(sampled_interactions),
            "sampled_truth_pps_protons": int(sampled_protons),
        },
        "single_pair_unrelated_vertex_probability": vertex_probabilities,
        "same_interaction_diagnostics": {
            "mean_pps_pairs_per_bx": float(same_pair_sum["pps"] / n_bx),
            "mean_pps_mass_pairs_per_bx": float(same_pair_sum["pps_mass"] / n_bx),
        },
        "criteria": criteria,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mu", type=float, default=200.0)
    parser.add_argument("--bunch-crossings", type=int, default=1_000_000)
    parser.add_argument(
        "--mass-window", type=float, nargs=2, default=DEFAULT_MASS_RANGE
    )
    parser.add_argument("--xi-resolution", type=float, default=None)
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7)
    parser.add_argument("--pps-time-resolution-ps", type=float, default=10.0)
    parser.add_argument("--pv-z-resolution-cm", type=float, default=0.1)
    parser.add_argument("--pv-time-resolution-ps", type=float, default=30.0)
    parser.add_argument("--nsigma", type=float, default=2.0)
    parser.add_argument("--bx-frequency-mhz", type=float, default=31.6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON result path")
    return parser.parse_args()


def print_report(report):
    config = report["configuration"]
    print(
        f"Sampled {config['n_bx']:,} BX at mu={config['mu']:g}; "
        f"frequency={config['bx_frequency_hz'] / 1.0e6:g} MHz"
    )
    print(
        f"{'criterion':<38} {'mean pairs/BX':>23} {'passing BX':>11} "
        f"{'P(BX >= 1)':>23} {'rate [kHz]':>23}"
    )
    for key, _label in CRITERIA:
        result = report["criteria"][key]
        print(
            f"{result['label']:<38} "
            f"{result['mean_pairs_per_bx']:>11.6g} +/- "
            f"{result['mean_pairs_per_bx_error']:<8.2g} "
            f"{result['passing_bx']:>11d} "
            f"{result['bx_probability']:>11.6g} +/- "
            f"{result['bx_probability_error']:<8.2g} "
            f"{result['rate_hz'] / 1.0e3:>11.6g} +/- "
            f"{result['rate_error_hz'] / 1.0e3:<8.2g}"
        )
    same = report["same_interaction_diagnostics"]
    print(
        "Same-interaction mean pairs/BX: "
        f"PPS={same['mean_pps_pairs_per_bx']:.6g}, "
        f"PPS+mass={same['mean_pps_mass_pairs_per_bx']:.6g}"
    )


def main():
    args = parse_args()
    try:
        print(f"Loading configuration: {Path(args.config).resolve()}", flush=True)
        config = load_config(args.config)
        print(
            f"Loading and validating proton artifact: {Path(args.artifact).resolve()}",
            flush=True,
        )
        flux = ProtonFlux.load(args.artifact)
        print(
            f"Loaded {flux.xi.size:,} stored protons from "
            f"{flux.n_inelastic_generated:,} inelastic interactions.",
            flush=True,
        )
        report = study_trigger_rates(
            flux,
            Acceptance(config["xi_windows"]),
            mu=args.mu,
            n_bx=args.bunch_crossings,
            mass_range=args.mass_window,
            xi_resolution=(
                config["xi_resolution"]
                if args.xi_resolution is None
                else args.xi_resolution
            ),
            beam_sigma_z_cm=args.beam_sigma_z_cm,
            single_arm_time_resolution_ps=args.pps_time_resolution_ps,
            pv_z_resolution_cm=args.pv_z_resolution_cm,
            pv_time_resolution_ps=args.pv_time_resolution_ps,
            nsigma=args.nsigma,
            bx_frequency_hz=args.bx_frequency_mhz * 1.0e6,
            seed=config["seed"] if args.seed is None else args.seed,
            progress=True,
        )
        if args.output:
            print(f"Writing JSON report: {Path(args.output).resolve()}", flush=True)
            write_json(Path(args.output), report)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print_report(report)
    if args.output:
        print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
