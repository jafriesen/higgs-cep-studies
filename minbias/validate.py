#!/usr/bin/env python3
"""Validate a minbias proton artifact and analytic pair-density closure."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from minbias.artifact import DEFAULT_CONFIG, load_config, metadata_path, read_json, write_json
from minbias.flux import Acceptance, PairDensity, ProtonFlux, Resolution


REFERENCE_MASS_RANGE = (117.0, 133.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--unfiltered-control", default=None)
    parser.add_argument("--expected-events", type=int, default=None)
    parser.add_argument("--legacy-pool", default=None)
    parser.add_argument("--bruteforce-per-arm", type=int, default=500)
    parser.add_argument("--report", default=None, help="Optionally write the validation JSON")
    return parser.parse_args()


def assert_dense_events(flux):
    unique = np.unique(flux.event)
    expected = np.arange(int(flux.metadata["n_events_written"]), dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise RuntimeError("Artifact event IDs are not dense over written interactions")


def compare_unfiltered(filtered_path, unfiltered_path, config):
    filtered = pq.read_table(filtered_path)
    unfiltered = pq.read_table(unfiltered_path)
    filtered_metadata = read_json(metadata_path(filtered_path))
    unfiltered_metadata = read_json(metadata_path(unfiltered_path))
    for key in ("seed", "n_inelastic_generated"):
        if filtered_metadata[key] != unfiltered_metadata[key]:
            raise RuntimeError(f"Filtered/control metadata differ in {key}")
    if filtered_metadata["cross_sections"]["processes"] != unfiltered_metadata["cross_sections"]["processes"]:
        raise RuntimeError("Filtered/control process counts differ")

    xi = np.asarray(unfiltered["xi"], dtype=np.float64)
    low, high = config["store_window"]
    selected = (xi > low) & (xi < high)
    control_event = np.asarray(unfiltered["event"], dtype=np.int64)[selected]
    _events, compressed = np.unique(control_event, return_inverse=True)
    if not np.array_equal(compressed.astype(np.int32), np.asarray(filtered["event"])):
        raise RuntimeError("Filtered event grouping is not the offline-filtered control grouping")
    for name in ("arm", "xi", "px", "py", "process"):
        if not np.array_equal(np.asarray(unfiltered[name])[selected], np.asarray(filtered[name])):
            raise RuntimeError(f"Filtered column {name} differs from the same-seed control")
    return {
        "unfiltered_protons": unfiltered.num_rows,
        "offline_selected_protons": int(np.sum(selected)),
        "filtered_protons": filtered.num_rows,
    }


def make_subset_flux(flux, acceptance, per_arm):
    selected = acceptance.mask(flux.xi)
    indices = []
    for sign in (-1, 1):
        arm_indices = np.flatnonzero(selected & (flux.arm == sign))[:per_arm]
        indices.append(arm_indices)
    indices = np.concatenate(indices)
    metadata = dict(flux.metadata)
    metadata["n_protons_written"] = int(indices.size)
    return ProtonFlux(
        flux.event[indices],
        flux.arm[indices],
        flux.xi[indices],
        flux.px[indices],
        flux.py[indices],
        flux.process[indices],
        metadata,
    )


def brute_pair_intensity(flux, acceptance, mass_range, yx_range=None):
    selected = acceptance.mask(flux.xi)
    left_mask = selected & (flux.arm < 0)
    right_mask = selected & (flux.arm > 0)
    left = np.asarray(flux.xi[left_mask], dtype=np.float64)
    right = np.asarray(flux.xi[right_mask], dtype=np.float64)
    if left.size == 0 or right.size == 0:
        return 0.0
    mx = np.sqrt(left[:, None] * right[None, :]) * flux.sqrt_s_gev
    distinct = flux.event[left_mask, None] != flux.event[right_mask][None, :]
    keep = distinct & (mx >= mass_range[0]) & (mx < mass_range[1])
    if yx_range is not None:
        yx = 0.5 * np.log(right[None, :] / left[:, None])
        keep &= (yx >= yx_range[0]) & (yx < yx_range[1])
    denominator = flux.n_inelastic_generated * (flux.n_inelastic_generated - 1)
    return float(np.sum(keep) / denominator) if denominator else 0.0


def brute_mass_histogram(flux, acceptance, mass_bins):
    selected = acceptance.mask(flux.xi)
    left_mask = selected & (flux.arm < 0)
    right_mask = selected & (flux.arm > 0)
    left = np.asarray(flux.xi[left_mask], dtype=np.float64)
    right = np.asarray(flux.xi[right_mask], dtype=np.float64)
    if left.size == 0 or right.size == 0 or flux.n_inelastic_generated < 2:
        return np.zeros(len(mass_bins) - 1, dtype=np.float64)
    mx = np.sqrt(left[:, None] * right[None, :]) * flux.sqrt_s_gev
    distinct = flux.event[left_mask, None] != flux.event[right_mask][None, :]
    counts, _ = np.histogram(mx[distinct], bins=mass_bins)
    denominator = flux.n_inelastic_generated * (flux.n_inelastic_generated - 1)
    return counts.astype(np.float64) / denominator


def pair_closure(flux, acceptance, per_arm):
    subset = make_subset_flux(flux, acceptance, per_arm)
    density = PairDensity(subset, acceptance, Resolution(0.0, 12345), bins=2048)
    checks = {}
    for label, yx_range in (("unrestricted", None), ("central_yx", (-0.2, 0.2))):
        analytic = density.integrate(REFERENCE_MASS_RANGE, yx_range)
        brute = brute_pair_intensity(subset, acceptance, REFERENCE_MASS_RANGE, yx_range)
        relative = abs(analytic - brute) / brute if brute else abs(analytic - brute)
        checks[label] = {"analytic": analytic, "brute": brute, "relative_difference": relative}
        if relative > 0.08:
            raise RuntimeError(f"Pair-density closure failed for {label}: relative difference {relative:.3g}")

    mass_bins = np.arange(117.0, 134.0, 1.0)
    fft_mass = density.marginal_mass(mass_bins)
    rectangle_mass = np.asarray(
        [density.integrate((low, high)) for low, high in zip(mass_bins[:-1], mass_bins[1:])]
    )
    brute_mass = brute_mass_histogram(subset, acceptance, mass_bins)
    absolute_difference = float(np.max(np.abs(fft_mass - rectangle_mass)))
    scale = float(np.max(np.abs(rectangle_mass)))
    fft_difference = absolute_difference / scale if scale > 1.0e-15 else 0.0
    if (scale > 1.0e-15 and fft_difference > 0.08) or absolute_difference > 1.0e-12:
        raise RuntimeError(
            "FFT/direct mass marginal closure failed: "
            f"relative={fft_difference:.3g}, absolute={absolute_difference:.3g}"
        )
    brute_scale = float(np.max(np.abs(brute_mass)))
    brute_difference = (
        float(np.max(np.abs(fft_mass - brute_mass))) / brute_scale
        if brute_scale > 1.0e-15
        else float(np.max(np.abs(fft_mass - brute_mass)))
    )
    if brute_difference > 0.08:
        raise RuntimeError(
            "FFT/brute-force mass marginal closure failed: "
            f"relative-to-peak={brute_difference:.3g}"
        )
    checks["fft_max_relative_to_peak"] = fft_difference
    checks["fft_max_absolute_difference"] = absolute_difference
    checks["fft_bruteforce_max_relative_to_peak"] = brute_difference
    return checks


def legacy_diagnostic(density, pool_path):
    table = pq.read_table(pool_path, columns=["mx", "weight"])
    bins = np.arange(117.0, 134.0, 1.0)
    old, _ = np.histogram(
        np.asarray(table["mx"], dtype=np.float64),
        bins=bins,
        weights=np.asarray(table["weight"], dtype=np.float64),
    )
    new = density.marginal_mass(bins)
    if old.sum() <= 0.0 or new.sum() <= 0.0:
        return {"available": False}
    old /= old.sum()
    new /= new.sum()
    return {
        "available": True,
        "total_variation_1gev_mass_shape": float(0.5 * np.sum(np.abs(old - new))),
    }


def validate(args):
    config = load_config(args.config)
    acceptance = Acceptance(config["xi_windows"])
    resolution = Resolution(config["xi_resolution"], config["seed"])
    flux = ProtonFlux.load(args.artifact)
    if args.expected_events is not None and flux.n_inelastic_generated != args.expected_events:
        raise RuntimeError(
            f"Artifact denominator {flux.n_inelastic_generated} != expected {args.expected_events}"
        )
    if flux.metadata["pythia"]["elastic_enabled"]:
        raise RuntimeError("Artifact metadata reports elastic production enabled")
    process_events = sum(
        int(process["events"])
        for process in flux.metadata["cross_sections"]["processes"].values()
    )
    if process_events != flux.n_inelastic_generated:
        raise RuntimeError("Per-process event counts do not sum to the inelastic denominator")
    assert_dense_events(flux)

    statistics = flux.arm_statistics(acceptance)
    zero = PairDensity(flux, acceptance, Resolution(0.0, config["seed"]), bins=config["log_xi_bins"])
    du = zero.du
    intensity_integrals = {
        "left": float(np.sum(zero.left) * du),
        "right": float(np.sum(zero.right) * du),
    }
    for arm in ("left", "right"):
        expected = statistics[arm]["mean_multiplicity"]
        if not np.isclose(intensity_integrals[arm], expected, rtol=1e-12, atol=1e-15):
            raise RuntimeError(f"Zero-resolution {arm} intensity does not reproduce direct counting")

    first = PairDensity(flux, acceptance, resolution, bins=min(config["log_xi_bins"], 2048))
    second = PairDensity(flux, acceptance, resolution, bins=min(config["log_xi_bins"], 2048))
    if not np.array_equal(first.left, second.left) or not np.array_equal(first.right, second.right):
        raise RuntimeError("Resolution smearing is not deterministic")

    report = {
        "artifact": str(Path(args.artifact).resolve()),
        "denominator": flux.n_inelastic_generated,
        "protons": int(flux.xi.size),
        "arm_statistics": {
            arm: statistics[arm] for arm in ("left", "right")
        },
        "joint_probability": statistics["joint_probability"].tolist(),
        "zero_resolution_intensity_integrals": intensity_integrals,
        "resolution_diagnostics": first.diagnostics,
        "pair_closure": pair_closure(flux, acceptance, args.bruteforce_per_arm),
        "reference_expected_pairs_poisson_mu200": float(
            first.expected_pairs_poisson(200.0, REFERENCE_MASS_RANGE)
        ),
    }
    if args.unfiltered_control:
        report["filtered_control"] = compare_unfiltered(
            Path(args.artifact), Path(args.unfiltered_control), config
        )
    if args.legacy_pool:
        report["legacy_diagnostic"] = legacy_diagnostic(first, args.legacy_pool)
    return report


def main():
    args = parse_args()
    if args.bruteforce_per_arm <= 0:
        raise SystemExit("ERROR: --bruteforce-per-arm must be positive")
    try:
        report = validate(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Validation passed")


if __name__ == "__main__":
    main()
