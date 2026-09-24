"""Shared correlated PPS pileup sampling for trigger studies."""

import math

import numpy as np

from minbias.vertex import C_CM_PER_PS, pps_vertex_resolution_cm


CRITERIA = (
    ("pps", "PPS"),
    ("pps_mass", "PPS + mass"),
    ("pps_z", "PPS + PV z"),
    ("pps_mass_z", "PPS + mass + PV z"),
    ("pps_z_luminous", "PPS + PV z + luminous time"),
    ("pps_mass_z_luminous", "PPS + mass + PV z + luminous time"),
    ("pps_z_central", "PPS + PV z + central time"),
    ("pps_mass_z_central", "PPS + mass + PV z + central time"),
)


def accepted_groups(flux, acceptance):
    """Group truth-accepted protons by their source inelastic interaction."""
    selected = acceptance.mask(flux.xi)
    event = np.asarray(flux.event[selected], dtype=np.int64)
    xi = np.asarray(flux.xi[selected], dtype=np.float64)
    arm = np.asarray(flux.arm[selected], dtype=np.int8)
    if event.size and np.any(event[1:] < event[:-1]):
        order = np.argsort(event, kind="stable")
        event, xi, arm = event[order], xi[order], arm[order]
    _unique, starts = np.unique(event, return_index=True)
    counts = np.diff(np.r_[starts, event.size])
    return {
        "xi": xi,
        "arm": arm,
        "starts": starts,
        "counts": counts,
        "probability": float(starts.size / flux.n_inelastic_generated),
    }


def vertex_cuts(
    *,
    beam_sigma_z_cm,
    single_arm_time_resolution_ps,
    pv_z_resolution_cm,
    pv_time_resolution_ps,
    nsigma,
):
    """Return rectangular PPS vertex cuts in cm (or equivalently ct)."""
    pps_sigma = pps_vertex_resolution_cm(single_arm_time_resolution_ps)
    return {
        "z": float(nsigma) * math.sqrt(pps_sigma**2 + float(pv_z_resolution_cm) ** 2),
        "luminous": float(nsigma) * float(beam_sigma_z_cm),
        "central": float(nsigma)
        * math.sqrt(pps_sigma**2 + (C_CM_PER_PS * float(pv_time_resolution_ps)) ** 2),
    }


def pair_observables(
    xi,
    arm,
    arrival,
    interaction,
    proton_is_signal=None,
    *,
    sqrt_s_gev,
    mass_range,
    pv_z,
    pv_u,
    cuts,
):
    """Build every reconstructed left-right pair from one sampled BX."""
    left = arm < 0
    right = arm > 0
    if not np.any(left) or not np.any(right):
        empty_float = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=bool)
        return {
            "mass": empty_float,
            "rapidity": empty_float,
            "mass_ok": empty_bool,
            "z_ok": empty_bool,
            "luminous_ok": empty_bool,
            "central_ok": empty_bool,
            "same_interaction": empty_bool,
            "signal_signal": empty_bool,
            "signal_pileup": empty_bool,
            "pileup_pileup": empty_bool,
        }

    if proton_is_signal is None:
        proton_is_signal = np.zeros(arm.size, dtype=bool)
    proton_is_signal = np.asarray(proton_is_signal, dtype=bool)
    if proton_is_signal.shape != arm.shape:
        raise ValueError("proton_is_signal must align with proton arrays")

    xi_left = xi[left, None]
    xi_right = xi[None, right]
    mass = np.sqrt(xi_left * xi_right) * float(sqrt_s_gev)
    rapidity = 0.5 * np.log(xi_right / xi_left)
    left_arrival = arrival[left, None]
    right_arrival = arrival[None, right]
    pp_z = 0.5 * (left_arrival - right_arrival)
    pp_u = 0.5 * (left_arrival + right_arrival)
    same = interaction[left, None] == interaction[None, right]
    left_signal = proton_is_signal[left, None]
    right_signal = proton_is_signal[None, right]
    low, high = mass_range
    return {
        "mass": mass.ravel(),
        "rapidity": rapidity.ravel(),
        "mass_ok": ((mass >= low) & (mass < high)).ravel(),
        "z_ok": (np.abs(pp_z - pv_z) <= cuts["z"]).ravel(),
        "luminous_ok": (np.abs(pp_u) <= cuts["luminous"]).ravel(),
        "central_ok": (np.abs(pp_u - pv_u) <= cuts["central"]).ravel(),
        "same_interaction": same.ravel(),
        "signal_signal": (left_signal & right_signal).ravel(),
        "signal_pileup": (left_signal ^ right_signal).ravel(),
        "pileup_pileup": (~left_signal & ~right_signal).ravel(),
    }


def criterion_masks(pairs):
    """Return the eight standard PPS criterion masks for pair observables."""
    count = pairs["mass"].size
    all_pairs = np.ones(count, dtype=bool)
    mass = pairs["mass_ok"]
    z = pairs["z_ok"]
    luminous = pairs["luminous_ok"]
    central = pairs["central_ok"]
    return {
        "pps": all_pairs,
        "pps_mass": mass,
        "pps_z": z,
        "pps_mass_z": mass & z,
        "pps_z_luminous": z & luminous,
        "pps_mass_z_luminous": mass & z & luminous,
        "pps_z_central": z & central,
        "pps_mass_z_central": mass & z & central,
    }


def sample_pps_pairs(
    groups,
    n_inelastic,
    rng,
    *,
    sqrt_s_gev,
    mass_range,
    xi_resolution,
    beam_sigma_z_cm,
    single_arm_time_resolution_ps,
    pv_z_resolution_cm,
    pv_time_resolution_ps,
    nsigma,
    matched_xi=None,
    matched_arm=None,
):
    """Sample PPS pairs for exact pileup, optionally injecting matched protons.

    ``matched_xi`` values must already pass truth-level PPS acceptance. Their
    common production vertex is the central primary vertex.
    """
    if n_inelastic < 0:
        raise ValueError("n_inelastic must be non-negative")
    n_relevant = int(rng.binomial(int(n_inelastic), groups["probability"]))
    n_groups = int(groups["starts"].size)
    pv_truth_z = rng.normal(0.0, beam_sigma_z_cm)
    pv_truth_u = rng.normal(0.0, beam_sigma_z_cm)
    pv_z = pv_truth_z + rng.normal(0.0, pv_z_resolution_cm)
    pv_u = pv_truth_u + C_CM_PER_PS * rng.normal(0.0, pv_time_resolution_ps)

    xi_parts = []
    arm_parts = []
    arrival_parts = []
    interaction_parts = []
    signal_parts = []
    if n_relevant and n_groups:
        chosen = rng.integers(0, n_groups, size=n_relevant)
        interaction_z = rng.normal(0.0, beam_sigma_z_cm, size=n_relevant)
        interaction_u = rng.normal(0.0, beam_sigma_z_cm, size=n_relevant)
        rows = groups["counts"][chosen]
        n_protons = int(np.sum(rows))
        starts = np.repeat(groups["starts"][chosen], rows)
        offsets = np.concatenate([np.arange(count) for count in rows])
        source = starts + offsets
        proton_interaction = np.repeat(np.arange(n_relevant), rows)
        pileup_arm = groups["arm"][source]
        pileup_xi = groups["xi"][source] + float(xi_resolution) * rng.standard_normal(
            n_protons
        )
        pileup_arrival = (
            interaction_u[proton_interaction]
            - pileup_arm * interaction_z[proton_interaction]
            + C_CM_PER_PS
            * rng.normal(0.0, single_arm_time_resolution_ps, size=n_protons)
        )
        positive = pileup_xi > 0.0
        xi_parts.append(pileup_xi[positive])
        arm_parts.append(pileup_arm[positive])
        arrival_parts.append(pileup_arrival[positive])
        interaction_parts.append(proton_interaction[positive])
        signal_parts.append(np.zeros(np.count_nonzero(positive), dtype=bool))

    if matched_xi is not None or matched_arm is not None:
        if matched_xi is None or matched_arm is None:
            raise ValueError("matched_xi and matched_arm must be provided together")
        signal_xi = np.asarray(matched_xi, dtype=np.float64)
        signal_arm = np.asarray(matched_arm, dtype=np.int8)
        if signal_xi.ndim != 1 or signal_xi.shape != signal_arm.shape:
            raise ValueError(
                "matched proton arrays must be aligned and one-dimensional"
            )
        if np.any((signal_arm != -1) & (signal_arm != 1)):
            raise ValueError("matched proton arms must be -1 or +1")
        signal_xi = signal_xi + float(xi_resolution) * rng.standard_normal(
            signal_xi.size
        )
        signal_arrival = (
            pv_truth_u
            - signal_arm * pv_truth_z
            + C_CM_PER_PS
            * rng.normal(0.0, single_arm_time_resolution_ps, size=signal_xi.size)
        )
        positive = signal_xi > 0.0
        xi_parts.append(signal_xi[positive])
        arm_parts.append(signal_arm[positive])
        arrival_parts.append(signal_arrival[positive])
        interaction_parts.append(
            np.full(np.count_nonzero(positive), -1, dtype=np.int64)
        )
        signal_parts.append(np.ones(np.count_nonzero(positive), dtype=bool))

    xi = np.concatenate(xi_parts) if xi_parts else np.empty(0, dtype=np.float64)
    arm = np.concatenate(arm_parts) if arm_parts else np.empty(0, dtype=np.int8)
    arrival = (
        np.concatenate(arrival_parts)
        if arrival_parts
        else np.empty(0, dtype=np.float64)
    )
    proton_interaction = (
        np.concatenate(interaction_parts)
        if interaction_parts
        else np.empty(0, dtype=np.int64)
    )
    proton_is_signal = (
        np.concatenate(signal_parts) if signal_parts else np.empty(0, dtype=bool)
    )
    cuts = vertex_cuts(
        beam_sigma_z_cm=beam_sigma_z_cm,
        single_arm_time_resolution_ps=single_arm_time_resolution_ps,
        pv_z_resolution_cm=pv_z_resolution_cm,
        pv_time_resolution_ps=pv_time_resolution_ps,
        nsigma=nsigma,
    )
    return pair_observables(
        xi,
        arm,
        arrival,
        proton_interaction,
        proton_is_signal,
        sqrt_s_gev=sqrt_s_gev,
        mass_range=mass_range,
        pv_z=pv_z,
        pv_u=pv_u,
        cuts=cuts,
    )
