"""Analytic PPS-to-primary-vertex overlap probabilities."""

import math
from functools import lru_cache


C_CM_PER_PS = 2.99792458e-2
TRUTH_STATUSES = ("matched", "unrelated")
TIMING_MODES = ("z_only", "luminous", "central")


def pps_vertex_resolution_cm(single_arm_time_resolution_ps):
    """PPS resolution on z and ct from two equal-resolution timing arms."""
    resolution = float(single_arm_time_resolution_ps)
    if resolution <= 0.0:
        raise ValueError("single_arm_time_resolution_ps must be positive")
    return C_CM_PER_PS * resolution / math.sqrt(2.0)


def _window_probability(cut, sigma):
    if sigma == 0.0:
        return 1.0
    return math.erf(cut / (math.sqrt(2.0) * sigma))


def vertex_overlap_probability(
    *,
    truth_status,
    timing_mode="central",
    beam_sigma_z_cm=5.7,
    single_arm_time_resolution_ps=10.0,
    pv_z_resolution_cm=0.001,
    pv_time_resolution_ps=30.0,
    nsigma=2.0,
):
    """Probability that a reconstructed PPS vertex passes rectangular cuts.

    ``truth_status`` is ``"matched"`` when both protons originate from the
    central primary vertex and ``"unrelated"`` when the PPS-implied vertex is
    independent of it. ``timing_mode="z_only"`` applies only the z match,
    ``"luminous"`` also requires the PPS-implied ct to lie in the luminous
    region, and ``"central"`` compares it with a measured primary-vertex time.
    """
    if truth_status not in TRUTH_STATUSES:
        raise ValueError(f"truth_status must be one of {TRUTH_STATUSES}")
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"timing_mode must be one of {TIMING_MODES}")

    beam_sigma = float(beam_sigma_z_cm)
    pv_z_sigma = float(pv_z_resolution_cm)
    nsigma = float(nsigma)
    if beam_sigma < 0.0:
        raise ValueError("beam_sigma_z_cm must be non-negative")
    if pv_z_sigma < 0.0:
        raise ValueError("pv_z_resolution_cm must be non-negative")
    if nsigma <= 0.0:
        raise ValueError("nsigma must be positive")

    pps_sigma = pps_vertex_resolution_cm(single_arm_time_resolution_ps)
    z_measurement_variance = pps_sigma**2 + pv_z_sigma**2
    z_variance = z_measurement_variance
    if truth_status == "unrelated":
        z_variance += 2.0 * beam_sigma**2
    z_probability = _window_probability(
        nsigma * math.sqrt(z_measurement_variance), math.sqrt(z_variance)
    )

    if timing_mode == "z_only":
        return z_probability
    if timing_mode == "luminous":
        luminous_probability = _window_probability(
            nsigma * beam_sigma, math.sqrt(beam_sigma**2 + pps_sigma**2)
        )
        return z_probability * luminous_probability

    if pv_time_resolution_ps is None:
        raise ValueError("pv_time_resolution_ps is required for central timing")
    pv_time_sigma = float(pv_time_resolution_ps)
    if pv_time_sigma <= 0.0:
        raise ValueError("pv_time_resolution_ps must be positive")
    time_measurement_variance = pps_sigma**2 + (C_CM_PER_PS * pv_time_sigma) ** 2
    time_variance = time_measurement_variance
    if truth_status == "unrelated":
        time_variance += 2.0 * beam_sigma**2
    time_probability = _window_probability(
        nsigma * math.sqrt(time_measurement_variance), math.sqrt(time_variance)
    )
    return z_probability * time_probability


def vertex_likelihood_parameters(
    *,
    beam_sigma_z_cm=5.7,
    single_arm_time_resolution_ps=10.0,
    pv_z_resolution_cm=0.001,
    pv_time_resolution_ps=30.0,
):
    """Widths and coefficients of the exact matched/unrelated Gaussian LR."""
    beam_sigma = float(beam_sigma_z_cm)
    pv_z_sigma = float(pv_z_resolution_cm)
    pv_time_sigma = float(pv_time_resolution_ps)
    if beam_sigma <= 0.0:
        raise ValueError("beam_sigma_z_cm must be positive for a vertex likelihood")
    if pv_z_sigma < 0.0:
        raise ValueError("pv_z_resolution_cm must be non-negative")
    if pv_time_sigma <= 0.0:
        raise ValueError("pv_time_resolution_ps must be positive")

    pps_sigma = pps_vertex_resolution_cm(single_arm_time_resolution_ps)
    matched_z = math.hypot(pps_sigma, pv_z_sigma)
    matched_ct = math.hypot(pps_sigma, C_CM_PER_PS * pv_time_sigma)
    unrelated_z = math.sqrt(matched_z**2 + 2.0 * beam_sigma**2)
    unrelated_ct = math.sqrt(matched_ct**2 + 2.0 * beam_sigma**2)
    coefficient_z = 1.0 / matched_z**2 - 1.0 / unrelated_z**2
    coefficient_ct = 1.0 / matched_ct**2 - 1.0 / unrelated_ct**2
    return {
        "matched_sigma_z_cm": matched_z,
        "matched_sigma_ct_cm": matched_ct,
        "unrelated_sigma_z_cm": unrelated_z,
        "unrelated_sigma_ct_cm": unrelated_ct,
        "quadratic_coefficient_z_cm2_inv": coefficient_z,
        "quadratic_coefficient_ct_cm2_inv": coefficient_ct,
        "maximum_log_likelihood_ratio": math.log(
            unrelated_z * unrelated_ct / (matched_z * matched_ct)
        ),
    }


def vertex_log_likelihood_ratio(dz_cm, dct_cm, **resolution):
    """Exact ``log(p_matched / p_unrelated)`` for measured dz and dct."""
    import numpy as np

    parameters = vertex_likelihood_parameters(**resolution)
    dz, dct = np.broadcast_arrays(
        np.asarray(dz_cm, dtype=np.float64),
        np.asarray(dct_cm, dtype=np.float64),
    )
    quadratic = (
        parameters["quadratic_coefficient_z_cm2_inv"] * dz**2
        + parameters["quadratic_coefficient_ct_cm2_inv"] * dct**2
    )
    return parameters["maximum_log_likelihood_ratio"] - 0.5 * quadratic


@lru_cache(maxsize=None)
def _angular_quadrature(order):
    import numpy as np

    if order < 16:
        raise ValueError("quadrature_order must be at least 16")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angles = 0.25 * math.pi * (nodes + 1.0)
    weights = 0.25 * math.pi * weights
    return angles, weights


def _quadratic_cdf(values, scale_z, scale_ct, quadrature_order):
    """CDF of ``scale_z*X**2 + scale_ct*Y**2`` for standard normals."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    output = np.zeros(values.shape, dtype=np.float64)
    output[np.isposinf(values)] = 1.0
    selected = np.isfinite(values) & (values > 0.0)
    if not np.any(selected):
        return output
    if scale_z <= 0.0 or scale_ct <= 0.0:
        raise ValueError("quadratic scales must be positive")
    if math.isclose(scale_z, scale_ct, rel_tol=1.0e-14):
        output[selected] = -np.expm1(-values[selected] / (2.0 * scale_z))
        return output

    angles, weights = _angular_quadrature(int(quadrature_order))
    radial_scale = (
        scale_z * np.cos(angles) ** 2 + scale_ct * np.sin(angles) ** 2
    )
    survival = (2.0 / math.pi) * np.sum(
        weights[:, np.newaxis]
        * np.exp(-values[selected][np.newaxis, :] / (2.0 * radial_scale[:, np.newaxis])),
        axis=0,
    )
    output[selected] = np.clip(1.0 - survival, 0.0, 1.0)
    return output


def _quadratic_scales(parameters, truth_status):
    if truth_status not in TRUTH_STATUSES:
        raise ValueError(f"truth_status must be one of {TRUTH_STATUSES}")
    prefix = "matched" if truth_status == "matched" else "unrelated"
    return (
        parameters["quadratic_coefficient_z_cm2_inv"]
        * parameters[f"{prefix}_sigma_z_cm"] ** 2,
        parameters["quadratic_coefficient_ct_cm2_inv"]
        * parameters[f"{prefix}_sigma_ct_cm"] ** 2,
    )


def _quadratic_quantiles(probabilities, scales, quadrature_order):
    import numpy as np

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("quadratic quantiles require probabilities inside (0, 1)")
    scale_max = max(scales)
    low = np.zeros(probabilities.shape, dtype=np.float64)
    high = -2.0 * scale_max * np.log1p(-probabilities)
    for _iteration in range(60):
        middle = 0.5 * (low + high)
        below = _quadratic_cdf(
            middle, scales[0], scales[1], quadrature_order
        ) < probabilities
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    return 0.5 * (low + high)


def vertex_likelihood_table(
    *,
    bins=8,
    quadratic_edges=None,
    quadrature_order=96,
    beam_sigma_z_cm=5.7,
    single_arm_time_resolution_ps=10.0,
    pv_z_resolution_cm=0.001,
    pv_time_resolution_ps=30.0,
):
    """Analytic bin fractions and discrete log LR for the timing likelihood.

    When edges are omitted, bins have equal probability under the matched
    hypothesis.  The returned per-bin score is the exact discrete likelihood
    ratio ``log(P_matched / P_unrelated)`` rather than a value sampled inside
    the bin.
    """
    import numpy as np

    if int(bins) != bins or bins < 2:
        raise ValueError("bins must be an integer of at least two")
    parameters = vertex_likelihood_parameters(
        beam_sigma_z_cm=beam_sigma_z_cm,
        single_arm_time_resolution_ps=single_arm_time_resolution_ps,
        pv_z_resolution_cm=pv_z_resolution_cm,
        pv_time_resolution_ps=pv_time_resolution_ps,
    )
    matched_scales = _quadratic_scales(parameters, "matched")
    if quadratic_edges is None:
        interior = _quadratic_quantiles(
            np.arange(1, bins, dtype=np.float64) / bins,
            matched_scales,
            quadrature_order,
        )
        edges = np.r_[0.0, interior, np.inf]
    else:
        edges = np.asarray(quadratic_edges, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 3
            or edges[0] != 0.0
            or not np.isposinf(edges[-1])
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError(
                "quadratic_edges must increase strictly from zero to positive infinity"
            )
        bins = edges.size - 1

    probabilities = {}
    for truth_status in TRUTH_STATUSES:
        scales = _quadratic_scales(parameters, truth_status)
        cdf = _quadratic_cdf(edges, scales[0], scales[1], quadrature_order)
        values = np.diff(cdf)
        values = np.clip(values, 0.0, 1.0)
        values /= values.sum()
        probabilities[truth_status] = values
    with np.errstate(divide="ignore"):
        discrete_log_lr = np.log(
            probabilities["matched"] / probabilities["unrelated"]
        )
    return {
        "quadratic_edges": edges,
        "matched_probability": probabilities["matched"],
        "unrelated_probability": probabilities["unrelated"],
        "log_likelihood_ratio": discrete_log_lr,
        "parameters": parameters,
        "quadrature_order": int(quadrature_order),
    }
