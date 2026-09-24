#!/usr/bin/env python3
"""Shared vectorized API for saved JetPUPPI corrections and L1 smearing."""

import math
from pathlib import Path


SCHEMA_VERSION = 4
# 5 adds the FSR-recovered object: same layout, plus a "recovery" block that tells
# the feature builder which jets and muons to add back before applying the factors.
SUPPORTED_SCHEMA_VERSIONS = (4, 5)


def load_correction_map(path, fsr_state, source_sample):
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported correction schema {document.get('schema_version')!r}; "
            f"expected one of {SUPPORTED_SCHEMA_VERSIONS}"
        )
    try:
        return document["maps"][fsr_state][source_sample]
    except KeyError as error:
        raise KeyError(
            f"No correction map for fsr_state={fsr_state!r}, "
            f"source_sample={source_sample!r}"
        ) from error


def correction_factors(pt, eta, correction_map):
    """Return continuous raw-pT factors and a validity mask."""
    import numpy as np

    pt_array, eta_array = np.broadcast_arrays(
        np.asarray(pt, dtype=np.float64), np.asarray(eta, dtype=np.float64)
    )
    factors = np.full(pt_array.shape, np.nan, dtype=np.float64)
    valid = np.zeros(pt_array.shape, dtype=bool)
    abs_eta = np.abs(eta_array)
    support_min, support_max = correction_map["raw_pt_support"]

    for eta_bin in correction_map["eta_bins"]:
        nodes = [node for node in eta_bin["nodes"] if node.get("usable", False)]
        if not eta_bin.get("supported", False) or len(nodes) < 2:
            continue
        raw_nodes = np.asarray(
            [node["raw_reco_pt_median"] for node in nodes], dtype=np.float64
        )
        target_nodes = np.asarray(
            [node["target_pt"] for node in nodes], dtype=np.float64
        )
        if np.any(np.diff(raw_nodes) <= 0.0) or np.any(np.diff(target_nodes) <= 0.0):
            raise ValueError("Correction-map nodes must be strictly increasing")
        mask = (
            (abs_eta >= eta_bin["eta_min"])
            & (abs_eta < eta_bin["eta_max"])
            & (pt_array >= support_min)
            & (pt_array < support_max)
        )
        raw_pt = pt_array[mask]
        target_pt = np.interp(raw_pt, raw_nodes, target_nodes)
        below = raw_pt < raw_nodes[0]
        above = raw_pt > raw_nodes[-1]
        target_pt[below] = raw_pt[below] * target_nodes[0] / raw_nodes[0]
        target_pt[above] = raw_pt[above] * target_nodes[-1] / raw_nodes[-1]
        factor = target_pt / raw_pt
        factors[mask] = factor
        valid[mask] = np.isfinite(factor) & (factor > 0.0)
    return factors, valid


def correct_jet_kinematics(pt, eta, phi, mass, correction_map):
    """Scale pT and mass while preserving eta and phi."""
    import numpy as np

    pt_array, eta_array, phi_array, mass_array = np.broadcast_arrays(
        np.asarray(pt, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
        np.asarray(mass, dtype=np.float64),
    )
    factors, valid = correction_factors(pt_array, eta_array, correction_map)
    return {
        "pt": np.where(valid, pt_array * factors, np.nan),
        "eta": eta_array.copy(),
        "phi": phi_array.copy(),
        "mass": np.where(valid, mass_array * factors, np.nan),
        "factor": factors,
        "valid": valid,
    }


def l1_relative_sigma(pt, eta, resolution_config):
    """Evaluate sigma(pT)/pT = a + b/pT in the configured absolute-eta bins."""
    import numpy as np

    pt_array, eta_array = np.broadcast_arrays(
        np.asarray(pt, dtype=np.float64), np.asarray(eta, dtype=np.float64)
    )
    sigma = np.full(pt_array.shape, np.nan, dtype=np.float64)
    valid = np.zeros(pt_array.shape, dtype=bool)
    abs_eta = np.abs(eta_array)
    for eta_bin in resolution_config["resolutions"]:
        mask = (
            (abs_eta >= eta_bin["eta_min"])
            & (abs_eta < eta_bin["eta_max"])
            & (pt_array > 0.0)
        )
        sigma[mask] = eta_bin["a"] + eta_bin["b"] / pt_array[mask]
        valid[mask] = np.isfinite(sigma[mask]) & (sigma[mask] >= 0.0)
    return sigma, valid


def smear_gen_jets(pt, mass, eta, rng, resolution_config, replicas=1):
    """Apply independent unit-mean log-normal L1 response factors."""
    import numpy as np

    if replicas <= 0:
        raise ValueError("replicas must be positive")
    pt_array, mass_array, eta_array = np.broadcast_arrays(
        np.asarray(pt, dtype=np.float64),
        np.asarray(mass, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
    )
    relative_sigma, valid = l1_relative_sigma(pt_array, eta_array, resolution_config)
    sigma_log = np.sqrt(np.log1p(relative_sigma * relative_sigma))
    mu_log = -0.5 * sigma_log * sigma_log
    shape = (replicas, *pt_array.shape)
    normal = rng.normal(size=shape)
    factors = np.exp(mu_log[np.newaxis, ...] + sigma_log[np.newaxis, ...] * normal)
    factors = np.where(valid[np.newaxis, ...], factors, np.nan)
    return {
        "pt": factors * pt_array[np.newaxis, ...],
        "mass": factors * mass_array[np.newaxis, ...],
        "factor": factors,
        "valid": np.broadcast_to(valid, shape).copy(),
    }


def response_metrics(values):
    """Return point estimates used throughout the calibration study."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "entries": 0,
            "median": None,
            "q16": None,
            "q84": None,
            "central68_resolution": None,
            "rms_over_mean": None,
        }
    q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    mean = float(np.mean(values))
    return {
        "entries": int(values.size),
        "median": float(median),
        "q16": float(q16),
        "q84": float(q84),
        "central68_resolution": (
            float(0.5 * (q84 - q16) / median) if median != 0.0 else None
        ),
        "rms_over_mean": (
            float(np.std(values) / mean) if mean != 0.0 else None
        ),
    }


def response_metrics_with_ci(values, rng, replicas=200, include_ci=True):
    """Return response metrics and paired-bootstrap 68% intervals."""
    import numpy as np

    metrics = response_metrics(values)
    metrics["closure_bias"] = (
        metrics["median"] - 1.0 if metrics["median"] is not None else None
    )
    for field in (
        "median_ci68",
        "closure_bias_ci68",
        "central68_resolution_ci68",
        "rms_over_mean_ci68",
    ):
        metrics[field] = [None, None]
    if not include_ci or metrics["entries"] == 0:
        return metrics

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    medians = np.empty(replicas, dtype=np.float64)
    resolutions = np.empty(replicas, dtype=np.float64)
    rms_over_means = np.empty(replicas, dtype=np.float64)
    for index in range(replicas):
        sample = values[rng.integers(0, values.size, size=values.size)]
        q16, median, q84 = np.quantile(sample, [0.16, 0.50, 0.84])
        mean = np.mean(sample)
        medians[index] = median
        resolutions[index] = 0.5 * (q84 - q16) / median if median else math.nan
        rms_over_means[index] = np.std(sample) / mean if mean else math.nan

    metrics["median_ci68"] = [
        float(value) for value in np.quantile(medians, [0.16, 0.84])
    ]
    metrics["closure_bias_ci68"] = [
        value - 1.0 for value in metrics["median_ci68"]
    ]
    finite_resolution = resolutions[np.isfinite(resolutions)]
    metrics["central68_resolution_ci68"] = [
        float(value) for value in np.quantile(finite_resolution, [0.16, 0.84])
    ]
    finite_rms = rms_over_means[np.isfinite(rms_over_means)]
    metrics["rms_over_mean_ci68"] = [
        float(value) for value in np.quantile(finite_rms, [0.16, 0.84])
    ]
    return metrics


def bootstrap_interval(values, statistic, rng, replicas=200):
    """Return the central 68% paired-bootstrap interval for a 1D statistic."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or replicas <= 0:
        return [None, None]
    estimates = np.empty(replicas, dtype=np.float64)
    for index in range(replicas):
        sample = values[rng.integers(0, values.size, size=values.size)]
        estimates[index] = statistic(sample)
    low, high = np.quantile(estimates[np.isfinite(estimates)], [0.16, 0.84])
    return [float(low), float(high)]


def central68_resolution(values):
    metrics = response_metrics(values)
    value = metrics["central68_resolution"]
    return value if value is not None else math.nan
