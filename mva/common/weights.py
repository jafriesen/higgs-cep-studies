"""Physical weights and score-normalization helpers shared by both channels."""

import numpy as np

from analysis.cross_sections import generator_cross_section_fb, generator_weight


LUMI_FB = 3000.0


def tag_probability(parameters, source_flavor, target_flavor):
    tagging = parameters["tagging"]
    flavor_key = {"bb": "b", "cc": "c", "light": "light"}
    if source_flavor == target_flavor:
        return float(tagging[f"eff_{flavor_key[target_flavor]}"])
    return float(
        tagging[f"mistag_{flavor_key[source_flavor]}_to_{flavor_key[target_flavor]}"]
    )


def tag_factor(parameters, source_flavor, target_flavor):
    return tag_probability(parameters, source_flavor, target_flavor) ** 2


def physical_event_weight(
    spec,
    n_generated,
    n_draws,
    pair_acceptance,
    tag,
    campaign=None,
    luminosity_fb=LUMI_FB,
):
    """Legacy-compatible per-draw physical weight.

    Nominal analytic MadGraph datasets instead store the central cross-section
    weight and multiply it by a min-bias pair intensity at evaluation time.
    """
    if n_generated <= 0 or n_draws <= 0 or not np.isfinite(pair_acceptance):
        return 0.0
    xsec_fb, _source = generator_cross_section_fb(
        spec["generator"], spec["process"], campaign
    )
    process_weight = generator_weight(spec["generator"], spec["process"])
    acceptance = (
        float(spec.get("combinatorial_acceptance_factor", 0.005))
        if spec["generator"] == "madgraph"
        else 1.0
    )
    return (
        xsec_fb
        * luminosity_fb
        * process_weight
        * acceptance
        * pair_acceptance
        * tag
        / (n_generated * n_draws)
    )


def central_cross_section_weight(spec, n_generated, campaign=None):
    """Cross section in fb represented by one unstitched generated event."""
    if n_generated <= 0:
        raise ValueError("n_generated must be positive")
    xsec_fb, source = generator_cross_section_fb(
        spec["generator"], spec["process"], campaign
    )
    scale = generator_weight(spec["generator"], spec["process"])
    return xsec_fb * scale / n_generated, xsec_fb, source


def stitched_cross_section_weights(own_campaign, phase_masks, luminosities):
    """Cross section represented by each event in overlapping samples."""
    coverage = np.zeros_like(next(iter(phase_masks.values())), dtype=np.float64)
    for campaign, luminosity in luminosities.items():
        coverage += float(luminosity) * np.asarray(phase_masks[campaign], dtype=bool)
    own = np.asarray(phase_masks[own_campaign], dtype=bool)
    weights = np.zeros(own.shape, dtype=np.float64)
    valid = own & (coverage > 0.0)
    weights[valid] = 1.0 / coverage[valid]
    return weights


def disjoint_campaign_weights(campaign, generated, cross_sections_fb):
    """Per-event fb weights for explicitly disjoint campaign components."""
    if set(generated) != set(cross_sections_fb):
        raise ValueError("generated and cross_sections_fb campaigns must match")
    if campaign not in generated or generated[campaign] <= 0:
        raise ValueError(f"Invalid generated denominator for {campaign}")
    return float(cross_sections_fb[campaign]) / int(generated[campaign])


def subsample_scale(n_total, n_drawn):
    if n_total <= 0 or n_drawn <= 0 or n_drawn > n_total:
        raise ValueError("subsample counts must satisfy 0 < n_drawn <= n_total")
    return float(n_total) / float(n_drawn)


def plugin_score(probabilities, kappas):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0)
    kappas = np.asarray(kappas, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != kappas.size + 1:
        raise ValueError("Probability columns must contain signal followed by every background")
    background = probabilities[:, 1:] @ kappas
    return np.log(probabilities[:, 0]) - np.log(np.clip(background, 1.0e-12, None))


def normalization_scales(components, kind, value, nominal):
    if value < 0.0 or nominal <= 0.0:
        raise ValueError("Normalization values must be non-negative with positive nominal")
    ratio = value / nominal
    scales = np.ones(len(components), dtype=np.float64)
    for index, component in enumerate(components):
        if kind == "survival" and component["survival_scaled"]:
            scales[index] = ratio
        elif kind == "eff_c" and component["source_flavor"] == "cc":
            scales[index] = ratio**2
        elif kind == "mistag_b_to_c" and component["source_flavor"] == "bb":
            scales[index] = ratio**2
        elif kind == "eff_b" and component["source_flavor"] == "bb":
            scales[index] = ratio**2
        elif kind == "mistag_c_to_b" and component["source_flavor"] == "cc":
            scales[index] = ratio**2
    if kind not in {"survival", "eff_c", "mistag_b_to_c", "eff_b", "mistag_c_to_b"}:
        raise ValueError(f"Unknown normalization scan: {kind}")
    return scales
