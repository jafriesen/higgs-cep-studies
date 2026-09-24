"""Shared definitions for the five-class H(cc) proton MVA workflow."""
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]

CLASS_NAMES = (
    "Hcc",
    "exclusive_QCD",
    "exclusive_QED",
    "nonexclusive_QCD",
    "Hbb_resonant",
)

COMPONENT_SPECS = (
    {
        "name": "Hcc",
        "generator": "superchic",
        "process": "Hcc",
        "class_name": "Hcc",
        "source_flavor": "cc",
        "signal": True,
        "survival_scaled": True,
        "campaigns": None,
    },
    {
        "name": "QCDcc_superchic",
        "generator": "superchic",
        "process": "QCDcc",
        "class_name": "exclusive_QCD",
        "source_flavor": "cc",
        "signal": False,
        "survival_scaled": True,
        "campaigns": None,
    },
    {
        "name": "QCDbb_superchic",
        "generator": "superchic",
        "process": "QCDbb",
        "class_name": "exclusive_QCD",
        "source_flavor": "bb",
        "signal": False,
        "survival_scaled": True,
        "campaigns": None,
    },
    {
        "name": "QEDcc_superchic",
        "generator": "superchic",
        "process": "QEDcc",
        "class_name": "exclusive_QED",
        "source_flavor": "cc",
        "signal": False,
        "survival_scaled": False,
        "campaigns": None,
    },
    {
        "name": "QCDcc_madgraph",
        "generator": "madgraph",
        "process": "QCDcc",
        "class_name": "nonexclusive_QCD",
        "source_flavor": "cc",
        "signal": False,
        "survival_scaled": False,
        "campaigns": ("QCDcc__v01",),
    },
    {
        "name": "QCDbb_madgraph",
        "generator": "madgraph",
        "process": "QCDbb",
        "class_name": "nonexclusive_QCD",
        "source_flavor": "bb",
        "signal": False,
        "survival_scaled": False,
        "campaigns": ("QCDbb__v02", "QCDbb__v03"),
    },
    {
        "name": "Hbb_superchic",
        "generator": "superchic",
        "process": "Hbb",
        "class_name": "Hbb_resonant",
        "source_flavor": "bb",
        "signal": False,
        "survival_scaled": True,
        "campaigns": None,
    },
)

LOCKED_FEATURES = (
    "delta_phi_jj",
    "yx_minus_dijet_rapidity",
    "dijet_mass",
    "pt_asymmetry",
    "interjet_bridge_asymmetry_projected",
    "delta_eta_jj",
    "n_tracks_interjet",
    "sum_track_pt_outer_negative",
    "jet_multiplicity",
    "jet1_eta",
    "jet2_pull_magnitude",
    "n_vertices",
    "n_tracks_outer_positive",
    "interjet_bridge_asymmetry",
    "jet1_charged_fraction",
    "dijet_rapidity",
    "n_outer_track_jets",
    "sum_track_pt_outside_pt2",
    "minimum_gap_size",
    "n_tracks_projected_bridge",
)


def read_parameters(path=None):
    target = Path(path) if path else REPO / "parameters.yaml"
    with open(target, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tag_probability(parameters, source_flavor):
    tagging = parameters["tagging"]
    if source_flavor == "cc":
        return float(tagging["eff_c"])
    if source_flavor == "bb":
        return float(tagging["mistag_b_to_c"])
    raise ValueError(f"Unsupported source flavor: {source_flavor}")


def tag_factor(parameters, source_flavor):
    return tag_probability(parameters, source_flavor) ** 2


def class_index(class_name):
    return CLASS_NAMES.index(class_name)


def component_index(component_name):
    return tuple(spec["name"] for spec in COMPONENT_SPECS).index(component_name)


def component_manifest(parameters, specs=COMPONENT_SPECS, class_names=CLASS_NAMES):
    manifest = []
    for component_id, spec in enumerate(specs):
        item = dict(spec)
        item["id"] = component_id
        item["class_id"] = list(class_names).index(spec["class_name"])
        item["tag_probability"] = tag_probability(parameters, spec["source_flavor"])
        item["tag_factor"] = tag_factor(parameters, spec["source_flavor"])
        item["campaigns"] = list(spec["campaigns"]) if spec["campaigns"] else None
        manifest.append(item)
    return manifest


def stitched_cross_section_weights(own_campaign, phase_masks, luminosities):
    """Return sigma represented by each retained event, stitching one flavor."""
    coverage = np.zeros_like(next(iter(phase_masks.values())), dtype=np.float64)
    for campaign, luminosity in luminosities.items():
        coverage += float(luminosity) * np.asarray(phase_masks[campaign], dtype=bool)
    own = np.asarray(phase_masks[own_campaign], dtype=bool)
    weights = np.zeros(own.shape, dtype=np.float64)
    valid = own & (coverage > 0.0)
    weights[valid] = 1.0 / coverage[valid]
    return weights


def class_yields(classes, weights, n_classes):
    return np.bincount(classes, weights=weights, minlength=n_classes)


def plugin_score(probabilities, kappas):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0)
    kappas = np.asarray(kappas, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != kappas.size + 1:
        raise ValueError("Probability columns must contain signal followed by every background")
    background = probabilities[:, 1:] @ kappas
    return np.log(probabilities[:, 0]) - np.log(np.clip(background, 1.0e-12, None))


def normalization_scales(components, kind, value, nominal):
    """Component scale factors for a locked-score normalization variation."""
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
    if kind not in {"survival", "eff_c", "mistag_b_to_c"}:
        raise ValueError(f"Unknown normalization scan: {kind}")
    return scales


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(plain(payload), handle, sort_keys=False)
