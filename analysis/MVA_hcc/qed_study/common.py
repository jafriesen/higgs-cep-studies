"""Shared helpers for the H(cc) QED-background study."""
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.MVA_hcc.common import plain, write_yaml  # noqa: E402,F401

MASS_BINS = np.arange(117.0, 134.0, 1.0)
MAX_ABS_RAPIDITY_DIFFERENCE = 0.2
DEFAULT_DATA = SCRIPT_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "output"

# Okabe-Ito, validated colorblind-safe. Order is fixed; never cycled.
PALETTE = {
    "Hcc": "#0072B2",
    "QEDcc_superchic": "#D55E00",
    "QCDcc_superchic": "#E69F00",
    "QCDcc_madgraph": "#009E73",
    "QCDbb_madgraph": "#CC79A7",
    "QCDbb_superchic": "#56B4E9",
    "Hbb_superchic": "#F0E442",
}
SERIES = ("#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")

ARRAY_NAMES = (
    "x", "class", "component", "group_id", "mx",
    "physical_weight", "training_mixture_weight", "central_weight", "band_probability",
)


def load_dataset(data_dir, mmap=True):
    data_dir = Path(data_dir)
    with open(data_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    arrays = {
        name: np.load(data_dir / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
        for name in ARRAY_NAMES
    }
    rows = arrays["class"].size
    if arrays["x"].shape != (rows, len(metadata["features"])):
        raise RuntimeError("Feature matrix does not match the recorded feature list")
    arrays["features"] = list(metadata["features"])
    arrays["metadata"] = metadata
    arrays["component_names"] = [item["name"] for item in metadata["components"]]
    return arrays


def component_rows(data, name):
    return np.flatnonzero(np.asarray(data["component"]) == data["component_names"].index(name))


def weighted_auc(values_a, weights_a, values_b, weights_b):
    """AUC of a over b with mid-rank tie handling. NaN rows are dropped."""
    values = np.r_[values_a, values_b]
    weights = np.r_[weights_a, weights_b]
    is_a = np.r_[np.ones(values_a.size, bool), np.zeros(values_b.size, bool)]
    finite = np.isfinite(values)
    values, weights, is_a = values[finite], weights[finite], is_a[finite]
    if values.size == 0:
        return np.nan
    unique, inverse = np.unique(values, return_inverse=True)
    weight_b = np.bincount(inverse, weights=np.where(is_a, 0.0, weights), minlength=unique.size)
    weight_a = np.bincount(inverse, weights=np.where(is_a, weights, 0.0), minlength=unique.size)
    if weight_a.sum() <= 0.0 or weight_b.sum() <= 0.0:
        return np.nan
    below = np.cumsum(weight_b) - weight_b
    return float(np.sum(weight_a * (below + 0.5 * weight_b)) / (weight_a.sum() * weight_b.sum()))


def separation(values_a, weights_a, values_b, weights_b):
    """AUC folded to [0.5, 1]: direction-free discriminating power."""
    area = weighted_auc(values_a, weights_a, values_b, weights_b)
    return np.nan if not np.isfinite(area) else max(area, 1.0 - area)


def conditional_separation(values_a, weights_a, values_b, weights_b,
                           cond_a, cond_b, n_bins=8):
    """Separation measured inside bins of a conditioning variable, then pooled.

    Answers 'what is left once the conditioning variable is known', which is the
    test that distinguishes real information from a re-expression of it.
    """
    edges = np.quantile(np.r_[cond_a, cond_b][np.isfinite(np.r_[cond_a, cond_b])],
                        np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return np.nan
    index_a = np.clip(np.searchsorted(edges, cond_a, side="right") - 1, 0, edges.size - 2)
    index_b = np.clip(np.searchsorted(edges, cond_b, side="right") - 1, 0, edges.size - 2)
    total, pooled = 0.0, 0.0
    for cell in range(edges.size - 1):
        mask_a, mask_b = index_a == cell, index_b == cell
        if mask_a.sum() < 20 or mask_b.sum() < 20:
            continue
        area = separation(values_a[mask_a], weights_a[mask_a],
                          values_b[mask_b], weights_b[mask_b])
        if not np.isfinite(area):
            continue
        share = weights_a[mask_a].sum()
        pooled += share * area
        total += share
    return float(pooled / total) if total > 0.0 else np.nan


def mass_binned_z(signal_mass, background_mass):
    """Z = sqrt(sum_m s^2/(s+b)) over the fixed 1 GeV mass bins."""
    signal = np.asarray(signal_mass, dtype=np.float64)
    total = signal + np.asarray(background_mass, dtype=np.float64)
    return float(np.sqrt(np.sum(np.divide(signal * signal, total,
                                          out=np.zeros_like(signal), where=total > 0.0))))


def mass_histogram(mx, weight, mask=None):
    if mask is not None:
        mx, weight = mx[mask], weight[mask]
    histogram, _ = np.histogram(mx, MASS_BINS, weights=weight)
    return histogram


def scan_threshold_z(score, mx, weight, is_signal, n_points=200):
    """Best mass-binned Z over score thresholds, plus the full scan."""
    finite = np.isfinite(score)
    thresholds = np.quantile(score[finite & is_signal], np.linspace(0.0, 0.98, n_points))
    thresholds = np.unique(thresholds)
    values = np.empty(thresholds.size)
    for index, threshold in enumerate(thresholds):
        keep = finite & (score >= threshold)
        values[index] = mass_binned_z(
            mass_histogram(mx, weight, keep & is_signal),
            mass_histogram(mx, weight, keep & ~is_signal),
        )
    best = int(np.argmax(values))
    return thresholds, values, thresholds[best], values[best]


def held_out_threshold_z(score, mx, weight, is_signal, folds, n_points=200):
    """Mass-binned Z with the cut chosen on one half and measured on the other.

    scan_threshold_z takes its maximum on the same events it reports, which is
    biased upward whenever the surviving sample is small. Weights are doubled so
    each half still represents the full expected yield.
    """
    values = []
    for fold in (0, 1):
        choose, measure = folds == fold, folds != fold
        doubled = 2.0 * weight
        thresholds, scan, _cut, _best = scan_threshold_z(
            score[choose], mx[choose], doubled[choose], is_signal[choose], n_points
        )
        cut = thresholds[int(np.argmax(scan))]
        keep = measure & np.isfinite(score) & (score >= cut)
        values.append(mass_binned_z(
            mass_histogram(mx, doubled, keep & is_signal),
            mass_histogram(mx, doubled, keep & ~is_signal),
        ))
    return float(np.mean(values))


def assign_folds(groups, seed, n_folds=2):
    """Fold label per row, constant within a hard-event group."""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    assignment = np.arange(unique.size, dtype=np.int8) % n_folds
    rng.shuffle(assignment)
    folds = assignment[inverse]
    order = np.argsort(groups, kind="stable")
    if np.any((groups[order][1:] == groups[order][:-1]) & (folds[order][1:] != folds[order][:-1])):
        raise RuntimeError("A hard-event group crosses folds")
    return folds


def balanced_weights(labels, weights, n_classes):
    """Rescale each class to an equal total, so a plug-in score is the likelihood ratio."""
    output = np.asarray(weights, dtype=np.float64).copy()
    target = labels.size / n_classes
    for label in range(n_classes):
        mask = labels == label
        total = output[mask].sum()
        if total <= 0.0:
            raise RuntimeError(f"Class {label} carries no positive training weight")
        output[mask] *= target / total
    return output


def effective_count(weights):
    weights = np.asarray(weights, dtype=np.float64)
    squared = np.sum(weights * weights)
    return float(weights.sum() ** 2 / squared) if squared > 0.0 else 0.0
