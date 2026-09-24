"""Forward-proton reconstruction and nominal analytic pair intensities."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from common.config_utils import resolve_minbias_campaign
from minbias import Acceptance, PairDensity, ProtonFlux, Resolution


MASS_WINDOW_GEV = (117.0, 133.0)
MAX_ABS_RAPIDITY_DIFFERENCE = 0.2
DEFAULT_MINBIAS_PATH = Path("output/minbias/minbias_inelastic_100m_v1/protons.parquet")


def load_pps_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sqrt_s = float((config.get("beam") or {}).get("sqrt_s_gev", 0.0))
    if sqrt_s <= 0.0:
        raise RuntimeError(f"{path} must define a positive beam.sqrt_s_gev")
    xi_ranges = []
    for station, bounds in ((config.get("pps") or {}).get("xi_ranges") or {}).items():
        if len(bounds) != 2:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        xi_ranges.append((str(station), low, high))
    if not xi_ranges:
        raise RuntimeError(f"{path} must define at least one pps.xi_ranges entry")
    return {
        "sqrt_s": sqrt_s,
        "xi_ranges": xi_ranges,
        "xi_res": float((config.get("pps") or {}).get("xi_res", 0.0)),
    }


def passes_pps(xi, xi_ranges):
    xi = np.asarray(xi, dtype=np.float64)
    passed = np.zeros(xi.shape, dtype=bool)
    for _station, low, high in xi_ranges:
        passed |= (xi >= low) & (xi < high)
    return passed


def pair_observables(xi_left, xi_right, sqrt_s):
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    return np.sqrt(xi_left * xi_right) * sqrt_s, 0.5 * np.log(xi_right / xi_left)


def smear_pair_observables(xi_left, xi_right, pps, rng):
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    if pps["xi_res"] > 0.0:
        xi_left = rng.normal(xi_left, pps["xi_res"])
        xi_right = rng.normal(xi_right, pps["xi_res"])
    valid = (xi_left > 0.0) & (xi_right > 0.0)
    mx = np.full(xi_left.shape, np.nan, dtype=np.float64)
    yx = np.full(xi_left.shape, np.nan, dtype=np.float64)
    mx[valid], yx[valid] = pair_observables(
        xi_left[valid], xi_right[valid], pps["sqrt_s"]
    )
    return xi_left, xi_right, mx, yx, valid


def real_proton_pass(xi_left_truth, xi_right_truth, pps, rng):
    valid_pair = (
        np.isfinite(xi_left_truth)
        & np.isfinite(xi_right_truth)
        & (xi_left_truth > 0.0)
        & (xi_right_truth > 0.0)
    )
    accepted = valid_pair & passes_pps(xi_left_truth, pps["xi_ranges"])
    accepted &= passes_pps(xi_right_truth, pps["xi_ranges"])
    xi_left, xi_right, mx, yx, reco_valid = smear_pair_observables(
        xi_left_truth, xi_right_truth, pps, rng
    )
    return accepted & reco_valid, xi_left, xi_right, mx, yx


def _parse_protons(input_file, selected_event_indices, sqrt_s, record_kind):
    selected_event_indices = np.asarray(selected_event_indices, dtype=np.int64)
    output = {
        name: np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
        for name in (
            "xi_left",
            "xi_right",
            "proton_px_left",
            "proton_py_left",
            "proton_px_right",
            "proton_py_right",
        )
    }
    if selected_event_indices.size == 0:
        return output
    selected_positions = {
        int(event_index): output_index
        for output_index, event_index in enumerate(selected_event_indices)
    }
    last_selected = int(selected_event_indices[-1])
    beam_energy = sqrt_s / 2.0
    event_index = -1
    output_index = None
    left = right = None
    left_abs_pz = right_abs_pz = -1.0

    def store_event():
        if output_index is None or left is None or right is None:
            return
        left_px, left_py, left_energy = left
        right_px, right_py, right_energy = right
        output["xi_left"][output_index] = (beam_energy - left_energy) / beam_energy
        output["xi_right"][output_index] = (beam_energy - right_energy) / beam_energy
        output["proton_px_left"][output_index] = left_px
        output["proton_py_left"][output_index] = left_py
        output["proton_px_right"][output_index] = right_px
        output["proton_py_right"][output_index] = right_py

    with open(input_file, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if (record_kind == "hepmc" and line.startswith("E ")) or (
                record_kind == "lhe" and stripped.startswith("<event")
            ):
                if record_kind == "hepmc" and event_index >= 0:
                    store_event()
                event_index += 1
                if event_index > last_selected:
                    break
                output_index = selected_positions.get(event_index)
                left = right = None
                left_abs_pz = right_abs_pz = -1.0
                continue
            if record_kind == "lhe" and stripped.startswith("</event"):
                store_event()
                output_index = None
                continue
            if output_index is None:
                continue
            fields = stripped.split()
            try:
                if record_kind == "hepmc":
                    if not line.startswith("P ") or int(fields[3]) != 2212 or int(fields[9]) != 1:
                        continue
                    px, py = float(fields[4]), float(fields[5])
                    pz, energy = float(fields[6]), float(fields[7])
                else:
                    if len(fields) < 10 or int(fields[0]) != 2212 or int(fields[1]) != 1:
                        continue
                    px, py = float(fields[6]), float(fields[7])
                    pz, energy = float(fields[8]), float(fields[9])
            except (ValueError, IndexError):
                continue
            abs_pz = abs(pz)
            if pz < 0.0 and abs_pz > left_abs_pz:
                left, left_abs_pz = (px, py, energy), abs_pz
            elif pz > 0.0 and abs_pz > right_abs_pz:
                right, right_abs_pz = (px, py, energy), abs_pz
        if record_kind == "hepmc":
            store_event()
    return output


def parse_hepmc_protons(input_file, selected_event_indices, sqrt_s):
    return _parse_protons(input_file, selected_event_indices, sqrt_s, "hepmc")


def parse_lhe_protons(input_file, selected_event_indices, sqrt_s):
    return _parse_protons(input_file, selected_event_indices, sqrt_s, "lhe")


def parse_hepmc_proton_xi(input_file, selected_event_indices, sqrt_s):
    protons = parse_hepmc_protons(input_file, selected_event_indices, sqrt_s)
    return protons["xi_left"], protons["xi_right"]


def parse_lhe_proton_xi(input_file, selected_event_indices, sqrt_s):
    protons = parse_lhe_protons(input_file, selected_event_indices, sqrt_s)
    return protons["xi_left"], protons["xi_right"]


def build_pair_density(path, pps, seed=12345, bins=4096, verify_hash=True):
    path = Path(path).resolve()
    flux = ProtonFlux.load(path, verify_hash=verify_hash)
    acceptance = Acceptance([(low, high) for _station, low, high in pps["xi_ranges"]])
    return PairDensity(flux, acceptance, Resolution(pps["xi_res"], seed), bins=bins)


def proton_cell_intensities(
    pairs, dijet_rapidity, cell_edges, mass_window=MASS_WINDOW_GEV, pileup_mu=200.0
):
    """Poisson pair intensities for event-relative rapidity cells."""
    rapidity = np.asarray(dijet_rapidity, dtype=np.float64)
    edges = np.asarray(cell_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("cell_edges must be strictly increasing")
    if pileup_mu < 0.0:
        raise ValueError("pileup_mu must be non-negative")
    low = rapidity[:, np.newaxis] + edges[:-1]
    high = rapidity[:, np.newaxis] + edges[1:]
    return pileup_mu**2 * pairs.integrate_yx_ranges(mass_window, low, high)


def load_bootstrap_pool(minbias_campaign=None):
    """Load the legacy sampled pool, for the explicit parity profile only."""
    campaign_dir, campaign = resolve_minbias_campaign(minbias_campaign)
    pool_path = campaign_dir / "pairs" / "proton_pairs.parquet"
    meta_path = campaign_dir / "pairs" / "metadata.json"
    if not pool_path.is_file():
        raise RuntimeError(f"Missing legacy proton-pair pool: {pool_path}")
    table = pq.read_table(pool_path, columns=["mx", "yx", "weight"])
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {
        "mx": np.asarray(table["mx"], dtype=np.float64),
        "yx": np.asarray(table["yx"], dtype=np.float64),
        "weight": np.asarray(table["weight"], dtype=np.float64),
        "acceptance": float(metadata.get("bx_pair_acceptance_poisson", np.nan)),
        "path": str(pool_path),
        "campaign": campaign,
    }


class LegacyPairDensity:
    """Weighted sampled-pool adapter used only by the parity profile."""

    def __init__(self, pool):
        self.mx = np.asarray(pool["mx"], dtype=np.float64)
        self.yx = np.asarray(pool["yx"], dtype=np.float64)
        self.weight = np.asarray(pool["weight"], dtype=np.float64)
        if self.mx.shape != self.yx.shape or self.mx.shape != self.weight.shape:
            raise ValueError("Legacy pool columns must be aligned")
        self.total = float(np.sum(self.weight))
        if self.total <= 0.0:
            raise ValueError("Legacy pool has non-positive total weight")
        self._profiles = {}

    def _profile(self, mass_range):
        key = tuple(float(value) for value in mass_range)
        if key not in self._profiles:
            selected = (self.mx >= key[0]) & (self.mx < key[1])
            order = np.argsort(self.yx[selected], kind="stable")
            yx = self.yx[selected][order]
            cumulative = np.r_[0.0, np.cumsum(self.weight[selected][order]) / self.total]
            self._profiles[key] = yx, cumulative
        return self._profiles[key]

    def integrate_yx_ranges(self, mass_range, yx_low, yx_high, **_unused):
        low, high = np.broadcast_arrays(
            np.asarray(yx_low, dtype=np.float64), np.asarray(yx_high, dtype=np.float64)
        )
        if np.any(low >= high):
            raise ValueError("Every yx interval must satisfy low < high")
        yx, cumulative = self._profile(mass_range)
        left = np.searchsorted(yx, low, side="left")
        right = np.searchsorted(yx, high, side="left")
        return cumulative[right] - cumulative[left]

    def integrate(self, mass_range, yx_range=None):
        low, high = (-np.inf, np.inf) if yx_range is None else yx_range
        return float(self.integrate_yx_ranges(mass_range, low, high))


def conditional_pair_sample(
    central_matrix,
    dijet_rapidity,
    central_group,
    central_campaign,
    central_split,
    stitch_weight_fb,
    coverage_mask,
    pool,
    train_pairs,
    evaluation_pairs,
    seed,
):
    """Legacy conditional pool sampling retained solely for parity."""
    if train_pairs <= 0 or evaluation_pairs <= 0:
        raise RuntimeError("Conditional proton copy counts must be positive")
    probability = np.array(pool["weight"], dtype=np.float64, copy=True)
    probability /= np.sum(probability)
    order = np.argsort(pool["yx"], kind="stable")
    sorted_yx = np.asarray(pool["yx"], dtype=np.float64)[order]
    sorted_mx = np.asarray(pool["mx"], dtype=np.float64)[order]
    cdf = np.cumsum(probability[order])
    cdf[-1] = 1.0
    low_index = np.searchsorted(
        sorted_yx, dijet_rapidity - MAX_ABS_RAPIDITY_DIFFERENCE, side="right"
    )
    high_index = np.searchsorted(
        sorted_yx, dijet_rapidity + MAX_ABS_RAPIDITY_DIFFERENCE, side="left"
    )
    cdf_low = np.zeros(dijet_rapidity.shape[0], dtype=np.float64)
    cdf_low[low_index > 0] = cdf[low_index[low_index > 0] - 1]
    cdf_high = np.zeros(dijet_rapidity.shape[0], dtype=np.float64)
    cdf_high[high_index > 0] = cdf[high_index[high_index > 0] - 1]
    band_probability = np.maximum(cdf_high - cdf_low, 0.0)
    copies = np.full(central_split.shape, evaluation_pairs, dtype=np.int64)
    valid = band_probability > 0.0
    central_index = np.repeat(np.flatnonzero(valid), copies[valid])
    if central_index.size == 0:
        raise RuntimeError("No central event has a proton pair in the rapidity band")
    rng = np.random.default_rng(seed)
    quantile = cdf_low[central_index] + rng.random(central_index.size) * band_probability[central_index]
    pair_index = np.minimum(np.searchsorted(cdf, quantile, side="right"), cdf.size - 1)
    delta_y = sorted_yx[pair_index] - dijet_rapidity[central_index]
    return {
        "features": np.asarray(
            np.column_stack([central_matrix[central_index], delta_y]), dtype=np.float32
        ),
        "mx": sorted_mx[pair_index],
        "group": central_group[central_index],
        "source_campaign": central_campaign[central_index],
        "split": central_split[central_index],
        "band_probability": band_probability[central_index],
        "central_stitch_weight_fb": stitch_weight_fb[central_index],
        "coverage_mask": coverage_mask[central_index],
        "stitch_weight_fb": stitch_weight_fb[central_index] * band_probability[central_index] / copies[central_index],
    }
