"""Single-proton intensities and analytic cross-interaction pair densities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import fftconvolve

from .artifact import metadata_path, read_json, validate_parquet


class Acceptance:
    """Union of truth-xi acceptance windows."""

    def __init__(self, windows):
        if isinstance(windows, dict):
            windows = windows.values()
        self.windows = tuple(sorted((float(low), float(high)) for low, high in windows))
        if not self.windows or any(not 0.0 < low < high < 1.0 for low, high in self.windows):
            raise ValueError("Acceptance windows must satisfy 0 < low < high < 1")

    @property
    def minimum(self):
        return self.windows[0][0]

    @property
    def maximum(self):
        return max(high for _low, high in self.windows)

    def mask(self, xi):
        xi = np.asarray(xi)
        selected = np.zeros(xi.shape, dtype=bool)
        for low, high in self.windows:
            selected |= (xi > low) & (xi < high)
        return selected


@dataclass(frozen=True)
class Resolution:
    xi_sigma: float = 0.0003
    seed: int = 12345

    def __post_init__(self):
        if self.xi_sigma < 0.0:
            raise ValueError("xi_sigma must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


class ProtonFlux:
    """Stored truth protons together with the inelastic denominator."""

    def __init__(self, event, arm, xi, px, py, process, metadata, path=None):
        arrays = [np.asarray(value) for value in (event, arm, xi, px, py, process)]
        if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 1:
            raise ValueError("Proton columns must be aligned one-dimensional arrays")
        self.event, self.arm, self.xi, self.px, self.py, self.process = arrays
        self.metadata = metadata
        self.path = Path(path) if path is not None else None
        self.n_inelastic_generated = int(metadata["n_inelastic_generated"])
        self.sqrt_s_gev = float(metadata["sqrt_s_gev"])
        if self.n_inelastic_generated <= 0 or self.sqrt_s_gev <= 0.0:
            raise ValueError("Artifact denominator and sqrt(s) must be positive")
        if np.any((self.arm != -1) & (self.arm != 1)):
            raise ValueError("Proton arm values must be -1 or +1")

    @classmethod
    def load(cls, path, verify_hash=True):
        path = Path(path).resolve()
        metadata = read_json(metadata_path(path))
        validate_parquet(path, metadata, verify_hash=verify_hash)
        table = pq.read_table(path)
        return cls(
            *(np.asarray(table[name]) for name in ("event", "arm", "xi", "px", "py", "process")),
            metadata,
            path,
        )

    def arm_statistics(self, acceptance):
        selected = acceptance.mask(self.xi)
        denominator = self.n_inelastic_generated
        counts = {}
        present = {}
        for name, sign in (("left", -1), ("right", 1)):
            mask = selected & (self.arm == sign)
            events, event_counts = np.unique(self.event[mask], return_counts=True)
            counts[name] = {
                "protons": int(np.sum(mask)),
                "mean_multiplicity": float(np.sum(mask) / denominator),
                "occupied_events": int(events.size),
                "occupancy": float(events.size / denominator),
                "multi_proton_events": int(np.sum(event_counts > 1)),
                "multi_proton_event_probability": float(
                    np.sum(event_counts > 1) / denominator
                ),
            }
            present[name] = events
        both = np.intersect1d(present["left"], present["right"], assume_unique=True).size
        left_only = present["left"].size - both
        right_only = present["right"].size - both
        neither = denominator - left_only - right_only - both
        if neither < 0:
            raise RuntimeError("Arm occupancy exceeds the artifact denominator")
        joint_counts = np.array([[neither, right_only], [left_only, both]], dtype=np.int64)
        return {
            **counts,
            "joint_counts": joint_counts,
            "joint_probability": joint_counts / denominator,
        }

    def log_xi_intensity(self, acceptance, resolution=Resolution(), bins=4096):
        if bins < 2:
            raise ValueError("bins must be at least two")
        truth_selected = acceptance.mask(self.xi)
        selected_indices = np.flatnonzero(truth_selected)
        truth_xi = np.asarray(self.xi[selected_indices], dtype=np.float64)
        rng = np.random.default_rng(resolution.seed)
        smeared = truth_xi + resolution.xi_sigma * rng.standard_normal(truth_xi.size)
        positive = smeared > 0.0
        lower = acceptance.minimum - 6.0 * resolution.xi_sigma
        lower = max(lower, acceptance.minimum * 1.0e-3, np.finfo(np.float64).tiny)
        upper = acceptance.maximum + 6.0 * resolution.xi_sigma
        edges = np.linspace(np.log(lower), np.log(upper), bins + 1)
        du = edges[1] - edges[0]
        intensities = {}
        out_of_grid = {}
        arm_diagnostics = {}
        for name, sign in (("left", -1), ("right", 1)):
            arm = self.arm[selected_indices] == sign
            valid = arm & positive
            values = np.log(smeared[valid])
            counts, _ = np.histogram(values, bins=edges)
            intensities[name] = counts.astype(np.float64) / (self.n_inelastic_generated * du)
            in_grid = valid & (smeared >= lower) & (smeared <= upper)
            out_of_grid[name] = int(np.sum(valid) - np.sum(in_grid))
            arm_diagnostics[name] = {
                "truth_selected": int(np.sum(arm)),
                "nonpositive_after_smearing": int(np.sum(arm & ~positive)),
                "nonpositive_lost_intensity": float(
                    np.sum(arm & ~positive) / self.n_inelastic_generated
                ),
                "out_of_grid": out_of_grid[name],
                "out_of_grid_lost_intensity": float(
                    out_of_grid[name] / self.n_inelastic_generated
                ),
            }
        diagnostics = {
            "truth_selected": int(truth_xi.size),
            "nonpositive_after_smearing": int(np.sum(~positive)),
            "out_of_grid": out_of_grid,
            "arms": arm_diagnostics,
            "du": float(du),
        }
        return edges, intensities["left"], intensities["right"], diagnostics


class PairDensity:
    """Analytic intensity of cross-arm pairs from distinct interactions."""

    def __init__(self, flux, acceptance, resolution=Resolution(), bins=4096):
        self.flux = flux
        self.acceptance = acceptance
        self.resolution = resolution
        self.edges, self.left, self.right, self.diagnostics = flux.log_xi_intensity(
            acceptance, resolution, bins
        )
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.du = float(self.edges[1] - self.edges[0])
        self._right_cumulative = np.r_[0.0, np.cumsum(self.right) * self.du]
        self._mass_sum_centers = 2.0 * self.edges[0] + self.du + np.arange(
            2 * self.left.size - 1
        ) * self.du
        self._mass_sum_density = fftconvolve(self.left, self.right, mode="full") * self.du
        self._mass_sum_edges = np.r_[
            self._mass_sum_centers - 0.5 * self.du,
            self._mass_sum_centers[-1] + 0.5 * self.du,
        ]
        self._mass_sum_cumulative = np.r_[
            0.0, np.cumsum(self._mass_sum_density) * self.du
        ]
        self._rapidity_profiles = {}

    @staticmethod
    def _validate_mass_range(mass_range):
        low, high = (float(value) for value in mass_range)
        if not 0.0 < low < high:
            raise ValueError("mass_range must satisfy 0 < low < high")
        return low, high

    @staticmethod
    def _validate_yx_range(yx_range):
        if yx_range is None:
            return -np.inf, np.inf
        low, high = (float(value) for value in yx_range)
        if not low < high:
            raise ValueError("yx_range must satisfy low < high")
        return low, high

    @staticmethod
    def _piecewise_cdf(values, edges, density, cumulative):
        values = np.asarray(values, dtype=np.float64)
        result = np.empty(values.shape, dtype=np.float64)
        below = values <= edges[0]
        above = values >= edges[-1]
        middle = ~(below | above)
        result[below] = 0.0
        result[above] = cumulative[-1]
        index = np.searchsorted(edges, values[middle], side="right") - 1
        result[middle] = cumulative[index] + density[index] * (values[middle] - edges[index])
        return result

    def density(self, mx, yx):
        mx, yx = np.broadcast_arrays(
            np.asarray(mx, dtype=np.float64), np.asarray(yx, dtype=np.float64)
        )
        if np.any(mx <= 0.0):
            raise ValueError("mx must be positive")
        v = np.log(mx / self.flux.sqrt_s_gev)
        left = np.interp(v - yx, self.centers, self.left, left=0.0, right=0.0)
        right = np.interp(v + yx, self.centers, self.right, left=0.0, right=0.0)
        return 2.0 * left * right / mx

    def integrate(self, mass_range, yx_range=None):
        mass_low, mass_high = self._validate_mass_range(mass_range)
        yx_low, yx_high = self._validate_yx_range(yx_range)
        sum_low = 2.0 * np.log(mass_low / self.flux.sqrt_s_gev)
        sum_high = 2.0 * np.log(mass_high / self.flux.sqrt_s_gev)
        right_low = np.maximum(sum_low - self.centers, self.centers + 2.0 * yx_low)
        right_high = np.minimum(sum_high - self.centers, self.centers + 2.0 * yx_high)
        valid = right_high > right_low
        right_integral = np.zeros(self.left.shape, dtype=np.float64)
        if np.any(valid):
            right_integral[valid] = self._piecewise_cdf(
                right_high[valid], self.edges, self.right, self._right_cumulative
            ) - self._piecewise_cdf(
                right_low[valid], self.edges, self.right, self._right_cumulative
            )
        return float(np.sum(self.left * right_integral) * self.du)

    def rapidity_profile(self, mass_range, bins=8192, quadrature_order=64):
        """Return a cached CDF of pair intensity versus pair rapidity.

        The mass-window integral is evaluated with Gauss--Legendre quadrature
        in ``s = log(xi_left * xi_right)``.  Interpolating this CDF makes large
        arrays of event-dependent rapidity bands inexpensive while retaining
        the same histogrammed single-proton flux as :meth:`integrate`.
        """
        mass_low, mass_high = self._validate_mass_range(mass_range)
        bins = int(bins)
        quadrature_order = int(quadrature_order)
        if bins < 2 or quadrature_order < 2:
            raise ValueError("bins and quadrature_order must be at least two")
        key = (mass_low, mass_high, bins, quadrature_order)
        cached = self._rapidity_profiles.get(key)
        if cached is not None:
            return cached

        # y = (u_right - u_left) / 2 for u = log(xi).  The full histogram
        # support is safe for every mass slice; values outside the physical
        # support of a slice simply interpolate to zero below.
        y_edges = np.linspace(
            0.5 * (self.edges[0] - self.edges[-1]),
            0.5 * (self.edges[-1] - self.edges[0]),
            bins + 1,
        )
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        sum_low = 2.0 * np.log(mass_low / self.flux.sqrt_s_gev)
        sum_high = 2.0 * np.log(mass_high / self.flux.sqrt_s_gev)
        sums = 0.5 * (sum_high - sum_low) * nodes + 0.5 * (sum_high + sum_low)
        sum_weights = 0.5 * (sum_high - sum_low) * weights
        density = np.zeros(y_centers.shape, dtype=np.float64)
        for value, weight in zip(sums, sum_weights):
            left = np.interp(
                0.5 * value - y_centers,
                self.centers,
                self.left,
                left=0.0,
                right=0.0,
            )
            right = np.interp(
                0.5 * value + y_centers,
                self.centers,
                self.right,
                left=0.0,
                right=0.0,
            )
            density += weight * left * right
        widths = np.diff(y_edges)
        cumulative = np.r_[0.0, np.cumsum(density * widths)]
        profile = {
            "edges": y_edges,
            "density": density,
            "cumulative": cumulative,
            "mass_range": (mass_low, mass_high),
        }
        self._rapidity_profiles[key] = profile
        return profile

    def integrate_yx_ranges(
        self, mass_range, yx_low, yx_high, bins=8192, quadrature_order=64
    ):
        """Vectorized mass-window intensities for many rapidity intervals."""
        low, high = np.broadcast_arrays(
            np.asarray(yx_low, dtype=np.float64),
            np.asarray(yx_high, dtype=np.float64),
        )
        if np.any(low >= high):
            raise ValueError("Every yx interval must satisfy low < high")
        profile = self.rapidity_profile(mass_range, bins, quadrature_order)
        lower = self._piecewise_cdf(
            low, profile["edges"], profile["density"], profile["cumulative"]
        )
        upper = self._piecewise_cdf(
            high, profile["edges"], profile["density"], profile["cumulative"]
        )
        return np.maximum(upper - lower, 0.0)

    def marginal_mass(self, mass_bins, yx_range=None):
        mass_bins = np.asarray(mass_bins, dtype=np.float64)
        if mass_bins.ndim != 1 or mass_bins.size < 2 or np.any(np.diff(mass_bins) <= 0.0):
            raise ValueError("mass_bins must be a strictly increasing one-dimensional array")
        if mass_bins[0] <= 0.0:
            raise ValueError("mass bins must be positive")
        if yx_range is not None:
            return np.asarray(
                [self.integrate((low, high), yx_range) for low, high in zip(mass_bins[:-1], mass_bins[1:])]
            )
        sums = 2.0 * np.log(mass_bins / self.flux.sqrt_s_gev)
        cumulative = self._piecewise_cdf(
            sums,
            self._mass_sum_edges,
            self._mass_sum_density,
            self._mass_sum_cumulative,
        )
        return np.diff(cumulative)

    def conditional_yx(self, mx, yx_range=None):
        mx = float(mx)
        if mx <= 0.0:
            raise ValueError("mx must be positive")
        v = np.log(mx / self.flux.sqrt_s_gev)
        support_low = max(v - self.edges[-1], self.edges[0] - v)
        support_high = min(v - self.edges[0], self.edges[-1] - v)
        requested_low, requested_high = self._validate_yx_range(yx_range)
        low, high = max(support_low, requested_low), min(support_high, requested_high)
        if not low < high:
            return np.empty(0), np.empty(0)
        n = max(2, int(np.ceil((high - low) / self.du)))
        edges = np.linspace(low, high, n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = self.density(mx, centers)
        normalization = np.sum(values * np.diff(edges))
        if normalization <= 0.0:
            return np.empty(0), np.empty(0)
        return centers, values / normalization

    def expected_pairs_fixed_n(self, n_interactions, mass_range, yx_range=None):
        n_interactions = np.asarray(n_interactions)
        if np.any(n_interactions < 0):
            raise ValueError("n_interactions must be non-negative")
        return n_interactions * (n_interactions - 1) * self.integrate(mass_range, yx_range)

    def expected_pairs_poisson(self, mu, mass_range, yx_range=None):
        mu = np.asarray(mu, dtype=np.float64)
        if np.any(mu < 0.0):
            raise ValueError("mu must be non-negative")
        return mu * mu * self.integrate(mass_range, yx_range)
