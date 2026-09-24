import unittest

import numpy as np

from minbias.flux import Acceptance, PairDensity, ProtonFlux, Resolution


class FluxTest(unittest.TestCase):
    def setUp(self):
        self.flux = ProtonFlux(
            event=np.array([0, 0, 1, 2, 2, 3], dtype=np.int32),
            arm=np.array([-1, 1, -1, -1, -1, 1], dtype=np.int8),
            xi=np.array([0.005, 0.006, 0.010, 0.020, 0.021, 0.022], dtype=np.float32),
            px=np.zeros(6),
            py=np.zeros(6),
            process=np.ones(6, dtype=np.int16),
            metadata={"n_inelastic_generated": 10, "sqrt_s_gev": 14000.0},
        )
        self.acceptance = Acceptance([(0.003, 0.012), (0.019, 0.024)])

    def test_arm_statistics_distinguish_occupancy_and_multiplicity(self):
        stats = self.flux.arm_statistics(self.acceptance)
        self.assertAlmostEqual(stats["left"]["mean_multiplicity"], 0.4)
        self.assertAlmostEqual(stats["left"]["occupancy"], 0.3)
        self.assertEqual(stats["left"]["multi_proton_events"], 1)
        self.assertAlmostEqual(stats["left"]["multi_proton_event_probability"], 0.1)
        self.assertAlmostEqual(stats["right"]["mean_multiplicity"], 0.2)
        np.testing.assert_array_equal(stats["joint_counts"], [[6, 1], [2, 1]])

    def test_acceptance_windows_are_disjoint_and_open(self):
        acceptance = Acceptance([(0.003, 0.004), (0.010, 0.020)])
        values = np.array([0.003, 0.0035, 0.004, 0.009, 0.010, 0.015, 0.020])
        np.testing.assert_array_equal(
            acceptance.mask(values),
            [False, True, False, False, False, True, False],
        )

    def test_intensity_normalization_and_resolution_repeatability(self):
        edges, left, right, _ = self.flux.log_xi_intensity(
            self.acceptance, Resolution(0.0, 10), bins=512
        )
        du = edges[1] - edges[0]
        self.assertAlmostEqual(np.sum(left) * du, 0.4)
        self.assertAlmostEqual(np.sum(right) * du, 0.2)
        first = self.flux.log_xi_intensity(self.acceptance, Resolution(0.0003, 17), 512)
        second = self.flux.log_xi_intensity(self.acceptance, Resolution(0.0003, 17), 512)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])

    def test_pair_scaling_and_mass_marginal(self):
        density = PairDensity(self.flux, self.acceptance, Resolution(0.0, 1), bins=2048)
        region = (50.0, 400.0)
        intensity = density.integrate(region)
        self.assertGreater(intensity, 0.0)
        self.assertAlmostEqual(density.expected_pairs_fixed_n(10, region), 90 * intensity)
        self.assertAlmostEqual(density.expected_pairs_poisson(10.0, region), 100 * intensity)
        bins = np.linspace(50.0, 400.0, 15)
        marginal = density.marginal_mass(bins)
        direct = np.array(
            [density.integrate((low, high)) for low, high in zip(bins[:-1], bins[1:])]
        )
        np.testing.assert_allclose(marginal, direct, rtol=0.08, atol=1e-8)

    def test_density_jacobian_and_conditional_yx(self):
        density = PairDensity(self.flux, self.acceptance, Resolution(0.0, 1), bins=1024)
        values = density.density(np.array([100.0, 200.0]), np.array([0.0, 0.0]))
        self.assertEqual(values.shape, (2,))
        yx, conditional = density.conditional_yx(140.0)
        if yx.size:
            self.assertAlmostEqual(np.trapezoid(conditional, yx), 1.0, places=2)

    def test_vectorized_rapidity_ranges_match_direct_integrals(self):
        density = PairDensity(self.flux, self.acceptance, Resolution(0.0, 1), bins=2048)
        mass_range = (80.0, 240.0)
        low = np.array([-1.0, -0.4, 0.0, 0.3])
        high = low + np.array([0.2, 0.3, 0.25, 0.4])
        fast = density.integrate_yx_ranges(
            mass_range, low, high, bins=16384, quadrature_order=4096
        )
        direct = np.array(
            [density.integrate(mass_range, bounds) for bounds in zip(low, high)]
        )
        np.testing.assert_allclose(fast, direct, rtol=0.03, atol=2.0e-8)
        self.assertIs(
            density.rapidity_profile(mass_range, bins=16384, quadrature_order=4096),
            density.rapidity_profile(mass_range, bins=16384, quadrature_order=4096),
        )


if __name__ == "__main__":
    unittest.main()
