import unittest

import numpy as np

from minbias.vertex import (
    C_CM_PER_PS,
    pps_vertex_resolution_cm,
    vertex_likelihood_parameters,
    vertex_likelihood_table,
    vertex_log_likelihood_ratio,
    vertex_overlap_probability,
)


class VertexTest(unittest.TestCase):
    def test_default_unrelated_probabilities(self):
        self.assertAlmostEqual(
            vertex_overlap_probability(truth_status="unrelated", timing_mode="z_only"),
            0.041932,
            places=5,
        )
        self.assertAlmostEqual(
            vertex_overlap_probability(truth_status="unrelated", timing_mode="luminous"),
            0.040017,
            places=5,
        )
        self.assertAlmostEqual(
            vertex_overlap_probability(truth_status="unrelated"),
            0.007555,
            places=5,
        )

    def test_default_matched_probabilities(self):
        self.assertAlmostEqual(
            vertex_overlap_probability(truth_status="matched", timing_mode="z_only"),
            0.954500,
            places=5,
        )
        self.assertAlmostEqual(
            vertex_overlap_probability(truth_status="matched"),
            0.911070,
            places=5,
        )

    def test_zero_beam_width_makes_truth_statuses_equal(self):
        matched = vertex_overlap_probability(truth_status="matched", beam_sigma_z_cm=0.0)
        unrelated = vertex_overlap_probability(truth_status="unrelated", beam_sigma_z_cm=0.0)
        self.assertEqual(matched, unrelated)

    def test_unrelated_probabilities_close_against_direct_sampling(self):
        rng = np.random.default_rng(9)
        size = 300_000
        beam_sigma = 5.7
        pps_time_sigma = 10.0
        pv_z_sigma = 0.001
        pv_time_sigma = 30.0
        nsigma = 2.0
        left_z, right_z, left_u, right_u, pv_z, pv_u = rng.normal(
            0.0, beam_sigma, size=(6, size)
        )
        left_arrival = left_u + left_z + C_CM_PER_PS * rng.normal(
            0.0, pps_time_sigma, size=size
        )
        right_arrival = right_u - right_z + C_CM_PER_PS * rng.normal(
            0.0, pps_time_sigma, size=size
        )
        pp_z = 0.5 * (left_arrival - right_arrival)
        pp_u = 0.5 * (left_arrival + right_arrival)
        pv_z += rng.normal(0.0, pv_z_sigma, size=size)
        pv_u += C_CM_PER_PS * rng.normal(0.0, pv_time_sigma, size=size)
        pps_sigma = pps_vertex_resolution_cm(pps_time_sigma)
        z_ok = np.abs(pp_z - pv_z) <= nsigma * np.hypot(pps_sigma, pv_z_sigma)
        sampled = {
            "z_only": np.mean(z_ok),
            "luminous": np.mean(z_ok & (np.abs(pp_u) <= nsigma * beam_sigma)),
            "central": np.mean(
                z_ok
                & (
                    np.abs(pp_u - pv_u)
                    <= nsigma * np.hypot(pps_sigma, C_CM_PER_PS * pv_time_sigma)
                )
            ),
        }
        for mode, probability in sampled.items():
            with self.subTest(mode=mode):
                self.assertAlmostEqual(
                    probability,
                    vertex_overlap_probability(
                        truth_status="unrelated", timing_mode=mode
                    ),
                    delta=0.001,
                )

    def test_pps_resolution_and_invalid_inputs(self):
        self.assertAlmostEqual(pps_vertex_resolution_cm(10.0), 0.211985, places=5)
        invalid = (
            {"truth_status": "other"},
            {"truth_status": "matched", "timing_mode": "ellipse"},
            {"truth_status": "matched", "beam_sigma_z_cm": -1.0},
            {"truth_status": "matched", "pv_z_resolution_cm": -1.0},
            {"truth_status": "matched", "single_arm_time_resolution_ps": 0.0},
            {"truth_status": "matched", "nsigma": 0.0},
            {"truth_status": "matched", "pv_time_resolution_ps": None},
            {"truth_status": "matched", "pv_time_resolution_ps": 0.0},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                vertex_overlap_probability(**arguments)

    def test_exact_likelihood_widths_and_monotonicity(self):
        parameters = vertex_likelihood_parameters()
        self.assertAlmostEqual(parameters["matched_sigma_z_cm"], 0.211987, places=5)
        self.assertAlmostEqual(parameters["matched_sigma_ct_cm"], 0.92402, places=5)
        self.assertAlmostEqual(parameters["unrelated_sigma_z_cm"], 8.0638, places=3)
        self.assertAlmostEqual(parameters["unrelated_sigma_ct_cm"], 8.1137, places=3)
        at_origin = float(vertex_log_likelihood_ratio(0.0, 0.0))
        away = vertex_log_likelihood_ratio(
            np.asarray([0.1, 0.5, 1.0]), np.asarray([0.1, 0.5, 1.0])
        )
        self.assertEqual(at_origin, parameters["maximum_log_likelihood_ratio"])
        self.assertTrue(np.all(np.diff(away) < 0.0))

    def test_likelihood_table_is_normalized_and_ordered(self):
        table = vertex_likelihood_table(bins=8)
        np.testing.assert_allclose(table["matched_probability"], 0.125, atol=2.0e-12)
        self.assertAlmostEqual(float(table["unrelated_probability"].sum()), 1.0)
        self.assertTrue(np.all(table["matched_probability"] > 0.0))
        self.assertTrue(np.all(table["unrelated_probability"] > 0.0))
        self.assertTrue(np.all(np.diff(table["log_likelihood_ratio"]) < 0.0))

    def test_likelihood_table_agrees_with_direct_sampling(self):
        rng = np.random.default_rng(17)
        table = vertex_likelihood_table(bins=6)
        parameters = table["parameters"]
        edges = table["quadratic_edges"]
        for truth_status in ("matched", "unrelated"):
            prefix = truth_status
            dz = rng.normal(0.0, parameters[f"{prefix}_sigma_z_cm"], 300_000)
            dct = rng.normal(0.0, parameters[f"{prefix}_sigma_ct_cm"], 300_000)
            quadratic = (
                parameters["quadratic_coefficient_z_cm2_inv"] * dz**2
                + parameters["quadratic_coefficient_ct_cm2_inv"] * dct**2
            )
            sampled, _ = np.histogram(quadratic, bins=edges)
            sampled = sampled / sampled.sum()
            np.testing.assert_allclose(
                sampled, table[f"{truth_status}_probability"], atol=0.0025
            )


if __name__ == "__main__":
    unittest.main()
