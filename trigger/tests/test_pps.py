import unittest

import numpy as np

from minbias.flux import Acceptance, ProtonFlux
from trigger.pps import accepted_groups, criterion_masks, sample_pps_pairs


class ExactPPSPileupTest(unittest.TestCase):
    def setUp(self):
        self.flux = ProtonFlux(
            event=np.array([0, 0]),
            arm=np.array([-1, 1]),
            xi=np.array([0.01, 0.01]),
            px=np.zeros(2),
            py=np.zeros(2),
            process=np.ones(2),
            metadata={"n_inelastic_generated": 1, "sqrt_s_gev": 10_000.0},
        )
        self.groups = accepted_groups(self.flux, Acceptance([(0.005, 0.02)]))

    def sample(self, n_inelastic, seed=4):
        return sample_pps_pairs(
            self.groups,
            n_inelastic,
            np.random.default_rng(seed),
            sqrt_s_gev=10_000.0,
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            beam_sigma_z_cm=0.0,
            single_arm_time_resolution_ps=10.0,
            pv_z_resolution_cm=0.0,
            pv_time_resolution_ps=30.0,
            nsigma=2.0,
        )

    def test_zero_exact_interactions_produces_no_pairs(self):
        self.assertEqual(self.sample(0)["mass"].size, 0)

    def test_one_exact_interaction_preserves_group_and_mass(self):
        pairs = self.sample(1)
        self.assertEqual(pairs["mass"].tolist(), [100.0])
        self.assertTrue(pairs["same_interaction"][0])
        self.assertTrue(criterion_masks(pairs)["pps_mass"][0])

    def test_fixed_seed_is_repeatable(self):
        first = self.sample(3, 11)
        second = self.sample(3, 11)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_matched_signal_pair_is_injected_at_primary_vertex(self):
        pairs = sample_pps_pairs(
            self.groups,
            0,
            np.random.default_rng(5),
            sqrt_s_gev=10_000.0,
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            beam_sigma_z_cm=0.0,
            single_arm_time_resolution_ps=1.0e-12,
            pv_z_resolution_cm=0.0,
            pv_time_resolution_ps=1.0e-12,
            nsigma=2.0,
            matched_xi=np.array([0.01, 0.01]),
            matched_arm=np.array([-1, 1]),
        )
        self.assertEqual(pairs["mass"].tolist(), [100.0])
        self.assertTrue(pairs["signal_signal"][0])
        self.assertFalse(pairs["signal_pileup"][0])
        self.assertFalse(pairs["pileup_pileup"][0])
        self.assertTrue(pairs["z_ok"][0])
        self.assertTrue(pairs["central_ok"][0])

    def test_pair_sources_distinguish_signal_mixed_and_pileup(self):
        pairs = sample_pps_pairs(
            self.groups,
            1,
            np.random.default_rng(6),
            sqrt_s_gev=10_000.0,
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            beam_sigma_z_cm=0.0,
            single_arm_time_resolution_ps=1.0e-12,
            pv_z_resolution_cm=0.0,
            pv_time_resolution_ps=1.0e-12,
            nsigma=2.0,
            matched_xi=np.array([0.01, 0.01]),
            matched_arm=np.array([-1, 1]),
        )
        self.assertEqual(np.count_nonzero(pairs["signal_signal"]), 1)
        self.assertEqual(np.count_nonzero(pairs["signal_pileup"]), 2)
        self.assertEqual(np.count_nonzero(pairs["pileup_pileup"]), 1)

    def test_matched_proton_arguments_must_be_aligned(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            sample_pps_pairs(
                self.groups,
                0,
                np.random.default_rng(7),
                sqrt_s_gev=10_000.0,
                mass_range=(90.0, 110.0),
                xi_resolution=0.0,
                beam_sigma_z_cm=0.0,
                single_arm_time_resolution_ps=0.0,
                pv_z_resolution_cm=0.0,
                pv_time_resolution_ps=0.0,
                nsigma=2.0,
                matched_xi=np.array([0.01]),
            )


if __name__ == "__main__":
    unittest.main()
