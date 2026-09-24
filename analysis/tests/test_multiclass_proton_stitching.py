import unittest

import numpy as np

from analysis.MVA import run_dijet_mva_multiclass_protons as mva


class MulticlassProtonStitchingTest(unittest.TestCase):
    def test_phase_space_masks_match_campaign_run_cards(self):
        hard_bb = {
            "hard_bb_valid": np.ones(5, dtype=bool),
            "hard_b1_pt": np.asarray([20.0, 30.0, 90.0, 30.0, 40.0]),
            "hard_b2_pt": np.asarray([20.0, 30.0, 90.0, 30.0, 40.0]),
            "hard_b1_eta": np.asarray([0.0, 0.0, 0.0, 2.7, 1.0]),
            "hard_b2_eta": np.asarray([0.0, 0.0, 0.0, 2.7, 1.0]),
            "hard_bb_mass": np.asarray([60.0, 100.0, 170.0, 100.0, 120.0]),
        }
        masks = mva.madgraph_phase_space_masks(hard_bb)
        np.testing.assert_array_equal(
            masks["QCDbb__v01"], [True, True, False, True, True]
        )
        np.testing.assert_array_equal(
            masks["QCDbb__v02"], [False, True, True, False, True]
        )
        np.testing.assert_array_equal(
            masks["QCDbb__v03"], [False, True, False, False, True]
        )

    def test_overlap_uses_combined_effective_mc_luminosity(self):
        phase_masks = {
            "QCDbb__v01": np.asarray([True, True, False]),
            "QCDbb__v02": np.asarray([False, True, True]),
        }
        luminosities = {"QCDbb__v01": 10.0, "QCDbb__v02": 20.0}
        np.testing.assert_allclose(
            mva.stitched_cross_section_weights(
                "QCDbb__v01", phase_masks, luminosities
            ),
            [0.1, 1.0 / 30.0, 0.0],
        )
        np.testing.assert_allclose(
            mva.stitched_cross_section_weights(
                "QCDbb__v02", phase_masks, luminosities
            ),
            [0.0, 1.0 / 30.0, 0.05],
        )

    def test_three_way_overlap_uses_all_effective_luminosities(self):
        phase_masks = {
            "QCDbb__v01": np.asarray([True]),
            "QCDbb__v02": np.asarray([True]),
            "QCDbb__v03": np.asarray([True]),
        }
        luminosities = {
            "QCDbb__v01": 1.0,
            "QCDbb__v02": 2.0,
            "QCDbb__v03": 7.0,
        }
        for campaign in luminosities:
            np.testing.assert_allclose(
                mva.stitched_cross_section_weights(
                    campaign, phase_masks, luminosities
                ),
                [0.1],
            )

    def test_configured_phase_space_matches_saved_run_cards(self):
        manifest = mva.validate_madgraph_campaign_cards(
            ("QCDbb__v01", "QCDbb__v02", "QCDbb__v03")
        )
        self.assertEqual(set(manifest), {
            "QCDbb__v01", "QCDbb__v02", "QCDbb__v03"
        })

    def test_class_balance_preserves_within_class_weight_ratios(self):
        classes = np.asarray([0, 0, 1, 1])
        mixture = np.asarray([1.0, 3.0, 2.0, 2.0])
        weights = mva.class_balanced_weights(classes, mixture)
        self.assertAlmostEqual(np.sum(weights[classes == 0]), 2.0)
        self.assertAlmostEqual(np.sum(weights[classes == 1]), 2.0)
        self.assertAlmostEqual(weights[1] / weights[0], 3.0)


if __name__ == "__main__":
    unittest.main()
