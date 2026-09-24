import unittest

import numpy as np

from analysis.MVA import optimize_dijet_mva_multiclass_protons as optimize
from analysis.MVA import run_dijet_mva_multiclass_protons as prepare
from analysis.MVA import scan_multiclass_proton_crossfit as crossfit_scan


class ConditionalProtonSamplingTest(unittest.TestCase):
    def test_conditional_sampling_matches_rejection_distribution_and_weight(self):
        pool = {
            "yx": np.asarray([-1.0, -0.1, 0.1, 1.0]),
            "mx": np.asarray([110.0, 120.0, 130.0, 140.0]),
            "weight": np.asarray([2.0, 1.0, 3.0, 4.0]),
        }
        sampled = prepare.conditional_pair_sample(
            central_matrix=np.asarray([[7.0]]),
            dijet_rapidity=np.asarray([0.0]),
            central_group=np.asarray(["mg:0"]),
            central_campaign=np.asarray(["v01"]),
            central_split=np.asarray([3], dtype=np.uint8),
            stitch_weight_fb=np.asarray([2.0]),
            coverage_mask=np.asarray([7], dtype=np.uint8),
            pool=pool,
            train_pairs=1,
            evaluation_pairs=100000,
            seed=19,
        )
        # The rejection sampler accepts weights 1 and 3 out of total weight 10.
        self.assertAlmostEqual(np.sum(sampled["stitch_weight_fb"]), 0.8, places=12)
        observed_high = np.mean(sampled["mx"] == 130.0)
        self.assertAlmostEqual(observed_high, 0.75, delta=0.006)
        self.assertTrue(np.all(np.abs(sampled["features"][:, -1]) < 0.2))

    def test_copies_of_one_event_keep_one_split(self):
        event_ids = np.arange(10000)
        split = prepare.deterministic_group_split(event_ids, 31, 123)
        repeated = np.repeat(split, 4)
        groups = np.repeat(event_ids, 4)
        optimize.assert_group_safe(repeated, groups)
        fractions = np.bincount(split, minlength=4) / split.size
        np.testing.assert_allclose(fractions, [0.60, 0.10, 0.15, 0.15], atol=0.02)


class SignificanceTest(unittest.TestCase):
    def test_cumulative_group_support_matches_direct_aggregation(self):
        thresholds = np.asarray([-1.0, 0.0, 1.0])
        score = np.asarray([-0.5, 0.5, 1.5, 1.2])
        groups = np.asarray([0, 0, 1, 2])
        weights = np.asarray([1.0, 2.0, 3.0, 4.0])
        counts, neff = crossfit_scan.cumulative_group_support(
            score, groups, weights, thresholds
        )
        np.testing.assert_array_equal(counts, [3, 3, 2])
        expected_neff = [
            10.0**2 / (3.0**2 + 3.0**2 + 4.0**2),
            9.0**2 / (2.0**2 + 3.0**2 + 4.0**2),
            7.0**2 / (3.0**2 + 4.0**2),
        ]
        np.testing.assert_allclose(neff, expected_neff)

    def test_threshold_candidates_include_minority_class_tail(self):
        score = np.r_[
            np.full(100000, -20.0),
            np.linspace(-9.0, -5.0, 100),
            np.linspace(-8.0, -4.0, 100),
        ]
        classes = np.r_[
            np.full(100000, 2),
            np.ones(100),
            np.zeros(100),
        ].astype(np.int8)
        thresholds = optimize.candidate_thresholds(score, classes)
        self.assertGreater(np.max(thresholds), -5.0)
        self.assertTrue(np.any((thresholds > -7.0) & (thresholds < -5.0)))

    def test_weighted_mass_bins_and_categories(self):
        classes = np.asarray([0, 0, 1, 1, 2, 2])
        mass = np.asarray([117.5, 118.5, 117.5, 118.5, 117.5, 118.5])
        weights = np.asarray([3.0, 4.0, 1.0, 3.0, 2.0, 1.0])
        all_rows = np.ones(classes.size, dtype=bool)
        significance, yields = optimize.histogram_significance(
            classes, mass, weights, [all_rows]
        )
        expected = np.sqrt(3.0**2 / (3.0 + 3.0) + 4.0**2 / (4.0 + 4.0))
        self.assertAlmostEqual(significance, expected)
        np.testing.assert_allclose(yields, [7.0, 4.0, 3.0])

        first_bin = mass < 118.0
        categorized, _ = optimize.histogram_significance(
            classes, mass, weights, [first_bin, ~first_bin]
        )
        self.assertAlmostEqual(categorized, expected)

    def test_group_level_mc_requirement_aggregates_copies(self):
        groups = np.repeat(np.arange(120), 4)
        classes = np.repeat(np.r_[np.ones(60), np.full(60, 2)], 4).astype(int)
        weights = np.ones(groups.size)
        selection = np.ones(groups.size, dtype=bool)
        valid, diagnostics = optimize.eligible(
            classes, groups, weights, [selection]
        )
        self.assertTrue(valid)
        self.assertEqual(diagnostics[0]["QCDbb"]["groups"], 60)
        self.assertEqual(diagnostics[0]["QCDbb_madgraph"]["groups"], 60)
        self.assertAlmostEqual(diagnostics[0]["QCDbb"]["neff"], 60.0)


if __name__ == "__main__":
    unittest.main()
