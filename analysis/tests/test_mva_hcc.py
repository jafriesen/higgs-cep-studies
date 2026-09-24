import tempfile
import unittest
from pathlib import Path

import awkward as ak
import numpy as np

from analysis.MVA_hcc import common
from analysis.MVA_hcc import prepare_dataset as prepare
from analysis.MVA_hcc import prepare_cc_only_dataset as prepare_cc_only
from analysis.MVA_hcc import scan_normalizations as hcc_scan
from analysis.MVA_hcc import train_model as train
from analysis.MVA_new import scan_survival_factors as hbb_scan


class HccComponentContractTest(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "tagging": {"eff_c": 0.65, "mistag_b_to_c": 0.055},
            "normalization": {"combinatorial_acceptance_factor": 0.005},
        }

    def test_component_to_class_mapping(self):
        manifest = common.component_manifest(self.parameters)
        observed = {item["name"]: item["class_name"] for item in manifest}
        self.assertEqual(observed["Hcc"], "Hcc")
        self.assertEqual(observed["QCDcc_superchic"], "exclusive_QCD")
        self.assertEqual(observed["QCDbb_superchic"], "exclusive_QCD")
        self.assertEqual(observed["QEDcc_superchic"], "exclusive_QED")
        self.assertEqual(observed["QCDcc_madgraph"], "nonexclusive_QCD")
        self.assertEqual(observed["QCDbb_madgraph"], "nonexclusive_QCD")
        self.assertEqual(observed["Hbb_superchic"], "Hbb_resonant")
        self.assertEqual(len(set(item["id"] for item in manifest)), 7)
        self.assertEqual(len(common.CLASS_NAMES), 5)

    def test_double_tag_factors_are_applied_once(self):
        manifest = common.component_manifest(self.parameters)
        for item in manifest:
            expected = 0.65**2 if item["source_flavor"] == "cc" else 0.055**2
            self.assertAlmostEqual(item["tag_factor"], expected)
        self.assertAlmostEqual(common.tag_factor(self.parameters, "cc"), 0.4225)
        self.assertAlmostEqual(common.tag_factor(self.parameters, "bb"), 0.003025)

    def test_normalization_scales_select_only_matching_components(self):
        components = common.component_manifest(self.parameters)
        charm = common.normalization_scales(components, "eff_c", 0.5, 1.0)
        mistag = common.normalization_scales(components, "mistag_b_to_c", 0.5, 1.0)
        survival = common.normalization_scales(components, "survival", 0.5, 1.0)
        for index, component in enumerate(components):
            self.assertEqual(charm[index], 0.25 if component["source_flavor"] == "cc" else 1.0)
            self.assertEqual(mistag[index], 0.25 if component["source_flavor"] == "bb" else 1.0)
            self.assertEqual(survival[index], 0.5 if component["survival_scaled"] else 1.0)

    def test_cc_only_comparison_contract(self):
        metadata = {
            "classes": list(common.CLASS_NAMES),
            "components": common.component_manifest(self.parameters),
        }
        classes, components = prepare_cc_only.comparison_contract(metadata)
        self.assertEqual(
            classes,
            ["Hcc", "exclusive_QCD", "exclusive_QED", "nonexclusive_QCD"],
        )
        self.assertEqual(
            [item["name"] for item in components],
            list(prepare_cc_only.COMPONENT_NAMES),
        )
        self.assertEqual([item["id"] for item in components], list(range(4)))
        scales = common.normalization_scales(components, "survival", 0.06, 0.03)
        np.testing.assert_allclose(scales, [2.0, 2.0, 1.0, 1.0])


class HccPreparationTest(unittest.TestCase):
    def test_stitching_is_independent_within_one_flavor(self):
        masks = {
            "v02": np.asarray([True, True, False]),
            "v03": np.asarray([False, True, True]),
        }
        luminosities = {"v02": 10.0, "v03": 30.0}
        np.testing.assert_allclose(
            common.stitched_cross_section_weights("v02", masks, luminosities),
            [0.1, 0.025, 0.0],
        )
        np.testing.assert_allclose(
            common.stitched_cross_section_weights("v03", masks, luminosities),
            [0.0, 0.025, 1.0 / 30.0],
        )

    def test_nonpileup_proton_extraction_ignores_higher_energy_pileup(self):
        key = prepare.source.branch_name
        arrays = {
            key("Particle", "PID"): ak.Array([[2212, 2212, 2212], [2212, 2212]]),
            key("Particle", "Status"): ak.Array([[1, 1, 1], [1, 1]]),
            key("Particle", "IsPU"): ak.Array([[0, 1, 0], [0, 0]]),
            key("Particle", "Pz"): ak.Array([[-6800.0, -6990.0, 6750.0], [-6700.0, 6600.0]]),
            key("Particle", "E"): ak.Array([[6801.0, 6991.0, 6751.0], [6701.0, 6601.0]]),
        }
        left, right = prepare.extract_nonpileup_proton_energies(arrays, np.asarray([0, 1]))
        np.testing.assert_allclose(left, [6801.0, 6701.0])
        np.testing.assert_allclose(right, [6751.0, 6601.0])

    def test_lhe_proton_extraction_uses_selected_final_protons(self):
        event = """<event>
3 0 1 1 1 1
2212 -1 0 0 0 0 0 0 7000 7000
2212 1 0 0 0 0 0 0 6800 6800
2212 1 0 0 0 0 0 0 -6700 6700
</event>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.dat"
            path.write_text(event + event.replace("6800", "6600"), encoding="utf-8")
            left, right = prepare.parse_lhe_proton_xi(
                path, np.asarray([1], dtype=np.int64), 14000.0
            )
        np.testing.assert_allclose(left, [300.0 / 7000.0])
        np.testing.assert_allclose(right, [400.0 / 7000.0])

    def test_conditional_sampling_keeps_group_and_weight(self):
        pool = {
            "yx": np.asarray([-1.0, -0.1, 0.1, 1.0]),
            "mx": np.asarray([110.0, 120.0, 130.0, 140.0]),
            "weight": np.asarray([2.0, 1.0, 3.0, 4.0]),
        }
        sampled = prepare.source.conditional_pair_sample(
            central_matrix=np.asarray([[7.0]]),
            dijet_rapidity=np.asarray([0.0]),
            central_group=np.asarray([12]),
            central_campaign=np.asarray(["v01"]),
            central_split=np.asarray([0], dtype=np.uint8),
            stitch_weight_fb=np.asarray([2.0]),
            coverage_mask=np.asarray([1], dtype=np.uint8),
            pool=pool,
            train_pairs=1,
            evaluation_pairs=10000,
            seed=19,
        )
        self.assertTrue(np.all(sampled["group"] == 12))
        self.assertAlmostEqual(sampled["stitch_weight_fb"].sum(), 0.8, places=12)


class HccTrainingTest(unittest.TestCase):
    def test_group_folds_never_split_proton_copies(self):
        groups = np.repeat(np.arange(100), 4)
        folds = train.assign_folds(groups, 12345)
        train.assert_group_safe(folds, groups)
        for group in np.unique(groups):
            self.assertEqual(np.unique(folds[groups == group]).size, 1)

    def test_five_class_plugin_score(self):
        probabilities = np.asarray(
            [[0.5, 0.1, 0.1, 0.2, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]]
        )
        kappas = np.asarray([2.0, 3.0, 4.0, 5.0])
        expected = np.log(probabilities[:, 0]) - np.log(probabilities[:, 1:] @ kappas)
        np.testing.assert_allclose(common.plugin_score(probabilities, kappas), expected)

    def test_saved_model_and_calibration_reload(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(250, 4)).astype(np.float32)
        classes = np.tile(np.arange(5, dtype=np.int8), 50)
        order = rng.permutation(classes.size)
        data = {
            "x": x[order],
            "class": classes[order],
            "training_mixture_weight": np.ones(classes.size),
        }
        train_rows = np.arange(200)
        stop_rows = np.arange(200, 250)
        model, calibrator = train.fit_calibrated(
            data, train_rows, stop_rows, seed=9, n_estimators=5, n_classes=5
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            record = train.save_and_check_model(
                model, calibrator, data["x"][stop_rows[:16]], 0, Path(temp_dir)
            )
        self.assertLess(record["reload_margin_max_abs_difference"], 1.0e-12)
        self.assertLess(record["reload_probability_max_abs_difference"], 1.0e-7)


class SurvivalCategoryScanTest(unittest.TestCase):
    def test_hbb_category_significance_scales_only_superchic_components(self):
        category_mass = np.asarray(
            [
                [[2.0], [1.0], [1.0]],
                [[1.0], [0.0], [3.0]],
            ]
        )
        expected = np.sqrt(4.0**2 / 7.0 + 2.0**2 / 5.0)
        self.assertAlmostEqual(
            hbb_scan.categorized_significance(category_mass, 2.0), expected
        )

    def test_hcc_category_significance_combines_categories_and_mass_bins(self):
        category_mass = np.asarray(
            [
                [[2.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                [[1.0, 2.0], [3.0, 0.0], [0.0, 2.0]],
            ]
        )
        signal = category_mass[:, 0]
        total = category_mass.sum(axis=1)
        expected = np.sqrt(np.sum(signal * signal / total))
        self.assertAlmostEqual(hcc_scan.categorized_significance(category_mass), expected)


if __name__ == "__main__":
    unittest.main()
