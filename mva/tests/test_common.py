import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import awkward as ak
import numpy as np
import yaml

from mva.common.config import load_channel_config
from mva.common.dataset import (
    PROTON_TRANSVERSE_FIELDS,
    component_truth_pid_abs,
    correction_source_sample,
    load_component_correction_map,
)
from mva.common.features import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    PROTON_FEATURE_NAMES,
    TRACKER_ETA,
    TRACK_MAX_ABS_ETA,
    TRANSVERSE_TRACK_MAX_ABS_ETA,
    _match_two_to_many,
    _recovered_chosen_pair,
    stored_central_features,
    transverse_track_mask,
)
from mva.common.protons import (
    LegacyPairDensity,
    parse_hepmc_protons,
    parse_lhe_proton_xi,
    parse_lhe_protons,
)
from mva.common import training
from mva.common.training import (
    architecture_classes,
    architecture_training_mask,
    assign_folds,
    balanced_weights,
    fold_rows,
    hierarchical_probabilities,
)
from mva.common.weights import (
    disjoint_campaign_weights,
    stitched_cross_section_weights,
    subsample_scale,
    tag_factor,
)


REPO = Path(__file__).resolve().parents[2]
CHARM_LOCKED_FEATURES = (
    "delta_phi_jj",
    "yx_minus_dijet_rapidity",
    "sum_gap_size",
    "sum_track_pt_outside_jets",
    "dijet_mass",
    "jet2_eta",
    "n_tracks_outside_jets",
    "jet1_eta",
    "dijet_mass_fsr",
    "eta_rms_outside",
    "max_track_pt_outside_jets",
    "sum_track_pt_interjet",
    "sum_track_pt_outside_pt2",
    "pt_asymmetry",
    "jet1_charged_fraction",
    "jet2_pt_over_mjj",
    "jet2_pull_magnitude",
    "dijet_rapidity",
    "delta_eta_jj",
    "jet_multiplicity",
)


class ConfigurationTest(unittest.TestCase):
    def test_full_schema_has_62_features(self):
        self.assertEqual(len(ALL_FEATURE_NAMES), 62)
        self.assertIn("jet1_pt", FEATURE_NAMES)
        self.assertIn("jet2_pt", FEATURE_NAMES)
        self.assertIn("n_tracks_transverse", FEATURE_NAMES)
        self.assertIn("sum_track_pt_transverse", FEATURE_NAMES)

    def test_track_acceptance_extends_to_eta_four(self):
        self.assertEqual(TRACK_MAX_ABS_ETA, 4.0)
        self.assertEqual(TRACKER_ETA, 4.0)
        self.assertEqual(TRANSVERSE_TRACK_MAX_ABS_ETA, 1.75)

    def test_transverse_track_mask_uses_wrapped_phi_and_paper_eta_range(self):
        track_eta = ak.Array(
            [[0.0, 0.0, 1.75, -1.74], [0.0, 0.0]]
        )
        track_phi = ak.Array(
            [[0.0, np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0], [-2.0, 0.0]]
        )
        mask = transverse_track_mask(track_eta, track_phi, np.array([0.0, 3.0]))
        self.assertEqual(
            ak.to_list(mask), [[False, True, False, True], [True, False]]
        )

    def test_hcc_profiles_and_disjoint_nominal_campaigns(self):
        path = REPO / "mva/Hcc/config.yaml"
        nominal = load_channel_config(path, "nominal_full")
        qcdcc = next(item for item in nominal["components"] if item["name"] == "QCDcc_madgraph")
        self.assertEqual(qcdcc["campaigns"], [f"QCDcc__v0{index}" for index in range(2, 8)])
        self.assertEqual(qcdcc["campaign_combination"], "disjoint")
        self.assertEqual(
            nominal["classes"],
            [
                "Hcc",
                "exclusive_QCD",
                "exclusive_QED",
                "nonexclusive_QCD",
                "Hbb_resonant",
            ],
        )
        for profile_name in (
            "nominal_full",
            "nominal_no_hbb_resonant",
            "charm_only_three_class",
            "charm_only_four_class",
        ):
            profile = load_channel_config(path, profile_name)
            self.assertEqual(profile["profile"]["max_abs_jet_eta"], 3.0)
        parity = load_channel_config(path, "parity_five_class")
        self.assertIsNone(parity["profile"]["max_abs_jet_eta"])
        self.assertEqual(parity["profile"]["feature_set"], "legacy_locked")
        no_hbb = load_channel_config(path, "nominal_no_hbb_resonant")
        self.assertNotIn("Hbb_superchic", [item["name"] for item in no_hbb["components"]])
        self.assertEqual(
            no_hbb["classes"],
            ["Hcc", "exclusive_QCD", "exclusive_QED", "nonexclusive_QCD"],
        )
        charm_split = load_channel_config(path, "charm_only_four_class")
        self.assertEqual(
            charm_split["classes"],
            ["Hcc", "exclusive_QCD", "exclusive_QED", "nonexclusive_QCD"],
        )
        self.assertEqual(tuple(nominal["feature_sets"]["locked"]), CHARM_LOCKED_FEATURES)
        self.assertFalse(set(PROTON_TRANSVERSE_FIELDS) & set(nominal["feature_sets"]["locked"]))
        self.assertFalse(set(PROTON_TRANSVERSE_FIELDS) & set(ALL_FEATURE_NAMES))

    def test_hbb_locked_profile_has_reference_feature_count(self):
        config = load_channel_config(REPO / "mva/Hbb/config.yaml", "nominal_three_class")
        # The nominal set is the permutation-ranked twenty; feature_count_v1 does not
        # resolve 42 as better than 20, so the larger set is not the default.
        self.assertEqual(len(config["feature_sets"]["locked"]), 20)
        # The parity control must keep the 48 that significance_v3_all_campaigns used.
        self.assertEqual(len(config["feature_sets"]["legacy_locked"]), 48)
        self.assertEqual(config["classes"], ["Hbb", "exclusive_QCDbb", "nonexclusive_QCDbb"])
        self.assertEqual(config["profile"]["max_abs_jet_eta"], 3.0)
        candidates = yaml.safe_load(
            (REPO / "mva/feature_candidates.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["feature_sets"]["new_list"], candidates["Hbb"]["new_list"])
        self.assertFalse(set(config["feature_sets"]["new_list"]) - set(ALL_FEATURE_NAMES))
        self.assertTrue(
            {"n_tracks_transverse", "sum_track_pt_transverse"}
            <= set(config["feature_sets"]["new_list_mass_estimator"])
        )
        parity = load_channel_config(REPO / "mva/Hbb/config.yaml", "parity_three_class")
        self.assertEqual(parity["profile"]["feature_set"], "legacy_locked")

    def test_hbb_fsr_corrections_resolve_per_sample_and_slice(self):
        config = load_channel_config(REPO / "mva/Hbb/config.yaml", "fsr_mtd_four_class")
        resolved = {}
        for spec in config["components"]:
            campaigns = spec.get("campaigns") or [None]
            for campaign in campaigns:
                key = correction_source_sample(spec, campaign)
                load_component_correction_map(REPO, spec, campaign)
                resolved[(spec["name"], campaign)] = key

        self.assertEqual(resolved[("Hbb_fsr", None)], "Hbb")
        self.assertEqual(resolved[("QCDbb_fsr", None)], "QCDbb_superchic")
        self.assertEqual(resolved[("QEDbb_fsr", None)], "QEDbb")
        for version in range(4, 10):
            campaign = f"QCDbb__v0{version}"
            self.assertEqual(
                resolved[("QCDbb_madgraph_fsr", campaign)],
                f"QCDbb_madgraph_v0{version}",
            )

    def test_hbb_gg_profile_uses_gluon_truth_and_double_light_mistag(self):
        config = load_channel_config(REPO / "mva/Hbb/config.yaml", "fsr_mtd_five_class_gg")
        gg = next(item for item in config["components"] if item["name"] == "QCDgg_fsr")
        parameters = yaml.safe_load((REPO / "parameters.yaml").read_text(encoding="utf-8"))
        features = set(config["feature_sets"][config["profile"]["feature_set"]])

        self.assertEqual(component_truth_pid_abs(gg), 21)
        self.assertEqual(gg["source_flavor"], "light")
        self.assertEqual(gg["campaign"], "QCDgg__v04")
        self.assertEqual(gg["max_files"], 700)
        self.assertAlmostEqual(tag_factor(parameters, "light", "bb"), 1.0e-4)
        self.assertTrue({"jet1_pull_magnitude", "jet2_pull_magnitude"} <= features)
        self.assertFalse(
            {
                "jet1_ptd",
                "jet2_ptd",
                "jet1_width",
                "jet2_width",
            }
            & features
        )
        self.assertEqual(
            config["classes"],
            [
                "Hbb",
                "exclusive_QCDbb",
                "exclusive_QCDgg",
                "exclusive_QEDbb",
                "nonexclusive_QCDbb",
            ],
        )

    def test_campaign_correction_template_requires_campaign(self):
        spec = {"name": "background", "source_sample": "sample_{campaign}"}
        with self.assertRaisesRegex(ValueError, "needs a campaign"):
            correction_source_sample(spec)


class MatchingTest(unittest.TestCase):
    def test_greedy_matching_is_unique_and_smallest_first(self):
        first_eta = np.array([[0.0, 0.18], [0.0, 2.0]])
        first_phi = np.zeros_like(first_eta)
        second_eta = ak.Array([[0.08, 0.20], [0.05]])
        second_phi = ak.Array([[0.0, 0.0], [0.0]])
        indices, distances, complete = _match_two_to_many(
            first_eta, first_phi, second_eta, second_phi, 0.4
        )
        np.testing.assert_array_equal(indices[0], [0, 1])
        np.testing.assert_allclose(distances[0], [0.08, 0.02])
        self.assertTrue(complete[0])
        self.assertFalse(complete[1])


class RecoveryTest(unittest.TestCase):
    def test_recovered_ncharged_includes_unique_donor_and_muon(self):
        raw_pt = ak.Array([[50.0, 45.0, 10.0]])
        raw_eta = ak.Array([[0.0, 2.0, 1.0]])
        raw_phi = ak.Array([[0.0, 0.0, 0.0]])
        raw_mass = ak.Array([[5.0, 5.0, 1.0]])
        raw_ncharged = ak.Array([[5.0, 7.0, 3.0]])
        original_index = ak.local_index(raw_pt)
        chosen_indices = np.array([[0, 1]])
        correction_map = {
            "raw_pt_support": [0.0, 500.0],
            "eta_bins": [
                {
                    "eta_min": 0.0,
                    "eta_max": 4.0,
                    "supported": True,
                    "nodes": [
                        {"usable": True, "raw_reco_pt_median": 1.0, "target_pt": 1.0},
                        {"usable": True, "raw_reco_pt_median": 499.0, "target_pt": 499.0},
                    ],
                }
            ],
        }
        recovered, used_donors = _recovered_chosen_pair(
            chosen_indices,
            ak.ones_like(raw_pt, dtype=bool),
            original_index,
            raw_pt,
            raw_eta,
            raw_phi,
            raw_mass,
            raw_ncharged,
            ak.Array([[2.0]]),
            ak.Array([[0.1]]),
            ak.Array([[0.0]]),
            correction_map,
        )

        # The equidistant donor is assigned to jet one only; its three charged
        # constituents and the recovered muon augment that final jet object.
        np.testing.assert_array_equal(recovered["ncharged"], [[9.0, 7.0]])
        self.assertEqual(ak.to_list(used_donors), [[2]])


class ProtonRecordTest(unittest.TestCase):
    def test_lhe_parser_returns_unsmeared_transverse_components(self):
        content = """<LesHouchesEvents>
<event>
2212 1 0 0 0 0 -0.10 0.20 -6800.0 6800.0
2212 1 0 0 0 0 0.30 -0.40 6900.0 6900.0
</event>
<event>
2212 1 0 0 0 0 -0.50 -0.60 -6700.0 6700.0
2212 1 0 0 0 0 0.70 0.80 6600.0 6600.0
</event>
</LesHouchesEvents>
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.lhe"
            path.write_text(content, encoding="utf-8")
            protons = parse_lhe_protons(path, np.array([0, 1]), 14000.0)
            legacy_xi = parse_lhe_proton_xi(path, np.array([0, 1]), 14000.0)
        np.testing.assert_allclose(protons["proton_px_left"], [-0.10, -0.50])
        np.testing.assert_allclose(protons["proton_py_left"], [0.20, -0.60])
        np.testing.assert_allclose(protons["proton_px_right"], [0.30, 0.70])
        np.testing.assert_allclose(protons["proton_py_right"], [-0.40, 0.80])
        np.testing.assert_allclose(protons["xi_left"], [200.0 / 7000.0, 300.0 / 7000.0])
        np.testing.assert_allclose(protons["xi_right"], [100.0 / 7000.0, 400.0 / 7000.0])
        np.testing.assert_allclose(legacy_xi[0], protons["xi_left"])
        np.testing.assert_allclose(legacy_xi[1], protons["xi_right"])

    def test_hepmc_parser_uses_final_outgoing_protons(self):
        content = """HepMC::Version 3.02.05
E 0 0 0
P 1 0 2212 9.0 9.0 6999.0 7000.0 0.938 21
P 2 -1 2212 0.30 -0.40 6900.0 6900.0 0.938 1
P 3 -1 2212 -0.10 0.20 -6800.0 6800.0 0.938 1
E 1 0 0
P 1 -1 2212 0.70 0.80 6600.0 6600.0 0.938 1
P 2 -1 2212 -0.50 -0.60 -6700.0 6700.0 0.938 1
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.hepmc"
            path.write_text(content, encoding="utf-8")
            protons = parse_hepmc_protons(path, np.array([0, 1]), 14000.0)
        np.testing.assert_allclose(protons["proton_px_left"], [-0.10, -0.50])
        np.testing.assert_allclose(protons["proton_py_left"], [0.20, -0.60])
        np.testing.assert_allclose(protons["proton_px_right"], [0.30, 0.70])
        np.testing.assert_allclose(protons["proton_py_right"], [-0.40, 0.80])


class WeightTest(unittest.TestCase):
    def test_disjoint_and_stitched_weights_have_different_contracts(self):
        self.assertEqual(
            disjoint_campaign_weights("a", {"a": 20, "b": 10}, {"a": 4.0, "b": 3.0}),
            0.2,
        )
        masks = {
            "a": np.array([True, True, False]),
            "b": np.array([False, True, True]),
        }
        weights = stitched_cross_section_weights("a", masks, {"a": 5.0, "b": 3.0})
        np.testing.assert_allclose(weights, [0.2, 0.125, 0.0])
        self.assertEqual(subsample_scale(100, 20), 5.0)

    def test_legacy_pool_adapter_returns_global_weight_fraction(self):
        pool = {
            "mx": np.array([120.0, 125.0, 140.0]),
            "yx": np.array([-0.1, 0.1, 0.0]),
            "weight": np.array([1.0, 3.0, 6.0]),
        }
        density = LegacyPairDensity(pool)
        values = density.integrate_yx_ranges(
            (117.0, 133.0), np.array([-0.2, 0.0]), np.array([0.0, 0.2])
        )
        np.testing.assert_allclose(values, [0.1, 0.3])


class TrainingInvariantTest(unittest.TestCase):
    def test_folds_are_group_safe_and_weights_balance_classes(self):
        groups = np.array([0, 0, 1, 2, 2, 3])
        folds = assign_folds(groups, seed=17)
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[3], folds[4])
        labels = np.array([0, 0, 1, 1])
        weights = balanced_weights(labels, np.array([1.0, 3.0, 2.0, 2.0]), 2)
        np.testing.assert_allclose(
            [weights[labels == class_id].sum() for class_id in (0, 1)], [2.0, 2.0]
        )

    def test_architecture_class_maps(self):
        classes = [
            "Hbb",
            "exclusive_QCDbb",
            "exclusive_QCDgg",
            "exclusive_QEDbb",
            "nonexclusive_QCDbb",
        ]
        binary, binary_names = architecture_classes(classes, "binary")
        merged, merged_names = architecture_classes(classes, "merged_exclusive")
        np.testing.assert_array_equal(binary, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(merged, [0, 1, 1, 1, 2])
        self.assertEqual(binary_names, ["Hbb", "all_backgrounds"])
        self.assertEqual(merged_names[1], "combined_exclusive_background")

        hcc_binary, hcc_names = architecture_classes(
            ["Hcc", "exclusive_QCD", "nonexclusive_QCD"], "binary"
        )
        np.testing.assert_array_equal(hcc_binary, [0, 1, 1])
        self.assertEqual(hcc_names, ["Hcc", "all_backgrounds"])

    def test_specialists_fit_only_the_requested_background_family(self):
        classes = [
            "Hbb",
            "exclusive_QCDbb",
            "exclusive_QCDgg",
            "exclusive_QEDbb",
            "nonexclusive_QCDbb",
        ]
        labels = np.arange(5)
        nonexclusive = architecture_training_mask(
            labels, classes, "nonexclusive_specialist"
        )
        exclusive = architecture_training_mask(
            labels, classes, "exclusive_specialist"
        )
        np.testing.assert_array_equal(nonexclusive, [True, False, False, False, True])
        np.testing.assert_array_equal(exclusive, [True, True, True, True, False])

    def test_stopping_split_is_stratified_and_group_safe(self):
        labels = np.repeat([0, 1], 20)
        groups = np.tile(np.repeat(np.arange(10), 2), 2)
        components = np.repeat([0, 1], 20)
        folds = np.ones(40, dtype=np.int8)
        eligible = np.ones(40, dtype=bool)
        pooled = labels == 1
        train, stop = fold_rows(
            folds,
            fold=0,
            eligible=eligible,
            pooled=pooled,
            cap=None,
            stop_cap=None,
            seed=17,
            labels=labels,
            groups=groups,
            components=components,
        )
        self.assertEqual(np.bincount(labels[stop], minlength=2).tolist(), [2, 2])
        self.assertEqual(np.bincount(labels[train], minlength=2).tolist(), [18, 18])
        stop_keys = set(zip(components[stop], groups[stop]))
        train_keys = set(zip(components[train], groups[train]))
        self.assertFalse(stop_keys & train_keys)

    def test_tail_metric_uses_fixed_weighted_signal_efficiency(self):
        scores = np.array([0.9, 0.8, 0.7, 0.85, 0.1, 0.05])
        labels = np.array([0, 0, 0, 1, 1, 1])
        weights = np.ones(6)
        nonexclusive = labels == 1
        metric, thresholds, efficiencies = training.tail_rejection_metric(
            scores, labels, weights, nonexclusive, [1 / 3, 2 / 3]
        )
        np.testing.assert_allclose(thresholds, [0.9, 0.8])
        np.testing.assert_allclose(efficiencies, [1 / 3, 2 / 3])
        self.assertAlmostEqual(metric, np.mean(np.log([1 / 3, 2 / 3])))

    def test_hard_negative_mining_is_campaign_stratified(self):
        scores = np.array([0.1, 0.9, 0.2, 0.8, np.nan, np.nan])
        rows = np.arange(6)
        nonexclusive = np.array([True, True, True, True, False, False])
        campaigns = np.array([0, 0, 1, 1, 0, 0])
        factors, selected, diagnostics = training.hard_negative_factors(
            scores, rows, nonexclusive, campaigns, fraction=0.5, boost=7.0
        )
        np.testing.assert_array_equal(selected, [1, 3])
        np.testing.assert_allclose(factors, [1, 7, 1, 7, 1, 1])
        self.assertEqual([item["selected"] for item in diagnostics], [1, 1])

    def test_hierarchy_reconstructs_density_ratios(self):
        yields = np.array([2.0, 3.0, 20.0])
        densities = np.array([4.0, 2.0, 1.0])
        exclusive_like = (yields[0] * densities[0] + yields[1] * densities[1]) / (
            yields[0] + yields[1]
        )
        stage_a = np.array([[exclusive_like, densities[2]]])
        stage_a /= stage_a.sum(axis=1, keepdims=True)
        stage_b = np.array([[densities[0], densities[1]]])
        stage_b /= stage_b.sum(axis=1, keepdims=True)
        recovered = hierarchical_probabilities(stage_a, stage_b, yields)
        expected = densities / densities.sum()
        np.testing.assert_allclose(recovered[0], expected)


class MassEstimatorTest(unittest.TestCase):
    def test_proton_feature_names_agree_and_inputs_are_stored(self):
        self.assertEqual(training.PROTON_FEATURE_NAMES, PROTON_FEATURE_NAMES)
        self.assertEqual(
            stored_central_features(["dijet_mass", "yx_minus_dijet_rapidity"]), ["dijet_mass"]
        )
        self.assertEqual(
            stored_central_features(["jet1_mt", "jet1_mass_estimator", "yx_minus_dijet_rapidity"]),
            ["jet1_mt", "jet1_rapidity"],
        )

    def test_estimator_is_twice_the_jet_energy_in_the_proton_frame(self):
        import vector

        pt, eta, phi, mass, y_x = 40.0, 1.2, 0.3, 9.0, 0.7
        jet = vector.obj(pt=pt, eta=eta, phi=phi, mass=mass)
        boosted = jet.boostZ(beta=-np.tanh(y_x))
        estimator = {
            "jet1_mt": np.array([jet.mt]),
            "jet1_rapidity": np.array([jet.rapidity]),
            "dijet_rapidity": np.array([0.5]),
        }
        value = training.mass_estimator_values(estimator, np.array([0]), np.array([y_x - 0.5]))
        self.assertAlmostEqual(float(value[0]), 2.0 * boosted.E, places=9)

    def test_real_rows_use_proton_rapidity_and_pooled_cells_are_rewritten(self):
        central = ["jet1_mt", "jet1_rapidity", "dijet_mass"]
        x = np.array([[41.0, 0.8, 100.0], [30.0, -0.2, 90.0]], dtype=np.float32)
        data = {
            "metadata": {"central_features": central, "features": []},
            "x": x,
            "dijet_rapidity": np.array([0.5, -0.1]),
            "proton_yx": np.array([0.6, np.nan]),  # second row is a pooled event
        }
        features = ["dijet_mass", "yx_minus_dijet_rapidity", "jet1_mass_estimator"]
        matrix, pair_column, estimator = training.read_logical_matrix(data, 1, features)
        self.assertEqual(pair_column, 1)
        self.assertEqual(estimator["column"], 2)
        np.testing.assert_allclose(matrix[:, 1], [0.1, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            matrix[:, 2], [2 * 41.0 * np.cosh(0.8 - 0.6), 2 * 30.0 * np.cosh(-0.2 + 0.1)], rtol=1e-6
        )
        working = np.repeat(matrix[[1]], 3, axis=0)
        delta = np.array([-0.15, 0.0, 0.15])
        training.set_proton_cells(working, pair_column, estimator, np.array([1, 1, 1]), delta)
        np.testing.assert_allclose(working[:, 1], delta, atol=1e-7)
        np.testing.assert_allclose(
            working[:, 2], 2 * 30.0 * np.cosh(-0.2 - (-0.1 + delta)), rtol=1e-6
        )

    def test_estimator_requires_stored_inputs(self):
        data = {
            "metadata": {"central_features": ["dijet_mass"], "features": []},
            "x": np.zeros((1, 1), dtype=np.float32),
            "dijet_rapidity": np.zeros(1),
            "proton_yx": np.zeros(1),
        }
        with self.assertRaisesRegex(RuntimeError, "rebuild the dataset"):
            training.read_logical_matrix(
                data, 1, ["yx_minus_dijet_rapidity", "jet1_mass_estimator"]
            )


if __name__ == "__main__":
    unittest.main()
