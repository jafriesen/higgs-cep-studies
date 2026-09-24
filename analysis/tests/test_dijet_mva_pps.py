import tempfile
import unittest
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import vector

from analysis.MVA import run_dijet_mva as mva
from analysis.MVA import plot_dijet_mva as mva_plot
from analysis.MVA import scan_pp_mass_significance as scan


class DijetMvaPpsTest(unittest.TestCase):
    def setUp(self):
        self.pps = {
            "sqrt_s": 14000.0,
            "xi_ranges": [("test", 0.005, 0.02)],
            "xi_res": 0.0,
        }

    def test_combinatorial_acceptance_factor_is_explicit(self):
        combinatorial = next(
            spec for spec in mva.SAMPLE_SPECS if spec["name"] == "QCDbb_madgraph_comb"
        )
        self.assertEqual(mva.COMBINATORIAL_ACCEPTANCE_FACTOR, 0.005)
        self.assertEqual(
            combinatorial["acceptance_factor"], mva.COMBINATORIAL_ACCEPTANCE_FACTOR
        )

    def test_pps_boundaries_and_pair_observables(self):
        xi = np.asarray([0.0049, 0.005, 0.019999, 0.02])
        np.testing.assert_array_equal(
            mva.passes_pps(np, xi, self.pps["xi_ranges"]),
            [False, True, True, False],
        )
        mass, rapidity = mva.pair_observables(
            np, np.asarray([0.008]), np.asarray([0.01]), self.pps["sqrt_s"]
        )
        self.assertAlmostEqual(mass[0], np.sqrt(0.008 * 0.01) * 14000.0)
        self.assertAlmostEqual(rapidity[0], 0.5 * np.log(0.01 / 0.008))

    def test_superchic_selection_and_signed_features(self):
        central = {
            "xi_left_truth": np.asarray([0.008, 0.003]),
            "xi_right_truth": np.asarray([0.01, 0.01]),
            "features": np.zeros((2, len(mva.FEATURE_NAMES) - 2)),
            "dijet_mass": np.asarray([120.0, 120.0]),
            "dijet_rapidity": np.asarray([0.2, 0.2]),
        }
        pairs = mva.select_superchic_pairs(
            np, central, self.pps, np.random.default_rng(7)
        )
        np.testing.assert_array_equal(pairs["selected"], [True, False])
        features = mva.append_proton_features(np, central, pairs)
        self.assertEqual(features.shape, (1, len(mva.FEATURE_NAMES)))
        self.assertAlmostEqual(features[0, -2], pairs["mx"][0] - 120.0)
        self.assertAlmostEqual(features[0, -1], pairs["yx"][0] - 0.2)

    def test_pre_mva_cuts_are_strict_and_preserve_alignment(self):
        features = np.zeros((3, len(mva.FEATURE_NAMES) - 2))
        features[:, mva.BASE_FEATURE_NAMES.index("delta_phi_jj")] = [3.1, 3.0, 3.2]
        features[
            :, mva.BASE_FEATURE_NAMES.index("track_multiplicity_r_gt_0p4_pt1")
        ] = [4, 0, 5]
        central = {
            "features": features,
            "dijet_mass": np.asarray([50.0, 100.0, 150.0]),
            "dijet_rapidity": np.zeros(3),
        }
        pairs = {
            "selected": np.asarray([True, True, True]),
            "xi_left": np.asarray([0.01, 0.02, 0.03]),
            "xi_right": np.asarray([0.011, 0.021, 0.031]),
            "mx": np.asarray([120.0, 121.0, 122.0]),
            "yx": np.asarray([0.14, 0.0, 0.10]),
            "bx_id": np.asarray([1, 2, 3]),
            "proton_source": np.asarray(["a", "b", "c"]),
        }
        filtered, cutflow = mva.apply_pre_mva_cuts(np, central, pairs)
        np.testing.assert_array_equal(filtered["selected"], [True, False, False])
        np.testing.assert_array_equal(filtered["bx_id"], [1])
        self.assertEqual(
            cutflow,
            {
                "n_mass_window": 3,
                "n_delta_phi": 2,
                "n_rapidity": 2,
                "n_track_multiplicity": 1,
                "n_dijet_mass": 1,
                "n_pre_mva": 1,
            },
        )

    def test_pre_mva_dijet_mass_boundaries_are_inclusive(self):
        features = np.zeros((3, len(mva.FEATURE_NAMES) - 2))
        features[:, mva.BASE_FEATURE_NAMES.index("delta_phi_jj")] = 3.1
        central = {
            "features": features,
            "dijet_mass": np.asarray([50.0, 150.0, 150.01]),
            "dijet_rapidity": np.zeros(3),
        }
        pairs = {
            "selected": np.asarray([True, True, True]),
            "xi_left": np.asarray([0.01, 0.02, 0.03]),
            "xi_right": np.asarray([0.011, 0.021, 0.031]),
            "mx": np.asarray([120.0, 121.0, 122.0]),
            "yx": np.zeros(3),
            "bx_id": np.asarray([1, 2, 3]),
            "proton_source": np.asarray(["a", "b", "c"]),
        }
        filtered, cutflow = mva.apply_pre_mva_cuts(np, central, pairs)
        np.testing.assert_array_equal(filtered["selected"], [True, True, False])
        self.assertEqual(cutflow["n_dijet_mass"], 2)

    def test_scan_score_cut_uses_only_actual_passing_events(self):
        data = {
            "label": np.asarray([1, 0, 0]),
            "weight": np.asarray([5.0, 2.0, 6.0]),
            "score": np.asarray([0.9, 0.8, 0.2]),
            "mass": np.asarray([120.0, 120.0, 122.0]),
        }
        result = scan.scan_significance(
            np,
            data,
            np.asarray([119.0, 121.0, 123.0]),
            np.asarray([0.5]),
        )
        self.assertAlmostEqual(result[0]["background_yield"], 2.0)
        self.assertAlmostEqual(result[0]["signal_yield"], 5.0)
        self.assertEqual(result[0]["background_events"], 1)
        self.assertEqual(result[0]["signal_events"], 1)
        self.assertAlmostEqual(result[0]["significance"], 5.0 / np.sqrt(7.0))

    def test_mva_binned_significance_matches_mass_bin_formula(self):
        values = np.asarray([0.2, 0.2, 0.8])
        labels = np.asarray([1, 0, 0])
        weights = np.asarray([5.0, 2.0, 6.0])
        significance = mva_plot.binned_significance(
            np, values, labels, weights, np.asarray([0.0, 0.5, 1.0])
        )
        self.assertAlmostEqual(significance, 5.0 / np.sqrt(7.0))

    def test_scan_channel_weights_do_not_change_rows_or_scores(self):
        data = {
            "process": np.asarray(["Hbb", "QCDbb", "QCDbb_madgraph_comb"]),
            "weight": np.asarray([1.0, 2.0, 3.0]),
            "score": np.asarray([0.9, 0.4, 0.2]),
        }
        factors = scan.parse_channel_weights(
            ["QCDbb=10", "QCDbb_madgraph_comb=0.5"]
        )
        weighted, summary = scan.apply_channel_weights(np, data, factors)
        np.testing.assert_allclose(weighted["weight"], [1.0, 20.0, 1.5])
        np.testing.assert_array_equal(weighted["score"], data["score"])
        self.assertEqual(len(weighted["weight"]), 3)
        self.assertEqual(summary["Hbb"]["factor"], 1.0)

    def test_scan_channel_weight_validation(self):
        with self.assertRaises(RuntimeError):
            scan.parse_channel_weights(["unknown=2"])
        with self.assertRaises(RuntimeError):
            scan.parse_channel_weights(["Hbb=-1"])

    def test_hepmc_protons_follow_selected_event_indices(self):
        contents = """HepMC::Version 3.03.01
E 1 0 0
P 1 0 2212 0 0 -6943 6944 0.938 1
P 2 0 2212 0 0 6929 6930 0.938 1
E 2 0 0
P 1 0 2212 0 0 -6999 7000 0.938 4
P 2 0 2212 0 0 -6860 6861 0.938 1
P 3 0 2212 0 0 6789 6790 0.938 1
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.hepmc"
            path.write_text(contents)
            left, right = mva.parse_hepmc_proton_xi(
                np, path, np.asarray([1]), self.pps["sqrt_s"]
            )
        np.testing.assert_allclose(left, [(7000.0 - 6861.0) / 7000.0])
        np.testing.assert_allclose(right, [(7000.0 - 6790.0) / 7000.0])

    def test_seeded_bx_assignment_uses_each_bx_before_reuse(self):
        first = mva.shuffled_bx_assignments(
            np, 8, 3, np.random.default_rng(12345)
        )
        second = mva.shuffled_bx_assignments(
            np, 8, 3, np.random.default_rng(12345)
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first[:3]), {0, 1, 2})
        self.assertEqual(set(first[3:6]), {0, 1, 2})

    def test_minbias_pair_pool_and_random_choice_are_deterministic(self):
        bunch_crossings = ak.Array(
            [
                {
                    "bx_id": 0,
                    "protons": [
                        {"side": -1, "xi": 0.008},
                        {"side": -1, "xi": 0.009},
                        {"side": 1, "xi": 0.01},
                    ],
                },
                {
                    "bx_id": 1,
                    "protons": [{"side": -1, "xi": 0.008}],
                },
            ]
        )
        pool, has_pair = mva.build_minbias_pair_pool(
            ak, np, bunch_crossings, self.pps, np.random.default_rng(11), chunk_size=1
        )
        np.testing.assert_array_equal(has_pair, [True, False])
        self.assertEqual(ak.to_list(ak.num(pool, axis=1)), [2, 0])
        choice_a = mva.choose_minbias_pairs(
            ak, np, pool, np.asarray([0, 1, 0]), np.random.default_rng(9)
        )
        choice_b = mva.choose_minbias_pairs(
            ak, np, pool, np.asarray([0, 1, 0]), np.random.default_rng(9)
        )
        np.testing.assert_array_equal(choice_a["selected"], [True, False, True])
        np.testing.assert_allclose(choice_a["mx"], choice_b["mx"])
        self.assertTrue(np.all((choice_a["mx"] >= 117.0) & (choice_a["mx"] <= 133.0)))

    def test_preselected_minbias_bx_assignment_is_fully_efficient(self):
        pair_pool = ak.Array(
            [
                [{"xi_left": 0.008, "xi_right": 0.01, "mx": 125.2, "yx": 0.1}],
                [],
            ]
        )
        has_pair = np.asarray([True, False])
        first, acceptance = mva.assign_qualifying_minbias_pairs(
            ak,
            np,
            pair_pool,
            has_pair,
            6,
            np.asarray([0, 2, 5]),
            np.random.default_rng(17),
        )
        second, _ = mva.assign_qualifying_minbias_pairs(
            ak,
            np,
            pair_pool,
            has_pair,
            6,
            np.asarray([0, 2, 5]),
            np.random.default_rng(17),
        )
        self.assertAlmostEqual(acceptance, 0.5)
        np.testing.assert_array_equal(first["selected"], [True, True, True])
        np.testing.assert_array_equal(first["bx_id"], [0, 0, 0])
        np.testing.assert_allclose(first["mx"], second["mx"])

    def test_dataset_keeps_physical_weights_and_cached_protons(self):
        sample = {
            "features": np.zeros((2, len(mva.FEATURE_NAMES)), dtype=np.float64),
            "label": 0,
            "event_weight": 2.5,
            "name": "QCDbb_madgraph_comb",
            "campaign": "test",
            "xi_left": np.asarray([0.008, 0.009]),
            "xi_right": np.asarray([0.01, 0.01]),
            "mx": np.asarray([125.2, 132.8]),
            "yx": np.asarray([0.1, 0.05]),
            "dijet_mass": np.asarray([120.0, 121.0]),
            "dijet_rapidity": np.asarray([0.0, 0.1]),
            "delta_phi_jj": np.asarray([3.05, 2.9]),
            "bx_id": np.asarray([2, 3]),
            "proton_source": np.asarray(["minbias_bx", "minbias_bx"]),
        }
        dataset = mva.build_dataset(np, [sample])
        np.testing.assert_allclose(dataset["physical_weight"], [2.5, 2.5])
        cached = scan.cached_data(
            np,
            dataset,
            {"all": np.asarray([0.2, 0.7])},
            (117.0, 133.0),
        )
        np.testing.assert_allclose(cached["weight"], [2.5, 2.5])
        np.testing.assert_allclose(cached["mass"], sample["mx"])
        np.testing.assert_array_equal(cached["proton_source"], sample["proton_source"])

    def test_feature_set_variants(self):
        base = {
            "x": np.arange(2 * len(mva.BASE_FEATURE_NAMES), dtype=np.float64).reshape(
                2, len(mva.BASE_FEATURE_NAMES)
            ),
            "dijet_mass": np.asarray([100.0, 200.0]),
        }
        jet1_index = mva.BASE_FEATURE_NAMES.index("jet1_pt_over_mjj")
        for feature_set, expected_count in (
            ("current", 14),
            ("no_dijet_mass", 13),
            ("raw_pt_with_dijet_mass", 14),
            ("raw_pt_no_dijet_mass", 13),
        ):
            dataset = {name: value.copy() for name, value in base.items()}
            transformed = mva.transform_feature_set(
                np, dataset, mva.BASE_FEATURE_NAMES, feature_set
            )
            self.assertEqual(transformed["x"].shape[1], expected_count)
            names = list(transformed["feature_names"])
            if feature_set.startswith("raw_pt"):
                self.assertIn("jet1_pt", names)
                self.assertNotIn("jet1_pt_over_mjj", names)
                np.testing.assert_allclose(
                    transformed["x"][:, jet1_index],
                    base["x"][:, jet1_index] * base["dijet_mass"],
                )
            if "no_dijet_mass" in feature_set:
                self.assertNotIn("dijet_mass", names)

    def test_delphes_fixture_keeps_event_and_proton_alignment(self):
        vector.register_awkward()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.root"
            with uproot.recreate(path) as root_file:
                root_file["Delphes"] = {
                    "Jet/Jet.PT": ak.Array([[70.0, 55.0], [12.0]]),
                    "Jet/Jet.Eta": ak.Array([[0.2, -0.2], [0.0]]),
                    "Jet/Jet.Phi": ak.Array([[0.0, np.pi], [0.0]]),
                    "Jet/Jet.Mass": ak.Array([[5.0, 5.0], [1.0]]),
                    "EFlowTrack/EFlowTrack.PT": ak.Array(
                        [[2.0, 1.0, 0.5], [10.0]]
                    ),
                    "EFlowTrack/EFlowTrack.Eta": ak.Array(
                        [[0.2, 1.0, 1.0], [3.0]]
                    ),
                    "EFlowTrack/EFlowTrack.Phi": ak.Array(
                        [[0.0, 1.0, 1.0], [3.0]]
                    ),
                    "Particle/Particle.PID": ak.Array([[2212, 2212], [2212, 2212]]),
                    "Particle/Particle.Status": ak.Array([[1, 1], [1, 1]]),
                    "Particle/Particle.Pz": ak.Array([[-6900.0, 6900.0], [-6900.0, 6900.0]]),
                    "Particle/Particle.E": ak.Array([[6944.0, 6930.0], [6944.0, 6930.0]]),
                }
            central = mva.load_file_central(
                ak, np, uproot, path, "Delphes", "Jet", with_protons=True
            )
            self.assertEqual(central["n_generated"], 2)
            self.assertEqual(central["n_two_jet"], 1)
            self.assertEqual(central["n_finite"], 1)
            feature_index = mva.BASE_FEATURE_NAMES.index(
                "track_multiplicity_r_gt_0p4_pt1"
            )
            self.assertEqual(central["features"][0, feature_index], 1.0)
            np.testing.assert_allclose(central["left_energy"], [6944.0])
            np.testing.assert_allclose(central["right_energy"], [6930.0])


if __name__ == "__main__":
    unittest.main()
