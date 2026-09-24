#!/usr/bin/env python3
"""Focused tests for the JetPUPPI calibration API and study helpers."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

import jet_calibration as calibration
import jet_cache
import study_jet_corrections as study


def node(gen_pt_min, gen_pt_max, raw_pt, target_pt, usable=True, entries=100):
    return {
        "gen_pt_min": gen_pt_min,
        "gen_pt_max": gen_pt_max,
        "raw_reco_pt_median": raw_pt,
        "gen_pt_median": target_pt,
        "target_pt": target_pt,
        "correction_factor": target_pt / raw_pt,
        "correction_ci68": [0.9 * target_pt / raw_pt, 1.1 * target_pt / raw_pt],
        "entries": entries,
        "usable": usable,
        "invalid_reason": None if usable else "insufficient_entries",
    }


def correction_map():
    return {
        "raw_pt_support": [5.0, 40.0],
        "eta_bins": [
            {
                "eta_min": 0.0,
                "eta_max": 1.3,
                "supported": True,
                "nodes": [
                    node(10.0, 20.0, 10.0, 20.0),
                    node(20.0, 40.0, 20.0, 30.0),
                ],
            },
            {
                "eta_min": 1.3,
                "eta_max": 3.0,
                "supported": True,
                "nodes": [
                    node(10.0, 20.0, 10.0, 10.0),
                    node(20.0, 40.0, 30.0, 30.0),
                ],
            },
        ]
    }


def resolution_config():
    return {
        "resolutions": [
            {"eta_min": 0.0, "eta_max": 1.3, "a": 0.1, "b": 10.0},
            {"eta_min": 1.3, "eta_max": 3.0, "a": 0.2, "b": 5.0},
        ]
    }


def hard_flavor_event_dataset():
    record = {
        "gen_pt": np.asarray([30.0, 32.0, 35.0, 20.0]),
        "gen_eta": np.asarray([0.2, -0.3, 0.4, -0.5]),
        "gen_phi": np.asarray([0.0, 2.0, 0.2, 2.2]),
        "gen_mass": np.asarray([4.0, 3.0, 4.0, 3.0]),
        "reco_pt": np.asarray([20.0, 25.0, 18.0]),
        "reco_eta": np.asarray([0.21, -0.29, 0.41]),
        "reco_phi": np.asarray([0.01, 2.01, 0.21]),
        "reco_mass": np.asarray([3.0, 2.5, 2.0]),
        "gen_offsets": np.asarray([0, 2, 4]),
        "reco_offsets": np.asarray([0, 2, 3]),
        "hard_flavor_gen_offsets": np.asarray([0, 2, 4]),
        "hard_flavor_gen_index": np.asarray([0, 1, 0, 1]),
        "hard_flavor_match_offsets": np.asarray([0, 2, 3]),
        "hard_flavor_match_gen_index": np.asarray([0, 1, 0]),
        "hard_flavor_match_reco_index": np.asarray([0, 1, 0]),
        "hard_flavor_match_dr": np.asarray([0.01, 0.02, 0.01]),
    }
    return {"sources": [(record, 2)]}


class JetCalibrationTest(unittest.TestCase):
    def test_inclusive_dataset_uses_all_genjet_puppi_matches(self):
        record = {
            "metadata": {
                "events_read": 2,
                "fsr_state": "FSR",
                "sample": "HardQCD",
                "match_selection": "inclusive",
                "truth_pid_abs": None,
                "closure_mode": "single_jet",
                "eta_max": 3.0,
                "match_dr_max": 0.2,
                "truth_genjet_dr_max": 0.4,
                "source_path": "hardqcd.root",
            },
            "path": "hardqcd.npz",
            "gen_pt": np.asarray([30.0, 25.0, 20.0]),
            "gen_eta": np.asarray([0.1, -0.2, 0.3]),
            "gen_phi": np.asarray([0.0, 1.0, 2.0]),
            "gen_mass": np.asarray([3.0, 2.5, 2.0]),
            "reco_pt": np.asarray([27.0, 22.0, 18.0]),
            "reco_eta": np.asarray([0.1, -0.2, 0.3]),
            "reco_phi": np.asarray([0.0, 1.0, 2.0]),
            "reco_mass": np.asarray([2.7, 2.2, 1.8]),
            "gen_offsets": np.asarray([0, 2, 3]),
            "reco_offsets": np.asarray([0, 2, 3]),
            "match_offsets": np.asarray([0, 2, 3]),
            "match_gen_index": np.asarray([0, 1, 0]),
            "match_reco_index": np.asarray([0, 1, 0]),
            "match_dr": np.asarray([0.01, 0.02, 0.03]),
        }
        config = {
            "fsr_state": "FSR",
            "sample": "HardQCD",
            "match_selection": "inclusive",
            "truth_pid_abs": None,
            "closure_mode": "single_jet",
            "manifest_path": "datasets.yaml",
        }
        with patch.object(jet_cache, "load_cache", return_value=record):
            loaded = study.load_dataset(np, config, [Path("cache.npz")], None)
        np.testing.assert_array_equal(loaded["pairs"]["gen_pt"], [30.0, 25.0, 20.0])
        self.assertEqual(loaded["pair_selection"], "inclusive_hard_interaction_genjets")

    def test_selects_only_unique_hard_process_bottom_genjets(self):
        particles = [
            SimpleNamespace(PID=5, Status=23, IsPU=0, PT=40.0, Eta=0.1, Phi=0.0),
            SimpleNamespace(PID=-5, Status=23, IsPU=0, PT=35.0, Eta=-0.2, Phi=2.0),
            SimpleNamespace(PID=5, Status=51, IsPU=0, PT=30.0, Eta=1.0, Phi=1.0),
            SimpleNamespace(PID=5, Status=23, IsPU=1, PT=25.0, Eta=2.0, Phi=2.0),
        ]
        gen_jets = [
            {"pt": 39.0, "eta": 0.12, "phi": 0.01, "mass": 4.0},
            {"pt": 34.0, "eta": -0.18, "phi": 2.02, "mass": 4.0},
            {"pt": 50.0, "eta": 1.0, "phi": 1.0, "mass": 5.0},
        ]
        self.assertEqual(
            [
                match[1]
                for match in jet_cache.match_jets(
                    jet_cache.hard_process_partons(particles, 5), gen_jets, 0.4
                )
            ],
            [0, 1],
        )

    def test_rejects_ambiguous_hard_bottom_record(self):
        particles = [
            SimpleNamespace(PID=5, Status=23, IsPU=0, PT=40.0, Eta=0.1, Phi=0.0),
            SimpleNamespace(PID=5, Status=23, IsPU=0, PT=30.0, Eta=0.2, Phi=0.1),
            SimpleNamespace(PID=-5, Status=23, IsPU=0, PT=35.0, Eta=-0.2, Phi=2.0),
        ]
        self.assertEqual(jet_cache.hard_process_partons(particles, 5), [])

    def test_interpolates_corrected_pt_between_nodes(self):
        factors, valid = calibration.correction_factors(
            np.asarray([10.0, 15.0, 20.0]),
            np.asarray([0.2, 0.2, 0.2]),
            correction_map(),
        )
        np.testing.assert_array_equal(valid, [True, True, True])
        np.testing.assert_allclose(factors * [10.0, 15.0, 20.0], [20.0, 25.0, 30.0])

    def test_rejects_pt_and_eta_outside_support(self):
        factors, valid = calibration.correction_factors(
            np.asarray([4.9, 40.0, 15.0, 15.0]),
            np.asarray([0.2, 0.2, 3.0, 1.3]),
            correction_map(),
        )
        np.testing.assert_array_equal(valid, [False, False, False, True])
        self.assertTrue(np.all(np.isnan(factors[:3])))
        self.assertAlmostEqual(factors[3], 1.0)

    def test_endpoint_factors_extend_to_full_support(self):
        terminal_map = correction_map()
        factors, valid = calibration.correction_factors(
            [5.0, 10.0, 20.0, 39.9, 40.0], [0.2] * 5, terminal_map
        )
        np.testing.assert_array_equal(valid, [True, True, True, True, False])
        np.testing.assert_allclose(factors[:2], [2.0, 2.0])
        np.testing.assert_allclose(factors[2:4], [1.5, 1.5])
        self.assertTrue(math.isnan(factors[4]))

    def test_corrected_pt_is_continuous_and_strictly_increasing(self):
        raw = np.linspace(5.0, 39.999, 10000)
        factors, valid = calibration.correction_factors(
            raw, np.full(raw.size, 0.2), correction_map()
        )
        self.assertTrue(np.all(valid))
        self.assertTrue(np.all(np.diff(raw * factors) > 0.0))
        for boundary in (10.0, 20.0):
            nearby = np.asarray([boundary - 1e-8, boundary, boundary + 1e-8])
            local, _valid = calibration.correction_factors(
                nearby, np.full(3, 0.2), correction_map()
            )
            self.assertLess(float(np.max(np.diff(nearby * local))), 1e-6)

    def test_scales_pt_and_mass_but_preserves_angles(self):
        corrected = calibration.correct_jet_kinematics(
            [10.0], [0.2], [1.1], [4.0], correction_map()
        )
        self.assertAlmostEqual(corrected["pt"][0], 20.0)
        self.assertAlmostEqual(corrected["mass"][0], 8.0)
        self.assertAlmostEqual(corrected["eta"][0], 0.2)
        self.assertAlmostEqual(corrected["phi"][0], 1.1)

    def test_loads_requested_yaml_map(self):
        document = {
            "schema_version": calibration.SCHEMA_VERSION,
            "maps": {"FSR": {"Hbb": correction_map()}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrections.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            loaded = calibration.load_correction_map(path, "FSR", "Hbb")
        self.assertEqual(loaded["eta_bins"][0]["eta_max"], 1.3)

    def test_rejects_old_correction_schema(self):
        document = {"schema_version": 2, "maps": {"FSR": {"Hbb": correction_map()}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrections.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported correction schema"):
                calibration.load_correction_map(path, "FSR", "Hbb")

    def test_l1_formula_and_eta_boundary(self):
        sigma, valid = calibration.l1_relative_sigma(
            [20.0, 20.0, 20.0], [0.2, 1.3, 3.0], resolution_config()
        )
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(sigma[:2], [0.6, 0.45])
        self.assertTrue(math.isnan(sigma[2]))

    def test_log_normal_smearing_is_deterministic_and_has_requested_variance(self):
        first = calibration.smear_gen_jets(
            [50.0],
            [5.0],
            [0.2],
            np.random.default_rng(123),
            resolution_config(),
            replicas=100000,
        )
        second = calibration.smear_gen_jets(
            [50.0],
            [5.0],
            [0.2],
            np.random.default_rng(123),
            resolution_config(),
            replicas=100000,
        )
        np.testing.assert_array_equal(first["factor"], second["factor"])
        factors = first["factor"][:, 0]
        expected_sigma = 0.1 + 10.0 / 50.0
        self.assertAlmostEqual(float(np.mean(factors)), 1.0, delta=0.01)
        self.assertAlmostEqual(
            float(np.std(factors) / np.mean(factors)), expected_sigma, delta=0.01
        )
        np.testing.assert_allclose(first["mass"][:, 0], factors * 5.0)

    def test_derivation_inverts_synthetic_half_response_nodes(self):
        gen_pt = np.asarray([42.0, 46.0, 55.0, 60.0])
        pairs = {
            "gen_pt": gen_pt,
            "gen_eta": np.full(gen_pt.size, 0.2),
            "reco_pt": 0.5 * gen_pt,
            "reco_eta": np.full(gen_pt.size, 0.2),
        }
        dataset = {
            "fsr_state": "FSR",
            "sample": "Hbb",
            "files": ["derive.root"],
            "events_read": 6,
            "pair_selection": "hard_process_b_matched_genjets",
            "match_selection": "hard_flavor",
            "truth_pid_abs": 5,
            "closure_mode": "dijet",
            "selection_pair_events": 4,
            "selected_genjets": 4,
            "matched_puppijets": 4,
            "complete_genjet_pair_events": 2,
            "complete_puppi_pair_events": 2,
            "truth_genjet_dr_max": 0.4,
            "pairs": pairs,
        }
        args = SimpleNamespace(
            min_bin_entries=2, bootstrap_replicas=20, match_dr_max=0.2
        )
        derived = study.derive_correction_map(
            np, dataset, args, np.random.default_rng(7)
        )
        factors, valid = calibration.correction_factors(
            np.asarray([22.0, 28.75]), np.asarray([0.2, 0.2]), derived
        )
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(
            factors * np.asarray([22.0, 28.75]), [44.0, 57.5]
        )

    def test_nonmonotonic_nodes_discard_lower_occupancy_then_higher_on_tie(self):
        nodes = [
            node(5.0, 10.0, 8.0, 7.0, entries=80),
            node(10.0, 20.0, 7.0, 15.0, entries=20),
            node(20.0, 30.0, 18.0, 25.0, entries=50),
        ]
        study.discard_nonmonotonic_nodes(nodes)
        self.assertTrue(nodes[0]["usable"])
        self.assertFalse(nodes[1]["usable"])
        self.assertEqual(nodes[1]["invalid_reason"], "nonmonotonic_raw_median")
        self.assertTrue(nodes[2]["usable"])

        tied = [
            node(5.0, 10.0, 8.0, 7.0, entries=50),
            node(10.0, 20.0, 7.0, 15.0, entries=50),
            node(20.0, 30.0, 18.0, 25.0, entries=50),
        ]
        study.discard_nonmonotonic_nodes(tied)
        self.assertTrue(tied[0]["usable"])
        self.assertFalse(tied[1]["usable"])

    def test_eta_bin_requires_two_monotonic_nodes(self):
        nodes = [
            node(5.0, 10.0, 8.0, 7.0),
            node(10.0, 20.0, 7.0, 15.0, entries=20),
        ]
        study.discard_nonmonotonic_nodes(nodes)
        self.assertFalse(any(item["usable"] for item in nodes))
        self.assertIn(
            "insufficient_monotonic_nodes",
            {item["invalid_reason"] for item in nodes},
        )

    def test_bootstrap_is_reproducible(self):
        values = np.arange(1.0, 21.0)
        first = calibration.bootstrap_interval(
            values, np.median, np.random.default_rng(99), replicas=50
        )
        second = calibration.bootstrap_interval(
            values, np.median, np.random.default_rng(99), replicas=50
        )
        self.assertEqual(first, second)

    def test_validation_metrics_include_reproducible_confidence_intervals(self):
        values = np.linspace(0.8, 1.2, 101)
        first = calibration.response_metrics_with_ci(
            values, np.random.default_rng(99), replicas=50, include_ci=True
        )
        second = calibration.response_metrics_with_ci(
            values, np.random.default_rng(99), replicas=50, include_ci=True
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["closure_bias"], first["median"] - 1.0)
        self.assertEqual(
            first["closure_bias_ci68"],
            [value - 1.0 for value in first["median_ci68"]],
        )
        self.assertTrue(all(value is not None for value in first["rms_over_mean_ci68"]))

    def test_raw_pt_closure_uses_complete_calibration_bins(self):
        pairs = {
            "gen_pt": np.asarray([26.0, 26.0]),
            "reco_pt": np.asarray([16.0, 16.0]),
            "reco_eta": np.asarray([0.2, 0.2]),
        }
        rows = study.evaluate_raw_pt_response_bins(
            np,
            pairs,
            correction_map(),
            SimpleNamespace(bootstrap_replicas=10),
            np.random.default_rng(7),
        )
        first = rows[1]
        self.assertEqual(
            (first["raw_pt_min"], first["raw_pt_max"]),
            tuple(study.CALIBRATION_GEN_PT_EDGES[1:3]),
        )
        self.assertEqual(first["matched_jets"], 2)
        self.assertAlmostEqual(first["raw"]["median"], 16.0 / 26.0)
        self.assertAlmostEqual(first["corrected"]["median"], 1.0)

    def test_correction_can_reorder_jets_across_eta_bins(self):
        jets = [
            {"pt": 20.0, "eta": 0.2, "phi": 0.0, "mass": 2.0},
            {"pt": 25.0, "eta": 1.4, "phi": 1.0, "mass": 3.0},
        ]
        corrected, raw_supported, unsupported = study.corrected_event_jets(
            np, jets, correction_map()
        )
        selected = study.selected_pair(corrected, eta_max=3.0, pt_min=15.0)
        self.assertEqual(unsupported, 0)
        self.assertEqual(raw_supported, jets)
        self.assertAlmostEqual(selected[0]["pt"], 30.0)
        self.assertAlmostEqual(selected[1]["pt"], 25.0)

    def test_event_closure_uses_hard_flavor_genjet_pair_denominator(self):
        summary, profiles, residuals = study.event_closure(
            np,
            hard_flavor_event_dataset(),
            correction_map(),
            resolution_config(),
            SimpleNamespace(smear_replicas=1),
            np.random.default_rng(9),
        )
        self.assertEqual(summary["eligible_hard_flavor_genjet_pair_events"], 2)
        self.assertEqual(summary["complete_hard_flavor_puppi_pair_events"], 1)
        self.assertEqual(summary["fully_correctable_pair_events"], 1)
        self.assertEqual(profiles["gen"]["trials"], 2)
        self.assertEqual(profiles["raw"]["trials"], 2)
        self.assertEqual(len(profiles["raw"]["leading_pt"]), 1)
        self.assertEqual(summary["closure_selected_pair_events"], 1)
        self.assertEqual(len(residuals["raw"]["mass_ratio"]), 1)
        self.assertEqual(len(residuals["corrected"]["mass_ratio"]), 1)

    def test_event_closure_residuals_use_generator_side_selection(self):
        dataset = hard_flavor_event_dataset()
        dataset["sources"][0][0]["gen_pt"] = np.asarray([30.0, 20.0, 35.0, 20.0])
        summary, _profiles, residuals = study.event_closure(
            np,
            dataset,
            correction_map(),
            resolution_config(),
            SimpleNamespace(smear_replicas=1),
            np.random.default_rng(9),
        )
        self.assertEqual(summary["eligible_hard_flavor_genjet_pair_events"], 2)
        self.assertEqual(summary["closure_selected_pair_events"], 0)
        self.assertEqual(len(residuals["raw"]["mass_ratio"]), 0)
        self.assertEqual(len(residuals["corrected"]["mass_ratio"]), 0)
        self.assertEqual(len(residuals["smeared"]["mass_ratio"]), 0)

    def test_jet_activity_uses_all_selected_jets_for_ht(self):
        profiles = study.empty_jet_activity_profiles()
        jets = [
            {"pt": 50.0, "eta": 0.2, "phi": 0.0, "mass": 2.0},
            {"pt": 30.0, "eta": -0.3, "phi": 1.0, "mass": 2.0},
            {"pt": 25.0, "eta": 0.4, "phi": 2.0, "mass": 2.0},
            {"pt": 20.0, "eta": 0.5, "phi": 2.5, "mass": 2.0},
        ]
        study.append_jet_activity_observables(
            profiles, "raw", jets, [jets[1], jets[2]]
        )

        matched = profiles["truth_matched"]["raw"]
        self.assertEqual(matched["leading_pt"], [30.0])
        self.assertEqual(matched["subleading_pt"], [25.0])
        self.assertEqual(matched["all_pt"], [30.0, 25.0])
        self.assertAlmostEqual(matched["dijet_ht_fraction"][0], 55.0 / 105.0)

        hardest = profiles["two_hardest"]["raw"]
        self.assertEqual(hardest["leading_pt"], [50.0])
        self.assertEqual(hardest["subleading_pt"], [30.0])
        self.assertEqual(hardest["all_pt"], [50.0, 30.0, 25.0])
        self.assertAlmostEqual(hardest["dijet_ht_fraction"][0], 80.0 / 105.0)

    def test_jet_activity_tracks_truth_matches_after_correction(self):
        summary, profiles = study.jet_activity_profiles(
            np,
            hard_flavor_event_dataset(),
            correction_map(),
            resolution_config(),
            SimpleNamespace(smear_replicas=1),
            np.random.default_rng(11),
        )
        self.assertEqual(summary["jet_pt_min"], 20.0)
        self.assertEqual(summary["abs_eta_max"], 2.4)
        self.assertEqual(profiles["truth_matched"]["raw"]["leading_pt"], [])
        np.testing.assert_allclose(
            profiles["truth_matched"]["corrected"]["leading_pt"], [37.5]
        )
        np.testing.assert_allclose(
            profiles["truth_matched"]["corrected"]["subleading_pt"], [30.0]
        )

    def test_selected_pair_respects_eta_acceptance(self):
        jets = [
            {"pt": 30.0, "eta": 2.6, "phi": 0.0, "mass": 2.0},
            {"pt": 25.0, "eta": -2.7, "phi": 1.0, "mass": 2.0},
        ]
        self.assertIsNone(study.selected_pair(jets, eta_max=2.4))
        self.assertEqual(len(study.selected_pair(jets, eta_max=3.0)), 2)

    def test_hardqcd_source_or_target_skips_dijet_closure(self):
        dijet = {"closure_mode": "dijet"}
        single = {"closure_mode": "single_jet"}
        self.assertTrue(study.should_run_dijet_closure(dijet, dijet))
        self.assertFalse(study.should_run_dijet_closure(single, dijet))
        self.assertFalse(study.should_run_dijet_closure(dijet, single))


if __name__ == "__main__":
    unittest.main()
