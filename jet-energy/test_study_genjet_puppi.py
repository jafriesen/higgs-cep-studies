#!/usr/bin/env python3
"""Focused tests for the stage-1 GenJet--PUPPI study."""

import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("study_genjet_puppi.py")
SPEC = importlib.util.spec_from_file_location("study_genjet_puppi", MODULE_PATH)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def jet(pt, eta, phi, mass=0.0):
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass}


class JetEnergyStage1Test(unittest.TestCase):
    def test_delta_phi_wraps_at_pi(self):
        self.assertAlmostEqual(study.delta_phi(-math.pi + 0.1, math.pi - 0.1), 0.2)
        self.assertAlmostEqual(study.delta_phi(math.pi - 0.1, -math.pi + 0.1), -0.2)

    def test_delta_r_uses_wrapped_phi(self):
        first = jet(10.0, 0.2, -math.pi + 0.05)
        second = jet(10.0, 0.1, math.pi - 0.05)
        self.assertAlmostEqual(study.delta_r(first, second), math.sqrt(0.02))

    def test_matching_is_greedy_and_unique(self):
        gen_jets = [jet(50.0, 0.00, 0.0), jet(40.0, 0.10, 0.0)]
        reco_jets = [jet(45.0, 0.04, 0.0), jet(35.0, 0.18, 0.0)]
        matches = study.match_jets(gen_jets, reco_jets, 0.2)
        self.assertEqual([(gen, reco) for gen, reco, _distance in matches], [(0, 0), (1, 1)])
        self.assertEqual(len({reco for _gen, reco, _distance in matches}), 2)

    def test_event_ranks_by_gen_pt_before_matching(self):
        gen_jets = [jet(30.0, 0.4, 0.4), jet(50.0, -0.2, -0.2)]
        reco_jets = [jet(28.0, 0.41, 0.41), jet(48.0, -0.19, -0.19)]
        result = study.analyze_event(gen_jets, reco_jets, 0.2)
        self.assertEqual(result["leading"]["gen"]["pt"], 50.0)
        self.assertEqual(result["subleading"]["gen"]["pt"], 30.0)
        self.assertIsNotNone(result["dijet"])
        self.assertTrue(all(result["eligible"].values()))

    def test_four_vector_addition_and_dijet_mass(self):
        first = jet(50.0, 0.0, 0.0)
        second = jet(50.0, 0.0, math.pi)
        summed = study.add_four_vectors(
            study.jet_four_vector(first), study.jet_four_vector(second)
        )
        self.assertAlmostEqual(summed[0], 0.0, places=12)
        self.assertAlmostEqual(summed[1], 0.0, places=12)
        self.assertAlmostEqual(study.dijet_kinematics(first, second)["mass"], 100.0)

    def test_dijet_response_uses_combined_four_vectors(self):
        gen = [jet(50.0, 0.0, 0.0), jet(50.0, 0.0, math.pi)]
        reco = [jet(45.0, 0.0, 0.0), jet(45.0, 0.0, math.pi)]
        observation = study.dijet_observation(gen, reco)
        self.assertAlmostEqual(observation["gen"]["mass"], 100.0)
        self.assertAlmostEqual(observation["reco"]["mass"], 90.0)
        self.assertAlmostEqual(observation["response"]["mass_ratio"], 0.9)

    def test_zero_denominator_ratio_is_nonfinite(self):
        self.assertTrue(math.isnan(study.safe_ratio(1.0, 0.0)))
        gen = [jet(20.0, 0.0, 0.0), jet(20.0, 0.0, 0.0)]
        reco = [jet(18.0, 0.0, 0.0), jet(18.0, 0.0, 0.0)]
        observation = study.dijet_observation(gen, reco)
        self.assertTrue(math.isnan(observation["response"]["mass_ratio"]))
        self.assertEqual(study.zero_denominator_entries([observation], "mass_ratio"), 1)


if __name__ == "__main__":
    unittest.main()
