import math
import unittest

from trigger.jet_response import dijet_rapidity, match_jets


class JetResponseTest(unittest.TestCase):
    def test_matching_is_unique_and_uses_delta_r(self):
        gen = [
            {"pt": 30.0, "eta": 0.0, "phi": 0.0, "mass": 0.0},
            {"pt": 20.0, "eta": 1.0, "phi": 1.0, "mass": 0.0},
        ]
        reco = [
            {"pt": 29.0, "eta": 0.05, "phi": 0.02, "mass": 0.0},
            {"pt": 21.0, "eta": 1.3, "phi": 1.0, "mass": 0.0},
        ]
        self.assertEqual([(a, b) for a, b, _dr in match_jets(gen, reco, 0.2)], [(0, 0)])

    def test_dijet_rapidity_uses_summed_four_vector(self):
        first = {"pt": 25.0, "eta": 0.6, "phi": 0.0, "mass": 0.0}
        second = {"pt": 25.0, "eta": 0.6, "phi": math.pi, "mass": 0.0}
        self.assertAlmostEqual(dijet_rapidity(first, second), 0.6)


if __name__ == "__main__":
    unittest.main()
