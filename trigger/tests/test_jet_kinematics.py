import unittest

import numpy as np

from trigger.jet_kinematics import distribution_summary, leading_subleading_pt


class JetKinematicsTest(unittest.TestCase):
    def test_selects_any_two_jets_without_pt_cut_and_orders_them(self):
        self.assertEqual(
            leading_subleading_pt([6.0, 20.0, 10.0], [0.1, -1.0, 2.0], 2.4),
            (20.0, 10.0),
        )
        self.assertEqual(
            leading_subleading_pt([7.0, 5.2], [0.2, -0.3], 2.4),
            (7.0, 5.2),
        )
        self.assertIsNone(leading_subleading_pt([5.1], [0.0], 2.4))

    def test_eta_acceptance_is_applied_before_two_jet_requirement(self):
        self.assertEqual(
            leading_subleading_pt([30.0, 20.0, 10.0], [2.5, 0.5, -0.5], 2.4),
            (20.0, 10.0),
        )

    def test_distribution_summary(self):
        summary = distribution_summary(np.array([5.0, 10.0, 15.0]))
        self.assertEqual(summary["entries"], 3)
        self.assertEqual(summary["minimum"], 5.0)
        self.assertEqual(summary["maximum"], 15.0)
        self.assertEqual(summary["quantiles"]["0.50"], 10.0)


if __name__ == "__main__":
    unittest.main()
