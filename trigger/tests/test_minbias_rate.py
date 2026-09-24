from contextlib import redirect_stdout
import io
import unittest

import numpy as np

from minbias.flux import Acceptance, ProtonFlux
from trigger.minbias_rate import _count_pair_categories, study_trigger_rates


class TriggerRateTest(unittest.TestCase):
    def setUp(self):
        self.flux = ProtonFlux(
            event=np.array([0, 0, 1], dtype=np.int32),
            arm=np.array([-1, 1, -1], dtype=np.int8),
            xi=np.array([0.01, 0.01, 0.01], dtype=np.float32),
            px=np.zeros(3),
            py=np.zeros(3),
            process=np.ones(3, dtype=np.int16),
            metadata={"n_inelastic_generated": 4, "sqrt_s_gev": 10000.0},
        )
        self.acceptance = Acceptance([(0.005, 0.02)])

    def test_pair_categories_use_reconstructed_values_and_same_interaction(self):
        counts, same = _count_pair_categories(
            xi=np.array([0.01, 0.01, 0.04]),
            arm=np.array([-1, 1, 1]),
            arrival=np.array([0.2, 0.0, 0.4]),
            interaction=np.array([0, 0, 1]),
            sqrt_s_gev=10000.0,
            mass_range=(90.0, 110.0),
            pv_z=0.0,
            pv_u=0.0,
            z_cut=0.15,
            luminous_u_cut=0.2,
            central_u_cut=0.15,
        )
        self.assertEqual(counts["pps"], 2)
        self.assertEqual(counts["pps_mass"], 1)
        self.assertEqual(counts["pps_z"], 2)
        self.assertEqual(counts["pps_z_luminous"], 1)
        self.assertEqual(counts["pps_z_central"], 1)
        self.assertEqual(same, {"pps": 1, "pps_mass": 1})

    def test_poisson_thinning_same_pairs_and_repeatability(self):
        arguments = dict(
            mu=4.0,
            n_bx=2000,
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            seed=19,
            batch_size=2000,
        )
        first = study_trigger_rates(self.flux, self.acceptance, **arguments)
        second = study_trigger_rates(self.flux, self.acceptance, **arguments)
        self.assertEqual(first, second)
        self.assertEqual(first["sampling"]["truth_pps_interactions"], 2)
        self.assertEqual(first["sampling"]["truth_pps_interaction_probability"], 0.5)
        self.assertGreater(
            first["same_interaction_diagnostics"]["mean_pps_mass_pairs_per_bx"], 0.0
        )

    def test_truth_acceptance_is_applied_before_smearing(self):
        flux = ProtonFlux(
            event=np.array([0, 0], dtype=np.int32),
            arm=np.array([-1, 1], dtype=np.int8),
            xi=np.array([0.004, 0.004], dtype=np.float32),
            px=np.zeros(2),
            py=np.zeros(2),
            process=np.ones(2, dtype=np.int16),
            metadata={"n_inelastic_generated": 100, "sqrt_s_gev": 1000.0},
        )
        result = study_trigger_rates(
            flux,
            self.acceptance,
            mu=200.0,
            n_bx=100,
            xi_resolution=1.0,
            seed=2,
            batch_size=100,
        )
        self.assertEqual(result["sampling"]["truth_pps_interactions"], 0)
        self.assertEqual(result["criteria"]["pps"]["mean_pairs_per_bx"], 0.0)

    def test_progress_reports_stages_and_eta(self):
        output = io.StringIO()
        with redirect_stdout(output):
            study_trigger_rates(
                self.flux,
                self.acceptance,
                mu=4.0,
                n_bx=20,
                xi_resolution=0.0,
                seed=3,
                batch_size=10,
                progress=True,
            )
        text = output.getvalue()
        self.assertIn("Preparing truth-PPS interaction groups", text)
        self.assertIn("BX 20/20", text)
        self.assertIn("ETA", text)
        self.assertIn("Finalizing probabilities", text)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            study_trigger_rates(self.flux, self.acceptance, n_bx=0)
        with self.assertRaises(ValueError):
            study_trigger_rates(self.flux, self.acceptance, mass_range=(133.0, 117.0))
        with self.assertRaises(ValueError):
            study_trigger_rates(self.flux, self.acceptance, xi_resolution=-1.0)


if __name__ == "__main__":
    unittest.main()
