import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minbias.flux import Acceptance, ProtonFlux
from trigger.dijet_rate import load_hardqcd_campaign
from trigger.signal import (
    load_hbb_signal,
    paired_signal_files,
    parse_hepmc_proton_xi,
    study_hbb_signal_efficiency,
)


class SignalSelectionParityTest(unittest.TestCase):
    """The signal must be corrected and cut exactly like the background."""

    def test_signal_requires_a_correction_map(self):
        parameters = inspect.signature(load_hbb_signal).parameters
        for name in ("correction_map", "correction_info"):
            self.assertIs(parameters[name].default, inspect.Parameter.empty)

    def test_signal_and_background_share_selection_defaults(self):
        signal = inspect.signature(load_hbb_signal).parameters
        background = inspect.signature(load_hardqcd_campaign).parameters
        for name in ("eta_max", "leading_pt_min", "subleading_pt_min"):
            self.assertEqual(signal[name].default, background[name].default)
        self.assertEqual(signal["leading_pt_min"].default, 20.0)


class HbbSignalTest(unittest.TestCase):
    def test_paired_files_are_naturally_sorted_and_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = base / "root"
            hepmc = base / "hepmc"
            roots.mkdir()
            hepmc.mkdir()
            for stem in ("sample_10", "sample_2"):
                (roots / f"{stem}.root").touch()
                (hepmc / f"{stem}.hepmc").touch()
            pairs = paired_signal_files(roots, hepmc)
            self.assertEqual(
                [root.stem for root, _source in pairs], ["sample_2", "sample_10"]
            )

    def test_hepmc_parser_uses_highest_momentum_final_proton_per_arm(self):
        content = """HepMC::Version 3.03.01
HepMC::Asciiv3-START_EVENT_LISTING
E 1 0 0
P 1 0 2212 0 0 -6800 6800 1 1
P 2 0 2212 0 0 -6900 6900 1 1
P 3 0 2212 0 0 6860 6860 1 1
E 2 0 0
P 1 0 2212 0 0 -6930 6930 1 1
P 2 0 2212 0 0 6790 6790 1 1
HepMC::Asciiv3-END_EVENT_LISTING
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.hepmc"
            path.write_text(content, encoding="utf-8")
            event_ids, left, right = parse_hepmc_proton_xi(path, 2, 14_000.0)
        np.testing.assert_array_equal(event_ids, [1, 2])
        np.testing.assert_allclose(left, [100.0 / 7000.0, 70.0 / 7000.0])
        np.testing.assert_allclose(right, [140.0 / 7000.0, 210.0 / 7000.0])

    @staticmethod
    def empty_flux():
        empty = np.empty(0)
        return ProtonFlux(
            event=np.empty(0, dtype=np.int64),
            arm=np.empty(0, dtype=np.int8),
            xi=empty,
            px=empty,
            py=empty,
            process=empty,
            metadata={"n_inelastic_generated": 1, "sqrt_s_gev": 10_000.0},
        )

    def test_signal_efficiency_is_unconditional_and_genuine_is_separate(self):
        signal = {
            "n_pileup": np.array([0, 0]),
            "dijet_rapidity": np.array([0.0, np.nan]),
            "xi_left_truth": np.array([0.01, 0.01]),
            "xi_right_truth": np.array([0.01, 0.01]),
            "root_dir": "synthetic-root",
            "hepmc_dir": "synthetic-hepmc",
            "files": 1,
            "jet_selection": {},
        }
        kwargs = dict(
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            beam_sigma_z_cm=0.0,
            single_arm_time_resolution_ps=1.0e-12,
            pv_z_resolution_cm=0.0,
            pv_time_resolution_ps=1.0e-12,
            seed=8,
        )
        first = study_hbb_signal_efficiency(
            signal,
            self.empty_flux(),
            Acceptance([(0.005, 0.02)]),
            **kwargs,
        )
        second = study_hbb_signal_efficiency(
            signal,
            self.empty_flux(),
            Acceptance([(0.005, 0.02)]),
            **kwargs,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["stages"]["central_dijet"]["efficiency"], 0.5)
        result = first["criteria"]["pps_mass__dy_0.1"]
        self.assertEqual(result["efficiency"], 0.5)
        self.assertEqual(result["genuine_pair_efficiency"], 0.5)
        self.assertEqual(result["rescue_efficiency"], 0.0)

    def test_pileup_only_pass_is_reported_as_rescue(self):
        flux = ProtonFlux(
            event=np.array([0, 0]),
            arm=np.array([-1, 1]),
            xi=np.array([0.01, 0.01]),
            px=np.zeros(2),
            py=np.zeros(2),
            process=np.zeros(2),
            metadata={"n_inelastic_generated": 1, "sqrt_s_gev": 10_000.0},
        )
        signal = {
            "n_pileup": np.array([1]),
            "dijet_rapidity": np.array([0.0]),
            "xi_left_truth": np.array([0.001]),
            "xi_right_truth": np.array([0.001]),
            "root_dir": "synthetic-root",
            "hepmc_dir": "synthetic-hepmc",
            "files": 1,
            "jet_selection": {},
        }
        report = study_hbb_signal_efficiency(
            signal,
            flux,
            Acceptance([(0.005, 0.02)]),
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            beam_sigma_z_cm=0.0,
            single_arm_time_resolution_ps=1.0e-12,
            pv_z_resolution_cm=0.0,
            pv_time_resolution_ps=1.0e-12,
            seed=9,
        )
        result = report["criteria"]["pps_mass__dy_0.1"]
        self.assertEqual(result["efficiency"], 1.0)
        self.assertEqual(result["genuine_pair_efficiency"], 0.0)
        self.assertEqual(result["pileup_pair_efficiency"], 1.0)
        self.assertEqual(result["rescue_efficiency"], 1.0)


if __name__ == "__main__":
    unittest.main()
