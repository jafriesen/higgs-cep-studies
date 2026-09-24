import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from minbias.flux import Acceptance, ProtonFlux
from trigger.dijet_rate import (
    correct_jets,
    load_hardqcd_campaign,
    load_jet_correction,
    select_dijet,
    select_hardest_indices,
    study_dijet_trigger_rates,
)


def correction_map():
    def node(raw, target):
        return {
            "raw_reco_pt_median": raw,
            "target_pt": target,
            "usable": True,
        }

    return {
        "raw_pt_support": [5.0, 250.0],
        "eta_bins": [
            {
                "eta_min": 0.0,
                "eta_max": 1.0,
                "supported": True,
                "nodes": [node(10.0, 20.0), node(20.0, 40.0)],
            },
            {
                "eta_min": 1.0,
                "eta_max": 2.5,
                "supported": True,
                "nodes": [node(10.0, 10.0), node(20.0, 20.0)],
            },
        ],
    }


class FixedCandidates:
    def integers(self, low, high, size):
        self.asserted = (low, high, size)
        return np.array([0, 2, 1, 0, 2], dtype=np.int64)


class FakeCollection:
    def __init__(self, values):
        self.values = values

    def GetEntriesFast(self):
        return len(self.values)

    def At(self, index):
        return self.values[index]


class FakeTree:
    def __init__(self, events):
        self.events = events

    def GetBranch(self, _name):
        return True

    def GetEntries(self):
        return len(self.events)

    def GetEntry(self, entry):
        event = self.events[entry]
        self.Event = FakeCollection([SimpleNamespace(Number=event["number"])])
        self.Vertex = FakeCollection([object()] * event["vertices"])
        self.JetPUPPI = FakeCollection(event["jets"])


class FakeRootFile:
    def __init__(self, tree):
        self.tree = tree

    def Get(self, name):
        return self.tree if name == "Delphes" else None

    def Close(self):
        pass


def fake_root(trees):
    return SimpleNamespace(
        TFile=SimpleNamespace(Open=lambda path: FakeRootFile(trees[str(path)]))
    )


def fake_jet(pt, eta=0.2, mass=2.0):
    return SimpleNamespace(PT=pt, Eta=eta, Phi=0.0, Mass=mass)


def fake_campaign(directory, specifications):
    shards = []
    trees = {}
    for order, (job_index, pthat, event_number) in enumerate(specifications):
        shard_dir = directory / f"job_{job_index:05d}"
        shard_dir.mkdir()
        sidecar = shard_dir / "events.parquet"
        pq.write_table(
            pa.table(
                {
                    "event_id": [0],
                    "pthat_gev": [pthat],
                    "weight": [1.0],
                }
            ),
            sidecar,
        )
        root_path = shard_dir / "delphes.root"
        root_path.touch()
        trees[str(root_path)] = FakeTree(
            [
                {
                    "number": event_number,
                    "vertices": 2,
                    "jets": [fake_jet(11.0), fake_jet(11.0), fake_jet(250.0)],
                }
            ]
        )
        shards.append(
            {
                "job": {
                    "index": job_index,
                    "role": "rate_validation",
                    "events": 1,
                },
                "event_path": sidecar,
                "root_path": root_path,
                "order": order,
            }
        )
    return {
        "campaign": directory,
        "metadata": {"pythia": {"sigma_gen_mb": 1.0, "sigma_err_mb": 0.1}},
        "shards": shards,
    }, trees


class DijetRateTest(unittest.TestCase):
    def test_campaign_concatenates_manifest_order_and_reports_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, trees = fake_campaign(
                Path(directory), [(7, 30.0, 0), (3, 20.0, 0)]
            )
            with (
                patch("trigger.dijet_rate.load_campaign", return_value=campaign),
                patch("trigger.dijet_rate.load_delphes_root", return_value=fake_root(trees)),
            ):
                loaded = load_hardqcd_campaign(
                    directory,
                    correction_map=correction_map(),
                    correction_info={"sample": "HardQCD"},
                )
        np.testing.assert_array_equal(loaded["pthat_gev"], [30.0, 20.0])
        np.testing.assert_allclose(loaded["leading_pt_gev"], [22.0, 22.0])
        self.assertEqual(loaded["jet_counts"]["total_jets"], 6)
        self.assertEqual(loaded["jet_counts"]["supported_jets"], 4)
        self.assertEqual(loaded["jet_counts"]["rejected_jets"], 2)
        self.assertAlmostEqual(loaded["jet_counts"]["correction_coverage"], 2 / 3)
        self.assertEqual(loaded["jet_selection"]["correction"]["sample"], "HardQCD")

    def test_campaign_rejects_event_number_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, trees = fake_campaign(Path(directory), [(0, 30.0, 1)])
            with (
                patch("trigger.dijet_rate.load_campaign", return_value=campaign),
                patch("trigger.dijet_rate.load_delphes_root", return_value=fake_root(trees)),
            ):
                with self.assertRaisesRegex(RuntimeError, "sidecar"):
                    load_hardqcd_campaign(
                        directory,
                        correction_map=correction_map(),
                        correction_info={"sample": "HardQCD"},
                    )

    def test_loads_hardqcd_correction_with_provenance(self):
        saved_map = correction_map()
        saved_map.update({"fsr_state": "FSR", "source_sample": "HardQCD"})
        document = {"schema_version": 4, "maps": {"FSR": {"HardQCD": saved_map}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrections.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            loaded, provenance = load_jet_correction(path)
        self.assertEqual(loaded["source_sample"], "HardQCD")
        self.assertEqual(provenance["schema_version"], 4)
        self.assertEqual(len(provenance["sha256"]), 64)

    def test_select_dijet_uses_leading_two_and_strict_thresholds(self):
        jets = [
            {"pt": 15.0, "eta": 0.0, "phi": 0.0, "mass": 0.0},
            {"pt": 30.0, "eta": 0.0, "phi": 1.0, "mass": 0.0},
            {"pt": 21.0, "eta": 0.0, "phi": 2.0, "mass": 0.0},
        ]
        leading, subleading, rapidity = select_dijet(jets)
        self.assertEqual((leading, subleading), (30.0, 21.0))
        self.assertAlmostEqual(rapidity, 0.0)
        self.assertTrue(np.isnan(select_dijet(jets, subleading_pt_min=21.0)[2]))

    def test_leading_supported_forward_jet_fails_eta_requirement(self):
        jets = [
            {"pt": 40.0, "eta": 2.5, "phi": 0.0, "mass": 2.0},
            {"pt": 30.0, "eta": 0.2, "phi": 1.0, "mass": 2.0},
            {"pt": 25.0, "eta": -0.2, "phi": 2.0, "mass": 2.0},
        ]
        leading, subleading, rapidity = select_dijet(jets)
        self.assertEqual((leading, subleading), (40.0, 30.0))
        self.assertTrue(np.isnan(rapidity))

    def test_correction_scales_mass_reorders_jets_and_drops_high_pt(self):
        jets = [
            {"pt": 15.0, "eta": 0.2, "phi": 0.0, "mass": 2.0},
            {"pt": 20.0, "eta": 1.2, "phi": 1.0, "mass": 3.0},
            {"pt": 250.0, "eta": 0.2, "phi": 2.0, "mass": 10.0},
        ]
        corrected, counts = correct_jets(jets, correction_map())
        self.assertEqual([jet["pt"] for jet in corrected], [30.0, 20.0])
        self.assertEqual(corrected[0]["mass"], 4.0)
        self.assertEqual(counts["rejected_above_pt_support"], 1)
        leading, subleading, rapidity = select_dijet(corrected)
        self.assertEqual((leading, subleading), (30.0, 20.0))
        self.assertTrue(np.isnan(rapidity))

    def test_selects_largest_generator_pthat_and_handles_zero(self):
        selected = select_hardest_indices(
            FixedCandidates(), np.array([0, 2, 3]), np.array([10.0, 30.0, 20.0])
        )
        np.testing.assert_array_equal(selected, [-1, 2, 1])

    def test_small_study_is_repeatable_and_unconditional(self):
        flux = ProtonFlux(
            event=np.array([0, 0]),
            arm=np.array([-1, 1]),
            xi=np.array([0.01, 0.01]),
            px=np.zeros(2),
            py=np.zeros(2),
            process=np.ones(2),
            metadata={
                "n_inelastic_generated": 1,
                "sqrt_s_gev": 10_000.0,
                "cross_sections": {"sigma_gen_mb": 2.0},
            },
        )
        hard = {
            "campaign": "synthetic",
            "metadata": {"pythia": {"sigma_gen_mb": 1.0, "sigma_err_mb": 0.0}},
            "pthat_gev": np.array([10.0, 20.0]),
            "n_pileup": np.array([1, 1]),
            "dijet_rapidity": np.array([0.0, 0.0]),
        }
        kwargs = dict(
            mu=1.0,
            n_bx=300,
            mass_range=(90.0, 110.0),
            xi_resolution=0.0,
            beam_sigma_z_cm=0.0,
            pv_z_resolution_cm=0.0,
            seed=9,
            batch_size=100,
            n_bootstrap=5,
        )
        first = study_dijet_trigger_rates(
            hard, flux, Acceptance([(0.005, 0.02)]), **kwargs
        )
        second = study_dijet_trigger_rates(
            hard, flux, Acceptance([(0.005, 0.02)]), **kwargs
        )
        self.assertEqual(first, second)
        expected = 1.0 - np.exp(-0.5)
        self.assertAlmostEqual(first["hardqcd"]["probability_at_least_one"], expected)
        result = first["criteria"]["pps_mass__dy_none"]
        self.assertGreater(result["passing_bx"], 0)
        self.assertLessEqual(
            result["passing_bx"], first["hardqcd"]["sampled_bx_with_hardqcd"]
        )


if __name__ == "__main__":
    unittest.main()
