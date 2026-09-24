import sys
import unittest
from pathlib import Path

import awkward as ak
import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

import plot_eflow_tracks as tracks  # noqa: E402


class EFlowTrackAnalysisTest(unittest.TestCase):
    def test_available_track_collections_returns_both(self):
        keys = {
            tracks.analyzer.branch_name(collection, field)
            for collection in ("EFlowTrack", "EFlowTrackPUPPI")
            for field in ("PT", "Eta", "Phi")
        }

        available = tracks.available_track_collections(keys)

        self.assertEqual(
            [collection["name"] for collection in available],
            ["EFlowTrack", "EFlowTrackPUPPI"],
        )

    def test_available_track_collections_falls_back_to_present_collection(self):
        keys = {
            tracks.analyzer.branch_name("EFlowTrackPUPPI", field)
            for field in ("PT", "Eta", "Phi")
        }

        available = tracks.available_track_collections(keys)

        self.assertEqual(
            [collection["name"] for collection in available],
            ["EFlowTrackPUPPI"],
        )

    def test_selection_nearest_jet_and_wrapped_phi(self):
        jet_pt = ak.Array([[30.0, 25.0], [50.0], [30.0, 19.0]])
        jet_eta = ak.Array([[0.0, 2.0], [0.0], [0.0, 1.0]])
        jet_phi = ak.Array([[-np.pi + 0.01, 1.0], [0.0], [0.0, 1.0]])
        track_pt = ak.Array([[2.0, 3.0, 0.5, 1.0], [100.0], [200.0]])
        track_eta = ak.Array([[0.0, 2.0, 1.0, 1.0], [0.0], [0.0]])
        track_phi = ak.Array([[np.pi - 0.01, 1.0, 2.0, 2.0], [0.0], [0.0]])

        delta_r, selected_pt, outer_multiplicity, n_selected = tracks.selected_track_values(
            ak,
            np,
            jet_pt,
            jet_eta,
            jet_phi,
            track_pt,
            track_eta,
            track_phi,
        )

        self.assertEqual(n_selected, 1)
        np.testing.assert_allclose(delta_r[:2], [0.02, 0.0], atol=1.0e-12)
        np.testing.assert_array_equal(selected_pt, [2.0, 3.0, 0.5, 1.0])
        np.testing.assert_array_equal(outer_multiplicity, [1])

    def test_normalized_density_integrates_to_one(self):
        widths = np.asarray([0.1, 0.2])
        density = tracks.normalized_density(np, np.asarray([1.0, 3.0]), widths)
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertAlmostEqual(float(np.sum(density * widths)), 1.0)


if __name__ == "__main__":
    unittest.main()
