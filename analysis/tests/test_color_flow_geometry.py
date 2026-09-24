import unittest

import awkward as ak
import numpy as np

from analysis.MVA import prototype_color_flow as prototype
from analysis.MVA import run_dijet_mva_exclusive_central as central
from analysis.MVA import run_dijet_mva_multiclass_protons as proton_mva
from analysis.MVA.color_flow_geometry import (
    activity_region_masks,
    jet_frame_coordinates,
    projected_bridge_asymmetry,
)


def jets(eta1, phi1, eta2, phi2):
    return (
        ak.zip({"eta": ak.Array([eta1]), "phi": ak.Array([phi1])}),
        ak.zip({"eta": ak.Array([eta2]), "phi": ak.Array([phi2])}),
    )


class ColorFlowGeometryTest(unittest.TestCase):
    def test_bridge_is_well_defined_for_equal_jet_eta(self):
        jet1, jet2 = jets(0.0, 0.0, 0.0, 3.0)
        eta = ak.Array([[0.0]])
        phi = ak.Array([[1.5]])
        u, v = jet_frame_coordinates(eta, phi, jet1, jet2)
        self.assertAlmostEqual(float(u[0, 0]), 0.5)
        self.assertAlmostEqual(float(v[0, 0]), 0.0)
        bridge, outer_pos, outer_neg, side = activity_region_masks(
            eta, phi, jet1, jet2
        )
        self.assertTrue(bool(bridge[0, 0]))
        self.assertFalse(bool(outer_pos[0, 0] | outer_neg[0, 0] | side[0, 0]))

    def test_coordinates_wrap_across_phi_boundary(self):
        jet1, jet2 = jets(0.0, 3.0, 0.0, -3.0)
        eta = ak.Array([[0.0]])
        phi = ak.Array([[np.pi]])
        u, v = jet_frame_coordinates(eta, phi, jet1, jet2)
        self.assertAlmostEqual(float(u[0, 0]), 0.5, places=12)
        self.assertAlmostEqual(float(v[0, 0]), 0.0, places=12)

    def test_regions_are_mutually_exclusive_and_complete(self):
        jet1, jet2 = jets(-1.0, 0.0, 1.0, 3.0)
        eta = ak.Array([[0.0, 2.0, -2.0, 0.0]])
        phi = ak.Array([[1.5, 3.0, 0.0, 0.0]])
        masks = activity_region_masks(eta, phi, jet1, jet2)
        total = sum(ak.to_numpy(mask).astype(np.int8) for mask in masks)
        np.testing.assert_array_equal(total, np.ones((1, 4), dtype=np.int8))

    def test_projected_bridge_asymmetry_uses_perpendicular_side_bands(self):
        jet1, jet2 = jets(0.0, 0.0, 0.0, 3.0)
        pt = ak.Array([[2.0, 1.0, 1.0]])
        eta = ak.Array([[0.0, 0.8, -0.8]])
        phi = ak.Array([[1.5, 1.5, 1.5]])
        outside = ak.Array([[True, True, True]])
        value = projected_bridge_asymmetry(pt, eta, phi, outside, jet1, jet2)
        self.assertAlmostEqual(value[0], 1.0 / 3.0)


class ColorFlowFeatureSchemaTest(unittest.TestCase):
    def test_mva_and_prototype_share_new_feature_names(self):
        expected = {
            "n_tracks_projected_bridge",
            "sum_track_pt_projected_bridge",
            "n_tracks_projected_side",
            "sum_track_pt_projected_side",
            "n_extra_track_jets",
            "n_outer_track_jets",
            "leading_outer_track_jet_pt",
            "interjet_bridge_asymmetry_projected",
        }
        self.assertTrue(expected <= set(central.FEATURE_NAMES))
        self.assertTrue(expected <= set(proton_mva.FEATURE_NAMES))
        self.assertTrue(expected <= set(prototype.OBSERVABLE_NAMES))
        self.assertEqual(len(central.FEATURE_NAMES), len(set(central.FEATURE_NAMES)))

    def test_prototype_matches_mva_track_definition(self):
        self.assertEqual(prototype.TRACK_MIN_PT, proton_mva.TRACK_MIN_PT)
        self.assertEqual(
            prototype.TRACK_MAX_ABS_ETA, proton_mva.TRACK_MAX_ABS_ETA
        )
        self.assertEqual(prototype.TRACK_JET_R, proton_mva.TRACK_JET_R)
        self.assertEqual(
            prototype.TRACK_JET_MIN_PT, proton_mva.TRACK_JET_MIN_PT
        )


if __name__ == "__main__":
    unittest.main()
