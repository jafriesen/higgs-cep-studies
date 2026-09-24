"""Shared projected geometry for dijet color-flow observables."""

import awkward as ak
import numpy as np


BRIDGE_HALF_WIDTH = 0.4
SIDE_BAND_OUTER = 1.2


def wrap_phi(values):
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def jet_frame_coordinates(object_eta, object_phi, jet1, jet2):
    """Return position along and distance from the finite jet--jet segment.

    ``u`` is the fractional projection along the segment (zero at jet1 and one
    at jet2). ``v`` is the signed perpendicular distance in wrapped
    ``(eta, phi)`` coordinates.
    """
    eta1 = ak.to_numpy(jet1.eta)
    phi1 = ak.to_numpy(jet1.phi)
    axis_eta = ak.to_numpy(jet2.eta - jet1.eta)
    axis_phi = ak.to_numpy(wrap_phi(jet2.phi - jet1.phi))
    axis_length_sq = axis_eta**2 + axis_phi**2
    axis_length = np.sqrt(axis_length_sq)
    object_deta = object_eta - eta1[:, np.newaxis]
    object_dphi = wrap_phi(object_phi - phi1[:, np.newaxis])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (
            object_deta * axis_eta[:, np.newaxis]
            + object_dphi * axis_phi[:, np.newaxis]
        ) / axis_length_sq[:, np.newaxis]
        v = (
            object_deta * axis_phi[:, np.newaxis]
            - object_dphi * axis_eta[:, np.newaxis]
        ) / axis_length[:, np.newaxis]
    return u, v


def activity_region_masks(object_eta, object_phi, jet1, jet2):
    """Return mutually exclusive projected bridge, outer, and side masks."""
    u, v = jet_frame_coordinates(object_eta, object_phi, jet1, jet2)
    bridge = (u > 0.0) & (u < 1.0) & (np.abs(v) < BRIDGE_HALF_WIDTH)
    eta_pos = np.maximum(ak.to_numpy(jet1.eta), ak.to_numpy(jet2.eta))
    eta_neg = np.minimum(ak.to_numpy(jet1.eta), ak.to_numpy(jet2.eta))
    outer_pos = (~bridge) & (object_eta > eta_pos[:, np.newaxis])
    outer_neg = (~bridge) & (object_eta < eta_neg[:, np.newaxis])
    side = (~bridge) & (~outer_pos) & (~outer_neg)
    return bridge, outer_pos, outer_neg, side


def projected_bridge_asymmetry(track_pt, track_eta, track_phi, outside, jet1, jet2):
    """Return bridge-versus-side pT asymmetry in projected dijet coordinates."""
    u, v = jet_frame_coordinates(track_eta, track_phi, jet1, jet2)
    between_endpoints = outside & (u > 0.0) & (u < 1.0)
    bridge = between_endpoints & (np.abs(v) < BRIDGE_HALF_WIDTH)
    side = between_endpoints & (np.abs(v) >= BRIDGE_HALF_WIDTH) & (
        np.abs(v) < SIDE_BAND_OUTER
    )
    pt_bridge = ak.to_numpy(ak.sum(track_pt[bridge], axis=1))
    pt_up = ak.to_numpy(ak.sum(track_pt[side & (v > 0.0)], axis=1))
    pt_down = ak.to_numpy(ak.sum(track_pt[side & (v < 0.0)], axis=1))
    mean_side = 0.5 * (pt_up + pt_down)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (pt_bridge - mean_side) / (pt_bridge + mean_side)
