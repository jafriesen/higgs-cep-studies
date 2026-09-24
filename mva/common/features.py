"""Shared central-feature construction for the H(bb) and H(cc) MVAs."""

import site
import sys

import awkward as ak
import numpy as np
import uproot
import vector
from uproot.source.file import MemmapSource

try:
    import fastjet
except ImportError:
    sys.path.append(site.getusersitepackages())
    import fastjet

vector.register_awkward()

from common.jet_calibration import correct_jet_kinematics, correction_factors
from mva.common.color_flow_geometry import activity_region_masks, projected_bridge_asymmetry

MIN_JET_PT = 15.0
JET_CONE_R = 0.4
TRACK_MIN_PT = 0.5
HARD_TRACK_MIN_PT = 2.0
TRACK_MAX_ABS_ETA = 4.0
TRANSVERSE_TRACK_MAX_ABS_ETA = 1.75
TRACKER_ETA = 4.0
MIN_DIJET_DELTA_PHI = 3.0
PRE_MVA_DIJET_MASS_RANGE_GEV = (50.0, 150.0)
TRACK_JET_R = 0.2
TRACK_JET_MIN_PT = 3.0
BRIDGE_HALF_WIDTH = 0.4
SIDE_BAND_OUTER = 1.2
SOFT_LEPTON_MIN_PT = 2.0
FSR_RECOVERY_MAX_R = 1.0
MAX_MATCH_ABS_ETA = 3.0
PARTON_GENJET_DR = 0.4
GENJET_PUPPI_DR = 0.2

# FSR recovery. Radiation off the b/c quarks lands outside the 0.4 jet cones, so
# the pair is rebuilt as (leading jet + nearby softer jets + in-jet muons) before
# calibration. Measured on H(bb) at 200 pileup: m(jj)/m(bb) 0.769 -> 0.840 and the
# mass resolution 15.4% -> 14.2%; muons alone are worth +3% because Delphes jets
# exclude them. Corrections must be derived on this same object, otherwise the
# recovered energy is counted twice.
RECOVERY_JET_PT_MIN = 5.0
RECOVERY_JET_MAX_DR = 1.5
RECOVERY_MUON_MAX_DR = 0.4
PARTON_MATCH_DR = 0.3

FEATURE_NAMES = (
    # dijet / single-jet kinematics
    "jet1_pt",
    "jet2_pt",
    "jet1_pt_over_mjj",
    "jet2_pt_over_mjj",
    "jet1_eta",
    "jet2_eta",
    # leading-jet transverse mass and rapidity: inputs to the proton-frame mass estimator
    "jet1_mt",
    "jet1_rapidity",
    "delta_eta_jj",
    "delta_phi_jj",
    "dijet_pt",
    "dijet_rapidity",
    "dijet_mass",
    "dijet_mass_fsr",
    "fsr_recovered_pt",
    "jet_multiplicity",
    "cos_theta_star",
    "pt_asymmetry",
    # per-jet substructure
    "jet1_ncharged",
    "jet2_ncharged",
    "jet1_ptd",
    "jet2_ptd",
    "jet1_width",
    "jet2_width",
    "jet1_charged_fraction",
    "jet2_charged_fraction",
    "jet1_pull_magnitude",
    "jet2_pull_magnitude",
    # color flow / exclusivity (PV tracks)
    "n_tracks_outside_jets",
    "sum_track_pt_outside_jets",
    "n_tracks_outside_pt2",
    "sum_track_pt_outside_pt2",
    "max_track_pt_outside_jets",
    # underlying-event transverse region relative to the leading jet
    "n_tracks_transverse",
    "sum_track_pt_transverse",
    # legacy eta-strip activity, retained for a controlled feature ablation
    "n_tracks_interjet",
    "sum_track_pt_interjet",
    # finite jet--jet segment in wrapped (eta, phi)
    "n_tracks_projected_bridge",
    "sum_track_pt_projected_bridge",
    "n_tracks_projected_side",
    "sum_track_pt_projected_side",
    "n_tracks_outer_positive",
    "n_tracks_outer_negative",
    "sum_track_pt_outer_positive",
    "sum_track_pt_outer_negative",
    "sum_gap_size",
    "minimum_gap_size",
    "track_clumping_fraction",
    "leading_track_jet_fraction",
    "n_extra_track_jets",
    "n_outer_track_jets",
    "leading_outer_track_jet_pt",
    "interjet_bridge_asymmetry",
    "interjet_bridge_asymmetry_projected",
    "eta_rms_outside",
    # leptons / MET / vertices
    "n_soft_leptons_in_jets",
    "max_in_jet_lepton_pt",
    "puppi_met",
    "met_min_dphi_jet",
    "n_vertices",
)
# Proton-dependent features are built at training time, per real proton pair or per
# analytic delta-y cell. jet1_mass_estimator = 2 m_T cosh(y_j1 - y_X) is twice the
# leading jet energy in the frame where the proton system has no longitudinal momentum.
PROTON_FEATURE_NAMES = ("yx_minus_dijet_rapidity", "jet1_mass_estimator")
ALL_FEATURE_NAMES = FEATURE_NAMES + PROTON_FEATURE_NAMES
MASS_ESTIMATOR_INPUTS = ("jet1_mt", "jet1_rapidity")


def stored_central_features(requested):
    """Central columns a dataset must store for the requested logical features.

    Proton features are not stored; the mass estimator needs its jet inputs stored
    even when they are not themselves training features.
    """
    names = [name for name in requested if name not in PROTON_FEATURE_NAMES]
    if "jet1_mass_estimator" in requested:
        names += [name for name in MASS_ESTIMATOR_INPUTS if name not in names]
    return names


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def wrap_phi(np_like):
    return (np_like + np.pi) % (2.0 * np.pi) - np.pi


def transverse_track_mask(track_eta, track_phi, leading_jet_phi):
    """Tracks in the paper's transverse azimuthal region, with its eta acceptance."""
    abs_dphi = np.abs(wrap_phi(track_phi - leading_jet_phi[:, np.newaxis]))
    return (
        (abs_dphi >= np.pi / 3.0)
        & (abs_dphi <= 2.0 * np.pi / 3.0)
        & (np.abs(track_eta) < TRANSVERSE_TRACK_MAX_ABS_ETA)
    )


def scalar_max(values, mask, empty=0.0):
    out = ak.to_numpy(ak.max(values[mask], axis=1, mask_identity=False))
    return np.where(np.isfinite(out), out, empty)


def jet_pull_magnitude(track_pt, track_eta, track_phi, jet):
    deta = track_eta - jet.eta[:, np.newaxis]
    dphi = wrap_phi(track_phi - jet.phi[:, np.newaxis])
    dr = np.hypot(deta, dphi)
    in_jet = (dr < JET_CONE_R) & (track_pt >= TRACK_MIN_PT)
    jet_pt = ak.to_numpy(jet.pt)
    pull_eta = ak.to_numpy(ak.sum((track_pt * deta * dr)[in_jet], axis=1)) / jet_pt
    pull_phi = ak.to_numpy(ak.sum((track_pt * dphi * dr)[in_jet], axis=1)) / jet_pt
    return np.hypot(pull_eta, pull_phi)


def cluster_track_jets(track_pt, track_eta, track_phi, mask):
    particles = ak.zip(
        {
            "pt": track_pt[mask],
            "eta": track_eta[mask],
            "phi": track_phi[mask],
            "mass": ak.zeros_like(track_pt[mask]),
        },
        with_name="Momentum4D",
    )
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, TRACK_JET_R)
    raw = fastjet.ClusterSequence(particles, jet_def).inclusive_jets(min_pt=TRACK_JET_MIN_PT)
    return ak.zip({"px": raw.px, "py": raw.py, "pz": raw.pz, "E": raw.E}, with_name="Momentum4D")


def bridge_asymmetry(track_pt, track_eta, track_phi, interjet, jet1, jet2):
    eta1 = ak.to_numpy(jet1.eta)
    eta2 = ak.to_numpy(jet2.eta)
    phi1 = ak.to_numpy(jet1.phi)
    phi2_unwrapped = phi1 + ak.to_numpy(wrap_phi(jet2.phi - jet1.phi))
    deta_jj = eta2 - eta1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (track_eta - eta1[:, np.newaxis]) / deta_jj[:, np.newaxis]
    line_phi = phi1[:, np.newaxis] + frac * (phi2_unwrapped - phi1)[:, np.newaxis]
    dphi_line = wrap_phi(track_phi - line_phi)
    in_bridge = interjet & (np.abs(dphi_line) < BRIDGE_HALF_WIDTH)
    in_side = interjet & (np.abs(dphi_line) >= BRIDGE_HALF_WIDTH) & (
        np.abs(dphi_line) < SIDE_BAND_OUTER
    )
    pt_bridge = ak.to_numpy(ak.sum(track_pt[in_bridge], axis=1))
    pt_up = ak.to_numpy(ak.sum(track_pt[in_side & (dphi_line > 0)], axis=1))
    pt_down = ak.to_numpy(ak.sum(track_pt[in_side & (dphi_line < 0)], axis=1))
    mean_side = 0.5 * (pt_up + pt_down)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (pt_bridge - mean_side) / (pt_bridge + mean_side)


def _padded_numpy(values, width=None, fill=np.nan):
    if width is None:
        width = int(ak.max(ak.num(values), initial=0))
    if width == 0:
        return np.empty((len(values), 0), dtype=np.float64)
    padded = ak.pad_none(values, width, clip=True)
    return ak.to_numpy(ak.fill_none(padded, fill))


def _match_two_to_many(first_eta, first_phi, second_eta, second_phi, max_delta_r):
    """Vectorized form of greedy unique smallest-first matching for two objects."""
    first_eta = np.asarray(first_eta, dtype=np.float64)
    first_phi = np.asarray(first_phi, dtype=np.float64)
    if first_eta.ndim != 2 or first_eta.shape[1] != 2:
        raise ValueError("first objects must have shape (events, 2)")
    second_eta = _padded_numpy(second_eta)
    second_phi = _padded_numpy(second_phi, width=second_eta.shape[1])
    events = first_eta.shape[0]
    indices = np.full((events, 2), -1, dtype=np.int64)
    distances = np.full((events, 2), np.nan, dtype=np.float64)
    if second_eta.shape[1] == 0:
        return indices, distances, np.zeros(events, dtype=bool)

    deta = first_eta[:, :, np.newaxis] - second_eta[:, np.newaxis, :]
    dphi = wrap_phi(first_phi[:, :, np.newaxis] - second_phi[:, np.newaxis, :])
    candidates = np.hypot(deta, dphi)
    candidates = np.where(candidates < max_delta_r, candidates, np.inf)
    flat = candidates.reshape(events, -1)
    first_flat = np.argmin(flat, axis=1)
    first_distance = flat[np.arange(events), first_flat]
    first_object = first_flat // second_eta.shape[1]
    first_match = first_flat % second_eta.shape[1]
    has_first = np.isfinite(first_distance)
    rows = np.flatnonzero(has_first)
    indices[rows, first_object[rows]] = first_match[rows]
    distances[rows, first_object[rows]] = first_distance[rows]

    other_object = 1 - first_object
    remaining = candidates[np.arange(events), other_object].copy()
    remaining[np.arange(events), first_match] = np.inf
    second_match = np.argmin(remaining, axis=1)
    second_distance = remaining[np.arange(events), second_match]
    has_second = has_first & np.isfinite(second_distance)
    rows = np.flatnonzero(has_second)
    indices[rows, other_object[rows]] = second_match[rows]
    distances[rows, other_object[rows]] = second_distance[rows]
    return indices, distances, has_second


def _gather_jagged(values, indices, fill=np.nan):
    padded = _padded_numpy(values, fill=fill)
    output = np.full(indices.shape, fill, dtype=padded.dtype if padded.size else np.float64)
    if padded.shape[1] == 0:
        return output
    valid = (indices >= 0) & (indices < padded.shape[1])
    rows, columns = np.nonzero(valid)
    output[rows, columns] = padded[rows, indices[rows, columns]]
    return output


def _pair_assignment_masks(pair, objects, max_delta_r, min_pt=None):
    """Assign each object to at most one jet, with exact ties going to jet one."""
    first, second = pair[:, 0], pair[:, 1]
    delta_first = objects.deltaR(first)
    delta_second = objects.deltaR(second)
    first_mask = (delta_first < max_delta_r) & (delta_first <= delta_second)
    second_mask = (delta_second < max_delta_r) & (delta_second < delta_first)
    if min_pt is not None:
        first_mask = first_mask & (objects.pt > min_pt)
        second_mask = second_mask & (objects.pt > min_pt)
    return first_mask, second_mask


def _recovered_chosen_pair(
    chosen_indices, pool_mask, original_index,
    raw_pt, raw_eta, raw_phi, raw_mass, raw_ncharged,
    muon_pt, muon_eta, muon_phi, correction_map,
):
    """Recover and calibrate a chosen pair, also returning absorbed donor indices."""
    core = ak.zip(
        {
            "pt": _gather_jagged(raw_pt, chosen_indices),
            "eta": _gather_jagged(raw_eta, chosen_indices),
            "phi": _gather_jagged(raw_phi, chosen_indices),
            "mass": _gather_jagged(raw_mass, chosen_indices),
        },
        with_name="Momentum4D",
    )
    donor_mask = pool_mask
    for column in (0, 1):
        donor_mask = donor_mask & (original_index != chosen_indices[:, column][:, np.newaxis])
    donors = ak.zip(
        {"pt": raw_pt[donor_mask], "eta": raw_eta[donor_mask],
         "phi": raw_phi[donor_mask], "mass": raw_mass[donor_mask],
         "ncharged": raw_ncharged[donor_mask],
         "original_index": original_index[donor_mask]},
        with_name="Momentum4D",
    )
    muons = ak.zip(
        {"pt": muon_pt, "eta": muon_eta, "phi": muon_phi, "mass": 0.0 * muon_pt},
        with_name="Momentum4D",
    )
    first, second, _n1, _n2 = recovered_pair(core, donors, muons)
    donor_assignments = _pair_assignment_masks(
        core, donors, RECOVERY_JET_MAX_DR, RECOVERY_JET_PT_MIN
    )
    muon_assignments = _pair_assignment_masks(core, muons, RECOVERY_MUON_MAX_DR)
    stacked = {
        name: np.stack([ak.to_numpy(getattr(jet, name)) for jet in (first, second)], axis=1)
        for name in ("pt", "eta", "phi", "mass")
    }
    core_ncharged = _gather_jagged(raw_ncharged, chosen_indices)
    stacked["ncharged"] = np.stack(
        [
            core_ncharged[:, index]
            + ak.to_numpy(ak.sum(donors.ncharged[donor_assignments[index]], axis=1))
            + ak.to_numpy(ak.num(muons[muon_assignments[index]]))
            for index in (0, 1)
        ],
        axis=1,
    )
    used_donor_indices = ak.concatenate(
        [
            donors.original_index[donor_assignments[0]],
            donors.original_index[donor_assignments[1]],
        ],
        axis=1,
    )
    factors, valid = correction_factors(stacked["pt"], stacked["eta"], correction_map)
    return (
        {
            "pt": np.where(valid, stacked["pt"] * factors, np.nan),
            "eta": stacked["eta"],
            "phi": stacked["phi"],
            "mass": np.where(valid, stacked["mass"] * factors, np.nan),
            "ncharged": stacked["ncharged"],
            "valid": valid & (chosen_indices >= 0),
        },
        used_donor_indices,
    )


def recovered_pair(pair, others, muons):
    """Add nearby softer jets and in-jet muons to each jet of the selected pair.

    `pair` holds the two selected jets (fields pt/eta/phi/mass, one row per event),
    `others` the remaining jets in the event, `muons` the loose muons. Each extra
    object joins whichever jet of the pair is nearer, when inside the recovery cone.
    Returns the two recovered Momentum4D jets and the number of jets added to each.
    """
    recovered = []
    counts = []
    first, second = pair[:, 0], pair[:, 1]
    donor_assignments = (
        _pair_assignment_masks(pair, others, RECOVERY_JET_MAX_DR, RECOVERY_JET_PT_MIN)
        if others is not None else (None, None)
    )
    muon_assignments = (
        _pair_assignment_masks(pair, muons, RECOVERY_MUON_MAX_DR)
        if muons is not None else (None, None)
    )
    for index, mine in enumerate((first, second)):
        components = {axis: getattr(mine, axis) for axis in ("px", "py", "pz", "E")}
        added = None
        for objects, assignments in ((others, donor_assignments), (muons, muon_assignments)):
            if objects is None:
                continue
            near = assignments[index]
            if objects is others:
                added = ak.num(objects[near])
            selected = objects[near]
            for axis in components:
                components[axis] = components[axis] + ak.sum(getattr(selected, axis), axis=1)
        recovered.append(ak.zip(components, with_name="Momentum4D"))
        counts.append(added)
    return recovered[0], recovered[1], counts[0], counts[1]


def _correct_jagged_jets(pt, eta, phi, mass, correction_map):
    counts = ak.to_numpy(ak.num(pt))
    flat = [ak.to_numpy(ak.flatten(values)) for values in (pt, eta, phi, mass)]
    if correction_map is None:
        corrected = {
            "pt": flat[0],
            "eta": flat[1],
            "phi": flat[2],
            "mass": flat[3],
            "valid": np.ones(flat[0].shape, dtype=bool),
        }
    else:
        corrected = correct_jet_kinematics(*flat, correction_map)
    return {
        name: ak.unflatten(np.asarray(corrected[name]), counts)
        for name in ("pt", "eta", "phi", "mass", "valid")
    }


def central_features(
    path,
    tree_name,
    collection,
    *,
    truth_pid_abs,
    jet_mode="leading",
    correction_map=None,
    track_min_pt=HARD_TRACK_MIN_PT,
    max_abs_jet_eta=None,
    cutflow_only=False,
):
    """Build central features from a configurable selected pair."""
    if truth_pid_abs not in (4, 5, 21):
        raise ValueError("truth_pid_abs must be 4, 5, or 21")
    if jet_mode not in {"leading", "truth"}:
        raise ValueError("jet_mode must be 'leading' or 'truth'")
    if track_min_pt < TRACK_MIN_PT:
        raise ValueError(f"track_min_pt must be at least {TRACK_MIN_PT}")
    if max_abs_jet_eta is not None and max_abs_jet_eta <= 0:
        raise ValueError("max_abs_jet_eta must be positive or None")
    jet_fields = (
        "PT", "Eta", "Phi", "Mass", "NCharged", "PTD", "MeanSqDeltaR",
        "ChargedEnergyFraction",
    )
    required = [branch_name(collection, field) for field in jet_fields]
    required.extend(
        branch_name("EFlowTrack", field) for field in ("PT", "Eta", "Phi", "IsRecoPU")
    )
    required.extend(branch_name("MuonLoose", field) for field in ("PT", "Eta", "Phi"))
    required.extend(branch_name("Electron", field) for field in ("PT", "Eta", "Phi"))
    required.extend(branch_name("PuppiMissingET", field) for field in ("MET", "Phi"))
    required.append(branch_name("Vertex", "Z"))
    required.extend(
        branch_name("Particle", field)
        for field in ("PID", "Status", "IsPU", "PT", "Eta", "Phi", "Mass")
    )
    required.extend(
        branch_name("GenJet", field) for field in ("PT", "Eta", "Phi", "Mass")
    )
    try:
        with uproot.open(path, handler=MemmapSource) as root_file:
            tree = root_file[tree_name]
            arrays = tree.arrays(required, library="ak")
            n_generated = int(tree.num_entries)
    except Exception as exc:
        raise RuntimeError(f"Failed to read required data from {path}: {exc}") from exc

    def float_field(coll, field):
        return ak.values_astype(arrays[branch_name(coll, field)], "float64")

    raw_pt = float_field(collection, "PT")
    raw_eta = float_field(collection, "Eta")
    raw_phi = float_field(collection, "Phi")
    raw_mass = float_field(collection, "Mass")
    corrected = _correct_jagged_jets(
        raw_pt, raw_eta, raw_phi, raw_mass, correction_map
    )
    original_index = ak.local_index(raw_pt)
    jet_acceptance = ak.ones_like(raw_eta, dtype=bool)
    if max_abs_jet_eta is not None:
        jet_acceptance = np.abs(raw_eta) < max_abs_jet_eta
    jet_mask = (
        jet_acceptance
        & corrected["valid"]
        & (corrected["pt"] >= MIN_JET_PT)
    )
    jets = ak.zip(
        {
            "pt": corrected["pt"][jet_mask],
            "eta": corrected["eta"][jet_mask],
            "phi": corrected["phi"][jet_mask],
            "mass": corrected["mass"][jet_mask],
            "ncharged": float_field(collection, "NCharged")[jet_mask],
            "ptd": float_field(collection, "PTD")[jet_mask],
            "mean_sq_dr": float_field(collection, "MeanSqDeltaR")[jet_mask],
            "charged_fraction": float_field(collection, "ChargedEnergyFraction")[jet_mask],
            "original_index": original_index[jet_mask],
        },
        with_name="Momentum4D",
    )
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    leading_indices = _padded_numpy(jets.original_index, width=2, fill=-1).astype(np.int64)

    # A map derived on the FSR-recovered pair carries its recovery definition, so the
    # object calibrated here is the object that was calibrated there. The pair is then
    # chosen on raw pT from a pool reaching down to the donor threshold: the 15 GeV cut
    # belongs after recovery, otherwise a jet that only clears it once its radiation is
    # added back would be lost before it could be recovered.
    recovery = (correction_map or {}).get("recovery") if correction_map else None
    if recovery:
        pool_mask = jet_acceptance & (raw_pt >= float(recovery["jet_pt_min_gev"]))
        pool_order = ak.argsort(raw_pt[pool_mask], axis=1, ascending=False)
        leading_indices = _padded_numpy(
            original_index[pool_mask][pool_order], width=2, fill=-1
        ).astype(np.int64)

    particle_pid = arrays[branch_name("Particle", "PID")]
    particle_status = arrays[branch_name("Particle", "Status")]
    particle_is_pu = arrays[branch_name("Particle", "IsPU")]
    hard_base = (particle_status == 23) & (particle_is_pu == 0)
    if truth_pid_abs == 21:
        hard_gluons = hard_base & (particle_pid == 21)
        hard_valid = ak.to_numpy(ak.sum(hard_gluons, axis=1) == 2)

        def hard_values(field):
            return _padded_numpy(float_field("Particle", field)[hard_gluons], width=2)
    else:
        quark = hard_base & (particle_pid == truth_pid_abs)
        antiquark = hard_base & (particle_pid == -truth_pid_abs)
        hard_valid = ak.to_numpy(
            (ak.sum(quark, axis=1) == 1) & (ak.sum(antiquark, axis=1) == 1)
        )

        def hard_values(field):
            values = float_field("Particle", field)
            return np.column_stack(
                [
                    ak.to_numpy(ak.fill_none(ak.firsts(values[mask]), np.nan))
                    for mask in (quark, antiquark)
                ]
            )

    parton_pt = hard_values("PT")
    parton_eta = hard_values("Eta")
    parton_phi = hard_values("Phi")
    parton_mass = hard_values("Mass")

    gen_mask = np.abs(float_field("GenJet", "Eta")) < MAX_MATCH_ABS_ETA
    gen_eta = float_field("GenJet", "Eta")[gen_mask]
    gen_phi = float_field("GenJet", "Phi")[gen_mask]
    parton_gen_index, _parton_gen_dr, complete_gen = _match_two_to_many(
        parton_eta, parton_phi, gen_eta, gen_phi, PARTON_GENJET_DR
    )
    complete_gen &= hard_valid
    selected_gen_eta = _gather_jagged(gen_eta, parton_gen_index)
    selected_gen_phi = _gather_jagged(gen_phi, parton_gen_index)

    reco_match_mask = np.abs(raw_eta) < MAX_MATCH_ABS_ETA
    reco_match_eta = raw_eta[reco_match_mask]
    reco_match_phi = raw_phi[reco_match_mask]
    reco_original_index = original_index[reco_match_mask]
    gen_reco_local, match_dr, complete_reco = _match_two_to_many(
        selected_gen_eta,
        selected_gen_phi,
        reco_match_eta,
        reco_match_phi,
        GENJET_PUPPI_DR,
    )
    truth_indices = _gather_jagged(
        reco_original_index, gen_reco_local, fill=-1
    ).astype(np.int64)
    truth_matched = complete_gen & complete_reco

    use_truth = truth_matched if jet_mode == "truth" else np.zeros_like(truth_matched)
    chosen_indices = np.where(use_truth[:, np.newaxis], truth_indices, leading_indices)
    recovered = None
    used_donor_indices = None
    if recovery:
        recovered, used_donor_indices = _recovered_chosen_pair(
            chosen_indices, pool_mask, original_index,
            raw_pt, raw_eta, raw_phi, raw_mass, float_field(collection, "NCharged"),
            float_field("MuonLoose", "PT"), float_field("MuonLoose", "Eta"),
            float_field("MuonLoose", "Phi"), correction_map,
        )
        chosen_pt = recovered["pt"]
    else:
        chosen_pt = _gather_jagged(corrected["pt"], chosen_indices)
    order = np.argsort(-chosen_pt, axis=1)
    chosen_indices = np.take_along_axis(chosen_indices, order, axis=1)
    chosen_pt = np.take_along_axis(chosen_pt, order, axis=1)
    if recovered is not None:
        recovered = {
            name: np.take_along_axis(values, order, axis=1)
            for name, values in recovered.items()
        }

        selected_core = ak.zeros_like(raw_pt, dtype=bool)
        for column in (0, 1):
            selected_core = selected_core | (
                original_index == chosen_indices[:, column][:, np.newaxis]
            )
        absorbed_donor = ak.any(
            original_index[:, :, np.newaxis] == used_donor_indices[:, np.newaxis, :],
            axis=2,
        )
        # Recovery defines only the selected pair. Count those two final objects,
        # then raw AK4 jets above threshold that were not absorbed as donors.
        remaining_jet = (
            jet_acceptance
            & (raw_pt >= MIN_JET_PT)
            & ~selected_core
            & ~absorbed_donor
        )
        jet_multiplicity = 2.0 + ak.to_numpy(ak.sum(remaining_jet, axis=1))
    else:
        jet_multiplicity = ak.to_numpy(ak.num(jets.pt)).astype(np.float64)

    if recovered is not None:
        correction_valid = np.all(recovered["valid"].astype(bool), axis=1)
    else:
        correction_valid = np.all(
            _gather_jagged(corrected["valid"], chosen_indices, fill=False).astype(bool),
            axis=1,
        )
    raw_order = ak.argsort(raw_pt[jet_acceptance], axis=1, ascending=False)
    raw_leading_indices = _padded_numpy(
        original_index[jet_acceptance][raw_order], width=2, fill=-1
    ).astype(np.int64)
    raw_trigger_mask = jet_acceptance & (raw_pt >= MIN_JET_PT)
    raw_trigger_order = ak.argsort(
        raw_pt[raw_trigger_mask], axis=1, ascending=False
    )
    raw_trigger_indices = _padded_numpy(
        original_index[raw_trigger_mask][raw_trigger_order], width=2, fill=-1
    ).astype(np.int64)
    raw_leading_valid = np.all(
        _gather_jagged(corrected["valid"], raw_leading_indices, fill=False).astype(bool),
        axis=1,
    )
    diagnostic_indices = np.where(
        use_truth[:, np.newaxis], truth_indices, raw_trigger_indices
    )
    correction_candidate = np.all(diagnostic_indices >= 0, axis=1)
    correction_diagnostic_valid = np.all(
        _gather_jagged(corrected["valid"], diagnostic_indices, fill=False).astype(bool),
        axis=1,
    )
    fallback = ~use_truth
    if correction_map is not None:
        correction_valid &= (~fallback) | raw_leading_valid
    candidate_pair = np.all(chosen_indices >= 0, axis=1)
    has_two = (
        candidate_pair
        & correction_valid
        & np.all(np.isfinite(chosen_pt) & (chosen_pt >= MIN_JET_PT), axis=1)
    )
    if not np.any(has_two):
        return {
            "n_generated": n_generated,
            "empty": True,
            "cutflow": {"two_jets": 0, "delta_phi": 0, "dijet_mass": 0},
            "correction_candidate_events": int(np.sum(correction_candidate)),
            "correction_invalid_events": int(
                np.sum(correction_candidate & ~correction_diagnostic_valid)
            ),
        }

    if recovered is not None:
        pair_kinematics = {name: recovered[name] for name in ("pt", "eta", "phi", "mass")}
    else:
        pair_kinematics = {
            name: _gather_jagged(corrected[name], chosen_indices)
            for name in ("pt", "eta", "phi", "mass")
        }
    chosen = ak.zip(
        {
            **pair_kinematics,
            "ncharged": (
                recovered["ncharged"]
                if recovered is not None
                else _gather_jagged(float_field(collection, "NCharged"), chosen_indices)
            ),
            "ptd": _gather_jagged(float_field(collection, "PTD"), chosen_indices),
            "mean_sq_dr": _gather_jagged(float_field(collection, "MeanSqDeltaR"), chosen_indices),
            "charged_fraction": _gather_jagged(
                float_field(collection, "ChargedEnergyFraction"), chosen_indices
            ),
        },
        with_name="Momentum4D",
    )
    jets = jets[has_two]
    jet_multiplicity = jet_multiplicity[has_two]
    chosen = chosen[has_two]
    leading, subleading = chosen[:, 0], chosen[:, 1]
    dijet = leading + subleading
    mjj = ak.to_numpy(dijet.mass)
    abs_dphi_jj = np.abs(ak.to_numpy(wrap_phi(leading.phi - subleading.phi)))
    delta_phi_count = int(np.sum(abs_dphi_jj > MIN_DIJET_DELTA_PHI))
    keep = (
        (abs_dphi_jj > MIN_DIJET_DELTA_PHI)
        & (mjj >= PRE_MVA_DIJET_MASS_RANGE_GEV[0])
        & (mjj <= PRE_MVA_DIJET_MASS_RANGE_GEV[1])
    )
    if not np.any(keep):
        return {
            "n_generated": n_generated,
            "empty": True,
            "cutflow": {
                "two_jets": int(np.sum(has_two)),
                "delta_phi": delta_phi_count,
                "dijet_mass": 0,
            },
            "correction_candidate_events": int(np.sum(correction_candidate)),
            "correction_invalid_events": int(
                np.sum(correction_candidate & ~correction_diagnostic_valid)
            ),
        }
    keep_ak = ak.Array(keep)
    jets = jets[keep_ak]
    jet_multiplicity = jet_multiplicity[keep]
    chosen = chosen[keep_ak]
    leading, subleading = chosen[:, 0], chosen[:, 1]
    dijet = leading + subleading
    mjj = mjj[keep]
    abs_dphi_jj = abs_dphi_jj[keep]
    event_indices = np.flatnonzero(has_two)[keep]
    if cutflow_only:
        return {
            "n_generated": n_generated,
            "cutflow": {
                "two_jets": int(np.sum(has_two)),
                "delta_phi": delta_phi_count,
                "dijet_mass": int(np.sum(keep)),
            },
            "event_indices": event_indices,
            "dijet_rapidity": ak.to_numpy(dijet.rapidity),
        }
    selected_truth_matched = truth_matched[has_two][keep]
    sorted_match_dr = np.take_along_axis(match_dr, order, axis=1)
    selected_match_dr = sorted_match_dr[has_two][keep]
    leading_pair_indices = leading_indices
    truth_is_leading = truth_matched & (
        np.sort(truth_indices, axis=1) == np.sort(leading_pair_indices, axis=1)
    ).all(axis=1)
    selected_truth_is_leading = truth_is_leading[has_two][keep]

    partons = ak.zip(
        {"pt": parton_pt, "eta": parton_eta, "phi": parton_phi, "mass": parton_mass},
        with_name="Momentum4D",
    )
    parton_dijet_mass = ak.to_numpy((partons[:, 0] + partons[:, 1]).mass)
    parton_values = {
        "valid": hard_valid[event_indices],
        "pt1": parton_pt[event_indices, 0],
        "pt2": parton_pt[event_indices, 1],
        "eta1": parton_eta[event_indices, 0],
        "eta2": parton_eta[event_indices, 1],
        "mass": parton_dijet_mass[event_indices],
    }

    def event_slice(coll, field):
        values = ak.values_astype(arrays[branch_name(coll, field)], "float64")
        return values[has_two][keep_ak]

    track_pt = event_slice("EFlowTrack", "PT")
    track_eta = event_slice("EFlowTrack", "Eta")
    track_phi = event_slice("EFlowTrack", "Phi")
    track_is_pu = event_slice("EFlowTrack", "IsRecoPU")
    pv_track = (
        (track_is_pu == 0)
        & (track_pt >= track_min_pt)
        & (np.abs(track_eta) < TRACK_MAX_ABS_ETA)
    )
    track_pt, track_eta, track_phi = track_pt[pv_track], track_eta[pv_track], track_phi[pv_track]

    features = {}
    jet1_pt = ak.to_numpy(leading.pt)
    jet2_pt = ak.to_numpy(subleading.pt)
    y1 = ak.to_numpy(leading.rapidity)
    y2 = ak.to_numpy(subleading.rapidity)
    features["jet1_pt"] = jet1_pt
    features["jet2_pt"] = jet2_pt
    features["jet1_pt_over_mjj"] = jet1_pt / mjj
    features["jet2_pt_over_mjj"] = jet2_pt / mjj
    features["jet1_eta"] = ak.to_numpy(leading.eta)
    features["jet1_mt"] = ak.to_numpy(leading.mt)
    features["jet1_rapidity"] = y1
    features["jet2_eta"] = ak.to_numpy(subleading.eta)
    features["delta_eta_jj"] = np.abs(ak.to_numpy(leading.eta - subleading.eta))
    features["delta_phi_jj"] = abs_dphi_jj
    features["dijet_pt"] = ak.to_numpy(dijet.pt)
    features["dijet_rapidity"] = ak.to_numpy(dijet.rapidity)
    features["dijet_mass"] = mjj
    features["jet_multiplicity"] = jet_multiplicity
    features["cos_theta_star"] = np.abs(np.tanh(0.5 * (y1 - y2)))
    features["pt_asymmetry"] = (jet1_pt - jet2_pt) / (jet1_pt + jet2_pt)

    for tag, jet in (("jet1", leading), ("jet2", subleading)):
        features[f"{tag}_ncharged"] = ak.to_numpy(jet.ncharged)
        features[f"{tag}_ptd"] = ak.to_numpy(jet.ptd)
        features[f"{tag}_width"] = np.sqrt(np.clip(ak.to_numpy(jet.mean_sq_dr), 0.0, None))
        features[f"{tag}_charged_fraction"] = ak.to_numpy(jet.charged_fraction)
        features[f"{tag}_pull_magnitude"] = jet_pull_magnitude(
            track_pt, track_eta, track_phi, jet
        )

    dr1 = np.hypot(
        track_eta - leading.eta[:, np.newaxis],
        wrap_phi(track_phi - leading.phi[:, np.newaxis]),
    )
    dr2 = np.hypot(
        track_eta - subleading.eta[:, np.newaxis],
        wrap_phi(track_phi - subleading.phi[:, np.newaxis]),
    )
    outside = (dr1 > JET_CONE_R) & (dr2 > JET_CONE_R)
    outside_hard = outside & (track_pt >= HARD_TRACK_MIN_PT)
    transverse = transverse_track_mask(
        track_eta, track_phi, ak.to_numpy(leading.phi)
    )
    eta_pos = np.maximum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    eta_neg = np.minimum(ak.to_numpy(leading.eta), ak.to_numpy(subleading.eta))
    interjet = outside & (track_eta > eta_neg[:, np.newaxis]) & (track_eta < eta_pos[:, np.newaxis])
    outer_pos = outside & (track_eta > eta_pos[:, np.newaxis])
    outer_neg = outside & (track_eta < eta_neg[:, np.newaxis])
    projected_bridge, _projected_outer_pos, _projected_outer_neg, projected_side = (
        activity_region_masks(track_eta, track_phi, leading, subleading)
    )
    projected_bridge = outside & projected_bridge
    projected_side = outside & projected_side

    sum_pt_outside = ak.to_numpy(ak.sum(track_pt[outside], axis=1))
    features["n_tracks_outside_jets"] = ak.to_numpy(ak.sum(outside, axis=1)).astype(np.float64)
    features["sum_track_pt_outside_jets"] = sum_pt_outside
    features["n_tracks_outside_pt2"] = ak.to_numpy(ak.sum(outside_hard, axis=1)).astype(np.float64)
    features["sum_track_pt_outside_pt2"] = ak.to_numpy(ak.sum(track_pt[outside_hard], axis=1))
    features["max_track_pt_outside_jets"] = scalar_max(track_pt, outside)
    features["n_tracks_transverse"] = ak.to_numpy(ak.sum(transverse, axis=1)).astype(np.float64)
    features["sum_track_pt_transverse"] = ak.to_numpy(ak.sum(track_pt[transverse], axis=1))
    features["n_tracks_interjet"] = ak.to_numpy(ak.sum(interjet, axis=1)).astype(np.float64)
    features["sum_track_pt_interjet"] = ak.to_numpy(ak.sum(track_pt[interjet], axis=1))
    features["n_tracks_projected_bridge"] = ak.to_numpy(
        ak.sum(projected_bridge, axis=1)
    ).astype(np.float64)
    features["sum_track_pt_projected_bridge"] = ak.to_numpy(
        ak.sum(track_pt[projected_bridge], axis=1)
    )
    features["n_tracks_projected_side"] = ak.to_numpy(
        ak.sum(projected_side, axis=1)
    ).astype(np.float64)
    features["sum_track_pt_projected_side"] = ak.to_numpy(
        ak.sum(track_pt[projected_side], axis=1)
    )
    features["n_tracks_outer_positive"] = ak.to_numpy(ak.sum(outer_pos, axis=1)).astype(np.float64)
    features["n_tracks_outer_negative"] = ak.to_numpy(ak.sum(outer_neg, axis=1)).astype(np.float64)
    features["sum_track_pt_outer_positive"] = ak.to_numpy(ak.sum(track_pt[outer_pos], axis=1))
    features["sum_track_pt_outer_negative"] = ak.to_numpy(ak.sum(track_pt[outer_neg], axis=1))

    pos_edge = eta_pos + JET_CONE_R
    neg_edge = eta_neg - JET_CONE_R
    nearest_pos = ak.to_numpy(
        ak.min(track_eta[outside & (track_eta > pos_edge[:, np.newaxis])], axis=1, mask_identity=False)
    )
    nearest_pos = np.where(np.isfinite(nearest_pos), nearest_pos, TRACKER_ETA)
    nearest_neg = ak.to_numpy(
        ak.max(track_eta[outside & (track_eta < neg_edge[:, np.newaxis])], axis=1, mask_identity=False)
    )
    nearest_neg = np.where(np.isfinite(nearest_neg), nearest_neg, -TRACKER_ETA)
    pos_gap = np.clip(np.minimum(nearest_pos, TRACKER_ETA) - pos_edge, 0.0, None)
    neg_gap = np.clip(neg_edge - np.maximum(nearest_neg, -TRACKER_ETA), 0.0, None)
    features["sum_gap_size"] = pos_gap + neg_gap
    features["minimum_gap_size"] = np.minimum(pos_gap, neg_gap)

    track_jets = cluster_track_jets(track_pt, track_eta, track_phi, outside)
    tj_ht = ak.to_numpy(ak.sum(track_jets.pt, axis=1))
    leading_tj_pt = scalar_max(track_jets.pt, track_jets.pt > 0)
    _tj_bridge, tj_outer_pos, tj_outer_neg, _tj_side = activity_region_masks(
        track_jets.eta, track_jets.phi, leading, subleading
    )
    tj_outer = tj_outer_pos | tj_outer_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        features["track_clumping_fraction"] = np.where(
            sum_pt_outside > 0, tj_ht / sum_pt_outside, np.nan
        )
        features["leading_track_jet_fraction"] = np.where(
            sum_pt_outside > 0, leading_tj_pt / sum_pt_outside, np.nan
        )
    features["n_extra_track_jets"] = ak.to_numpy(ak.num(track_jets.pt)).astype(np.float64)
    features["n_outer_track_jets"] = ak.to_numpy(ak.sum(tj_outer, axis=1)).astype(np.float64)
    features["leading_outer_track_jet_pt"] = scalar_max(track_jets.pt, tj_outer)
    features["interjet_bridge_asymmetry"] = bridge_asymmetry(
        track_pt, track_eta, track_phi, interjet, leading, subleading
    )
    features["interjet_bridge_asymmetry_projected"] = projected_bridge_asymmetry(
        track_pt, track_eta, track_phi, outside, leading, subleading
    )

    tj_dr1 = np.hypot(
        track_jets.eta - leading.eta[:, np.newaxis],
        wrap_phi(track_jets.phi - leading.phi[:, np.newaxis]),
    )
    tj_dr2 = np.hypot(
        track_jets.eta - subleading.eta[:, np.newaxis],
        wrap_phi(track_jets.phi - subleading.phi[:, np.newaxis]),
    )
    recovered = track_jets[np.minimum(tj_dr1, tj_dr2) < FSR_RECOVERY_MAX_R]
    total_px = ak.to_numpy(dijet.px) + ak.to_numpy(ak.sum(recovered.px, axis=1))
    total_py = ak.to_numpy(dijet.py) + ak.to_numpy(ak.sum(recovered.py, axis=1))
    total_pz = ak.to_numpy(dijet.pz) + ak.to_numpy(ak.sum(recovered.pz, axis=1))
    total_e = ak.to_numpy(dijet.E) + ak.to_numpy(ak.sum(recovered.E, axis=1))
    mass_sq = total_e**2 - total_px**2 - total_py**2 - total_pz**2
    features["dijet_mass_fsr"] = np.sqrt(np.clip(mass_sq, 0.0, None))
    features["fsr_recovered_pt"] = ak.to_numpy(ak.sum(recovered.pt, axis=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        eta_centroid = ak.to_numpy(ak.sum((track_pt * track_eta)[outside], axis=1)) / sum_pt_outside
        eta_var = (
            ak.to_numpy(
                ak.sum((track_pt * (track_eta - eta_centroid[:, np.newaxis]) ** 2)[outside], axis=1)
            )
            / sum_pt_outside
        )
    features["eta_rms_outside"] = np.sqrt(np.clip(eta_var, 0.0, None))

    lepton_pt = ak.concatenate(
        [event_slice("MuonLoose", "PT"), event_slice("Electron", "PT")], axis=1
    )
    lepton_eta = ak.concatenate(
        [event_slice("MuonLoose", "Eta"), event_slice("Electron", "Eta")], axis=1
    )
    lepton_phi = ak.concatenate(
        [event_slice("MuonLoose", "Phi"), event_slice("Electron", "Phi")], axis=1
    )
    lep_dr1 = np.hypot(
        lepton_eta - leading.eta[:, np.newaxis],
        wrap_phi(lepton_phi - leading.phi[:, np.newaxis]),
    )
    lep_dr2 = np.hypot(
        lepton_eta - subleading.eta[:, np.newaxis],
        wrap_phi(lepton_phi - subleading.phi[:, np.newaxis]),
    )
    in_jet_lepton = (
        (np.minimum(lep_dr1, lep_dr2) < JET_CONE_R) & (lepton_pt >= SOFT_LEPTON_MIN_PT)
    )
    features["n_soft_leptons_in_jets"] = ak.to_numpy(ak.sum(in_jet_lepton, axis=1)).astype(np.float64)
    features["max_in_jet_lepton_pt"] = scalar_max(lepton_pt, in_jet_lepton)

    met = ak.to_numpy(ak.firsts(event_slice("PuppiMissingET", "MET")))
    met_phi = ak.to_numpy(ak.firsts(event_slice("PuppiMissingET", "Phi")))
    features["puppi_met"] = met
    features["met_min_dphi_jet"] = np.minimum(
        np.abs(wrap_phi(met_phi - ak.to_numpy(leading.phi))),
        np.abs(wrap_phi(met_phi - ak.to_numpy(subleading.phi))),
    )
    features["n_vertices"] = ak.to_numpy(
        ak.num(arrays[branch_name("Vertex", "Z")][has_two][keep_ak])
    ).astype(np.float64)

    matrix = np.asarray(
        np.column_stack([features[name] for name in FEATURE_NAMES]),
        dtype=np.float32,
    )
    return {
        "matrix": matrix,
        "event_indices": event_indices,
        "dijet_mass": mjj,
        "dijet_rapidity": features["dijet_rapidity"],
        "n_generated": n_generated,
        "cutflow": {
            "two_jets": int(np.sum(has_two)),
            "delta_phi": delta_phi_count,
            "dijet_mass": int(np.sum(keep)),
        },
        "truth_matched": selected_truth_matched,
        "truth_is_leading": selected_truth_is_leading,
        "match_dr1": selected_match_dr[:, 0],
        "match_dr2": selected_match_dr[:, 1],
        "correction_valid": correction_valid[event_indices],
        "correction_candidate_events": int(np.sum(correction_candidate)),
        "correction_invalid_events": int(
            np.sum(correction_candidate & ~correction_diagnostic_valid)
        ),
        "parton": parton_values,
    }
