import argparse
import glob
import math
import os

import numpy as np

S_GEV2 = 14000.0**2
C_CM_PER_PS = 2.99792458e-2
F_COLL_AVG_HZ = 31.6e6

STATION_XI = {
    "192": (0.08, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.014, 0.0263),
    "420": (0.00325, 0.0116),
}

PROTON_FIELDS = (
    "bx_id",
    "interaction_id",
    "proton_idx",
    "side",
    "px",
    "py",
    "pz",
    "E",
    "m",
    "pt",
    "xi",
)

TRACK_FIELDS = (
    "trk_bx_id",
    "trk_interaction_id",
    "trk_pt",
    "trk_eta",
    "trk_idx",
    "trk_pdg_id",
    "trk_charge",
    "trk_px",
    "trk_py",
    "trk_pz",
    "trk_E",
)

def input_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not files:
            raise RuntimeError(f"No .npz files found in directory {path}")
        return files
    return [path]

def require_fields(data, fields, filename):
    missing = [name for name in fields if name not in data.files]
    if missing:
        if any(name.startswith("trk_") for name in missing):
            raise RuntimeError(
                f"Required track (charged particle) arrays missing in {filename}: {', '.join(missing)}. "
            )
        raise RuntimeError(f"Required arrays missing in {filename}: {', '.join(missing)}")

def encoded_keys(bx, interaction, base):
    return np.asarray(bx, dtype=np.int64) * base + np.asarray(interaction, dtype=np.int64)

def interaction_universe_from_file(data, filename):
    if all(name in data.files for name in ("mu_per_bx", "bx_offset", "n_bx")):
        mu_per_bx = np.asarray(data["mu_per_bx"], dtype=np.int64)
        bx_offset = int(np.asarray(data["bx_offset"]).item())
        n_bx = int(np.asarray(data["n_bx"]).item())
        if mu_per_bx.size != n_bx:
            raise RuntimeError(
                f"mu_per_bx length ({mu_per_bx.size}) does not match n_bx ({n_bx}) in {filename}"
            )

        bx_parts = []
        int_parts = []
        for local_bx, n_interactions in enumerate(mu_per_bx):
            if n_interactions <= 0:
                continue
            bx = bx_offset + local_bx
            bx_parts.append(np.full(int(n_interactions), bx, dtype=np.int64))
            int_parts.append(np.arange(int(n_interactions), dtype=np.int64))

        if bx_parts:
            print(f"Constructed interaction universe from {filename} with {len(bx_parts)} BX and total {sum(len(part) for part in int_parts)} interactions")
            return np.column_stack((np.concatenate(bx_parts), np.concatenate(int_parts)))

    proton_keys = np.column_stack(
        (
            np.asarray(data["bx_id"], dtype=np.int64),
            np.asarray(data["interaction_id"], dtype=np.int64),
        )
    )
    track_keys = np.column_stack(
        (
            np.asarray(data["trk_bx_id"], dtype=np.int64),
            np.asarray(data["trk_interaction_id"], dtype=np.int64),
        )
    )
    keys = np.concatenate((proton_keys, track_keys), axis=0)
    if keys.size == 0:
        print(f"No proton or track interactions found in {filename}, returning empty universe")
        return np.empty((0, 2), dtype=np.int64)
    print(f"Constructed interaction universe from {filename} with {keys.shape[0]} total interactions (protons: {proton_keys.shape[0]}, tracks: {track_keys.shape[0]})")
    return np.unique(keys, axis=0)


def load_minbias(path, max_files=None, max_bx=None):
    files = input_files(path)
    if max_files is not None:
        files = files[:max_files]
    proton_parts = {name: [] for name in PROTON_FIELDS}
    track_parts = {name: [] for name in TRACK_FIELDS}
    universe_parts = []

    for filename in files:
        with np.load(filename) as data:
            require_fields(data, PROTON_FIELDS, filename)
            require_fields(data, TRACK_FIELDS, filename)

            for name in PROTON_FIELDS:
                proton_parts[name].append(np.asarray(data[name]))
            for name in TRACK_FIELDS:
                track_parts[name].append(np.asarray(data[name]))
            n_tracks_file = len(data["trk_pt"])
            universe_parts.append(interaction_universe_from_file(data, filename))

    protons = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in proton_parts.items()
    }
    tracks = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in track_parts.items()
    }

    if universe_parts:
        universe = np.unique(np.concatenate(universe_parts, axis=0), axis=0)
    else:
        universe = np.empty((0, 2), dtype=np.int64)

    if max_bx is not None:
        universe = universe[universe[:, 0] < max_bx] 
        protons = {name: arr[protons["bx_id"] < max_bx] for name, arr in protons.items()}
        tracks = {name: arr[tracks["trk_bx_id"] < max_bx] for name, arr in tracks.items()}

    return files, protons, tracks, universe


def build_interaction_model(universe, protons, tracks, args, verbosity=0):
    if universe.size == 0:
        raise RuntimeError("No interactions found in input.")

    max_interaction = int(np.max(universe[:, 1]))
    if protons["interaction_id"].size:
        max_interaction = max(max_interaction, int(np.max(protons["interaction_id"])))
    if tracks["trk_interaction_id"].size:
        max_interaction = max(max_interaction, int(np.max(tracks["trk_interaction_id"])))
    key_base = max_interaction + 1

    universe_codes = encoded_keys(universe[:, 0], universe[:, 1], key_base)
    order = np.argsort(universe_codes, kind="mergesort")
    universe = universe[order]
    universe_codes = universe_codes[order]

    rng = np.random.default_rng(args.seed)
    z = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm, size=universe.shape[0])
    t = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm / C_CM_PER_PS, size=universe.shape[0])

    sum_pt2 = np.zeros(universe.shape[0], dtype=np.float64)
    n_tracks = np.zeros(universe.shape[0], dtype=np.int32)
    
    n_protons = np.zeros(universe.shape[0], dtype=np.int32)
    if protons["interaction_id"].size:
        proton_codes = encoded_keys(protons["bx_id"], protons["interaction_id"], key_base)
        pos = np.searchsorted(universe_codes, proton_codes)
        valid = pos < universe_codes.size
        valid[valid] = universe_codes[pos[valid]] == proton_codes[valid]
        pos = pos[valid]
        np.add.at(n_protons, pos, 1)

        if verbosity > 2:
            print("Number of protons in input:", protons["interaction_id"].size)
            print("Number of protons matched to universe interactions:", pos.size)
            print("Number of valid matches:", valid.sum())

    track_mask = (
        (tracks["trk_pt"] > args.trk_pt_min)
        & (np.abs(tracks["trk_eta"]) < args.trk_eta_max)
    )

    if np.any(track_mask):
        track_codes = encoded_keys(
            tracks["trk_bx_id"][track_mask],
            tracks["trk_interaction_id"][track_mask],
            key_base,
        )
        pos = np.searchsorted(universe_codes, track_codes)
        valid = pos < universe_codes.size
        valid[valid] = universe_codes[pos[valid]] == track_codes[valid]
        pos = pos[valid]
        pt = np.asarray(tracks["trk_pt"][track_mask], dtype=np.float64)[valid]
        np.add.at(sum_pt2, pos, pt * pt)
        np.add.at(n_tracks, pos, 1)

        if verbosity > 2:
            print("Number of track interactions:", tracks["trk_interaction_id"].size)
            print("Number of tracks matched to universe interactions:", pos.size)
            print("Number of valid matches:", valid.sum())

    if verbosity > 2:
        print("Number of interactions in universe:", universe.shape[0])
        print("Number of tracks before cuts:", tracks["trk_pt"].size)
        print("Number of tracks after pT and eta cuts:", np.sum(track_mask))

    return {
        "universe": universe,
        "codes": universe_codes,
        "key_base": key_base,
        "z": z,
        "t": t,
        "sum_pt2": sum_pt2,
        "n_tracks": n_tracks,
        "n_protons": n_protons,
    }

def get_interaction_info(model, bx, interaction):
    code = encoded_keys(bx, interaction, model["key_base"])
    idx = np.searchsorted(model["codes"], code)
    if idx >= model["codes"].size or model["codes"][idx] != code:
        return None
    return interaction_info_from_index(model, idx)

def interaction_info_from_index(model, idx):
    return {
        "z": model["z"][idx],
        "t": model["t"][idx],
        "sum_pt2": model["sum_pt2"][idx],
        "n_tracks": model["n_tracks"][idx],
        "n_protons": model["n_protons"][idx],
    }

def get_interaction_tracks(model, tracks, bx, interaction):
    track_mask = (tracks["trk_bx_id"] == bx) & (tracks["trk_interaction_id"] == interaction)
    interaction_tracks = {name: arr[track_mask] for name, arr in tracks.items()}
    return interaction_tracks

def get_interaction_protons(model, protons, bx, interaction):
    proton_mask = (protons["bx_id"] == bx) & (protons["interaction_id"] == interaction)
    interaction_protons = {name: arr[proton_mask] for name, arr in protons.items()}
    return interaction_protons

def get_bx_protons(model, protons, bx):
    proton_mask = protons["bx_id"] == bx
    bx_protons = {name: arr[proton_mask] for name, arr in protons.items()}
    return bx_protons

def group_indices_by_bx(bx_id):
    order = np.argsort(bx_id, kind="mergesort")
    bx_sorted = bx_id[order]
    unique_bx, starts, counts = np.unique(bx_sorted, return_index=True, return_counts=True)
    groups = [order[start:start + count] for start, count in zip(starts, counts)]
    return unique_bx, groups

def proton_interaction_indices(protons, model):
    codes = encoded_keys(protons["bx_id"], protons["interaction_id"], model["key_base"])
    pos = np.searchsorted(model["codes"], codes)
    valid = pos < model["codes"].size
    valid[valid] = model["codes"][pos[valid]] == codes[valid]
    if not np.all(valid):
        raise RuntimeError("Some proton (bx_id, interaction_id) keys are missing from the interaction universe.")
    return pos

def print_bx(model, protons, tracks, bx=None):
    bx_to_print = [bx] if bx is not None else np.unique(model["universe"][:, 0])
    for bx in bx_to_print:
        interactions_in_bx = model["universe"][model["universe"][:, 0] == bx][:, 1]
        if interactions_in_bx.size == 0:
            print(f"BX {bx}: No interactions found to print")
            continue
        for interaction in interactions_in_bx:
            info = get_interaction_info(model, bx, interaction)
            if info is None:
                continue
            z = info["z"]
            t = info["t"]
            sum_pt2 = info["sum_pt2"]
            n_tracks = info["n_tracks"]
            n_protons = info["n_protons"]
            print(f"Interaction (bx={bx}, id={interaction}): z={z:.2f} cm, t={t:.2f} ps, sum_pt2={sum_pt2:.2f} GeV^2, n_tracks={n_tracks}, n_protons={n_protons}")
            interaction_tracks = get_interaction_tracks(model, tracks, bx, interaction)
            interaction_protons = get_interaction_protons(model, protons, bx, interaction)
            print(f"  Tracks (after cuts): {n_tracks}")
            if n_tracks > 0:
                for i in range(len(interaction_tracks["trk_pt"])):
                    print(f"    Track {i}: pt={interaction_tracks['trk_pt'][i]:.2f} GeV, eta={interaction_tracks['trk_eta'][i]:.2f}, pdg_id={interaction_tracks['trk_pdg_id'][i]}, charge={interaction_tracks['trk_charge'][i]}")
            print(f"  Protons: {n_protons}")
            if n_protons > 0:
                for i in range(len(interaction_protons["pt"])):
                    xi = interaction_protons["xi"][i]
                    print(f"    Proton {i}: side={interaction_protons['side'][i]}, pt={interaction_protons['pt'][i]:.2f} GeV, xi={xi:.4f}, station_hits={station_hits(xi)}")


def get_primary_vertex(model, bx):
    idx_bx = np.where(model["universe"][:, 0] == bx)[0]
    if idx_bx.size == 0:
        return None
    idx = idx_bx[np.argmax(model["sum_pt2"][idx_bx])]
    n_tracks = model["n_tracks"][idx]
    if n_tracks == 0:
        return None
    return primary_vertex_from_index(model, idx)

def primary_vertex_from_index(model, idx):
    pv = interaction_info_from_index(model, idx)
    pv["idx"] = idx
    pv["interaction_id"] = model["universe"][idx, 1]
    return pv

def primary_vertices_by_bx(model):
    bx_values = model["universe"][:, 0]
    unique_bx, starts, counts = np.unique(bx_values, return_index=True, return_counts=True)

    pv_by_bx = {}
    for bx, start, count in zip(unique_bx, starts, counts):
        idx_bx = np.arange(start, start + count)
        idx = idx_bx[np.argmax(model["sum_pt2"][idx_bx])]
        pv_by_bx[int(bx)] = None if model["n_tracks"][idx] == 0 else primary_vertex_from_index(model, idx)

    return unique_bx, pv_by_bx

def cartesian_pair_indices(left_idx, right_idx):
    n_left = int(left_idx.size)
    n_right = int(right_idx.size)
    if n_left == 0 or n_right == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.repeat(left_idx, n_right), np.tile(right_idx, n_left)

def pps_sigma_z_cm(pps_time_res_ps):
    return 0.5 * C_CM_PER_PS * math.sqrt(2.0) * pps_time_res_ps

def station_tags(xi):
    xi_192_min, xi_192_max = STATION_XI["192"]
    xi_213_min, xi_213_max = STATION_XI["213"]
    xi_220_min, xi_220_max = STATION_XI["220"]
    xi_420_min, xi_420_max = STATION_XI["420"]
    tag200 = (
        ((xi >= xi_192_min) & (xi < xi_192_max))
        | ((xi >= xi_213_min) & (xi < xi_213_max))
        | ((xi >= xi_220_min) & (xi < xi_220_max))
    )
    tag400 = (xi >= xi_420_min) & (xi < xi_420_max)
    return tag200, tag400

def station_hits(xi):
    hits = [
        station
        for station, (xi_min, xi_max) in STATION_XI.items()
        if xi >= xi_min and xi < xi_max
    ]
    return ",".join(hits) if hits else "none"

def smear_xi(xi, xi_res, rng):
    if xi_res <= 0.0 or xi.size == 0:
        return xi
    return xi + rng.normal(loc=0.0, scale=xi_res, size=xi.size)

def analyze_bx(bx, model, bx_protons, bx_proton_model_idx, pv, constants, args, protons, tracks):
    rng = np.random.default_rng(args.seed + bx * 1000)
    xi_rng = np.random.default_rng(args.seed + bx * 1000 + 2)
    if pv is None:
        if args.verbose > 1:
            print(f"BX {bx}: No PV found")
    else:
        pv_z, pv_t = pv["z"], pv["t"]
        pv_z_reco = pv_z + rng.normal(loc=0.0, scale=args.pv_z_res_cm)
        pv_t_reco = pv_t + rng.normal(loc=0.0, scale=args.pv_t_res_ps) if args.pv_t_res_ps is not None else None
        if args.verbose > 1:
            print(f"BX {bx}: Primary vertex at z={pv_z:.2f} cm, t={pv_t:.2f} ps with n_tracks={pv['n_tracks']} and sum_pt2={pv['sum_pt2']:.2f} GeV^2")
            print(f"  Reco PV: z={pv_z_reco:.2f} cm, t={pv_t_reco} ps", end="")
    
    n_protons = len(bx_protons["pt"])
    if args.verbose > 1:
        print(f"BX {bx}: Found {n_protons} protons")
    
    pps_sigma_z_vertex = constants["pps_sigma_z_vertex"]
    zcut = constants["zcut"]
    tcut = constants["tcut"]

    bx_p_xi_nominal = np.asarray(bx_protons["xi"], dtype=np.float64)
    bx_p_xi = smear_xi(bx_p_xi_nominal, args.xi_res, xi_rng)
    bx_p_side = np.asarray(bx_protons["side"], dtype=np.int32)
    bx_p_tag200, bx_p_tag400 = station_tags(bx_p_xi_nominal)
    bx_p_tag_any = bx_p_tag200 | bx_p_tag400
    bx_p_interactions = bx_protons["interaction_id"]

    bx_pL = np.where((bx_p_side == -1) & bx_p_tag_any)[0]
    bx_pR = np.where((bx_p_side == +1) & bx_p_tag_any)[0]
    if pv is not None :
        bx_p_pv = (bx_p_interactions == pv["interaction_id"])
        bx_p_left_is_pv = np.where((bx_p_side == -1) & bx_p_tag_any & bx_p_pv)[0]
        bx_p_right_is_pv = np.where((bx_p_side == +1) & bx_p_tag_any & bx_p_pv)[0]
        if args.verbose > 1:
            print(f"BX {bx}: Among tagged protons, {len(bx_p_left_is_pv)} on left and {len(bx_p_right_is_pv)} on right are from PV interaction")

    bx_pL_400 = bx_pL[bx_p_tag400[bx_pL]]
    bx_pL_not400 = bx_pL[~bx_p_tag400[bx_pL]]
    bx_pR_400 = bx_pR[bx_p_tag400[bx_pR]]
    iL_a, iR_a = cartesian_pair_indices(bx_pL_400, bx_pR)
    iL_b, iR_b = cartesian_pair_indices(bx_pL_not400, bx_pR_400)
    iL = np.concatenate((iL_a, iL_b)) if iL_b.size else iL_a
    iR = np.concatenate((iR_a, iR_b)) if iR_b.size else iR_a
    bx_pp_mass = np.sqrt(bx_p_xi[iL] * bx_p_xi[iR] * args.s)

    if args.mass_window is not None and args.mass_window >= 0.0:
        M_H_GEV = 125.0
        keep = (bx_pp_mass >= (M_H_GEV - args.mass_window)) & (bx_pp_mass <= (M_H_GEV + args.mass_window))
    else:
        keep = np.ones(iL.size, dtype=bool)

    iL = iL[keep]
    iR = iR[keep]
    bx_pp_mass = bx_pp_mass[keep]
    bx_pp = np.column_stack((iL, iR)) if iL.size else np.empty((0, 2), dtype=np.int64)

    if args.verbose > 1:
        print(f"BX {bx}: Found {len(bx_pp)} proton pairs with at least one tagged in 420 station")
        if len(bx_pp) > 0:
            print(f"  Masses of proton pairs: {bx_pp_mass}")
    
    bx_pp_vertex_reco = np.zeros((len(bx_pp), 2), dtype=np.float64)
    if len(bx_pp) > 0:
        bx_p_z = model["z"][bx_proton_model_idx]
        bx_p_t = model["t"][bx_proton_model_idx]
        pL_t_truth = bx_p_t[iL] + bx_p_z[iL] / C_CM_PER_PS
        pR_t_truth = bx_p_t[iR] - bx_p_z[iR] / C_CM_PER_PS
        pL_t_reco = pL_t_truth + rng.normal(loc=0.0, scale=args.pps_time_res_ps, size=len(bx_pp))
        pR_t_reco = pR_t_truth + rng.normal(loc=0.0, scale=args.pps_time_res_ps, size=len(bx_pp))
        bx_pp_vertex_reco[:, 0] = 0.5 * (pL_t_reco - pR_t_reco) * C_CM_PER_PS
        bx_pp_vertex_reco[:, 1] = 0.5 * (pL_t_reco + pR_t_reco)

    bx_pp_z_compatible = np.abs(bx_pp_vertex_reco[:, 0] - pv_z_reco) <= zcut if pv is not None else np.zeros(len(bx_pp), dtype=bool)
    if args.pv_t_res_ps is not None:
        bx_pp_t_compatible = np.abs(bx_pp_vertex_reco[:, 1] - pv_t_reco) <= tcut if pv is not None else np.zeros(len(bx_pp), dtype=bool)
    else :
        bx_pp_t_compatible = np.abs(bx_pp_vertex_reco[:, 1]) <= tcut if pv is not None else np.zeros(len(bx_pp), dtype=bool)
    bx_pp_vtx_compatible = bx_pp_z_compatible & bx_pp_t_compatible if pv is not None else np.zeros(len(bx_pp), dtype=bool)
    bx_pp_is_pv = (
        (bx_p_interactions[iL] == pv["interaction_id"]).astype(np.int32)
        + (bx_p_interactions[iR] == pv["interaction_id"]).astype(np.int32)
    ) if pv is not None else np.zeros(len(bx_pp), dtype=np.int32)
    counters = {
        "n_protons": n_protons,
        "n_pp": len(bx_pp),
        "n_pp_z_compatible": np.sum(bx_pp_z_compatible),
        "n_pp_t_compatible": np.sum(bx_pp_t_compatible),
        "n_pp_vtx_compatible": np.sum(bx_pp_vtx_compatible),
        "n_pp_vtx_compatible_one_pv": np.sum(bx_pp_vtx_compatible & (bx_pp_is_pv == 1)),
        "n_pp_vtx_compatible_both_pv": np.sum(bx_pp_vtx_compatible & (bx_pp_is_pv == 2)),
    }

    if args.verbose > 1:
        print(f"BX {bx}: PPS vertex compatibility results:")
        print(f"  Total proton pairs: {counters['n_pp']}")
        print(f"  Pairs with z compatible with PV: {counters['n_pp_z_compatible']}")
        print(f"  Pairs with t compatible with PV: {counters['n_pp_t_compatible']}")
        print(f"  Pairs with both z and t compatible with PV: {counters['n_pp_vtx_compatible']}")
        print(f"  Pairs with one PV compatible: {counters['n_pp_vtx_compatible_one_pv']}")
        print(f"  Pairs with both PV compatible: {counters['n_pp_vtx_compatible_both_pv']}")
        if args.verbose > 1 and len(bx_pp) > 0:
            for i, (iL, iR) in enumerate(bx_pp):
                print(f"  Pair {i}: xi_L={bx_p_xi[iL]:.4f}, xi_R={bx_p_xi[iR]:.4f}, mass={bx_pp_mass[i]:.2f} GeV, vertex z={bx_pp_vertex_reco[i, 0]:.2f} cm, vertex t={bx_pp_vertex_reco[i, 1]:.2f} ps, z compatible: {bx_pp_z_compatible[i]}, t compatible: {bx_pp_t_compatible[i]}")
        if args.verbose > 2:
            print_bx(model, protons, tracks, bx)

    return {
        "counters": counters,
        "pps_sigma_z_vertex": pps_sigma_z_vertex,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze minbias proton pairs and vertex compatibility"
    )
    parser.add_argument("-i", "--input", required=True, help="Input minbias NPZ file or directory of NPZ files")
    parser.add_argument("--max-bx", type=int, default=None, help="Optional maximum number of BX to process")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of files to process")
    parser.add_argument("--min-bx-pair-candidates", type=int, default=None, help="Trigger-level requirement: BX must have at least this many accepted PPS+mass proton pair candidates")
    parser.add_argument("--trk-pt-min", type=float, default=2.0, help="Track (charged particles) proxy pT cut in GeV (default: 0.5)")
    parser.add_argument("--trk-eta-max", type=float, default=2.4, help="Track (charged particles) proxy |eta| cut (default: 2.5)")
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7, help="Gaussian beam spot sigma z in cm")
    parser.add_argument("--pps-time-res-ps", type=float, default=10.0, help="Single-arm timing resolution in ps")
    parser.add_argument("--pv-z-res-cm", type=float, default=0.1, help="Primary vertex z resolution in cm (central detector)")
    parser.add_argument("--pv-t-res-ps", type=float, default=None, help="Primary vertex timing resolution in ps (central detector)")
    parser.add_argument("--nsigma", type=float, default=2.0, help="Compatibility cut in units of sigma_dz")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for vertex and timing smearing")
    parser.add_argument("--mass-window", type=float, default=6, help="Mass half-width in GeV around 125")
    parser.add_argument("--xi-res", type=float, default=0.0003, help="Gaussian fractional xi resolution sigma; 0 disables smearing")
    parser.add_argument("--s", type=float, default=S_GEV2, help="s value for M=sqrt(xi1*xi2*s)")
    parser.add_argument("--no-dz-smear", dest="smear_dz", action="store_false", help="Disable timing smearing")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Increase output verbosity")
    parser.set_defaults(smear_dz=True)
    
    args = parser.parse_args()

    if args.max_bx is not None and args.max_bx <= 0:
        raise RuntimeError("--max-bx must be > 0")
    if args.min_bx_pair_candidates is not None and args.min_bx_pair_candidates < 0:
        raise RuntimeError("--min-bx-pair-candidates must be >= 0")
    if args.trk_pt_min < 0.0:
        raise RuntimeError("--trk-pt-min must be >= 0")
    if args.trk_eta_max <= 0.0:
        raise RuntimeError("--trk-eta-max must be > 0")
    if args.beam_sigma_z_cm < 0.0:
        raise RuntimeError("--beam-sigma-z-cm must be >= 0")
    if args.pps_time_res_ps <= 0.0:
        raise RuntimeError("--pps-time-res-ps must be > 0")
    if args.pv_z_res_cm < 0.0:
        raise RuntimeError("--pv-z-res-cm must be >= 0")
    if args.pv_t_res_ps is not None and args.pv_t_res_ps <= 0.0:
        raise RuntimeError("--pv-t-res-ps must be > 0")
    if args.nsigma <= 0.0:
        raise RuntimeError("--nsigma must be > 0")
    if args.mass_window is not None and args.mass_window < 0.0:
        raise RuntimeError("--mass-window must be >= 0")
    if args.xi_res < 0.0:
        raise RuntimeError("--xi-res must be >= 0")

    files, protons, tracks, universe = load_minbias(args.input, max_files=args.max_files, max_bx=args.max_bx)
    model = build_interaction_model(universe, protons, tracks, args, verbosity=args.verbose)
    
    selected_bx, pv_by_bx = primary_vertices_by_bx(model)
    n_bx = len(selected_bx)
    print(f"Processing {n_bx} unique BX from {len(files)} files")
    proton_model_idx = proton_interaction_indices(protons, model)
    unique_pbx, proton_groups = group_indices_by_bx(protons["bx_id"])
    proton_group_by_bx = {
        int(bx): group for bx, group in zip(unique_pbx, proton_groups)
    }
    pps_sigma_z_vertex = pps_sigma_z_cm(args.pps_time_res_ps)
    pps_sigma_t_vertex = pps_sigma_z_vertex / C_CM_PER_PS
    constants = {
        "pps_sigma_z_vertex": pps_sigma_z_vertex,
        "zcut": args.nsigma * math.sqrt(pps_sigma_z_vertex**2 + args.pv_z_res_cm**2),
        "tcut": (
            args.nsigma * math.sqrt(pps_sigma_t_vertex**2 + args.pv_t_res_ps**2)
            if args.pv_t_res_ps is not None
            else args.nsigma * args.beam_sigma_z_cm / C_CM_PER_PS
        ),
    }
    counters = {
        "n_protons": 0,
        "n_pp": 0,
        "n_pp_z_compatible": 0,
        "n_pp_t_compatible": 0,
        "n_pp_vtx_compatible": 0,
        "n_pp_vtx_compatible_one_pv": 0,
        "n_pp_vtx_compatible_both_pv": 0,
    }
    n_pair_requirement_failed = 0
    for bx in selected_bx:
        bx = int(bx)
        if bx % 100 == 0:
            print(f"Analyzing BX {bx}...")
        idx_bx = proton_group_by_bx.get(bx)
        if idx_bx is None:
            idx_bx = np.empty(0, dtype=np.int64)
        bx_protons = {name: arr[idx_bx] for name, arr in protons.items()}
        bx_result = analyze_bx(
            bx,
            model,
            bx_protons,
            proton_model_idx[idx_bx],
            pv_by_bx[bx],
            constants,
            args,
            protons,
            tracks,
        )
        if (
            args.min_bx_pair_candidates is not None
            and bx_result["counters"]["n_pp"] < args.min_bx_pair_candidates
        ):
            n_pair_requirement_failed += 1
            continue
        for key in counters.keys():
            if bx_result["counters"][key] is not None:
                counters[key] += 1 if bx_result["counters"][key] > 0.1 else 0

    print(f"Summary of results across all processed BX ({n_bx}):")
    if args.min_bx_pair_candidates is not None:
        print(
            f"BX failing --min-bx-pair-candidates ({args.min_bx_pair_candidates}): "
            f"{n_pair_requirement_failed}"
        )
    print(f"Total protons: {counters['n_protons']}")
    print(f"Total proton pairs: {counters['n_pp']}")
    print(f"Proton pairs with z compatible with PV: {counters['n_pp_z_compatible']}")
    print(f"Proton pairs with t compatible with PV: {counters['n_pp_t_compatible']}")
    print(f"Proton pairs with both z and t compatible with PV: {counters['n_pp_vtx_compatible']}")
    print(f"Proton pairs with one PV compatible: {counters['n_pp_vtx_compatible_one_pv']}")
    print(f"Proton pairs with both PV compatible: {counters['n_pp_vtx_compatible_both_pv']}")

    print("Expected rates for 31.6 MHz collision rate:")
    print(f"Expected rate for single protons: {counters['n_protons'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} +/- {math.sqrt(counters['n_protons']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")
    print(f"Expected rate for proton pairs: {counters['n_pp'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} +/- {math.sqrt(counters['n_pp']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")
    print(f"Expected rate for proton pairs with z compatible with PV: {counters['n_pp_z_compatible'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} +/- {math.sqrt(counters['n_pp_z_compatible']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")
    print(f"Expected rate for proton pairs with t compatible with PV: {counters['n_pp_t_compatible'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} +/- {math.sqrt(counters['n_pp_t_compatible']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")
    print(f"Expected rate for proton pairs with both z and t compatible with PV: {counters['n_pp_vtx_compatible'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} +/- {math.sqrt(counters['n_pp_vtx_compatible']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")
    print(f"Expected rate for proton pairs with one PV compatible: {counters['n_pp_vtx_compatible_one_pv'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} +/- {math.sqrt(counters['n_pp_vtx_compatible_one_pv']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")
    print(f"Expected rate for proton pairs with both PV compatible: {counters['n_pp_vtx_compatible_both_pv'] * F_COLL_AVG_HZ / n_bx / 1000:.2f} \pm {math.sqrt(counters['n_pp_vtx_compatible_both_pv']) * F_COLL_AVG_HZ / n_bx / 1000:.2f} kHz")

    #all_bx, pv_by_bx = primary_vertices_by_bx(model)
    #selected_bx = all_bx
    #if args.max_bx is not None:
    #    selected_bx = selected_bx[:args.max_bx]

    # if args.print_track_candidates:
    #     print_track_candidates(tracks, model, pv_by_bx, selected_bx, args)

    # result = analyze(protons, model, pv_by_bx, selected_bx, args)
    # print_summary(files, args, result)

if __name__ == "__main__":
    main()
