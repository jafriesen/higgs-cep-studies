#!/usr/bin/env python3
"""
Analyze minbias-only proton pairs against a charged-track primary vertex.

The input must be a minbias NPZ produced by generate_minbias_protons.py with
--store-tracks. For each BX, the script assigns random vertices to all
interactions, chooses the interaction with the largest selected-track sum(pT^2)
as the primary vertex, and compares tagged minbias proton pairs to that PV.
"""

import argparse
import glob
import math
import os

import numpy as np


C_CM_PER_PS = 2.99792458e-2
POT_DISTANCE_CM = 420.0 * 100.0
M_H_GEV = 125.0
S_GEV2_DEFAULT = 14000.0**2

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
    "pt",
    "xi",
)

TRACK_FIELDS = (
    "trk_bx_id",
    "trk_interaction_id",
    "trk_pt",
    "trk_eta",
)

OPTIONAL_TRACK_FIELDS = (
    "trk_idx",
    "trk_pdg_id",
    "trk_charge",
    "trk_px",
    "trk_py",
    "trk_pz",
    "trk_E",
)


def safe_fraction(num, den):
    return (float(num) / float(den)) if den else 0.0


def sigma_dz_cm(single_arm_time_res_ps):
    return 0.5 * C_CM_PER_PS * math.sqrt(2.0) * single_arm_time_res_ps


def print_count(label, count, total):
    print(f"{label}: {count} ({safe_fraction(count, total):.6f})")


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
                f"Required track arrays missing in {filename}: {', '.join(missing)}. "
                "Regenerate the minbias NPZ with --store-tracks."
            )
        raise RuntimeError(f"Required arrays missing in {filename}: {', '.join(missing)}")


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
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(keys, axis=0)


def load_minbias(path, max_files=None):
    files = input_files(path)
    if max_files is not None:
        files = files[:max_files]
    proton_parts = {name: [] for name in PROTON_FIELDS}
    track_parts = {name: [] for name in TRACK_FIELDS}
    optional_track_parts = {name: [] for name in OPTIONAL_TRACK_FIELDS}
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
            for name in OPTIONAL_TRACK_FIELDS:
                if name in data.files:
                    optional_track_parts[name].append(np.asarray(data[name]))
                elif name in ("trk_idx", "trk_pdg_id", "trk_charge"):
                    optional_track_parts[name].append(np.full(n_tracks_file, -1, dtype=np.int32))
                else:
                    optional_track_parts[name].append(np.full(n_tracks_file, np.nan, dtype=np.float32))
            universe_parts.append(interaction_universe_from_file(data, filename))

    protons = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in proton_parts.items()
    }
    tracks = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in track_parts.items()
    }
    for name, parts in optional_track_parts.items():
        if parts:
            tracks[name] = np.concatenate(parts)

    if universe_parts:
        universe = np.unique(np.concatenate(universe_parts, axis=0), axis=0)
    else:
        universe = np.empty((0, 2), dtype=np.int64)

    return files, protons, tracks, universe


def encoded_keys(bx, interaction, base):
    return np.asarray(bx, dtype=np.int64) * base + np.asarray(interaction, dtype=np.int64)


def build_interaction_model(universe, protons, tracks, args):
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

    rng = np.random.default_rng(args.smear_seed)
    z = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm, size=universe.shape[0])
    t = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm / C_CM_PER_PS, size=universe.shape[0])

    sum_pt2 = np.zeros(universe.shape[0], dtype=np.float64)
    n_tracks = np.zeros(universe.shape[0], dtype=np.int32)

    track_mask = (
        (tracks["trk_pt"] > args.track_pt_min)
        & (np.abs(tracks["trk_eta"]) < args.track_eta_max)
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

        if args.print_track_candidates:
            for track in tracks.keys():
                print(
                    f"Track array '{track}': {tracks[track].shape} entries, "
                    f"{np.sum(track_mask)} passing pT and eta cuts"
                )

    return {
        "universe": universe,
        "codes": universe_codes,
        "key_base": key_base,
        "z": z,
        "t": t,
        "sum_pt2": sum_pt2,
        "n_tracks": n_tracks,
    }


def proton_interaction_indices(protons, model):
    codes = encoded_keys(protons["bx_id"], protons["interaction_id"], model["key_base"])
    pos = np.searchsorted(model["codes"], codes)
    valid = pos < model["codes"].size
    valid[valid] = model["codes"][pos[valid]] == codes[valid]
    if not np.all(valid):
        raise RuntimeError("Some proton (bx_id, interaction_id) keys are missing from the interaction universe.")
    return pos


def primary_vertices_by_bx(model):
    universe = model["universe"]
    bx_values = universe[:, 0]
    unique_bx, starts, counts = np.unique(bx_values, return_index=True, return_counts=True)

    pv = {}
    for bx, start, count in zip(unique_bx, starts, counts):
        idx = np.arange(start, start + count)
        best_local = int(np.argmax(model["sum_pt2"][idx]))
        best_idx = int(idx[best_local])
        has_tracks = bool(model["sum_pt2"][best_idx] > 0.0)
        if not int(model["n_tracks"][best_idx]) > 0:
            print(f"Warning: No tracks above pT cut for PV in BX {bx}, interaction {model['universe'][best_idx, 1]}")
        pv[int(bx)] = {
            "idx": best_idx,
            "interaction_id": int(universe[best_idx, 1]),
            "z": float(model["z"][best_idx]),
            "t": float(model["t"][best_idx]),
            "sum_pt2": float(model["sum_pt2"][best_idx]),
            "n_tracks": int(model["n_tracks"][best_idx]),
            "has_tracks": has_tracks,
        }

    return unique_bx.astype(np.int64, copy=False), pv


def print_track_candidates(tracks, model, pv_by_bx, selected_bx, args):
    if args.print_track_candidate_bx:
        bx_to_print = [int(bx) for bx in args.print_track_candidate_bx]
    else:
        bx_to_print = [int(bx) for bx in selected_bx[:args.max_track_candidate_bx]]

    track_codes = encoded_keys(
        tracks["trk_bx_id"],
        tracks["trk_interaction_id"],
        model["key_base"],
    )
    track_pass = (
        (tracks["trk_pt"] > args.track_pt_min)
        & (np.abs(tracks["trk_eta"]) < args.track_eta_max)
    )

    print("")
    print("=== Track Candidate Debug ===")
    print(f"Track pT cut: > {args.track_pt_min} GeV")
    print(f"Track eta cut: abs(eta) < {args.track_eta_max}")
    print(f"BX printed: {', '.join(str(bx) for bx in bx_to_print) if bx_to_print else '(none)'}")

    for bx in bx_to_print:
        if bx not in pv_by_bx:
            print(f"\nBX {bx}: not present in interaction universe")
            continue

        pv = pv_by_bx[bx]
        interaction_indices = np.where(model["universe"][:, 0] == bx)[0]
        print("")
        print(
            f"BX {bx}: PV interaction={pv['interaction_id']} "
            f"sum_pt2={pv['sum_pt2']:.6g} n_selected_tracks={pv['n_tracks']} "
            f"has_tracks={pv['has_tracks']} z={pv['z']:.3f} cm"
        )

        for model_idx in interaction_indices:
            interaction_id = int(model["universe"][model_idx, 1])
            code = model["codes"][model_idx]
            idx_tracks = np.where(track_codes == code)[0]
            order = np.lexsort((-tracks["trk_pt"][idx_tracks], ~track_pass[idx_tracks]))
            idx_tracks = idx_tracks[order]
            is_pv = interaction_id == pv["interaction_id"]
            marker = " <== PV" if is_pv else ""
            n_all = int(idx_tracks.size)
            n_selected = int(model["n_tracks"][model_idx])
            sum_pt2 = float(model["sum_pt2"][model_idx])
            print(
                f"  interaction {interaction_id:4d}: "
                f"tracks={n_all:4d} selected={n_selected:4d} "
                f"sum_pt2={sum_pt2:10.5f}{marker}"
            )

            if n_all == 0:
                continue

            limit = args.max_tracks_per_interaction
            shown = idx_tracks[:limit]
            print(
                "      "
                f"{'trk_idx':>7} {'pdg':>7} {'q':>3} {'pt':>10} {'eta':>9} "
                f"{'pass':>5} {'px':>10} {'py':>10} {'pz':>10}"
            )
            for idx in shown:
                trk_idx = int(tracks["trk_idx"][idx]) if "trk_idx" in tracks else -1
                pdg_id = int(tracks["trk_pdg_id"][idx]) if "trk_pdg_id" in tracks else 0
                charge = int(tracks["trk_charge"][idx]) if "trk_charge" in tracks else 0
                px = float(tracks["trk_px"][idx]) if "trk_px" in tracks else float("nan")
                py = float(tracks["trk_py"][idx]) if "trk_py" in tracks else float("nan")
                pz = float(tracks["trk_pz"][idx]) if "trk_pz" in tracks else float("nan")
                print(
                    "      "
                    f"{trk_idx:7d} {pdg_id:7d} {charge:3d} "
                    f"{float(tracks['trk_pt'][idx]):10.4f} "
                    f"{float(tracks['trk_eta'][idx]):9.4f} "
                    f"{str(bool(track_pass[idx])):>5} "
                    f"{px:10.4f} {py:10.4f} {pz:10.4f}"
                )

            if n_all > limit:
                print(f"      ... {n_all - limit} more tracks not shown")


def group_indices_by_bx(bx_id):
    order = np.argsort(bx_id, kind="mergesort")
    bx_sorted = bx_id[order]
    unique_bx, starts, counts = np.unique(bx_sorted, return_index=True, return_counts=True)
    groups = [order[start:start + count] for start, count in zip(starts, counts)]
    return unique_bx, groups


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


def cartesian_pair_indices(left_idx, right_idx):
    n_left = int(left_idx.size)
    n_right = int(right_idx.size)
    if n_left == 0 or n_right == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.repeat(left_idx, n_right), np.tile(right_idx, n_left)


def init_counters():
    counters = {}
    for prefix in ("passing", "vertex", "pv_z", "pv_z_has_tracks", "pv_z_no_tracks"):
        counters[prefix] = 0
        for category in ("both_pv", "one_pv", "neither_pv"):
            counters[f"{prefix}_{category}"] = 0
    return counters


def update_category_flags(flags, prefix, category):
    flags[prefix] = True
    flags[f"{prefix}_{category}"] = True


def append_root_chunk(chunks, chunk):
    if chunks is None:
        return
    for name, values in chunk.items():
        chunks[name].append(values)


def analyze(protons, model, pv_by_bx, selected_bx, args):
    p_int_idx = proton_interaction_indices(protons, model)
    z_vertex = model["z"][p_int_idx]
    t_vertex = model["t"][p_int_idx]

    unique_pbx, proton_groups = group_indices_by_bx(protons["bx_id"])
    proton_group_by_bx = {
        int(bx): group for bx, group in zip(unique_pbx, proton_groups)
    }

    timing_sigma_dz = sigma_dz_cm(args.single_arm_time_res_ps)
    zcut_cm = args.nsigma * timing_sigma_dz
    smear_rng = np.random.default_rng(args.smear_seed + 1000003)

    counters = init_counters()
    n_track_supported_pv = 0
    n_pairs_total = 0

    int_branch_names = [
        "bx",
        "interaction_L",
        "interaction_R",
        "p_idx_L",
        "p_idx_R",
        "side_L",
        "side_R",
        "tag200_L",
        "tag200_R",
        "tag400_L",
        "tag400_R",
        "double_tag_420",
        "vtx_ok",
        "in_Hwin",
        "pv_interaction_id",
        "pv_n_tracks",
        "interaction_L_is_pv",
        "interaction_R_is_pv",
        "pv_z_compatible",
    ]
    float_branch_names = [
        "xi_L",
        "xi_R",
        "z_L",
        "z_R",
        "dz",
        "M",
        "ln_xi_ratio",
        "yX",
        "abs_yX",
        "pT_L",
        "pT_R",
        "t1_abs",
        "t2_abs",
        "t_sum",
        "t_diff_abs",
        "pt_bal",
        "pt_bal_ratio",
        "abs_dphi",
        "dphi_from_pi",
        "pv_z",
        "pv_t",
        "pv_sum_track_pt2",
        "z_reco",
        "t_reco",
    ]
    root_chunks = None
    if args.root_out:
        root_chunks = {name: [] for name in int_branch_names + float_branch_names}

    for ib, bx in enumerate(selected_bx):
        bx = int(bx)
        if ib % 1000 == 0:
            progress = 100.0 * ib / len(selected_bx) if len(selected_bx) else 100.0
            print(f"[BX {ib}/{len(selected_bx)}] {progress:.1f}% done, pairs stored: {n_pairs_total}", flush=True)

        pv = pv_by_bx[bx]
        if pv["has_tracks"]:
            n_track_supported_pv += 1

        flags = {name: False for name in counters}
        idx_bx = proton_group_by_bx.get(bx)
        if idx_bx is None or idx_bx.size == 0:
            for name, value in flags.items():
                if value:
                    counters[name] += 1
            continue

        xi_bx = np.asarray(protons["xi"][idx_bx], dtype=np.float64)
        side_bx = np.asarray(protons["side"][idx_bx], dtype=np.int32)
        tag200_bx, tag400_bx = station_tags(xi_bx)
        tag_any_bx = tag200_bx | tag400_bx

        left_tagged = np.where((side_bx == -1) & tag_any_bx)[0]
        right_tagged = np.where((side_bx == +1) & tag_any_bx)[0]
        left_tagged_400 = left_tagged[tag400_bx[left_tagged]]
        left_tagged_not400 = left_tagged[~tag400_bx[left_tagged]]
        right_tagged_400 = right_tagged[tag400_bx[right_tagged]]

        iL_a, iR_a = cartesian_pair_indices(left_tagged_400, right_tagged)
        iL_b, iR_b = cartesian_pair_indices(left_tagged_not400, right_tagged_400)
        iL = np.concatenate((iL_a, iL_b)) if iL_b.size else iL_a
        iR = np.concatenate((iR_a, iR_b)) if iR_b.size else iR_a
        if iL.size == 0:
            for name, value in flags.items():
                if value:
                    counters[name] += 1
            continue

        xi_L = xi_bx[iL]
        xi_R = xi_bx[iR]
        mass = np.sqrt(np.maximum(xi_L * xi_R * args.s, 0.0))
        if args.mass_window is not None and args.mass_window >= 0.0:
            in_mass_window = (
                (mass >= (M_H_GEV - args.mass_window))
                & (mass <= (M_H_GEV + args.mass_window))
            )
        else :
            in_mass_window = np.ones(iL.size, dtype=bool)
        keep = in_mass_window
        if not np.any(keep):
            for name, value in flags.items():
                if value:
                    counters[name] += 1
            continue

        iL = iL[keep]
        iR = iR[keep]
        xi_L = xi_L[keep]
        xi_R = xi_R[keep]
        mass = mass[keep]
        n_pairs = int(iL.size)
        n_pairs_total += n_pairs

        global_L = idx_bx[iL]
        global_R = idx_bx[iR]
        inter_L = np.asarray(protons["interaction_id"][global_L], dtype=np.int32)
        inter_R = np.asarray(protons["interaction_id"][global_R], dtype=np.int32)
        is_pv_L = inter_L == pv["interaction_id"]
        is_pv_R = inter_R == pv["interaction_id"]
        both_pv = is_pv_L & is_pv_R
        one_pv = np.logical_xor(is_pv_L, is_pv_R)
        neither_pv = ~(both_pv | one_pv)

        z_L = z_vertex[global_L].astype(np.float64, copy=False)
        z_R = z_vertex[global_R].astype(np.float64, copy=False)
        t_L0 = t_vertex[global_L].astype(np.float64, copy=False)
        t_R0 = t_vertex[global_R].astype(np.float64, copy=False)
        dz = z_L - z_R
        if args.smear_dz:
            dz_obs = dz + smear_rng.normal(loc=0.0, scale=timing_sigma_dz, size=n_pairs)
        else:
            dz_obs = dz
        vtx_ok = np.abs(dz_obs) <= zcut_cm

        t_left = t_L0 + (z_L + POT_DISTANCE_CM) / C_CM_PER_PS
        t_right = t_R0 + (-z_R + POT_DISTANCE_CM) / C_CM_PER_PS
        if args.smear_dz:
            t_left = t_left + smear_rng.normal(loc=0.0, scale=args.single_arm_time_res_ps, size=n_pairs)
            t_right = t_right + smear_rng.normal(loc=0.0, scale=args.single_arm_time_res_ps, size=n_pairs)
        z_reco = 0.5 * (t_left - t_right) * C_CM_PER_PS
        t_reco = 0.5 * (t_left + t_right) - POT_DISTANCE_CM / C_CM_PER_PS
        pv_z_compatible = np.abs(z_reco - pv["z"]) <= zcut_cm

        categories = (
            ("both_pv", both_pv),
            ("one_pv", one_pv),
            ("neither_pv", neither_pv),
        )
        flags["passing"] = True
        flags["passing_both_pv"] = flags["passing_both_pv"] or bool(np.any(both_pv))
        flags["passing_one_pv"] = flags["passing_one_pv"] or bool(np.any(one_pv))
        flags["passing_neither_pv"] = flags["passing_neither_pv"] or bool(np.any(neither_pv))
        flags["pv_z_has_tracks"] = flags["pv_z_has_tracks"] or (pv["has_tracks"] and bool(np.any(pv_z_compatible)))
        flags["pv_z_no_tracks"] = flags["pv_z_no_tracks"] or (not pv["has_tracks"] and bool(np.any(pv_z_compatible)))
        for category, mask in categories:
            if np.any(vtx_ok & mask):
                update_category_flags(flags, "vertex", category)
            if np.any(pv_z_compatible & mask):
                update_category_flags(flags, "pv_z", category)
            if pv["has_tracks"] and np.any(pv_z_compatible & mask):
                update_category_flags(flags, "pv_z_has_tracks", category)
            if (not pv["has_tracks"]) and np.any(pv_z_compatible & mask):
                update_category_flags(flags, "pv_z_no_tracks", category)


        pT_L = np.asarray(protons["pt"][global_L], dtype=np.float64)
        pT_R = np.asarray(protons["pt"][global_R], dtype=np.float64)
        t1_abs = pT_L * pT_L
        t2_abs = pT_R * pT_R
        t_sum = t1_abs + t2_abs
        t_diff_abs = np.abs(t1_abs - t2_abs)

        px_L = np.asarray(protons["px"][global_L], dtype=np.float64)
        py_L = np.asarray(protons["py"][global_L], dtype=np.float64)
        px_R = np.asarray(protons["px"][global_R], dtype=np.float64)
        py_R = np.asarray(protons["py"][global_R], dtype=np.float64)
        phi_L = np.arctan2(py_L, px_L)
        phi_R = np.arctan2(py_R, px_R)
        dphi = (phi_L - phi_R + math.pi) % (2.0 * math.pi) - math.pi
        abs_dphi = np.abs(dphi)
        pt_bal = np.hypot(px_L + px_R, py_L + py_R)
        denom_pt = pT_L + pT_R
        pt_bal_ratio = np.zeros(n_pairs, dtype=np.float64)
        np.divide(pt_bal, denom_pt, out=pt_bal_ratio, where=denom_pt > 0.0)
        dphi_from_pi = math.pi - abs_dphi

        ln_xi_ratio = np.zeros(n_pairs, dtype=np.float64)
        valid_xi = (xi_L > 0.0) & (xi_R > 0.0)
        ln_xi_ratio[valid_xi] = np.log(xi_L[valid_xi] / xi_R[valid_xi])
        yX = 0.5 * ln_xi_ratio

        append_root_chunk(
            root_chunks,
            {
                "bx": np.full(n_pairs, bx, dtype=np.int32),
                "interaction_L": inter_L.astype(np.int32, copy=False),
                "interaction_R": inter_R.astype(np.int32, copy=False),
                "p_idx_L": np.asarray(protons["proton_idx"][global_L], dtype=np.int32),
                "p_idx_R": np.asarray(protons["proton_idx"][global_R], dtype=np.int32),
                "side_L": np.asarray(protons["side"][global_L], dtype=np.int32),
                "side_R": np.asarray(protons["side"][global_R], dtype=np.int32),
                "tag200_L": tag200_bx[iL].astype(np.int32, copy=False),
                "tag200_R": tag200_bx[iR].astype(np.int32, copy=False),
                "tag400_L": tag400_bx[iL].astype(np.int32, copy=False),
                "tag400_R": tag400_bx[iR].astype(np.int32, copy=False),
                "double_tag_420": np.ones(n_pairs, dtype=np.int32),
                "vtx_ok": vtx_ok.astype(np.int32, copy=False),
                "in_Hwin": np.ones(n_pairs, dtype=np.int32),
                "pv_interaction_id": np.full(n_pairs, pv["interaction_id"], dtype=np.int32),
                "pv_n_tracks": np.full(n_pairs, pv["n_tracks"], dtype=np.int32),
                "interaction_L_is_pv": is_pv_L.astype(np.int32, copy=False),
                "interaction_R_is_pv": is_pv_R.astype(np.int32, copy=False),
                "pv_z_compatible": pv_z_compatible.astype(np.int32, copy=False),
                "xi_L": xi_L.astype(np.float32, copy=False),
                "xi_R": xi_R.astype(np.float32, copy=False),
                "z_L": z_L.astype(np.float32, copy=False),
                "z_R": z_R.astype(np.float32, copy=False),
                "dz": dz.astype(np.float32, copy=False),
                "M": mass.astype(np.float32, copy=False),
                "ln_xi_ratio": ln_xi_ratio.astype(np.float32, copy=False),
                "yX": yX.astype(np.float32, copy=False),
                "abs_yX": np.abs(yX).astype(np.float32, copy=False),
                "pT_L": pT_L.astype(np.float32, copy=False),
                "pT_R": pT_R.astype(np.float32, copy=False),
                "t1_abs": t1_abs.astype(np.float32, copy=False),
                "t2_abs": t2_abs.astype(np.float32, copy=False),
                "t_sum": t_sum.astype(np.float32, copy=False),
                "t_diff_abs": t_diff_abs.astype(np.float32, copy=False),
                "pt_bal": pt_bal.astype(np.float32, copy=False),
                "pt_bal_ratio": pt_bal_ratio.astype(np.float32, copy=False),
                "abs_dphi": abs_dphi.astype(np.float32, copy=False),
                "dphi_from_pi": dphi_from_pi.astype(np.float32, copy=False),
                "pv_z": np.full(n_pairs, pv["z"], dtype=np.float32),
                "pv_t": np.full(n_pairs, pv["t"], dtype=np.float32),
                "pv_sum_track_pt2": np.full(n_pairs, pv["sum_pt2"], dtype=np.float32),
                "z_reco": z_reco.astype(np.float32, copy=False),
                "t_reco": t_reco.astype(np.float32, copy=False),
            },
        )

        for name, value in flags.items():
            if value:
                counters[name] += 1

    return {
        "counters": counters,
        "n_bx": len(selected_bx),
        "n_track_supported_pv": n_track_supported_pv,
        "n_pairs": n_pairs_total,
        "zcut_cm": zcut_cm,
        "timing_sigma_dz": timing_sigma_dz,
        "root_chunks": root_chunks,
        "int_branch_names": int_branch_names,
        "float_branch_names": float_branch_names,
    }


def write_root(root_out, root_chunks, int_branch_names, float_branch_names):
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError(
            "uproot is required for --root-out. Install with: python3 -m pip install --user uproot awkward"
        ) from exc

    arrays = {}
    for name in int_branch_names:
        if root_chunks[name]:
            arrays[name] = np.concatenate(root_chunks[name]).astype(np.int32, copy=False)
        else:
            arrays[name] = np.empty(0, dtype=np.int32)
    for name in float_branch_names:
        if root_chunks[name]:
            arrays[name] = np.concatenate(root_chunks[name]).astype(np.float32, copy=False)
        else:
            arrays[name] = np.empty(0, dtype=np.float32)

    print(f"[uproot] Writing output file: {root_out}", flush=True)
    with uproot.recreate(root_out) as fout:
        fout["MinbiasTrackPV"] = arrays
    print(f"Wrote ROOT file: {root_out}")
    print(f"Pairs stored in ROOT tree: {len(arrays['bx'])}")


def print_summary(files, args, result):
    counters = result["counters"]
    n_bx = result["n_bx"]

    print("")
    print("=== MinBias Track-PV Analysis ===")
    print(f"Input files: {len(files)}")
    for filename in files[:5]:
        print(f"  {filename}")
    if len(files) > 5:
        print(f"  ... {len(files) - 5} more")
    print(f"BX processed: {n_bx}")
    print(f"Track pT cut: > {args.track_pt_min} GeV")
    print(f"Track eta cut: abs(eta) < {args.track_eta_max}")
    print(f"Beam sigma z: {args.beam_sigma_z_cm} cm")
    print(f"Single-arm timing resolution: {args.single_arm_time_res_ps} ps")
    print(f"sigma_dz: {result['timing_sigma_dz']:.6f} cm")
    print(f"PV z compatibility cut: {result['zcut_cm']:.6f} cm")
    if args.mass_window is None:
        print("Mass window: disabled")
    else:
        print(f"Mass window: ±{args.mass_window} GeV around {M_H_GEV} GeV")
    print(f"Pairs passing tag and mass rules: {result['n_pairs']}")
    print("")

    print_count("BX with track-supported PV", result["n_track_supported_pv"], n_bx)
    print_count("BX with >=1 passing proton pair", counters["passing"], n_bx)
    print_count("BX with >=1 vertex-compatible pair", counters["vertex"], n_bx)
    print_count("BX with >=1 pair compatible with charged-track PV", counters["pv_z"], n_bx)
    print_count("BX with >=1 pair compatible with charged-track PV and tracks", counters["pv_z_has_tracks"], n_bx)
    print_count("BX with >=1 pair compatible with charged-track PV and no tracks", counters["pv_z_no_tracks"], n_bx)
    print("")

    for prefix, label in (
        ("passing", "Passing pairs"),
        ("vertex", "Vertex-compatible pairs"),
        ("pv_z", "Pairs compatible with charged-track PV"),
        ("pv_z_has_tracks", "Pairs compatible with charged-track PV with tracks"),
        ("pv_z_no_tracks", "Pairs compatible with charged-track PV and no tracks"),
    ):
        print(f"=== {label} by PV membership ===")
        print_count("Both protons from PV interaction", counters[f"{prefix}_both_pv"], n_bx)
        print_count("Exactly one proton from PV interaction", counters[f"{prefix}_one_pv"], n_bx)
        print_count("Neither proton from PV interaction", counters[f"{prefix}_neither_pv"], n_bx)
        print("")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze minbias proton pairs against a charged-track primary vertex."
    )
    parser.add_argument("-i", "--input", required=True, help="Input minbias NPZ file or directory of NPZ files")
    parser.add_argument("--max-bx", type=int, default=None, help="Optional maximum number of BX to process")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of files to process")
    parser.add_argument("--track-pt-min", type=float, default=0.5, help="Track proxy pT cut in GeV (default: 0.5)")
    parser.add_argument("--track-eta-max", type=float, default=2.5, help="Track proxy |eta| cut (default: 2.5)")
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7, help="Gaussian beam spot sigma z in cm")
    parser.add_argument("--single-arm-time-res-ps", type=float, default=3.0, help="Single-arm timing resolution in ps")
    parser.add_argument("--nsigma", type=float, default=2.0, help="Compatibility cut in units of sigma_dz")
    parser.add_argument("--smear-seed", type=int, default=12345, help="Seed for vertex and timing smearing")
    parser.add_argument("--mass-window", type=float, default=None, help="Mass half-width in GeV around 125")
    parser.add_argument("--s", type=float, default=S_GEV2_DEFAULT, help="s value for M=sqrt(xi1*xi2*s)")
    parser.add_argument("--no-dz-smear", dest="smear_dz", action="store_false", help="Disable timing smearing")
    parser.set_defaults(smear_dz=True)
    parser.add_argument("--root-out", default=None, help="Optional ROOT output file with tree MinbiasTrackPV")
    parser.add_argument(
        "--print-track-candidates",
        action="store_true",
        help="Print per-interaction track candidates used for PV scoring",
    )
    parser.add_argument(
        "--print-track-candidate-bx",
        type=int,
        action="append",
        default=[],
        help="Specific BX to print for --print-track-candidates; may be repeated",
    )
    parser.add_argument(
        "--max-track-candidate-bx",
        type=int,
        default=5,
        help="Max BX to print when --print-track-candidates is set without explicit BX values",
    )
    parser.add_argument(
        "--max-tracks-per-interaction",
        type=int,
        default=20,
        help="Max track rows to print per interaction in track candidate debug output",
    )
    args = parser.parse_args()

    if args.max_bx is not None and args.max_bx <= 0:
        raise RuntimeError("--max-bx must be > 0")
    if args.track_pt_min < 0.0:
        raise RuntimeError("--track-pt-min must be >= 0")
    if args.track_eta_max <= 0.0:
        raise RuntimeError("--track-eta-max must be > 0")
    if args.beam_sigma_z_cm < 0.0:
        raise RuntimeError("--beam-sigma-z-cm must be >= 0")
    if args.single_arm_time_res_ps <= 0.0:
        raise RuntimeError("--single-arm-time-res-ps must be > 0")
    if args.nsigma <= 0.0:
        raise RuntimeError("--nsigma must be > 0")
    if args.mass_window is not None and args.mass_window < 0.0:
        raise RuntimeError("--mass-window must be >= 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_track_candidate_bx <= 0:
        raise RuntimeError("--max-track-candidate-bx must be > 0")
    if args.max_tracks_per_interaction <= 0:
        raise RuntimeError("--max-tracks-per-interaction must be > 0")

    files, protons, tracks, universe = load_minbias(args.input, max_files=args.max_files)
    model = build_interaction_model(universe, protons, tracks, args)
    all_bx, pv_by_bx = primary_vertices_by_bx(model)
    selected_bx = all_bx
    if args.max_bx is not None:
        selected_bx = selected_bx[:args.max_bx]

    if args.print_track_candidates:
        print_track_candidates(tracks, model, pv_by_bx, selected_bx, args)

    result = analyze(protons, model, pv_by_bx, selected_bx, args)
    print_summary(files, args, result)

    if args.root_out:
        write_root(
            args.root_out,
            result["root_chunks"],
            result["int_branch_names"],
            result["float_branch_names"],
        )


if __name__ == "__main__":
    main()
