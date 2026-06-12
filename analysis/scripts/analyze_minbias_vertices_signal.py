import argparse
import glob
import math
import os

import numpy as np

S_GEV2 = 14000.0**2
C_CM_PER_PS = 2.99792458e-2
F_COLL_AVG_HZ = 31.6e6
M_HIGGS_GEV = 126.0

STATION_XI = {
    "192": (0.08, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.014, 0.0263),
    "420": (0.00325, 0.0116),
}
STATIONS_200 = ("192", "213", "220")

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

ROOT_INT_BRANCHES = (
    "combo_id",
    "signal_event_idx",
    "bx",
    "source_L",
    "source_R",
    "signal_only",
    "minbias_only",
    "mixed",
    "non_signal",
    "interaction_L",
    "interaction_R",
    "proton_idx_L",
    "proton_idx_R",
    "side_L",
    "side_R",
    "tag200_L",
    "tag200_R",
    "tag400_L",
    "tag400_R",
    "tag192_L",
    "tag192_R",
    "tag213_L",
    "tag213_R",
    "tag220_L",
    "tag220_R",
    "pair_tag200",
    "pair_tag192",
    "pair_tag213",
    "pair_tag220",
    "pair_tag400",
    "pair_double_tag400",
    "bx_compatible",
    "z_compatible",
    "t_compatible",
    "vtx_compatible",
    "bx_z_compatible",
    "bx_t_compatible",
    "bx_vtx_compatible",
    "y_compatible_2sigma",
    "y_compatible_3sigma",
    "left_is_signal",
    "right_is_signal",
    "left_is_minbias",
    "right_is_minbias",
    "minbias_pair_no_central_vertex_overlap",
    "minbias_pair_bx_no_central_vertex_overlap",
    "require_two_400_tags",
    "smear_dz",
    "has_pv_t",
    "qq_pdg_abs",
)

ROOT_FLOAT_BRANCHES = (
    "xi_L",
    "xi_R",
    "xi_nominal_L",
    "xi_nominal_R",
    "mass",
    "z_truth_L",
    "t_truth_L",
    "z_truth_R",
    "t_truth_R",
    "signal_z_truth",
    "signal_t_truth",
    "pp_z_reco",
    "pp_t_reco",
    "pv_z_reco",
    "pv_t_reco",
    "nearest_detected_mb_vertex_dz",
    "nearest_detected_mb_vertex_dt",
    "zcut",
    "tcut",
    "bx_zcut",
    "bx_tcut",
    "mass_window",
    "s",
    "ln_xi_ratio",
    "yX",
    "abs_yX",
    "qq_y_truth",
    "qq_y_reco",
    "delta_y",
    "abs_delta_y",
    "sigma_y_pps",
    "sigma_y_total",
    "ycut_2sigma",
    "ycut_3sigma",
)

def input_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not files:
            raise RuntimeError(f"No .npz files found in directory {path}")
        return files
    return [path]

def signal_input_files(path):
    if os.path.isdir(path):
        files = []
        for pattern in ("*.lhe", "*.dat"):
            files.extend(sorted(glob.glob(os.path.join(path, pattern))))
        if not files:
            raise RuntimeError(f"No .lhe/.dat files found in directory {path}")
        return files
    return [path]

def parse_lhe_with_xsec(filename):
    events = []
    xsec_pb = None

    in_event = False
    expect_event_header = False
    current_particles = []
    current_header = None

    in_init = False
    init_line_count = 0

    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("<init>"):
                in_init = True
                init_line_count = 0
                continue
            if line.startswith("</init>"):
                in_init = False
                continue

            if in_init:
                init_line_count += 1
                if init_line_count == 2:
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            xsec_pb = float(parts[0])
                        except ValueError:
                            pass
                continue

            if line.startswith("<event"):
                in_event = True
                expect_event_header = True
                current_particles = []
                current_header = None
                continue

            if line.startswith("</event"):
                if current_header is not None:
                    events.append((current_header, current_particles))
                in_event = False
                expect_event_header = False
                continue

            if not in_event:
                continue

            if expect_event_header:
                parts = line.split()
                if len(parts) < 6:
                    raise RuntimeError(f"Unexpected event header: {line}")
                current_header = {
                    "NUP": int(parts[0]),
                    "IDPRUP": int(parts[1]),
                    "XWGTUP": float(parts[2]),
                    "SCALUP": float(parts[3]),
                    "AQEDUP": float(parts[4]),
                    "AQCDUP": float(parts[5]),
                }
                expect_event_header = False
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            current_particles.append(
                {
                    "pid": int(parts[0]),
                    "status": int(parts[1]),
                    "moth1": int(parts[2]),
                    "moth2": int(parts[3]),
                    "col1": int(parts[4]),
                    "col2": int(parts[5]),
                    "px": float(parts[6]),
                    "py": float(parts[7]),
                    "pz": float(parts[8]),
                    "E": float(parts[9]),
                    "m": float(parts[10]),
                }
            )

    return events, xsec_pb

def load_signal_events(path):
    files = signal_input_files(path)
    events = []
    xsec_from_file = None
    for filename in files:
        file_events, xsec_pb = parse_lhe_with_xsec(filename)
        if xsec_pb is not None and xsec_from_file is None:
            xsec_from_file = xsec_pb
        events.extend(file_events)
    if not events:
        raise RuntimeError("No signal events parsed from input files.")
    return files, events, xsec_from_file

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
                    print(f"    Proton {i}: side={interaction_protons['side'][i]}, pt={interaction_protons['pt'][i]:.2f} GeV, xi={interaction_protons['xi'][i]:.4f}")


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

def station_tag(xi, station):
    xi_min, xi_max = STATION_XI[station]
    return (xi >= xi_min) & (xi < xi_max)

def smear_xi(xi, xi_res, rng):
    if xi_res <= 0.0 or xi.size == 0:
        return xi
    return xi + rng.normal(loc=0.0, scale=xi_res, size=xi.size)

def rapidity(E, pz):
    if E <= abs(pz):
        return math.nan
    return 0.5 * math.log((E + pz) / (E - pz))

def signal_qq_rapidity(signal_event):
    _header, particles = signal_event
    for pdg_abs in (5, 4):
        q = [p for p in particles if p["pid"] == pdg_abs and p["status"] == 1]
        qbar = [p for p in particles if p["pid"] == -pdg_abs and p["status"] == 1]
        if len(q) == 1 and len(qbar) == 1:
            E = q[0]["E"] + qbar[0]["E"]
            pz = q[0]["pz"] + qbar[0]["pz"]
            return rapidity(E, pz), pdg_abs

    return math.nan, 0

def signal_protons_from_event(signal_event):
    _header, particles = signal_event
    protons_out = [p for p in particles if p["pid"] == 2212 and p["status"] == 1]
    if len(protons_out) != 2:
        return None

    incoming = [p for p in particles if p["pid"] == 2212 and p["status"] == -1]
    beam_E = incoming[0]["E"] if incoming else None

    out = []
    for proton_idx, proton in enumerate(protons_out):
        side = +1 if proton["pz"] >= 0.0 else -1
        if beam_E is not None and beam_E > 0.0:
            xi = (beam_E - proton["E"]) / beam_E
        else:
            xi = 0.0
        tag200, tag400 = station_tags(np.asarray([xi], dtype=np.float64))
        out.append(
            {
                "source": "signal",
                "interaction_id": -1,
                "proton_idx": proton_idx,
                "side": side,
                "xi": float(xi) * 13.6 / 14,
                "px": float(proton["px"]),
                "py": float(proton["py"]),
                "pt": math.hypot(proton["px"], proton["py"]),
                "pz": float(proton["pz"]),
                "tag200": bool(tag200[0]),
                "tag400": bool(tag400[0]),
            }
        )

    return out

def add_source_counter_sums(counters, prefix, mask, source_left, source_right):
    n_pass = int(np.sum(mask))
    counters[f"{prefix}_any"] += n_pass
    if n_pass == 0:
        return

    signal_only = mask & (source_left == 0) & (source_right == 0)
    minbias_only = mask & (source_left == 1) & (source_right == 1)
    n_signal_only = int(np.sum(signal_only))
    n_minbias_only = int(np.sum(minbias_only))
    counters[f"{prefix}_signal_only"] += n_signal_only
    counters[f"{prefix}_minbias_only"] += n_minbias_only
    counters[f"{prefix}_mixed"] += n_pass - n_signal_only - n_minbias_only
    counters[f"{prefix}_non_signal"] += n_pass - n_signal_only

def empty_root_chunks():
    return {name: [] for name in ROOT_INT_BRANCHES + ROOT_FLOAT_BRANCHES}

def append_root_chunk(root_chunks, root_chunk):
    if root_chunks is None or root_chunk is None:
        return
    for name, values in root_chunk.items():
        root_chunks[name].append(values)

def nearest_detected_vertex_deltas(pp_z_reco, pp_t_reco, detected_z, detected_t):
    n_pairs = pp_z_reco.size
    nearest_dz = np.full(n_pairs, np.nan, dtype=np.float64)
    nearest_dt = np.full(n_pairs, np.nan, dtype=np.float64)
    if detected_z.size == 0:
        return nearest_dz, nearest_dt

    dz_all = detected_z[np.newaxis, :] - pp_z_reco[:, np.newaxis]
    nearest_idx = np.argmin(np.abs(dz_all), axis=1)
    nearest_dz = dz_all[np.arange(n_pairs), nearest_idx]
    if detected_t is not None:
        nearest_dt = detected_t[nearest_idx] - pp_t_reco
    return nearest_dz, nearest_dt

def no_detected_vertex_overlap_flags(pp_z_reco, pp_t_reco, detected_z, detected_t, zcut, tcut, pair_idx):
    no_overlap = np.zeros(pp_z_reco.size, dtype=bool)
    if pair_idx.size == 0:
        return no_overlap
    if detected_z.size == 0:
        no_overlap[pair_idx] = True
        return no_overlap

    order = np.argsort(detected_z, kind="mergesort")
    z_sorted = detected_z[order]
    t_sorted = detected_t[order] if detected_t is not None else None

    for idx in pair_idx:
        lo = np.searchsorted(z_sorted, pp_z_reco[idx] - zcut, side="left")
        hi = np.searchsorted(z_sorted, pp_z_reco[idx] + zcut, side="right")
        if lo == hi:
            no_overlap[idx] = True
            continue
        if t_sorted is None:
            continue
        if not np.any(np.abs(t_sorted[lo:hi] - pp_t_reco[idx]) <= tcut):
            no_overlap[idx] = True

    return no_overlap

def write_root(root_out, root_chunks):
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError(
            "uproot is required for --root-out. Install with: python3 -m pip install --user uproot awkward"
        ) from exc

    arrays = {}
    for name in ROOT_INT_BRANCHES:
        if root_chunks[name]:
            arrays[name] = np.concatenate(root_chunks[name]).astype(np.int32, copy=False)
        else:
            arrays[name] = np.empty(0, dtype=np.int32)
    for name in ROOT_FLOAT_BRANCHES:
        if root_chunks[name]:
            arrays[name] = np.concatenate(root_chunks[name]).astype(np.float32, copy=False)
        else:
            arrays[name] = np.empty(0, dtype=np.float32)

    out_dir = os.path.dirname(root_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"[uproot] Writing output file: {root_out}", flush=True)
    with uproot.recreate(root_out) as fout:
        fout["SignalMinbiasPairs"] = arrays
    print(f"Wrote ROOT file: {root_out}")
    print(f"Pairs stored in ROOT tree: {len(arrays['bx'])}")

def analyze_combo(combo_id, bx, signal_event_idx, signal_event, model, bx_protons, bx_proton_model_idx, mb_vertex_idx, constants, args, protons, tracks):
    rng = np.random.default_rng(args.seed + combo_id * 1000)
    y_rng = np.random.default_rng(args.seed + combo_id * 1000 + 3)
    xi_rng = np.random.default_rng(args.seed + combo_id * 1000 + 2)
    signal_protons = signal_protons_from_event(signal_event)
    if signal_protons is None:
        return None

    qq_y_truth, qq_pdg_abs = signal_qq_rapidity(signal_event)
    qq_y_reco = (
        qq_y_truth + y_rng.normal(loc=0.0, scale=args.central_y_res)
        if np.isfinite(qq_y_truth) and args.central_y_res > 0.0
        else qq_y_truth
    )

    signal_z = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm)
    signal_t = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm / C_CM_PER_PS)
    pv_z_reco = signal_z + rng.normal(loc=0.0, scale=args.pv_z_res_cm)
    pv_t_reco = signal_t + rng.normal(loc=0.0, scale=args.pv_t_res_ps) if args.pv_t_res_ps is not None else None

    vertex_rng = np.random.default_rng(args.seed + combo_id * 1000 + 1)
    mb_vertex_z_reco = model["z"][mb_vertex_idx] + vertex_rng.normal(
        loc=0.0,
        scale=args.pv_z_res_cm,
        size=mb_vertex_idx.size,
    )
    mb_vertex_z_overlap = np.abs(mb_vertex_z_reco - pv_z_reco) <= constants["central_zcut"]
    if args.pv_t_res_ps is not None:
        mb_vertex_t_reco = model["t"][mb_vertex_idx] + vertex_rng.normal(
            loc=0.0,
            scale=args.pv_t_res_ps,
            size=mb_vertex_idx.size,
        )
        mb_vertex_t_overlap = np.abs(mb_vertex_t_reco - pv_t_reco) <= constants["central_tcut"]
    else:
        mb_vertex_t_reco = None
        mb_vertex_t_overlap = np.zeros(mb_vertex_idx.size, dtype=bool)
    mb_vertex_has_tracks = model["n_tracks"][mb_vertex_idx] > 0
    mb_vertex_z_overlap &= mb_vertex_has_tracks
    mb_vertex_t_overlap &= mb_vertex_has_tracks
    mb_vertex_overlap = mb_vertex_z_overlap & mb_vertex_t_overlap
    detected_mb_vertex_z_reco = mb_vertex_z_reco[mb_vertex_has_tracks]
    detected_mb_vertex_t_reco = (
        mb_vertex_t_reco[mb_vertex_has_tracks]
        if mb_vertex_t_reco is not None
        else None
    )

    sig_xi_nominal = np.asarray([p["xi"] for p in signal_protons], dtype=np.float64)
    sig_xi = smear_xi(sig_xi_nominal, args.xi_res, xi_rng)
    sig_side = np.asarray([p["side"] for p in signal_protons], dtype=np.int32)
    sig_tag200, sig_tag400 = station_tags(sig_xi_nominal)
    sig_station_tags = {
        station: station_tag(sig_xi_nominal, station)
        for station in STATIONS_200
    }
    sig_z = np.full(sig_xi.size, signal_z, dtype=np.float64)
    sig_t = np.full(sig_xi.size, signal_t, dtype=np.float64)
    sig_source = np.zeros(sig_xi.size, dtype=np.int8)
    sig_interaction = np.full(sig_xi.size, -1, dtype=np.int32)
    sig_proton_idx = np.asarray([p["proton_idx"] for p in signal_protons], dtype=np.int32)

    bx_p_xi_nominal = np.asarray(bx_protons["xi"], dtype=np.float64)
    bx_p_xi = smear_xi(bx_p_xi_nominal, args.xi_res, xi_rng)
    bx_p_side = np.asarray(bx_protons["side"], dtype=np.int32)
    bx_p_tag200, bx_p_tag400 = station_tags(bx_p_xi_nominal)
    bx_p_station_tags = {
        station: station_tag(bx_p_xi_nominal, station)
        for station in STATIONS_200
    }
    bx_p_z = model["z"][bx_proton_model_idx] if bx_proton_model_idx.size else np.empty(0)
    bx_p_t = model["t"][bx_proton_model_idx] if bx_proton_model_idx.size else np.empty(0)

    cand_xi = np.concatenate((sig_xi, bx_p_xi))
    cand_xi_nominal = np.concatenate((sig_xi_nominal, bx_p_xi_nominal))
    cand_side = np.concatenate((sig_side, bx_p_side))
    cand_tag200 = np.concatenate((sig_tag200, bx_p_tag200))
    cand_tag400 = np.concatenate((sig_tag400, bx_p_tag400))
    cand_tagged = cand_tag200 | cand_tag400
    cand_station_tags = {
        station: np.concatenate((sig_station_tags[station], bx_p_station_tags[station]))
        for station in STATIONS_200
    }
    cand_z = np.concatenate((sig_z, bx_p_z))
    cand_t = np.concatenate((sig_t, bx_p_t))
    cand_source = np.concatenate((sig_source, np.ones(bx_p_xi.size, dtype=np.int8)))
    cand_interaction = np.concatenate((sig_interaction, np.asarray(bx_protons["interaction_id"], dtype=np.int32)))
    cand_proton_idx = np.concatenate((sig_proton_idx, np.asarray(bx_protons["proton_idx"], dtype=np.int32)))

    counters = {
        "n_candidate_protons": int(cand_xi.size),
        "n_tagged_protons": int(np.sum(cand_tagged)),
        "n_minbias_tagged_protons": int(np.sum(cand_tagged & (cand_source == 1))),
        "n_minbias_vertices": int(mb_vertex_idx.size),
        "n_minbias_vertices_with_tracks": int(np.sum(mb_vertex_has_tracks)),
        "n_minbias_vertices_z_overlap": int(np.sum(mb_vertex_z_overlap)),
        "n_minbias_vertices_t_overlap": int(np.sum(mb_vertex_t_overlap)),
        "n_minbias_vertices_vtx_overlap": int(np.sum(mb_vertex_overlap)),
        "minbias_vertex_z_overlap_any": int(np.any(mb_vertex_z_overlap)),
        "minbias_vertex_t_overlap_any": int(np.any(mb_vertex_t_overlap)),
        "minbias_vertex_vtx_overlap_any": int(np.any(mb_vertex_overlap)),
        "n_pp": 0,
        "n_pp_z_compatible": 0,
        "n_pp_t_compatible": 0,
        "n_pp_vtx_compatible": 0,
        "n_pp_bx_compatible": 0,
        "n_pp_bx_z_compatible": 0,
        "n_pp_bx_t_compatible": 0,
        "n_pp_bx_vtx_compatible": 0,
        "n_pp_y_compatible_2sigma": 0,
        "n_pp_y_compatible_3sigma": 0,
        "n_pp_z_y_compatible_2sigma": 0,
        "n_pp_z_y_compatible_3sigma": 0,
        "n_pp_vtx_y_compatible_2sigma": 0,
        "n_pp_vtx_y_compatible_3sigma": 0,
        "n_pp_tag200": 0,
        "n_pp_tag200_signal_only": 0,
        "n_pp_tag200_minbias_only": 0,
        "n_minbias_pp_bx_compatible": 0,
        "n_minbias_pp_no_central_vertex_overlap": 0,
        "n_minbias_pp_bx_no_central_vertex_overlap": 0,
        "minbias_pp_no_central_vertex_overlap_any": 0,
        "minbias_pp_bx_no_central_vertex_overlap_any": 0,
        "minbias_vertex_z_overlap_pp_accept_mass_any": 0,
        "minbias_vertex_t_overlap_pp_accept_mass_any": 0,
        "minbias_vertex_vtx_overlap_pp_accept_mass_any": 0,
        "minbias_vertex_pp_z_overlap_signal_pair_any": 0,
        "minbias_vertex_pp_t_overlap_signal_pair_any": 0,
        "minbias_vertex_pp_vtx_overlap_signal_pair_any": 0,
        "passing_any": 0,
        "passing_non_signal": 0,
        "passing_signal_only": 0,
        "passing_minbias_only": 0,
        "passing_mixed": 0,
        "z_any": 0,
        "z_non_signal": 0,
        "z_signal_only": 0,
        "z_minbias_only": 0,
        "z_mixed": 0,
        "t_any": 0,
        "t_non_signal": 0,
        "t_signal_only": 0,
        "t_minbias_only": 0,
        "t_mixed": 0,
        "vtx_any": 0,
        "vtx_non_signal": 0,
        "vtx_signal_only": 0,
        "vtx_minbias_only": 0,
        "vtx_mixed": 0,
        "bx_any": 0,
        "bx_non_signal": 0,
        "bx_signal_only": 0,
        "bx_minbias_only": 0,
        "bx_mixed": 0,
        "bx_z_any": 0,
        "bx_z_non_signal": 0,
        "bx_z_signal_only": 0,
        "bx_z_minbias_only": 0,
        "bx_z_mixed": 0,
        "bx_t_any": 0,
        "bx_t_non_signal": 0,
        "bx_t_signal_only": 0,
        "bx_t_minbias_only": 0,
        "bx_t_mixed": 0,
        "bx_vtx_any": 0,
        "bx_vtx_non_signal": 0,
        "bx_vtx_signal_only": 0,
        "bx_vtx_minbias_only": 0,
        "bx_vtx_mixed": 0,
        "y2_any": 0,
        "y2_non_signal": 0,
        "y2_signal_only": 0,
        "y2_minbias_only": 0,
        "y2_mixed": 0,
        "y3_any": 0,
        "y3_non_signal": 0,
        "y3_signal_only": 0,
        "y3_minbias_only": 0,
        "y3_mixed": 0,
        "z_y2_any": 0,
        "z_y2_non_signal": 0,
        "z_y2_signal_only": 0,
        "z_y2_minbias_only": 0,
        "z_y2_mixed": 0,
        "z_y3_any": 0,
        "z_y3_non_signal": 0,
        "z_y3_signal_only": 0,
        "z_y3_minbias_only": 0,
        "z_y3_mixed": 0,
        "vtx_y2_any": 0,
        "vtx_y2_non_signal": 0,
        "vtx_y2_signal_only": 0,
        "vtx_y2_minbias_only": 0,
        "vtx_y2_mixed": 0,
        "vtx_y3_any": 0,
        "vtx_y3_non_signal": 0,
        "vtx_y3_signal_only": 0,
        "vtx_y3_minbias_only": 0,
        "vtx_y3_mixed": 0,
    }
    for station in STATIONS_200:
        counters[f"n_pp_tag{station}"] = 0
        counters[f"n_pp_tag{station}_signal_only"] = 0
        counters[f"n_pp_tag{station}_minbias_only"] = 0

    zcut = constants["zcut"]
    tcut = constants["tcut"]
    n_candidates = int(cand_xi.size)

    if args.verbose > 1:
        print(
            f"Combo {combo_id}: signal_evt={signal_event_idx}, bx={bx}, "
            f"signal vertex z={signal_z:.2f} cm, t={signal_t:.2f} ps"
        )
        print(f"  Signal qq rapidity: |pdg|={qq_pdg_abs}, truth={qq_y_truth:.4f}, reco={qq_y_reco:.4f}")
        print(f"  Reco signal PV: z={pv_z_reco:.2f} cm, t={pv_t_reco} ps")
        print(
            f"  Minbias reco vertices overlapping PV: "
            f"z={counters['n_minbias_vertices_z_overlap']}, "
            f"t={counters['n_minbias_vertices_t_overlap']}, "
            f"z+t={counters['n_minbias_vertices_vtx_overlap']} "
            f"out of {counters['n_minbias_vertices_with_tracks']} vertices with tracks"
        )
        print(f"  Candidate protons: {n_candidates} ({counters['n_tagged_protons']} tagged)")

    cand_left = np.where((cand_side == -1) & cand_tagged)[0]
    cand_right = np.where((cand_side == +1) & cand_tagged)[0]
    iL, iR = cartesian_pair_indices(cand_left, cand_right)
    if args.require_two_400_tags:
        keep = cand_tag400[iL] & cand_tag400[iR]
    else:
        keep = cand_tag400[iL] | cand_tag400[iR]
    xi_prod = cand_xi[iL] * cand_xi[iR]
    keep &= xi_prod > 0.0
    iL = iL[keep]
    iR = iR[keep]
    xi_prod = xi_prod[keep]
    if iL.size:
        pair_order = np.lexsort((np.maximum(iL, iR), np.minimum(iL, iR)))
        iL = iL[pair_order]
        iR = iR[pair_order]
        xi_prod = xi_prod[pair_order]

    mass = np.sqrt(xi_prod * args.s)
    if args.mass_window is not None:
        keep = (mass >= M_HIGGS_GEV - args.mass_window) & (mass <= M_HIGGS_GEV + args.mass_window)
        iL = iL[keep]
        iR = iR[keep]
        mass = mass[keep]

    n_pairs = int(iL.size)
    counters["n_pp"] = n_pairs
    root_chunk = None
    if n_pairs > 0:
        counters["minbias_vertex_z_overlap_pp_accept_mass_any"] = int(np.any(mb_vertex_z_overlap))
        counters["minbias_vertex_t_overlap_pp_accept_mass_any"] = int(np.any(mb_vertex_t_overlap))
        counters["minbias_vertex_vtx_overlap_pp_accept_mass_any"] = int(np.any(mb_vertex_overlap))
    if n_pairs > 0:
        source_left = cand_source[iL]
        source_right = cand_source[iR]
        all_pairs = np.ones(n_pairs, dtype=bool)
        add_source_counter_sums(counters, "passing", all_pairs, source_left, source_right)

        pair_tag200 = cand_tag200[iL] | cand_tag200[iR]
        signal_only = (source_left == 0) & (source_right == 0)
        minbias_only = (source_left == 1) & (source_right == 1)
        counters["n_pp_tag200"] = int(np.sum(pair_tag200))
        counters["n_pp_tag200_signal_only"] = int(np.sum(pair_tag200 & signal_only))
        counters["n_pp_tag200_minbias_only"] = int(np.sum(pair_tag200 & minbias_only))
        for station in STATIONS_200:
            pair_station_tag = cand_station_tags[station][iL] | cand_station_tags[station][iR]
            counters[f"n_pp_tag{station}"] = int(np.sum(pair_station_tag))
            counters[f"n_pp_tag{station}_signal_only"] = int(np.sum(pair_station_tag & signal_only))
            counters[f"n_pp_tag{station}_minbias_only"] = int(np.sum(pair_station_tag & minbias_only))

        pL_t_truth = cand_t[iL] + cand_z[iL] / C_CM_PER_PS
        pR_t_truth = cand_t[iR] - cand_z[iR] / C_CM_PER_PS
        if args.smear_dz:
            smear = rng.normal(loc=0.0, scale=args.pps_time_res_ps, size=(n_pairs, 2))
            pL_t_reco = pL_t_truth + smear[:, 0]
            pR_t_reco = pR_t_truth + smear[:, 1]
        else:
            pL_t_reco = pL_t_truth
            pR_t_reco = pR_t_truth

        pp_z_reco = 0.5 * (pL_t_reco - pR_t_reco) * C_CM_PER_PS
        pp_t_reco = 0.5 * (pL_t_reco + pR_t_reco)
        ln_xi_ratio = np.full(n_pairs, np.nan, dtype=np.float64)
        valid_xi = (cand_xi[iL] > 0.0) & (cand_xi[iR] > 0.0)
        ln_xi_ratio[valid_xi] = np.log(cand_xi[iR][valid_xi] / cand_xi[iL][valid_xi])
        yX = 0.5 * ln_xi_ratio
        sigma_y_pps = np.full(n_pairs, np.nan, dtype=np.float64)
        if args.xi_res > 0.0:
            sigma_y_pps[valid_xi] = 0.5 * np.sqrt(
                (args.xi_res / cand_xi[iL][valid_xi]) ** 2
                + (args.xi_res / cand_xi[iR][valid_xi]) ** 2
            )
        else:
            sigma_y_pps[valid_xi] = 0.0
        sigma_y_total = np.sqrt(sigma_y_pps * sigma_y_pps + args.central_y_res * args.central_y_res)
        delta_y = yX - qq_y_reco
        abs_delta_y = np.abs(delta_y)
        ycut_2sigma = 2.0 * sigma_y_total
        ycut_3sigma = 3.0 * sigma_y_total
        valid_y_compare = np.isfinite(abs_delta_y) & np.isfinite(sigma_y_total)
        y_compatible_2sigma = valid_y_compare & (abs_delta_y <= ycut_2sigma)
        y_compatible_3sigma = valid_y_compare & (abs_delta_y <= ycut_3sigma)
        signal_pair = (source_left == 0) & (source_right == 0)
        if np.any(signal_pair):
            signal_pair_idx = np.where(signal_pair)[0][0]
            mb_vertex_pp_z_overlap = (
                np.abs(detected_mb_vertex_z_reco - pp_z_reco[signal_pair_idx]) <= zcut
            )
            if detected_mb_vertex_t_reco is not None:
                mb_vertex_pp_t_overlap = (
                    np.abs(detected_mb_vertex_t_reco - pp_t_reco[signal_pair_idx]) <= tcut
                )
                mb_vertex_pp_overlap = mb_vertex_pp_z_overlap & mb_vertex_pp_t_overlap
            else:
                mb_vertex_pp_t_overlap = np.zeros(detected_mb_vertex_z_reco.size, dtype=bool)
                mb_vertex_pp_overlap = np.zeros(detected_mb_vertex_z_reco.size, dtype=bool)
            counters["minbias_vertex_pp_z_overlap_signal_pair_any"] = int(np.any(mb_vertex_pp_z_overlap))
            counters["minbias_vertex_pp_t_overlap_signal_pair_any"] = int(np.any(mb_vertex_pp_t_overlap))
            counters["minbias_vertex_pp_vtx_overlap_signal_pair_any"] = int(np.any(mb_vertex_pp_overlap))
        bx_compatible = (
            (np.abs(pp_z_reco) <= constants["bx_zcut"])
            & (np.abs(pp_t_reco) <= constants["bx_tcut"])
        )
        z_compatible = np.abs(pp_z_reco - pv_z_reco) <= zcut
        if args.pv_t_res_ps is not None:
            t_compatible = np.abs(pp_t_reco - pv_t_reco) <= tcut
        else:
            t_compatible = np.abs(pp_t_reco) <= tcut
        vtx_compatible = z_compatible & t_compatible
        z_y_compatible_2sigma = z_compatible & y_compatible_2sigma
        z_y_compatible_3sigma = z_compatible & y_compatible_3sigma
        vtx_y_compatible_2sigma = vtx_compatible & y_compatible_2sigma
        vtx_y_compatible_3sigma = vtx_compatible & y_compatible_3sigma

        counters["n_pp_bx_compatible"] = int(np.sum(bx_compatible))
        counters["n_pp_z_compatible"] = int(np.sum(z_compatible))
        counters["n_pp_t_compatible"] = int(np.sum(t_compatible))
        counters["n_pp_vtx_compatible"] = int(np.sum(vtx_compatible))
        counters["n_pp_bx_z_compatible"] = int(np.sum(bx_compatible & z_compatible))
        counters["n_pp_bx_t_compatible"] = int(np.sum(bx_compatible & t_compatible))
        counters["n_pp_bx_vtx_compatible"] = int(np.sum(bx_compatible & vtx_compatible))
        counters["n_pp_y_compatible_2sigma"] = int(np.sum(y_compatible_2sigma))
        counters["n_pp_y_compatible_3sigma"] = int(np.sum(y_compatible_3sigma))
        counters["n_pp_z_y_compatible_2sigma"] = int(np.sum(z_y_compatible_2sigma))
        counters["n_pp_z_y_compatible_3sigma"] = int(np.sum(z_y_compatible_3sigma))
        counters["n_pp_vtx_y_compatible_2sigma"] = int(np.sum(vtx_y_compatible_2sigma))
        counters["n_pp_vtx_y_compatible_3sigma"] = int(np.sum(vtx_y_compatible_3sigma))

        add_source_counter_sums(counters, "bx", bx_compatible, source_left, source_right)
        add_source_counter_sums(counters, "z", z_compatible, source_left, source_right)
        add_source_counter_sums(counters, "t", t_compatible, source_left, source_right)
        add_source_counter_sums(counters, "vtx", vtx_compatible, source_left, source_right)
        add_source_counter_sums(counters, "bx_z", bx_compatible & z_compatible, source_left, source_right)
        add_source_counter_sums(counters, "bx_t", bx_compatible & t_compatible, source_left, source_right)
        add_source_counter_sums(counters, "bx_vtx", bx_compatible & vtx_compatible, source_left, source_right)
        add_source_counter_sums(counters, "y2", y_compatible_2sigma, source_left, source_right)
        add_source_counter_sums(counters, "y3", y_compatible_3sigma, source_left, source_right)
        add_source_counter_sums(counters, "z_y2", z_y_compatible_2sigma, source_left, source_right)
        add_source_counter_sums(counters, "z_y3", z_y_compatible_3sigma, source_left, source_right)
        add_source_counter_sums(counters, "vtx_y2", vtx_y_compatible_2sigma, source_left, source_right)
        add_source_counter_sums(counters, "vtx_y3", vtx_y_compatible_3sigma, source_left, source_right)

        counters["n_minbias_pp_bx_compatible"] = int(np.sum(minbias_only & bx_compatible))
        minbias_pair_no_central_vertex_overlap = np.zeros(n_pairs, dtype=bool)
        minbias_pair_bx_no_central_vertex_overlap = np.zeros(n_pairs, dtype=bool)
        minbias_pair_idx = np.where(minbias_only)[0]
        minbias_pair_no_central_vertex_overlap = no_detected_vertex_overlap_flags(
            pp_z_reco,
            pp_t_reco,
            detected_mb_vertex_z_reco,
            detected_mb_vertex_t_reco,
            zcut,
            tcut,
            minbias_pair_idx,
        )
        minbias_pair_bx_no_central_vertex_overlap = minbias_pair_no_central_vertex_overlap & bx_compatible
        counters["n_minbias_pp_no_central_vertex_overlap"] = int(np.sum(minbias_pair_no_central_vertex_overlap))
        counters["minbias_pp_no_central_vertex_overlap_any"] = int(np.any(minbias_pair_no_central_vertex_overlap))
        counters["n_minbias_pp_bx_no_central_vertex_overlap"] = int(np.sum(minbias_pair_bx_no_central_vertex_overlap))
        counters["minbias_pp_bx_no_central_vertex_overlap_any"] = int(np.any(minbias_pair_bx_no_central_vertex_overlap))

        if args.root_out:
            nearest_mb_dz, nearest_mb_dt = nearest_detected_vertex_deltas(
                pp_z_reco,
                pp_t_reco,
                detected_mb_vertex_z_reco,
                detected_mb_vertex_t_reco,
            )
            tag192 = cand_station_tags["192"]
            tag213 = cand_station_tags["213"]
            tag220 = cand_station_tags["220"]
            pair_tag192 = tag192[iL] | tag192[iR]
            pair_tag213 = tag213[iL] | tag213[iR]
            pair_tag220 = tag220[iL] | tag220[iR]
            pair_tag400 = cand_tag400[iL] | cand_tag400[iR]
            pair_double_tag400 = cand_tag400[iL] & cand_tag400[iR]
            left_is_signal = source_left == 0
            right_is_signal = source_right == 0
            left_is_minbias = source_left == 1
            right_is_minbias = source_right == 1
            mixed = ~(signal_only | minbias_only)
            non_signal = ~signal_only
            pv_t_value = np.nan if pv_t_reco is None else pv_t_reco
            root_chunk = {
                "combo_id": np.full(n_pairs, combo_id, dtype=np.int32),
                "signal_event_idx": np.full(n_pairs, signal_event_idx, dtype=np.int32),
                "bx": np.full(n_pairs, bx, dtype=np.int32),
                "source_L": source_left.astype(np.int32, copy=False),
                "source_R": source_right.astype(np.int32, copy=False),
                "signal_only": signal_only.astype(np.int32, copy=False),
                "minbias_only": minbias_only.astype(np.int32, copy=False),
                "mixed": mixed.astype(np.int32, copy=False),
                "non_signal": non_signal.astype(np.int32, copy=False),
                "interaction_L": cand_interaction[iL].astype(np.int32, copy=False),
                "interaction_R": cand_interaction[iR].astype(np.int32, copy=False),
                "proton_idx_L": cand_proton_idx[iL].astype(np.int32, copy=False),
                "proton_idx_R": cand_proton_idx[iR].astype(np.int32, copy=False),
                "side_L": cand_side[iL].astype(np.int32, copy=False),
                "side_R": cand_side[iR].astype(np.int32, copy=False),
                "tag200_L": cand_tag200[iL].astype(np.int32, copy=False),
                "tag200_R": cand_tag200[iR].astype(np.int32, copy=False),
                "tag400_L": cand_tag400[iL].astype(np.int32, copy=False),
                "tag400_R": cand_tag400[iR].astype(np.int32, copy=False),
                "tag192_L": tag192[iL].astype(np.int32, copy=False),
                "tag192_R": tag192[iR].astype(np.int32, copy=False),
                "tag213_L": tag213[iL].astype(np.int32, copy=False),
                "tag213_R": tag213[iR].astype(np.int32, copy=False),
                "tag220_L": tag220[iL].astype(np.int32, copy=False),
                "tag220_R": tag220[iR].astype(np.int32, copy=False),
                "pair_tag200": pair_tag200.astype(np.int32, copy=False),
                "pair_tag192": pair_tag192.astype(np.int32, copy=False),
                "pair_tag213": pair_tag213.astype(np.int32, copy=False),
                "pair_tag220": pair_tag220.astype(np.int32, copy=False),
                "pair_tag400": pair_tag400.astype(np.int32, copy=False),
                "pair_double_tag400": pair_double_tag400.astype(np.int32, copy=False),
                "bx_compatible": bx_compatible.astype(np.int32, copy=False),
                "z_compatible": z_compatible.astype(np.int32, copy=False),
                "t_compatible": t_compatible.astype(np.int32, copy=False),
                "vtx_compatible": vtx_compatible.astype(np.int32, copy=False),
                "bx_z_compatible": (bx_compatible & z_compatible).astype(np.int32, copy=False),
                "bx_t_compatible": (bx_compatible & t_compatible).astype(np.int32, copy=False),
                "bx_vtx_compatible": (bx_compatible & vtx_compatible).astype(np.int32, copy=False),
                "y_compatible_2sigma": y_compatible_2sigma.astype(np.int32, copy=False),
                "y_compatible_3sigma": y_compatible_3sigma.astype(np.int32, copy=False),
                "left_is_signal": left_is_signal.astype(np.int32, copy=False),
                "right_is_signal": right_is_signal.astype(np.int32, copy=False),
                "left_is_minbias": left_is_minbias.astype(np.int32, copy=False),
                "right_is_minbias": right_is_minbias.astype(np.int32, copy=False),
                "minbias_pair_no_central_vertex_overlap": minbias_pair_no_central_vertex_overlap.astype(np.int32, copy=False),
                "minbias_pair_bx_no_central_vertex_overlap": minbias_pair_bx_no_central_vertex_overlap.astype(np.int32, copy=False),
                "require_two_400_tags": np.full(n_pairs, int(args.require_two_400_tags), dtype=np.int32),
                "smear_dz": np.full(n_pairs, int(args.smear_dz), dtype=np.int32),
                "has_pv_t": np.full(n_pairs, int(args.pv_t_res_ps is not None), dtype=np.int32),
                "qq_pdg_abs": np.full(n_pairs, qq_pdg_abs, dtype=np.int32),
                "xi_L": cand_xi[iL].astype(np.float32, copy=False),
                "xi_R": cand_xi[iR].astype(np.float32, copy=False),
                "xi_nominal_L": cand_xi_nominal[iL].astype(np.float32, copy=False),
                "xi_nominal_R": cand_xi_nominal[iR].astype(np.float32, copy=False),
                "mass": mass.astype(np.float32, copy=False),
                "z_truth_L": cand_z[iL].astype(np.float32, copy=False),
                "t_truth_L": cand_t[iL].astype(np.float32, copy=False),
                "z_truth_R": cand_z[iR].astype(np.float32, copy=False),
                "t_truth_R": cand_t[iR].astype(np.float32, copy=False),
                "signal_z_truth": np.full(n_pairs, signal_z, dtype=np.float32),
                "signal_t_truth": np.full(n_pairs, signal_t, dtype=np.float32),
                "pp_z_reco": pp_z_reco.astype(np.float32, copy=False),
                "pp_t_reco": pp_t_reco.astype(np.float32, copy=False),
                "pv_z_reco": np.full(n_pairs, pv_z_reco, dtype=np.float32),
                "pv_t_reco": np.full(n_pairs, pv_t_value, dtype=np.float32),
                "nearest_detected_mb_vertex_dz": nearest_mb_dz.astype(np.float32, copy=False),
                "nearest_detected_mb_vertex_dt": nearest_mb_dt.astype(np.float32, copy=False),
                "zcut": np.full(n_pairs, zcut, dtype=np.float32),
                "tcut": np.full(n_pairs, tcut, dtype=np.float32),
                "bx_zcut": np.full(n_pairs, constants["bx_zcut"], dtype=np.float32),
                "bx_tcut": np.full(n_pairs, constants["bx_tcut"], dtype=np.float32),
                "mass_window": np.full(n_pairs, np.nan if args.mass_window is None else args.mass_window, dtype=np.float32),
                "s": np.full(n_pairs, args.s, dtype=np.float32),
                "ln_xi_ratio": ln_xi_ratio.astype(np.float32, copy=False),
                "yX": yX.astype(np.float32, copy=False),
                "abs_yX": np.abs(yX).astype(np.float32, copy=False),
                "qq_y_truth": np.full(n_pairs, qq_y_truth, dtype=np.float32),
                "qq_y_reco": np.full(n_pairs, qq_y_reco, dtype=np.float32),
                "delta_y": delta_y.astype(np.float32, copy=False),
                "abs_delta_y": abs_delta_y.astype(np.float32, copy=False),
                "sigma_y_pps": sigma_y_pps.astype(np.float32, copy=False),
                "sigma_y_total": sigma_y_total.astype(np.float32, copy=False),
                "ycut_2sigma": ycut_2sigma.astype(np.float32, copy=False),
                "ycut_3sigma": ycut_3sigma.astype(np.float32, copy=False),
            }

        if args.verbose > 1:
            for pair_idx in range(n_pairs):
                source_left_name = "signal" if source_left[pair_idx] == 0 else "minbias"
                source_right_name = "signal" if source_right[pair_idx] == 0 else "minbias"
                print(
                    f"  Pair {pair_idx + 1}: sources=({source_left_name},{source_right_name}), "
                    f"xi=({cand_xi[iL[pair_idx]]:.4f},{cand_xi[iR[pair_idx]]:.4f}), mass={mass[pair_idx]:.2f} GeV, "
                    f"yPPS={yX[pair_idx]:.4f}, delta_y={delta_y[pair_idx]:.4f}, "
                    f"y2 compatible: {y_compatible_2sigma[pair_idx]}, y3 compatible: {y_compatible_3sigma[pair_idx]}, "
                    f"vertex z={pp_z_reco[pair_idx]:.2f} cm, t={pp_t_reco[pair_idx]:.2f} ps, "
                    f"z compatible: {z_compatible[pair_idx]}, t compatible: {t_compatible[pair_idx]}, "
                    f"BX compatible: {bx_compatible[pair_idx]}"
                )

    if args.verbose > 2:
        print_bx(model, protons, tracks, bx)

    return {
        "counters": counters,
        "pps_sigma_z_vertex": constants["pps_sigma_z_vertex"],
        "root_chunk": root_chunk,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze signal+minbias proton pairs and signal-PV compatibility"
    )
    parser.add_argument("-i", "--input", required=True, help="Input minbias NPZ file or directory of NPZ files")
    parser.add_argument("-s", "--signal-in", required=True, help="Signal LHE file or directory of .lhe/.dat files")
    parser.add_argument("--max-bx", type=int, default=None, help="Optional maximum number of BX to process")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of files to process")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional maximum number of signal+minbias combos to process")
    parser.add_argument("--min-bx-pair-candidates", type=int, default=None, help="Require each processed BX/signal combo to have at least this many accepted PPS+mass proton pair candidates")
    parser.add_argument("--min-bx-minbias-pair-candidates", type=int, default=None, help="Require each processed BX/signal combo to have at least this many accepted PPS+mass minbias-only proton pair candidates")
    parser.add_argument("--max-bx-passing-protons", type=int, default=None, help="Require each processed BX/signal combo to have no more than this many PPS-tagged protons")
    parser.add_argument("--max-bx-minbias-passing-protons", type=int, default=None, help="Require each processed BX/signal combo to have no more than this many PPS-tagged minbias protons")
    parser.add_argument("--trk-pt-min", type=float, default=2.0, help="Track (charged particles) proxy pT cut in GeV (default: 0.5)")
    parser.add_argument("--trk-eta-max", type=float, default=2.4, help="Track (charged particles) proxy |eta| cut (default: 2.5)")
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7, help="Gaussian beam spot sigma z in cm")
    parser.add_argument("--pps-time-res-ps", type=float, default=10.0, help="Single-arm timing resolution in ps")
    parser.add_argument("--pv-z-res-cm", type=float, default=0.1, help="Primary vertex z resolution in cm (central detector)")
    parser.add_argument("--pv-t-res-ps", type=float, default=50, help="Primary vertex timing resolution in ps (central detector)")
    parser.add_argument("--nsigma", type=float, default=2.0, help="Compatibility cut in units of sigma_dz")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for vertex and timing smearing")
    parser.add_argument("--mass-window", type=float, default=6, help="Mass half-width in GeV around 125")
    parser.add_argument("--xi-res", type=float, default=0.0003, help="Gaussian fractional xi resolution sigma; 0 disables smearing")
    parser.add_argument("--central-y-res", type=float, default=0.1, help="Central detector qq rapidity resolution sigma; 0 disables smearing")
    parser.add_argument("--s", type=float, default=S_GEV2, help="s value for M=sqrt(xi1*xi2*s)")
    parser.add_argument("--require-two-400-tags", action="store_true", help="Require both protons in a pair to be tagged at the 420m station")
    parser.add_argument("--no-dz-smear", dest="smear_dz", action="store_false", help="Disable timing smearing")
    parser.add_argument("--root-out", default=None, help="Optional ROOT output file with tree SignalMinbiasPairs")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Increase output verbosity")
    parser.set_defaults(smear_dz=True)
    
    args = parser.parse_args()

    if args.max_bx is not None and args.max_bx <= 0:
        raise RuntimeError("--max-bx must be > 0")
    if args.max_combos is not None and args.max_combos <= 0:
        raise RuntimeError("--max-combos must be > 0")
    if args.min_bx_pair_candidates is not None and args.min_bx_pair_candidates < 0:
        raise RuntimeError("--min-bx-pair-candidates must be >= 0")
    if args.min_bx_minbias_pair_candidates is not None and args.min_bx_minbias_pair_candidates < 0:
        raise RuntimeError("--min-bx-minbias-pair-candidates must be >= 0")
    if args.max_bx_passing_protons is not None and args.max_bx_passing_protons < 0:
        raise RuntimeError("--max-bx-passing-protons must be >= 0")
    if args.max_bx_minbias_passing_protons is not None and args.max_bx_minbias_passing_protons < 0:
        raise RuntimeError("--max-bx-minbias-passing-protons must be >= 0")
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
    if args.central_y_res < 0.0:
        raise RuntimeError("--central-y-res must be >= 0")

    signal_files, signal_events, xsec_from_file = load_signal_events(args.signal_in)
    files, protons, tracks, universe = load_minbias(args.input, max_files=args.max_files, max_bx=args.max_bx)
    model = build_interaction_model(universe, protons, tracks, args, verbosity=args.verbose)

    selected_bx = np.unique(model["universe"][:, 0])
    n_available_bx = len(selected_bx)
    n_combos = min(len(signal_events), n_available_bx)
    if args.max_combos is not None:
        n_combos = min(n_combos, args.max_combos)
    selected_bx = selected_bx[:n_combos]

    print(f"Loaded {len(signal_events)} signal events from {len(signal_files)} files")
    if xsec_from_file is not None:
        print(f"Signal cross section from file: {xsec_from_file} pb")
    print(f"Loaded {n_available_bx} unique minbias BX from {len(files)} files")
    print(f"Processing {n_combos} signal+minbias combos")

    if n_combos == 0:
        raise RuntimeError("No signal+minbias combos available to process.")

    proton_model_idx = proton_interaction_indices(protons, model)
    unique_pbx, proton_groups = group_indices_by_bx(protons["bx_id"])
    proton_group_by_bx = {
        int(bx): group for bx, group in zip(unique_pbx, proton_groups)
    }
    unique_vbx, vertex_groups = group_indices_by_bx(model["universe"][:, 0])
    vertex_group_by_bx = {
        int(bx): group for bx, group in zip(unique_vbx, vertex_groups)
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
        "central_zcut": args.nsigma * math.sqrt(2.0) * args.pv_z_res_cm,
        "central_tcut": (
            args.nsigma * math.sqrt(2.0) * args.pv_t_res_ps
            if args.pv_t_res_ps is not None
            else None
        ),
        "bx_zcut": args.nsigma * args.beam_sigma_z_cm,
        "bx_tcut": args.nsigma * args.beam_sigma_z_cm / C_CM_PER_PS,
    }
    station_200_counter_keys = tuple(
        key
        for station in STATIONS_200
        for key in (
            f"n_pp_tag{station}",
            f"n_pp_tag{station}_signal_only",
            f"n_pp_tag{station}_minbias_only",
        )
    )
    station_200_total_summary_items = [
        item
        for station in STATIONS_200
        for item in (
            (f"passing proton pairs with any {station}m station tag", f"n_pp_tag{station}"),
            (f"signal-only passing proton pairs with any {station}m station tag", f"n_pp_tag{station}_signal_only"),
            (f"minbias-only passing proton pairs with any {station}m station tag", f"n_pp_tag{station}_minbias_only"),
        )
    ]
    station_200_combo_summary_items = [
        item
        for station in STATIONS_200
        for item in (
            (f"Combos with >=1 passing pair with any {station}m station tag", f"n_pp_tag{station}"),
            (f"Combos with >=1 passing signal-only pair with any {station}m station tag", f"n_pp_tag{station}_signal_only"),
            (f"Combos with >=1 passing minbias-only pair with any {station}m station tag", f"n_pp_tag{station}_minbias_only"),
        )
    ]
    pair_counter_keys = (
        "n_candidate_protons",
        "n_tagged_protons",
        "n_minbias_tagged_protons",
        "n_minbias_vertices",
        "n_minbias_vertices_with_tracks",
        "n_minbias_vertices_z_overlap",
        "n_minbias_vertices_t_overlap",
        "n_minbias_vertices_vtx_overlap",
        "n_pp",
        "n_pp_z_compatible",
        "n_pp_t_compatible",
        "n_pp_vtx_compatible",
        "n_pp_bx_compatible",
        "n_pp_bx_z_compatible",
        "n_pp_bx_t_compatible",
        "n_pp_bx_vtx_compatible",
        "n_pp_y_compatible_2sigma",
        "n_pp_y_compatible_3sigma",
        "n_pp_z_y_compatible_2sigma",
        "n_pp_z_y_compatible_3sigma",
        "n_pp_vtx_y_compatible_2sigma",
        "n_pp_vtx_y_compatible_3sigma",
        "n_pp_tag200",
        "n_pp_tag200_signal_only",
        "n_pp_tag200_minbias_only",
        *station_200_counter_keys,
        "n_minbias_pp_bx_compatible",
        "n_minbias_pp_no_central_vertex_overlap",
        "n_minbias_pp_bx_no_central_vertex_overlap",
    )
    combo_counter_keys = (
        "n_candidate_protons",
        "n_tagged_protons",
        "n_minbias_tagged_protons",
        "n_minbias_vertices",
        "n_minbias_vertices_with_tracks",
        "n_minbias_vertices_z_overlap",
        "n_minbias_vertices_t_overlap",
        "n_minbias_vertices_vtx_overlap",
        "n_pp",
        "n_pp_z_compatible",
        "n_pp_t_compatible",
        "n_pp_vtx_compatible",
        "n_pp_bx_compatible",
        "n_pp_bx_z_compatible",
        "n_pp_bx_t_compatible",
        "n_pp_bx_vtx_compatible",
        "n_pp_y_compatible_2sigma",
        "n_pp_y_compatible_3sigma",
        "n_pp_z_y_compatible_2sigma",
        "n_pp_z_y_compatible_3sigma",
        "n_pp_vtx_y_compatible_2sigma",
        "n_pp_vtx_y_compatible_3sigma",
        "n_pp_tag200",
        "n_pp_tag200_signal_only",
        "n_pp_tag200_minbias_only",
        *station_200_counter_keys,
        "n_minbias_pp_bx_compatible",
        "n_minbias_pp_no_central_vertex_overlap",
        "n_minbias_pp_bx_no_central_vertex_overlap",
        "minbias_vertex_z_overlap_pp_accept_mass_any",
        "minbias_vertex_t_overlap_pp_accept_mass_any",
        "minbias_vertex_vtx_overlap_pp_accept_mass_any",
        "minbias_vertex_pp_z_overlap_signal_pair_any",
        "minbias_vertex_pp_t_overlap_signal_pair_any",
        "minbias_vertex_pp_vtx_overlap_signal_pair_any",
        "passing_any",
        "passing_non_signal",
        "passing_signal_only",
        "passing_minbias_only",
        "passing_mixed",
        "z_any",
        "z_non_signal",
        "z_signal_only",
        "z_minbias_only",
        "z_mixed",
        "t_any",
        "t_non_signal",
        "t_signal_only",
        "t_minbias_only",
        "t_mixed",
        "vtx_any",
        "vtx_non_signal",
        "vtx_signal_only",
        "vtx_minbias_only",
        "vtx_mixed",
        "bx_any",
        "bx_non_signal",
        "bx_signal_only",
        "bx_minbias_only",
        "bx_mixed",
        "bx_z_any",
        "bx_z_non_signal",
        "bx_z_signal_only",
        "bx_z_minbias_only",
        "bx_z_mixed",
        "bx_t_any",
        "bx_t_non_signal",
        "bx_t_signal_only",
        "bx_t_minbias_only",
        "bx_t_mixed",
        "bx_vtx_any",
        "bx_vtx_non_signal",
        "bx_vtx_signal_only",
        "bx_vtx_minbias_only",
        "bx_vtx_mixed",
        "y2_any",
        "y2_non_signal",
        "y2_signal_only",
        "y2_minbias_only",
        "y2_mixed",
        "y3_any",
        "y3_non_signal",
        "y3_signal_only",
        "y3_minbias_only",
        "y3_mixed",
        "z_y2_any",
        "z_y2_non_signal",
        "z_y2_signal_only",
        "z_y2_minbias_only",
        "z_y2_mixed",
        "z_y3_any",
        "z_y3_non_signal",
        "z_y3_signal_only",
        "z_y3_minbias_only",
        "z_y3_mixed",
        "vtx_y2_any",
        "vtx_y2_non_signal",
        "vtx_y2_signal_only",
        "vtx_y2_minbias_only",
        "vtx_y2_mixed",
        "vtx_y3_any",
        "vtx_y3_non_signal",
        "vtx_y3_signal_only",
        "vtx_y3_minbias_only",
        "vtx_y3_mixed",
        "minbias_vertex_z_overlap_any",
        "minbias_vertex_t_overlap_any",
        "minbias_vertex_vtx_overlap_any",
        "minbias_pp_no_central_vertex_overlap_any",
        "minbias_pp_bx_no_central_vertex_overlap_any",
    )
    pair_totals = {key: 0 for key in pair_counter_keys}
    combo_totals = {key: 0 for key in combo_counter_keys}
    root_chunks = empty_root_chunks() if args.root_out else None
    skipped_signal = 0
    skipped_pair_requirement = 0
    skipped_minbias_pair_requirement = 0
    skipped_passing_proton_requirement = 0
    skipped_minbias_passing_proton_requirement = 0

    for combo_id, bx in enumerate(selected_bx):
        bx = int(bx)
        if combo_id % 100 == 0:
            print(f"Analyzing combo {combo_id} (BX {bx})...")
        idx_bx = proton_group_by_bx.get(bx)
        if idx_bx is None:
            idx_bx = np.empty(0, dtype=np.int64)
        mb_vertex_idx = vertex_group_by_bx.get(bx)
        if mb_vertex_idx is None:
            mb_vertex_idx = np.empty(0, dtype=np.int64)
        bx_protons = {name: arr[idx_bx] for name, arr in protons.items()}
        bx_result = analyze_combo(
            combo_id,
            bx,
            combo_id,
            signal_events[combo_id],
            model,
            bx_protons,
            proton_model_idx[idx_bx],
            mb_vertex_idx,
            constants,
            args,
            protons,
            tracks,
        )
        if bx_result is None:
            skipped_signal += 1
            continue
        if (
            args.min_bx_pair_candidates is not None
            and bx_result["counters"]["n_pp"] < args.min_bx_pair_candidates
        ):
            skipped_pair_requirement += 1
            continue
        if (
            args.min_bx_minbias_pair_candidates is not None
            and bx_result["counters"]["passing_minbias_only"] < args.min_bx_minbias_pair_candidates
        ):
            skipped_minbias_pair_requirement += 1
            continue
        if (
            args.max_bx_passing_protons is not None
            and bx_result["counters"]["n_tagged_protons"] > args.max_bx_passing_protons
        ):
            skipped_passing_proton_requirement += 1
            continue
        if (
            args.max_bx_minbias_passing_protons is not None
            and bx_result["counters"]["n_minbias_tagged_protons"] > args.max_bx_minbias_passing_protons
        ):
            skipped_minbias_passing_proton_requirement += 1
            continue

        for key in pair_counter_keys:
            pair_totals[key] += int(bx_result["counters"][key])
        for key in combo_counter_keys:
            combo_totals[key] += 1 if bx_result["counters"][key] > 0 else 0
        append_root_chunk(root_chunks, bx_result["root_chunk"])

    n_processed = n_combos
    if n_processed <= 0:
        raise RuntimeError("No valid signal+minbias combos were processed.")

    if args.root_out:
        write_root(args.root_out, root_chunks)

    def print_combo_count(label, key):
        count = combo_totals[key]
        frac = count / n_processed
        err = math.sqrt(count) * F_COLL_AVG_HZ / n_processed / 1000 if count > 0 else 0.0
        rate = count * F_COLL_AVG_HZ / n_processed / 1000
        print(f"{label}: {count} ({frac:.6f}), rate={rate:.2f} +/- {err:.2f} kHz")

    def print_totals(items):
        print("Total counts:")
        for label, key in items:
            print(f"  {label}: {pair_totals[key]}")

    def print_combo_counts(title, items):
        print(title)
        for label, key in items:
            print_combo_count(label, key)

    print(f"Summary of results across processed combos ({n_processed}):")
    if skipped_signal:
        print(f"Signal events skipped (malformed): {skipped_signal}")
    if skipped_pair_requirement:
        print(
            f"Combos skipped by --min-bx-pair-candidates "
            f"({args.min_bx_pair_candidates}): {skipped_pair_requirement}"
        )
    if skipped_minbias_pair_requirement:
        print(
            f"Combos skipped by --min-bx-minbias-pair-candidates "
            f"({args.min_bx_minbias_pair_candidates}): {skipped_minbias_pair_requirement}"
        )
    if skipped_passing_proton_requirement:
        print(
            f"Combos skipped by --max-bx-passing-protons "
            f"({args.max_bx_passing_protons}): {skipped_passing_proton_requirement}"
        )
    if skipped_minbias_passing_proton_requirement:
        print(
            f"Combos skipped by --max-bx-minbias-passing-protons "
            f"({args.max_bx_minbias_passing_protons}): {skipped_minbias_passing_proton_requirement}"
        )

    total_summary_items = [
        ("candidate protons", "n_candidate_protons"),
        ("tagged protons", "n_tagged_protons"),
        ("minbias tagged protons", "n_minbias_tagged_protons"),
        ("minbias reco vertices", "n_minbias_vertices"),
        ("minbias reco vertices with tracks", "n_minbias_vertices_with_tracks"),
        ("minbias reco vertices z-overlapping signal PV", "n_minbias_vertices_z_overlap"),
        ("minbias reco vertices t-overlapping signal PV", "n_minbias_vertices_t_overlap"),
        ("minbias reco vertices z+t-overlapping signal PV", "n_minbias_vertices_vtx_overlap"),
        ("passing proton pairs", "n_pp"),
        ("passing proton pairs with z compatible with signal PV", "n_pp_z_compatible"),
        ("passing proton pairs with t compatible with signal PV", "n_pp_t_compatible"),
        ("passing proton pairs with both z and t compatible with signal PV", "n_pp_vtx_compatible"),
        ("passing proton pairs with PPS z and t compatible with bunch crossing", "n_pp_bx_compatible"),
        ("passing proton pairs with PPS-central qq rapidity compatible at 2 sigma", "n_pp_y_compatible_2sigma"),
        ("passing proton pairs with PPS-central qq rapidity compatible at 3 sigma", "n_pp_y_compatible_3sigma"),
        ("passing proton pairs with signal PV z and rapidity compatible at 2 sigma", "n_pp_z_y_compatible_2sigma"),
        ("passing proton pairs with signal PV z and rapidity compatible at 3 sigma", "n_pp_z_y_compatible_3sigma"),
        ("passing proton pairs with signal PV z+t and rapidity compatible at 2 sigma", "n_pp_vtx_y_compatible_2sigma"),
        ("passing proton pairs with signal PV z+t and rapidity compatible at 3 sigma", "n_pp_vtx_y_compatible_3sigma"),
        ("passing proton pairs with any 200m station tag", "n_pp_tag200"),
        ("signal-only passing proton pairs with any 200m station tag", "n_pp_tag200_signal_only"),
        ("minbias-only passing proton pairs with any 200m station tag", "n_pp_tag200_minbias_only"),
        *station_200_total_summary_items,
        ("minbias-only passing proton pairs with PPS z and t compatible with bunch crossing", "n_minbias_pp_bx_compatible"),
        ("passing proton pairs compatible with both bunch crossing and signal PV z", "n_pp_bx_z_compatible"),
        ("passing proton pairs compatible with both bunch crossing and signal PV t", "n_pp_bx_t_compatible"),
        ("passing proton pairs compatible with both bunch crossing and signal PV z+t", "n_pp_bx_vtx_compatible"),
        ("minbias-only passing proton pairs with no detected minbias central vertex overlap", "n_minbias_pp_no_central_vertex_overlap"),
        (
            "minbias-only passing proton pairs with bunch-crossing compatibility and no detected minbias central vertex overlap",
            "n_minbias_pp_bx_no_central_vertex_overlap",
        ),
    ]
    combo_summary_groups = [
        (
            "Expected combo rates for 31.6 MHz collision rate:",
            [
                ("Combos with >=1 minbias reco vertex z-overlapping signal PV", "minbias_vertex_z_overlap_any"),
                ("Combos with >=1 minbias reco vertex t-overlapping signal PV", "minbias_vertex_t_overlap_any"),
                ("Combos with >=1 minbias reco vertex z+t-overlapping signal PV", "minbias_vertex_vtx_overlap_any"),
            ],
        ),
        (
            "Combos with minbias reco vertex overlap and proton pairs passing acceptance and mass-window cuts:",
            [
                (
                    "Combos with >=1 minbias reco vertex z-overlapping signal PV and >=1 pair passing acceptance and mass-window cuts",
                    "minbias_vertex_z_overlap_pp_accept_mass_any",
                ),
                (
                    "Combos with >=1 minbias reco vertex t-overlapping signal PV and >=1 pair passing acceptance and mass-window cuts",
                    "minbias_vertex_t_overlap_pp_accept_mass_any",
                ),
                (
                    "Combos with >=1 minbias reco vertex z+t-overlapping signal PV and >=1 pair passing acceptance and mass-window cuts",
                    "minbias_vertex_vtx_overlap_pp_accept_mass_any",
                ),
            ],
        ),
        (
            "Combos with minbias reco vertex overlap with PPS-reconstructed signal-pair PV:",
            [
                (
                    "Combos with >=1 minbias reco vertex z-overlapping PPS-reconstructed signal-pair PV",
                    "minbias_vertex_pp_z_overlap_signal_pair_any",
                ),
                (
                    "Combos with >=1 minbias reco vertex t-overlapping PPS-reconstructed signal-pair PV",
                    "minbias_vertex_pp_t_overlap_signal_pair_any",
                ),
                (
                    "Combos with >=1 minbias reco vertex z+t-overlapping PPS-reconstructed signal-pair PV",
                    "minbias_vertex_pp_vtx_overlap_signal_pair_any",
                ),
            ],
        ),
        (
            "Combos with minbias-only PPS pairs and no detected minbias central vertex overlap:",
            [
                ("Combos with >=1 minbias-only passing PPS pair and no detected minbias central vertex overlap", "minbias_pp_no_central_vertex_overlap_any"),
                (
                    "Combos with >=1 minbias-only passing PPS pair compatible with bunch crossing and no detected minbias central vertex overlap",
                    "minbias_pp_bx_no_central_vertex_overlap_any",
                ),
            ],
        ),
        (
            "Passing pair combo summaries:",
            [
                ("Combos with >=1 passing pair", "passing_any"),
                ("Combos with >=1 passing pair and >=1 non-signal proton", "passing_non_signal"),
                ("Combos with >=1 passing signal-only pair", "passing_signal_only"),
                ("Combos with >=1 passing minbias-only pair", "passing_minbias_only"),
                ("Combos with >=1 passing mixed signal+minbias pair", "passing_mixed"),
                ("Combos with >=1 passing pair with any 200m station tag", "n_pp_tag200"),
                ("Combos with >=1 passing signal-only pair with any 200m station tag", "n_pp_tag200_signal_only"),
                ("Combos with >=1 passing minbias-only pair with any 200m station tag", "n_pp_tag200_minbias_only"),
                *station_200_combo_summary_items,
            ],
        ),
        (
            "PPS-central qq rapidity-compatible pair combo summaries:",
            [
                ("Combos with >=1 passing pair rapidity-compatible at 2 sigma", "y2_any"),
                ("Combos with >=1 passing non-signal pair rapidity-compatible at 2 sigma", "y2_non_signal"),
                ("Combos with >=1 passing signal-only pair rapidity-compatible at 2 sigma", "y2_signal_only"),
                ("Combos with >=1 passing minbias-only pair rapidity-compatible at 2 sigma", "y2_minbias_only"),
                ("Combos with >=1 passing mixed pair rapidity-compatible at 2 sigma", "y2_mixed"),
                ("Combos with >=1 passing pair rapidity-compatible at 3 sigma", "y3_any"),
                ("Combos with >=1 passing non-signal pair rapidity-compatible at 3 sigma", "y3_non_signal"),
                ("Combos with >=1 passing signal-only pair rapidity-compatible at 3 sigma", "y3_signal_only"),
                ("Combos with >=1 passing minbias-only pair rapidity-compatible at 3 sigma", "y3_minbias_only"),
                ("Combos with >=1 passing mixed pair rapidity-compatible at 3 sigma", "y3_mixed"),
            ],
        ),
        (
            "Signal-PV z and rapidity-compatible pair combo summaries:",
            [
                ("Combos with >=1 passing pair z-compatible and rapidity-compatible at 2 sigma", "z_y2_any"),
                ("Combos with >=1 passing non-signal pair z-compatible and rapidity-compatible at 2 sigma", "z_y2_non_signal"),
                ("Combos with >=1 passing signal-only pair z-compatible and rapidity-compatible at 2 sigma", "z_y2_signal_only"),
                ("Combos with >=1 passing minbias-only pair z-compatible and rapidity-compatible at 2 sigma", "z_y2_minbias_only"),
                ("Combos with >=1 passing mixed pair z-compatible and rapidity-compatible at 2 sigma", "z_y2_mixed"),
                ("Combos with >=1 passing pair z-compatible and rapidity-compatible at 3 sigma", "z_y3_any"),
                ("Combos with >=1 passing non-signal pair z-compatible and rapidity-compatible at 3 sigma", "z_y3_non_signal"),
                ("Combos with >=1 passing signal-only pair z-compatible and rapidity-compatible at 3 sigma", "z_y3_signal_only"),
                ("Combos with >=1 passing minbias-only pair z-compatible and rapidity-compatible at 3 sigma", "z_y3_minbias_only"),
                ("Combos with >=1 passing mixed pair z-compatible and rapidity-compatible at 3 sigma", "z_y3_mixed"),
            ],
        ),
        (
            "Signal-PV z+t and rapidity-compatible pair combo summaries:",
            [
                ("Combos with >=1 passing pair z+t-compatible and rapidity-compatible at 2 sigma", "vtx_y2_any"),
                ("Combos with >=1 passing non-signal pair z+t-compatible and rapidity-compatible at 2 sigma", "vtx_y2_non_signal"),
                ("Combos with >=1 passing signal-only pair z+t-compatible and rapidity-compatible at 2 sigma", "vtx_y2_signal_only"),
                ("Combos with >=1 passing minbias-only pair z+t-compatible and rapidity-compatible at 2 sigma", "vtx_y2_minbias_only"),
                ("Combos with >=1 passing mixed pair z+t-compatible and rapidity-compatible at 2 sigma", "vtx_y2_mixed"),
                ("Combos with >=1 passing pair z+t-compatible and rapidity-compatible at 3 sigma", "vtx_y3_any"),
                ("Combos with >=1 passing non-signal pair z+t-compatible and rapidity-compatible at 3 sigma", "vtx_y3_non_signal"),
                ("Combos with >=1 passing signal-only pair z+t-compatible and rapidity-compatible at 3 sigma", "vtx_y3_signal_only"),
                ("Combos with >=1 passing minbias-only pair z+t-compatible and rapidity-compatible at 3 sigma", "vtx_y3_minbias_only"),
                ("Combos with >=1 passing mixed pair z+t-compatible and rapidity-compatible at 3 sigma", "vtx_y3_mixed"),
            ],
        ),
        (
            "Z-compatible pair combo summaries:",
            [
                ("Combos with >=1 passing pair z-compatible with signal PV", "z_any"),
                ("Combos with >=1 passing non-signal pair z-compatible with signal PV", "z_non_signal"),
                ("Combos with >=1 passing signal-only pair z-compatible with signal PV", "z_signal_only"),
                ("Combos with >=1 passing minbias-only pair z-compatible with signal PV", "z_minbias_only"),
                ("Combos with >=1 passing mixed pair z-compatible with signal PV", "z_mixed"),
            ],
        ),
        (
            "z+t-compatible pair combo summaries:",
            [
                ("Combos with >=1 passing pair z+t-compatible with signal PV", "vtx_any"),
                ("Combos with >=1 passing non-signal pair z+t-compatible with signal PV", "vtx_non_signal"),
                ("Combos with >=1 passing signal-only pair z+t-compatible with signal PV", "vtx_signal_only"),
                ("Combos with >=1 passing minbias-only pair z+t-compatible with signal PV", "vtx_minbias_only"),
                ("Combos with >=1 passing mixed pair z+t-compatible with signal PV", "vtx_mixed"),
            ],
        ),
        (
            "z+t-compatible bunch-crossing pair combo summaries:",
            [
                ("Combos with >=1 passing pair z+t-compatible with bunch crossing", "bx_any"),
                ("Combos with >=1 passing non-signal pair z+t-compatible with bunch crossing", "bx_non_signal"),
                ("Combos with >=1 passing signal-only pair z+t-compatible with bunch crossing", "bx_signal_only"),
                ("Combos with >=1 passing minbias-only pair z+t-compatible with bunch crossing", "bx_minbias_only"),
                ("Combos with >=1 passing mixed pair z+t-compatible with bunch crossing", "bx_mixed"),
            ],
        ),
        (
            "Bunch-crossing and signal-PV compatible pair combo summaries:",
            [
                ("Combos with >=1 passing pair compatible with bunch crossing and signal PV z", "bx_z_any"),
                ("Combos with >=1 passing non-signal pair compatible with bunch crossing and signal PV z", "bx_z_non_signal"),
                ("Combos with >=1 passing signal-only pair compatible with bunch crossing and signal PV z", "bx_z_signal_only"),
                ("Combos with >=1 passing minbias-only pair compatible with bunch crossing and signal PV z", "bx_z_minbias_only"),
                ("Combos with >=1 passing mixed pair compatible with bunch crossing and signal PV z", "bx_z_mixed"),
                ("Combos with >=1 passing pair compatible with bunch crossing and signal PV t", "bx_t_any"),
                ("Combos with >=1 passing non-signal pair compatible with bunch crossing and signal PV t", "bx_t_non_signal"),
                ("Combos with >=1 passing signal-only pair compatible with bunch crossing and signal PV t", "bx_t_signal_only"),
                ("Combos with >=1 passing minbias-only pair compatible with bunch crossing and signal PV t", "bx_t_minbias_only"),
                ("Combos with >=1 passing mixed pair compatible with bunch crossing and signal PV t", "bx_t_mixed"),
                ("Combos with >=1 passing pair compatible with bunch crossing and signal PV z+t", "bx_vtx_any"),
                ("Combos with >=1 passing non-signal pair compatible with bunch crossing and signal PV z+t", "bx_vtx_non_signal"),
                ("Combos with >=1 passing signal-only pair compatible with bunch crossing and signal PV z+t", "bx_vtx_signal_only"),
                ("Combos with >=1 passing minbias-only pair compatible with bunch crossing and signal PV z+t", "bx_vtx_minbias_only"),
                ("Combos with >=1 passing mixed pair compatible with bunch crossing and signal PV z+t", "bx_vtx_mixed"),
            ],
        ),
    ]

    print_totals(total_summary_items)
    print("")
    for title, items in combo_summary_groups:
        print_combo_counts(title, items)
        print("")

if __name__ == "__main__":
    main()
