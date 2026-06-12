import argparse
import glob
import itertools
import math
import os

import numpy as np


S_GEV2 = 13600.0**2
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
                    if parts:
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
                    raise RuntimeError(f"Unexpected event header in {filename}: {line}")
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
            total = sum(len(part) for part in int_parts)
            print(f"Constructed interaction universe from {filename} with {len(bx_parts)} BX and total {total} interactions")
            return np.column_stack((np.concatenate(bx_parts), np.concatenate(int_parts)))

    proton_keys = np.column_stack(
        (np.asarray(data["bx_id"], dtype=np.int64), np.asarray(data["interaction_id"], dtype=np.int64))
    )
    track_keys = np.column_stack(
        (np.asarray(data["trk_bx_id"], dtype=np.int64), np.asarray(data["trk_interaction_id"], dtype=np.int64))
    )
    keys = np.concatenate((proton_keys, track_keys), axis=0)
    return np.unique(keys, axis=0) if keys.size else np.empty((0, 2), dtype=np.int64)


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
            universe_parts.append(interaction_universe_from_file(data, filename))

    protons = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in proton_parts.items()
    }
    tracks = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in track_parts.items()
    }
    universe = (
        np.unique(np.concatenate(universe_parts, axis=0), axis=0)
        if universe_parts
        else np.empty((0, 2), dtype=np.int64)
    )

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
        np.add.at(n_protons, pos[valid], 1)

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
        pt = np.asarray(tracks["trk_pt"][track_mask], dtype=np.float64)[valid]
        np.add.at(sum_pt2, pos[valid], pt * pt)
        np.add.at(n_tracks, pos[valid], 1)

    if verbosity > 1:
        print(f"Interactions in universe: {universe.shape[0]}")
        print(f"Tracks after pT and eta cuts: {int(np.sum(track_mask))}")

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


def cartesian_pair_indices(left_idx, right_idx):
    if left_idx.size == 0 or right_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.repeat(left_idx, right_idx.size), np.tile(right_idx, left_idx.size)


def pps_sigma_z_cm(pps_time_res_ps):
    return 0.5 * C_CM_PER_PS * math.sqrt(2.0) * pps_time_res_ps


def station_tag(xi, station):
    xi_min, xi_max = STATION_XI[station]
    return (xi >= xi_min) & (xi < xi_max)


def station_tags(xi):
    tag200 = np.zeros(np.asarray(xi).shape, dtype=bool)
    for station in STATIONS_200:
        tag200 |= station_tag(xi, station)
    tag400 = station_tag(xi, "420")
    return tag200, tag400


def smear_xi(xi, xi_res, rng):
    if xi_res <= 0.0 or xi.size == 0:
        return xi
    return xi + rng.normal(loc=0.0, scale=xi_res, size=xi.size)


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
        xi = (beam_E - proton["E"]) / beam_E if beam_E is not None and beam_E > 0.0 else 0.0
        out.append(
            {
                "proton_idx": proton_idx,
                "side": side,
                "xi": float(xi),
            }
        )
    return out


def build_candidates(combo_id, bx, signal_event, model, bx_protons, bx_proton_model_idx, args):
    rng = np.random.default_rng(args.seed + combo_id * 1000)
    xi_rng = np.random.default_rng(args.seed + combo_id * 1000 + 2)
    signal_protons = signal_protons_from_event(signal_event)
    if signal_protons is None:
        return None

    signal_z = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm)
    signal_t = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm / C_CM_PER_PS)

    sig_xi_nominal = np.asarray([p["xi"] for p in signal_protons], dtype=np.float64)
    sig_xi = smear_xi(sig_xi_nominal, args.xi_res, xi_rng)
    sig_side = np.asarray([p["side"] for p in signal_protons], dtype=np.int32)
    sig_tag200, sig_tag400 = station_tags(sig_xi_nominal)
    sig_station_tags = {
        station: station_tag(sig_xi_nominal, station)
        for station in STATIONS_200
    }

    bx_xi_nominal = np.asarray(bx_protons["xi"], dtype=np.float64)
    bx_xi = smear_xi(bx_xi_nominal, args.xi_res, xi_rng)
    bx_side = np.asarray(bx_protons["side"], dtype=np.int32)
    bx_tag200, bx_tag400 = station_tags(bx_xi_nominal)
    bx_station_tags = {
        station: station_tag(bx_xi_nominal, station)
        for station in STATIONS_200
    }

    return {
        "xi": np.concatenate((sig_xi, bx_xi)),
        "xi_nominal": np.concatenate((sig_xi_nominal, bx_xi_nominal)),
        "side": np.concatenate((sig_side, bx_side)),
        "tag200": np.concatenate((sig_tag200, bx_tag200)),
        "tag400": np.concatenate((sig_tag400, bx_tag400)),
        "tag192": np.concatenate((sig_station_tags["192"], bx_station_tags["192"])),
        "tag213": np.concatenate((sig_station_tags["213"], bx_station_tags["213"])),
        "tag220": np.concatenate((sig_station_tags["220"], bx_station_tags["220"])),
        "z": np.concatenate((np.full(sig_xi.size, signal_z), model["z"][bx_proton_model_idx])),
        "t": np.concatenate((np.full(sig_xi.size, signal_t), model["t"][bx_proton_model_idx])),
        "source": np.concatenate((np.zeros(sig_xi.size, dtype=np.int8), np.ones(bx_xi.size, dtype=np.int8))),
    }


def build_detected_minbias_vertices(combo_id, bx, model, args):
    vertex_rng = np.random.default_rng(args.seed + combo_id * 1000 + 1)
    mb_vertex_idx = np.where(model["universe"][:, 0] == bx)[0]
    z_reco = model["z"][mb_vertex_idx] + vertex_rng.normal(
        loc=0.0,
        scale=args.pv_z_res_cm,
        size=mb_vertex_idx.size,
    )
    if args.pv_t_res_ps is not None:
        t_reco = model["t"][mb_vertex_idx] + vertex_rng.normal(
            loc=0.0,
            scale=args.pv_t_res_ps,
            size=mb_vertex_idx.size,
        )
    else:
        t_reco = None

    has_tracks = model["n_tracks"][mb_vertex_idx] > 0
    return {
        "z": z_reco[has_tracks],
        "t": t_reco[has_tracks] if t_reco is not None else None,
        "n_all": int(mb_vertex_idx.size),
        "n_detected": int(np.sum(has_tracks)),
    }


def build_pairs(combo_id, candidates, detected_vertices, constants, args):
    tagged = candidates["tag200"] | candidates["tag400"]
    cand_left = np.where((candidates["side"] == -1) & tagged)[0]
    cand_right = np.where((candidates["side"] == +1) & tagged)[0]
    iL, iR = cartesian_pair_indices(cand_left, cand_right)

    if iL.size:
        keep = candidates["tag400"][iL] | candidates["tag400"][iR]
        xi_prod = candidates["xi"][iL] * candidates["xi"][iR]
        keep &= xi_prod > 0.0
        iL = iL[keep]
        iR = iR[keep]
        xi_prod = xi_prod[keep]
    else:
        xi_prod = np.empty(0, dtype=np.float64)

    if iL.size:
        order = np.lexsort((np.maximum(iL, iR), np.minimum(iL, iR)))
        iL = iL[order]
        iR = iR[order]
        xi_prod = xi_prod[order]

    mass = np.sqrt(xi_prod * args.s)
    if args.mass_window is not None:
        keep = (mass >= M_HIGGS_GEV - args.mass_window) & (mass <= M_HIGGS_GEV + args.mass_window)
        iL = iL[keep]
        iR = iR[keep]
        mass = mass[keep]

    n_pairs = int(iL.size)
    pair_flags = {
        "source_L": candidates["source"][iL] if n_pairs else np.empty(0, dtype=np.int8),
        "source_R": candidates["source"][iR] if n_pairs else np.empty(0, dtype=np.int8),
        "any200": np.zeros(n_pairs, dtype=bool),
        "double400": np.zeros(n_pairs, dtype=bool),
        "bx_z": np.zeros(n_pairs, dtype=bool),
        "bx_t": np.zeros(n_pairs, dtype=bool),
        "bx_zt": np.zeros(n_pairs, dtype=bool),
        "central_overlap": np.zeros(n_pairs, dtype=bool),
        "no_central_overlap": np.zeros(n_pairs, dtype=bool),
    }
    if n_pairs == 0:
        return pair_flags

    pair_flags["any200"] = candidates["tag200"][iL] | candidates["tag200"][iR]
    pair_flags["double400"] = candidates["tag400"][iL] & candidates["tag400"][iR]

    pL_t_truth = candidates["t"][iL] + candidates["z"][iL] / C_CM_PER_PS
    pR_t_truth = candidates["t"][iR] - candidates["z"][iR] / C_CM_PER_PS
    if args.smear_dz:
        rng = np.random.default_rng(args.seed + combo_id * 1000)
        smear = rng.normal(loc=0.0, scale=args.pps_time_res_ps, size=(n_pairs, 2))
        pL_t_reco = pL_t_truth + smear[:, 0]
        pR_t_reco = pR_t_truth + smear[:, 1]
    else:
        pL_t_reco = pL_t_truth
        pR_t_reco = pR_t_truth

    pp_z_reco = 0.5 * (pL_t_reco - pR_t_reco) * C_CM_PER_PS
    pp_t_reco = 0.5 * (pL_t_reco + pR_t_reco)

    pair_flags["bx_z"] = np.abs(pp_z_reco) <= constants["bx_zcut"]
    pair_flags["bx_t"] = np.abs(pp_t_reco) <= constants["bx_tcut"]
    pair_flags["bx_zt"] = pair_flags["bx_z"] & pair_flags["bx_t"]

    detected_z = detected_vertices["z"]
    if detected_z.size:
        central_z_overlap = np.abs(detected_z[np.newaxis, :] - pp_z_reco[:, np.newaxis]) <= constants["zcut"]
        if detected_vertices["t"] is not None:
            central_t_overlap = np.abs(detected_vertices["t"][np.newaxis, :] - pp_t_reco[:, np.newaxis]) <= constants["tcut"]
            pair_flags["central_overlap"] = np.any(central_z_overlap & central_t_overlap, axis=1)
        else:
            pair_flags["central_overlap"] = np.any(central_z_overlap, axis=1)
    pair_flags["no_central_overlap"] = ~pair_flags["central_overlap"]
    return pair_flags


def selection_specs():
    station_flags = [
        (),
        ("any200",),
        ("double400",),
        ("any200", "double400"),
    ]
    bx_flags = [
        (),
        ("bx_z",),
        ("bx_t",),
        ("bx_zt",),
        ("bx_z", "bx_t"),
        ("bx_z", "bx_zt"),
        ("bx_t", "bx_zt"),
        ("bx_z", "bx_t", "bx_zt"),
    ]
    central_modes = (None, "central_overlap", "no_central_overlap")
    max_protons_options = (None, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    min_protons_400_options = (None, 1, 2, 3, 4, 5, 6)
    specs = []
    for min_pairs in (1, 2, 3):
        for max_protons in max_protons_options:
            for min_protons_400 in min_protons_400_options:
                for central in central_modes:
                    for station in station_flags:
                        for bx in bx_flags:
                            flags = []
                            if central is not None:
                                flags.append(central)
                            flags.extend(station)
                            flags.extend(bx)
                            label = f">={min_pairs} pair"
                            if max_protons is not None:
                                label += f" + <={max_protons} protons"
                            if min_protons_400 is not None:
                                label += f" + >={min_protons_400} 420m protons"
                            if flags:
                                label += " + " + " + ".join(flags)
                            specs.append((min_pairs, max_protons, min_protons_400, tuple(flags), label))
    return specs


def count_passing_pairs(pair_flags, source_mask, flags):
    if source_mask.size == 0:
        return 0
    keep = source_mask.copy()
    for flag in flags:
        keep &= pair_flags[flag]
    return int(np.sum(keep))


def sorted_result_indices(specs, minbias_counts, total_counts, signal_counts, n_valid, args):
    rows = []
    for spec_idx, (min_pairs, max_protons, min_protons_400, flags, _label) in enumerate(specs):
        minbias_count = int(minbias_counts[spec_idx])
        rate_khz = minbias_count * F_COLL_AVG_HZ / n_valid / 1000
        sig_eff = signal_counts[spec_idx] / n_valid
        values = {
            "sig_eff": sig_eff,
            "signal": int(signal_counts[spec_idx]),
            "minbias": minbias_count,
            "rate": rate_khz,
            "total": int(total_counts[spec_idx]),
            "min_pairs": min_pairs,
            "max_protons": max_protons if max_protons is not None else 999,
            "min_protons_400": min_protons_400 if min_protons_400 is not None else 0,
            "n_flags": len(flags),
            "input": spec_idx,
        }
        rows.append((spec_idx, values))

    def sort_key(row):
        _spec_idx, values = row
        primary = values[args.sort_by]
        if args.sort_by in ("sig_eff", "signal", "minbias", "rate", "total"):
            primary = -primary if not args.sort_ascending else primary
        elif not args.sort_ascending:
            primary = -primary
        return (
            primary,
            values["min_pairs"],
            values["max_protons"],
            values["min_protons_400"],
            values["n_flags"],
            values["input"],
        )

    return [spec_idx for spec_idx, _values in sorted(rows, key=sort_key)]


def analyze_combo(combo_id, bx, signal_event, model, bx_protons, bx_proton_model_idx, constants, args, specs):
    candidates = build_candidates(combo_id, bx, signal_event, model, bx_protons, bx_proton_model_idx, args)
    if candidates is None:
        return None

    detected_vertices = build_detected_minbias_vertices(combo_id, bx, model, args)
    pair_flags = build_pairs(combo_id, candidates, detected_vertices, constants, args)
    source_left = pair_flags["source_L"]
    source_right = pair_flags["source_R"]
    minbias_only = (source_left == 1) & (source_right == 1)
    signal_pair = (source_left == 0) | (source_right == 0)
    all_pairs = np.ones(source_left.size, dtype=bool)

    count_cache = {}
    for flags in {flags for _min_pairs, _max_protons, _min_protons_400, flags, _label in specs}:
        count_cache[flags] = {
            "minbias": count_passing_pairs(pair_flags, minbias_only, flags),
            "total": count_passing_pairs(pair_flags, all_pairs, flags),
            "signal": count_passing_pairs(pair_flags, signal_pair, flags),
        }

    tagged = candidates["tag200"] | candidates["tag400"]
    total_tagged_protons = int(np.sum(tagged))
    minbias_tagged_protons = int(np.sum(tagged & (candidates["source"] == 1)))
    tag400 = candidates["tag400"]
    total_tag400_protons = int(np.sum(tag400))
    minbias_tag400_protons = int(np.sum(tag400 & (candidates["source"] == 1)))
    proton_count_cache = {}
    for max_protons in {max_protons for _min_pairs, max_protons, _min_protons_400, _flags, _label in specs}:
        proton_count_cache[max_protons] = {
            "minbias": max_protons is None or minbias_tagged_protons <= max_protons,
            "total": max_protons is None or total_tagged_protons <= max_protons,
        }
    tag400_count_cache = {}
    for min_protons_400 in {
        min_protons_400 for _min_pairs, _max_protons, min_protons_400, _flags, _label in specs
    }:
        tag400_count_cache[min_protons_400] = {
            "minbias": min_protons_400 is None or minbias_tag400_protons >= min_protons_400,
            "total": min_protons_400 is None or total_tag400_protons >= min_protons_400,
        }

    rows = []
    for spec_idx, (min_pairs, max_protons, min_protons_400, flags, _label) in enumerate(specs):
        counts = count_cache[flags]
        proton_count_pass = proton_count_cache[max_protons]
        tag400_count_pass = tag400_count_cache[min_protons_400]
        rows.append(
            (
                spec_idx,
                proton_count_pass["minbias"] and tag400_count_pass["minbias"] and counts["minbias"] >= min_pairs,
                proton_count_pass["total"] and tag400_count_pass["total"] and counts["total"] >= min_pairs,
                (
                    proton_count_pass["total"]
                    and tag400_count_pass["total"]
                    and counts["total"] >= min_pairs
                    and counts["signal"] > 0
                ),
            )
        )

    if args.verbose > 1:
        print(
            f"Combo {combo_id}: BX {bx}, accepted pairs={source_left.size}, "
            f"minbias-only={int(np.sum(minbias_only))}, signal-including={int(np.sum(signal_pair))}, "
            f"detected minbias vertices={detected_vertices['n_detected']}/{detected_vertices['n_all']}"
        )
    return rows


def validate_args(args):
    if args.max_bx is not None and args.max_bx <= 0:
        raise RuntimeError("--max-bx must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_combos is not None and args.max_combos <= 0:
        raise RuntimeError("--max-combos must be > 0")
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


def main():
    parser = argparse.ArgumentParser(description="Scan invisible-signal and minbias PPS-pair trigger selections")
    parser.add_argument("-i", "--input", required=True, help="Input minbias NPZ file or directory of NPZ files")
    parser.add_argument("-s", "--signal-in", required=True, help="Signal LHE file or directory of .lhe/.dat files")
    parser.add_argument("--max-bx", type=int, default=None, help="Optional maximum number of BX to process")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of minbias files to process")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional maximum number of signal+minbias combos to process")
    parser.add_argument("--trk-pt-min", type=float, default=2.0, help="Track proxy pT cut in GeV")
    parser.add_argument("--trk-eta-max", type=float, default=2.4, help="Track proxy |eta| cut")
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7, help="Gaussian beam spot sigma z in cm")
    parser.add_argument("--pps-time-res-ps", type=float, default=3.0, help="Single-arm timing resolution in ps")
    parser.add_argument("--pv-z-res-cm", type=float, default=0.1, help="Central vertex z resolution in cm")
    parser.add_argument("--pv-t-res-ps", type=float, default=None, help="Central vertex timing resolution in ps")
    parser.add_argument("--nsigma", type=float, default=2.0, help="Compatibility cut in sigma units")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for vertex and timing smearing")
    parser.add_argument("--mass-window", type=float, default=6.0, help="Mass half-width in GeV around 126")
    parser.add_argument("--xi-res", type=float, default=0.0003, help="Gaussian xi resolution sigma; 0 disables smearing")
    parser.add_argument("--s", type=float, default=S_GEV2, help="s value for M=sqrt(xi1*xi2*s)")
    parser.add_argument("--no-dz-smear", dest="smear_dz", action="store_false", help="Disable PPS timing smearing")
    parser.add_argument(
        "--sort-by",
        choices=(
            "sig_eff",
            "signal",
            "minbias",
            "rate",
            "total",
            "min_pairs",
            "max_protons",
            "min_protons_400",
            "n_flags",
            "input",
        ),
        default="sig_eff",
        help="Sort output table by this column or derived quantity",
    )
    parser.add_argument("--sort-ascending", action="store_true", help="Sort output table in ascending order")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Increase output verbosity")
    parser.set_defaults(smear_dz=True)
    args = parser.parse_args()
    validate_args(args)

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
    print(f"Processing {n_combos} invisible-signal+minbias combos")
    if n_combos == 0:
        raise RuntimeError("No signal+minbias combos available to process.")

    proton_model_idx = proton_interaction_indices(protons, model)
    unique_pbx, proton_groups = group_indices_by_bx(protons["bx_id"])
    proton_group_by_bx = {int(bx): group for bx, group in zip(unique_pbx, proton_groups)}

    pps_sigma_z_vertex = pps_sigma_z_cm(args.pps_time_res_ps)
    pps_sigma_t_vertex = pps_sigma_z_vertex / C_CM_PER_PS
    constants = {
        "zcut": args.nsigma * math.sqrt(pps_sigma_z_vertex**2 + args.pv_z_res_cm**2),
        "tcut": (
            args.nsigma * math.sqrt(pps_sigma_t_vertex**2 + args.pv_t_res_ps**2)
            if args.pv_t_res_ps is not None
            else args.nsigma * args.beam_sigma_z_cm / C_CM_PER_PS
        ),
        "bx_zcut": args.nsigma * args.beam_sigma_z_cm,
        "bx_tcut": args.nsigma * args.beam_sigma_z_cm / C_CM_PER_PS,
    }

    specs = selection_specs()
    minbias_counts = np.zeros(len(specs), dtype=np.int64)
    total_counts = np.zeros(len(specs), dtype=np.int64)
    signal_counts = np.zeros(len(specs), dtype=np.int64)
    skipped_signal = 0

    for combo_id, bx in enumerate(selected_bx):
        bx = int(bx)
        if combo_id % 100 == 0:
            print(f"Analyzing combo {combo_id} (BX {bx})...")
        idx_bx = proton_group_by_bx.get(bx)
        if idx_bx is None:
            idx_bx = np.empty(0, dtype=np.int64)
        bx_protons = {name: arr[idx_bx] for name, arr in protons.items()}
        combo_rows = analyze_combo(
            combo_id,
            bx,
            signal_events[combo_id],
            model,
            bx_protons,
            proton_model_idx[idx_bx],
            constants,
            args,
            specs,
        )
        if combo_rows is None:
            skipped_signal += 1
            continue
        for spec_idx, minbias_pass, total_pass, signal_pass in combo_rows:
            minbias_counts[spec_idx] += int(minbias_pass)
            total_counts[spec_idx] += int(total_pass)
            signal_counts[spec_idx] += int(signal_pass)

    n_valid = n_combos - skipped_signal
    if n_valid <= 0:
        raise RuntimeError("No valid signal events were processed.")

    print(f"Summary of invisible trigger selections across valid combos ({n_valid}):")
    if skipped_signal:
        print(f"Signal events skipped (malformed): {skipped_signal}")
    print(f"Sorted by: {args.sort_by} ({'ascending' if args.sort_ascending else 'descending'})")
    selection_width = max(
        90,
        max(len(label) for _min_pairs, _max_protons, _min_protons_400, _flags, label in specs),
    )
    print(
        f"{'selection':{selection_width}s}  {'minbias':>8s}  {'rate_kHz':>12s}  "
        f"{'total_with_signal':>17s}  {'signal':>8s}  {'sig_eff':>10s}"
    )
    print("-" * (selection_width + 68))
    for spec_idx in sorted_result_indices(specs, minbias_counts, total_counts, signal_counts, n_valid, args):
        _min_pairs, _max_protons, _min_protons_400, _flags, label = specs[spec_idx]
        minbias_count = int(minbias_counts[spec_idx])
        rate_khz = minbias_count * F_COLL_AVG_HZ / n_valid / 1000
        sig_eff = signal_counts[spec_idx] / n_valid
        print(
            f"{label:{selection_width}s}  {minbias_count:8d}  {rate_khz:12.2f}  "
            f"{int(total_counts[spec_idx]):17d}  {int(signal_counts[spec_idx]):8d}  {sig_eff:10.6f}"
        )


if __name__ == "__main__":
    main()
