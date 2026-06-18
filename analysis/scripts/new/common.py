#!/usr/bin/env python3
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


CENTRAL_SOURCE = "central"
MINBIAS_SOURCE = "minbias"

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
)


def repo_root():
    return Path(__file__).resolve().parents[3]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def resolve_path(path, base=None):
    if path is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return path
    if base is not None:
        candidate = Path(base) / path
        if candidate.exists():
            return candidate
    return repo_root() / path


def discover_npz_files(path, max_files=None):
    path = resolve_path(path)
    if path.is_dir():
        files = sorted(path.glob("*.npz"))
    else:
        files = [path]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No .npz files found at {path}")
    return files


def discover_lhe_files(path, max_files=None):
    path = resolve_path(path)
    if path.is_dir():
        patterns = ("*.lhe", "evrec*.dat", "*.dat")
        files = []
        for pattern in patterns:
            files.extend(path.rglob(pattern))
        files = sorted(set(files))
        files = [
            f
            for f in files
            if not f.name.endswith("_summary.dat") and not f.name.startswith("output")
        ]
    else:
        files = [path]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No .lhe/.dat files found at {path}")
    return files


def require_fields(data, fields, filename):
    missing = [name for name in fields if name not in data.files]
    if missing:
        raise RuntimeError(f"Required arrays missing in {filename}: {', '.join(missing)}")


def parse_lhe_events(filename):
    events = []
    xsec_pb = None
    in_event = False
    in_init = False
    expect_event_header = False
    init_line_count = 0
    current_header = None
    current_particles = []

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
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
                current_header = None
                current_particles = []
                continue
            if line.startswith("</event"):
                if current_header is not None:
                    events.append((current_header, current_particles))
                in_event = False
                expect_event_header = False
                continue
            if not in_event:
                continue

            parts = line.split()
            if expect_event_header:
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
                    "mass": float(parts[10]),
                }
            )

    return events, xsec_pb


def load_hard_events(path, max_files=None):
    files = discover_lhe_files(path, max_files=max_files)
    events = []
    xsec_pb = None
    for filename in files:
        file_events, file_xsec = parse_lhe_events(filename)
        if xsec_pb is None and file_xsec is not None:
            xsec_pb = file_xsec
        events.extend(file_events)
    if not events:
        raise RuntimeError(f"No events parsed from {path}")
    return files, events, xsec_pb


def load_minbias(path, max_files=None):
    files = discover_npz_files(path, max_files=max_files)
    proton_parts = {name: [] for name in PROTON_FIELDS}
    track_parts = {name: [] for name in TRACK_FIELDS}
    universe_parts = []
    metadata = {}

    for filename in files:
        with np.load(filename) as data:
            require_fields(data, PROTON_FIELDS, filename)
            for name in PROTON_FIELDS:
                proton_parts[name].append(np.asarray(data[name]))

            has_tracks = all(name in data.files for name in TRACK_FIELDS)
            if has_tracks:
                for name in TRACK_FIELDS:
                    track_parts[name].append(np.asarray(data[name]))

            universe_parts.append(interaction_universe_from_file(data, filename, has_tracks))
            for key in ("mu_mean", "mu_mode"):
                if key in data.files and key not in metadata:
                    metadata[key] = np.asarray(data[key]).item()

    protons = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in proton_parts.items()
    }
    tracks = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in track_parts.items()
    }
    universe = np.unique(np.concatenate(universe_parts, axis=0), axis=0)
    return files, protons, tracks, universe, metadata


def interaction_universe_from_file(data, filename, has_tracks):
    if all(name in data.files for name in ("mu_per_bx", "bx_offset", "n_bx")):
        mu_per_bx = np.asarray(data["mu_per_bx"], dtype=np.int64)
        bx_offset = int(np.asarray(data["bx_offset"]).item())
        n_bx = int(np.asarray(data["n_bx"]).item())
        if mu_per_bx.size != n_bx:
            raise RuntimeError(
                f"mu_per_bx length ({mu_per_bx.size}) does not match n_bx ({n_bx}) in {filename}"
            )
        bx_parts = []
        interaction_parts = []
        for local_bx, n_interactions in enumerate(mu_per_bx):
            if n_interactions <= 0:
                continue
            bx = bx_offset + local_bx
            bx_parts.append(np.full(int(n_interactions), bx, dtype=np.int64))
            interaction_parts.append(np.arange(int(n_interactions), dtype=np.int64))
        if bx_parts:
            return np.column_stack((np.concatenate(bx_parts), np.concatenate(interaction_parts)))

    keys = [
        np.column_stack(
            (
                np.asarray(data["bx_id"], dtype=np.int64),
                np.asarray(data["interaction_id"], dtype=np.int64),
            )
        )
    ]
    if has_tracks:
        keys.append(
            np.column_stack(
                (
                    np.asarray(data["trk_bx_id"], dtype=np.int64),
                    np.asarray(data["trk_interaction_id"], dtype=np.int64),
                )
            )
        )
    all_keys = np.concatenate(keys, axis=0)
    if all_keys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(all_keys, axis=0)


def build_interaction_summaries(universe, tracks, trk_pt_min, trk_eta_max):
    summaries = {}
    for bx in np.unique(universe[:, 0]):
        idx = np.where(universe[:, 0] == bx)[0]
        summaries[int(bx)] = [
            {"interaction_id": int(interaction), "n_tracks": 0, "sum_pt2": 0.0}
            for interaction in universe[idx, 1]
        ]

    if tracks["trk_pt"].size == 0:
        return summaries

    mask = (tracks["trk_pt"] > trk_pt_min) & (np.abs(tracks["trk_eta"]) < trk_eta_max)
    grouped = {}
    for bx, interaction, pt in zip(
        tracks["trk_bx_id"][mask], tracks["trk_interaction_id"][mask], tracks["trk_pt"][mask]
    ):
        key = (int(bx), int(interaction))
        count, sum_pt2 = grouped.get(key, (0, 0.0))
        grouped[key] = (count + 1, sum_pt2 + float(pt * pt))

    for bx, rows in summaries.items():
        for row in rows:
            count, sum_pt2 = grouped.get((bx, row["interaction_id"]), (0, 0.0))
            row["n_tracks"] = int(count)
            row["sum_pt2"] = float(sum_pt2)
    return summaries


def infer_jet_pdg_ids(sample_name):
    name = sample_name.lower()
    if name.endswith("_bb") or "bb" in name:
        return [5, -5]
    if name.endswith("_cc") or "cc" in name:
        return [4, -4]
    if name.endswith("_gg") or "gg" in name:
        return [21, 21]
    if name.endswith("_qq") or "qq" in name:
        return [1, -1, 2, -2, 3, -3]
    return [5, -5]


def select_hard_jets(sample_name, sample_cfg, particles):
    wanted = sample_cfg.get("jet_pdg_ids")
    if wanted is None:
        wanted = infer_jet_pdg_ids(sample_name)
    wanted = [int(pid) for pid in wanted]
    final = [p for p in particles if p["status"] == 1]
    if len(wanted) == 2 and wanted[0] != wanted[1]:
        jets = []
        for pid in wanted:
            matching = [p for p in final if p["pid"] == pid]
            if len(matching) != 1:
                return None
            jets.append(matching[0])
        return jets

    matching = [p for p in final if p["pid"] in wanted]
    if len(matching) != 2:
        return None
    return matching


def outgoing_protons(particles, sqrt_s_gev):
    incoming = [p for p in particles if p["pid"] == 2212 and p["status"] == -1]
    protons = [p for p in particles if p["pid"] == 2212 and p["status"] == 1]
    if len(protons) != 2:
        return None
    beam_e = incoming[0]["E"] if incoming else sqrt_s_gev / 2.0
    out = []
    for proton_idx, proton in enumerate(protons):
        side = 1 if proton["pz"] >= 0.0 else -1
        xi = (beam_e - proton["E"]) / beam_e if beam_e > 0.0 else 0.0
        out.append(
            {
                "input_proton_idx": int(proton_idx),
                "source": CENTRAL_SOURCE,
                "interaction_id": -1,
                "side": side,
                "px": float(proton["px"]),
                "py": float(proton["py"]),
                "pz": float(proton["pz"]),
                "E": float(proton["E"]),
                "pt": pt(proton["px"], proton["py"]),
                "xi_truth": float(xi),
            }
        )
    return out


def particle_record(particle):
    record = {
        "pid": int(particle["pid"]),
        "px": float(particle["px"]),
        "py": float(particle["py"]),
        "pz": float(particle["pz"]),
        "E": float(particle["E"]),
    }
    record["pt"] = pt(record["px"], record["py"])
    record["eta"] = eta(record["px"], record["py"], record["pz"])
    record["phi"] = phi(record["px"], record["py"])
    record["mass"] = mass(record["px"], record["py"], record["pz"], record["E"])
    return record


def smear_jet(jet, smearing, rng):
    pt_reco = smear_positive(jet["pt"], smearing.get("pt_rel", 0.0) * jet["pt"], rng)
    eta_reco = smear_value(jet["eta"], smearing.get("eta", 0.0), rng)
    phi_reco = smear_value(jet["phi"], smearing.get("phi", 0.0), rng)
    mass_reco = smear_positive(jet["mass"], smearing.get("mass_rel", 0.0) * jet["mass"], rng)
    px = pt_reco * math.cos(phi_reco)
    py = pt_reco * math.sin(phi_reco)
    pz = pt_reco * math.sinh(eta_reco)
    energy = math.sqrt(max(mass_reco * mass_reco + px * px + py * py + pz * pz, 0.0))
    return {
        "pid": jet["pid"],
        "px": px,
        "py": py,
        "pz": pz,
        "E": energy,
        "pt": pt_reco,
        "eta": eta_reco,
        "phi": phi_reco,
        "mass": mass_reco,
    }


def dijet_record(j1, j2):
    px = j1["px"] + j2["px"]
    py = j1["py"] + j2["py"]
    pz = j1["pz"] + j2["pz"]
    energy = j1["E"] + j2["E"]
    return {
        "px": px,
        "py": py,
        "pz": pz,
        "E": energy,
        "pt": pt(px, py),
        "y": rapidity(energy, pz),
        "mass": mass(px, py, pz, energy),
    }


def build_central_record(hard_event_id, header, jets, smearing, rng):
    j1_truth = particle_record(jets[0])
    j2_truth = particle_record(jets[1])
    j1_reco = smear_jet(j1_truth, smearing, rng)
    j2_reco = smear_jet(j2_truth, smearing, rng)
    return {
        "hard_event_id": int(hard_event_id),
        "xwgtup": float(header["XWGTUP"]),
        "j1_truth": j1_truth,
        "j2_truth": j2_truth,
        "j1_reco": j1_reco,
        "j2_reco": j2_reco,
        "jj_truth": dijet_record(j1_truth, j2_truth),
        "jj_reco": dijet_record(j1_reco, j2_reco),
    }


def station_tag_counts(xi, xi_ranges):
    passing = 0
    n_400 = 0
    for station, bounds in xi_ranges.items():
        xi_min, xi_max = float(bounds[0]), float(bounds[1])
        tagged = xi >= xi_min and xi < xi_max
        if tagged:
            passing += 1
            if str(station) == "420":
                n_400 += 1
    return passing > 0, n_400


def smear_xi(xi, xi_res, rng):
    if xi_res <= 0.0:
        return xi
    return max(0.0, float(xi + rng.normal(0.0, xi_res)))


def build_pairs(protons, sqrt_s_gev):
    pairs = []
    passing = [p for p in protons if p["passes_pps"]]
    left = [p for p in passing if p["side"] == -1]
    right = [p for p in passing if p["side"] == 1]
    for left_p in left:
        for right_p in right:
            n_signal = int(left_p["source"] == CENTRAL_SOURCE) + int(right_p["source"] == CENTRAL_SOURCE)
            pairs.append(
                {
                    "pair_id": len(pairs),
                    "left_proton_id": int(left_p["proton_id"]),
                    "right_proton_id": int(right_p["proton_id"]),
                    "n_signal_protons": int(n_signal),
                    "n_400_tags": int(left_p["n_400_tags"] + right_p["n_400_tags"]),
                    "mass_truth": pair_mass(left_p["xi_truth"], right_p["xi_truth"], sqrt_s_gev),
                    "y_truth": pair_rapidity(left_p["xi_truth"], right_p["xi_truth"]),
                    "mass_reco": pair_mass(left_p["xi_reco"], right_p["xi_reco"], sqrt_s_gev),
                    "y_reco": pair_rapidity(left_p["xi_reco"], right_p["xi_reco"]),
                }
            )
    return pairs


def selected_bx_values(universe, max_bx=None):
    values = np.unique(universe[:, 0])
    if max_bx is not None:
        values = values[:max_bx]
    return [int(v) for v in values]


def pt(px, py):
    return math.hypot(px, py)


def phi(px, py):
    return math.atan2(py, px)


def eta(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return math.copysign(float("inf"), pz)
    arg = (p + pz) / (p - pz)
    return 0.5 * math.log(arg) if arg > 0.0 else float("nan")


def mass(px, py, pz, energy):
    m2 = energy * energy - px * px - py * py - pz * pz
    return math.sqrt(m2) if m2 > 0.0 else 0.0


def rapidity(energy, pz):
    if energy <= abs(pz):
        return float("nan")
    arg = (energy + pz) / (energy - pz)
    return 0.5 * math.log(arg) if arg > 0.0 else float("nan")


def pair_mass(xi_left, xi_right, sqrt_s_gev):
    product = xi_left * xi_right
    return math.sqrt(product) * sqrt_s_gev if product > 0.0 else float("nan")


def pair_rapidity(xi_left, xi_right):
    if xi_left <= 0.0 or xi_right <= 0.0:
        return float("nan")
    return 0.5 * math.log(xi_right / xi_left)


def smear_value(value, sigma, rng):
    if sigma <= 0.0:
        return float(value)
    return float(value + rng.normal(0.0, sigma))


def smear_positive(value, sigma, rng):
    if sigma <= 0.0:
        return float(value)
    return max(0.0, float(value + rng.normal(0.0, sigma)))


def counter_summary(counters):
    return ", ".join(f"{key}={value}" for key, value in sorted(Counter(counters).items()))
