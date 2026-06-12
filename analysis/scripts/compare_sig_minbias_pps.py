#!/usr/bin/env python3
import argparse
import glob
import math
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


S_GEV2 = 14000.0**2
M_HIGGS_GEV = 126.0
PROTON_MASS_GEV = 0.9382720813

STATION_XI = {
    "192": (0.08, 0.1967),
    "213": (0.0375, 0.0688),
    "220": (0.014, 0.0263),
    "420": (0.00325, 0.0116),
}
STATIONS_200 = ("192", "213", "220")
STATIONS = (*STATIONS_200, "420")

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

INT_BRANCHES = (
    "event",
    "bx",
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
    "pair_tag400",
    "pair_double_tag400",
    "require_two_400_tags",
)

PROTON_FLOAT_BASES = (
    "xi",
    "px",
    "py",
    "pz",
    "E",
    "m",
    "pt",
    "t",
    "theta_x",
    "theta_y",
    "theta",
    "phi",
)

PAIR_FLOAT_BRANCHES = (
    "MX",
    "yX",
    "pX_px",
    "pX_py",
    "pX_pz",
    "pX_E",
    "pX_pt",
    "delta_phi",
    "abs_delta_phi",
    "delta_t",
    "abs_delta_t",
    "delta_theta_x",
    "abs_delta_theta_x",
    "delta_theta_y",
    "abs_delta_theta_y",
    "delta_theta",
    "abs_delta_theta",
    "xi_asym",
    "pt_asym",
    "t_asym",
    "pp_pt_sumvec",
    "xi_product",
    "theta_x_sum",
    "theta_y_sum",
    "mass_window",
    "s",
)

FLOAT_BRANCHES = tuple(
    f"{name}_{side}"
    for side in ("L", "R")
    for name in PROTON_FLOAT_BASES
) + PAIR_FLOAT_BRANCHES

def plot_config(label, plot_range=None, bins=None):
    return {
        "label": label,
        "range": plot_range,
        "bins": bins,
    }


PLOT_VARIABLES = {
    "xi_L": plot_config(r"$\xi_L$"),
    "xi_R": plot_config(r"$\xi_R$"),
    "px_L": plot_config(r"$p_{x,L}$ [GeV]"),
    "px_R": plot_config(r"$p_{x,R}$ [GeV]"),
    "py_L": plot_config(r"$p_{y,L}$ [GeV]"),
    "py_R": plot_config(r"$p_{y,R}$ [GeV]"),
    "pz_L": plot_config(r"$p_{z,L}$ [GeV]"),
    "pz_R": plot_config(r"$p_{z,R}$ [GeV]"),
    "pt_L": plot_config(r"$p_{T,L}$ [GeV]"),
    "pt_R": plot_config(r"$p_{T,R}$ [GeV]"),
    "t_L": plot_config(r"$t_L$ [GeV$^2$]"),
    "t_R": plot_config(r"$t_R$ [GeV$^2$]"),
    "theta_x_L": plot_config(r"$\theta_{x,L}$"),
    "theta_x_R": plot_config(r"$\theta_{x,R}$"),
    "theta_y_L": plot_config(r"$\theta_{y,L}$"),
    "theta_y_R": plot_config(r"$\theta_{y,R}$"),
    "theta_L": plot_config(r"$\theta_L$"),
    "theta_R": plot_config(r"$\theta_R$"),
    "phi_L": plot_config(r"$\phi_L$"),
    "phi_R": plot_config(r"$\phi_R$"),
    "MX": plot_config(r"$M_X$ [GeV]"),
    "yX": plot_config(r"$y_X$"),
    "pX_px": plot_config(r"$p_{X,x}$ [GeV]"),
    "pX_py": plot_config(r"$p_{X,y}$ [GeV]"),
    "pX_pz": plot_config(r"$p_{X,z}$ [GeV]"),
    "pX_E": plot_config(r"$E_X$ [GeV]"),
    "pX_pt": plot_config(r"$p_{T,X}$ [GeV]"),
    "delta_phi": plot_config(r"$\Delta\phi$"),
    "abs_delta_phi": plot_config(r"$|\Delta\phi|$"),
    "delta_t": plot_config(r"$\Delta t$ [GeV$^2$]"),
    "abs_delta_t": plot_config(r"$|\Delta t|$ [GeV$^2$]", plot_range=(0,2),bins=40),
    "delta_theta_x": plot_config(r"$\Delta\theta_x$"),
    "abs_delta_theta_x": plot_config(r"$|\Delta\theta_x|$"),
    "delta_theta_y": plot_config(r"$\Delta\theta_y$"),
    "abs_delta_theta_y": plot_config(r"$|\Delta\theta_y|$"),
    "delta_theta": plot_config(r"$\Delta\theta$"),
    "abs_delta_theta": plot_config(r"$|\Delta\theta|$"),
    "xi_asym": plot_config(r"$(\xi_L-\xi_R)/(\xi_L+\xi_R)$"),
    "pt_asym": plot_config(r"$(p_{T,L}-p_{T,R})/(p_{T,L}+p_{T,R})$"),
    "t_asym": plot_config(r"$(t_L-t_R)/(t_L+t_R)$"),
    "pp_pt_sumvec": plot_config(r"$|\vec p_{T,L}+\vec p_{T,R}|$ [GeV]"),
    "xi_product": plot_config(r"$\xi_L\xi_R$"),
    "theta_x_sum": plot_config(r"$\theta_{x,L}+\theta_{x,R}$"),
    "theta_y_sum": plot_config(r"$\theta_{y,L}+\theta_{y,R}$"),
}


def input_files(path, pattern):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, pattern)))
        if not files:
            raise RuntimeError(f"No {pattern} files found in directory {path}")
        return files
    return [path]


def signal_input_files(path):
    if os.path.isdir(path):
        files = []
        for pattern in ("*.lhe", "*.dat"):
            files.extend(sorted(glob.glob(os.path.join(path, pattern))))
        files = [
            filename for filename in files
            if not os.path.basename(filename).endswith("_summary.dat")
        ]
        if not files:
            raise RuntimeError(f"No .lhe/.dat files found in directory {path}")
        return files
    return [path]


def iter_lhe_records(filename):
    in_event = False
    expect_event_header = False
    current_particles = []
    current_header = None
    in_init = False
    init_line_count = 0
    xsec_reported = False

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
                if init_line_count == 2 and not xsec_reported:
                    parts = line.split()
                    if parts:
                        try:
                            yield "xsec", float(parts[0])
                            xsec_reported = True
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
                    yield "event", (current_header, current_particles)
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


def require_fields(data, fields, filename):
    missing = [name for name in fields if name not in data.files]
    if missing:
        raise RuntimeError(f"Required arrays missing in {filename}: {', '.join(missing)}")


def station_tags(xi):
    tag_by_station = {
        station: (xi >= xi_min) & (xi < xi_max)
        for station, (xi_min, xi_max) in STATION_XI.items()
    }
    tag200 = tag_by_station["192"] | tag_by_station["213"] | tag_by_station["220"]
    tag400 = tag_by_station["420"]
    return tag_by_station, tag200, tag400


def cartesian_pair_indices(left_idx, right_idx):
    if left_idx.size == 0 or right_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.repeat(left_idx, right_idx.size), np.tile(right_idx, left_idx.size)


def empty_pair_arrays():
    arrays = {}
    for name in INT_BRANCHES:
        arrays[name] = np.empty(0, dtype=np.int32)
    for name in FLOAT_BRANCHES:
        arrays[name] = np.empty(0, dtype=np.float64)
    return arrays


def concatenate_pair_chunks(chunks):
    if not chunks:
        return empty_pair_arrays()
    arrays = {}
    for name in INT_BRANCHES:
        arrays[name] = np.concatenate([chunk[name] for chunk in chunks]).astype(np.int32, copy=False)
    for name in FLOAT_BRANCHES:
        arrays[name] = np.concatenate([chunk[name] for chunk in chunks]).astype(np.float64, copy=False)
    return arrays


def infer_incoming_proton(side, s, mass=PROTON_MASS_GEV):
    beam_E = math.sqrt(s) / 2.0
    pz_abs = math.sqrt(max(beam_E * beam_E - mass * mass, 0.0))
    return {
        "px": 0.0,
        "py": 0.0,
        "pz": float(side) * pz_abs,
        "E": beam_E,
        "m": mass,
    }


def incoming_arrays_from_side(side, s, mass):
    side = np.asarray(side, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    beam_E = math.sqrt(s) / 2.0
    pz_abs = np.sqrt(np.maximum(beam_E * beam_E - mass * mass, 0.0))
    return {
        "px_in": np.zeros(side.size, dtype=np.float64),
        "py_in": np.zeros(side.size, dtype=np.float64),
        "pz_in": side * pz_abs,
        "E_in": np.full(side.size, beam_E, dtype=np.float64),
        "m_in": mass,
    }


def select_pair_indices(candidates, args):
    tagged = candidates["tag200"] | candidates["tag400"]
    left_idx = np.where((candidates["side"] == -1) & tagged)[0]
    right_idx = np.where((candidates["side"] == +1) & tagged)[0]
    iL, iR = cartesian_pair_indices(left_idx, right_idx)
    if iL.size == 0:
        return iL, iR, np.empty(0, dtype=np.float64)

    if args.require_two_400_tags:
        keep = candidates["tag400"][iL] & candidates["tag400"][iR]
    else:
        keep = candidates["tag400"][iL] | candidates["tag400"][iR]

    xi_prod = candidates["xi"][iL] * candidates["xi"][iR]
    keep &= xi_prod > 0.0
    iL = iL[keep]
    iR = iR[keep]
    xi_prod = xi_prod[keep]
    if iL.size == 0:
        return iL, iR, np.empty(0, dtype=np.float64)

    mass = np.sqrt(xi_prod * args.s)
    if args.mass_window is not None:
        keep = np.abs(mass - M_HIGGS_GEV) <= args.mass_window
        iL = iL[keep]
        iR = iR[keep]
        mass = mass[keep]

    return iL, iR, mass


def vector_t(candidates, indices):
    dE = candidates["E"][indices] - candidates["E_in"][indices]
    dpx = candidates["px"][indices] - candidates["px_in"][indices]
    dpy = candidates["py"][indices] - candidates["py_in"][indices]
    dpz = candidates["pz"][indices] - candidates["pz_in"][indices]
    return dE * dE - dpx * dpx - dpy * dpy - dpz * dpz


def vector_theta(px, py, pz):
    theta_x = np.full(px.size, np.nan, dtype=np.float64)
    theta_y = np.full(px.size, np.nan, dtype=np.float64)
    valid = pz != 0.0
    theta_x[valid] = px[valid] / pz[valid]
    theta_y[valid] = py[valid] / pz[valid]
    theta = np.sqrt(theta_x * theta_x + theta_y * theta_y)
    return theta_x, theta_y, theta


def safe_asym(numerator, denominator):
    out = np.full(numerator.size, np.nan, dtype=np.float64)
    valid = denominator != 0.0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def build_pair_chunk(candidates, iL, iR, mass, event, bx, args):
    n_pairs = iL.size
    if n_pairs == 0:
        return None

    chunk = {}
    chunk["event"] = np.full(n_pairs, event, dtype=np.int32)
    chunk["bx"] = np.full(n_pairs, bx, dtype=np.int32)
    chunk["interaction_L"] = candidates["interaction"][iL].astype(np.int32, copy=False)
    chunk["interaction_R"] = candidates["interaction"][iR].astype(np.int32, copy=False)
    chunk["proton_idx_L"] = candidates["proton_idx"][iL].astype(np.int32, copy=False)
    chunk["proton_idx_R"] = candidates["proton_idx"][iR].astype(np.int32, copy=False)
    chunk["side_L"] = candidates["side"][iL].astype(np.int32, copy=False)
    chunk["side_R"] = candidates["side"][iR].astype(np.int32, copy=False)
    chunk["tag200_L"] = candidates["tag200"][iL].astype(np.int32, copy=False)
    chunk["tag200_R"] = candidates["tag200"][iR].astype(np.int32, copy=False)
    chunk["tag400_L"] = candidates["tag400"][iL].astype(np.int32, copy=False)
    chunk["tag400_R"] = candidates["tag400"][iR].astype(np.int32, copy=False)
    for station in STATIONS_200:
        chunk[f"tag{station}_L"] = candidates[f"tag{station}"][iL].astype(np.int32, copy=False)
        chunk[f"tag{station}_R"] = candidates[f"tag{station}"][iR].astype(np.int32, copy=False)
    chunk["pair_tag200"] = (candidates["tag200"][iL] | candidates["tag200"][iR]).astype(np.int32, copy=False)
    chunk["pair_tag400"] = (candidates["tag400"][iL] | candidates["tag400"][iR]).astype(np.int32, copy=False)
    chunk["pair_double_tag400"] = (candidates["tag400"][iL] & candidates["tag400"][iR]).astype(np.int32, copy=False)
    chunk["require_two_400_tags"] = np.full(n_pairs, int(args.require_two_400_tags), dtype=np.int32)

    pt_L = np.hypot(candidates["px"][iL], candidates["py"][iL])
    pt_R = np.hypot(candidates["px"][iR], candidates["py"][iR])
    t_L = vector_t(candidates, iL)
    t_R = vector_t(candidates, iR)
    theta_x_L, theta_y_L, theta_L = vector_theta(candidates["px"][iL], candidates["py"][iL], candidates["pz"][iL])
    theta_x_R, theta_y_R, theta_R = vector_theta(candidates["px"][iR], candidates["py"][iR], candidates["pz"][iR])
    phi_L = np.arctan2(candidates["py"][iL], candidates["px"][iL])
    phi_R = np.arctan2(candidates["py"][iR], candidates["px"][iR])

    proton_values = {
        "xi_L": candidates["xi"][iL],
        "xi_R": candidates["xi"][iR],
        "px_L": candidates["px"][iL],
        "px_R": candidates["px"][iR],
        "py_L": candidates["py"][iL],
        "py_R": candidates["py"][iR],
        "pz_L": candidates["pz"][iL],
        "pz_R": candidates["pz"][iR],
        "E_L": candidates["E"][iL],
        "E_R": candidates["E"][iR],
        "m_L": candidates["m"][iL],
        "m_R": candidates["m"][iR],
        "pt_L": pt_L,
        "pt_R": pt_R,
        "t_L": t_L,
        "t_R": t_R,
        "theta_x_L": theta_x_L,
        "theta_x_R": theta_x_R,
        "theta_y_L": theta_y_L,
        "theta_y_R": theta_y_R,
        "theta_L": theta_L,
        "theta_R": theta_R,
        "phi_L": phi_L,
        "phi_R": phi_R,
    }
    for name, values in proton_values.items():
        chunk[name] = values.astype(np.float64, copy=False)

    pX_px = candidates["px_in"][iL] + candidates["px_in"][iR] - candidates["px"][iL] - candidates["px"][iR]
    pX_py = candidates["py_in"][iL] + candidates["py_in"][iR] - candidates["py"][iL] - candidates["py"][iR]
    pX_pz = candidates["pz_in"][iL] + candidates["pz_in"][iR] - candidates["pz"][iL] - candidates["pz"][iR]
    pX_E = candidates["E_in"][iL] + candidates["E_in"][iR] - candidates["E"][iL] - candidates["E"][iR]
    delta_phi = np.arctan2(np.sin(phi_L - phi_R), np.cos(phi_L - phi_R))
    delta_t = t_L - t_R
    delta_theta_x = theta_x_L - theta_x_R
    delta_theta_y = theta_y_L - theta_y_R
    delta_theta = theta_L - theta_R
    xi_L = candidates["xi"][iL]
    xi_R = candidates["xi"][iR]
    xi_asym = safe_asym(xi_L - xi_R, xi_L + xi_R)
    pt_asym = safe_asym(pt_L - pt_R, pt_L + pt_R)
    t_asym = safe_asym(t_L - t_R, t_L + t_R)
    pp_pt_sumvec = np.hypot(candidates["px"][iL] + candidates["px"][iR], candidates["py"][iL] + candidates["py"][iR])
    chunk["MX"] = mass.astype(np.float64, copy=False)
    chunk["yX"] = (0.5 * np.log(xi_L / xi_R)).astype(np.float64, copy=False)
    chunk["pX_px"] = pX_px.astype(np.float64, copy=False)
    chunk["pX_py"] = pX_py.astype(np.float64, copy=False)
    chunk["pX_pz"] = pX_pz.astype(np.float64, copy=False)
    chunk["pX_E"] = pX_E.astype(np.float64, copy=False)
    chunk["pX_pt"] = np.hypot(pX_px, pX_py).astype(np.float64, copy=False)
    chunk["delta_phi"] = delta_phi.astype(np.float64, copy=False)
    chunk["abs_delta_phi"] = np.abs(delta_phi).astype(np.float64, copy=False)
    chunk["delta_t"] = delta_t.astype(np.float64, copy=False)
    chunk["abs_delta_t"] = np.abs(delta_t).astype(np.float64, copy=False)
    chunk["delta_theta_x"] = delta_theta_x.astype(np.float64, copy=False)
    chunk["abs_delta_theta_x"] = np.abs(delta_theta_x).astype(np.float64, copy=False)
    chunk["delta_theta_y"] = delta_theta_y.astype(np.float64, copy=False)
    chunk["abs_delta_theta_y"] = np.abs(delta_theta_y).astype(np.float64, copy=False)
    chunk["delta_theta"] = delta_theta.astype(np.float64, copy=False)
    chunk["abs_delta_theta"] = np.abs(delta_theta).astype(np.float64, copy=False)
    chunk["xi_asym"] = xi_asym
    chunk["pt_asym"] = pt_asym
    chunk["t_asym"] = t_asym
    chunk["pp_pt_sumvec"] = pp_pt_sumvec.astype(np.float64, copy=False)
    chunk["xi_product"] = (xi_L * xi_R).astype(np.float64, copy=False)
    chunk["theta_x_sum"] = (theta_x_L + theta_x_R).astype(np.float64, copy=False)
    chunk["theta_y_sum"] = (theta_y_L + theta_y_R).astype(np.float64, copy=False)
    chunk["mass_window"] = np.full(n_pairs, math.nan if args.mass_window is None else args.mass_window, dtype=np.float64)
    chunk["s"] = np.full(n_pairs, args.s, dtype=np.float64)
    return chunk


def should_report_incremental(processed, interval):
    return processed == 1 or (interval > 0 and processed % interval == 0)


def signal_event_candidates(signal_event, event_idx, args, rng):
    _header, particles = signal_event
    outgoing = [p for p in particles if p["pid"] == 2212 and p["status"] == 1]
    if len(outgoing) != 2:
        return None

    incoming = [p for p in particles if p["pid"] == 2212 and p["status"] == -1]
    incoming_by_side = {}
    for proton in incoming:
        side = +1 if proton["pz"] >= 0.0 else -1
        incoming_by_side[side] = proton
    beam_E = incoming[0]["E"] if incoming else math.sqrt(args.s) / 2.0

    side = np.asarray([+1 if proton["pz"] >= 0.0 else -1 for proton in outgoing], dtype=np.int32)
    px = np.asarray([proton["px"] for proton in outgoing], dtype=np.float64)
    py = np.asarray([proton["py"] for proton in outgoing], dtype=np.float64)
    pz = np.asarray([proton["pz"] for proton in outgoing], dtype=np.float64)
    E = np.asarray([proton["E"] for proton in outgoing], dtype=np.float64)
    m = np.asarray([proton["m"] for proton in outgoing], dtype=np.float64)
    xi_nominal = ((beam_E - E) / beam_E) if beam_E > 0.0 else np.zeros(E.size, dtype=np.float64)
    xi_nominal *= math.sqrt(args.s) / 14000.0
    xi = xi_nominal.copy()
    if args.xi_res > 0.0:
        xi += rng.normal(loc=0.0, scale=args.xi_res, size=xi.size)

    tags, tag200, tag400 = station_tags(xi_nominal)
    candidates = {
        "side": side,
        "xi": xi,
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
        "m": m,
        "interaction": np.full(side.size, event_idx, dtype=np.int32),
        "proton_idx": np.arange(side.size, dtype=np.int32),
        "tag200": tag200,
        "tag400": tag400,
    }
    for station in STATIONS_200:
        candidates[f"tag{station}"] = tags[station]

    incoming_px = np.empty(side.size, dtype=np.float64)
    incoming_py = np.empty(side.size, dtype=np.float64)
    incoming_pz = np.empty(side.size, dtype=np.float64)
    incoming_E = np.empty(side.size, dtype=np.float64)
    incoming_m = np.empty(side.size, dtype=np.float64)
    for proton_idx, proton in enumerate(outgoing):
        incoming = incoming_by_side.get(int(side[proton_idx]), infer_incoming_proton(side[proton_idx], args.s, proton["m"]))
        incoming_px[proton_idx] = incoming["px"]
        incoming_py[proton_idx] = incoming["py"]
        incoming_pz[proton_idx] = incoming["pz"]
        incoming_E[proton_idx] = incoming["E"]
        incoming_m[proton_idx] = incoming["m"]
    candidates["px_in"] = incoming_px
    candidates["py_in"] = incoming_py
    candidates["pz_in"] = incoming_pz
    candidates["E_in"] = incoming_E
    candidates["m_in"] = incoming_m
    return candidates


def signal_pair_chunk_from_event(signal_event, event_idx, args, rng):
    candidates = signal_event_candidates(signal_event, event_idx, args, rng)
    if candidates is None:
        return None, True
    iL, iR, mass = select_pair_indices(candidates, args)
    return build_pair_chunk(candidates, iL, iR, mass, event_idx, event_idx, args), False


def process_signal_pairs(path, args):
    files = signal_input_files(path)
    if args.signal_max_files is not None:
        files = files[:args.signal_max_files]

    print(f"Found {len(files)} signal input files")
    chunks = []
    rng = np.random.default_rng(args.seed)
    xsec_from_file = None
    n_events = 0
    malformed = 0
    n_pairs = 0
    n_pairs_before = 0
    stop = False
    for file_idx, filename in enumerate(files, start=1):
        print(f"Reading signal file {file_idx}/{len(files)}: {filename}", flush=True)
        n_file_events = 0
        for record_type, value in iter_lhe_records(filename):
            if record_type == "xsec":
                if xsec_from_file is None:
                    xsec_from_file = value
                continue
            event = value
            n_file_events += 1
            if args.signal_max_events is not None and n_events >= args.signal_max_events:
                stop = True
                break
            chunk, is_malformed = signal_pair_chunk_from_event(event, n_events, args, rng)
            malformed += int(is_malformed)
            if chunk is not None:
                chunks.append(chunk)
                n_pairs += len(chunk["MX"])
            n_events += 1
            if should_report_incremental(n_events, args.progress_interval):
                print(
                    f"  Signal events processed: {n_events}; "
                    f"accepted pairs: {n_pairs} (+{n_pairs - n_pairs_before} since last report)",
                    flush=True,
                )
                n_pairs_before = n_pairs
        print(f"  Parsed {n_file_events} events from file; kept {n_events} total signal events so far", flush=True)
        if stop:
            print(f"Reached --signal-max-events={args.signal_max_events}; stopping signal loading", flush=True)
            break

    if n_events == 0:
        raise RuntimeError("No signal events parsed from input files.")
    return files, concatenate_pair_chunks(chunks), malformed, xsec_from_file, n_events


def candidate_arrays_from_minbias(protons, indices, args, rng):
    xi_nominal = np.asarray(protons["xi"][indices], dtype=np.float64)
    xi = xi_nominal.copy()
    if args.xi_res > 0.0 and xi.size:
        xi += rng.normal(loc=0.0, scale=args.xi_res, size=xi.size)
    tags, tag200, tag400 = station_tags(xi_nominal)

    side = np.asarray(protons["side"][indices], dtype=np.int32)
    mass = np.asarray(protons["m"][indices], dtype=np.float64)
    incoming = incoming_arrays_from_side(side, args.s, mass)
    candidates = {
        "side": side,
        "xi": xi,
        "px": np.asarray(protons["px"][indices], dtype=np.float64),
        "py": np.asarray(protons["py"][indices], dtype=np.float64),
        "pz": np.asarray(protons["pz"][indices], dtype=np.float64),
        "E": np.asarray(protons["E"][indices], dtype=np.float64),
        "m": mass,
        "interaction": np.asarray(protons["interaction_id"][indices], dtype=np.int32),
        "proton_idx": np.asarray(protons["proton_idx"][indices], dtype=np.int32),
        "tag200": tag200,
        "tag400": tag400,
        **incoming,
    }
    for station in STATIONS_200:
        candidates[f"tag{station}"] = tags[station]
    return candidates


def grouped_indices(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    unique, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    return unique, [order[start:start + count] for start, count in zip(starts, counts)]


def minbias_pair_chunk_for_bx(protons, indices, event_idx, bx, args, rng):
    candidates = candidate_arrays_from_minbias(protons, indices, args, rng)
    iL, iR, mass = select_pair_indices(candidates, args)
    return build_pair_chunk(candidates, iL, iR, mass, event_idx, int(bx), args)


def process_minbias_pairs(path, args):
    files = input_files(path, "*.npz")
    if args.minbias_max_files is not None:
        files = files[:args.minbias_max_files]

    print(f"Found {len(files)} minbias input files")
    chunks = []
    rng = np.random.default_rng(args.seed + 1000003)
    n_protons_loaded = 0
    n_protons_kept = 0
    n_bx = 0
    n_pairs = 0
    n_pairs_before = 0
    for file_idx, filename in enumerate(files, start=1):
        print(f"Reading minbias file {file_idx}/{len(files)}: {filename}", flush=True)
        with np.load(filename) as data:
            require_fields(data, PROTON_FIELDS, filename)
            n_file_protons = len(data["xi"])
            n_protons_loaded += n_file_protons
            protons = {name: np.asarray(data[name]) for name in PROTON_FIELDS}
        if args.minbias_max_bx is not None:
            keep = protons["bx_id"] < args.minbias_max_bx
            protons = {name: values[keep] for name, values in protons.items()}
        n_protons_kept += protons["xi"].size
        print(
            f"  Loaded {n_file_protons} protons; kept {protons['xi'].size} after BX filter",
            flush=True,
        )
        if protons["bx_id"].size == 0:
            continue
        bx_values, groups = grouped_indices(np.asarray(protons["bx_id"], dtype=np.int64))
        print(f"  Processing {len(bx_values)} BX from this file", flush=True)
        for bx, indices in zip(bx_values, groups):
            chunk = minbias_pair_chunk_for_bx(protons, indices, n_bx, bx, args, rng)
            if chunk is not None:
                chunks.append(chunk)
                n_pairs += len(chunk["MX"])
            n_bx += 1
            if should_report_incremental(n_bx, args.progress_interval):
                print(
                    f"  Minbias BX processed: {n_bx}; "
                    f"accepted pairs: {n_pairs} (+{n_pairs - n_pairs_before} since last report)",
                    flush=True,
                )
                n_pairs_before = n_pairs
    return files, concatenate_pair_chunks(chunks), n_protons_loaded, n_protons_kept, n_bx


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(sig_values, mb_values):
    finite_sets = []
    for values in (sig_values, mb_values):
        values = finite_values(values)
        if values.size:
            finite_sets.append(values)
    if not finite_sets:
        return None
    values = np.concatenate(finite_sets)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0.0 else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def normalized_histogram(values, bins, value_range):
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(np.sum(counts))
    if total > 0.0:
        y = counts / (total * widths)
        yerr = np.sqrt(counts) / (total * widths)
    else:
        y = np.zeros_like(centers, dtype=np.float64)
        yerr = np.zeros_like(centers, dtype=np.float64)
    return centers, edges, y, yerr


def draw_hist_with_errors(ax, values, bins, value_range, color, marker, label):
    centers, edges, y, yerr = normalized_histogram(values, bins, value_range)
    nonzero = y > 0.0
    ax.stairs(y, edges, color=color, linewidth=2.0, label=label if not np.any(nonzero) else None)
    if np.any(nonzero):
        ax.errorbar(
            centers[nonzero],
            y[nonzero],
            yerr=yerr[nonzero],
            fmt=marker,
            markersize=4.5,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=1.8,
            linestyle="none",
            label=label,
        )


def plot_variable(signal_pairs, minbias_pairs, branch, xlabel, bins, output_dir, value_range_override=None):
    sig_values = finite_values(signal_pairs[branch])
    mb_values = finite_values(minbias_pairs[branch])
    value_range = value_range_override if value_range_override is not None else histogram_range(sig_values, mb_values)
    if value_range is None:
        print(f"Warning: no finite values for {branch}; skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    plotted = 0
    if mb_values.size:
        draw_hist_with_errors(
            ax,
            mb_values,
            bins=bins,
            value_range=value_range,
            color="#1f5fbf",
            marker="s",
            label=f"Minbias (n={mb_values.size})",
        )
        plotted += 1
    if sig_values.size:
        draw_hist_with_errors(
            ax,
            sig_values,
            bins=bins,
            value_range=value_range,
            color="#d62728",
            marker="o",
            label=f"Signal (n={sig_values.size})",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized entries")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=10, frameon=False)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{branch}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {out_path}")
    return out_path


def write_root(root_out, signal_pairs, minbias_pairs):
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError(
            "uproot is required for --root-out. Install with: python3 -m pip install --user uproot awkward"
        ) from exc

    out_dir = os.path.dirname(root_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def root_arrays(arrays):
        out = {}
        for name in INT_BRANCHES:
            out[name] = arrays[name].astype(np.int32, copy=False)
        for name in FLOAT_BRANCHES:
            out[name] = arrays[name].astype(np.float32, copy=False)
        return out

    print(f"[uproot] Writing output file: {root_out}", flush=True)
    with uproot.recreate(root_out) as fout:
        fout["SignalPairs"] = root_arrays(signal_pairs)
        fout["MinbiasPairs"] = root_arrays(minbias_pairs)
    print(f"Wrote ROOT file: {root_out}")


def validate_args(args):
    checks = (
        ("--signal-max-files", args.signal_max_files, 1),
        ("--signal-max-events", args.signal_max_events, 1),
        ("--minbias-max-files", args.minbias_max_files, 1),
        ("--minbias-max-bx", args.minbias_max_bx, 1),
        ("--bins", args.bins, 1),
        ("--progress-interval", args.progress_interval, 1),
    )
    for name, value, minimum in checks:
        if value is not None and value < minimum:
            raise RuntimeError(f"{name} must be >= {minimum}")
    if args.mass_window is not None and args.mass_window < 0.0:
        raise RuntimeError("--mass-window must be >= 0")
    if args.xi_res < 0.0:
        raise RuntimeError("--xi-res must be >= 0")
    if args.s <= 0.0:
        raise RuntimeError("--s must be > 0")


def validate_plot_configs():
    known_float_branches = set(FLOAT_BRANCHES)
    for branch, plot_options in PLOT_VARIABLES.items():
        if branch not in known_float_branches:
            raise RuntimeError(f"Unknown plot branch in PLOT_VARIABLES: {branch}")
        plot_range = plot_options["range"]
        if plot_range is not None:
            if len(plot_range) != 2 or not plot_range[0] < plot_range[1]:
                raise RuntimeError(f"Invalid plot range for {branch}; expected (min, max)")
        bins = plot_options["bins"]
        if bins is not None and bins <= 0:
            raise RuntimeError(f"Invalid plot bins for {branch}; expected a positive integer")


def main():
    default_output_dir = os.path.join("analysis", "output", "sig_minbias_pps")
    parser = argparse.ArgumentParser(
        description="Compare selected signal and minbias proton-pair kinematics."
    )
    parser.add_argument("--signal-in", required=True, help="Signal LHE/dat file or directory")
    parser.add_argument("--minbias-in", required=True, help="Minbias NPZ file or directory")
    parser.add_argument("--signal-max-files", type=int, default=None, help="Optional max signal files to read")
    parser.add_argument("--signal-max-events", type=int, default=None, help="Optional max signal events to process")
    parser.add_argument("--minbias-max-files", type=int, default=None, help="Optional max minbias NPZ files to read")
    parser.add_argument("--minbias-max-bx", type=int, default=None, help="Keep minbias protons with bx_id < this value")
    parser.add_argument("--output-dir", default=default_output_dir, help=f"Output plot directory (default: {default_output_dir})")
    parser.add_argument("--root-out", default=None, help="Optional ROOT output with SignalPairs and MinbiasPairs trees")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--mass-window", type=float, default=6.0, help="Mass half-width in GeV around the Higgs mass")
    parser.add_argument("--xi-res", type=float, default=0.0003, help="Gaussian fractional xi resolution sigma; 0 disables smearing")
    parser.add_argument("--s", type=float, default=S_GEV2, help="s value for MX=sqrt(xi_L*xi_R*s)")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for xi smearing")
    parser.add_argument("--require-two-400-tags", action="store_true", help="Require both protons in a pair to be tagged at 420 m")
    parser.add_argument("--progress-interval", type=int, default=10000, help="Report event/BX processing progress every N entries")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Increase output verbosity")
    args = parser.parse_args()
    validate_args(args)
    validate_plot_configs()

    signal_files, signal_pairs, malformed_signal, xsec_from_file, n_signal_events = process_signal_pairs(
        args.signal_in,
        args,
    )
    minbias_files, minbias_pairs, n_minbias_protons_loaded, n_minbias_protons_kept, n_minbias_bx = process_minbias_pairs(
        args.minbias_in,
        args,
    )

    print(f"Loaded and processed {n_signal_events} signal events from {len(signal_files)} files")
    if xsec_from_file is not None:
        print(f"Signal cross section from file: {xsec_from_file} pb")
    print(
        f"Loaded {n_minbias_protons_loaded} minbias protons from {len(minbias_files)} files; "
        f"kept {n_minbias_protons_kept} protons across {n_minbias_bx} BX"
    )
    if malformed_signal:
        print(f"Skipped {malformed_signal} signal events without exactly two outgoing protons")

    print(f"Accepted signal pairs: {signal_pairs['MX'].size}")
    print(f"Accepted minbias pairs: {minbias_pairs['MX'].size}")

    os.makedirs(args.output_dir, exist_ok=True)
    written = []
    for branch, plot_options in PLOT_VARIABLES.items():
        out_path = plot_variable(
            signal_pairs,
            minbias_pairs,
            branch,
            plot_options["label"],
            plot_options["bins"] if plot_options["bins"] is not None else args.bins,
            args.output_dir,
            value_range_override=plot_options["range"],
        )
        if out_path is not None:
            written.append(out_path)

    if args.root_out:
        write_root(args.root_out, signal_pairs, minbias_pairs)

    print(f"Wrote {len(written)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
