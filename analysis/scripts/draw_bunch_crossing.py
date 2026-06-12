#!/usr/bin/env python3
import argparse
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_minbias_vertices import (
    C_CM_PER_PS,
    S_GEV2,
    STATION_XI,
    build_interaction_model,
    group_indices_by_bx,
    load_minbias,
    primary_vertices_by_bx,
    proton_interaction_indices,
)
from analyze_minbias_vertices_signal import (
    M_HIGGS_GEV,
    load_signal_events,
    signal_protons_from_event,
)


STATION_ORDER = ("192", "213", "220", "420")


def station_names_for_xi(xi):
    names = []
    for station in STATION_ORDER:
        xi_min, xi_max = STATION_XI[station]
        if xi >= xi_min and xi < xi_max:
            names.append(station)
    return names


def is_tagged_xi(xi):
    return len(station_names_for_xi(xi)) > 0


def proton_hit_count_from_xis(xis):
    return sum(len(station_names_for_xi(xi)) for xi in xis)


def simple_proton_rows(protons, mask=None):
    if mask is None:
        mask = np.ones(len(protons["xi"]), dtype=bool)
    return [
        {
            "side": protons["side"][idx],
            "xi": protons["xi"][idx],
        }
        for idx in np.where(mask)[0]
    ]


def signal_candidate_rows(signal_protons):
    return [
        {
            "side": proton["side"],
            "xi": proton["xi"],
        }
        for proton in signal_protons
    ]


def bx_requirements_met(args, protons, bx):
    bx_mask = protons["bx_id"] == bx
    if args.min_bx_proton_hits is not None:
        n_hits = proton_hit_count_from_xis(protons["xi"][bx_mask])
        if n_hits < args.min_bx_proton_hits:
            return False
    if args.min_bx_pair_candidates is not None:
        n_pairs = len(passing_pair_candidates(simple_proton_rows(protons, bx_mask), args))
        if n_pairs < args.min_bx_pair_candidates:
            return False
    return True


def select_bx(args, model, protons):
    bx_values = np.unique(model["universe"][:, 0])
    if bx_values.size == 0:
        raise RuntimeError("No bunch crossings found in input.")

    if args.bx is not None:
        if args.bx not in set(int(bx) for bx in bx_values):
            raise RuntimeError(f"Requested BX {args.bx} is not present in the loaded input.")
        if args.min_bx_proton_hits is not None:
            bx_mask = protons["bx_id"] == args.bx
            n_hits = proton_hit_count_from_xis(protons["xi"][bx_mask])
            if n_hits < args.min_bx_proton_hits:
                raise RuntimeError(
                    f"Requested BX {args.bx} has {n_hits} proton detector hits, "
                    f"below --min-bx-proton-hits {args.min_bx_proton_hits}."
                )
        if args.min_bx_pair_candidates is not None:
            bx_mask = protons["bx_id"] == args.bx
            n_pairs = len(passing_pair_candidates(simple_proton_rows(protons, bx_mask), args))
            if n_pairs < args.min_bx_pair_candidates:
                raise RuntimeError(
                    f"Requested BX {args.bx} has {n_pairs} PPS+mass pair candidates, "
                    f"below --min-bx-pair-candidates {args.min_bx_pair_candidates}."
                )
        return int(args.bx)

    if args.min_bx_proton_hits is not None or args.min_bx_pair_candidates is not None:
        for bx in bx_values:
            if bx_requirements_met(args, protons, bx):
                return int(bx)
        requirements = []
        if args.min_bx_proton_hits is not None:
            requirements.append(f"{args.min_bx_proton_hits} proton detector hits")
        if args.min_bx_pair_candidates is not None:
            requirements.append(f"{args.min_bx_pair_candidates} PPS+mass pair candidates")
        raise RuntimeError(f"No loaded BX satisfies: {', '.join(requirements)}.")

    tagged = np.asarray([is_tagged_xi(xi) for xi in protons["xi"]], dtype=bool)
    if np.any(tagged):
        tagged_bx = np.unique(protons["bx_id"][tagged])
        available = np.intersect1d(tagged_bx, bx_values)
        if available.size:
            return int(available[0])

    return int(bx_values[0])


def parse_figsize(value):
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--figsize must look like WIDTHxHEIGHT, e.g. 9x6")
    try:
        width = float(parts[0])
        height = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--figsize values must be numeric") from exc
    if width <= 0.0 or height <= 0.0:
        raise argparse.ArgumentTypeError("--figsize values must be positive")
    return width, height


def detector_columns(vertex_z, signal_z=None, beam_sigma_z_cm=5.7):
    z_values = np.asarray(vertex_z, dtype=np.float64)
    if signal_z is not None:
        z_values = np.append(z_values, signal_z)

    if z_values.size:
        center = 0.5 * (float(np.min(z_values)) + float(np.max(z_values)))
        half_width = 0.5 * (float(np.max(z_values)) - float(np.min(z_values)))
    else:
        center = 0.0
        half_width = 0.0

    half_width = max(half_width + 0.15 * beam_sigma_z_cm, 4.0 * beam_sigma_z_cm, 20.0)
    vertex_min = center - half_width
    vertex_max = center + half_width
    span = vertex_max - vertex_min
    gap = 0.28 * span
    spacing = 0.075 * span

    left_x = {}
    right_x = {}
    for idx, station in enumerate(STATION_ORDER):
        left_x[station] = vertex_min - gap - (idx * spacing)
        right_x[station] = vertex_max + gap + (idx * spacing)

    x_min = left_x[STATION_ORDER[-1]] - 0.10 * span
    x_max = right_x[STATION_ORDER[-1]] + 0.10 * span
    return left_x, right_x, (x_min, x_max)


def draw_detector_columns(ax, left_x, right_x):
    for station in STATION_ORDER:
        for side, x in (("-", left_x[station]), ("+", right_x[station])):
            ax.axvline(x, color="0.82", linewidth=1.0, linestyle="--", zorder=0)
            ax.text(
                x,
                1.01,
                f"{side}{station}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.35",
            )


def plot_proton_lines(ax, proton_rows, source_label, color, marker, left_x, right_x):
    n_lines = 0
    for proton in proton_rows:
        station_names = station_names_for_xi(proton["xi"])
        if not station_names:
            continue

        side = int(proton["side"])
        if side < 0:
            endpoint_t = proton["t"] + proton["z"] / C_CM_PER_PS
            station_x = left_x
        else:
            endpoint_t = proton["t"] - proton["z"] / C_CM_PER_PS
            station_x = right_x

        for station in station_names:
            x_end = station_x[station]
            ax.plot(
                [proton["z"], x_end],
                [proton["t"], endpoint_t],
                color=color,
                alpha=0.65,
                linewidth=1.1,
                zorder=2,
            )
            ax.scatter(
                [x_end],
                [endpoint_t],
                color=color,
                marker=marker,
                s=34,
                linewidths=0.8,
                edgecolors="black",
                zorder=4,
                label=source_label if n_lines == 0 else None,
            )
            n_lines += 1
    return n_lines


def plot_scaled_forward_proton_lines(ax, proton_rows, source_label, color, marker, zlim, tlim):
    n_lines = 0
    z_min, z_max = zlim
    t_min, t_max = tlim

    for proton in proton_rows:
        if not station_names_for_xi(proton["xi"]):
            continue

        side = int(proton["side"])
        direction = -1.0 if side < 0 else 1.0
        z0 = float(proton["z"])
        t0 = float(proton["t"])

        if t0 >= t_max:
            continue

        t_start = max(t0, t_min)
        z_start = z0 + direction * C_CM_PER_PS * (t_start - t0)
        if z_start < z_min or z_start > z_max:
            continue

        z_edge = z_min if side < 0 else z_max
        t_at_z_edge = t0 + abs(z_edge - z0) / C_CM_PER_PS
        if t_at_z_edge <= t_max:
            z_end = z_edge
            t_end = t_at_z_edge
        else:
            t_end = t_max
            z_end = z0 + direction * C_CM_PER_PS * (t_end - t0)

        if z_end < z_min or z_end > z_max:
            continue

        ax.plot(
            [z_start, z_end],
            [t_start, t_end],
            color=color,
            alpha=0.65,
            linewidth=1.1,
            zorder=2,
            label=source_label if n_lines == 0 else None,
        )
        # ax.scatter(
        #     [z_start],
        #     [t_start],
        #     color=color,
        #     marker=marker,
        #     s=20,
        #     linewidths=0.7,
        #     edgecolors="black",
        #     zorder=4,
        # )
        n_lines += 1

    return n_lines


def proton_endpoint(proton, station, left_x, right_x):
    side = int(proton["side"])
    if side < 0:
        return left_x[station], proton["t"] + proton["z"] / C_CM_PER_PS
    return right_x[station], proton["t"] - proton["z"] / C_CM_PER_PS


def selected_station_for_pair(proton, prefer_420=False):
    station_names = station_names_for_xi(proton["xi"])
    if not station_names:
        return None
    if prefer_420 and "420" in station_names:
        return "420"
    return station_names[0]


def passing_pair_candidates(proton_rows, args):
    left_idx = [
        idx for idx, proton in enumerate(proton_rows)
        if int(proton["side"]) < 0 and is_tagged_xi(proton["xi"])
    ]
    right_idx = [
        idx for idx, proton in enumerate(proton_rows)
        if int(proton["side"]) > 0 and is_tagged_xi(proton["xi"])
    ]

    pairs = []
    for i_left in left_idx:
        for i_right in right_idx:
            left = proton_rows[i_left]
            right = proton_rows[i_right]
            left_stations = station_names_for_xi(left["xi"])
            right_stations = station_names_for_xi(right["xi"])
            left_400 = "420" in left_stations
            right_400 = "420" in right_stations
            if not (left_400 or right_400):
                continue
            if args.require_two_400_tags and not (left_400 and right_400):
                continue

            xi_prod = left["xi"] * right["xi"]
            if xi_prod <= 0.0:
                continue
            mass = np.sqrt(xi_prod * args.s)
            if args.mass_window is not None and abs(mass - args.mass_center) > args.mass_window:
                continue

            left_station = selected_station_for_pair(left, prefer_420=left_400)
            right_station = selected_station_for_pair(right, prefer_420=right_400)
            pairs.append((left, right, left_station, right_station, mass))

    return pairs


def plot_pair_candidates(ax, pair_candidates, left_x, right_x):
    for pair_idx, (left, right, left_station, right_station, _mass) in enumerate(pair_candidates):
        x_left, t_left = proton_endpoint(left, left_station, left_x, right_x)
        x_right, t_right = proton_endpoint(right, right_station, left_x, right_x)
        ax.plot(
            [x_left, x_right],
            [t_left, t_right],
            color="#6d3fb3",
            alpha=0.85,
            linewidth=1.3,
            linestyle="--",
            zorder=1,
            label="passing PPS+mass proton pair" if pair_idx == 0 else None,
        )
    return len(pair_candidates)


def scaled_ray_segment(z0, t0, side, zlim, tlim):
    z_min, z_max = zlim
    t_min, t_max = tlim
    direction = -1.0 if int(side) < 0 else 1.0

    if t0 >= t_max:
        return None

    t_start = max(t0, t_min)
    z_start = z0 + direction * C_CM_PER_PS * (t_start - t0)
    if z_start < z_min or z_start > z_max:
        return None

    z_edge = z_min if int(side) < 0 else z_max
    t_at_z_edge = t0 + abs(z_edge - z0) / C_CM_PER_PS
    if t_at_z_edge <= t_max:
        z_end = z_edge
        t_end = t_at_z_edge
    else:
        t_end = t_max
        z_end = z0 + direction * C_CM_PER_PS * (t_end - t0)

    if z_end < z_min or z_end > z_max:
        return None

    return z_start, t_start, z_end, t_end


def implied_pair_vertex(left, right):
    z_left = float(left["z"])
    t_left = float(left["t"])
    z_right = float(right["z"])
    t_right = float(right["t"])
    z_vertex = 0.5 * (z_left + z_right) + 0.5 * C_CM_PER_PS * (t_left - t_right)
    t_vertex = 0.5 * (t_left + t_right) + 0.5 * (z_left - z_right) / C_CM_PER_PS
    return z_vertex, t_vertex


def pair_pt_balance(left, right):
    pt_balance = np.hypot(float(left["px"]) + float(right["px"]), float(left["py"]) + float(right["py"]))
    pt_sum = np.hypot(float(left["px"]), float(left["py"])) + np.hypot(float(right["px"]), float(right["py"]))
    pt_balance_ratio = pt_balance / pt_sum if pt_sum > 0.0 else 0.0
    return pt_balance, pt_balance_ratio


def plot_scaled_pair_candidates(ax, pair_candidates, zlim, tlim):
    n_pairs = 0
    for pair_idx, (left, right, _left_station, _right_station, _mass) in enumerate(pair_candidates):
        if left.get("source") == "signal" and right.get("source") == "signal":
            continue

        z_vertex, t_vertex = implied_pair_vertex(left, right)
        label = "passing PPS+mass proton pair" if n_pairs == 0 else None
        for proton in (left, right):
            ax.plot(
                [float(proton["z"]), z_vertex],
                [float(proton["t"]), t_vertex],
                color="#6d3fb3",
                alpha=0.85,
                linewidth=1.3,
                linestyle=":",
                zorder=1,
                label=label,
            )
            label = None

        if zlim[0] <= z_vertex <= zlim[1] and tlim[0] <= t_vertex <= tlim[1]:
            ax.scatter(
                [z_vertex],
                [t_vertex],
                marker="x",
                s=34,
                color="#6d3fb3",
                linewidths=1.1,
                zorder=6,
            )
        n_pairs += 1

    return n_pairs


def load_signal(args):
    if args.signal_in is None:
        return None, None, []

    signal_files, signal_events, _xsec = load_signal_events(args.signal_in)
    if args.signal_event_index < 0 or args.signal_event_index >= len(signal_events):
        raise RuntimeError(
            f"--signal-event-index {args.signal_event_index} is outside the available "
            f"range 0..{len(signal_events) - 1}"
        )

    if args.min_signal_proton_hits is not None or args.min_signal_pair_candidates is not None:
        for event_idx in range(args.signal_event_index, len(signal_events)):
            signal_protons = signal_protons_from_event(signal_events[event_idx])
            if signal_protons is None:
                continue
            if args.min_signal_proton_hits is not None:
                xis = [proton["xi"] for proton in signal_protons]
                if proton_hit_count_from_xis(xis) < args.min_signal_proton_hits:
                    continue
            if args.min_signal_pair_candidates is not None:
                n_pairs = len(passing_pair_candidates(signal_candidate_rows(signal_protons), args))
                if n_pairs < args.min_signal_pair_candidates:
                    continue
            return signal_files, event_idx, signal_protons

        requirements = []
        if args.min_signal_proton_hits is not None:
            requirements.append(f"{args.min_signal_proton_hits} proton detector hits")
        if args.min_signal_pair_candidates is not None:
            requirements.append(f"{args.min_signal_pair_candidates} PPS+mass pair candidates")
        raise RuntimeError(
            f"No signal event from index {args.signal_event_index} onward satisfies: "
            f"{', '.join(requirements)}."
        )

    signal_event = signal_events[args.signal_event_index]
    signal_protons = signal_protons_from_event(signal_event)
    if signal_protons is None:
        raise RuntimeError(
            f"Signal event {args.signal_event_index} does not contain exactly two final-state protons."
        )
    return signal_files, args.signal_event_index, signal_protons


def build_signal_rows(args, signal_protons):
    if not signal_protons:
        return None, []

    rng = np.random.default_rng(args.seed + 999001)
    signal_z = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm)
    signal_t = rng.normal(loc=0.0, scale=args.beam_sigma_z_cm / C_CM_PER_PS)
    rows = []
    for proton in signal_protons:
        rows.append(
            {
                "z": signal_z,
                "t": signal_t,
                "side": proton["side"],
                "xi": proton["xi"],
                "px": proton["px"],
                "py": proton["py"],
                "source": "signal",
            }
        )
    return {"z": signal_z, "t": signal_t}, rows


def make_minbias_rows(model, bx_protons, bx_proton_model_idx):
    rows = []
    for local_idx in range(len(bx_protons["xi"])):
        model_idx = bx_proton_model_idx[local_idx]
        rows.append(
            {
                "z": model["z"][model_idx],
                "t": model["t"][model_idx],
                "side": bx_protons["side"][local_idx],
                "xi": bx_protons["xi"][local_idx],
                "px": bx_protons["px"][local_idx],
                "py": bx_protons["py"][local_idx],
                "source": "minbias",
            }
        )
    return rows


def format_float(value, digits=6):
    return f"{float(value):.{digits}g}"


def track_passes_cuts(tracks, idx, args):
    return (
        float(tracks["trk_pt"][idx]) > args.trk_pt_min
        and abs(float(tracks["trk_eta"][idx])) < args.trk_eta_max
    )


def print_minbias_proton(protons, idx, prefix):
    xi = float(protons["xi"][idx])
    stations = station_names_for_xi(xi)
    station_text = ",".join(stations) if stations else "none"
    print(
        prefix
        + (
            f"proton_idx={int(protons['proton_idx'][idx])} "
            f"side={int(protons['side'][idx]):+d} "
            f"px={format_float(protons['px'][idx])} GeV "
            f"py={format_float(protons['py'][idx])} GeV "
            f"pz={format_float(protons['pz'][idx])} GeV "
            f"E={format_float(protons['E'][idx])} GeV "
            f"m={format_float(protons['m'][idx])} GeV "
            f"pt={format_float(protons['pt'][idx])} GeV "
            f"xi={format_float(xi)} "
            f"stations={station_text}"
        )
    )


def print_track(tracks, idx, args, prefix):
    passes = track_passes_cuts(tracks, idx, args)
    print(
        prefix
        + (
            f"trk_idx={int(tracks['trk_idx'][idx])} "
            f"pdg_id={int(tracks['trk_pdg_id'][idx])} "
            f"charge={int(tracks['trk_charge'][idx])} "
            f"px={format_float(tracks['trk_px'][idx])} GeV "
            f"py={format_float(tracks['trk_py'][idx])} GeV "
            f"pz={format_float(tracks['trk_pz'][idx])} GeV "
            f"E={format_float(tracks['trk_E'][idx])} GeV "
            f"pt={format_float(tracks['trk_pt'][idx])} GeV "
            f"eta={format_float(tracks['trk_eta'][idx])} "
            f"passes_cuts={passes}"
        )
    )


def print_signal_proton(proton, prefix):
    xi = float(proton["xi"])
    stations = station_names_for_xi(xi)
    station_text = ",".join(stations) if stations else "none"
    print(
        prefix
        + (
            f"proton_idx={int(proton['proton_idx'])} "
            f"side={int(proton['side']):+d} "
            f"pz={format_float(proton['pz'])} GeV "
            f"pt={format_float(proton['pt'])} GeV "
            f"xi={format_float(xi)} "
            f"stations={station_text}"
        )
    )


def print_pair_candidate(pair_idx, pair):
    left, right, left_station, right_station, mass = pair
    z_vertex, t_vertex = implied_pair_vertex(left, right)
    pt_balance, pt_balance_ratio = pair_pt_balance(left, right)
    print(
        "  "
        + (
            f"pair {pair_idx}: "
            f"left_source={left.get('source', 'unknown')} "
            f"left_side={int(left['side']):+d} left_xi={format_float(left['xi'])} "
            f"left_station={left_station} "
            f"right_source={right.get('source', 'unknown')} "
            f"right_side={int(right['side']):+d} right_xi={format_float(right['xi'])} "
            f"right_station={right_station} "
            f"mass={format_float(mass)} GeV "
            f"pt_balance={format_float(pt_balance)} GeV "
            f"pt_balance_ratio={format_float(pt_balance_ratio)} "
            f"implied_z={format_float(z_vertex)} cm "
            f"implied_t={format_float(t_vertex)} ps"
        )
    )


def print_bunch_crossing_details(
    args,
    model,
    protons,
    tracks,
    selected_bx,
    pv,
    bx_vertex_idx,
    signal_vertex,
    signal_protons,
    signal_event_idx,
    pair_candidates,
):
    print("")
    print(f"Detailed bunch crossing dump: BX {selected_bx}")
    print(
        f"Track selection for vertex ranking: pt > {args.trk_pt_min:g} GeV, "
        f"|eta| < {args.trk_eta_max:g}"
    )
    print(f"Minbias vertices/interactions: {bx_vertex_idx.size}")
    if pv is None:
        print("Primary vertex: none among interactions with selected tracks")
    else:
        print(
            "Primary vertex: "
            f"interaction_id={int(pv['interaction_id'])} "
            f"model_idx={int(pv['idx'])} "
            f"z={format_float(pv['z'])} cm "
            f"t={format_float(pv['t'])} ps "
            f"selected_tracks={int(pv['n_tracks'])} "
            f"sum_pt2={format_float(pv['sum_pt2'])} GeV^2"
        )

    for model_idx in bx_vertex_idx:
        interaction = int(model["universe"][model_idx, 1])
        proton_idx = np.where(
            (protons["bx_id"] == selected_bx)
            & (protons["interaction_id"] == interaction)
        )[0]
        track_idx = np.where(
            (tracks["trk_bx_id"] == selected_bx)
            & (tracks["trk_interaction_id"] == interaction)
        )[0]
        n_selected_tracks = sum(track_passes_cuts(tracks, idx, args) for idx in track_idx)
        pv_marker = " primary_vertex=True" if pv is not None and interaction == int(pv["interaction_id"]) else ""
        print(
            f"Vertex interaction_id={interaction} model_idx={int(model_idx)} "
            f"z={format_float(model['z'][model_idx])} cm "
            f"t={format_float(model['t'][model_idx])} ps "
            f"sum_pt2={format_float(model['sum_pt2'][model_idx])} GeV^2 "
            f"selected_tracks={int(model['n_tracks'][model_idx])} "
            f"selected_tracks_recounted={n_selected_tracks} "
            f"all_tracks={len(track_idx)} "
            f"protons={len(proton_idx)}"
            f"{pv_marker}"
        )
        for local_idx, idx in enumerate(proton_idx):
            print_minbias_proton(protons, idx, f"  Proton {local_idx}: ")
        for local_idx, idx in enumerate(track_idx):
            print_track(tracks, idx, args, f"  Track {local_idx}: ")

    if signal_vertex is not None:
        print(
            f"Signal vertex: event_index={signal_event_idx} "
            f"z={format_float(signal_vertex['z'])} cm "
            f"t={format_float(signal_vertex['t'])} ps "
            f"protons={len(signal_protons)}"
        )
        for idx, proton in enumerate(signal_protons):
            print_signal_proton(proton, f"  Signal proton {idx}: ")
    else:
        print("Signal vertex: none")

    print(f"Passing PPS+mass proton pair candidates: {len(pair_candidates)}")
    for pair_idx, pair in enumerate(pair_candidates):
        print_pair_candidate(pair_idx, pair)


def draw_bunch_crossing(args):
    files, protons, tracks, universe = load_minbias(
        args.input,
        max_files=args.max_files,
        max_bx=args.max_bx,
    )
    model_args = SimpleNamespace(
        seed=args.seed,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        trk_pt_min=args.trk_pt_min,
        trk_eta_max=args.trk_eta_max,
    )
    model = build_interaction_model(universe, protons, tracks, model_args, verbosity=args.verbose)
    selected_bx = select_bx(args, model, protons)

    all_bx, pv_by_bx = primary_vertices_by_bx(model)
    if selected_bx not in set(int(bx) for bx in all_bx):
        raise RuntimeError(f"Selected BX {selected_bx} is not present after model construction.")
    pv = pv_by_bx[selected_bx]

    proton_model_idx = proton_interaction_indices(protons, model)
    unique_pbx, proton_groups = group_indices_by_bx(protons["bx_id"])
    proton_group_by_bx = {
        int(bx): group for bx, group in zip(unique_pbx, proton_groups)
    }
    idx_bx = proton_group_by_bx.get(selected_bx, np.empty(0, dtype=np.int64))
    bx_protons = {name: arr[idx_bx] for name, arr in protons.items()}
    bx_proton_model_idx = proton_model_idx[idx_bx]
    minbias_proton_rows = make_minbias_rows(model, bx_protons, bx_proton_model_idx)

    signal_files, signal_event_idx, signal_protons = load_signal(args)
    signal_vertex, signal_proton_rows = build_signal_rows(args, signal_protons)

    bx_vertex_idx = np.where(model["universe"][:, 0] == selected_bx)[0]
    if bx_vertex_idx.size == 0:
        raise RuntimeError(f"Selected BX {selected_bx} has no interactions.")

    vertex_z = model["z"][bx_vertex_idx]
    vertex_t = model["t"][bx_vertex_idx]
    left_x, right_x, xlim = detector_columns(
        vertex_z,
        signal_z=signal_vertex["z"] if signal_vertex is not None else None,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
    )

    fig, ax = plt.subplots(figsize=args.figsize)
    vertex_has_tracks = model["n_tracks"][bx_vertex_idx] > 0
    if np.any(~vertex_has_tracks):
        ax.scatter(
            vertex_z[~vertex_has_tracks],
            vertex_t[~vertex_has_tracks],
            s=10,
            color="0.72",
            alpha=0.55,
            marker=".",
            label="minbias vertices without selected tracks",
            zorder=3,
        )
    if np.any(vertex_has_tracks):
        ax.scatter(
            vertex_z[vertex_has_tracks],
            vertex_t[vertex_has_tracks],
            s=20,
            color="0.55",
            alpha=0.75,
            label="minbias interactions",
            zorder=3,
        )

    proton_vertex_idx = np.unique(bx_proton_model_idx) if bx_proton_model_idx.size else np.empty(0, dtype=np.int64)
    tagged_model_idx = []
    for row, model_idx in zip(minbias_proton_rows, bx_proton_model_idx):
        if is_tagged_xi(row["xi"]):
            tagged_model_idx.append(model_idx)
    tagged_model_idx = np.unique(np.asarray(tagged_model_idx, dtype=np.int64)) if tagged_model_idx else np.empty(0, dtype=np.int64)

    # if proton_vertex_idx.size:
    #     ax.scatter(
    #         model["z"][proton_vertex_idx],
    #         model["t"][proton_vertex_idx],
    #         s=44,
    #         facecolors="none",
    #         edgecolors="#3b73b9",
    #         linewidths=1.2,
    #         label="interactions with protons",
    #         zorder=5,
    #     )
    # if tagged_model_idx.size:
    #     ax.scatter(
    #         model["z"][tagged_model_idx],
    #         model["t"][tagged_model_idx],
    #         s=62,
    #         facecolors="none",
    #         edgecolors="#1f9d55",
    #         linewidths=1.4,
    #         label="interactions with tagged protons",
    #         zorder=6,
    #     )

    if pv is not None and signal_vertex is None:
        ax.scatter(
            [pv["z"]],
            [pv["t"]],
            marker="*",
            s=150,
            color="#f2b01e",
            edgecolors="black",
            linewidths=0.7,
            label="central PV",
            zorder=7,
        )

    if signal_vertex is not None:
        ax.scatter(
            [signal_vertex["z"]],
            [signal_vertex["t"]],
            marker="D",
            s=10,
            color="#d64f4f",
            edgecolors="black",
            linewidths=0.7,
            label="signal vertex",
            zorder=8,
        )

    if args.scaled_forward_time:
        zlim = (-20.0, 20.0)
        tlim = (-600.0, 600.0)
        n_mb_lines = plot_scaled_forward_proton_lines(
            ax,
            minbias_proton_rows,
            "minbias proton detector hit",
            "#2f6fb0",
            "o",
            zlim,
            tlim,
        )
        n_signal_lines = plot_scaled_forward_proton_lines(
            ax,
            signal_proton_rows,
            "signal proton detector hit",
            "#c93c3c",
            "D",
            zlim,
            tlim,
        )
        pair_candidates = passing_pair_candidates(minbias_proton_rows + signal_proton_rows, args)
        n_pair_lines = plot_scaled_pair_candidates(ax, pair_candidates, zlim, tlim)
        ax.set_xlim(*zlim)
        ax.set_ylim(*tlim)
        ax.set_aspect(C_CM_PER_PS, adjustable="box")
        ax.set_xlabel("z [cm]")
    else:
        draw_detector_columns(ax, left_x, right_x)
        n_mb_lines = plot_proton_lines(
            ax,
            minbias_proton_rows,
            "minbias proton detector hit",
            "#2f6fb0",
            "o",
            left_x,
            right_x,
        )
        n_signal_lines = plot_proton_lines(
            ax,
            signal_proton_rows,
            "signal proton detector hit",
            "#c93c3c",
            "D",
            left_x,
            right_x,
        )
        pair_candidates = passing_pair_candidates(minbias_proton_rows + signal_proton_rows, args)
        n_pair_lines = plot_pair_candidates(ax, pair_candidates, left_x, right_x)
        ax.set_xlim(*xlim)
        ax.set_xlabel("z [cm] with compressed PPS detector columns")

    ax.set_ylabel("t [ps]")
    ax.set_title(f"Bunch crossing {selected_bx}")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)

    n_tagged_mb = sum(1 for row in minbias_proton_rows if is_tagged_xi(row["xi"]))
    n_tagged_signal = sum(1 for row in signal_proton_rows if is_tagged_xi(row["xi"]))
    n_mb_hits = proton_hit_count_from_xis([row["xi"] for row in minbias_proton_rows])
    n_signal_hits = proton_hit_count_from_xis([row["xi"] for row in signal_proton_rows])
    print(f"Wrote plot: {args.output}")
    if args.verbose:
        print(f"Input files loaded: {len(files)}")
        print(f"Selected BX: {selected_bx}")
        print(f"Minbias interactions in BX: {bx_vertex_idx.size}")
        print(f"Minbias interactions passing track cuts: {int(np.sum(vertex_has_tracks))}")
        print(f"Minbias interactions failing track cuts: {int(np.sum(~vertex_has_tracks))}")
        print(f"Minbias protons in BX: {len(minbias_proton_rows)}")
        print(f"Tagged minbias protons in BX: {n_tagged_mb}")
        print(f"Minbias proton detector hits in BX: {n_mb_hits}")
        print(f"Drawn minbias proton detector lines: {n_mb_lines}")
        print(f"Drawn passing PPS+mass proton pair lines: {n_pair_lines}")
        if signal_files is not None:
            print(f"Signal files loaded: {len(signal_files)}")
            print(f"Selected signal event: {signal_event_idx}")
            print(f"Signal protons: {len(signal_proton_rows)}")
            print(f"Tagged signal protons: {n_tagged_signal}")
            print(f"Signal proton detector hits: {n_signal_hits}")
            print(f"Drawn signal proton detector lines: {n_signal_lines}")

    print_bunch_crossing_details(
        args,
        model,
        protons,
        tracks,
        selected_bx,
        pv,
        bx_vertex_idx,
        signal_vertex,
        signal_protons,
        signal_event_idx,
        pair_candidates,
    )


def main():
    default_output = os.path.join("analysis", "output", "bunch_crossing.png")
    parser = argparse.ArgumentParser(description="Draw one minbias bunch crossing in z-t spacetime.")
    parser.add_argument("-i", "--input", required=True, help="Input minbias NPZ file or directory of NPZ files")
    parser.add_argument("-o", "--output", default=default_output, help=f"Output image path (default: {default_output})")
    parser.add_argument("--bx", type=int, default=None, help="Specific BX id to draw")
    parser.add_argument("--min-bx-proton-hits", type=int, default=None, help="Require/select a BX with at least this many proton detector hits")
    parser.add_argument("--min-bx-pair-candidates", type=int, default=None, help="Require/select a BX with at least this many passing PPS+mass pair candidates")
    parser.add_argument("--signal-in", default=None, help="Optional signal LHE/dat file or directory")
    parser.add_argument("--signal-event-index", type=int, default=0, help="Signal event index to draw")
    parser.add_argument("--min-signal-proton-hits", type=int, default=None, help="Find a signal event with at least this many proton detector hits, starting at --signal-event-index")
    parser.add_argument("--min-signal-pair-candidates", type=int, default=None, help="Find a signal event with at least this many passing PPS+mass pair candidates, starting at --signal-event-index")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of minbias files to load")
    parser.add_argument("--max-bx", type=int, default=None, help="Optional maximum BX id to load")
    parser.add_argument("--trk-pt-min", type=float, default=2.0, help="Track proxy pT cut in GeV")
    parser.add_argument("--trk-eta-max", type=float, default=2.4, help="Track proxy |eta| cut")
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7, help="Gaussian beam spot sigma z in cm")
    parser.add_argument("--mass-center", type=float, default=M_HIGGS_GEV, help="Central mass value for PPS pair selection in GeV")
    parser.add_argument("--mass-window", type=float, default=6.0, help="Mass half-width in GeV for PPS pair selection")
    parser.add_argument("--s", type=float, default=S_GEV2, help="s value for M=sqrt(xi1*xi2*s)")
    parser.add_argument("--require-two-400-tags", action="store_true", help="Require both paired protons to be tagged at the 420m station")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for vertex sampling")
    parser.add_argument("--figsize", type=parse_figsize, default=(9.0, 6.0), help="Figure size as WIDTHxHEIGHT")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI")
    parser.add_argument("--scaled-forward-time", action="store_true",
                        help="Draw a to-scale z-t view with all proton worldlines going forward in time over +/-20 cm and +/-600 ps")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase output verbosity")
    args = parser.parse_args()

    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_bx is not None and args.max_bx <= 0:
        raise RuntimeError("--max-bx must be > 0")
    if args.trk_pt_min < 0.0:
        raise RuntimeError("--trk-pt-min must be >= 0")
    if args.trk_eta_max <= 0.0:
        raise RuntimeError("--trk-eta-max must be > 0")
    if args.beam_sigma_z_cm < 0.0:
        raise RuntimeError("--beam-sigma-z-cm must be >= 0")
    if args.mass_window is not None and args.mass_window < 0.0:
        raise RuntimeError("--mass-window must be >= 0")
    if args.s <= 0.0:
        raise RuntimeError("--s must be > 0")
    if args.signal_event_index < 0:
        raise RuntimeError("--signal-event-index must be >= 0")
    if args.min_bx_proton_hits is not None and args.min_bx_proton_hits < 0:
        raise RuntimeError("--min-bx-proton-hits must be >= 0")
    if args.min_bx_pair_candidates is not None and args.min_bx_pair_candidates < 0:
        raise RuntimeError("--min-bx-pair-candidates must be >= 0")
    if args.min_signal_proton_hits is not None and args.min_signal_proton_hits < 0:
        raise RuntimeError("--min-signal-proton-hits must be >= 0")
    if args.min_signal_pair_candidates is not None and args.min_signal_pair_candidates < 0:
        raise RuntimeError("--min-signal-pair-candidates must be >= 0")
    if args.dpi <= 0:
        raise RuntimeError("--dpi must be > 0")

    draw_bunch_crossing(args)


if __name__ == "__main__":
    main()
