#!/usr/bin/env python3
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


C_CM_PER_PS = 2.99792458e-2

DEFAULT_OUTPUT = os.path.join("analysis", "output", "bx_model.png")
DEFAULT_SEED = 12345

ZLIM_CM = (-15.0, 20.0)
TLIM_PS = (-400.0, 400.0)
FIGSIZE = (9.0, 6.0)
DPI = 160

BEAM_SIGMA_Z_CM = 5.7
CENTRAL_SIGMA_Z_CM = 0.1
CENTRAL_SIGMA_T_PS = 50.0
PPS_SINGLE_ARM_SIGMA_T_PS = (3.0, 10.0)

DRAW_BEAM_SPOT = True
DRAW_RESOLUTION_GUIDE = True
DRAW_CENTRAL_TIMING_OVAL = True

VERTEX_COLOR = "#555555"
RIGHT_PROTON_COLOR = "#2f6fb0"
LEFT_PROTON_COLOR = "#c93c3c"
PAIR_COLOR = "#6d3fb3"
BEAM_SPOT_COLOR = "#777777"
PPS_GUIDE_COLOR = "#1f7a5a"
CENTRAL_GUIDE_COLOR = "#c07a1f"

VERTEX_MARKER_SIZE = 34
IMPLIED_VERTEX_MARKER_SIZE = 58
LINE_WIDTH = 1.4
GUIDE_LINE_WIDTH = 1.2
GUIDE_TEXT_SIZE = 8

GUIDE_X_CM = 13.0
GUIDE_Y_PS = (290.0, 110.0, -120.0)
GUIDE_LABEL_DX_CM = 1.0
GUIDE_LABEL_DY_PS = 34.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a standalone toy bunch-crossing z-t model with PPS timing guides."
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help=f"Output image path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    return parser.parse_args()


def beam_sigma_t_ps():
    return BEAM_SIGMA_Z_CM / C_CM_PER_PS


def pps_vertex_sigmas(single_arm_sigma_t_ps):
    sigma_z = 0.5 * C_CM_PER_PS * math.sqrt(2.0) * single_arm_sigma_t_ps
    return sigma_z, sigma_z / C_CM_PER_PS


def generate_vertices(seed):
    rng = np.random.default_rng(seed)
    z = [-5.0, -5.0] #rng.normal(loc=0.0, scale=BEAM_SIGMA_Z_CM, size=2)
    t = [-200,100] #rng.normal(loc=0.0, scale=beam_sigma_t_ps(), size=2)
    return [
        {"z": float(z[0]), "t": float(t[0]), "label": "vertex 0"},
        {"z": float(z[1]), "t": float(t[1]), "label": "vertex 1"},
    ]


def ray_segment(vertex, side):
    z_min, z_max = ZLIM_CM
    t_min, t_max = TLIM_PS
    z0 = float(vertex["z"])
    t0 = float(vertex["t"])
    direction = -1.0 if side < 0 else 1.0

    if t0 >= t_max:
        return None

    t_start = max(t0, t_min)
    z_start = z0 + direction * C_CM_PER_PS * (t_start - t0)
    if z_start < z_min or z_start > z_max:
        return None

    candidates = []
    for z_edge in (z_min, z_max):
        dt = (z_edge - z0) / (direction * C_CM_PER_PS)
        t_edge = t0 + dt
        if t_edge >= t_start and t_edge <= t_max:
            candidates.append((dt, z_edge, t_edge))

    z_at_t_max = z0 + direction * C_CM_PER_PS * (t_max - t0)
    if z_min <= z_at_t_max <= z_max:
        candidates.append((t_max - t0, z_at_t_max, t_max))

    if not candidates:
        return None

    _dt, z_end, t_end = min(candidates, key=lambda item: item[0])
    return z_start, t_start, z_end, t_end


def implied_pair_vertex(left_vertex, right_vertex):
    z_left = float(left_vertex["z"])
    t_left = float(left_vertex["t"])
    z_right = float(right_vertex["z"])
    t_right = float(right_vertex["t"])
    z_vertex = 0.5 * (z_left + z_right) + 0.5 * C_CM_PER_PS * (t_left - t_right)
    t_vertex = 0.5 * (t_left + t_right) + 0.5 * (z_left - z_right) / C_CM_PER_PS
    return {"z": z_vertex, "t": t_vertex, "label": "PPS-implied vertex"}


def add_sigma_ellipse(ax, center, sigma_z_cm, sigma_t_ps, nsigma, color, linestyle=":", alpha=0.85):
    ellipse = Ellipse(
        xy=center,
        width=2.0 * nsigma * sigma_z_cm,
        height=2.0 * nsigma * sigma_t_ps,
        fill=False,
        edgecolor=color,
        linewidth=GUIDE_LINE_WIDTH,
        linestyle=linestyle,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(ellipse)
    return ellipse


def draw_beam_spot(ax):
    for nsigma in [2.0]:
        add_sigma_ellipse(
            ax,
            (0.0, 0.0),
            BEAM_SIGMA_Z_CM,
            beam_sigma_t_ps(),
            nsigma,
            BEAM_SPOT_COLOR,
            linestyle=":",
            alpha=0.65,
        )


def draw_vertices_and_protons(ax, vertices):
    right_vertex = vertices[0]
    left_vertex = vertices[1]
    implied_vertex = implied_pair_vertex(left_vertex, right_vertex)

    for idx, vertex in enumerate(vertices):
        ax.scatter(
            [vertex["z"]],
            [vertex["t"]],
            s=VERTEX_MARKER_SIZE,
            color=VERTEX_COLOR,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    for vertex, side, color, label in (
        (right_vertex, +1, RIGHT_PROTON_COLOR, "Right proton"),
        (left_vertex, -1, LEFT_PROTON_COLOR, "Left proton"),
    ):
        segment = ray_segment(vertex, side)
        if segment is None:
            continue
        z_start, t_start, z_end, t_end = segment
        ax.plot(
            [z_start, z_end],
            [t_start, t_end],
            color=color,
            linewidth=LINE_WIDTH,
            alpha=0.8,
            zorder=3,
        )
        label_t = t_end - 200.0 if t_end > TLIM_PS[1] - 30.0 else t_end
        ax.text(
            z_end - 7 if side > 0 else z_end + 7,
            label_t,
            label,
            fontsize=GUIDE_TEXT_SIZE,
            color=color,
            ha="right" if side > 0 else "left",
            va="top" if t_end > TLIM_PS[1] - 30.0 else "center",
        )

    for vertex in (left_vertex, right_vertex):
        ax.plot(
            [vertex["z"], implied_vertex["z"]],
            [vertex["t"], implied_vertex["t"]],
            color=PAIR_COLOR,
            linewidth=LINE_WIDTH,
            linestyle=":",
            alpha=0.85,
            zorder=2,
        )

    ax.scatter(
        [implied_vertex["z"]],
        [implied_vertex["t"]],
        marker="+",
        s=IMPLIED_VERTEX_MARKER_SIZE,
        color=PAIR_COLOR,
        linewidths=1.5,
        zorder=6,
    )
    ax.text(
        implied_vertex["z"] + 0.6,
        implied_vertex["t"],
        "PPS-reco vtx",
        fontsize=GUIDE_TEXT_SIZE,
        color=PAIR_COLOR,
        ha="left",
        va="top",
    )


def draw_pps_guide(ax, center, single_arm_sigma_t_ps):
    sigma_z, sigma_t = pps_vertex_sigmas(single_arm_sigma_t_ps)
    for nsigma in [2.0]:
        add_sigma_ellipse(ax, center, sigma_z, sigma_t, nsigma, PPS_GUIDE_COLOR, linestyle="-")
    #ax.scatter([center[0]], [center[1]], marker=".", s=5, color=PPS_GUIDE_COLOR, linewidths=1.1, zorder=4)
    ax.text(
        center[0] + GUIDE_LABEL_DX_CM,
        center[1],
        f"PPS {single_arm_sigma_t_ps:g} ps vertex",
        fontsize=GUIDE_TEXT_SIZE,
        color=PPS_GUIDE_COLOR,
        ha="left",
        va="center",
    )


def draw_central_guide(ax, center):
    z0, t0 = center
    ax.plot(
        [z0, z0],
        [t0 - CENTRAL_SIGMA_Z_CM / C_CM_PER_PS, t0 + CENTRAL_SIGMA_Z_CM / C_CM_PER_PS],
        color=CENTRAL_GUIDE_COLOR,
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=4,
    )
    if DRAW_CENTRAL_TIMING_OVAL:
        for nsigma in [2.0]:
            add_sigma_ellipse(
                ax,
                center,
                CENTRAL_SIGMA_Z_CM,
                CENTRAL_SIGMA_T_PS,
                nsigma,
                CENTRAL_GUIDE_COLOR,
                linestyle=":",
                alpha=0.75,
            )
            add_sigma_ellipse(
                ax,
                center,
                CENTRAL_SIGMA_Z_CM,
                1000,
                nsigma,
                CENTRAL_GUIDE_COLOR,
                linestyle=":",
                alpha=0.75,
            )
    ax.scatter([z0], [t0], marker="+", s=40, color=CENTRAL_GUIDE_COLOR, linewidths=1.1, zorder=5)
    ax.text(
        z0 + GUIDE_LABEL_DX_CM,
        t0,
        "L1T vertex\n(50ps timing res)",
        fontsize=GUIDE_TEXT_SIZE,
        color=CENTRAL_GUIDE_COLOR,
        ha="left",
        va="center",
    )


def draw_resolution_guide(ax):
    for y, single_arm_sigma_t_ps in zip(GUIDE_Y_PS[:2], PPS_SINGLE_ARM_SIGMA_T_PS):
        draw_pps_guide(ax, (GUIDE_X_CM, y), single_arm_sigma_t_ps)
    draw_central_guide(ax, (GUIDE_X_CM, GUIDE_Y_PS[2]))


def draw_model(args):
    vertices = generate_vertices(args.seed)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if DRAW_BEAM_SPOT:
        draw_beam_spot(ax)
    draw_vertices_and_protons(ax, vertices)
    if DRAW_RESOLUTION_GUIDE:
        draw_resolution_guide(ax)

    ax.set_xlim(*ZLIM_CM)
    ax.set_ylim(*TLIM_PS)
    ax.set_aspect(C_CM_PER_PS, adjustable="box")
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("t [ps]")
    #ax.set_title("Toy bunch crossing model")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=DPI)
    plt.close(fig)
    print(f"Wrote plot: {args.output}")


def main():
    draw_model(parse_args())


if __name__ == "__main__":
    main()
