#!/usr/bin/env python3
import argparse
import numpy as np
import math
import os
import faulthandler

# --- global constants ---
SIGMA_Z_CM   = 5.0        # HL-LHC-like luminous region RMS in z (cm)
Z_MATCH_CM   = 0.5         # 5 mm vertex matching resolution
E_BEAM_GEV   = 7000.0      # beam energy
SQRTS_GEV    = 2.0 * E_BEAM_GEV
S_GEV2       = SQRTS_GEV**2  # s = (2E_beam)^2

# Higgs mass window (for trigger / sideband)
M_H_GEV      = 125.0
M_WIN_LOW    = 110.0
M_WIN_HIGH   = 140.0

# Station xi windows
# STATION_XI = {
#     "192": (0.0140, 0.0250),   # ~196 m
#     "213": (0.0390, 0.0680),   # ~220 m
#     "220": (0.0390, 0.0680),   # ~234 m
#     "420": (0.00325, 0.0120),  # 420 m, low-xi region
# }
STATION_XI = {
    "192": (0.08, 0.1967),   # ~196 m
    "213": (0.0375, 0.0688),   # ~220 m
    "220": (0.014, 0.0263),   # ~234 m
    "420": (0.00325, 0.0116),  # 420 m, low-xi region
}

# HL-LHC bunch-crossing / collision rates
F_BX_PEAK_HZ  = 40.0e6    # 25 ns spacing → 40 MHz bunch crossings
F_COLL_AVG_HZ = 31.6e6    # ~2808 bunches × 11.245 kHz ≈ 31.6 MHz


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def wrap_dphi(dphi):
    """Wrap Δφ into [-π, π]."""
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi < -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def group_indices_by_bx(bx_id):
    """Return unique BX IDs and proton indices grouped by BX.

    This avoids rebuilding a full-length mask for each BX in all-BX mode.
    """
    order = np.argsort(bx_id, kind="mergesort")
    bx_sorted = bx_id[order]
    unique_bx, starts, counts = np.unique(
        bx_sorted, return_index=True, return_counts=True
    )
    groups = [order[s:s + c] for s, c in zip(starts, counts)]
    return unique_bx, groups


def cartesian_pair_indices(left_idx, right_idx):
    """Build flattened Cartesian-product pair indices for left/right arrays."""
    n_left = int(left_idx.size)
    n_right = int(right_idx.size)
    if n_left == 0 or n_right == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    i_left = np.repeat(left_idx, n_right)
    i_right = np.tile(right_idx, n_left)
    return i_left, i_right


# ----------------------------------------------------------------------
# Single-BX detailed mode (mostly unchanged, but can use px/py if present)
# ----------------------------------------------------------------------
def analyze_single_bx(
    bx_id, interaction_id, proton_idx,
    side, pz, pt, xi, z_vertex_cm, bx_to_inspect,
    px=None, py=None
):
    have_phi = (px is not None) and (py is not None)

    max_bx = int(bx_id.max())
    print(f"Available BX IDs: 0 .. {max_bx}")
    if not (0 <= bx_to_inspect <= max_bx):
        raise ValueError(f"Requested BX {bx_to_inspect} is out of range 0..{max_bx}")

    mask_bx = (bx_id == bx_to_inspect)
    indices = np.where(mask_bx)[0]

    print(f"\nInspecting BX = {bx_to_inspect}")
    print(f"Number of protons in this BX: {len(indices)}")
    print(f"(Beam spot: sigma_z = {SIGMA_Z_CM:.1f} cm)")
    print(f"(Mass formula: M^2 = xi_L * xi_R * s, sqrt(s) = {SQRTS_GEV:.1f} GeV)")
    print(f"(Vertex match: |z_L - z_R| <= {Z_MATCH_CM:.3f} cm)")
    print(f"(Higgs window for trigger / sideband: {M_WIN_LOW:.1f}–{M_WIN_HIGH:.1f} GeV)\n")

    # sort by interaction then proton index
    order = np.lexsort((proton_idx[indices], interaction_id[indices]))
    indices = indices[order]

    # helper masks for xi windows (on full arrays)
    xi_192_min, xi_192_max = STATION_XI["192"]
    xi_213_min, xi_213_max = STATION_XI["213"]
    xi_220_min, xi_220_max = STATION_XI["220"]
    xi_420_min, xi_420_max = STATION_XI["420"]

    in_192 = (xi >= xi_192_min) & (xi < xi_192_max)
    in_213 = (xi >= xi_213_min) & (xi < xi_213_max)
    in_220 = (xi >= xi_220_min) & (xi < xi_220_max)
    in_420 = (xi >= xi_420_min) & (xi < xi_420_max)

    # 200 m chamber tag: any of 192/213/220, NO pT or z cuts
    tag_200_all = (in_192 | in_213 | in_220)

    # 400 m chamber tag: 420 xi window, NO pT or z cuts
    tag_400_all = in_420

    # Restrict tag flags and z to this BX only
    tag_200 = tag_200_all[indices]
    tag_400 = tag_400_all[indices]
    z_bx    = z_vertex_cm[indices]
    pt_bx   = pt[indices]
    xi_bx   = xi[indices]
    side_bx = side[indices]
    if have_phi:
        px_bx = px[indices]
        py_bx = py[indices]
    else:
        px_bx = None
        py_bx = None

    # ---- FIRST PRINT: all protons in this BX ----
    print("All protons in this BX:")
    print(
        f"{'i':>5}  {'interaction':>11}  {'p_idx':>6}  "
        f"{'side':>6}  {'z [cm]':>8}  {'xi':>10}  "
        f"{'pz [GeV]':>12}  {'pT [GeV]':>10}  {'tag200':>7}  {'tag400':>7}"
    )
    print("-" * 120)

    tag_any_indices = []  # store (local_i, idx) for tagged protons

    for local_i, idx in enumerate(indices):
        inter = interaction_id[idx]
        p_i   = proton_idx[idx]
        sgn   = side[idx]
        direction = "right" if sgn > 0 else "left"
        pz_val = pz[idx]
        pt_val = pt_bx[local_i]
        z_val  = z_bx[local_i]
        xi_val = xi[idx]

        t200 = tag_200[local_i]
        t400 = tag_400[local_i]

        tag200_str = "Y" if t200 else "N"
        tag400_str = "Y" if t400 else "N"

        if t200 or t400:
            tag_any_indices.append((local_i, idx))

        print(
            f"{local_i:5d}  {inter:11d}  {p_i:6d}  "
            f"{direction:>6}  {z_val:8.2f}  {xi_val:10.5f}  "
            f"{pz_val:12.4f}  {pt_val:10.4f}  {tag200_str:>7}  {tag400_str:>7}"
        )

    # ---- SECOND PRINT: only tagged protons ----
    print("\nProtons in this BX that are tagged in 200 m and/or 400 m chambers:")
    if not tag_any_indices:
        print("  (None in this BX with current xi cuts.)")
    else:
        print(
            f"{'i':>5}  {'interaction':>11}  {'p_idx':>6}  "
            f"{'side':>6}  {'z [cm]':>8}  {'xi':>10}  "
            f"{'pz [GeV]':>12}  {'pT [GeV]':>10}  {'tag200':>7}  {'tag400':>7}"
        )
        print("-" * 120)

        for local_i, idx in tag_any_indices:
            inter = interaction_id[idx]
            p_i   = proton_idx[idx]
            sgn   = side[idx]
            direction = "right" if sgn > 0 else "left"
            pz_val = pz[idx]
            pt_val = pt_bx[local_i]
            z_val  = z_bx[local_i]
            xi_val = xi[idx]

            t200 = tag_200[local_i]
            t400 = tag_400[local_i]
            tag200_str = "Y" if t200 else "N"
            tag400_str = "Y" if t400 else "N"

            print(
                f"{local_i:5d}  {inter:11d}  {p_i:6d}  "
                f"{direction:>6}  {z_val:8.2f}  {xi_val:10.5f}  "
                f"{pz_val:12.4f}  {pt_val:10.4f}  {tag200_str:>7}  {tag400_str:>7}"
            )

    # ---- THIRD PART: mass candidates from tagged left/right protons ----
    print("\nMass candidates from tagged protons in this BX "
          "(with vertex consistency, Higgs-window flag and extra kinematics):")

    tag_any_bx = tag_200 | tag_400

    # left = side == -1, right = side == +1, and tagged
    left_idx  = np.where(tag_any_bx & (side_bx == -1))[0]
    right_idx = np.where(tag_any_bx & (side_bx == +1))[0]

    if len(left_idx) == 0 or len(right_idx) == 0:
        print("  No left-right tagged proton pair in this BX.")
    else:
        header = (
            f"{'pair':>4}  {'iL':>4}  {'iR':>4}  "
            f"{'xi_L':>10}  {'xi_R':>10}  "
            f"{'z_L [cm]':>9}  {'z_R [cm]':>9}  {'dz [cm]':>9}  "
            f"{'M [GeV]':>9}  {'yX':>7}  "
            f"{'|t1|':>7}  {'|t2|':>7}  {'|Δt|':>7}  "
            f"{'pt_bal':>9}  {'|Δφ|':>7}  "
            f"{'vtx_ok':>7}  {'in_Hwin':>9}"
        )
        print(header)
        print("-" * len(header))

        pair_counter = 0
        for iL in left_idx:
            for iR in right_idx:
                xi_L = xi_bx[iL]
                xi_R = xi_bx[iR]
                z_L  = z_bx[iL]
                z_R  = z_bx[iR]
                dz   = z_L - z_R
                M    = np.sqrt(xi_L * xi_R * S_GEV2)

                # vertex consistency
                vtx_ok = abs(dz) <= Z_MATCH_CM
                vtx_str = "True" if vtx_ok else "False"

                # Higgs window
                in_Hwin = (M >= M_WIN_LOW) and (M <= M_WIN_HIGH)
                Hwin_str = "True" if in_Hwin else "False"

                # extra kinematic variables (pair-level)
                # xi-based
                if (xi_L > 0.0) and (xi_R > 0.0):
                    ln_xi_ratio = math.log(xi_L / xi_R)
                else:
                    ln_xi_ratio = 0.0
                yX = 0.5 * ln_xi_ratio

                # t-like (using pT^2)
                pt_L = pt_bx[iL]
                pt_R = pt_bx[iR]
                t1_abs = pt_L * pt_L
                t2_abs = pt_R * pt_R
                t_diff_abs = abs(t1_abs - t2_abs)

                # phi & pT-balance if available
                if have_phi:
                    phi_L = math.atan2(py_bx[iL], px_bx[iL])
                    phi_R = math.atan2(py_bx[iR], px_bx[iR])
                    dphi = wrap_dphi(phi_L - phi_R)
                    abs_dphi = abs(dphi)
                    pt_bal_vec_x = px_bx[iL] + px_bx[iR]
                    pt_bal_vec_y = py_bx[iL] + py_bx[iR]
                    pt_bal = math.hypot(pt_bal_vec_x, pt_bal_vec_y)
                else:
                    abs_dphi = float("nan")
                    pt_bal = float("nan")

                print(
                    f"{pair_counter:4d}  {iL:4d}  {iR:4d}  "
                    f"{xi_L:10.5f}  {xi_R:10.5f}  "
                    f"{z_L:9.2f}  {z_R:9.2f}  {dz:9.3f}  "
                    f"{M:9.2f}  {yX:7.3f}  "
                    f"{t1_abs:7.3f}  {t2_abs:7.3f}  {t_diff_abs:7.3f}  "
                    f"{pt_bal:9.3f}  {abs_dphi:7.3f}  "
                    f"{vtx_str:>7}  {Hwin_str:>9}"
                )
                pair_counter += 1


# ----------------------------------------------------------------------
# All-BX summary mode (rates A/B/C, unchanged logic)
# ----------------------------------------------------------------------
def analyze_all_bx(bx_id, side, pt, xi, z_vertex_cm, unique_bx=None, bx_groups=None):
    """
    Whole-sample summary:
      - fraction of BX with >=1 proton tagged in 400m chamber,
      - fraction of BX with >=1 left-right pair of 400m-tagged protons
        with vtx_ok=True,
      - fraction of BX with >=1 such pair also in Higgs window.
      Then convert to trigger rates at 40 MHz and 31.6 MHz.
    """
    if unique_bx is None or bx_groups is None:
        unique_bx, bx_groups = group_indices_by_bx(bx_id)
    n_bx = int(unique_bx.size)

    # 400 m tag definition (global arrays): ONLY xi_420 window
    xi_420_min, xi_420_max = STATION_XI["420"]
    in_420 = (xi >= xi_420_min) & (xi < xi_420_max)

    tag_400_all = in_420

    # Event-level flags
    has_400_tag    = np.zeros(n_bx, dtype=bool)
    has_vtx_ok     = np.zeros(n_bx, dtype=bool)
    has_vtx_Hwin   = np.zeros(n_bx, dtype=bool)

    for ib, (bx, idx_bx) in enumerate(zip(unique_bx, bx_groups)):
        if ib % 1000 == 0:  # every 1000 bunch crossings
            prog = 100.0 * ib / n_bx if n_bx > 0 else 100.0
            print(f"[BX {ib}/{n_bx}]  {prog:.1f}% done")
        if idx_bx.size == 0:
            continue

        tag_400_bx = tag_400_all[idx_bx]
        side_bx    = side[idx_bx]
        xi_bx      = xi[idx_bx]
        z_bx       = z_vertex_cm[idx_bx]

        # Condition A: >= 1 proton tagged at 400 m
        n_tag = int(tag_400_bx.sum())
        if n_tag > 0:
            has_400_tag[ib] = True
        else:
            continue  # if no 400m tag, no chance for vtx_ok or Higgs-window pair

        # Left-right indices among 400m-tagged protons
        left_idx  = np.where(tag_400_bx & (side_bx == -1))[0]
        right_idx = np.where(tag_400_bx & (side_bx == +1))[0]

        if left_idx.size == 0 or right_idx.size == 0:
            continue

        z_L = z_bx[left_idx]
        z_R = z_bx[right_idx]
        vtx_ok_pairs = np.abs(z_L[:, None] - z_R[None, :]) <= Z_MATCH_CM

        if not np.any(vtx_ok_pairs):
            continue

        has_vtx_ok[ib] = True

        xi_L = xi_bx[left_idx]
        xi_R = xi_bx[right_idx]
        M_pairs = np.sqrt(np.maximum(np.multiply.outer(xi_L, xi_R) * S_GEV2, 0.0))
        in_Hwin_pairs = (M_pairs >= M_WIN_LOW) & (M_pairs <= M_WIN_HIGH)
        if np.any(vtx_ok_pairs & in_Hwin_pairs):
            has_vtx_Hwin[ib] = True

    # Now summary
    total_bx = float(n_bx)

    frac_400   = has_400_tag.sum()  / total_bx
    frac_vtx   = has_vtx_ok.sum()   / total_bx
    frac_Hwin  = has_vtx_Hwin.sum() / total_bx

    print("=== Whole-sample summary (per BX) ===")
    print(f"Total BX: {n_bx}")
    print(f"Condition A: >=1 proton tagged at 400 m (xi_420 window only)")
    print(f"  -> BX with A: {has_400_tag.sum()}  (fraction = {frac_400:.4e})")

    print(f"\nCondition B (cumulative): A AND >=1 left-right 400m-tagged pair with |dz| <= {Z_MATCH_CM:.3f} cm")
    print(f"  -> BX with B: {has_vtx_ok.sum()}  (fraction = {frac_vtx:.4e})")

    print(f"\nCondition C (cumulative): A AND B AND M in [{M_WIN_LOW:.1f}, {M_WIN_HIGH:.1f}] GeV")
    print(f"  -> BX with C: {has_vtx_Hwin.sum()}  (fraction = {frac_Hwin:.4e})")

    # --- convert to rates ---
    print("\n=== Estimated trigger rates ===")
    for label, freq in [
        ("Peak bunch-crossing clock (40 MHz)", F_BX_PEAK_HZ),
        ("Average pp collision rate (~31.6 MHz)", F_COLL_AVG_HZ),
    ]:
        R_A = frac_400  * freq
        R_B = frac_vtx  * freq
        R_C = frac_Hwin * freq

        print(f"\nAssuming {label}:")
        print(f"  Rate(A) = {R_A:9.3e} Hz  = {R_A/1e3:9.3f} kHz")
        print(f"  Rate(B) = {R_B:9.3e} Hz  = {R_B/1e3:9.3f} kHz")
        print(f"  Rate(C) = {R_C:9.3e} Hz  = {R_C/1e3:9.3f} kHz")
    print("======================================\n")


# ----------------------------------------------------------------------
# Build ROOT TTree with uproot (one entry per L-R pair passing double-tag+>=1 at 400 m)
# ----------------------------------------------------------------------
def build_pair_tree_all_bx(
    bx_id, interaction_id, proton_idx,
    side, pt, xi, z_vertex_cm,
    root_out, px=None, py=None, unique_bx=None, bx_groups=None
):
    have_phi = (px is not None) and (py is not None)
    print("[startup] Importing uproot...", flush=True)
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError(
            "uproot is required to write ROOT output. "
            "Install with: python3 -m pip install --user uproot awkward"
        ) from exc
    print(f"[startup] uproot imported (version {uproot.__version__})", flush=True)

    int_branches = [
        "bx",
        "interaction_L", "interaction_R",
        "p_idx_L", "p_idx_R",
        "side_L", "side_R",
        "tag200_L", "tag200_R",
        "tag400_L", "tag400_R",
        "double_tag_420",
        "vtx_ok", "in_Hwin",
    ]
    float_branches = [
        "xi_L", "xi_R",
        "z_L", "z_R", "dz",
        "M",
        "ln_xi_ratio", "yX", "abs_yX",
        "pT_L", "pT_R",
        "t1_abs", "t2_abs", "t_sum", "t_diff_abs",
        "pt_bal", "pt_bal_ratio",
        "abs_dphi", "dphi_from_pi",
    ]
    branch_chunks = {name: [] for name in int_branches + float_branches}

    # Station xi ranges
    xi_192_min, xi_192_max = STATION_XI["192"]
    xi_213_min, xi_213_max = STATION_XI["213"]
    xi_220_min, xi_220_max = STATION_XI["220"]
    xi_420_min, xi_420_max = STATION_XI["420"]

    if unique_bx is None or bx_groups is None:
        unique_bx, bx_groups = group_indices_by_bx(bx_id)
    n_bx = int(unique_bx.size)
    n_pairs_filled = 0

    for ib, (bx, idx_bx) in enumerate(zip(unique_bx, bx_groups)):
        if ib % 100 == 0:  # every 100 bunch crossings
            prog = 100.0 * ib / n_bx if n_bx > 0 else 100.0
            print(f"[BX {ib}/{n_bx}]  {prog:.1f}% done, pairs filled so far: {n_pairs_filled}")
        if idx_bx.size == 0:
            continue

        # Slice arrays for this BX
        interaction_bx = interaction_id[idx_bx]
        pidx_bx        = proton_idx[idx_bx]
        side_bx        = side[idx_bx]
        pt_bx          = pt[idx_bx]
        xi_bx          = xi[idx_bx]
        z_bx           = z_vertex_cm[idx_bx]
        if have_phi:
            px_bx = px[idx_bx]
            py_bx = py[idx_bx]
        else:
            px_bx = None
            py_bx = None

        # Tagging per proton
        in_192_bx = (xi_bx >= xi_192_min) & (xi_bx < xi_192_max)
        in_213_bx = (xi_bx >= xi_213_min) & (xi_bx < xi_213_max)
        in_220_bx = (xi_bx >= xi_220_min) & (xi_bx < xi_220_max)
        in_420_bx = (xi_bx >= xi_420_min) & (xi_bx < xi_420_max)

        tag200_bx = in_192_bx | in_213_bx | in_220_bx
        tag400_bx = in_420_bx
        tag_any_bx = tag200_bx | tag400_bx

        # Candidate L/R indices among tagged protons.
        left_tagged = np.where((side_bx == -1) & tag_any_bx)[0]
        right_tagged = np.where((side_bx == +1) & tag_any_bx)[0]
        if left_tagged.size == 0 or right_tagged.size == 0:
            continue

        left_tagged_400 = left_tagged[tag400_bx[left_tagged]]
        left_tagged_not400 = left_tagged[~tag400_bx[left_tagged]]
        right_tagged_400 = right_tagged[tag400_bx[right_tagged]]

        # Pairs that satisfy "at least one proton tagged at 400 m".
        iL_a, iR_a = cartesian_pair_indices(left_tagged_400, right_tagged)
        iL_b, iR_b = cartesian_pair_indices(left_tagged_not400, right_tagged_400)
        iL = np.concatenate((iL_a, iL_b)) if iL_b.size > 0 else iL_a
        iR = np.concatenate((iR_a, iR_b)) if iR_b.size > 0 else iR_a
        if iL.size == 0:
            continue

        n_pairs = int(iL.size)
        xi_L = xi_bx[iL].astype(np.float64, copy=False)
        xi_R = xi_bx[iR].astype(np.float64, copy=False)
        z_L = z_bx[iL].astype(np.float64, copy=False)
        z_R = z_bx[iR].astype(np.float64, copy=False)
        dz = z_L - z_R
        M = np.sqrt(np.maximum(xi_L * xi_R * S_GEV2, 0.0))

        vtx_ok = np.abs(dz) <= Z_MATCH_CM
        in_Hwin = (M >= M_WIN_LOW) & (M <= M_WIN_HIGH)

        ln_xi_ratio = np.zeros(n_pairs, dtype=np.float64)
        valid_xi = (xi_L > 0.0) & (xi_R > 0.0)
        np.log(xi_L[valid_xi] / xi_R[valid_xi], out=ln_xi_ratio[valid_xi])
        yX = 0.5 * ln_xi_ratio
        abs_yX = np.abs(yX)

        pT_L = pt_bx[iL].astype(np.float64, copy=False)
        pT_R = pt_bx[iR].astype(np.float64, copy=False)
        t1_abs = pT_L * pT_L
        t2_abs = pT_R * pT_R
        t_sum = t1_abs + t2_abs
        t_diff_abs = np.abs(t1_abs - t2_abs)

        if have_phi:
            phi_bx = np.arctan2(py_bx, px_bx)
            dphi = phi_bx[iL] - phi_bx[iR]
            dphi = (dphi + math.pi) % (2.0 * math.pi) - math.pi
            abs_dphi = np.abs(dphi)
            pt_bal_x = px_bx[iL] + px_bx[iR]
            pt_bal_y = py_bx[iL] + py_bx[iR]
            pt_bal = np.hypot(pt_bal_x, pt_bal_y)
            denom_pt = pT_L + pT_R
            pt_bal_ratio = np.zeros(n_pairs, dtype=np.float64)
            np.divide(pt_bal, denom_pt, out=pt_bal_ratio, where=denom_pt > 0.0)
            dphi_from_pi = math.pi - abs_dphi
        else:
            abs_dphi = np.full(n_pairs, np.nan, dtype=np.float64)
            dphi_from_pi = np.full(n_pairs, np.nan, dtype=np.float64)
            pt_bal = np.full(n_pairs, np.nan, dtype=np.float64)
            pt_bal_ratio = np.full(n_pairs, np.nan, dtype=np.float64)

        t200_L = tag200_bx[iL]
        t200_R = tag200_bx[iR]
        t400_L = tag400_bx[iL]
        t400_R = tag400_bx[iR]

        chunk = {
            "bx": np.full(n_pairs, int(bx), dtype=np.int32),
            "interaction_L": interaction_bx[iL].astype(np.int32, copy=False),
            "interaction_R": interaction_bx[iR].astype(np.int32, copy=False),
            "p_idx_L": pidx_bx[iL].astype(np.int32, copy=False),
            "p_idx_R": pidx_bx[iR].astype(np.int32, copy=False),
            "side_L": side_bx[iL].astype(np.int32, copy=False),
            "side_R": side_bx[iR].astype(np.int32, copy=False),
            "tag200_L": t200_L.astype(np.int32, copy=False),
            "tag200_R": t200_R.astype(np.int32, copy=False),
            "tag400_L": t400_L.astype(np.int32, copy=False),
            "tag400_R": t400_R.astype(np.int32, copy=False),
            "double_tag_420": np.ones(n_pairs, dtype=np.int32),
            "vtx_ok": vtx_ok.astype(np.int32, copy=False),
            "in_Hwin": in_Hwin.astype(np.int32, copy=False),
            "xi_L": xi_L.astype(np.float32, copy=False),
            "xi_R": xi_R.astype(np.float32, copy=False),
            "z_L": z_L.astype(np.float32, copy=False),
            "z_R": z_R.astype(np.float32, copy=False),
            "dz": dz.astype(np.float32, copy=False),
            "M": M.astype(np.float32, copy=False),
            "ln_xi_ratio": ln_xi_ratio.astype(np.float32, copy=False),
            "yX": yX.astype(np.float32, copy=False),
            "abs_yX": abs_yX.astype(np.float32, copy=False),
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
        }

        for name in int_branches + float_branches:
            branch_chunks[name].append(chunk[name])

        n_pairs_filled += n_pairs

    arrays = {}
    for name in int_branches:
        if branch_chunks[name]:
            arrays[name] = np.concatenate(branch_chunks[name]).astype(np.int32, copy=False)
        else:
            arrays[name] = np.empty(0, dtype=np.int32)
    for name in float_branches:
        if branch_chunks[name]:
            arrays[name] = np.concatenate(branch_chunks[name]).astype(np.float32, copy=False)
        else:
            arrays[name] = np.empty(0, dtype=np.float32)

    print(f"[uproot] Writing output file: {root_out}", flush=True)
    with uproot.recreate(root_out) as fout:
        fout["ProtonPairs"] = arrays

    print(f"Wrote ROOT file with proton pairs: {root_out}")
    print(f"Total pairs stored in tree: {n_pairs_filled}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)

    parser = argparse.ArgumentParser(
        description="Inspect protons in minbias_protons.npz (single BX or whole-sample summary + ROOT tree)."
    )
    parser.add_argument(
        "-i", "--input",
        default="minbias_protons.npz",
        help="Input .npz file (default: minbias_protons.npz)",
    )
    parser.add_argument(
        "--bx",
        type=int,
        default=0,
        help="Which bunch crossing (bx_id) to inspect in single-BX mode (default: 0)",
    )
    parser.add_argument(
        "--all-bx",
        action="store_true",
        help="If set, run in whole-sample summary mode and build ROOT TTree.",
    )
    parser.add_argument(
        "--root-out",
        default=None,
        help="Output ROOT file name for pair tree (default: <input_base>_pairs.root)",
    )
    args = parser.parse_args()

    # ---- load arrays ----
    print(f"[startup] Loading NPZ: {args.input}", flush=True)
    data = np.load(args.input)
    bx_id          = data["bx_id"]
    interaction_id = data["interaction_id"]
    proton_idx     = data["proton_idx"]
    side           = data["side"]
    pz             = data["pz"]
    pt             = data["pt"]
    xi             = data["xi"]

    # optional px, py (if present in the .npz)
    px = data["px"] if "px" in data.files else None
    py = data["py"] if "py" in data.files else None

    # ---- assign z-vertex per (bx, interaction) ----
    print("[startup] Building (bx, interaction) map...", flush=True)
    keys = np.stack((bx_id, interaction_id), axis=1)   # shape (N_protons, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    rng = np.random.default_rng(12345)
    z_per_interaction_cm = rng.normal(
        loc=0.0, scale=SIGMA_Z_CM, size=uniq.shape[0]
    )
    z_vertex_cm = z_per_interaction_cm[inv]  # one z per proton
    print(f"[startup] Assigned z-vertices for {uniq.shape[0]} interactions", flush=True)

    print("[startup] Grouping proton indices by BX...", flush=True)
    unique_bx, bx_groups = group_indices_by_bx(bx_id)
    print(f"[startup] Prepared BX groups: {unique_bx.size}", flush=True)

    print(f"File: {args.input}")
    print(f"Beam spot sigma_z: {SIGMA_Z_CM:.1f} cm")
    print(f"Vertex match |dz| <= {Z_MATCH_CM:.3f} cm")
    print(f"Higgs window [{M_WIN_LOW:.1f}, {M_WIN_HIGH:.1f}] GeV")
    print(f"HL-LHC rates used: 40 MHz (bunch crossings), 31.6 MHz (avg collisions)")
    if px is None or py is None:
        print("Note: px/py not found in file → Δφ and pT-balance variables in ROOT tree will be NaN.\n")
    else:
        print("px/py found → Δφ and pT-balance variables will be computed.\n")

    if args.all_bx:
        # Summary rates (as before)
        analyze_all_bx(
            bx_id,
            side,
            pt,
            xi,
            z_vertex_cm,
            unique_bx=unique_bx,
            bx_groups=bx_groups,
        )

        # Build ROOT pair tree
        if args.root_out is None:
            base, _ = os.path.splitext(args.input)
            root_out = base + "_pairs.root"
        else:
            root_out = args.root_out

        build_pair_tree_all_bx(
            bx_id, interaction_id, proton_idx,
            side, pt, xi, z_vertex_cm,
            root_out, px=px, py=py,
            unique_bx=unique_bx,
            bx_groups=bx_groups,
        )
    else:
        analyze_single_bx(
            bx_id, interaction_id, proton_idx,
            side, pz, pt, xi, z_vertex_cm, args.bx,
            px=px, py=py
        )

    faulthandler.cancel_dump_traceback_later()

if __name__ == "__main__":
    main()

