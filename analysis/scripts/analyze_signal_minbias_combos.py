#!/usr/bin/env python3
"""
Analyze signal+minbias combo ROOT output.

Reads the `SignalMinbiasCombos` tree produced by build_signal_minbias_combos.py,
constructs all candidate proton pairs with one proton at pz>0 and one at pz<0,
computes candidate mass from xi values,
and reports combo-level passing statistics.
"""

import argparse
import math
import random


# Timing <-> z conversion.
C_CM_PER_PS = 2.99792458e-2


def _safe_fraction(num, den):
    return (float(num) / float(den)) if den else 0.0


def _in_mass_window(mass, center, half_width):
    return (mass >= (center - half_width)) and (mass <= (center + half_width))


def sigma_dz_cm(single_arm_time_res_ps):
    # sigma(dt) = sqrt(2) * sigma_t(arm), and dz = (c/2) * dt
    return 0.5 * C_CM_PER_PS * math.sqrt(2.0) * single_arm_time_res_ps


def _print_count(label, count, n_total):
    print(f"{label}: {count} ({_safe_fraction(count, n_total):.6f})")


def analyze(
    inroot,
    tree_name="SignalMinbiasCombos",
    s_gev2=13600.0**2,
    mass_window=None,
    max_combos=None,
    beam_sigma_z_cm=5.0,
    single_arm_time_res_ps=10.0,
    nsigma=2.0,
    smear_seed=12345,
    smear_dz=True,
    truth_z_smear_cm=0.0,
    verbose=False,
):
    try:
        import uproot
        import awkward as ak
    except ImportError as exc:
        raise RuntimeError(
            "This script requires uproot and awkward. Install with: "
            "python3 -m pip install --user uproot awkward"
        ) from exc

    with uproot.open(inroot) as fin:
        if tree_name not in fin:
            raise RuntimeError(f"Tree '{tree_name}' not found in {inroot}")

        tree = fin[tree_name]
        arrays = tree.arrays(
            [
                "combo_id",
                "signal_xi",
                "signal_pz",
                "signal_tag200",
                "signal_tag400",
                "minbias_xi",
                "minbias_pz",
                "minbias_tag200",
                "minbias_tag400",
            ],
            library="ak",
        )

    combo_ids = arrays["combo_id"]
    n_total = len(combo_ids)
    if max_combos is not None:
        n_total = min(n_total, int(max_combos))

    rng = random.Random(smear_seed)
    timing_sigma_dz_cm = sigma_dz_cm(single_arm_time_res_ps)
    zcut_cm = nsigma * timing_sigma_dz_cm
    zcut_truth_cm = nsigma * math.sqrt(timing_sigma_dz_cm**2 + truth_z_smear_cm**2)

    combos_with_passing_pair = 0
    combos_with_non_signal_pair = 0
    combos_with_signal_only_pair = 0
    combos_with_minbias_only_pair = 0

    combos_with_vertex_pair = 0
    combos_with_non_signal_vertex_pair = 0
    combos_with_signal_only_vertex_pair = 0
    combos_with_minbias_only_vertex_pair = 0

    combos_with_good_z_reco = 0
    combos_with_non_signal_good_z_reco = 0
    combos_with_signal_only_good_z_reco = 0
    combos_with_minbias_only_good_z_reco = 0

    combos_with_good_t_reco = 0
    combos_with_non_signal_good_t_reco = 0
    combos_with_signal_only_good_t_reco = 0
    combos_with_minbias_only_good_t_reco = 0

    combos_with_good_z_and_t_reco = 0
    combos_with_non_signal_good_z_and_t_reco = 0
    combos_with_signal_only_good_z_and_t_reco = 0
    combos_with_minbias_only_good_z_and_t_reco = 0

    combos_with_truth_pair = 0
    combos_with_non_signal_truth_pair = 0
    combos_with_signal_only_truth_pair = 0
    combos_with_minbias_only_truth_pair = 0

    for i in range(n_total):
        cid = int(combo_ids[i])

        sig_xi = arrays["signal_xi"][i]
        sig_pz = arrays["signal_pz"][i]
        sig_tag200 = arrays["signal_tag200"][i]
        sig_tag400 = arrays["signal_tag400"][i]

        mb_xi = arrays["minbias_xi"][i]
        mb_pz = arrays["minbias_pz"][i]
        mb_tag200 = arrays["minbias_tag200"][i]
        mb_tag400 = arrays["minbias_tag400"][i]

        protons = []
        signal_z = float(rng.gauss(0.0, beam_sigma_z_cm))
        signal_t = float(rng.gauss(0.0, beam_sigma_z_cm / C_CM_PER_PS))
        if truth_z_smear_cm > 0.0:
            truth_z = signal_z + float(rng.gauss(0.0, truth_z_smear_cm))
        else:
            truth_z = signal_z

        for j in range(len(sig_xi)):
            tagged = bool(int(sig_tag200[j]) or int(sig_tag400[j]))
            tagged_400 = bool(int(sig_tag400[j]))
            if not tagged:
                continue
            protons.append(
                {
                    "source": "signal",
                    "xi": float(sig_xi[j]),
                    "pz": float(sig_pz[j]),
                    "z": signal_z,
                    "t": signal_t,
                    "tagged_400": tagged_400,
                    "tagged": tagged,
                }
            )

        for j in range(len(mb_xi)):
            tagged = bool(int(mb_tag200[j]) or int(mb_tag400[j]))
            tagged_400 = bool(int(mb_tag400[j]))
            if not tagged:
                continue
            minbias_z = float(rng.gauss(0.0, beam_sigma_z_cm))
            minbias_t = float(rng.gauss(0.0, beam_sigma_z_cm / C_CM_PER_PS))
            protons.append(
                {
                    "source": "minbias",
                    "xi": float(mb_xi[j]),
                    "pz": float(mb_pz[j]),
                    "z": minbias_z,
                    "t": minbias_t,
                    "tagged_400": tagged_400,
                    "tagged": tagged,
                }
            )

        has_any_passing = False
        has_non_signal_passing = False
        has_signal_only_passing = False
        has_minbias_only_passing = False

        has_any_vertex = False
        has_non_signal_vertex = False
        has_signal_only_vertex = False
        has_minbias_only_vertex = False

        has_good_z_reco = False
        has_non_signal_good_z_reco = False
        has_signal_only_good_z_reco = False
        has_minbias_only_good_z_reco = False

        has_good_t_reco = False
        has_non_signal_good_t_reco = False
        has_signal_only_good_t_reco = False
        has_minbias_only_good_t_reco = False

        has_good_z_and_t_reco = False
        has_non_signal_good_z_and_t_reco = False
        has_signal_only_good_z_and_t_reco = False
        has_minbias_only_good_z_and_t_reco = False

        has_any_truth = False
        has_non_signal_truth = False
        has_signal_only_truth = False
        has_minbias_only_truth = False

        n_prot = len(protons)
        for a in range(n_prot):
            pa = protons[a]
            for b in range(a + 1, n_prot):
                pb = protons[b]

                # Require at least one proton tagged in 400m
                if not (pa["tagged_400"] or pb["tagged_400"]):
                    continue

                if not (pa["tagged"] and pb["tagged"]):
                    continue

                # Require one proton at +pz and one at -pz
                if not ((pa["pz"] > 0.0 and pb["pz"] < 0.0) or (pa["pz"] < 0.0 and pb["pz"] > 0.0)):
                    continue

                xi_prod = pa["xi"] * pb["xi"]
                if xi_prod <= 0.0:
                    continue

                mass = math.sqrt(xi_prod * s_gev2)

                if mass_window is not None and not _in_mass_window(mass, 125.0, mass_window):
                    continue

                has_any_passing = True
                if pa["source"] == "signal" and pb["source"] == "signal":
                    has_signal_only_passing = True
                elif pa["source"] == "minbias" and pb["source"] == "minbias":
                    has_minbias_only_passing = True
                    has_non_signal_passing = True
                else:
                    has_non_signal_passing = True

                dz = pa["z"] - pb["z"]
                if smear_dz:
                    dz_obs = dz + float(rng.gauss(0.0, timing_sigma_dz_cm))
                else:
                    dz_obs = dz
                vertex_compatible = abs(dz_obs) <= zcut_cm

                # pair_z_mid = 0.5 * (pa["z"] + pb["z"])
                # if smear_dz:
                #     pair_z_mid_obs = pair_z_mid + float(rng.gauss(0.0, timing_sigma_dz_cm))
                # else:
                #     pair_z_mid_obs = pair_z_mid
                # truth_compatible = abs(pair_z_mid_obs - truth_z) <= zcut_truth_cm

                p_left = pa if pa["pz"] < 0.0 else pb
                p_right = pb if pa["pz"] < 0.0 else pa
                t_left = p_left["t"] + (p_left["z"] + 420*100) / C_CM_PER_PS
                t_right = p_right["t"] + (-p_right["z"] + 420*100) / C_CM_PER_PS
                if smear_dz:
                    t_left += float(rng.gauss(0.0, single_arm_time_res_ps))
                    t_right += float(rng.gauss(0.0, single_arm_time_res_ps))

                z_reco = 0.5 * (t_left - t_right) * C_CM_PER_PS
                t_reco = 0.5 * (t_left + t_right) - 420*100 / C_CM_PER_PS
                z_good = abs(z_reco) <= nsigma * beam_sigma_z_cm
                t_good = abs(t_reco) <= nsigma * beam_sigma_z_cm / C_CM_PER_PS
                if z_good and t_good:
                    has_good_z_and_t_reco = True
                    if pa["source"] == "signal" and pb["source"] == "signal":
                        has_signal_only_good_z_and_t_reco = True
                    elif pa["source"] == "minbias" and pb["source"] == "minbias":
                        has_minbias_only_good_z_and_t_reco = True
                        has_non_signal_good_z_and_t_reco = True
                    else:
                        has_non_signal_good_z_and_t_reco = True
                if z_good:
                    has_good_z_reco = True
                    if pa["source"] == "signal" and pb["source"] == "signal":
                        has_signal_only_good_z_reco = True
                    elif pa["source"] == "minbias" and pb["source"] == "minbias":
                        has_minbias_only_good_z_reco = True
                        has_non_signal_good_z_reco = True
                    else:
                        has_non_signal_good_z_reco = True
                if t_good:
                    has_good_t_reco = True
                    if pa["source"] == "signal" and pb["source"] == "signal":
                        has_signal_only_good_t_reco = True
                    elif pa["source"] == "minbias" and pb["source"] == "minbias":
                        has_minbias_only_good_t_reco = True
                        has_non_signal_good_t_reco = True
                    else:
                        has_non_signal_good_t_reco = True

                truth_compatible = abs(z_reco - truth_z) <= zcut_truth_cm

                # z_reco = 0.5 * (pa["z"] + pb["z"]) + 0.5 * (pa["t"] - pb["t"]) * C_CM_PER_PS
                # if smear_dz:
                #     z_reco += float(rng.gauss(0.0, timing_sigma_dz_cm))
                # truth_compatible = abs(z_reco - truth_z) <= zcut_truth_cm

                #t_reco = 0.5 * (pa["t"] + pb["t"]) + 0.5 * (pa["z"] - pb["z"]) / C_CM_PER_PS
                #if smear_dz:
                #    t_reco += float(rng.gauss(0.0, timing_sigma_dt_ps))

                if vertex_compatible:
                    has_any_vertex = True
                    if pa["source"] == "signal" and pb["source"] == "signal":
                        has_signal_only_vertex = True
                    elif pa["source"] == "minbias" and pb["source"] == "minbias":
                        has_minbias_only_vertex = True
                        has_non_signal_vertex = True
                    else:
                        has_non_signal_vertex = True

                if truth_compatible:
                    has_any_truth = True
                    if pa["source"] == "signal" and pb["source"] == "signal":
                        has_signal_only_truth = True
                    elif pa["source"] == "minbias" and pb["source"] == "minbias":
                        has_minbias_only_truth = True
                        has_non_signal_truth = True
                    else:
                        has_non_signal_truth = True

                if verbose:
                    print(
                        f"  bx={cid}, protons={a}({pa['source']}), {b}({pb['source']}), "
                        f"z=({pa['z']:.2f}, {pb['z']:.2f}), "
                        f"t=({pa['t']:.2f}, {pb['t']:.2f}), "
                        #f"t_left={t_left:.2f} ps, t_right={t_right:.2f} ps, "
                        f"z_reco={z_reco:.2f} cm, "
                        f"t_reco={t_reco:.2f} ps, "
                        f"tagged_400=({pa['tagged_400']}, {pb['tagged_400']}), "
                        f"mass={mass:.2f} GeV, "
                        f"vertex_compatible={vertex_compatible}, "
                        f"truth_compatible={truth_compatible}"
                        f"z_reco_good={z_good}, t_reco_good={t_good}, z_and_t_reco_good={z_good and t_good}"
                    )

        if has_any_passing:
            combos_with_passing_pair += 1
        if has_non_signal_passing:
            combos_with_non_signal_pair += 1
        if has_signal_only_passing:
            combos_with_signal_only_pair += 1
        if has_minbias_only_passing:
            combos_with_minbias_only_pair += 1

        if has_any_vertex:
            combos_with_vertex_pair += 1
        if has_non_signal_vertex:
            combos_with_non_signal_vertex_pair += 1
        if has_signal_only_vertex:
            combos_with_signal_only_vertex_pair += 1
        if has_minbias_only_vertex:
            combos_with_minbias_only_vertex_pair += 1

        if has_good_z_reco:
            combos_with_good_z_reco += 1
        if has_non_signal_good_z_reco:
            combos_with_non_signal_good_z_reco += 1
        if has_signal_only_good_z_reco:
            combos_with_signal_only_good_z_reco += 1
        if has_minbias_only_good_z_reco:
            combos_with_minbias_only_good_z_reco += 1
        if has_good_t_reco:
            combos_with_good_t_reco += 1
        if has_non_signal_good_t_reco:
            combos_with_non_signal_good_t_reco += 1
        if has_signal_only_good_t_reco:
            combos_with_signal_only_good_t_reco += 1
        if has_minbias_only_good_t_reco:
            combos_with_minbias_only_good_t_reco += 1
        if has_good_z_and_t_reco:
            combos_with_good_z_and_t_reco += 1
        if has_non_signal_good_z_and_t_reco:
            combos_with_non_signal_good_z_and_t_reco += 1
        if has_signal_only_good_z_and_t_reco:
            combos_with_signal_only_good_z_and_t_reco += 1
        if has_minbias_only_good_z_and_t_reco:
            combos_with_minbias_only_good_z_and_t_reco += 1

        if has_any_truth:
            combos_with_truth_pair += 1
        if has_non_signal_truth:
            combos_with_non_signal_truth_pair += 1
        if has_signal_only_truth:
            combos_with_signal_only_truth_pair += 1
        if has_minbias_only_truth:
            combos_with_minbias_only_truth_pair += 1

        if verbose:
            print(
                f"combo_id={cid}: n_tagged_protons={len(protons)} "
                f"pass_any={int(has_any_passing)} "
                f"pass_non_signal={int(has_non_signal_passing)} "
                f"pass_signal_only={int(has_signal_only_passing)} "
                f"pass_minbias_only={int(has_minbias_only_passing)} "
                f"pass_vertex_any={int(has_any_vertex)} "
                f"pass_truth_any={int(has_any_truth)}"
                f"good_z_reco_any={int(has_good_z_reco)} "
                f"good_t_reco_any={int(has_good_t_reco)} "
                f"good_z_and_t_reco_any={int(has_good_z_and_t_reco)} "
                #f"masses={[math.sqrt(p['xi'] * s_gev2) for p in protons]}"
            )

    print("=== Signal+MinBias Combo Analysis ===")
    print(f"Input ROOT: {inroot}")
    print(f"Tree: {tree_name}")
    print(f"Combos processed: {n_total}")
    print(f"s used in mass formula: {s_gev2}")
    if mass_window is not None:
        print(f"Mass window: [{125.0 - mass_window}, {125.0 + mass_window}] GeV")
    else:
        print("Mass window: disabled")
    print("")

    print("=== Vertex Model ===")
    print(f"Beam sigma z: {beam_sigma_z_cm} cm")
    print(f"Single-arm timing resolution: {single_arm_time_res_ps} ps")
    print(f"sigma_dz: {timing_sigma_dz_cm:.6f} cm")
    print(f"nsigma: {nsigma}")
    print(f"z cut: {zcut_cm:.6f} cm")
    print(f"dz smearing enabled: {smear_dz}")
    print(f"Truth z smear: {truth_z_smear_cm} cm")
    print("")

    _print_count("Combos with >=1 passing pair", combos_with_passing_pair, n_total)
    _print_count(
        "Combos with >=1 passing pair and >=1 non-signal proton in the pair",
        combos_with_non_signal_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing pair where both protons are signal",
        combos_with_signal_only_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing pair where both protons are minbias",
        combos_with_minbias_only_pair,
        n_total,
    )
    print("")

    _print_count(
        "Combos with >=1 passing pair and vertex-compatible pair",
        combos_with_vertex_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing non-signal pair and vertex-compatible pair",
        combos_with_non_signal_vertex_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing signal-only pair and vertex-compatible pair",
        combos_with_signal_only_vertex_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing minbias-only pair and vertex-compatible pair",
        combos_with_minbias_only_vertex_pair,
        n_total,
    )
    print("")

    _print_count(
        "Combos with >=1 passing pair and good z reco",
        combos_with_good_z_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing pair and good t reco",
        combos_with_good_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing pair and good z and t reco",
        combos_with_good_z_and_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing non-signal pair and good z reco",
        combos_with_non_signal_good_z_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing non-signal pair and good t reco",
        combos_with_non_signal_good_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing non-signal pair and good z and t reco",
        combos_with_non_signal_good_z_and_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing signal-only pair and good z reco",
        combos_with_signal_only_good_z_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing signal-only pair and good t reco",
        combos_with_signal_only_good_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing signal-only pair and good z and t reco",
        combos_with_signal_only_good_z_and_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing minbias-only pair and good z reco",
        combos_with_minbias_only_good_z_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing minbias-only pair and good t reco",
        combos_with_minbias_only_good_t_reco,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing minbias-only pair and good z and t reco",
        combos_with_minbias_only_good_z_and_t_reco,
        n_total,
    )
    print("")

    _print_count(
        "Combos with >=1 passing pair and truth-compatible pair midpoint",
        combos_with_truth_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing non-signal pair and truth-compatible pair midpoint",
        combos_with_non_signal_truth_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing signal-only pair and truth-compatible pair midpoint",
        combos_with_signal_only_truth_pair,
        n_total,
    )
    _print_count(
        "Combos with >=1 passing minbias-only pair and truth-compatible pair midpoint",
        combos_with_minbias_only_truth_pair,
        n_total,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze signal+minbias combo ROOT output.")
    parser.add_argument("-i", "--inroot", required=True, help="Input ROOT file from build_signal_minbias_combos.py")
    parser.add_argument("--tree", default="SignalMinbiasCombos", help="Tree name (default: SignalMinbiasCombos)")
    parser.add_argument("--s", type=float, default=13600.0**2, help="s value for mass formula M=sqrt(xi1*xi2*s)")
    parser.add_argument("--mass-window", type=float, default=10, help="Optional mass half-width in GeV around 125")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional max number of combos to process")
    parser.add_argument("--beam-sigma-z-cm", type=float, default=5.7, help="Gaussian beam spot sigma z in cm")
    parser.add_argument("--single-arm-time-res-ps", type=float, default=3.0, help="Single-arm timing resolution in ps")
    parser.add_argument("--nsigma", type=float, default=2.0, help="Timing compatibility cut in units of sigma_dz")
    parser.add_argument("--smear-seed", type=int, default=12345, help="Seed for vertex and dz smearing RNG")
    parser.add_argument("--no-dz-smear", action="store_true", help="Disable Gaussian measurement smearing on dz")
    parser.add_argument("--truth-z-smear-cm", type=float, default=0.05, help="Optional Gaussian truth-vertex z smear in cm")
    parser.add_argument("--verbose", action="store_true", help="Print per-combo pass flags")
    args = parser.parse_args()

    if args.mass_window is not None and args.mass_window < 0.0:
        raise RuntimeError("--mass-window must be >= 0")
    if args.beam_sigma_z_cm < 0.0:
        raise RuntimeError("--beam-sigma-z-cm must be >= 0")
    if args.single_arm_time_res_ps <= 0.0:
        raise RuntimeError("--single-arm-time-res-ps must be > 0")
    if args.nsigma <= 0.0:
        raise RuntimeError("--nsigma must be > 0")
    if args.truth_z_smear_cm < 0.0:
        raise RuntimeError("--truth-z-smear-cm must be >= 0")

    analyze(
        inroot=args.inroot,
        tree_name=args.tree,
        s_gev2=args.s,
        mass_window=args.mass_window,
        max_combos=args.max_combos,
        beam_sigma_z_cm=args.beam_sigma_z_cm,
        single_arm_time_res_ps=args.single_arm_time_res_ps,
        nsigma=args.nsigma,
        smear_seed=args.smear_seed,
        smear_dz=not args.no_dz_smear,
        truth_z_smear_cm=args.truth_z_smear_cm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
