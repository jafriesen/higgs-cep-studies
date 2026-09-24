#!/usr/bin/env python3
"""Summarize a parton-level FPMC inclusive-DPE LHE pilot."""

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PPS_XI_RANGES = (
    (0.00325, 0.0116),
    (0.0140, 0.0263),
    (0.0375, 0.0688),
    (0.0800, 0.1967),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lhe", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def read_cross_section(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Cross section\[pb\]=\s*([0-9.Ee+-]+)", text)
    if not matches:
        raise RuntimeError(f"No final cross section in {log_path}")
    return float(matches[-1])


def events(path):
    block = []
    inside = False
    for raw in path.open(encoding="utf-8"):
        line = raw.strip()
        if line == "<event>":
            inside = True
            block = []
        elif line == "</event>":
            header = block[0].split()
            particles = []
            for particle_line in block[1 : 1 + int(header[0])]:
                fields = particle_line.split()
                particles.append(
                    {
                        "pid": int(fields[0]),
                        "status": int(fields[1]),
                        "px": float(fields[6]),
                        "py": float(fields[7]),
                        "pz": float(fields[8]),
                        "energy": float(fields[9]),
                    }
                )
            yield particles
            inside = False
        elif inside and line:
            block.append(line)


def flavor(partons):
    ids = sorted(abs(p["pid"]) for p in partons)
    if ids == [5, 5]:
        return "bb"
    if ids == [4, 4]:
        return "cc"
    if ids == [21, 21]:
        return "gg"
    if 21 in ids and any(pid in (1, 2, 3) for pid in ids):
        return "light_qg"
    if all(pid in (1, 2, 3) for pid in ids):
        return "light_qq"
    if 5 in ids:
        return "single_b_or_mixed_b"
    if 4 in ids:
        return "single_c_or_mixed_c"
    return "other"


def accepted_xi(xi):
    return any(low <= xi < high for low, high in PPS_XI_RANGES)


def eta(particle):
    momentum = math.sqrt(
        particle["px"] ** 2 + particle["py"] ** 2 + particle["pz"] ** 2
    )
    return math.atanh(particle["pz"] / momentum)


def invariant_mass(particles):
    energy = sum(p["energy"] for p in particles)
    px = sum(p["px"] for p in particles)
    py = sum(p["py"] for p in particles)
    pz = sum(p["pz"] for p in particles)
    return math.sqrt(max(0.0, energy**2 - px**2 - py**2 - pz**2))


def main():
    args = parse_args()
    cross_section_pb = read_cross_section(args.log)
    counts = defaultdict(Counter)

    for particles in events(args.lhe):
        protons = [p for p in particles if p["pid"] == 2212]
        partons = [p for p in particles if abs(p["pid"]) <= 6 or p["pid"] == 21]
        if len(protons) != 2 or len(partons) != 2:
            counts["malformed"]["all"] += 1
            continue

        category = flavor(partons)
        counts["generated"][category] += 1
        xis = [1.0 - abs(p["pz"]) / 7000.0 for p in protons]
        pps = all(accepted_xi(xi) for xi in xis)
        mpp = 14000.0 * math.sqrt(max(0.0, xis[0] * xis[1]))
        central = all(
            math.hypot(p["px"], p["py"]) >= 15.0 and abs(eta(p)) < 3.0
            for p in partons
        )
        mjj = invariant_mass(partons)

        selections = {
            "central_partons": central,
            "pps": pps,
            "pps_and_central": pps and central,
            "pps_mpp_110_140": pps and 110.0 <= mpp <= 140.0,
            "pps_mpp_117_133": pps and 117.0 <= mpp <= 133.0,
            "analysis_genlevel": (
                pps
                and 117.0 <= mpp <= 133.0
                and central
                and 50.0 <= mjj <= 150.0
            ),
        }
        for name, passed in selections.items():
            if passed:
                counts[name][category] += 1

    total = sum(counts["generated"].values())
    print(f"events: {total}")
    print(f"fpmc_cross_section_pb: {cross_section_pb:.6g}")
    print("PPS xi ranges: " + ", ".join(f"[{a}, {b})" for a, b in PPS_XI_RANGES))
    for selection, selected in counts.items():
        if selection == "malformed":
            continue
        print(f"\n{selection}: {sum(selected.values())}/{total}")
        for category in sorted(counts["generated"]):
            number = selected[category]
            fraction = number / total
            uncertainty = math.sqrt(number) / total if number else 0.0
            print(
                f"  {category:20s} {number:7d}  fraction={fraction:.6g}"
                f" +/- {uncertainty:.2g}  xsec_pb={cross_section_pb * fraction:.6g}"
            )


if __name__ == "__main__":
    main()
