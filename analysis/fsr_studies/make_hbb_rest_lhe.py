#!/usr/bin/env python3
"""Rewrite SuperChic CEP evrec events as e+e- -> h0 -> b bbar in the Higgs rest frame.

The intact protons are spectators for the jets, so they are dropped and the event is
boosted to the Higgs rest frame.  Anti-kt clustering uses (y, phi) and pT, all of which
are invariant under the longitudinal boost, so the jet structure is unchanged.
The colour flow, the b/bbar directions and the 125 GeV shower start scale are kept.
"""
import argparse
import math
from pathlib import Path


def boost_to_rest(p, ref):
    """Boost four-vector p = (px,py,pz,E) into the rest frame of ref."""
    rx, ry, rz, re = ref
    m = math.sqrt(max(re * re - rx * rx - ry * ry - rz * rz, 0.0))
    bx, by, bz = -rx / re, -ry / re, -rz / re
    b2 = bx * bx + by * by + bz * bz
    gamma = re / m
    px, py, pz, e = p
    bp = bx * px + by * py + bz * pz
    factor = (gamma - 1.0) / b2 * bp + gamma * e
    return (px + factor * bx, py + factor * by, pz + factor * bz, gamma * (e + bp))


def parse_events(path):
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "<event>":
            j = i + 1
            while lines[j].strip() != "</event>":
                j += 1
            header = lines[i + 1].split()
            particles = [line.split() for line in lines[i + 2:j]]
            yield header, particles
            i = j
        i += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    out = args.output.open("w")
    out.write('<LesHouchesEvents version="1.0">\n<header>\n')
    out.write("SuperChic CEP H->bb rewritten to the Higgs rest frame with e+e- beams.\n")
    out.write("</header>\n<init>\n")
    out.write(" -11 11 0.625000000E+02 0.625000000E+02 0 0 -1 -1 3 1\n")
    out.write(" 0.100000000E+01 0.000000000E+00 0.100000000E+01 1\n")
    out.write("</init>\n")

    written = 0
    for header, particles in parse_events(args.source):
        higgs = next(p for p in particles if p[0] == "25")
        bs = [p for p in particles if p[0] in ("5", "-5")]
        if len(bs) != 2:
            raise SystemExit("expected exactly one b and one bbar per event")
        ref = tuple(float(higgs[k]) for k in (6, 7, 8, 9))
        mh = float(higgs[10])
        scale = header[3]

        out.write("<event>\n")
        out.write(f" 5  1  0.100000000E+01  {scale}  {header[4]}  {header[5]}\n")
        half = mh / 2.0
        out.write(f" -11 -1 0 0 0 0 0.0 0.0 {half:.9E} {half:.9E} 0.0 0. 9.\n")
        out.write(f"  11 -1 0 0 0 0 0.0 0.0 {-half:.9E} {half:.9E} 0.0 0. 9.\n")
        out.write(f"  25  2 1 2 0 0 0.0 0.0 0.0 {mh:.9E} {mh:.9E} 0. 9.\n")
        for p in bs:
            px, py, pz, e = boost_to_rest(tuple(float(p[k]) for k in (6, 7, 8, 9)), ref)
            out.write(
                f" {p[0]:>3} 1 3 3 {p[4]} {p[5]} {px:.9E} {py:.9E} {pz:.9E} {e:.9E} {p[10]} 0. 9.\n"
            )
        out.write("</event>\n")
        written += 1
        if args.max_events and written >= args.max_events:
            break

    out.write("</LesHouchesEvents>\n")
    out.close()
    print(f"wrote {written} events to {args.output}")


if __name__ == "__main__":
    main()
