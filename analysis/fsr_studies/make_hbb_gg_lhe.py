#!/usr/bin/env python3
"""Rewrite SuperChic CEP evrec events as p p -> (g g -> h0) -> b bbar.

The intact protons are dropped and the Higgs is produced from two incoming gluons,
so Pythia can add ISR, MPI and beam remnants around exactly the same H->bb decay.
The Higgs keeps its SuperChic rapidity; its ~0.5 GeV pT is removed (the gluons are
collinear), so the b's are boosted to the rest frame and then longitudinally to the
Higgs rapidity.  Anti-kt jets use (y, phi, pT), which the longitudinal boost leaves
unchanged, and make_hbb_rest_lhe.py showed the rest-frame rewrite shifts m(jj) < 0.3%.
"""
import argparse
import math
from pathlib import Path

from make_hbb_rest_lhe import boost_to_rest, parse_events

SQRT_S = 14000.0


def boost_z(p, rapidity):
    px, py, pz, e = p
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    return (px, py, ch * pz + sh * e, ch * e + sh * pz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    beam = SQRT_S / 2.0
    out = args.output.open("w")
    out.write('<LesHouchesEvents version="1.0">\n<header>\n')
    out.write("SuperChic CEP H->bb rewritten as g g -> h0 -> b bbar in 14 TeV pp.\n")
    out.write("</header>\n<init>\n")
    out.write(f" 2212 2212 {beam:.9E} {beam:.9E} 0 0 -1 -1 3 1\n")
    out.write(" 0.100000000E+01 0.000000000E+00 0.100000000E+01 1\n")
    out.write("</init>\n")

    written = 0
    events = (event for source in args.sources for event in parse_events(source))
    for header, particles in events:
        higgs = next(p for p in particles if p[0] == "25")
        bs = [p for p in particles if p[0] in ("5", "-5")]
        if len(bs) != 2:
            raise SystemExit("expected exactly one b and one bbar per event")
        ref = tuple(float(higgs[k]) for k in (6, 7, 8, 9))
        mh = float(higgs[10])
        y = 0.5 * math.log((ref[3] + ref[2]) / (ref[3] - ref[2]))
        x1, x2 = mh * math.exp(y) / SQRT_S, mh * math.exp(-y) / SQRT_S
        g1, g2 = x1 * beam, x2 * beam
        e_h, pz_h = mh * math.cosh(y), mh * math.sinh(y)

        out.write("<event>\n")
        out.write(f" 5  1  0.100000000E+01  {header[3]}  {header[4]}  {header[5]}\n")
        out.write(f" 21 -1 0 0 511 512 0.0 0.0 {g1:.9E} {g1:.9E} 0.0 0. 9.\n")
        out.write(f" 21 -1 0 0 512 511 0.0 0.0 {-g2:.9E} {g2:.9E} 0.0 0. 9.\n")
        out.write(f" 25  2 1 2 0 0 0.0 0.0 {pz_h:.9E} {e_h:.9E} {mh:.9E} 0. 9.\n")
        for p in bs:
            rest = boost_to_rest(tuple(float(p[k]) for k in (6, 7, 8, 9)), ref)
            px, py, pz, e = boost_z(rest, y)
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
