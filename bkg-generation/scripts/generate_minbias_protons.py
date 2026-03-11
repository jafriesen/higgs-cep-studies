#!/usr/bin/env python3
"""
Generate minimum-bias pp events with Pythia8 and store all final-state protons.

Concept:
  - Use SoftQCD (minimum bias) at a given sqrt(s) (default 14 TeV).
  - Group `mu` interactions into one "bunch crossing" (BX).
  - For each BX and each interaction, store *all* final-state protons.
  - Save a flat table (one row per proton) into a compressed NumPy .npz file.

Stored columns (arrays of the same length):
  - bx_id          : int, which bunch crossing
  - interaction_id : int, which interaction within that BX (0 .. mu-1)
  - proton_idx     : int, index of proton within that interaction
  - side           : int, +1 if pz > 0 (right), -1 if pz < 0 (left)
  - px, py, pz, E  : float, 4-momentum in GeV
  - m              : float, proton mass as given by Pythia (should be ~0.938)
  - pt             : float, transverse momentum in GeV
  - xi             : float, fractional momentum loss = 1 - |pz| / E_beam

Later, you can:
  - apply any (xi_min, xi_max, pT_max) acceptance,
  - group by bx_id and count protons on each side,
  - compute double-tag probabilities / trigger rates.
"""

import argparse
import math
import numpy as np
import pythia8mc as pythia8


def configure_pythia(e_cm=14000.0, processes="SoftQCD:all"):
    """
    Configure and initialize a Pythia instance for minimum-bias.

    Parameters
    ----------
    e_cm : float
        Center-of-mass energy in GeV (default: 14000 = HL-LHC).
    processes : str
        Which SoftQCD processes to turn on (default: "SoftQCD:all").

    Returns
    -------
    pythia : pythia8.Pythia
    """
    pythia = pythia8.Pythia()

    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString(f"Beams:eCM = {e_cm}")

    # Turn off everything, then turn on the requested SoftQCD mode
    pythia.readString("SoftQCD:nonDiffractive      = off")
    pythia.readString("SoftQCD:elastic             = off")
    pythia.readString("SoftQCD:singleDiffractive   = off")
    pythia.readString("SoftQCD:doubleDiffractive   = off")
    pythia.readString("SoftQCD:centralDiffractive  = off")

    pythia.readString(f"{processes} = on")

    # Optional tune if you want:
    # pythia.readString("Tune:pp = 14")  # Monash

    pythia.init()
    return pythia


def generate_protons(n_bx, mu, e_cm=14000.0, processes="SoftQCD:all"):
    """
    Generate min-bias events and collect all final-state protons.

    Parameters
    ----------
    n_bx : int
        Number of bunch crossings to simulate.
    mu : int
        Number of interactions per bunch crossing.
    e_cm : float
        Center-of-mass energy in GeV.
    processes : str
        SoftQCD processes string, e.g. "SoftQCD:all".

    Returns
    -------
    data : dict of str -> np.ndarray
        Flat table of proton-level info, see module docstring.
    """
    pythia = configure_pythia(e_cm=e_cm, processes=processes)
    e_beam = e_cm / 2.0

    bx_ids = []
    int_ids = []
    proton_idxs = []
    sides = []
    pxs = []
    pys = []
    pzs = []
    Es = []
    ms = []
    pts = []
    xis = []

    global_event_counter = 0  # if you ever want a unique event ID

    for bx_id in range(n_bx):
        for interaction_id in range(mu):
            if not pythia.next():
                # If Pythia fails, just retry this interaction index
                # (this is rare, but can happen)
                interaction_id -= 1
                continue

            event = pythia.event
            proton_index_in_event = 0

            for i in range(event.size()):
                p = event[i]
                if not p.isFinal():
                    continue
                if p.id() != 2212:  # proton PID
                    continue

                px = p.px()
                py = p.py()
                pz = p.pz()
                E = p.e()
                m = p.m()
                pt = p.pT()
                # fractional momentum loss, using |pz|
                xi = 1.0 - abs(pz) / e_beam

                side = 1 if pz > 0 else -1  # right vs left

                bx_ids.append(bx_id)
                int_ids.append(interaction_id)
                proton_idxs.append(proton_index_in_event)
                sides.append(side)
                pxs.append(px)
                pys.append(py)
                pzs.append(pz)
                Es.append(E)
                ms.append(m)
                pts.append(pt)
                xis.append(xi)

                proton_index_in_event += 1

            global_event_counter += 1

    data = {
        "bx_id":          np.array(bx_ids, dtype=np.int32),
        "interaction_id": np.array(int_ids, dtype=np.int32),
        "proton_idx":     np.array(proton_idxs, dtype=np.int32),
        "side":           np.array(sides, dtype=np.int8),
        "px":             np.array(pxs, dtype=np.float32),
        "py":             np.array(pys, dtype=np.float32),
        "pz":             np.array(pzs, dtype=np.float32),
        "E":              np.array(Es, dtype=np.float32),
        "m":              np.array(ms, dtype=np.float32),
        "pt":             np.array(pts, dtype=np.float32),
        "xi":             np.array(xis, dtype=np.float32),
    }

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate min-bias protons with Pythia8 and store in .npz"
    )
    parser.add_argument(
        "--n-bx",
        type=int,
        default=1000,
        help="Number of bunch crossings to simulate (default: 1000)",
    )
    parser.add_argument(
        "--mu",
        type=int,
        default=200,
        help="Number of interactions per bunch crossing (default: 200)",
    )
    parser.add_argument(
        "--e-cm",
        type=float,
        default=14000.0,
        help="Center-of-mass energy in GeV (default: 14000)",
    )
    parser.add_argument(
        "--processes",
        type=str,
        default="SoftQCD:all",
        help='SoftQCD processes to enable, e.g. "SoftQCD:all" (default)',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="minbias_protons.npz",
        help="Output .npz file (default: minbias_protons.npz)",
    )

    args = parser.parse_args()

    print(
        f"Generating min-bias protons: "
        f"n_bx={args.n_bx}, mu={args.mu}, "
        f"sqrt(s)={args.e_cm} GeV, processes={args.processes}"
    )

    data = generate_protons(
        n_bx=args.n_bx,
        mu=args.mu,
        e_cm=args.e_cm,
        processes=args.processes,
    )

    print(f"Number of protons stored: {len(data['px'])}")
    print(f"Saving to {args.output}")
    np.savez_compressed(args.output, **data)
    print("Done.")


if __name__ == "__main__":
    main()

