#!/usr/bin/env python3
"""
Generate minimum-bias pp events with Pythia8 and store all final-state protons.

Concept:
  - Use SoftQCD (minimum bias) at a given sqrt(s) (default 14 TeV).
  - Group `mu` interactions into one "bunch crossing" (BX), either fixed
    per BX or sampled per BX from a Poisson distribution with mean `mu`.
  - For each BX and each interaction, store *all* final-state protons.
  - Save a flat table (one row per proton) into a compressed NumPy .npz file.

Stored columns (arrays of the same length):
  - bx_id          : int, which bunch crossing
  - interaction_id : int, which interaction within that BX
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


def configure_pythia(e_cm=14000.0, processes="SoftQCD:all", seed=None):
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

    if seed is not None:
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")

    pythia.init()
    return pythia


def generate_protons(
    n_bx,
    mu,
    mu_mode="fixed",
    e_cm=14000.0,
    processes="SoftQCD:all",
    seed=None,
    bx_offset=0,
    store_tracks=False,
):
    """
    Generate min-bias events and collect all final-state protons.

    Parameters
    ----------
    n_bx : int
        Number of bunch crossings to simulate.
    mu : int
        Fixed number of interactions per bunch crossing, or Poisson mean
        when mu_mode="poisson".
    mu_mode : str
        "fixed" for exactly mu interactions per BX, or "poisson" to sample
        interactions per BX from Poisson(mu).
    e_cm : float
        Center-of-mass energy in GeV.
    processes : str
        SoftQCD processes string, e.g. "SoftQCD:all".

    Returns
    -------
    data : dict of str -> np.ndarray
        Flat table of proton-level info, see module docstring.
    """
    pythia = configure_pythia(e_cm=e_cm, processes=processes, seed=seed)
    e_beam = e_cm / 2.0
    rng = np.random.default_rng(seed)

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
    mu_per_bx = np.empty(n_bx, dtype=np.int32)

    if store_tracks:
        trk_bx_ids = []
        trk_int_ids = []
        trk_idxs = []
        trk_pdg_ids = []
        trk_charges = []
        trk_pxs = []
        trk_pys = []
        trk_pzs = []
        trk_Es = []
        trk_pts = []
        trk_etas = []

    global_event_counter = 0  # if you ever want a unique event ID

    for bx_local in range(n_bx):
        bx_id = bx_offset + bx_local
        if mu_mode == "fixed":
            n_interactions = mu
        elif mu_mode == "poisson":
            n_interactions = int(rng.poisson(mu))
        else:
            raise ValueError(f"Unsupported mu_mode: {mu_mode}")
        mu_per_bx[bx_local] = n_interactions

        interaction_id = 0
        while interaction_id < n_interactions:
            if not pythia.next():
                # If Pythia fails, just retry this interaction index.
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

            if store_tracks:
                track_index_in_event = 0

                for i in range(event.size()):
                    p = event[i]
                    if not p.isFinal():
                        continue
                    charge = p.charge()
                    if charge == 0:
                        continue
                    if abs(p.id()) == 2212:
                        continue
                    eta = p.eta()
                    if abs(eta) >= 2.5:
                        continue
                    pt = p.pT()
                    if pt <= 0.5:
                        continue

                    trk_bx_ids.append(bx_id)
                    trk_int_ids.append(interaction_id)
                    trk_idxs.append(track_index_in_event)
                    trk_pdg_ids.append(p.id())
                    trk_charges.append(int(round(charge)))
                    trk_pxs.append(p.px())
                    trk_pys.append(p.py())
                    trk_pzs.append(p.pz())
                    trk_Es.append(p.e())
                    trk_pts.append(pt)
                    trk_etas.append(eta)

                    track_index_in_event += 1

            global_event_counter += 1
            interaction_id += 1

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
        "mu_per_bx":      mu_per_bx,
        "bx_offset":      np.array(bx_offset, dtype=np.int32),
        "n_bx":           np.array(n_bx, dtype=np.int32),
        "mu_mean":        np.array(mu, dtype=np.float32),
        "mu_mode":        np.array(mu_mode),
    }

    if store_tracks:
        data.update({
            "trk_bx_id":          np.array(trk_bx_ids, dtype=np.int32),
            "trk_interaction_id": np.array(trk_int_ids, dtype=np.int32),
            "trk_idx":            np.array(trk_idxs, dtype=np.int32),
            "trk_pdg_id":         np.array(trk_pdg_ids, dtype=np.int32),
            "trk_charge":         np.array(trk_charges, dtype=np.int32),
            "trk_px":             np.array(trk_pxs, dtype=np.float32),
            "trk_py":             np.array(trk_pys, dtype=np.float32),
            "trk_pz":             np.array(trk_pzs, dtype=np.float32),
            "trk_E":              np.array(trk_Es, dtype=np.float32),
            "trk_pt":             np.array(trk_pts, dtype=np.float32),
            "trk_eta":            np.array(trk_etas, dtype=np.float32),
        })

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
        help="Interactions per BX for fixed mode, or Poisson mean for poisson mode (default: 200)",
    )
    parser.add_argument(
        "--mu-mode",
        choices=("fixed", "poisson"),
        default="fixed",
        help='How to choose interactions per BX: "fixed" or "poisson" (default: fixed)',
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Pythia random seed for reproducibility",
    )
    parser.add_argument(
        "--bx-offset",
        type=int,
        default=0,
        help="Global BX offset applied to local BX IDs (default: 0)",
    )
    parser.add_argument(
        "--store-tracks",
        action="store_true",
        help="Also store selected central charged final-state particles",
    )

    args = parser.parse_args()

    if args.n_bx <= 0:
        raise ValueError("--n-bx must be > 0")
    if args.mu <= 0:
        raise ValueError("--mu must be > 0")
    if args.bx_offset < 0:
        raise ValueError("--bx-offset must be >= 0")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be >= 0")

    print(
        f"Generating min-bias protons: "
        f"n_bx={args.n_bx}, mu={args.mu}, mu_mode={args.mu_mode}, "
        f"sqrt(s)={args.e_cm} GeV, processes={args.processes}, "
        f"seed={args.seed}, bx_offset={args.bx_offset}"
    )
    if args.store_tracks:
        print("Also storing selected central charged final-state particles")

    data = generate_protons(
        n_bx=args.n_bx,
        mu=args.mu,
        mu_mode=args.mu_mode,
        e_cm=args.e_cm,
        processes=args.processes,
        seed=args.seed,
        bx_offset=args.bx_offset,
        store_tracks=args.store_tracks,
    )

    print(f"Number of protons stored: {len(data['px'])}")
    if args.store_tracks:
        print(f"Number of central charged tracks stored: {len(data['trk_px'])}")
    print(f"Saving to {args.output}")
    np.savez_compressed(args.output, **data)
    print("Done.")


if __name__ == "__main__":
    main()
