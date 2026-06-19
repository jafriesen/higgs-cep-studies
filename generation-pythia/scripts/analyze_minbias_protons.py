#!/usr/bin/env python3
"""
Analyze min-bias protons stored in minbias_protons.npz.

New features:
  * Apply a pT cut (reject protons with pT > 1.5 GeV).
  * Emulate timing at 420 m: keep only protons whose vertex z is
    within |z| <= 10 cm of the "physics" vertex at z = 0.

We assign a pseudo z-vertex to each (bx_id, interaction_id) drawn
from a Gaussian beam spot with sigma_z (cm). This is an approximate
model of the longitudinal beam profile.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------- beam spot model (for timing emulation) ---------
SIGMA_Z_CM = 5.0  # RMS of luminous region in cm (tunable)

# Station acceptance: xi window, pT cut, and timing usage
STATION_ACCEPTANCE = {
    "192": {
        "xi": (0.0140, 0.0250),
        "pt_max": 1.5,      # GeV
        "use_timing": False,
        "z_window_cm": 10.0 # ignored if use_timing=False
    },
    "213": {
        "xi": (0.0390, 0.0680),
        "pt_max": 1.5,
        "use_timing": False,
        "z_window_cm": 10.0
    },
    "220": {
        "xi": (0.0390, 0.0680),
        "pt_max": 1.5,
        "use_timing": False,
        "z_window_cm": 10.0
    },
    "420": {
        "xi": (0.00325, 0.0120),
        "pt_max": 1.5,
        "use_timing": True,  # emulate timing here
        "z_window_cm": 10.0  # keep protons with |z| <= 10 cm
    },
}


def load_data(npz_path):
    data = np.load(npz_path)
    required = ["bx_id", "interaction_id", "side", "pt", "xi"]
    for key in required:
        if key not in data:
            raise RuntimeError(f"Missing key '{key}' in {npz_path}")
    return data


def assign_vertex_z_cm(bx_id, interaction_id, sigma_z_cm=SIGMA_Z_CM, seed=12345):
    """
    Assign a pseudo z-vertex (in cm) to each (bx_id, interaction_id)
    pair, drawn from a Gaussian beam spot.

    Returns an array z_vertex_cm with one entry per proton.
    """
    # Stack (bx, inter) as keys and find unique pairs
    keys = np.stack((bx_id, interaction_id), axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    rng = np.random.default_rng(seed)
    z_per_interaction = rng.normal(loc=0.0, scale=sigma_z_cm, size=uniq.shape[0])

    # Map back to per-proton array
    z_vertex_cm = z_per_interaction[inv]
    return z_vertex_cm


def global_stats(data):
    bx_id = data["bx_id"]
    pt = data["pt"]
    xi = data["xi"]

    n_protons = len(xi)
    n_bx = int(bx_id.max()) + 1

    protons_per_bx = np.bincount(bx_id, minlength=n_bx)
    mean_ppbx = protons_per_bx.mean()
    std_ppbx = protons_per_bx.std()

    print("=== GLOBAL STATS ===")
    print(f"  Total protons:          {n_protons}")
    print(f"  Total BX:               {n_bx}")
    print(f"  Mean protons per BX:    {mean_ppbx:.3f} ± {std_ppbx:.3f}")
    print(f"  xi range:               [{xi.min():.4e}, {xi.max():.4e}]")
    print(f"  pT range:               [{pt.min():.4e}, {pt.max():.4e}]")
    print("======================\n")

    return n_bx, protons_per_bx


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_global_distributions(data, outdir):
    make_dir(outdir)
    xi = data["xi"]
    pt = data["pt"]

    # xi distribution (all protons)
    plt.figure()
    n_bins = 100
    xi_max = min(0.25, xi.max() * 1.1)
    plt.hist(xi, bins=np.linspace(0.0, xi_max, n_bins), histtype="step", log=True)
    plt.xlabel(r"$\xi = 1 - |p_z| / E_{\mathrm{beam}}$")
    plt.ylabel("Protons / bin")
    plt.title("All protons: xi distribution")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "xi_all.png"))
    plt.close()

    # pT distribution (all protons)
    plt.figure()
    pt_max = min(5.0, pt.max() * 1.1)
    plt.hist(pt, bins=np.linspace(0.0, pt_max, n_bins), histtype="step", log=True)
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel("Protons / bin")
    plt.title("All protons: $p_T$ distribution")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pt_all.png"))
    plt.close()


def binom_err(successes, trials):
    if trials == 0:
        return 0.0
    p = successes / trials
    return np.sqrt(p * (1.0 - p) / trials)


def analyze_station(
    data,
    station_name,
    xi_min,
    xi_max,
    pt_max,
    use_timing,
    z_window_cm,
    n_bx,
    z_vertex_cm,
    bx_frequency_hz=None,
):
    """
    Compute station-level stats with:
      - xi window,
      - pt <= pt_max,
      - optional timing cut |z_vertex| <= z_window_cm.
    """
    bx_id = data["bx_id"]
    side = data["side"]
    xi = data["xi"]
    pt = data["pt"]

    # Base mask: xi window + pT cut
    mask = (xi >= xi_min) & (xi < xi_max) & (pt <= pt_max)

    # Timing emulation: keep only protons whose vertex z
    # is within the timing window
    if use_timing:
        mask &= (np.abs(z_vertex_cm) <= z_window_cm)

    n_tagged = int(mask.sum())
    frac_tagged = n_tagged / len(xi) if len(xi) > 0 else 0.0

    left_mask = mask & (side == -1)
    right_mask = mask & (side == +1)

    has_left = np.bincount(bx_id[left_mask], minlength=n_bx) > 0
    has_right = np.bincount(bx_id[right_mask], minlength=n_bx) > 0
    has_any = np.bincount(bx_id[mask], minlength=n_bx) > 0
    has_double = has_left & has_right

    n_bx_left = int(has_left.sum())
    n_bx_right = int(has_right.sum())
    n_bx_any = int(has_any.sum())
    n_bx_double = int(has_double.sum())

    p_left = n_bx_left / n_bx
    p_right = n_bx_right / n_bx
    p_any = n_bx_any / n_bx
    p_double = n_bx_double / n_bx

    p_left_err = binom_err(n_bx_left, n_bx)
    p_right_err = binom_err(n_bx_right, n_bx)
    p_any_err = binom_err(n_bx_any, n_bx)
    p_double_err = binom_err(n_bx_double, n_bx)

    results = {
        "station": station_name,
        "xi_min": xi_min,
        "xi_max": xi_max,
        "pt_max": pt_max,
        "use_timing": use_timing,
        "z_window_cm": z_window_cm,
        "n_tagged_protons": n_tagged,
        "frac_tagged_protons": frac_tagged,
        "n_bx_left": n_bx_left,
        "n_bx_right": n_bx_right,
        "n_bx_any": n_bx_any,
        "n_bx_double": n_bx_double,
        "p_left": (p_left, p_left_err),
        "p_right": (p_right, p_right_err),
        "p_any": (p_any, p_any_err),
        "p_double": (p_double, p_double_err),
        "xi_tagged": xi[mask],
        "pt_tagged": pt[mask],
        "n_tagged_per_bx": np.bincount(bx_id[mask], minlength=n_bx),
    }

    if bx_frequency_hz is not None:
        rate_left = p_left * bx_frequency_hz
        rate_right = p_right * bx_frequency_hz
        rate_any = p_any * bx_frequency_hz
        rate_double = p_double * bx_frequency_hz

        rate_left_err = p_left_err * bx_frequency_hz
        rate_right_err = p_right_err * bx_frequency_hz
        rate_any_err = p_any_err * bx_frequency_hz
        rate_double_err = p_double_err * bx_frequency_hz

        results.update(
            {
                "rate_left_hz": (rate_left, rate_left_err),
                "rate_right_hz": (rate_right, rate_right_err),
                "rate_any_hz": (rate_any, rate_any_err),
                "rate_double_hz": (rate_double, rate_double_err),
            }
        )

    return results


def print_station_summary(results, n_bx, bx_frequency_hz=None):
    name = results["station"]
    xi_min = results["xi_min"]
    xi_max = results["xi_max"]
    pt_max = results["pt_max"]
    use_timing = results["use_timing"]
    z_window_cm = results["z_window_cm"]

    n_tagged = results["n_tagged_protons"]
    frac_tagged = results["frac_tagged_protons"]

    p_left, p_left_err = results["p_left"]
    p_right, p_right_err = results["p_right"]
    p_any, p_any_err = results["p_any"]
    p_double, p_double_err = results["p_double"]

    print(f"=== STATION {name} ===")
    print(f"  xi window:      [{xi_min:.5f}, {xi_max:.5f}]")
    print(f"  pT cut:         pT <= {pt_max:.2f} GeV")
    if use_timing:
        print(f"  Timing window:  |z_vertex| <= {z_window_cm:.1f} cm (sigma_z = {SIGMA_Z_CM:.1f} cm)")
    else:
        print(f"  Timing:         OFF")

    print(f"  Tagged protons: {n_tagged} ({frac_tagged*100:.4f} % of all)")

    print(f"  BX with ≥1 tag (any side): {results['n_bx_any']} / {n_bx}"
          f"  => P = {p_any:.3e} ± {p_any_err:.3e}")
    print(f"  BX with ≥1 left:           {results['n_bx_left']} / {n_bx}"
          f"  => P = {p_left:.3e} ± {p_left_err:.3e}")
    print(f"  BX with ≥1 right:          {results['n_bx_right']} / {n_bx}"
          f"  => P = {p_right:.3e} ± {p_right_err:.3e}")
    print(f"  BX with double-tag (L&R):  {results['n_bx_double']} / {n_bx}"
          f"  => P = {p_double:.3e} ± {p_double_err:.3e}")

    if bx_frequency_hz is not None and "rate_any_hz" in results:
        r_any, r_any_err = results["rate_any_hz"]
        r_double, r_double_err = results["rate_double_hz"]
        print(f"  Rate (any tag):    {r_any/1e3:8.2f} ± {r_any_err/1e3:6.2f} kHz")
        print(f"  Rate (double tag): {r_double/1e3:8.2f} ± {r_double_err/1e3:6.2f} kHz")
        print(f"  (BX freq = {bx_frequency_hz/1e6:.2f} MHz)")
    print("======================\n")


def plot_station_distributions(results, outdir):
    make_dir(outdir)
    name = results["station"]
    xi = results["xi_tagged"]
    pt = results["pt_tagged"]
    n_tagged_per_bx = results["n_tagged_per_bx"]

    if len(xi) == 0:
        return

    # xi distribution
    plt.figure()
    plt.hist(
        xi,
        bins=np.linspace(results["xi_min"], results["xi_max"], 50),
        histtype="step",
        log=True,
    )
    plt.xlabel(r"$\xi$")
    plt.ylabel("Protons / bin")
    plt.title(f"Station {name}: tagged protons ($\\xi$)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"xi_station_{name}.png"))
    plt.close()

    # pT distribution
    plt.figure()
    pt_max_plot = min(results["pt_max"], pt.max() * 1.1)
    plt.hist(
        pt,
        bins=np.linspace(0.0, pt_max_plot, 50),
        histtype="step",
        log=True,
    )
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel("Protons / bin")
    plt.title(f"Station {name}: tagged protons ($p_T$)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pt_station_{name}.png"))
    plt.close()

    # Multiplicity per BX
    plt.figure()
    vals = n_tagged_per_bx
    if vals.max() > 0:
        max_mult = int(np.percentile(vals[vals > 0], 99))
    else:
        max_mult = 0
    bins = np.arange(0, max_mult + 2) - 0.5
    plt.hist(vals, bins=bins, histtype="step")
    plt.xlabel("Tagged protons per BX")
    plt.ylabel("BX count")
    plt.title(f"Station {name}: multiplicity per BX")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"multiplicity_station_{name}.png"))
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Analyze minbias_protons.npz and compute station tagging stats "
                    "with pT and timing cuts."
    )
    ap.add_argument(
        "-i", "--input",
        default="minbias_protons.npz",
        help="Input .npz file from generator (default: minbias_protons.npz)",
    )
    ap.add_argument(
        "-o", "--outdir",
        default="plots_minbias",
        help="Output directory for plots (default: plots_minbias)",
    )
    ap.add_argument(
        "--bx-frequency-hz",
        type=float,
        default=40e6,
        help="Bunch crossing frequency in Hz (default: 40e6 ~ 40 MHz)",
    )
    args = ap.parse_args()

    data = load_data(args.input)
    n_bx, _ = global_stats(data)
    plot_global_distributions(data, args.outdir)

    print(f"Using BX frequency: {args.bx_frequency_hz/1e6:.2f} MHz")
    print(f"Beam spot model: Gaussian sigma_z = {SIGMA_Z_CM:.1f} cm\n")

    # Assign vertex z for each proton
    z_vertex_cm = assign_vertex_z_cm(data["bx_id"], data["interaction_id"])

    # Analyze each station
    for name, cfg in STATION_ACCEPTANCE.items():
        (xi_min, xi_max) = cfg["xi"]
        pt_max = cfg["pt_max"]
        use_timing = cfg["use_timing"]
        z_window_cm = cfg["z_window_cm"]

        res = analyze_station(
            data,
            station_name=name,
            xi_min=xi_min,
            xi_max=xi_max,
            pt_max=pt_max,
            use_timing=use_timing,
            z_window_cm=z_window_cm,
            n_bx=n_bx,
            z_vertex_cm=z_vertex_cm,
            bx_frequency_hz=args.bx_frequency_hz,
        )
        print_station_summary(res, n_bx, bx_frequency_hz=args.bx_frequency_hz)
        plot_station_distributions(res, os.path.join(args.outdir, f"station_{name}"))


if __name__ == "__main__":
    main()

