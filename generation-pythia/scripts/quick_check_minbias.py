#!/usr/bin/env python3
"""
Quick cross-check for minbias_protons.npz:
  * No chamber / station cuts applied.
  * Print xi and pT quantiles.
  * Plot xi and pT histograms with quantile lines drawn on top.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on batch
import matplotlib.pyplot as plt


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input",
        default="minbias_protons.npz",
        help="Input .npz file (default: minbias_protons.npz)",
    )
    ap.add_argument(
        "-o", "--outdir",
        default="quick_check_plots",
        help="Output directory for plots (default: quick_check_plots)",
    )
    args = ap.parse_args()

    make_dir(args.outdir)

    data = np.load(args.input)
    xi = data["xi"]
    pt = data["pt"]

    print(f"Loaded {len(xi)} protons from {args.input}")

    # Quantiles to show
    quantiles = [0.50, 0.90, 0.95, 0.99, 0.999]

    # ---- xi quantiles ----
    xi_q = np.quantile(xi, quantiles)
    print("\nxi quantiles (no cuts):")
    for q, v in zip(quantiles, xi_q):
        print(f"  q={q:6.3f} -> xi = {v:.6e}")

    # ---- pT quantiles ----
    pt_q = np.quantile(pt, quantiles)
    print("\npT quantiles (no cuts):")
    for q, v in zip(quantiles, pt_q):
        print(f"  q={q:6.3f} -> pT = {v:.6f} GeV")

    # ---- xi histogram with quantile lines ----
    plt.figure()
    n_bins = 100
    xi_max = min(0.25, xi.max() * 1.1)
    bins_xi = np.linspace(0.0, xi_max, n_bins)

    plt.hist(xi, bins=bins_xi, histtype="step", log=True, label="All protons")

    # draw quantile lines
    for q, v in zip(quantiles, xi_q):
        if v > xi_max:
            continue
        plt.axvline(v, linestyle="--", linewidth=1)
        plt.text(
            v,
            plt.ylim()[1] / 5.0,  # somewhere in the middle of the y-range
            f"q={q:.3f}\n{v:.3e}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
        )

    plt.xlabel(r"$\xi = 1 - |p_z| / E_{\mathrm{beam}}$")
    plt.ylabel("Protons / bin")
    plt.title(r"All protons: $\xi$ distribution (no cuts)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "xi_all_quantiles.png"))
    plt.close()

    # ---- pT histogram with quantile lines ----
    plt.figure()
    pt_max = min(5.0, pt.max() * 1.1)
    bins_pt = np.linspace(0.0, pt_max, n_bins)

    plt.hist(pt, bins=bins_pt, histtype="step", log=True, label="All protons")

    for q, v in zip(quantiles, pt_q):
        if v > pt_max:
            continue
        plt.axvline(v, linestyle="--", linewidth=1)
        plt.text(
            v,
            plt.ylim()[1] / 5.0,
            f"q={q:.3f}\n{v:.3f}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
        )

    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel("Protons / bin")
    plt.title(r"All protons: $p_T$ distribution (no cuts)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pt_all_quantiles.png"))
    plt.close()

    print(f"\nPlots written to: {args.outdir}")
    print("  - xi_all_quantiles.png")
    print("  - pt_all_quantiles.png")


if __name__ == "__main__":
    main()

