#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = (
    ("signal-only", "signal_only", "#c93c3c"),
    ("minbias-only", "minbias_only", "#2f6fb0"),
    ("signal+minbias", "mixed", "#6d3fb3"),
)

VARIABLES = (
    ("xi_L", "xi_L", r"$\xi_L$"),
    ("xi_R", "xi_R", r"$\xi_R$"),
    ("mass", "mass", "M [GeV]"),
    ("pp_z_reco", "pp_z_reco", "PPS reconstructed z [cm]"),
    ("pp_t_reco", "pp_t_reco", "PPS reconstructed t [ps]"),
    ("log_xi_R_over_xi_L", None, r"$\log(\xi_R / \xi_L)$"),
)


def load_arrays(input_path, tree_name):
    try:
        import ROOT
    except ImportError as exc:
        raise RuntimeError("ROOT is required to read the input ROOT tree.") from exc

    required = {
        "signal_only",
        "minbias_only",
        "mixed",
        "xi_L",
        "xi_R",
        "mass",
        "pp_z_reco",
        "pp_t_reco",
    }
    root_file = ROOT.TFile.Open(input_path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file {input_path}")
    tree = root_file.Get(tree_name)
    if tree is None:
        raise RuntimeError(f"Tree '{tree_name}' not found in {input_path}")

    columns = {str(branch.GetName()) for branch in tree.GetListOfBranches()}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"Missing required branches in {input_path}: {', '.join(missing)}")

    dataframe = ROOT.RDataFrame(tree_name, input_path)
    return dataframe.AsNumpy(sorted(required))


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def variable_values(arrays, output_name):
    if output_name == "log_xi_R_over_xi_L":
        xi_l = np.asarray(arrays["xi_L"], dtype=np.float64)
        xi_r = np.asarray(arrays["xi_R"], dtype=np.float64)
        values = np.full(xi_l.size, np.nan, dtype=np.float64)
        valid = np.isfinite(xi_l) & np.isfinite(xi_r) & (xi_l > 0.0) & (xi_r > 0.0)
        values[valid] = np.log(xi_r[valid] / xi_l[valid])
        return values

    branches = {name: branch for name, branch, _label in VARIABLES}
    return np.asarray(arrays[branches[output_name]], dtype=np.float64)


def histogram_range(category_values):
    finite_sets = []
    for values in category_values:
        vals = finite_values(values)
        if vals.size:
            finite_sets.append(vals)
    if not finite_sets:
        return None

    all_values = np.concatenate(finite_sets)
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0.0 else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def plot_variable(arrays, output_name, xlabel, bins, output_dir):
    values = variable_values(arrays, output_name)
    masks = {
        label: np.asarray(arrays[branch], dtype=np.int32) != 0
        for label, branch, _color in CATEGORIES
    }
    value_range = histogram_range([values[mask] for mask in masks.values()])
    if value_range is None:
        print(f"Warning: no finite values for {output_name}; skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    n_plotted = 0
    for label, _branch, color in CATEGORIES:
        vals = finite_values(values[masks[label]])
        if vals.size == 0:
            print(f"Warning: no entries for {label} in {output_name}; omitting category")
            continue
        ax.hist(
            vals,
            bins=bins,
            range=value_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{label} (n={vals.size})",
        )
        n_plotted += 1

    if n_plotted == 0:
        plt.close(fig)
        print(f"Warning: no plottable categories for {output_name}; skipping plot")
        return None

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized entries")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{output_name}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {out_path}")
    return out_path


def main():
    default_output_dir = os.path.join("analysis", "output", "pair_variable_plots")
    parser = argparse.ArgumentParser(
        description="Plot normalized pair-variable histograms from SignalMinbiasPairs ROOT output."
    )
    parser.add_argument("-i", "--input", required=True, help="Input pair-level ROOT file")
    parser.add_argument("-o", "--output-dir", default=default_output_dir, help=f"Output plot directory (default: {default_output_dir})")
    parser.add_argument("--tree", default="SignalMinbiasPairs", help="Input tree name")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    args = parser.parse_args()

    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    arrays = load_arrays(args.input, args.tree)
    print(f"Loaded {len(arrays['xi_L'])} pairs from {args.input}:{args.tree}")

    written = []
    for output_name, _branch, xlabel in VARIABLES:
        out_path = plot_variable(arrays, output_name, xlabel, args.bins, args.output_dir)
        if out_path is not None:
            written.append(out_path)

    print(f"Wrote {len(written)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
