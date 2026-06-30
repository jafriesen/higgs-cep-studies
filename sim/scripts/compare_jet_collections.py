#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


COLLECTIONS = (
    ("GenJet", "generator jets", "#c93c3c"),
    ("FastJet", "raw reconstructed fast jets", "#2f6fb0"),
    ("Jet", "calibrated/reconstructed jets", "#3f8f45"),
    ("JetPUPPI", "PUPPI-filtered jets", "#999999"),
)

VARIABLES = (
    ("PT", "jet_pt", "Jet pT [GeV]"),
    ("Eta", "jet_eta", "Jet eta"),
    ("Phi", "jet_phi", "Jet phi"),
    ("Mass", "jet_mass", "Jet mass [GeV]"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GenJet, FastJet, and Jet kinematics from Delphes outputs."
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name to plot. May be repeated. Defaults to all processes.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Explicit Delphes ROOT file override. Requires exactly one --process.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to <campaign>/plots/compare_jet_collections/<process>_<campaign>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
    return parser.parse_args()


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def import_root():
    try:
        import ROOT
    except ImportError as exc:
        raise RuntimeError(
            "ROOT/PyROOT is required to read Delphes outputs. "
            "Run source setup_env.sh first, or source the Delphes LCG view."
        ) from exc

    return ROOT


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def resolve_input(process_name, campaign_name, input_arg, output_dir_arg):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    default_input = campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root"
    input_file = resolve_path(input_arg, base=ROOT) if input_arg else default_input
    output_dir = (
        resolve_path(output_dir_arg, base=ROOT)
        if output_dir_arg
        else campaign_dir / "plots" / "compare_jet_collections" / f"{process_name}_{campaign}"
    )
    return input_file, output_dir, campaign


def branch_names(tree):
    return {str(branch.GetName()) for branch in tree.GetListOfBranches()}


def draw_values(tree, expression, entries):
    selected = tree.Draw(expression, "", "goff", entries)
    if selected < 0:
        raise RuntimeError(f"Could not read expression '{expression}'")
    values = tree.GetV1()
    return [float(values[index]) for index in range(selected)]


def read_collections(ROOT, input_file, tree_name, max_events):
    root_file = ROOT.TFile.Open(str(input_file))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {input_file}")

    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")

    available = branch_names(tree)
    collections = [name for name, _label, _color in COLLECTIONS if name in available]
    for name, _label, _color in COLLECTIONS:
        if name not in available:
            print(f"Warning: missing branch {name}; skipping collection")
    if not collections:
        root_file.Close()
        raise RuntimeError(f"No requested jet collections found in {input_file}")

    entries = tree.GetEntries()
    if max_events is not None:
        entries = min(entries, max_events)
    tree.SetEstimate(max(1000000, entries * 1000))

    data = {
        name: {field: [] for field, _plot_name, _xlabel in VARIABLES}
        for name in collections
    }
    multiplicities = {name: [] for name in collections}

    for name in collections:
        size_branch = f"{name}_size"
        if size_branch in available:
            multiplicities[name] = [int(value) for value in draw_values(tree, size_branch, entries)]
        else:
            print(f"Warning: missing branch {size_branch}; multiplicity for {name} will be omitted")
            multiplicities[name] = []

        for field, _plot_name, _xlabel in VARIABLES:
            data[name][field] = draw_values(tree, f"{name}.{field}", entries)

    root_file.Close()
    return data, multiplicities, entries


def finite_values(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(np, value_sets):
    finite_sets = [finite_values(np, values) for values in value_sets]
    finite_sets = [values for values in finite_sets if values.size]
    if not finite_sets:
        return None

    values = np.concatenate(finite_sets)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def plot_overlay(plt, np, series_by_collection, xlabel, output_path, bins):
    value_range = histogram_range(np, series_by_collection.values())
    if value_range is None:
        print(f"Warning: no finite values for {output_path.name}; skipping")
        return False

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    plotted = 0
    for name, label, color in COLLECTIONS:
        if name not in series_by_collection:
            continue
        values = finite_values(np, series_by_collection[name])
        if values.size == 0:
            print(f"Warning: no entries for {name} in {output_path.name}; omitting")
            continue
        ax.hist(
            values,
            bins=bins,
            range=value_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{label} (n={values.size})",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized entries")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    return True


def write_plots(plt, np, data, multiplicities, output_dir, bins):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for field, plot_name, xlabel in VARIABLES:
        series = {
            name: fields[field]
            for name, fields in data.items()
            if field in fields
        }
        if plot_overlay(plt, np, series, xlabel, output_dir / f"{plot_name}.png", bins):
            written += 1

    if plot_overlay(
        plt,
        np,
        multiplicities,
        "Jets per event",
        output_dir / "jet_multiplicity.png",
        bins,
    ):
        written += 1

    return written


def main():
    args = parse_args()
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")
    if args.input and (not args.processes or len(args.processes) != 1):
        raise RuntimeError("--input requires exactly one --process so the output location is unambiguous")

    plt, np = import_plotting()
    ROOT_module = import_root()

    processes = load_yaml(ROOT / "processes.yaml")
    total_plots = 0

    for process_name in selected_processes(processes, args.processes):
        input_file, output_dir, campaign = resolve_input(
            process_name,
            args.campaign,
            args.input,
            args.output_dir,
        )
        if not input_file.is_file():
            raise RuntimeError(f"Missing Delphes ROOT input for {process_name}: {input_file}")

        print(f"Reading {input_file}")
        data, multiplicities, entries = read_collections(
            ROOT_module,
            input_file,
            args.tree,
            args.max_events,
        )
        print(f"Loaded {entries} event(s) for {process_name}_{campaign}")
        for name, fields in data.items():
            n_objects = len(fields["PT"])
            nonzero_events = sum(1 for value in multiplicities[name] if value)
            print(f"  {name}: {n_objects} jets, {nonzero_events} nonzero event(s)")

        total_plots += write_plots(plt, np, data, multiplicities, output_dir, args.bins)

    print(f"Wrote {total_plots} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
