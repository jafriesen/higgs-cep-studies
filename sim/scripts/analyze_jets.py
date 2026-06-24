#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


# Each entry is (key, output_name, xlabel) with an optional (lo, hi) x-range to
# override the auto-computed histogram range.
VARIABLES = (
    ("leading_pt", "leading_jet_pt", "Leading jet pT [GeV]"),
    ("subleading_pt", "subleading_jet_pt", "Subleading jet pT [GeV]"),
    ("leading_eta", "leading_jet_eta", "Leading jet eta"),
    ("subleading_eta", "subleading_jet_eta", "Subleading jet eta"),
    ("leading_phi", "leading_jet_phi", "Leading jet phi"),
    ("subleading_phi", "subleading_jet_phi", "Subleading jet phi"),
    ("dijet_mass", "dijet_mass", "Dijet mass [GeV]", (100, 150)),
    ("dijet_pt", "dijet_pt", "Dijet pT [GeV]"),
    ("dijet_eta", "dijet_eta", "Dijet eta"),
    ("dijet_phi", "dijet_phi", "Dijet phi"),
    ("delta_eta", "delta_eta", "Delta eta"),
    ("delta_r", "delta_r", "Delta R"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the two leading jets from a Delphes jet collection."
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name to analyze. May be repeated. Defaults to all processes.",
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
        help="Output plot directory. Defaults to <campaign>/plots/analyze_jets/<process>_<campaign>/<collection>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
    return parser.parse_args()


def ensure_delphes_python_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_ANALYZER_ENV") == "1":
        return

    lcg_view = Path(os.environ.get(
        "DELPHES_LCG_VIEW",
        "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh",
    ))
    delphes_dir = Path(os.environ.get("DELPHES_DIR", "/home/jfriesen/Delphes"))

    if not lcg_view.is_file():
        raise RuntimeError(f"Delphes LCG view setup script does not exist: {lcg_view}")

    command = "\n".join(
        [
            f"source {shlex.quote(str(lcg_view))}",
            f"export DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"export LD_LIBRARY_PATH={shlex.quote(str(delphes_dir))}:$LD_LIBRARY_PATH",
            "export HIGGS_CEP_DELPHES_ANALYZER_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        ]
    )
    subprocess.run(["bash", "-lc", command], cwd=ROOT, check=True)
    raise SystemExit(0)


def selected_processes(processes, requested):
    if not requested:
        return list(processes)
    unknown = [name for name in requested if name not in processes]
    if unknown:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
    return requested


def resolve_input(process_name, campaign_name, input_arg, output_dir_arg, collection):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    default_input = campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root"
    input_file = resolve_path(input_arg, base=ROOT) if input_arg else default_input
    output_dir = (
        resolve_path(output_dir_arg, base=ROOT)
        if output_dir_arg
        else campaign_dir / "plots" / "analyze_jets" / f"{process_name}_{campaign}" / collection
    )
    return input_file, output_dir, campaign


def import_libraries():
    import awkward as ak
    import matplotlib
    import numpy as np
    import uproot
    import vector

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vector.register_awkward()
    return ak, np, plt, uproot, vector


def finite_values(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(np, values):
    values = finite_values(np, values)
    if values.size == 0:
        return None
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def load_two_jet_observables(ak, uproot, input_file, tree_name, collection, max_events):
    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
        required = [branch_name(collection, field) for field in ("PT", "Eta", "Phi", "Mass")]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {input_file}: {', '.join(missing)}")

        stop = max_events if max_events is not None else None
        arrays = tree.arrays(required, entry_stop=stop, library="ak")

    # Cast from the float32 stored in the ROOT file to float64: the dijet mass is
    # computed via E^2 - p^2, and that cancellation loses all precision in float32
    # for boosted jets (yielding negative or wildly wrong masses).
    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    phi = ak.values_astype(arrays[branch_name(collection, "Phi")], "float64")
    mass = ak.values_astype(arrays[branch_name(collection, "Mass")], "float64")

    has_two_jets = ak.num(pt) >= 2
    selected_events = int(ak.sum(has_two_jets))
    total_events = len(pt)
    if selected_events == 0:
        raise RuntimeError(f"No events with at least two {collection} jets in {input_file}")

    jets = ak.zip(
        {"pt": pt[has_two_jets], "eta": eta[has_two_jets], "phi": phi[has_two_jets], "mass": mass[has_two_jets]},
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    leading = jets[order][:, 0]
    subleading = jets[order][:, 1]
    dijet = leading + subleading

    observables = {
        "leading_pt": ak.to_numpy(leading.pt),
        "subleading_pt": ak.to_numpy(subleading.pt),
        "leading_eta": ak.to_numpy(leading.eta),
        "subleading_eta": ak.to_numpy(subleading.eta),
        "leading_phi": ak.to_numpy(leading.phi),
        "subleading_phi": ak.to_numpy(subleading.phi),
        "dijet_mass": ak.to_numpy(dijet.mass),
        "dijet_pt": ak.to_numpy(dijet.pt),
        "dijet_eta": ak.to_numpy(dijet.eta),
        "dijet_phi": ak.to_numpy(dijet.phi),
        "delta_eta": ak.to_numpy(leading.eta - subleading.eta),
        "delta_r": ak.to_numpy(leading.deltaR(subleading)),
    }
    return observables, total_events, selected_events


def plot_histogram(np, plt, values, xlabel, output_path, bins, xrange=None):
    values = finite_values(np, values)
    value_range = xrange if xrange is not None else histogram_range(np, values)
    if value_range is None:
        print(f"Warning: no finite values for {output_path.name}; skipping")
        return False

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.hist(
        values,
        bins=bins,
        range=value_range,
        histtype="stepfilled",
        alpha=0.45,
        color="#2f6fb0",
        edgecolor="#1f4e7d",
        linewidth=1.2,
        label=f"n={values.size}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    return True


def write_plots(np, plt, observables, output_dir, bins):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for key, output_name, xlabel, *xrange in VARIABLES:
        output_path = output_dir / f"{output_name}.png"
        if plot_histogram(np, plt, observables[key], xlabel, output_path, bins, xrange[0] if xrange else None):
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

    ensure_delphes_python_runtime()
    ak, np, plt, uproot, vector = import_libraries()
    processes = load_yaml(ROOT / "processes.yaml")
    total_plots = 0

    for process_name in selected_processes(processes, args.processes):
        input_file, output_dir, campaign = resolve_input(
            process_name,
            args.campaign,
            args.input,
            args.output_dir,
            args.collection,
        )
        if not input_file.is_file():
            raise RuntimeError(f"Missing Delphes ROOT input for {process_name}: {input_file}")

        print(f"Reading {input_file}")
        observables, total_events, selected_events = load_two_jet_observables(
            ak,
            uproot,
            input_file,
            args.tree,
            args.collection,
            args.max_events,
        )
        print(
            f"Loaded {selected_events}/{total_events} event(s) with at least two "
            f"{args.collection} jets for {process_name}_{campaign}"
        )
        total_plots += write_plots(np, plt, observables, output_dir, args.bins)

    print(f"Wrote {total_plots} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
