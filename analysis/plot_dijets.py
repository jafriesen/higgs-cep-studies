#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


LUMI_FB = 3000.0


def repo_root():
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


# Edit these labels/ranges/bins in the script when presentation needs change.
# A range or bins of None falls back to the inferred range / --bins CLI default.
PLOT_VARIABLES = {
    "dijet_mass": {
        "output": "dijet_mass.png",
        "xlabel": "Dijet mass [GeV]",
        "range": (95,145),
        "bins": 50,
    },
    "dijet_pt": {
        "output": "dijet_pt.png",
        "xlabel": "Dijet pT [GeV]",
        "range": None,
        "bins": None,
    },
    "dijet_eta": {
        "output": "dijet_eta.png",
        "xlabel": "Dijet eta",
        "range": None,
        "bins": None,
    },
    "dijet_phi": {
        "output": "dijet_phi.png",
        "xlabel": "Dijet phi",
        "range": None,
        "bins": None,
    },
}


PROCESS_ORDER = (
    "qcd_gg",
    "qcd_qq",
    "qcd_bb",
    "qcd_cc",
    "qed_bb",
    "qed_cc",
    "h_bb",
    "h_cc",
)


# Edit these legend labels in the script when presentation needs change.
PROCESS_LABELS = {
    "qcd_gg": "QCD $gg$",
    "qcd_qq": "QCD $q\\bar{q}$",
    "qcd_bb": "QCD $b\\bar{b}$",
    "qcd_cc": "QCD $c\\bar{c}$",
    "qed_bb": "QED $b\\bar{b}$",
    "qed_cc": "QED $c\\bar{c}$",
    "h_bb": "$H\\rightarrow b\\bar{b}$",
    "h_cc": "$H\\rightarrow c\\bar{c}$",
}


# Flavor-dependent x-axis label for the dijet mass plot.
FLAVOR_MASS_XLABELS = {
    "bb": "$m_{b\\bar{b}}$ [GeV]",
    "cc": "$m_{c\\bar{c}}$ [GeV]",
}


# Okabe-Ito colorblind-friendly palette. Every process gets its own color since
# h_bb/h_cc, qed_bb/qed_cc, and qcd_bb/qcd_cc are all plotted together regardless
# of --flavor (only their tag weight differs).
COLORS = {
    "h_bb": "#0072B2",
    "h_cc": "#D55E00",
    "qed_bb": "#009E73",
    "qed_cc": "#CC79A7",
    "qcd_bb": "#E69F00",
    "qcd_cc": "#56B4E9",
    "qcd_qq": "#000000",
    "qcd_gg": "#F0E442",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot weighted Delphes dijet observables for bb or cc tag selections."
    )
    parser.add_argument("--flavor", choices=("bb", "cc"), required=True, help="Tag target")
    parser.add_argument(
        "--include-light-qcd",
        action="store_true",
        help="Include qcd_qq and qcd_gg with light-to-heavy mistag weights.",
    )
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis")
    parser.add_argument("--stacked", action="store_true", help="Stack sample histograms instead of overlaying them")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to analysis/output/dijets_<flavor>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    return parser.parse_args()


def ensure_delphes_python_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_ANALYZER_ENV") == "1":
        return

    lcg_view = Path(
        os.environ.get(
            "DELPHES_LCG_VIEW",
            "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh",
        )
    )
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
    completed = subprocess.run(["bash", "-lc", command], cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


def import_libraries():
    import awkward as ak
    import matplotlib
    import mplhep
    import numpy as np
    import uproot
    import vector

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vector.register_awkward()
    return ak, np, plt, uproot


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def process_flavor(process):
    jet_ids = [abs(int(pdg_id)) for pdg_id in process.get("jet_pdg_ids", [])]
    if any(pdg_id == 5 for pdg_id in jet_ids):
        return "bb"
    if any(pdg_id == 4 for pdg_id in jet_ids):
        return "cc"
    return "light"


def selected_processes(processes, include_light_qcd):
    selected = []
    for name in PROCESS_ORDER:
        if name not in processes:
            continue
        source_flavor = process_flavor(processes[name])
        if source_flavor in ("bb", "cc"):
            selected.append(name)
        elif include_light_qcd and name in ("qcd_qq", "qcd_gg"):
            selected.append(name)
    return selected


def tag_weight(parameters, target_flavor, source_flavor):
    tagging = parameters.get("tagging", {})
    if target_flavor == "cc":
        if source_flavor == "cc":
            probability = tagging["eff_c"]
        elif source_flavor == "bb":
            probability = tagging["mistag_b_to_c"]
        else:
            probability = tagging["mistag_light_to_c"]
    else:
        if source_flavor == "bb":
            probability = tagging["eff_b"]
        elif source_flavor == "cc":
            probability = tagging["mistag_c_to_b"]
        else:
            probability = tagging["mistag_light_to_b"]
    return float(probability) ** 2


def resolve_input(process_name, campaign_name):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    return campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root", campaign


def load_dijets(ak, uproot, input_file, tree_name, collection):
    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
        n_generated = int(tree.num_entries)
        required = [branch_name(collection, field) for field in ("PT", "Eta", "Phi", "Mass")]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {input_file}: {', '.join(missing)}")
        arrays = tree.arrays(required, library="ak")

    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    phi = ak.values_astype(arrays[branch_name(collection, "Phi")], "float64")
    mass = ak.values_astype(arrays[branch_name(collection, "Mass")], "float64")

    has_two_jets = ak.num(pt) >= 2
    n_selected = int(ak.sum(has_two_jets))
    if n_selected == 0:
        raise RuntimeError(f"No events with at least two {collection} jets in {input_file}")

    jets = ak.zip(
        {
            "pt": pt[has_two_jets],
            "eta": eta[has_two_jets],
            "phi": phi[has_two_jets],
            "mass": mass[has_two_jets],
        },
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    leading = jets[order][:, 0]
    subleading = jets[order][:, 1]
    dijet = leading + subleading

    return {
        "dijet_mass": ak.to_numpy(dijet.mass),
        "dijet_pt": ak.to_numpy(dijet.pt),
        "dijet_eta": ak.to_numpy(dijet.eta),
        "dijet_phi": ak.to_numpy(dijet.phi),
    }, n_generated, n_selected


def finite_values(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(np, datasets):
    values = [finite_values(np, dataset) for dataset in datasets]
    values = [dataset for dataset in values if dataset.size]
    if not values:
        return None
    values = np.concatenate(values)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def default_output_dir(flavor, include_light_qcd):
    suffix = f"dijets_{flavor}"
    if include_light_qcd:
        suffix += "_with_light_qcd"
    return ROOT / "analysis" / "output" / suffix


def read_samples(ak, np, uproot, processes, parameters, args):
    samples = []
    skipped = []
    for process_name in selected_processes(processes, args.include_light_qcd):
        process = processes[process_name]
        input_file, campaign = resolve_input(process_name, args.campaign)
        if not input_file.is_file():
            skipped.append((process_name, input_file, "missing file"))
            continue

        source_flavor = process_flavor(process)
        tag = tag_weight(parameters, args.flavor, source_flavor)
        try:
            observables, n_generated, n_selected = load_dijets(
                ak, uproot, input_file, args.tree, args.collection
            )
        except RuntimeError as exc:
            skipped.append((process_name, input_file, str(exc)))
            continue
        if n_generated <= 0:
            raise RuntimeError(f"{input_file} has zero generated events")

        event_weight = float(process["xsec_fb"]) * LUMI_FB * tag / float(n_generated)
        expected_yield = event_weight * n_selected
        samples.append(
            {
                "name": process_name,
                "campaign": campaign,
                "observables": observables,
                "event_weight": event_weight,
                "expected_yield": expected_yield,
                "tag_weight": tag,
                "n_generated": n_generated,
                "n_selected": n_selected,
            }
        )
        print(
            f"{process_name}_{campaign}: generated={n_generated}, "
            f"two_jet={n_selected}, tag_weight={tag:.6g}, "
            f"event_weight={event_weight:.6g}, expected_yield={expected_yield:.6g}"
        )

    for process_name, input_file, reason in skipped:
        print(f"Warning: skipping {process_name}: {reason} ({input_file})")

    if not samples:
        raise RuntimeError("No usable Delphes ROOT inputs found for the selected processes")

    for sample in samples:
        for key in PLOT_VARIABLES:
            sample["observables"][key] = finite_values(np, sample["observables"][key])
    return samples


def print_range_yields(np, samples, variable, value_range):
    lo, hi = value_range
    total = 0.0
    print(f"Yields for {variable} within range [{lo:.6g}, {hi:.6g}]:")
    for sample in samples:
        values = sample["observables"][variable]
        in_range = np.sum((values >= lo) & (values <= hi))
        range_yield = sample["event_weight"] * in_range
        total += range_yield
        print(f"  {sample['name']}_{sample['campaign']}: in_range={int(in_range)}, range_yield={range_yield:.6g}")
    print(f"  total range_yield={total:.6g}")


def plot_variable(np, plt, samples, variable, options, output_dir, default_bins, log_y, stacked, flavor):
    bins = options["bins"] or default_bins
    datasets = [sample["observables"][variable] for sample in samples]
    value_range = options["range"] or histogram_range(np, datasets)
    if value_range is None:
        print(f"Warning: no finite values for {variable}; skipping")
        return False

    if options["range"] is not None:
        print_range_yields(np, samples, variable, options["range"])

    plot_samples = [sample for sample in samples if sample["observables"][variable].size]
    if not plot_samples:
        print(f"Warning: no finite values for {variable}; skipping")
        return False

    values = [sample["observables"][variable] for sample in plot_samples]
    weights = [
        np.full(sample["observables"][variable].shape, sample["event_weight"], dtype=np.float64)
        for sample in plot_samples
    ]
    labels = [
        f"{PROCESS_LABELS.get(sample['name'], sample['name'])} ({sample['expected_yield']:.3g})"
        for sample in plot_samples
    ]
    colors = [COLORS.get(sample["name"], None) for sample in plot_samples]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    if stacked:
        hist_kwargs = dict(histtype="stepfilled", alpha=0.75, linewidth=0.8, edgecolor="#333333")
    else:
        # Unfilled outlines so overlapping samples stay distinguishable instead of blending/hiding each other.
        hist_kwargs = dict(histtype="step", alpha=1.0, linewidth=1.5)
    _, _, patches = ax.hist(
        values,
        bins=bins,
        range=value_range,
        weights=weights,
        stacked=stacked,
        label=labels,
        color=colors,
        **hist_kwargs,
    )
    xlabel = FLAVOR_MASS_XLABELS[flavor] if variable == "dijet_mass" else options["xlabel"]
    ax.set_xlabel(xlabel)
    ax.set_xlim(value_range)
    ax.set_ylabel(options.get("ylabel", "Expected events / bin"))
    if log_y:
        ax.set_yscale("log")
        positive = []
        for sample in plot_samples:
            counts, _ = np.histogram(
                sample["observables"][variable],
                bins=bins,
                range=value_range,
                weights=np.full(
                    sample["observables"][variable].shape,
                    sample["event_weight"],
                    dtype=np.float64,
                ),
            )
            positive.extend(counts[counts > 0.0])
        if positive:
            ax.set_ylim(bottom=float(np.min(positive)) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = output_dir / options["output"]
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    return bool(patches)


def write_plots(np, plt, samples, output_dir, bins, log_y, stacked, flavor):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for variable, options in PLOT_VARIABLES.items():
        if plot_variable(np, plt, samples, variable, options, output_dir, bins, log_y, stacked, flavor):
            written += 1
    return written


def main():
    args = parse_args()
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")

    ensure_delphes_python_runtime()
    ak, np, plt, uproot = import_libraries()
    processes = load_yaml(ROOT / "processes.yaml")
    parameters = load_yaml(ROOT / "parameters.yaml")
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else default_output_dir(args.flavor, args.include_light_qcd)
    )

    samples = read_samples(ak, np, uproot, processes, parameters, args)
    written = write_plots(np, plt, samples, output_dir, args.bins, args.log_y, args.stacked, args.flavor)
    print(f"Wrote {written} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
