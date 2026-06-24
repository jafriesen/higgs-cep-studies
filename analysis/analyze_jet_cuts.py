#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


def repo_root():
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


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


MAX_MULTIPLICITY_BIN = 8  # last bin is an overflow bin: ">= MAX_MULTIPLICITY_BIN"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the >=2 jet selection used by plot_dijets.py / analyze_jets.py: "
            "histogram jet multiplicity and show how pT/eta cuts erode the two-jet count, "
            "per process."
        )
    )
    parser.add_argument(
        "--process",
        action="append",
        dest="processes",
        help="Process name to analyze. May be repeated. Defaults to all processes in PROCESS_ORDER.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--pt-cuts",
        default="20,30,50",
        help="Comma-separated jet pT thresholds [GeV] to test, in addition to no cut.",
    )
    parser.add_argument(
        "--eta-cuts",
        default="2.5,4.7",
        help="Comma-separated |eta| thresholds to test, in addition to no cut.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output plot directory. Plots are written to <output-dir>/<collection>. Defaults to analysis/output/jet_cuts.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
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
    import numpy as np
    import uproot

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return ak, np, plt, uproot


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def parse_float_list(text):
    if not text.strip():
        return []
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    return sorted(values)


def selected_processes(processes, requested):
    if requested:
        unknown = [name for name in requested if name not in processes]
        if unknown:
            known = ", ".join(sorted(processes))
            raise RuntimeError(f"Unknown process(es): {', '.join(unknown)}. Known processes: {known}")
        return requested
    return [name for name in PROCESS_ORDER if name in processes]


def resolve_input(process_name, campaign_name):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    return campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root", campaign


def load_jets(ak, uproot, input_file, tree_name, collection, max_events):
    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
        required = [branch_name(collection, field) for field in ("PT", "Eta")]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {input_file}: {', '.join(missing)}")
        stop = max_events if max_events is not None else None
        arrays = tree.arrays(required, entry_stop=stop, library="ak")

    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    return pt, eta


def multiplicity_counts(ak, np, pt):
    n_jets = ak.to_numpy(ak.num(pt))
    overflow = MAX_MULTIPLICITY_BIN
    clipped = np.minimum(n_jets, overflow)
    counts = np.bincount(clipped, minlength=overflow + 1)[: overflow + 1]
    return n_jets, counts


def cut_stages(pt_cuts, eta_cuts):
    stages = [("no cut", None, None)]
    for cut in pt_cuts:
        stages.append((f"pT > {cut:g} GeV", cut, None))
    for cut in eta_cuts:
        stages.append((f"|eta| < {cut:g}", None, cut))
    if pt_cuts and eta_cuts:
        stages.append((f"pT > {pt_cuts[-1]:g} & |eta| < {eta_cuts[0]:g}", pt_cuts[-1], eta_cuts[0]))
    return stages


def two_jet_fraction(ak, np, pt, eta, pt_cut, eta_cut):
    mask = ak.ones_like(pt, dtype=bool)
    if pt_cut is not None:
        mask = mask & (pt > pt_cut)
    if eta_cut is not None:
        mask = mask & (np.abs(eta) < eta_cut)
    n_pass = ak.num(pt[mask])
    n_total = len(pt)
    n_two_jet = int(ak.sum(n_pass >= 2))
    fraction = n_two_jet / n_total if n_total else 0.0
    return n_two_jet, n_total, fraction


def analyze_process(ak, np, process_name, args):
    input_file, campaign = resolve_input(process_name, args.campaign)
    if not input_file.is_file():
        print(f"Warning: skipping {process_name}: missing file ({input_file})")
        return None

    import uproot

    pt, eta = load_jets(ak, uproot, input_file, args.tree, args.collection, args.max_events)
    n_total = len(pt)
    if n_total == 0:
        print(f"Warning: skipping {process_name}: no events in {input_file}")
        return None

    n_jets, mult_counts = multiplicity_counts(ak, np, pt)

    stages = cut_stages(args.pt_cuts, args.eta_cuts)
    cutflow = []
    for label, pt_cut, eta_cut in stages:
        n_two_jet, _, fraction = two_jet_fraction(ak, np, pt, eta, pt_cut, eta_cut)
        cutflow.append((label, n_two_jet, fraction))
        print(
            f"  {process_name}_{campaign}: [{label}] two_jet={n_two_jet}/{n_total} "
            f"({fraction:.4f})"
        )

    return {
        "name": process_name,
        "campaign": campaign,
        "collection": args.collection,
        "n_total": n_total,
        "n_jets": n_jets,
        "multiplicity_counts": mult_counts,
        "cutflow": cutflow,
    }


def plot_multiplicity(plt, np, results, output_dir):
    bin_labels = [str(i) for i in range(MAX_MULTIPLICITY_BIN)] + [f">={MAX_MULTIPLICITY_BIN}"]
    x = np.arange(len(bin_labels))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    n_results = len(results)
    width = 0.8 / max(n_results, 1)
    for index, result in enumerate(results):
        fractions = result["multiplicity_counts"] / result["n_total"]
        offset = (index - (n_results - 1) / 2.0) * width
        ax.bar(
            x + offset,
            fractions,
            width=width,
            label=result["name"],
            color=COLORS.get(result["name"]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(f"Number of {results[0].get('collection', 'Jet')} jets per event")
    ax.set_ylabel("Fraction of events")
    ax.set_title("Jet multiplicity by process")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "jet_multiplicity.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def plot_cutflow(plt, np, results, output_dir):
    stage_labels = [label for label, _, _ in results[0]["cutflow"]]
    x = np.arange(len(stage_labels))

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    n_results = len(results)
    width = 0.8 / max(n_results, 1)
    for index, result in enumerate(results):
        fractions = [fraction for _, _, fraction in result["cutflow"]]
        offset = (index - (n_results - 1) / 2.0) * width
        ax.bar(
            x + offset,
            fractions,
            width=width,
            label=result["name"],
            color=COLORS.get(result["name"]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction of events with >= 2 selected jets")
    ax.set_title("Two-jet selection efficiency vs. cut stage")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "two_jet_cutflow.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def output_dir_for_collection(args):
    base_output_dir = resolve_path(args.output_dir, base=ROOT) if args.output_dir else ROOT / "analysis" / "output" / "jet_cuts"
    return base_output_dir / args.collection


def main():
    args = parse_args()
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")
    args.pt_cuts = parse_float_list(args.pt_cuts)
    args.eta_cuts = parse_float_list(args.eta_cuts)

    ensure_delphes_python_runtime()
    ak, np, plt, uproot = import_libraries()
    processes = load_yaml(ROOT / "processes.yaml")
    output_dir = output_dir_for_collection(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for process_name in selected_processes(processes, args.processes):
        print(f"Analyzing {process_name}...")
        result = analyze_process(ak, np, process_name, args)
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No usable Delphes ROOT inputs found for the selected processes")

    plot_multiplicity(plt, np, results, output_dir)
    plot_cutflow(plt, np, results, output_dir)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
