#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import analyzer


PLOT_VARIABLES = {
    "jet1_pt_before_pps": {
        "output": "jet1_pt_before_pps.png",
        "xlabel": "Jet 1 pT [GeV]",
        "range": None,
        "bins": None,
    },
    "jet2_pt_before_pps": {
        "output": "jet2_pt_before_pps.png",
        "xlabel": "Jet 2 pT [GeV]",
        "range": None,
        "bins": None,
    },
    "jet1_pt_over_mjj_before_pps": {
        "output": "jet1_pt_over_mjj_before_pps.png",
        "xlabel": "Jet 1 pT / dijet mass",
        "range": (0, 1),
        "bins": None,
    },
    "jet2_pt_over_mjj_before_pps": {
        "output": "jet2_pt_over_mjj_before_pps.png",
        "xlabel": "Jet 2 pT / dijet mass",
        "range": (0, 1),
        "bins": None,
    },
    "jet1_eta_before_pps": {
        "output": "jet1_eta_before_pps.png",
        "xlabel": "Jet 1 eta",
        "range": None,
        "bins": None,
    },
    "jet2_eta_before_pps": {
        "output": "jet2_eta_before_pps.png",
        "xlabel": "Jet 2 eta",
        "range": None,
        "bins": None,
    },
    "jet1_phi_before_pps": {
        "output": "jet1_phi_before_pps.png",
        "xlabel": "Jet 1 phi",
        "range": (-3.141592653589793, 3.141592653589793),
        "bins": None,
    },
    "jet2_phi_before_pps": {
        "output": "jet2_phi_before_pps.png",
        "xlabel": "Jet 2 phi",
        "range": (-3.141592653589793, 3.141592653589793),
        "bins": None,
    },
    "jet1_pt_after_pps": {
        "output": "jet1_pt_after_pps.png",
        "xlabel": "Jet 1 pT [GeV]",
        "range": None,
        "bins": None,
    },
    "jet2_pt_after_pps": {
        "output": "jet2_pt_after_pps.png",
        "xlabel": "Jet 2 pT [GeV]",
        "range": None,
        "bins": None,
    },
    "jet1_pt_over_mjj_after_pps": {
        "output": "jet1_pt_over_mjj_after_pps.png",
        "xlabel": "Jet 1 pT / dijet mass",
        "range": (0, 1),
        "bins": None,
    },
    "jet2_pt_over_mjj_after_pps": {
        "output": "jet2_pt_over_mjj_after_pps.png",
        "xlabel": "Jet 2 pT / dijet mass",
        "range": (0, 1),
        "bins": None,
    },
    "jet1_eta_after_pps": {
        "output": "jet1_eta_after_pps.png",
        "xlabel": "Jet 1 eta",
        "range": None,
        "bins": None,
    },
    "jet2_eta_after_pps": {
        "output": "jet2_eta_after_pps.png",
        "xlabel": "Jet 2 eta",
        "range": None,
        "bins": None,
    },
    "jet1_phi_after_pps": {
        "output": "jet1_phi_after_pps.png",
        "xlabel": "Jet 1 phi",
        "range": (-3.141592653589793, 3.141592653589793),
        "bins": None,
    },
    "jet2_phi_after_pps": {
        "output": "jet2_phi_after_pps.png",
        "xlabel": "Jet 2 phi",
        "range": (-3.141592653589793, 3.141592653589793),
        "bins": None,
    },
}


PROCESS_ORDER = (
    "QCDgg",
    "QCDqq",
    "QCDbb",
    "QCDcc",
    "QEDbb",
    "QEDcc",
    "Hbb",
    "Hcc",
)


PROCESS_LABELS = {
    "QCDgg": "QCD $gg$",
    "QCDqq": "QCD $q\\bar{q}$",
    "QCDbb": "QCD $b\\bar{b}$",
    "QCDcc": "QCD $c\\bar{c}$",
    "QEDbb": "QED $b\\bar{b}$",
    "QEDcc": "QED $c\\bar{c}$",
    "Hbb": "$H\\rightarrow b\\bar{b}$",
    "Hcc": "$H\\rightarrow c\\bar{c}$",
}


COLORS = {
    "Hbb": "#0072B2",
    "Hcc": "#D55E00",
    "QEDbb": "#009E73",
    "QEDcc": "#CC79A7",
    "QCDbb": "#E69F00",
    "QCDcc": "#56B4E9",
    "QCDqq": "#000000",
    "QCDgg": "#F0E442",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot weighted jet observables for tagged Delphes dijet selections."
    )
    parser.add_argument("--flavor", choices=("bb", "cc"), required=True, help="Tag target")
    parser.add_argument(
        "--include-light-qcd",
        action="store_true",
        help="Include light QCD with light-to-heavy mistag weights.",
    )
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis")
    parser.add_argument("--stacked", action="store_true", help="Stack weighted sample histograms")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to analysis/output/jets_<flavor>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of matched Delphes/Pythia file pairs to load per process.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of file pairs to load in parallel per process")
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file with beam.sqrt_s_gev, pps.xi_ranges, pps.xi_res, and random.seed.",
    )
    return parser.parse_args()


def default_output_dir(flavor, include_light_qcd):
    suffix = f"jets_{flavor}"
    if include_light_qcd:
        suffix += "_with_light_qcd"
    return analyzer.ROOT / "analysis" / "output" / suffix


def build_observables(np, table):
    before_pps = analyzer.make_selection(np, table, require_valid_pp=True)
    after_pps = analyzer.make_selection(np, table, require_pps=True)
    observables = {}
    for key in (
        "jet1_pt",
        "jet2_pt",
        "jet1_pt_over_mjj",
        "jet2_pt_over_mjj",
        "jet1_eta",
        "jet2_eta",
        "jet1_phi",
        "jet2_phi",
    ):
        observables[f"{key}_before_pps"] = table[key][before_pps]
        observables[f"{key}_after_pps"] = table[key][after_pps]
    return observables


def main():
    args = parse_args()
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.workers <= 0:
        raise RuntimeError("--workers must be > 0")

    analyzer.ensure_analysis_runtime(Path(__file__), sys.argv[1:])
    ak, np, plt, uproot = analyzer.import_libraries()
    processes = analyzer.load_yaml(analyzer.ROOT / "processes.yaml")
    parameters = analyzer.load_yaml(analyzer.ROOT / "parameters.yaml")
    pps_config = analyzer.load_pps_config(analyzer.resolve_path(args.pps_config, base=analyzer.ROOT))
    output_dir = (
        analyzer.resolve_path(args.output_dir, base=analyzer.ROOT)
        if args.output_dir
        else default_output_dir(args.flavor, args.include_light_qcd)
    )

    samples = analyzer.read_samples(
        ak, np, uproot, processes, parameters, pps_config, args, PROCESS_ORDER, build_observables
    )
    written = analyzer.write_plots(
        np,
        plt,
        samples,
        PLOT_VARIABLES,
        output_dir,
        args.bins,
        args.log_y,
        args.stacked,
        PROCESS_LABELS,
        COLORS,
    )
    print(f"Wrote {written} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
