#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import analyzer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_utils import discover_event_files, natural_key, resolve_path  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_campaign_root,
    generation_config,
    generation_process_config,
    generation_stage_root,
)


LUMI_FB = 3000.0
MIN_JET_PT = 20.0
MAX_MULTIPLICITY_BIN = 10

PLOT_VARIABLES = {
    "jet_multiplicity": {
        "output": "jet_multiplicity.png",
        "xlabel": "Number of jets per event (10 includes >=10)",
        "range": (-0.5, MAX_MULTIPLICITY_BIN + 0.5),
        "bins": MAX_MULTIPLICITY_BIN + 1,
    },
    "leading_jet_pt": {
        "output": "leading_jet_pt.png",
        "xlabel": "Leading jet pT [GeV]",
        "range": None,
        "bins": None,
    },
    "subleading_jet_pt": {
        "output": "subleading_jet_pt.png",
        "xlabel": "Subleading jet pT [GeV]",
        "range": None,
        "bins": None,
    },
    "leading_jet_pt_over_mjj": {
        "output": "leading_jet_pt_over_mjj.png",
        "xlabel": "Leading jet pT / dijet mass",
        "range": (0, 1),
        "bins": None,
    },
    "subleading_jet_pt_over_mjj": {
        "output": "subleading_jet_pt_over_mjj.png",
        "xlabel": "Subleading jet pT / dijet mass",
        "range": (0, 1),
        "bins": None,
    },
    "leading_jet_eta": {
        "output": "leading_jet_eta.png",
        "xlabel": "Leading jet eta",
        "range": None,
        "bins": None,
    },
    "subleading_jet_eta": {
        "output": "subleading_jet_eta.png",
        "xlabel": "Subleading jet eta",
        "range": None,
        "bins": None,
    },
    "leading_jet_phi": {
        "output": "leading_jet_phi.png",
        "xlabel": "Leading jet phi",
        "range": (-math.pi, math.pi),
        "bins": None,
    },
    "subleading_jet_phi": {
        "output": "subleading_jet_phi.png",
        "xlabel": "Subleading jet phi",
        "range": (-math.pi, math.pi),
        "bins": None,
    },
    "dijet_mass": {
        "output": "dijet_mass.png",
        "xlabel": "Dijet mass [GeV]",
        "range": (0,200),
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
    "dijet_rapidity": {
        "output": "dijet_rapidity.png",
        "xlabel": "Dijet rapidity",
        "range": None,
        "bins": None,
    },
    "dijet_phi": {
        "output": "dijet_phi.png",
        "xlabel": "Dijet phi",
        "range": None,
        "bins": None,
    },
    "dijet_delta_eta": {
        "output": "dijet_delta_eta.png",
        "xlabel": "Dijet |Delta eta|",
        "range": None,
        "bins": None,
    },
    "dijet_delta_phi": {
        "output": "dijet_delta_phi.png",
        "xlabel": "Dijet |Delta phi|",
        "range": (0, math.pi),
        "bins": None,
    },
    "dijet_delta_r": {
        "output": "dijet_delta_r.png",
        "xlabel": "Dijet Delta R",
        "range": None,
        "bins": None,
    },
}

COLORS = {
    "superchic": "#0072B2",
    "fpmc": "#D55E00",
    "madgraph": "#E69F00",
}

JET_TYPE_MASS_XLABELS = {
    "b": "$m_{b\\bar{b}}$ [GeV]",
    "c": "$m_{c\\bar{c}}$ [GeV]",
    "q": "$m_{q\\bar{q}}$ [GeV]",
    "g": "$m_{gg}$ [GeV]",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot weighted generator Delphes jet dijet observables."
    )
    parser.add_argument("process", help="Process name from the generator process config")
    parser.add_argument("--generator", choices=tuple(COLORS), required=True)
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis")
    parser.add_argument("--stacked", action="store_true", help="Stack weighted sample histograms")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Main campaign key to use for the selected process. Defaults to its default_campaign.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to analysis/output/sim_jets/<generator>/<process>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of Delphes ROOT files to load.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum TTree entries to load per Delphes ROOT file.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")


def default_output_dir(generator, process):
    return ROOT / "analysis" / "output" / "sim_jets" / generator / process


def resolve_delphes_files(generator, process_name, campaign, max_files):
    campaign_name, _ = generation_campaign_config(generator, process_name, campaign)
    stage_root = generation_stage_root(
        generator,
        process_name,
        campaign_name,
        "sim-delphes",
    )
    root_dir = stage_root / "root"
    files = sorted(root_dir.glob("*.root"), key=natural_key)
    if not files:
        raise RuntimeError(f"No Delphes ROOT files found in {root_dir}")
    if max_files is not None:
        files = files[:max_files]
    label_suffix = campaign_name
    if stage_root.name != campaign_name:
        label_suffix = f"{campaign_name}/{stage_root.name}"
    return campaign_name, label_suffix, files


def resolve_lhe_files(generator, process_name, campaign_name):
    event_dir = (
        generation_campaign_root(generator, process_name, campaign_name)
        / generation_config(generator)["generation_dir"]
        / "evrecs"
    )
    return discover_event_files(event_dir, None)


def file_init(filename):
    import pylhe

    return pylhe.LHEFile.fromfile(filename, generator=True, with_attributes=False).init


def sample_cross_section_fb(generator, process_config, process_name, campaign_name):
    configured = process_config.get("xsec_fb")
    if configured is not None:
        xsec_fb = float(configured)
        source = "config"
    else:
        inits = [
            file_init(filename)
            for filename in resolve_lhe_files(generator, process_name, campaign_name)
        ]
        xsecs_pb = [
            sum(float(proc.xSection) for proc in init.procInfo) for init in inits
        ]
        if not xsecs_pb or not all(xsec > 0.0 for xsec in xsecs_pb):
            raise RuntimeError(
                f"{process_name}_{campaign_name} has no positive configured or LHE cross section"
            )
        xsec_fb = 1000.0 * float(sum(xsecs_pb) / len(xsecs_pb))
        source = "LHE"
    if xsec_fb <= 0.0:
        raise RuntimeError(f"{process_name}_{campaign_name} cross section must be > 0")
    return xsec_fb, source


def empty_observables(np):
    return {variable: np.empty(0, dtype=np.float64) for variable in PLOT_VARIABLES}


def load_jets(ak, uproot, input_file, tree_name, collection, max_events):
    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
        n_entries = int(tree.num_entries)
        n_loaded = min(n_entries, max_events) if max_events is not None else n_entries
        required = [
            analyzer.branch_name(collection, field)
            for field in ("PT", "Eta", "Phi", "Mass")
        ]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {input_file}: {', '.join(missing)}")
        arrays = tree.arrays(required, entry_stop=n_loaded, library="ak")
        pt = arrays[analyzer.branch_name(collection, "PT")]
        eta = arrays[analyzer.branch_name(collection, "Eta")]
        n_jets = ak.to_numpy(ak.sum((pt > 15) & (abs(eta) < 3.0), axis=1))

    return {
        "pt": ak.values_astype(arrays[analyzer.branch_name(collection, "PT")], "float64"),
        "eta": ak.values_astype(arrays[analyzer.branch_name(collection, "Eta")], "float64"),
        "phi": ak.values_astype(arrays[analyzer.branch_name(collection, "Phi")], "float64"),
        "mass": ak.values_astype(arrays[analyzer.branch_name(collection, "Mass")], "float64"),
        "n_jets": n_jets,
        "n_generated": n_loaded,
    }


def build_observables(np, table, selected):
    delta_eta = np.abs(table["jet1_eta"] - table["jet2_eta"])
    delta_phi = np.abs(
        np.arctan2(
            np.sin(table["jet1_phi"] - table["jet2_phi"]),
            np.cos(table["jet1_phi"] - table["jet2_phi"]),
        )
    )
    delta_r = np.hypot(delta_eta, delta_phi)
    return {
        "leading_jet_pt": table["jet1_pt"][selected],
        "subleading_jet_pt": table["jet2_pt"][selected],
        "leading_jet_pt_over_mjj": table["jet1_pt_over_mjj"][selected],
        "subleading_jet_pt_over_mjj": table["jet2_pt_over_mjj"][selected],
        "leading_jet_eta": table["jet1_eta"][selected],
        "subleading_jet_eta": table["jet2_eta"][selected],
        "leading_jet_phi": table["jet1_phi"][selected],
        "subleading_jet_phi": table["jet2_phi"][selected],
        "dijet_mass": table["dijet_mass"][selected],
        "dijet_pt": table["dijet_pt"][selected],
        "dijet_eta": table["dijet_eta"][selected],
        "dijet_rapidity": table["dijet_rapidity"][selected],
        "dijet_phi": table["dijet_phi"][selected],
        "dijet_delta_eta": delta_eta[selected],
        "dijet_delta_phi": delta_phi[selected],
        "dijet_delta_r": delta_r[selected],
    }


def concatenate_observables(np, observable_sets):
    if not observable_sets:
        return empty_observables(np)
    return {
        key: np.concatenate([observables[key] for observables in observable_sets])
        for key in observable_sets[0]
    }


def sample_label(generator, process_name, label_suffix):
    return f"{generator} {process_name} {label_suffix}"


def load_sample(
    ak,
    np,
    uproot,
    generator,
    process_name,
    process_config,
    campaign_name,
    label_suffix,
    files,
    args,
):
    label = sample_label(generator, process_name, label_suffix)
    observable_sets = []
    multiplicities = []
    n_generated = 0
    n_passed = 0

    print(f"Loading {label} from {len(files)} file(s)...", flush=True)
    for file_index, filename in enumerate(files, start=1):
        print(f"  Reading file {file_index}/{len(files)}: {filename.name}", flush=True)
        jets = load_jets(ak, uproot, filename, args.tree, args.collection, args.max_events)
        n_generated += int(jets["n_generated"])
        multiplicities.append(np.minimum(jets["n_jets"], MAX_MULTIPLICITY_BIN))
        try:
            table = analyzer.build_dijets(ak, np, jets, filename, args.collection)
        except RuntimeError as exc:
            print(f"  Warning: {exc}")
            continue
        selected = analyzer.make_selection(np, table, min_jet_pt=MIN_JET_PT)
        n_passed += int(np.sum(selected))
        observable_sets.append(build_observables(np, table, selected))

    if n_generated <= 0:
        raise RuntimeError(f"{label} has zero generated events")
    if n_passed <= 0:
        raise RuntimeError(f"{label} has no events passing two jets with pT >= {MIN_JET_PT:g} GeV")

    xsec_fb, xsec_source = sample_cross_section_fb(
        generator,
        process_config,
        process_name,
        campaign_name,
    )
    event_weight = (
        xsec_fb
        * LUMI_FB
        * float(process_config.get("weight", 1.0))
        / float(n_generated)
    )
    efficiency = 100.0 * n_passed / n_generated
    print(
        f"{label}: files={len(files)}, generated={n_generated}, "
        f"two_jet_pt20={n_passed}/{n_generated} ({efficiency:.2f}%), "
        f"xsec={xsec_fb:.8g} fb ({xsec_source}), event_weight={event_weight:.8g}, "
        f"yield={event_weight * n_passed:.8g}"
    )
    return {
        "name": f"{generator}:{process_name}",
        "campaign": campaign_name,
        "label": label,
        "observables": {
            **concatenate_observables(np, observable_sets),
            "jet_multiplicity": np.concatenate(multiplicities),
        },
        "event_weight": event_weight,
        "n_generated": n_generated,
        "n_selected": n_passed,
    }


def process_plot_variables(process_config):
    plot_variables = {key: dict(options) for key, options in PLOT_VARIABLES.items()}
    plot_variables["dijet_mass"]["xlabel"] = JET_TYPE_MASS_XLABELS.get(
        process_config.get("jet_type"),
        "Dijet mass [GeV]",
    )
    return plot_variables


def main():
    args = parse_args()
    validate_args(args)
    analyzer.ensure_analysis_runtime(Path(__file__), sys.argv[1:])
    ak, np, plt, uproot = analyzer.import_libraries()
    process_config = generation_process_config(args.generator, args.process)
    campaign_name, label_suffix, files = resolve_delphes_files(
        args.generator,
        args.process,
        args.campaign,
        args.max_files,
    )
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else default_output_dir(args.generator, args.process)
    )

    sample = load_sample(
        ak,
        np,
        uproot,
        args.generator,
        args.process,
        process_config,
        campaign_name,
        label_suffix,
        files,
        args,
    )
    labels = {sample["name"]: sample["label"]}
    colors = {sample["name"]: COLORS[args.generator]}
    written = analyzer.write_plots(
        np,
        plt,
        [sample],
        process_plot_variables(process_config),
        output_dir,
        args.bins,
        args.log_y,
        args.stacked,
        labels,
        colors,
    )
    print(f"Wrote {written} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
