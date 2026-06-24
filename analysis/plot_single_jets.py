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
        "range": (0,1),
        "bins": None,
    },
    "jet2_pt_over_mjj_before_pps": {
        "output": "jet2_pt_over_mjj_before_pps.png",
        "xlabel": "Jet 2 pT / dijet mass",
        "range": (0,1),
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
        "range": (0,1),
        "bins": None,
    },
    "jet2_pt_over_mjj_after_pps": {
        "output": "jet2_pt_over_mjj_after_pps.png",
        "xlabel": "Jet 2 pT / dijet mass",
        "range": (0,1),
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
    "qcd_gg",
    "qcd_qq",
    "qcd_bb",
    "qcd_cc",
    "qed_bb",
    "qed_cc",
    "h_bb",
    "h_cc",
)


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
        description="Plot weighted single-jet observables for bb or cc tagged Delphes dijet selections."
    )
    parser.add_argument("--flavor", choices=("bb", "cc"), required=True, help="Tag target")
    parser.add_argument(
        "--include-light-qcd",
        action="store_true",
        help="Include qcd_qq and qcd_gg with light-to-heavy mistag weights.",
    )
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to analysis/output/single_jets_<flavor>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file with pps.xi_ranges.",
    )
    parser.add_argument(
        "--proton-source",
        choices=("auto", "delphes", "pythia"),
        default="auto",
        help="Deprecated. Protons are always read from Pythia HepMC.",
    )
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


def resolve_inputs(process_name, campaign_name):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    delphes_file = campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root"
    pythia_file = campaign_dir / "GEN-pythia" / f"{process_name}_{campaign}.hepmc"
    return campaign_dir, campaign, delphes_file, pythia_file


def load_pps_config(path):
    config = load_yaml(path)
    pps = config.get("pps", {})

    xi_ranges = []
    for station, bounds in (pps.get("xi_ranges") or {}).items():
        if len(bounds) != 2:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        xi_ranges.append((str(station), float(bounds[0]), float(bounds[1])))
    if not xi_ranges:
        raise RuntimeError(f"No PPS xi ranges found in {path}")

    return {
        "xi_ranges": xi_ranges,
    }


def load_dijet_selection(ak, np, uproot, input_file, tree_name, collection):
    print(f"Reading Delphes jets: {input_file}")
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

    has_two_jets_ak = ak.num(pt) >= 2
    has_two_jets = ak.to_numpy(has_two_jets_ak)
    selected_event_indices = np.nonzero(has_two_jets)[0]
    n_selected = int(selected_event_indices.size)
    if n_selected == 0:
        raise RuntimeError(f"No events with at least two {collection} jets in {input_file}")

    selected_pt = pt[has_two_jets_ak]
    selected_eta = eta[has_two_jets_ak]
    selected_phi = phi[has_two_jets_ak]
    selected_mass = mass[has_two_jets_ak]
    jets = ak.zip(
        {
            "pt": selected_pt,
            "eta": selected_eta,
            "phi": selected_phi,
            "mass": selected_mass,
        },
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    jet1 = jets[order][:, 0]
    jet2 = jets[order][:, 1]
    dijet = jet1 + jet2
    dijet_mass = ak.to_numpy(dijet.mass)
    valid_mass = dijet_mass > 0.0
    jet1_pt = ak.to_numpy(jet1.pt)
    jet2_pt = ak.to_numpy(jet2.pt)
    jet1_pt_over_mjj = np.full(jet1_pt.shape, np.nan, dtype=np.float64)
    jet2_pt_over_mjj = np.full(jet2_pt.shape, np.nan, dtype=np.float64)
    jet1_pt_over_mjj[valid_mass] = jet1_pt[valid_mass] / dijet_mass[valid_mass]
    jet2_pt_over_mjj[valid_mass] = jet2_pt[valid_mass] / dijet_mass[valid_mass]

    return {
        "selected_event_indices": selected_event_indices,
        "jet1_pt": jet1_pt,
        "jet2_pt": jet2_pt,
        "jet1_pt_over_mjj": jet1_pt_over_mjj,
        "jet2_pt_over_mjj": jet2_pt_over_mjj,
        "jet1_eta": ak.to_numpy(jet1.eta),
        "jet2_eta": ak.to_numpy(jet2.eta),
        "jet1_phi": ak.to_numpy(jet1.phi),
        "jet2_phi": ak.to_numpy(jet2.phi),
    }, n_generated, n_selected


def parse_hepmc_protons(np, input_file, selected_event_indices):
    print(f"Reading Pythia protons: {input_file}")
    selected = set(int(index) for index in selected_event_indices)
    left = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    right = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    matched_events = []
    left_xi = []
    right_xi = []

    event_index = -1
    beam_pos = None
    beam_neg = None
    left_energy = None
    left_abs_pz = -1.0
    right_energy = None
    right_abs_pz = -1.0

    def finish_event():
        if event_index not in selected:
            return
        if beam_pos is None or beam_neg is None or left_energy is None or right_energy is None:
            return
        matched_events.append(event_index)
        left_xi.append((beam_neg - left_energy) / beam_neg)
        right_xi.append((beam_pos - right_energy) / beam_pos)

    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            tag = line[:2]
            if tag == "E ":
                if event_index >= 0:
                    finish_event()
                    if event_index >= int(selected_event_indices[-1]):
                        break
                event_index += 1
                beam_pos = None
                beam_neg = None
                left_energy = None
                left_abs_pz = -1.0
                right_energy = None
                right_abs_pz = -1.0
            elif tag == "P ":
                fields = line.split()
                if len(fields) < 10:
                    continue
                pdg_id = int(fields[3])
                if pdg_id != 2212:
                    continue
                pz = float(fields[6])
                energy = float(fields[7])
                status = int(fields[9])
                if status == 4:
                    if pz > 0.0:
                        beam_pos = energy
                    elif pz < 0.0:
                        beam_neg = energy
                elif status == 1:
                    abs_pz = abs(pz)
                    if pz < 0.0 and abs_pz > left_abs_pz:
                        left_energy = energy
                        left_abs_pz = abs_pz
                    elif pz > 0.0 and abs_pz > right_abs_pz:
                        right_energy = energy
                        right_abs_pz = abs_pz
        if event_index >= 0:
            finish_event()

    if matched_events:
        matched_events = np.asarray(matched_events, dtype=selected_event_indices.dtype)
        output_indices = np.searchsorted(selected_event_indices, matched_events)
        matched = output_indices < selected_event_indices.size
        matched[matched] = selected_event_indices[output_indices[matched]] == matched_events[matched]
        output_indices = output_indices[matched]
        left[output_indices] = np.asarray(left_xi, dtype=np.float64)[matched]
        right[output_indices] = np.asarray(right_xi, dtype=np.float64)[matched]

    return proton_observables(np, left, right, "pythia")


def proton_observables(np, xi_left, xi_right, source):
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    valid = np.isfinite(xi_left) & np.isfinite(xi_right) & (xi_left > 0.0) & (xi_right > 0.0)
    return {
        "source": source,
        "xi_left": xi_left,
        "xi_right": xi_right,
        "valid": valid,
    }


def load_protons(np, pythia_file, selected_event_indices):
    if not pythia_file.is_file():
        raise RuntimeError(f"Missing Pythia HepMC input for proton pairs: {pythia_file}")
    return parse_hepmc_protons(np, pythia_file, selected_event_indices)


def passes_pps(np, xi, xi_ranges):
    passed = np.zeros(np.asarray(xi).shape, dtype=bool)
    for _station, xi_min, xi_max in xi_ranges:
        passed |= (xi >= xi_min) & (xi < xi_max)
    return passed


def build_observables(np, dijets, protons, pps_config):
    valid = protons["valid"]
    pps = valid & passes_pps(np, protons["xi_left"], pps_config["xi_ranges"])
    pps &= passes_pps(np, protons["xi_right"], pps_config["xi_ranges"])

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
        observables[f"{key}_before_pps"] = dijets[key][valid]
        observables[f"{key}_after_pps"] = dijets[key][pps]

    return {
        "observables": observables,
        "n_valid_proton_pairs": int(np.sum(valid)),
        "n_pps": int(np.sum(pps)),
        "proton_source": protons["source"],
    }


def finite_values(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(np, datasets):
    values = [np.asarray(dataset, dtype=np.float64) for dataset in datasets if dataset.size]
    if not values:
        return None
    lo = min(float(np.min(dataset)) for dataset in values)
    hi = max(float(np.max(dataset)) for dataset in values)
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def default_output_dir(flavor, include_light_qcd):
    suffix = f"single_jets_{flavor}"
    if include_light_qcd:
        suffix += "_with_light_qcd"
    return ROOT / "analysis" / "output" / suffix


def read_samples(ak, np, uproot, processes, parameters, pps_config, args):
    samples = []
    skipped = []
    for process_name in selected_processes(processes, args.include_light_qcd):
        process = processes[process_name]
        _campaign_dir, campaign, delphes_file, pythia_file = resolve_inputs(process_name, args.campaign)
        if not delphes_file.is_file():
            skipped.append((process_name, delphes_file, "missing Delphes file"))
            continue

        source_flavor = process_flavor(process)
        tag = tag_weight(parameters, args.flavor, source_flavor)
        try:
            dijets, n_generated, n_selected = load_dijet_selection(
                ak, np, uproot, delphes_file, args.tree, args.collection
            )
            protons = load_protons(
                np,
                pythia_file,
                dijets["selected_event_indices"],
            )
            results = build_observables(np, dijets, protons, pps_config)
        except RuntimeError as exc:
            skipped.append((process_name, delphes_file, str(exc)))
            continue
        if n_generated <= 0:
            raise RuntimeError(f"{delphes_file} has zero generated events")

        event_weight = float(process["xsec_fb"]) * LUMI_FB * tag / float(n_generated)
        expected_yield = event_weight * results["n_valid_proton_pairs"]
        samples.append(
            {
                "name": process_name,
                "campaign": campaign,
                "observables": results["observables"],
                "event_weight": event_weight,
                "expected_yield": expected_yield,
                "tag_weight": tag,
                "n_generated": n_generated,
                "n_selected": n_selected,
                "n_valid_proton_pairs": results["n_valid_proton_pairs"],
                "n_pps": results["n_pps"],
                "proton_source": results["proton_source"],
            }
        )
        print(
            f"{process_name}_{campaign}: generated={n_generated}, "
            f"two_jet={n_selected}, proton_pairs={results['n_valid_proton_pairs']}, "
            f"pps%={100 * results['n_pps'] / results['n_valid_proton_pairs']:.2f}%,"
            f"pps={results['n_pps']}, "
            f"proton_source={results['proton_source']}, tag_weight={tag:.6g}, "
            f"event_weight={event_weight:.6g}, expected_yield={expected_yield:.6g}"
        )

    for process_name, input_file, reason in skipped:
        print(f"Warning: skipping {process_name}: {reason} ({input_file})")

    if not samples:
        raise RuntimeError("No usable inputs found for the selected processes")

    for sample in samples:
        for key in PLOT_VARIABLES:
            sample["observables"][key] = finite_values(np, sample["observables"][key])
    return samples


def print_range_yields(histograms):
    lo, hi = histograms["value_range"]
    variable = histograms["variable"]
    total = 0.0
    print(f"Yields for {histograms['variable']} within range [{lo:.6g}, {hi:.6g}]:")
    for row in histograms["rows"]:
        sample = row["sample"]
        in_range = row["entries"]
        range_yield = sample["event_weight"] * in_range
        total += range_yield
        total_values = sample["observables"][variable].size
        percentage = 100 * in_range / total_values if total_values else 0.0
        print(
            f"  {sample['name']}_{sample['campaign']}: in_range={int(in_range)}, "
            f"percentage={percentage:.2f}%, range_yield={range_yield:.6g}"
        )
    print(f"  total range_yield={total:.6g}")


def normalized_output_name(output_name):
    path = Path(output_name)
    return f"{path.stem}_normalized{path.suffix}"


def make_histograms(np, samples, variable, options, default_bins):
    bins = options["bins"] or default_bins
    datasets = [sample["observables"][variable] for sample in samples]
    value_range = options["range"] or histogram_range(np, datasets)
    if value_range is None:
        print(f"Warning: no finite values for {variable}; skipping")
        return None

    rows = []
    edges = None
    for sample in samples:
        counts, edges = np.histogram(sample["observables"][variable], bins=bins, range=value_range)
        rows.append(
            {
                "sample": sample,
                "counts": counts.astype(np.float64),
                "entries": int(np.sum(counts)),
            }
        )
    if not any(row["entries"] for row in rows):
        print(f"Warning: no finite values for {variable}; skipping")
        return None

    return {
        "variable": variable,
        "options": options,
        "value_range": value_range,
        "edges": edges,
        "rows": rows,
    }


def plot_histograms(np, plt, histograms, output_dir, log_y, normalized=False):
    options = histograms["options"]
    value_range = histograms["value_range"]
    edges = histograms["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    plot_rows = [row for row in histograms["rows"] if row["entries"]]
    labels = []
    y_values = []
    y_errors = []
    for row in plot_rows:
        sample = row["sample"]
        counts = row["counts"]
        if normalized:
            y = counts / row["entries"]
            yerr = np.sqrt(counts) / row["entries"]
            labels.append(PROCESS_LABELS.get(sample["name"], sample["name"]))
        else:
            y = counts * sample["event_weight"]
            yerr = np.sqrt(counts) * sample["event_weight"]
            sample_yield = sample["event_weight"] * row["entries"]
            labels.append(f"{PROCESS_LABELS.get(sample['name'], sample['name'])} ({sample_yield:.4g})")
        y_values.append(y)
        y_errors.append(yerr)
    colors = [COLORS.get(row["sample"]["name"], None) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    artists = []
    for y, yerr, label, color in zip(y_values, y_errors, labels, colors):
        artists.append(ax.stairs(y, edges, label=label, color=color, linewidth=1.5))
        nonzero = y > 0.0
        ax.errorbar(
            centers[nonzero],
            y[nonzero],
            yerr=yerr[nonzero],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=1.5,
        )
    ax.set_xlabel(options["xlabel"])
    ax.set_xlim(value_range)
    if normalized:
        ax.set_ylabel("Normalized events / bin")
    else:
        ax.set_ylabel(options.get("ylabel", "Expected events / bin"))
    if log_y:
        ax.set_yscale("log")
        positive = []
        for y in y_values:
            positive.extend(y[y > 0.0])
        if positive:
            ax.set_ylim(bottom=float(np.min(positive)) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_name = normalized_output_name(options["output"]) if normalized else options["output"]
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    return bool(artists)


def write_plots(np, plt, samples, output_dir, bins, log_y):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for variable, options in PLOT_VARIABLES.items():
        histograms = make_histograms(np, samples, variable, options, bins)
        if histograms is None:
            continue
        if options["range"] is not None:
            print_range_yields(histograms)
        if plot_histograms(np, plt, histograms, output_dir, log_y):
            written += 1
        if plot_histograms(np, plt, histograms, output_dir, log_y, normalized=True):
            written += 1
    return written


def main():
    args = parse_args()
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.proton_source != "auto":
        print("Warning: --proton-source is deprecated; reading protons from Pythia HepMC")

    ensure_delphes_python_runtime()
    ak, np, plt, uproot = import_libraries()
    processes = load_yaml(ROOT / "processes.yaml")
    parameters = load_yaml(ROOT / "parameters.yaml")
    pps_config = load_pps_config(resolve_path(args.pps_config, base=ROOT))
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else default_output_dir(args.flavor, args.include_light_qcd)
    )

    samples = read_samples(ak, np, uproot, processes, parameters, pps_config, args)
    written = write_plots(np, plt, samples, output_dir, args.bins, args.log_y)
    print(f"Wrote {written} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
