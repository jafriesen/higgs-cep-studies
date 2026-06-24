#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


LUMI_FB = 3000.0
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
    "qcd_gg": "QCD gg",
    "qcd_qq": "QCD qq",
    "qcd_bb": "QCD bb",
    "qcd_cc": "QCD cc",
    "qed_bb": "QED bb",
    "qed_cc": "QED cc",
    "h_bb": "H->bb",
    "h_cc": "H->cc",
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


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan TMVA score cuts using smeared associated-proton mass significance."
    )
    parser.add_argument("--flavor", choices=("bb", "cc"), required=True, help="Signal flavor")
    parser.add_argument(
        "--include-light-qcd",
        action="store_true",
        help="Use the MVA output and samples including qcd_qq and qcd_gg.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="MVA output/input directory. Defaults to analysis/MVA/output/mva_<flavor>.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for xi smearing")
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file with beam.sqrt_s_gev, pps.xi_ranges, pps.xi_res, and random.seed.",
    )
    parser.add_argument(
        "--proton-source",
        choices=("auto", "delphes", "pythia"),
        default="auto",
        help="Source for associated proton information.",
    )
    parser.add_argument(
        "--mass-range",
        default="100,150",
        help="Comma-separated smeared pp mass range in GeV.",
    )
    parser.add_argument("--mass-bin-width", type=float, default=1.0, help="Mass bin width in GeV")
    parser.add_argument("--score-step", type=float, default=0.01, help="TMVA score scan step")
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
    import vector
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vector.register_awkward()
    return ak, np, plt, uproot, yaml


def parse_range(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise RuntimeError(f"Invalid --mass-range '{value}', expected low,high")
    lo, hi = (float(parts[0]), float(parts[1]))
    if not hi > lo:
        raise RuntimeError("--mass-range high value must be larger than low value")
    return lo, hi


def default_output_dir(flavor, include_light_qcd):
    suffix = f"mva_{flavor}"
    if include_light_qcd:
        suffix += "_with_light_qcd"
    return ROOT / "analysis" / "MVA" / "output" / suffix


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


def resolve_inputs(process_name, campaign_name):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    delphes_file = campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root"
    pythia_file = campaign_dir / "GEN-pythia" / f"{process_name}_{campaign}.hepmc"
    return campaign, delphes_file, pythia_file


def load_pps_config(path):
    config = load_yaml(path)
    beam = config.get("beam", {})
    pps = config.get("pps", {})
    random = config.get("random", {})

    xi_ranges = []
    for station, bounds in (pps.get("xi_ranges") or {}).items():
        if len(bounds) != 2:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        xi_ranges.append((str(station), float(bounds[0]), float(bounds[1])))
    if not xi_ranges:
        raise RuntimeError(f"No PPS xi ranges found in {path}")

    return {
        "sqrt_s": float(beam.get("sqrt_s_gev", 14000.0)),
        "xi_ranges": xi_ranges,
        "xi_res": float(pps.get("xi_res", 0.0)),
        "seed": int(random.get("seed", 12345)),
    }


def event_key(name):
    return name.split(";")[0]


def load_selected_event_indices(ak, np, uproot, input_file, tree_name, collection):
    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
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
    selected_event_indices = np.nonzero(ak.to_numpy(has_two_jets_ak))[0]
    if selected_event_indices.size == 0:
        raise RuntimeError(f"No events with at least two {collection} jets in {input_file}")

    jets = ak.zip(
        {
            "pt": pt[has_two_jets_ak],
            "eta": eta[has_two_jets_ak],
            "phi": phi[has_two_jets_ak],
            "mass": mass[has_two_jets_ak],
        },
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    leading = jets[order][:, 0]
    subleading = jets[order][:, 1]
    dijet = leading + subleading

    deta = leading.eta - subleading.eta
    dphi = (leading.phi - subleading.phi + np.pi) % (2.0 * np.pi) - np.pi
    delta_r = np.sqrt(deta * deta + dphi * dphi)
    mjj = ak.to_numpy(dijet.mass)
    features = np.column_stack(
        [
            ak.to_numpy(leading.pt) / mjj,
            ak.to_numpy(subleading.pt) / mjj,
            ak.to_numpy(leading.eta),
            ak.to_numpy(subleading.eta),
            ak.to_numpy(delta_r),
            ak.to_numpy(dijet.pt),
            ak.to_numpy(dijet.eta),
        ]
    )
    finite = np.all(np.isfinite(features), axis=1) & np.isfinite(mjj) & (mjj > 0.0)
    return selected_event_indices[finite]


def proton_observables(np, xi_left, xi_right, sqrt_s, source):
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    valid = np.isfinite(xi_left) & np.isfinite(xi_right) & (xi_left > 0.0) & (xi_right > 0.0)
    return {
        "source": source,
        "xi_left": xi_left,
        "xi_right": xi_right,
        "valid": valid,
    }


def read_delphes_protons(np, uproot, input_file, selected_event_indices, sqrt_s):
    with uproot.open(input_file) as root_file:
        trees = {event_key(key): key for key in root_file.keys()}
        if "Protons" not in trees:
            return None
        tree = root_file[trees["Protons"]]
        keys = set(tree.keys())
        required = {"event_id", "side", "xi"}
        if not required.issubset(keys):
            return None
        arrays = tree.arrays(["event_id", "side", "xi"], library="np")

    by_event = {}
    for event_id, side, xi in zip(arrays["event_id"], arrays["side"], arrays["xi"]):
        event = by_event.setdefault(int(event_id), {})
        if int(side) < 0:
            event["left"] = float(xi)
        elif int(side) > 0:
            event["right"] = float(xi)

    left = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    right = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    for idx, event_index in enumerate(selected_event_indices):
        event = by_event.get(int(event_index))
        if not event:
            continue
        left[idx] = event.get("left", np.nan)
        right[idx] = event.get("right", np.nan)

    return proton_observables(np, left, right, sqrt_s, "delphes")


def parse_hepmc_protons(np, input_file, selected_event_indices, sqrt_s):
    selected = set(int(index) for index in selected_event_indices)
    left = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    right = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    output_index = {int(event_index): idx for idx, event_index in enumerate(selected_event_indices)}

    event_index = -1
    beam_pos = None
    beam_neg = None
    protons = []

    def finish_event():
        if event_index not in selected:
            return
        if beam_pos is None or beam_neg is None:
            return
        left_candidates = [p for p in protons if p["pz"] < 0.0]
        right_candidates = [p for p in protons if p["pz"] > 0.0]
        if not left_candidates or not right_candidates:
            return
        left_proton = max(left_candidates, key=lambda p: abs(p["pz"]))
        right_proton = max(right_candidates, key=lambda p: abs(p["pz"]))
        idx = output_index[event_index]
        left[idx] = (beam_neg - left_proton["energy"]) / beam_neg
        right[idx] = (beam_pos - right_proton["energy"]) / beam_pos

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
                protons = []
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
                    protons.append({"pz": pz, "energy": energy})
        if event_index >= 0:
            finish_event()

    return proton_observables(np, left, right, sqrt_s, "pythia")


def load_protons(np, uproot, delphes_file, pythia_file, selected_event_indices, pps_config, source):
    if source in ("auto", "delphes"):
        protons = read_delphes_protons(np, uproot, delphes_file, selected_event_indices, pps_config["sqrt_s"])
        if protons is not None:
            return protons
        if source == "delphes":
            raise RuntimeError(f"No usable Protons tree found in {delphes_file}")

    if not pythia_file.is_file():
        raise RuntimeError(f"Missing Pythia HepMC input for proton pairs: {pythia_file}")
    return parse_hepmc_protons(np, pythia_file, selected_event_indices, pps_config["sqrt_s"])


def passes_pps(np, xi, xi_ranges):
    passed = np.zeros(np.asarray(xi).shape, dtype=bool)
    for _station, xi_min, xi_max in xi_ranges:
        passed |= (xi >= xi_min) & (xi < xi_max)
    return passed


def load_mva_cache(np, output_dir):
    dataset_path = output_dir / "dataset.npz"
    scores_path = output_dir / "scores.npz"
    if not dataset_path.is_file():
        raise RuntimeError(f"Missing MVA dataset cache: {dataset_path}")
    if not scores_path.is_file():
        raise RuntimeError(f"Missing MVA scores cache: {scores_path}")
    dataset = np.load(dataset_path, allow_pickle=False)
    scores = np.load(scores_path, allow_pickle=False)
    if "all" not in scores.files:
        raise RuntimeError(f"{scores_path} does not contain scores['all']; rerun run_dijet_mva.py")
    n_rows = dataset["y"].shape[0]
    if scores["all"].shape[0] != n_rows:
        raise RuntimeError(
            f"MVA row mismatch: dataset has {n_rows} rows, scores['all'] has {scores['all'].shape[0]}"
        )
    return dataset, scores


def expected_process_counts(np, dataset):
    counts = {}
    for process_name in PROCESS_ORDER:
        count = int(np.sum(dataset["process"] == process_name))
        if count:
            counts[process_name] = count
    return counts


def build_pps_rows(ak, np, uproot, processes, dataset, scores, pps_config, args):
    rows = []
    skipped = []
    counts = expected_process_counts(np, dataset)
    offset = 0
    rng = np.random.default_rng(args.seed)
    signal_process = f"h_{args.flavor}"

    for process_name in selected_processes(processes, args.include_light_qcd):
        expected_count = counts.get(process_name, 0)
        cached_process = dataset["process"][offset : offset + expected_count]
        if expected_count == 0 or not np.all(cached_process == process_name):
            raise RuntimeError(f"MVA cache row order/count mismatch for {process_name}")

        campaign, delphes_file, pythia_file = resolve_inputs(process_name, args.campaign)
        if not delphes_file.is_file():
            skipped.append({"process": process_name, "input": str(delphes_file), "reason": "missing Delphes file"})
            offset += expected_count
            continue

        try:
            selected_event_indices = load_selected_event_indices(
                ak, np, uproot, delphes_file, args.tree, args.collection
            )
            if selected_event_indices.shape[0] != expected_count:
                raise RuntimeError(
                    f"selected row count {selected_event_indices.shape[0]} does not match MVA cache {expected_count}"
                )
            protons = load_protons(
                np,
                uproot,
                delphes_file,
                pythia_file,
                selected_event_indices,
                pps_config,
                args.proton_source,
            )
        except RuntimeError as exc:
            skipped.append({"process": process_name, "input": str(delphes_file), "reason": str(exc)})
            offset += expected_count
            continue

        xi_left = protons["xi_left"]
        xi_right = protons["xi_right"]
        pps = protons["valid"] & passes_pps(np, xi_left, pps_config["xi_ranges"])
        pps &= passes_pps(np, xi_right, pps_config["xi_ranges"])
        xi_left_pps = xi_left[pps]
        xi_right_pps = xi_right[pps]
        if pps_config["xi_res"] > 0.0:
            xi_left_smeared = rng.normal(xi_left_pps, pps_config["xi_res"])
            xi_right_smeared = rng.normal(xi_right_pps, pps_config["xi_res"])
        else:
            xi_left_smeared = xi_left_pps.copy()
            xi_right_smeared = xi_right_pps.copy()
        smeared_valid = (xi_left_smeared > 0.0) & (xi_right_smeared > 0.0)
        pps_indices = np.nonzero(pps)[0][smeared_valid]
        mx_smeared = (
            np.sqrt(xi_left_smeared[smeared_valid] * xi_right_smeared[smeared_valid])
            * pps_config["sqrt_s"]
        )
        row_slice = slice(offset, offset + expected_count)
        rows.append(
            {
                "process": process_name,
                "campaign": campaign,
                "proton_source": protons["source"],
                "mass": mx_smeared,
                "score": scores["all"][row_slice][pps_indices],
                "weight": dataset["physical_weight"][row_slice][pps_indices],
                "label": dataset["y"][row_slice][pps_indices],
                "is_signal": process_name == signal_process,
                "n_cached": expected_count,
                "n_valid_proton_pairs": int(np.sum(protons["valid"])),
                "n_pps": int(np.sum(pps)),
                "n_smeared_valid": int(mx_smeared.shape[0]),
            }
        )
        print(
            f"{process_name}_{campaign}: cache={expected_count}, valid_protons={np.sum(protons['valid'])}, "
            f"pps={np.sum(pps)}, smeared_valid={mx_smeared.shape[0]}, proton_source={protons['source']}"
        )
        offset += expected_count

    if offset != dataset["y"].shape[0]:
        raise RuntimeError(f"MVA cache row mismatch: consumed {offset}, dataset has {dataset['y'].shape[0]}")
    if not rows:
        raise RuntimeError("No usable proton-matched rows found")
    return rows, skipped


def concatenate_rows(np, rows):
    return {
        "mass": np.concatenate([row["mass"] for row in rows]),
        "score": np.concatenate([row["score"] for row in rows]),
        "weight": np.concatenate([row["weight"] for row in rows]),
        "label": np.concatenate([row["label"] for row in rows]),
        "process": np.concatenate([np.full(row["mass"].shape, row["process"]) for row in rows]),
    }


def scan_significance(np, mass, score, weight, label, mass_bins, score_cuts):
    results = []
    for cut in score_cuts:
        selected = score >= cut
        sig = selected & (label == 1)
        bkg = selected & (label == 0)
        s_counts, _edges = np.histogram(mass[sig], bins=mass_bins, weights=weight[sig])
        b_counts, _edges = np.histogram(mass[bkg], bins=mass_bins, weights=weight[bkg])
        denom = s_counts + b_counts
        z_bins = np.zeros_like(s_counts, dtype=np.float64)
        nonzero = denom > 0.0
        z_bins[nonzero] = s_counts[nonzero] / np.sqrt(denom[nonzero])
        results.append(
            {
                "score_cut": float(cut),
                "significance": float(np.sqrt(np.sum(z_bins * z_bins))),
                "signal_yield": float(np.sum(s_counts)),
                "background_yield": float(np.sum(b_counts)),
            }
        )
    return results


def write_scan_csv(path, results):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("score_cut", "significance", "signal_yield", "background_yield"),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def plot_scan(plt, path, results, best):
    cuts = [result["score_cut"] for result in results]
    significances = [result["significance"] for result in results]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(cuts, significances, color="#0072B2", linewidth=1.6)
    ax.axvline(best["score_cut"], color="#D55E00", linestyle="--", linewidth=1.2)
    ax.scatter([best["score_cut"]], [best["significance"]], color="#D55E00", zorder=3)
    ax.set_xlabel("TMVA score cut")
    ax.set_ylabel("Combined binned S/sqrt(S+B)")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.05,
        0.95,
        f"best cut = {best['score_cut']:.3g}\nbest Z = {best['significance']:.3g}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mass_before_after(np, plt, path, data, mass_bins, best_cut, log_y=False):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    selections = (
        ("Signal before", (data["label"] == 1), "#0072B2", "-"),
        ("Signal after", (data["label"] == 1) & (data["score"] >= best_cut), "#0072B2", "--"),
        ("Background before", (data["label"] == 0), "#D55E00", "-"),
        ("Background after", (data["label"] == 0) & (data["score"] >= best_cut), "#D55E00", "--"),
    )
    positive_counts = []
    for label, mask, color, linestyle in selections:
        counts, _edges, _patches = ax.hist(
            data["mass"][mask],
            bins=mass_bins,
            weights=data["weight"][mask],
            histtype="step",
            linewidth=1.5,
            linestyle=linestyle,
            color=color,
            label=f"{label} ({np.sum(data['weight'][mask]):.3g})",
        )
        positive_counts.extend(counts[counts > 0.0])
    ax.set_xlabel("$M_X$ [GeV]")
    ax.set_ylabel("Expected events / GeV")
    ax.set_xlim(float(mass_bins[0]), float(mass_bins[-1]))
    if log_y:
        ax.set_yscale("log")
        if positive_counts:
            ax.set_ylim(bottom=float(np.min(positive_counts)) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def binned_significance(np, mass, weight, label, mass_bins):
    sig = label == 1
    bkg = label == 0
    s_counts, _edges = np.histogram(mass[sig], bins=mass_bins, weights=weight[sig])
    b_counts, _edges = np.histogram(mass[bkg], bins=mass_bins, weights=weight[bkg])
    denom = s_counts + b_counts
    z_bins = np.zeros_like(s_counts, dtype=np.float64)
    nonzero = denom > 0.0
    z_bins[nonzero] = s_counts[nonzero] / np.sqrt(denom[nonzero])
    return float(np.sqrt(np.sum(z_bins * z_bins)))


def sorted_processes_by_yield(np, data, mask):
    entries = []
    for process_name in PROCESS_ORDER:
        process_mask = mask & (data["process"] == process_name)
        if not np.any(process_mask):
            continue
        yield_sum = float(np.sum(data["weight"][process_mask]))
        if yield_sum <= 0.0:
            continue
        is_signal = bool(np.any(data["label"][process_mask] == 1))
        entries.append((process_name, yield_sum, is_signal))
    return [
        process_name
        for process_name, _yield_sum, _is_signal in sorted(
            entries,
            key=lambda item: (0 if item[2] else 1, -item[1]),
        )
    ]


def plot_mass_by_process(np, plt, path, data, mass_bins, selection, title=None):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    significance = binned_significance(
        np,
        data["mass"][selection],
        data["weight"][selection],
        data["label"][selection],
        mass_bins,
    )
    for process_name in sorted_processes_by_yield(np, data, selection):
        mask = selection & (data["process"] == process_name)
        linestyle = "-" if np.any(data["label"][mask] == 1) else "--"
        ax.hist(
            data["mass"][mask],
            bins=mass_bins,
            weights=data["weight"][mask],
            histtype="step",
            linewidth=1.5,
            linestyle=linestyle,
            color=COLORS.get(process_name),
            label=f"{PROCESS_LABELS.get(process_name, process_name)} ({np.sum(data['weight'][mask]):.3g})",
        )
    if title:
        ax.set_title(title)
    ax.set_xlabel("$M_X$ [GeV]")
    ax.set_ylabel("Expected events / GeV")
    ax.set_xlim(float(mass_bins[0]), float(mass_bins[-1]))
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none"))
    labels.append(f"Z = {significance:.3g}")
    ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(np, yaml, path, args, pps_config, rows, skipped, best, outputs, data):
    before_sig = data["weight"][data["label"] == 1]
    before_bkg = data["weight"][data["label"] == 0]
    after = data["score"] >= best["score_cut"]
    after_sig = data["weight"][after & (data["label"] == 1)]
    after_bkg = data["weight"][after & (data["label"] == 0)]
    summary = {
        "flavor": args.flavor,
        "include_light_qcd": bool(args.include_light_qcd),
        "pps_config": str(resolve_path(args.pps_config, base=ROOT)),
        "proton_source_requested": args.proton_source,
        "sqrt_s_gev": float(pps_config["sqrt_s"]),
        "xi_res": float(pps_config["xi_res"]),
        "best_score_cut": float(best["score_cut"]),
        "best_significance": float(best["significance"]),
        "best_signal_yield_in_mass_range": float(best["signal_yield"]),
        "best_background_yield_in_mass_range": float(best["background_yield"]),
        "before_signal_yield_all_masses": float(np.sum(before_sig)),
        "before_background_yield_all_masses": float(np.sum(before_bkg)),
        "after_signal_yield_all_masses": float(np.sum(after_sig)),
        "after_background_yield_all_masses": float(np.sum(after_bkg)),
        "samples": [
            {
                "process": row["process"],
                "campaign": row["campaign"],
                "proton_source": row["proton_source"],
                "n_cached": int(row["n_cached"]),
                "n_valid_proton_pairs": int(row["n_valid_proton_pairs"]),
                "n_pps": int(row["n_pps"]),
                "n_smeared_valid": int(row["n_smeared_valid"]),
            }
            for row in rows
        ],
        "skipped": skipped,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def main():
    args = parse_args()
    if args.mass_bin_width <= 0.0:
        raise RuntimeError("--mass-bin-width must be > 0")
    if args.score_step <= 0.0 or args.score_step > 1.0:
        raise RuntimeError("--score-step must be in (0, 1]")

    ensure_delphes_python_runtime()
    ak, np, plt, uproot, yaml = import_libraries()
    processes = load_yaml(ROOT / "processes.yaml")
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else default_output_dir(args.flavor, args.include_light_qcd)
    )
    pps_config = load_pps_config(resolve_path(args.pps_config, base=ROOT))
    dataset, scores = load_mva_cache(np, output_dir)
    rows, skipped = build_pps_rows(ak, np, uproot, processes, dataset, scores, pps_config, args)
    data = concatenate_rows(np, rows)

    mass_lo, mass_hi = parse_range(args.mass_range)
    mass_bins = np.arange(mass_lo, mass_hi + 0.5 * args.mass_bin_width, args.mass_bin_width)
    score_cuts = np.arange(0.0, 1.0 + 0.5 * args.score_step, args.score_step)
    results = scan_significance(
        np,
        data["mass"],
        data["score"],
        data["weight"],
        data["label"],
        mass_bins,
        score_cuts,
    )
    best = max(results, key=lambda result: result["significance"])

    outputs = {
        "scan_csv": output_dir / "pp_mass_significance_scan.csv",
        "summary": output_dir / "pp_mass_significance_summary.yaml",
        "scan_plot": output_dir / "pp_mass_significance_scan.png",
        "mass_before_after": output_dir / "pp_mass_smeared_before_after_tmva.png",
        "mass_before_after_log": output_dir / "pp_mass_smeared_before_after_tmva_log.png",
        "mass_before_by_process": output_dir / "pp_mass_smeared_before_tmva_by_process.png",
        "mass_after_by_process": output_dir / "pp_mass_smeared_after_tmva_by_process.png",
    }
    write_scan_csv(outputs["scan_csv"], results)
    plot_scan(plt, outputs["scan_plot"], results, best)
    plot_mass_before_after(np, plt, outputs["mass_before_after"], data, mass_bins, best["score_cut"])
    plot_mass_before_after(
        np,
        plt,
        outputs["mass_before_after_log"],
        data,
        mass_bins,
        best["score_cut"],
        log_y=True,
    )
    plot_mass_by_process(
        np,
        plt,
        outputs["mass_before_by_process"],
        data,
        mass_bins,
        np.ones(data["mass"].shape, dtype=bool),
    )
    plot_mass_by_process(
        np,
        plt,
        outputs["mass_after_by_process"],
        data,
        mass_bins,
        data["score"] >= best["score_cut"],
        f"TMVA score >= {best['score_cut']:.3g}",
    )
    write_summary(np, yaml, outputs["summary"], args, pps_config, rows, skipped, best, outputs, data)

    print(f"Best TMVA score cut: {best['score_cut']:.6g}")
    print(f"Best combined significance: {best['significance']:.6g}")
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
