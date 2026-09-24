#!/usr/bin/env python3
"""Prototype in-jet track observables for MadGraph-combinatorial rejection.

Standalone study script: reads Delphes ROOT files directly, applies the
central-only pre-MVA selection from run_dijet_mva.py (two jets with
pT >= 15 GeV, |delta_phi_jj| > 3, outer-track multiplicity < 5,
50 <= mjj <= 150 GeV; no proton/PPS requirements), computes per-jet
track-based substructure observables from EFlowTrack, and plots normalized
comparisons for Hbb signal vs SuperChic QCDbb vs MadGraph inclusive QCDbb.

Observables (per jet, tracks with pT >= 0.5 GeV and dR(track, jet) < 0.4):
  1. in-jet charged multiplicity and charged pT sum / jet pT
  2. track girth = sum(pT * dR) / sum(pT) and pTD = sqrt(sum pT^2) / sum pT
  3. jet pull magnitude and pull angle w.r.t. the other jet

Run under the LCG view used by the analysis, e.g.:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
  python3 analysis/MVA/prototype_jet_substructure.py
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import awkward as ak
import matplotlib
import numpy as np
import uproot
import vector

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

vector.register_awkward()

MIN_JET_PT = 15.0
IN_TRACK_MAX_R = 0.4
IN_TRACK_MIN_PT = 0.5
OUTER_TRACK_MIN_R = 0.4
OUTER_TRACK_MIN_PT = 1.0
MAX_OUTER_TRACK_MULTIPLICITY = 5
MIN_DIJET_DELTA_PHI = 3.0
PRE_MVA_DIJET_MASS_RANGE_GEV = (50.0, 150.0)

REPO = Path(__file__).resolve().parents[2]
SAMPLES = (
    {
        "name": "Hbb",
        "label": "H->bb (SuperChic)",
        "input_dir": REPO / "output-superchic/Hbb/Hbb__v01/sim-Delphes/Hbb_noFSR__v01/root",
        "max_files": 5,
        "color": "#0072B2",
        "linestyle": "-",
    },
    {
        "name": "QCDbb",
        "label": "SuperChic QCDbb",
        "input_dir": REPO / "output-superchic/QCDbb/QCDbb__v01/sim-Delphes/QCDbb_noFSR__v01/root",
        "max_files": 5,
        "color": "#E69F00",
        "linestyle": "--",
    },
    {
        "name": "QCDbb_madgraph",
        "label": "MadGraph QCDbb (inclusive)",
        "input_dir": REPO / "output-madgraph/QCDbb/QCDbb__v01/sim-Delphes/QCDbb_noFSR__v01/root",
        "max_files": 5,
        "color": "#D55E00",
        "linestyle": "-.",
    },
)
OBSERVABLES = (
    ("n_trk_in", "In-jet charged multiplicity (pT > 0.5)", np.arange(-0.5, 20.5, 1.0)),
    ("chg_pt_frac", "In-jet charged pT sum / jet pT", np.linspace(0.0, 1.5, 41)),
    ("girth", "Track girth", np.linspace(0.0, 0.4, 41)),
    ("ptd", "Track pTD", np.linspace(0.0, 1.0, 41)),
    ("pull_mag", "Jet pull magnitude", np.linspace(0.0, 0.02, 41)),
    ("pull_angle", "Pull angle w.r.t. other jet [rad]", np.linspace(0.0, np.pi, 41)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch")
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "analysis/MVA/output/substructure_proto"),
        help="Output directory for plots and the separation summary.",
    )
    return parser.parse_args()


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def wrap_phi(np_like):
    return (np_like + np.pi) % (2.0 * np.pi) - np.pi


def jet_track_observables(track_pt, track_eta, track_phi, jet, other_jet):
    """Track-based substructure of one jet (per event)."""
    deta = track_eta - jet.eta[:, np.newaxis]
    dphi = wrap_phi(track_phi - jet.phi[:, np.newaxis])
    dr = np.hypot(deta, dphi)
    in_jet = (dr < IN_TRACK_MAX_R) & (track_pt >= IN_TRACK_MIN_PT)
    pt_in = track_pt[in_jet]
    sum_pt = ak.to_numpy(ak.sum(pt_in, axis=1))
    n_trk = ak.to_numpy(ak.sum(in_jet, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        girth = ak.to_numpy(ak.sum((track_pt * dr)[in_jet], axis=1)) / sum_pt
        ptd = np.sqrt(ak.to_numpy(ak.sum((track_pt**2)[in_jet], axis=1))) / sum_pt
    jet_pt = ak.to_numpy(jet.pt)
    pull_eta = ak.to_numpy(ak.sum((track_pt * deta * dr)[in_jet], axis=1)) / jet_pt
    pull_phi = ak.to_numpy(ak.sum((track_pt * dphi * dr)[in_jet], axis=1)) / jet_pt
    pull_mag = np.hypot(pull_eta, pull_phi)
    link_eta = ak.to_numpy(other_jet.eta - jet.eta)
    link_phi = ak.to_numpy(wrap_phi(other_jet.phi - jet.phi))
    link_mag = np.hypot(link_eta, link_phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_angle = (pull_eta * link_eta + pull_phi * link_phi) / (pull_mag * link_mag)
    pull_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return {
        "n_trk_in": n_trk.astype(np.float64),
        "chg_pt_frac": sum_pt / jet_pt,
        "girth": girth,
        "ptd": ptd,
        "pull_mag": pull_mag,
        "pull_angle": pull_angle,
    }


def process_file(path, tree_name, collection):
    jet_fields = ("PT", "Eta", "Phi", "Mass")
    required = [branch_name(collection, field) for field in jet_fields]
    required.extend(branch_name("EFlowTrack", field) for field in ("PT", "Eta", "Phi"))
    with uproot.open(path) as root_file:
        tree = root_file[tree_name]
        arrays = tree.arrays(required, library="ak")

    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    phi = ak.values_astype(arrays[branch_name(collection, "Phi")], "float64")
    mass = ak.values_astype(arrays[branch_name(collection, "Mass")], "float64")
    jet_mask = pt >= MIN_JET_PT
    pt, eta, phi, mass = pt[jet_mask], eta[jet_mask], phi[jet_mask], mass[jet_mask]
    has_two = ak.to_numpy(ak.num(pt) >= 2)
    if not np.any(has_two):
        return None

    jets = ak.zip(
        {"pt": pt[has_two], "eta": eta[has_two], "phi": phi[has_two], "mass": mass[has_two]},
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    leading, subleading = jets[order][:, 0], jets[order][:, 1]
    mjj = ak.to_numpy((leading + subleading).mass)
    abs_dphi_jj = np.abs(ak.to_numpy(wrap_phi(leading.phi - subleading.phi)))

    track_pt = ak.values_astype(arrays[branch_name("EFlowTrack", "PT")], "float64")[has_two]
    track_eta = ak.values_astype(arrays[branch_name("EFlowTrack", "Eta")], "float64")[has_two]
    track_phi = ak.values_astype(arrays[branch_name("EFlowTrack", "Phi")], "float64")[has_two]
    dr1 = np.hypot(
        track_eta - leading.eta[:, np.newaxis],
        wrap_phi(track_phi - leading.phi[:, np.newaxis]),
    )
    dr2 = np.hypot(
        track_eta - subleading.eta[:, np.newaxis],
        wrap_phi(track_phi - subleading.phi[:, np.newaxis]),
    )
    outer = ak.to_numpy(
        ak.sum(
            (np.minimum(dr1, dr2) > OUTER_TRACK_MIN_R) & (track_pt >= OUTER_TRACK_MIN_PT),
            axis=1,
        )
    )
    keep = (
        (abs_dphi_jj > MIN_DIJET_DELTA_PHI)
        & (outer < MAX_OUTER_TRACK_MULTIPLICITY)
        & (mjj >= PRE_MVA_DIJET_MASS_RANGE_GEV[0])
        & (mjj <= PRE_MVA_DIJET_MASS_RANGE_GEV[1])
    )
    if not np.any(keep):
        return None
    keep_ak = ak.Array(keep)
    leading, subleading = leading[keep_ak], subleading[keep_ak]
    track_pt, track_eta, track_phi = track_pt[keep_ak], track_eta[keep_ak], track_phi[keep_ak]

    obs1 = jet_track_observables(track_pt, track_eta, track_phi, leading, subleading)
    obs2 = jet_track_observables(track_pt, track_eta, track_phi, subleading, leading)
    result = {name: np.concatenate([obs1[name], obs2[name]]) for name, _t, _b in OBSERVABLES}
    result["outer_zero"] = np.tile(outer[keep] == 0, 2)
    result["n_events"] = int(np.sum(keep))
    return result


def load_sample(spec, tree_name, collection):
    files = sorted(spec["input_dir"].glob("*.root"))[: spec["max_files"]]
    if not files:
        raise RuntimeError(f"No ROOT files in {spec['input_dir']}")
    pieces = []
    n_events = 0
    for path in files:
        piece = process_file(path, tree_name, collection)
        if piece is not None:
            n_events += piece.pop("n_events")
            pieces.append(piece)
    merged = {
        key: np.concatenate([piece[key] for piece in pieces])
        for key in list(pieces[0])
    }
    print(
        f"{spec['name']}: files={len(files)} selected_events={n_events} "
        f"jets={merged['girth'].size} (outer==0 jets: {int(merged['outer_zero'].sum())})",
        flush=True,
    )
    return merged


def separation(values_a, values_b, bins):
    hist_a, _ = np.histogram(values_a[np.isfinite(values_a)], bins=bins, density=True)
    hist_b, _ = np.histogram(values_b[np.isfinite(values_b)], bins=bins, density=True)
    width = np.diff(bins)
    return 1.0 - float(np.sum(np.minimum(hist_a, hist_b) * width))


def plot_panels(samples, data, output_path, subset_key=None, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    lines = []
    for (name, xlabel, bins), ax in zip(OBSERVABLES, axes.ravel()):
        for spec in samples:
            values = data[spec["name"]][name]
            if subset_key is not None:
                values = values[data[spec["name"]][subset_key]]
            values = values[np.isfinite(values)]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=spec["color"],
                linestyle=spec["linestyle"],
                label=spec["label"],
            )
        sep_sc = separation(
            *(
                data[s["name"]][name][data[s["name"]][subset_key]] if subset_key else data[s["name"]][name]
                for s in (samples[0], samples[1])
            ),
            bins,
        )
        sep_mg = separation(
            *(
                data[s["name"]][name][data[s["name"]][subset_key]] if subset_key else data[s["name"]][name]
                for s in (samples[0], samples[2])
            ),
            bins,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized jets / bin")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"sep vs SC = {sep_sc:.2f}, vs MG = {sep_mg:.2f}", fontsize=10)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle(f"In-jet track observables{title_suffix}", y=1.06)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return lines


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {spec["name"]: load_sample(spec, args.tree, args.collection) for spec in SAMPLES}

    summary_lines = ["separation = 1 - histogram overlap (normalized), pooled leading+subleading jets", ""]
    for subset_key, tag in ((None, "pre-MVA central selection"), ("outer_zero", "outer-track multiplicity == 0")):
        summary_lines.append(f"[{tag}]")
        summary_lines.append(f"{'observable':16s} {'Hbb vs SC QCDbb':>16s} {'Hbb vs MG comb':>16s}")
        for name, _xlabel, bins in OBSERVABLES:
            def values(sample_name):
                out = data[sample_name][name]
                if subset_key is not None:
                    out = out[data[sample_name][subset_key]]
                return out

            sep_sc = separation(values("Hbb"), values("QCDbb"), bins)
            sep_mg = separation(values("Hbb"), values("QCDbb_madgraph"), bins)
            summary_lines.append(f"{name:16s} {sep_sc:16.3f} {sep_mg:16.3f}")
        summary_lines.append("")

    plot_panels(SAMPLES, data, output_dir / "substructure_premva.png", None, " (pre-MVA central selection)")
    plot_panels(
        SAMPLES, data, output_dir / "substructure_exclusive.png", "outer_zero",
        " (outer-track multiplicity == 0)",
    )
    summary = "\n".join(summary_lines)
    (output_dir / "separations.txt").write_text(summary + "\n")
    print(summary)
    print(f"Wrote plots and separations to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
