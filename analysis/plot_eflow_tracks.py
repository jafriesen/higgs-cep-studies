#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import analyzer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_utils import natural_key, resolve_path  # noqa: E402
from common.path_helper import generation_campaign_config, generation_stage_root  # noqa: E402


MIN_JET_PT = 20.0
OUTER_TRACK_MIN_R = 0.4
OUTER_TRACK_MIN_PT = 1.0
TRACK_MAX_ABS_ETA = 2.5
TRACK_COLLECTIONS = (
    {
        "name": "EFlowTrack",
        "label": "tracks",
        "linestyle": "-",
    },
    {
        "name": "EFlowTrackPUPPI",
        "label": "PUPPI tracks",
        "linestyle": "--",
    },
)
SAMPLE_SPECS = (
    {
        "name": "qcdbb_madgraph",
        "generator": "madgraph",
        "process": "QCDbb",
        "label": "MadGraph QCD $b\\bar{b}$",
        "color": "#009E73",
        "campaign_arg": "madgraph_qcdbb_campaign",
        "subcampaign_arg": "madgraph_qcdbb_subcampaign",
    },
    {
        "name": "hbb_superchic",
        "generator": "superchic",
        "process": "Hbb",
        "label": "SuperChic $H\\rightarrow b\\bar{b}$",
        "color": "#0072B2",
        "campaign_arg": "hbb_campaign",
        "subcampaign_arg": "hbb_subcampaign",
    },
    {
        "name": "qcdbb_superchic",
        "generator": "superchic",
        "process": "QCDbb",
        "label": "SuperChic CEP QCD $b\\bar{b}$",
        "color": "#D55E00",
        "campaign_arg": "cep_qcdbb_campaign",
        "subcampaign_arg": "cep_qcdbb_subcampaign",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare EFlowTrack distance to the nearest of the two leading jets "
            "in Hbb, CEP QCDbb, and MadGraph QCDbb Delphes samples."
        )
    )
    parser.add_argument("--hbb-campaign", default=None, help="SuperChic Hbb main campaign")
    parser.add_argument(
        "--hbb-subcampaign", default=None, help="SuperChic Hbb sim-delphes subcampaign"
    )
    parser.add_argument(
        "--cep-qcdbb-campaign", default=None, help="SuperChic CEP QCDbb main campaign"
    )
    parser.add_argument(
        "--cep-qcdbb-subcampaign",
        default=None,
        help="SuperChic CEP QCDbb sim-delphes subcampaign",
    )
    parser.add_argument(
        "--madgraph-qcdbb-campaign", default=None, help="MadGraph QCDbb main campaign"
    )
    parser.add_argument(
        "--madgraph-qcdbb-subcampaign",
        default=None,
        help="MadGraph QCDbb sim-delphes subcampaign",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to analysis/output/eflow_tracks.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--bins", type=int, default=50, help="Number of radial bins")
    parser.add_argument("--r-max", type=float, default=5.0, help="Maximum plotted track distance")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files per sample")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum TTree entries read from each file",
    )
    return parser.parse_args()


def validate_args(args):
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.r_max <= 0.0:
        raise RuntimeError("--r-max must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")


def resolve_sample_files(spec, args):
    campaign_arg = getattr(args, spec["campaign_arg"])
    subcampaign_arg = getattr(args, spec["subcampaign_arg"])
    campaign, _ = generation_campaign_config(
        spec["generator"], spec["process"], campaign_arg
    )
    root_dir = (
        generation_stage_root(
            spec["generator"],
            spec["process"],
            campaign,
            "sim-delphes",
            subcampaign=subcampaign_arg,
        )
        / "root"
    )
    files = sorted(root_dir.glob("*.root"), key=natural_key)
    if not files:
        raise RuntimeError(f"No Delphes ROOT files found in {root_dir}")
    if args.max_files is not None:
        files = files[: args.max_files]
    return campaign, root_dir.parent.name, files


def wrapped_delta_phi(np, phi1, phi2):
    difference = phi1 - phi2
    return np.arctan2(np.sin(difference), np.cos(difference))


def available_track_collections(keys):
    keys = set(keys)
    return [
        collection
        for collection in TRACK_COLLECTIONS
        if all(
            analyzer.branch_name(collection["name"], field) in keys
            for field in ("PT", "Eta", "Phi")
        )
    ]


def selected_track_values(
    ak,
    np,
    jet_pt,
    jet_eta,
    jet_phi,
    track_pt,
    track_eta,
    track_phi,
):
    order = ak.argsort(jet_pt, axis=1, ascending=False)
    sorted_pt = ak.pad_none(jet_pt[order], 2)
    sorted_eta = ak.pad_none(jet_eta[order], 2)
    sorted_phi = ak.pad_none(jet_phi[order], 2)

    selected = (ak.num(jet_pt) >= 2) & ak.fill_none(
        (sorted_pt[:, 0] >= MIN_JET_PT) & (sorted_pt[:, 1] >= MIN_JET_PT),
        False,
    )
    n_selected = int(ak.sum(selected))
    if n_selected == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64), np.empty(0), 0

    selected_track_pt = track_pt[selected]
    selected_track_eta = track_eta[selected]
    selected_track_phi = track_phi[selected]
    in_acceptance = np.abs(selected_track_eta) < TRACK_MAX_ABS_ETA
    selected_track_pt = selected_track_pt[in_acceptance]
    selected_track_eta = selected_track_eta[in_acceptance]
    selected_track_phi = selected_track_phi[in_acceptance]
    jet1_eta = sorted_eta[selected][:, 0, np.newaxis]
    jet1_phi = sorted_phi[selected][:, 0, np.newaxis]
    jet2_eta = sorted_eta[selected][:, 1, np.newaxis]
    jet2_phi = sorted_phi[selected][:, 1, np.newaxis]

    delta_r1 = np.hypot(
        selected_track_eta - jet1_eta,
        wrapped_delta_phi(np, selected_track_phi, jet1_phi),
    )
    delta_r2 = np.hypot(
        selected_track_eta - jet2_eta,
        wrapped_delta_phi(np, selected_track_phi, jet2_phi),
    )
    nearest_delta_r = np.minimum(delta_r1, delta_r2)
    outer_track = (
        np.isfinite(nearest_delta_r)
        & np.isfinite(selected_track_pt)
        & (nearest_delta_r > OUTER_TRACK_MIN_R)
        & (selected_track_pt >= OUTER_TRACK_MIN_PT)
    )
    outer_track_multiplicity = ak.sum(outer_track, axis=1)
    outer_track_sum_pt = ak.sum(selected_track_pt[outer_track], axis=1)
    return (
        ak.to_numpy(ak.flatten(nearest_delta_r)),
        ak.to_numpy(ak.flatten(selected_track_pt)),
        ak.to_numpy(outer_track_multiplicity),
        ak.to_numpy(outer_track_sum_pt),
        n_selected,
    )


def load_file(ak, np, uproot, filename, tree_name, max_events):
    jet_fields = [analyzer.branch_name("JetPUPPI", field) for field in ("PT", "Eta", "Phi")]

    with uproot.open(filename) as root_file:
        if tree_name not in root_file:
            print(f"WARNING: Could not find TTree '{tree_name}' in {filename}; skipping file")
            return {}, 0
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {filename}")
        tree = root_file[tree_name]
        keys = tree.keys()
        missing = [name for name in jet_fields if name not in keys]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {filename}: {', '.join(missing)}")
        collections = available_track_collections(keys)
        n_events = (
            min(tree.num_entries, max_events) if max_events is not None else tree.num_entries
        )
        if not collections:
            return {}, int(n_events)
        track_fields = {
            collection["name"]: [
                analyzer.branch_name(collection["name"], field)
                for field in ("PT", "Eta", "Phi")
            ]
            for collection in collections
        }
        required = jet_fields + [
            field for fields in track_fields.values() for field in fields
        ]
        arrays = tree.arrays(required, entry_stop=n_events, library="ak")

    jets = [ak.values_astype(arrays[name], "float64") for name in jet_fields]
    results = {}
    for collection in collections:
        tracks = [
            ak.values_astype(arrays[name], "float64")
            for name in track_fields[collection["name"]]
        ]
        results[collection["name"]] = selected_track_values(ak, np, *jets, *tracks)
    return results, int(n_events)


def analyze_sample(ak, np, uproot, spec, campaign, subcampaign, files, args, edges):
    accumulators = {}

    print(
        f"Loading {spec['label']} ({campaign}/{subcampaign}) from {len(files)} file(s)...",
        flush=True,
    )
    for index, filename in enumerate(files, start=1):
        print(f"  Reading {index}/{len(files)}: {filename.name}", flush=True)
        file_results, file_events = load_file(
            ak, np, uproot, filename, args.tree, args.max_events
        )
        if not file_results:
            print("    No supported track collection found; skipping file")
            continue
        for collection_name, result in file_results.items():
            delta_r, track_pt, outer_track_multiplicity, outer_track_sum_pt, file_selected = result
            accumulator = accumulators.setdefault(
                collection_name,
                {
                    "track_counts": np.zeros(args.bins, dtype=np.float64),
                    "track_pt_sums": np.zeros(args.bins, dtype=np.float64),
                    "n_files": 0,
                    "n_events": 0,
                    "n_selected": 0,
                    "n_tracks": 0,
                    "n_invalid": 0,
                    "n_outside": 0,
                    "outer_track_multiplicities": [],
                    "outer_track_sum_pts": [],
                },
            )
            accumulator["n_files"] += 1
            accumulator["n_events"] += file_events
            accumulator["n_selected"] += file_selected
            accumulator["n_tracks"] += delta_r.size
            accumulator["outer_track_multiplicities"].append(outer_track_multiplicity)
            accumulator["outer_track_sum_pts"].append(outer_track_sum_pt)

            finite_r = np.isfinite(delta_r)
            finite_pt = np.isfinite(track_pt)
            accumulator["n_invalid"] += int(np.sum(~finite_r | ~finite_pt))
            in_range = finite_r & (delta_r >= 0.0) & (delta_r < args.r_max)
            accumulator["n_outside"] += int(np.sum(finite_r & ~in_range))
            accumulator["track_counts"] += np.histogram(
                delta_r[in_range], bins=edges
            )[0]
            weighted = in_range & finite_pt
            accumulator["track_pt_sums"] += np.histogram(
                delta_r[weighted], bins=edges, weights=track_pt[weighted]
            )[0]

    samples = []
    for collection in TRACK_COLLECTIONS:
        collection_name = collection["name"]
        if collection_name not in accumulators:
            continue
        accumulator = accumulators[collection_name]
        label = f"{spec['label']} ({collection['label']})"
        if not np.sum(accumulator["track_counts"]):
            print(f"  Skipping {label}: no tracks in 0 <= r < {args.r_max:g}")
            continue
        if not np.sum(accumulator["track_pt_sums"]):
            print(f"  Skipping {label}: zero track pT in 0 <= r < {args.r_max:g}")
            continue

        print(
            f"  {collection_name}: files={accumulator['n_files']}, "
            f"events={accumulator['n_events']}, "
            f"selected_events={accumulator['n_selected']}, "
            f"tracks={accumulator['n_tracks']}, "
            f"invalid_tracks={accumulator['n_invalid']}, "
            f"tracks_outside_range={accumulator['n_outside']}"
        )
        samples.append(
            {
                **spec,
                "label": label,
                "linestyle": collection["linestyle"],
                "track_collection": collection_name,
                "campaign": campaign,
                "subcampaign": subcampaign,
                "track_counts": accumulator["track_counts"],
                "track_pt_sums": accumulator["track_pt_sums"],
                "outer_track_multiplicity": np.concatenate(
                    accumulator["outer_track_multiplicities"]
                ),
                "outer_track_sum_pt": np.concatenate(
                    accumulator["outer_track_sum_pts"]
                ),
            }
        )

    if not samples:
        names = ", ".join(collection["name"] for collection in TRACK_COLLECTIONS)
        raise RuntimeError(f"{spec['label']} has none of the track collections: {names}")
    return samples


def normalized_density(np, values, widths):
    total = float(np.sum(values))
    density = values / (total * widths)
    if not np.all(np.isfinite(density)):
        raise RuntimeError("Histogram density contains non-finite values")
    if not np.isclose(np.sum(density * widths), 1.0):
        raise RuntimeError("Histogram density does not integrate to one")
    return density


def plot_density(np, plt, samples, edges, key, ylabel, output_path):
    widths = np.diff(edges)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sample in samples:
        density = normalized_density(np, sample[key], widths)
        ax.stairs(
            density,
            edges,
            label=sample["label"],
            color=sample["color"],
            linestyle=sample["linestyle"],
            linewidth=1.7,
        )

    ax.set_xlabel(r"$r = \min\,\Delta R(\mathrm{track}, j_{1,2})$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def plot_outer_track_multiplicity(np, plt, samples, output_path):
    maximum = max(int(np.max(sample["outer_track_multiplicity"])) for sample in samples)
    edges = np.arange(-0.5, maximum + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sample in samples:
        values = sample["outer_track_multiplicity"]
        counts = np.histogram(values, bins=edges)[0].astype(np.float64)
        normalized = counts / values.size
        #if not np.isclose(np.sum(normalized), 1.0):
        #    raise RuntimeError("Outer-track multiplicity histogram does not integrate to one")
        ax.stairs(
            normalized,
            edges,
            label=sample["label"],
            color=sample["color"],
            linestyle=sample["linestyle"],
            linewidth=1.7,
        )

    ax.set_xlabel(
        r"Tracks per event with $r>0.4$ and $p_T\geq 1\,\mathrm{GeV}$"
    )
    ax.set_ylabel("Normalized events / bin")
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def plot_outer_track_sum_pt(np, plt, samples, output_path):
    cap = max(
        float(np.percentile(sample["outer_track_sum_pt"], 99.0)) for sample in samples
    )
    edges = np.linspace(0.0, max(cap, 1.0), 41)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sample in samples:
        values = sample["outer_track_sum_pt"]
        counts = np.histogram(values, bins=edges)[0].astype(np.float64)
        normalized = counts / values.size
        ax.stairs(
            normalized,
            edges,
            label=sample["label"],
            color=sample["color"],
            linestyle=sample["linestyle"],
            linewidth=1.7,
        )

    ax.set_xlabel(
        rf"$\sum p_T$ of tracks with $r>{OUTER_TRACK_MIN_R:g}$ and "
        rf"$p_T\geq {OUTER_TRACK_MIN_PT:g}\,\mathrm{{GeV}}$ [GeV]"
    )
    ax.set_ylabel("Normalized events / bin")
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def main():
    args = parse_args()
    validate_args(args)
    analyzer.ensure_analysis_runtime(Path(__file__), sys.argv[1:])
    ak, np, plt, uproot = analyzer.import_libraries()

    edges = np.linspace(0.0, args.r_max, args.bins + 1)
    samples = []
    for spec in SAMPLE_SPECS:
        campaign, subcampaign, files = resolve_sample_files(spec, args)
        samples.extend(
            analyze_sample(
                ak, np, uproot, spec, campaign, subcampaign, files, args, edges
            )
        )

    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else ROOT / "analysis" / "output" / "eflow_tracks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_density(
        np,
        plt,
        samples,
        edges,
        "track_counts",
        r"$(1/N_{\mathrm{tracks}})\,dN_{\mathrm{tracks}}/dr$",
        output_dir / "track_delta_r_normalized.png",
    )
    plot_density(
        np,
        plt,
        samples,
        edges,
        "track_pt_sums",
        r"$(1/\sum p_T)\,d(\sum p_T)/dr$",
        output_dir / "track_pt_density_normalized.png",
    )
    plot_outer_track_multiplicity(
        np,
        plt,
        samples,
        output_dir / "track_multiplicity_r_gt_0p4_pt1_normalized.png",
    )
    plot_outer_track_sum_pt(
        np,
        plt,
        samples,
        output_dir / "outer_track_sum_pt_normalized.png",
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
