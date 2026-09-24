#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import analyzer
import plot_random_protons

from common.path_helper import generation_campaign_config, generation_stage_root


PLOT_VARIABLES = {
    "delta_mx_smeared_minus_dijet": {
        "output": "delta_mx_smeared_minus_dijet.png",
        "xlabel": "$M_X^{smeared} - m_{jj}$ [GeV]",
        "range": (-40, 80),
        "bins": None,
    },
    "delta_yx_smeared_minus_dijet": {
        "output": "delta_yx_smeared_minus_dijet.png",
        "xlabel": "$y_X^{smeared} - y_{jj}$",
        "range": (-4.0, 4.0),
        "bins": None,
    },
}


REAL_PROCESS_ORDER = ("QCDbb", "Hbb")
MADGRAPH_PROCESS = "QCDbb"
MADGRAPH_SAMPLE_NAME = "QCDbbMadGraphRandom"


PROCESS_LABELS = {
    "Hbb": "$H\\rightarrow b\\bar{b}$ real protons",
    "QCDbb": "QCD $b\\bar{b}$ real protons",
    MADGRAPH_SAMPLE_NAME: "MadGraph QCD $b\\bar{b}$ random protons",
}


COLORS = {
    "Hbb": "#0072B2",
    "QCDbb": "#E69F00",
    MADGRAPH_SAMPLE_NAME: "#009E73",
}


DELTA_Y_CUTS = (0.1, 0.2, 0.3, 0.4, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot delta mass and rapidity for Hbb/QCDbb real protons and "
            "MadGraph QCDbb random minbias protons."
        )
    )
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key for SuperChic Hbb/QCDbb real-proton samples.",
    )
    parser.add_argument(
        "--madgraph-campaign",
        default=None,
        help="MadGraph QCDbb campaign. Defaults to processes-madgraph.yaml default_campaign.main.",
    )
    parser.add_argument(
        "--madgraph-subcampaign",
        default=None,
        help="MadGraph sim-delphes subcampaign. Defaults to processes-madgraph.yaml default_campaign.sim-delphes.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to analysis/output/real_vs_madgraph_random_bb.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to load per sample.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Accepted for consistency with nearby scripts; real-proton files are read sequentially.",
    )
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file with beam.sqrt_s_gev, pps.xi_ranges, pps.xi_res, and random.seed.",
    )
    parser.add_argument(
        "--minbias-campaign",
        default=None,
        help="Minbias campaign key. Defaults to config.yaml minbias.default_campaign.",
    )
    parser.add_argument(
        "--minbias-input",
        default=None,
        help="Override minbias input .npz/.parquet file or directory. Defaults to the minbias campaign directory.",
    )
    parser.add_argument(
        "--max-minbias-files",
        type=int,
        default=None,
        help="Maximum number of minbias .npz/.parquet files to load.",
    )
    parser.add_argument("--mu", type=float, default=200.0, help="Mean interactions per synthetic BX")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to random.seed from --pps-config.",
    )
    return parser.parse_args()


def default_output_dir():
    return analyzer.ROOT / "analysis" / "output" / "real_vs_madgraph_random_bb"


def build_real_observables(np, table):
    smeared = analyzer.make_selection(
        np,
        table,
        require_smeared_valid=True,
        min_jet_pt=20.0,
    )
    return {
        "delta_mx_smeared_minus_dijet": table["mx_smeared"][smeared] - table["dijet_mass"][smeared],
        "delta_yx_smeared_minus_dijet": table["yx_smeared"][smeared] - table["dijet_rapidity"][smeared],
    }


def new_observable_summary(np, default_bins):
    variables = {}
    for variable, options in PLOT_VARIABLES.items():
        bins = options["bins"] or default_bins
        lo, hi = options["range"]
        variables[variable] = {
            "counts": np.zeros(bins, dtype=np.int64),
            "edges": np.linspace(lo, hi, bins + 1),
            "total_values": 0,
        }
    return {
        "variables": variables,
        "delta_y_passed": {cut: 0 for cut in DELTA_Y_CUTS},
        "delta_y_total": 0,
    }


def update_observable_summary(np, summary, observables):
    for variable, values in observables.items():
        finite = analyzer.finite_values(np, values)
        variable_summary = summary["variables"][variable]
        counts, _ = np.histogram(finite, bins=variable_summary["edges"])
        variable_summary["counts"] += counts
        variable_summary["total_values"] += int(finite.size)

        if variable == "delta_yx_smeared_minus_dijet":
            abs_values = np.abs(finite)
            summary["delta_y_total"] += int(finite.size)
            for cut in DELTA_Y_CUTS:
                summary["delta_y_passed"][cut] += int(np.sum(abs_values <= cut))


def resolve_superchic_input_pairs(process_name, args):
    campaign, _ = generation_campaign_config("superchic", process_name, args.campaign)
    pythia_dir = (
        generation_stage_root("superchic", process_name, campaign, "hadr-pythia")
        / "hepmc"
    )
    delphes_dir = (
        generation_stage_root("superchic", process_name, campaign, "sim-delphes")
        / "root"
    )

    root_files = sorted(delphes_dir.glob("*.root"), key=analyzer.natural_key)
    if not root_files:
        raise RuntimeError(f"No SuperChic Delphes ROOT files found in {delphes_dir}")

    hepmc_files = sorted(pythia_dir.glob("*.hepmc"), key=analyzer.natural_key)
    if not hepmc_files:
        raise RuntimeError(f"No SuperChic Pythia HepMC files found in {pythia_dir}")

    hepmc_by_stem = {path.stem: path for path in hepmc_files}
    pairs = []
    for root_file in root_files:
        pythia_file = hepmc_by_stem.get(root_file.stem)
        if pythia_file is None:
            raise RuntimeError(f"Missing matching HepMC file for {root_file.name} in {pythia_dir}")
        pairs.append((root_file, pythia_file))

    if args.max_files is not None:
        pairs = pairs[: args.max_files]
    return campaign, pairs


def read_real_samples(ak, np, uproot, pps_config, args):
    samples = []
    skipped = []
    rng = np.random.default_rng(pps_config["seed"])
    for process_name in REAL_PROCESS_ORDER:
        try:
            campaign, pairs = resolve_superchic_input_pairs(process_name, args)
        except RuntimeError as exc:
            skipped.append((process_name, str(exc)))
            continue

        summary = new_observable_summary(np, args.bins)
        n_generated = 0
        n_selected = 0
        n_valid_pp = 0
        n_pps = 0
        n_smeared_valid = 0
        n_files = 0
        for pair in pairs:
            try:
                result = analyzer.load_pair_observables(
                    ak,
                    np,
                    uproot,
                    pair,
                    args,
                    pps_config,
                    rng,
                    build_real_observables,
                )
            except RuntimeError as exc:
                skipped.append((process_name, str(exc)))
                continue
            update_observable_summary(np, summary, result["observables"])
            n_generated += result["n_generated"]
            n_selected += result["n_selected"]
            n_valid_pp += result["n_valid_pp"]
            n_pps += result["n_pps"]
            n_smeared_valid += result["n_smeared_valid"]
            n_files += 1

        if n_files == 0:
            continue

        samples.append(
            {
                "name": process_name,
                "campaign": campaign,
                "summary": summary,
                "event_weight": 1.0,
                "tag_weight": 1.0,
                "n_generated": n_generated,
                "n_selected": n_selected,
                "n_valid_proton_pairs": n_valid_pp,
                "n_pps": n_pps,
                "n_smeared_valid": n_smeared_valid,
                "n_files": n_files,
            }
        )
        pps_percent = 100 * n_pps / n_valid_pp if n_valid_pp else 0.0
        print(
            f"{process_name}_{campaign}: files={n_files}, generated={n_generated}, "
            f"two_jet={n_selected}, proton_pairs={n_valid_pp}, pps%={pps_percent:.2f}%, "
            f"pps={n_pps}, smeared_valid={n_smeared_valid}"
        )

    for process_name, reason in skipped:
        print(f"Warning: skipping {process_name}: {reason}")
    if not samples:
        raise RuntimeError("No usable SuperChic real-proton inputs found")
    return samples


def resolve_madgraph_delphes_files(args):
    campaign, _ = generation_campaign_config("madgraph", MADGRAPH_PROCESS, args.madgraph_campaign)
    delphes_dir = generation_stage_root(
        "madgraph",
        MADGRAPH_PROCESS,
        campaign,
        "sim-delphes",
        args.madgraph_subcampaign,
    )
    candidates = [delphes_dir, delphes_dir / "root"]
    files = []
    for candidate in candidates:
        if candidate.is_dir():
            files = sorted(candidate.glob("*.root"), key=analyzer.natural_key)
            if files:
                break
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No MadGraph QCDbb Delphes ROOT files found under {delphes_dir}")
    return campaign, files


def append_random_deltas(np, summary, dijet_table, minbias, pps_config, rng):
    assigned_bx = 0
    for dijet_mass, dijet_rapidity in zip(dijet_table["dijet_mass"], dijet_table["dijet_rapidity"]):
        pairs = plot_random_protons.random_bx_pairs(np, minbias, pps_config, rng)
        if pairs is None:
            break
        assigned_bx += 1
        if pairs["mx"].size == 0:
            continue
        update_observable_summary(
            np,
            summary,
            {
                "delta_mx_smeared_minus_dijet": pairs["mx"] - dijet_mass,
                "delta_yx_smeared_minus_dijet": pairs["yx"] - dijet_rapidity,
            },
        )
    return assigned_bx


def read_madgraph_random_sample(ak, np, uproot, pps_config, args, minbias):
    campaign, files = resolve_madgraph_delphes_files(args)
    process_minbias = plot_random_protons.minbias_process_cursor(minbias)
    rng = np.random.default_rng(args.seed if args.seed is not None else pps_config["seed"])
    summary = new_observable_summary(np, args.bins)
    n_generated = 0
    n_selected = 0
    assigned_bx = 0
    files_read = 0

    for input_file in files:
        table = plot_random_protons.load_dijet_table(ak, np, uproot, input_file, args)
        files_read += 1
        n_generated += table["n_generated"]
        n_selected += table["n_selected"]
        assigned_bx += append_random_deltas(np, summary, table, process_minbias, pps_config, rng)
        if process_minbias["exhausted"]:
            break

    plot_random_protons.add_minbias_counters(minbias, process_minbias)
    print(
        f"{MADGRAPH_SAMPLE_NAME}_{campaign}: files={files_read}, generated={n_generated}, "
        f"selected_dijets={n_selected}, assigned_bx={assigned_bx}, "
        f"interactions_consumed={process_minbias['interactions_consumed']}, "
        f"random_pairs={process_minbias['pairs_kept']}, "
        f"multi_pair_bx={process_minbias['multi_pair_bx']}"
    )
    if process_minbias["exhausted"]:
        print(f"Warning: stopping {MADGRAPH_SAMPLE_NAME} because minbias interactions are exhausted")

    return {
        "name": MADGRAPH_SAMPLE_NAME,
        "campaign": campaign,
        "summary": summary,
        "event_weight": 1.0,
        "tag_weight": 1.0,
        "n_generated": n_generated,
        "n_selected": n_selected,
        "n_assigned_bx": assigned_bx,
        "n_interactions_consumed": process_minbias["interactions_consumed"],
        "n_random_pairs": process_minbias["pairs_kept"],
        "n_multi_pair_bx": process_minbias["multi_pair_bx"],
        "n_files": files_read,
    }


def write_normalized_plots(np, plt, samples, output_dir, log_y):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for variable, options in PLOT_VARIABLES.items():
        rows = []
        edges = None
        for sample in samples:
            variable_summary = sample["summary"]["variables"][variable]
            edges = variable_summary["edges"]
            rows.append(
                {
                    "sample": sample,
                    "counts": variable_summary["counts"].astype(np.float64),
                    "entries": int(np.sum(variable_summary["counts"])),
                    "total_values": variable_summary["total_values"],
                }
            )
        if not any(row["entries"] for row in rows):
            print(f"Warning: no finite values for {variable}; skipping")
            continue
        histograms = {
            "variable": variable,
            "options": options,
            "value_range": options["range"],
            "edges": edges,
            "rows": rows,
        }
        if analyzer.plot_histograms(
            np,
            plt,
            histograms,
            output_dir,
            log_y,
            False,
            PROCESS_LABELS,
            COLORS,
            normalized=True,
        ):
            written += 1
    return written


def print_delta_y_efficiencies(samples):
    print("Delta y cut efficiencies using finite delta_y entries:")
    for sample in samples:
        summary = sample["summary"]
        total = summary["delta_y_total"]
        label = PROCESS_LABELS.get(sample["name"], sample["name"])
        if total == 0:
            print(f"  {label}: no finite delta_y entries")
            continue
        parts = []
        for cut in DELTA_Y_CUTS:
            passed = summary["delta_y_passed"][cut]
            efficiency = passed / total
            parts.append(f"|dy|<={cut:.1f}: {efficiency:.6g} ({passed}/{total})")
        print(f"  {label}: " + ", ".join(parts))


def main():
    args = parse_args()
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.workers <= 0:
        raise RuntimeError("--workers must be > 0")
    if args.max_minbias_files is not None and args.max_minbias_files <= 0:
        raise RuntimeError("--max-minbias-files must be > 0")
    if args.mu <= 0.0:
        raise RuntimeError("--mu must be > 0")

    analyzer.ensure_analysis_runtime(Path(__file__), sys.argv[1:])
    ak, np, plt, uproot = analyzer.import_libraries()
    pps_config = analyzer.load_pps_config(analyzer.resolve_path(args.pps_config, base=analyzer.ROOT))
    pps_config["mu"] = float(args.mu)
    output_dir = analyzer.resolve_path(args.output_dir, base=analyzer.ROOT) if args.output_dir else default_output_dir()

    real_samples = read_real_samples(ak, np, uproot, pps_config, args)
    minbias = plot_random_protons.load_minbias(np, args, pps_config)
    madgraph_sample = read_madgraph_random_sample(ak, np, uproot, pps_config, args, minbias)
    samples = real_samples + [madgraph_sample]

    print_delta_y_efficiencies(samples)
    written = write_normalized_plots(np, plt, samples, output_dir, args.log_y)
    print(
        f"Minbias totals: synthetic_bx={minbias['bx_built']}, "
        f"interactions_consumed={minbias['interactions_consumed']}, "
        f"random_pairs={minbias['pairs_kept']}, multi_pair_bx={minbias['multi_pair_bx']}"
    )
    print(f"Wrote {written} normalized plot(s)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
