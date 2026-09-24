#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import analyzer

from common.config_utils import resolve_minbias_campaign
from common.path_helper import campaign_config, delphes_root


MASS_RANGE = (117, 133)
MASS_BINS = MASS_RANGE[1] - MASS_RANGE[0]
MINBIAS_BATCH_SIZE = 100_000

PLOT_VARIABLES = {
    "random_mx_smeared_after_pps": {
        "output": "random_mx_smeared_after_pps.png",
        "xlabel": "$M_X$ from random smeared protons [GeV]",
        "range": MASS_RANGE,
        "bins": MASS_BINS,
    },
    "random_yx_smeared_after_pps": {
        "output": "random_yx_smeared_after_pps.png",
        "xlabel": "$y_X$ from random smeared protons",
        "range": None,
        "bins": None,
    },
    "delta_mx_smeared_minus_dijet": {
        "output": "delta_mx_smeared_minus_dijet.png",
        "xlabel": "$M_X^{random,smeared} - m_{jj}$ [GeV]",
        "range": (-40, 80),
        "bins": None,
    },
    "delta_yx_smeared_minus_dijet": {
        "output": "delta_yx_smeared_minus_dijet.png",
        "xlabel": "$y_X^{random,smeared} - y_{jj}$",
        "range": (-4.0, 4.0),
        "bins": None,
    },
}


PROCESS_ORDER = (
    "QCDgg",
    #"QCDqq",
    #"QCDbb",
    "QCDcc",
    #"QEDbb",
    "QEDcc",
    #"Hbb",
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
        description="Plot random minbias proton-pair observables associated to Delphes dijets."
    )
    parser.add_argument("--flavor", choices=("bb", "cc"), required=True, help="Tag target")
    parser.add_argument(
        "--include-light-qcd",
        action="store_true",
        help="Include light QCD with light-to-heavy mistag weights.",
    )
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis")
    parser.add_argument(
        "--stacked",
        action="store_true",
        help="Accepted for consistency with nearby scripts; normalized plots are never stacked.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output plot directory. Defaults to analysis/output/random_protons_<flavor>.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of Delphes files to load per process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Accepted for consistency with nearby scripts; this script uses a sequential minbias stream.",
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


def default_output_dir(flavor, include_light_qcd):
    suffix = f"random_protons_{flavor}"
    if include_light_qcd:
        suffix += "_with_light_qcd"
    return analyzer.ROOT / "analysis" / "output" / suffix


def resolve_delphes_files(process_name, campaign_name):
    campaign, _ = campaign_config(process_name, campaign_name)
    input_dir = delphes_root(process_name, campaign)
    files = sorted(input_dir.glob("*.root"), key=analyzer.natural_key)
    if not files:
        raise RuntimeError(f"No Delphes ROOT files found in {input_dir}")
    return campaign, files


def minbias_input_path(args):
    if args.minbias_input is not None:
        path = analyzer.resolve_path(args.minbias_input, base=analyzer.ROOT)
        campaign = args.minbias_campaign or path.name
        return path, campaign
    path, campaign = resolve_minbias_campaign(args.minbias_campaign)
    return path, campaign


def discover_minbias_files(path, max_files):
    if path.is_file():
        files = [path]
    else:
        files = []
        files.extend(path.glob("*.npz"))
        files.extend(path.glob("*.parquet"))
        files.extend((path / "parquet").glob("*.parquet"))
        files = sorted(set(files), key=analyzer.natural_key)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No minbias .npz or .parquet files found in {path}")
    return files


def file_interactions(np, data, filename):
    if all(name in data.files for name in ("mu_per_bx", "bx_offset", "n_bx")):
        mu_per_bx = np.asarray(data["mu_per_bx"], dtype=np.int64)
        bx_offset = int(np.asarray(data["bx_offset"]).item())
        n_bx = int(np.asarray(data["n_bx"]).item())
        if mu_per_bx.size != n_bx:
            raise RuntimeError(
                f"mu_per_bx length ({mu_per_bx.size}) does not match n_bx ({n_bx}) in {filename}"
            )
        parts = []
        for local_bx, n_interactions in enumerate(mu_per_bx):
            if n_interactions <= 0:
                continue
            bx = bx_offset + local_bx
            interactions = np.arange(int(n_interactions), dtype=np.int64)
            parts.append(np.column_stack((np.full(interactions.shape, bx, dtype=np.int64), interactions)))
        if parts:
            return np.concatenate(parts, axis=0)
        return np.empty((0, 2), dtype=np.int64)

    if data["bx_id"].size == 0:
        return np.empty((0, 2), dtype=np.int64)
    keys = np.column_stack(
        (
            np.asarray(data["bx_id"], dtype=np.int64),
            np.asarray(data["interaction_id"], dtype=np.int64),
        )
    )
    return np.unique(keys, axis=0)


def compact_interaction_block(np, universe, bx_id, interaction_id, side, xi):
    if universe.size == 0:
        return {
            "offsets": np.zeros(1, dtype=np.int64),
            "side": np.empty(0, dtype=np.int8),
            "xi": np.empty(0, dtype=np.float64),
        }

    original_order = np.arange(len(xi), dtype=np.int64)
    order = np.lexsort((original_order, interaction_id, bx_id))
    sorted_bx = np.asarray(bx_id, dtype=np.int64)[order]
    sorted_interaction = np.asarray(interaction_id, dtype=np.int64)[order]

    universe_keys = np.rec.fromarrays(
        (universe[:, 0], universe[:, 1]), names=("bx", "interaction")
    )
    proton_keys = np.rec.fromarrays((sorted_bx, sorted_interaction), names=("bx", "interaction"))
    positions = np.searchsorted(universe_keys, proton_keys)
    if np.any(positions >= len(universe)) or np.any(universe_keys[positions] != proton_keys):
        raise RuntimeError(
            "Minbias proton references an interaction absent from the interaction universe"
        )

    counts = np.bincount(positions, minlength=len(universe))
    offsets = np.empty(len(universe) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return {
        "offsets": offsets,
        "side": np.asarray(side, dtype=np.int8)[order],
        "xi": np.asarray(xi, dtype=np.float64)[order],
    }


def load_npz_interaction_block(np, filename):
    fields = ("bx_id", "interaction_id", "side", "xi")
    with np.load(filename) as data:
        missing = [field for field in fields if field not in data.files]
        if missing:
            raise RuntimeError(f"Required arrays missing in {filename}: {', '.join(missing)}")
        arrays = {field: np.asarray(data[field]) for field in fields}
        universe = file_interactions(np, data, filename)
    return compact_interaction_block(np, universe, **arrays)


def parquet_interaction_block(np, event_id, pdg_id, is_final, pz, energy, sqrt_s):
    if event_id.size == 0:
        return None
    if np.any(event_id[1:] < event_id[:-1]):
        raise RuntimeError("Minbias Parquet event_id values must be ordered for streaming")

    event_starts = np.r_[0, np.nonzero(event_id[1:] != event_id[:-1])[0] + 1]
    event_numbers = np.searchsorted(event_starts, np.arange(event_id.size), side="right") - 1
    beam_energy = float(sqrt_s) / 2.0
    xi = (beam_energy - energy) / beam_energy
    valid = (pdg_id == 2212) & is_final & (pz != 0.0) & np.isfinite(xi) & (xi > 0.0)
    counts = np.bincount(event_numbers[valid], minlength=len(event_starts))
    offsets = np.empty(len(event_starts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return {
        "offsets": offsets,
        "side": np.where(pz[valid] < 0.0, -1, 1).astype(np.int8, copy=False),
        "xi": xi[valid].astype(np.float64, copy=False),
    }


def iter_parquet_interaction_blocks(np, filename, sqrt_s):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read minbias Parquet files; run `source setup_env.sh`") from exc

    columns = ["event_id", "pdg_id", "is_final", "pz", "E"]
    parquet_file = pq.ParquetFile(filename)
    missing = [column for column in columns if column not in parquet_file.schema_arrow.names]
    if missing:
        raise RuntimeError(f"Required columns missing in {filename}: {', '.join(missing)}")

    pending = None
    last_event_id = None
    for batch in parquet_file.iter_batches(batch_size=MINBIAS_BATCH_SIZE, columns=columns):
        arrays = {
            "event_id": np.asarray(batch.column("event_id")).astype(np.int64, copy=False),
            "pdg_id": np.asarray(batch.column("pdg_id")),
            "is_final": np.asarray(batch.column("is_final"), dtype=bool),
            "pz": np.asarray(batch.column("pz")).astype(np.float64, copy=False),
            "energy": np.asarray(batch.column("E")).astype(np.float64, copy=False),
        }
        if arrays["event_id"].size == 0:
            continue
        if last_event_id is not None and arrays["event_id"][0] < last_event_id:
            raise RuntimeError(f"Minbias Parquet event_id values must be ordered in {filename}")
        last_event_id = int(arrays["event_id"][-1])

        if pending is not None:
            for name in arrays:
                arrays[name] = np.concatenate((pending[name], arrays[name]))

        last_start = int(np.nonzero(arrays["event_id"] == arrays["event_id"][-1])[0][0])
        complete = {name: values[:last_start] for name, values in arrays.items()}
        pending = {name: values[last_start:] for name, values in arrays.items()}
        block = parquet_interaction_block(np, sqrt_s=sqrt_s, **complete)
        if block is not None:
            yield block

    if pending is not None:
        block = parquet_interaction_block(np, sqrt_s=sqrt_s, **pending)
        if block is not None:
            yield block


def iter_minbias_interaction_blocks(np, files, sqrt_s):
    for filename in files:
        if filename.suffix == ".npz":
            yield load_npz_interaction_block(np, filename)
        elif filename.suffix == ".parquet":
            yield from iter_parquet_interaction_blocks(np, filename, sqrt_s)
        else:
            raise RuntimeError(f"Unsupported minbias input file type: {filename}")


def new_minbias_cursor(np, campaign, files, sqrt_s):
    return {
        "campaign": campaign,
        "files": files,
        "np": np,
        "sqrt_s": sqrt_s,
        "blocks": iter_minbias_interaction_blocks(np, files, sqrt_s),
        "block": None,
        "block_cursor": 0,
        "exhausted": False,
        "interactions_consumed": 0,
        "bx_built": 0,
        "multi_pair_bx": 0,
        "pairs_kept": 0,
    }


def load_minbias(np, args, pps_config):
    path, campaign = minbias_input_path(args)
    files = discover_minbias_files(path, args.max_minbias_files)
    print(f"Streaming minbias campaign {campaign}: files={len(files)}")
    return new_minbias_cursor(np, campaign, files, pps_config["sqrt_s"])


def consume_interactions(minbias, n_interactions):
    np = minbias["np"]
    remaining = int(n_interactions)
    side_parts = []
    xi_parts = []
    while remaining:
        block = minbias["block"]
        if block is None or minbias["block_cursor"] >= len(block["offsets"]) - 1:
            try:
                block = next(minbias["blocks"])
            except StopIteration:
                minbias["exhausted"] = True
                return None
            minbias["block"] = block
            minbias["block_cursor"] = 0

        start_interaction = minbias["block_cursor"]
        available = len(block["offsets"]) - 1 - start_interaction
        take = min(remaining, available)
        start_proton = block["offsets"][start_interaction]
        stop_proton = block["offsets"][start_interaction + take]
        if stop_proton > start_proton:
            side_parts.append(block["side"][start_proton:stop_proton])
            xi_parts.append(block["xi"][start_proton:stop_proton])
        minbias["block_cursor"] += take
        minbias["interactions_consumed"] += take
        remaining -= take

    return {
        "side": np.concatenate(side_parts) if side_parts else np.empty(0, dtype=np.int8),
        "xi": np.concatenate(xi_parts) if xi_parts else np.empty(0, dtype=np.float64),
    }


def minbias_process_cursor(minbias):
    return new_minbias_cursor(
        minbias["np"], minbias["campaign"], minbias["files"], minbias["sqrt_s"]
    )


def add_minbias_counters(total, part):
    for key in ("interactions_consumed", "bx_built", "multi_pair_bx", "pairs_kept"):
        total[key] += part[key]


def random_bx_pairs(np, minbias, pps_config, rng):
    n_interactions = int(rng.poisson(200.0 if pps_config.get("mu") is None else pps_config["mu"]))
    interactions = consume_interactions(minbias, n_interactions)
    if interactions is None:
        return None

    minbias["bx_built"] += 1
    if interactions["xi"].size == 0:
        return empty_pairs(np)

    xi = interactions["xi"]
    side = interactions["side"]
    passed = analyzer.passes_pps(np, xi, pps_config["xi_ranges"])
    if not np.any(passed):
        return empty_pairs(np)

    xi_smeared = np.full(xi.shape, np.nan, dtype=np.float64)
    if pps_config["xi_res"] > 0.0:
        xi_smeared[passed] = rng.normal(xi[passed], pps_config["xi_res"])
    else:
        xi_smeared[passed] = xi[passed]

    left = np.nonzero(passed & (side == -1))[0]
    right = np.nonzero(passed & (side == 1))[0]
    if left.size == 0 or right.size == 0:
        return empty_pairs(np)

    left_idx = np.repeat(left, right.size)
    right_idx = np.tile(right, left.size)
    mx, yx, valid = analyzer.proton_observables(
        np,
        xi_smeared[left_idx],
        xi_smeared[right_idx],
        pps_config["sqrt_s"],
    )
    keep = valid & (mx >= MASS_RANGE[0]) & (mx <= MASS_RANGE[1])
    pairs = {"mx": mx[keep], "yx": yx[keep]}
    n_pairs = int(pairs["mx"].size)
    minbias["pairs_kept"] += n_pairs
    if n_pairs > 1:
        minbias["multi_pair_bx"] += 1
    return pairs


def empty_pairs(np):
    return {
        "mx": np.empty(0, dtype=np.float64),
        "yx": np.empty(0, dtype=np.float64),
    }


def load_dijet_table(ak, np, uproot, input_file, args):
    print(f"Reading Delphes jets: {input_file}")
    jets = analyzer.load_jets(ak, uproot, input_file, args.tree, args.collection)
    table = analyzer.build_dijets(ak, np, jets, input_file, args.collection)
    selected = analyzer.make_selection(np, table, min_jet_pt=20.0)
    return {
        "n_generated": int(jets["n_generated"]),
        "n_selected": int(np.sum(selected)),
        "dijet_mass": table["dijet_mass"][selected],
        "dijet_rapidity": table["dijet_rapidity"][selected],
    }


def append_random_observables(np, observables, dijet_table, minbias, pps_config, rng):
    assigned_bx = 0
    for dijet_mass, dijet_rapidity in zip(dijet_table["dijet_mass"], dijet_table["dijet_rapidity"]):
        pairs = random_bx_pairs(np, minbias, pps_config, rng)
        if pairs is None:
            break
        assigned_bx += 1
        if pairs["mx"].size == 0:
            continue
        observables["random_mx_smeared_after_pps"].append(pairs["mx"])
        observables["random_yx_smeared_after_pps"].append(pairs["yx"])
        observables["delta_mx_smeared_minus_dijet"].append(pairs["mx"] - dijet_mass)
        observables["delta_yx_smeared_minus_dijet"].append(pairs["yx"] - dijet_rapidity)
    return assigned_bx


def concatenate_or_empty(np, values):
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(values)


def read_random_samples(ak, np, uproot, processes, parameters, pps_config, args, minbias):
    samples = []
    skipped = []
    seed = args.seed if args.seed is not None else pps_config["seed"]
    selected_processes = analyzer.selected_processes(processes, PROCESS_ORDER, args.include_light_qcd)
    for process_index, process_name in enumerate(selected_processes):
        process_minbias = minbias_process_cursor(minbias)
        rng = np.random.default_rng(seed + process_index)
        process = processes[process_name]
        source_flavor = analyzer.process_flavor(process)
        tag = analyzer.tag_weight(parameters, args.flavor, source_flavor)
        try:
            campaign, files = resolve_delphes_files(process_name, args.campaign)
        except RuntimeError as exc:
            skipped.append((process_name, str(exc)))
            continue
        if args.max_files is not None:
            files = files[: args.max_files]

        observables = {key: [] for key in PLOT_VARIABLES}
        n_generated = 0
        n_selected = 0
        assigned_bx = 0
        files_read = 0

        for input_file in files:
            try:
                table = load_dijet_table(ak, np, uproot, input_file, args)
            except RuntimeError as exc:
                skipped.append((process_name, str(exc)))
                continue
            files_read += 1
            n_generated += table["n_generated"]
            n_selected += table["n_selected"]
            assigned_bx += append_random_observables(np, observables, table, process_minbias, pps_config, rng)
            if process_minbias["exhausted"]:
                break

        if files_read == 0:
            continue
        add_minbias_counters(minbias, process_minbias)
        samples.append(
            {
                "name": process_name,
                "campaign": campaign,
                "observables": {key: concatenate_or_empty(np, value) for key, value in observables.items()},
                "event_weight": 1.0,
                "tag_weight": tag,
                "n_generated": n_generated,
                "n_selected": n_selected,
                "n_assigned_bx": assigned_bx,
                "n_interactions_consumed": process_minbias["interactions_consumed"],
                "n_random_pairs": process_minbias["pairs_kept"],
                "n_multi_pair_bx": process_minbias["multi_pair_bx"],
                "n_files": files_read,
            }
        )
        print(
            f"{process_name}_{campaign}: files={files_read}, generated={n_generated}, "
            f"selected_dijets={n_selected}, assigned_bx={assigned_bx}, "
            f"interactions_consumed={process_minbias['interactions_consumed']}, "
            f"random_pairs={process_minbias['pairs_kept']}, "
            f"multi_pair_bx={process_minbias['multi_pair_bx']}, tag_weight={tag:.6g}"
        )
        if process_minbias["exhausted"]:
            print(f"Warning: stopping {process_name} because minbias interactions are exhausted")

    for process_name, reason in skipped:
        print(f"Warning: skipping {process_name}: {reason}")
    if not samples:
        raise RuntimeError("No usable inputs found for the selected processes")
    return samples


def write_normalized_plots(np, plt, samples, output_dir, bins, log_y):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for variable, options in PLOT_VARIABLES.items():
        histograms = analyzer.make_histograms(np, samples, variable, options, bins)
        if histograms is None:
            continue
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
    processes = analyzer.load_yaml(analyzer.ROOT / "processes.yaml")
    parameters = analyzer.load_yaml(analyzer.ROOT / "parameters.yaml")
    pps_config = analyzer.load_pps_config(analyzer.resolve_path(args.pps_config, base=analyzer.ROOT))
    pps_config["mu"] = float(args.mu)
    output_dir = (
        analyzer.resolve_path(args.output_dir, base=analyzer.ROOT)
        if args.output_dir
        else default_output_dir(args.flavor, args.include_light_qcd)
    )

    minbias = load_minbias(np, args, pps_config)
    samples = read_random_samples(ak, np, uproot, processes, parameters, pps_config, args, minbias)
    written = write_normalized_plots(np, plt, samples, output_dir, args.bins, args.log_y)
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
