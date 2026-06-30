import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

LUMI_FB = 3000.0


def repo_root():
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, natural_key, resolve_path  # noqa: E402
from common.path_helper import campaign_config, delphes_root, pythia_root  # noqa: E402


def ensure_analysis_runtime(script_path, argv):
    if os.environ.get("HIGGS_CEP_ANALYZER_ENV") == "1":
        return

    setup_script = ROOT / "setup_env.sh"
    if not setup_script.is_file():
        raise RuntimeError(f"Setup script does not exist: {setup_script}")

    command = "\n".join(
        [
            f"source {shlex.quote(str(setup_script))}",
            "test -n \"$LCG_VIEW\"",
            "test -n \"$DELPHES_DIR\"",
            "export HIGGS_CEP_ANALYZER_ENV=1",
            f"exec python3 {shlex.join([str(Path(script_path).resolve()), *argv])}",
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


def selected_processes(processes, process_order, include_light_qcd):
    selected = []
    for name in process_order:
        if name not in processes:
            continue
        source_flavor = process_flavor(processes[name])
        if source_flavor in ("bb", "cc"):
            selected.append(name)
        elif include_light_qcd and source_flavor == "light":
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


def resolve_input_pairs(process_name, campaign_name):
    campaign, _ = campaign_config(process_name, campaign_name)
    delphes_dir = delphes_root(process_name, campaign)
    pythia_dir = pythia_root(process_name, campaign)

    root_files = sorted(delphes_dir.glob("*.root"), key=natural_key)
    if not root_files:
        raise RuntimeError(f"No Delphes ROOT files found in {delphes_dir}")

    hepmc_files = sorted(pythia_dir.glob("*.hepmc"), key=natural_key)
    if not hepmc_files:
        raise RuntimeError(f"No Pythia HepMC files found in {pythia_dir}")

    hepmc_by_stem = {path.stem: path for path in hepmc_files}
    pairs = []
    for root_file in root_files:
        pythia_file = hepmc_by_stem.get(root_file.stem)
        if pythia_file is None:
            raise RuntimeError(f"Missing matching HepMC file for {root_file.name} in {pythia_dir}")
        pairs.append((root_file, pythia_file))
    return campaign, pairs


def load_jets(ak, uproot, input_file, tree_name, collection):
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

    return {
        "pt": ak.values_astype(arrays[branch_name(collection, "PT")], "float64"),
        "eta": ak.values_astype(arrays[branch_name(collection, "Eta")], "float64"),
        "phi": ak.values_astype(arrays[branch_name(collection, "Phi")], "float64"),
        "mass": ak.values_astype(arrays[branch_name(collection, "Mass")], "float64"),
        "n_generated": n_generated,
    }


def build_dijets(ak, np, jets, input_file, collection):
    pt = jets["pt"]
    has_two_jets_ak = ak.num(pt) >= 2
    has_two_jets = ak.to_numpy(has_two_jets_ak)
    selected_event_indices = np.nonzero(has_two_jets)[0]
    if selected_event_indices.size == 0:
        raise RuntimeError(f"No events with at least two {collection} jets in {input_file}")

    selected_jets = ak.zip(
        {
            "pt": jets["pt"][has_two_jets_ak],
            "eta": jets["eta"][has_two_jets_ak],
            "phi": jets["phi"][has_two_jets_ak],
            "mass": jets["mass"][has_two_jets_ak],
        },
        with_name="Momentum4D",
    )
    order = ak.argsort(selected_jets.pt, axis=1, ascending=False)
    sorted_jets = selected_jets[order]
    jet1 = sorted_jets[:, 0]
    jet2 = sorted_jets[:, 1]
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
        "event_indices": selected_event_indices,
        "jets": sorted_jets,
        "jet1": jet1,
        "jet2": jet2,
        "dijet": dijet,
        "jet1_pt": jet1_pt,
        "jet2_pt": jet2_pt,
        "jet1_pt_over_mjj": jet1_pt_over_mjj,
        "jet2_pt_over_mjj": jet2_pt_over_mjj,
        "jet1_eta": ak.to_numpy(jet1.eta),
        "jet2_eta": ak.to_numpy(jet2.eta),
        "jet1_phi": ak.to_numpy(jet1.phi),
        "jet2_phi": ak.to_numpy(jet2.phi),
        "dijet_mass": dijet_mass,
        "dijet_pt": ak.to_numpy(dijet.pt),
        "dijet_eta": ak.to_numpy(dijet.eta),
        "dijet_phi": ak.to_numpy(dijet.phi),
        "dijet_rapidity": ak.to_numpy(dijet.rapidity),
    }


def proton_observables(np, xi_left, xi_right, sqrt_s):
    valid = np.isfinite(xi_left) & np.isfinite(xi_right) & (xi_left > 0.0) & (xi_right > 0.0)
    mx = np.full(xi_left.shape, np.nan, dtype=np.float64)
    yx = np.full(xi_left.shape, np.nan, dtype=np.float64)
    mx[valid] = np.sqrt(xi_left[valid] * xi_right[valid]) * sqrt_s
    yx[valid] = 0.5 * np.log(xi_right[valid] / xi_left[valid])
    return mx, yx, valid


def parse_hepmc_protons(np, input_file, selected_event_indices, sqrt_s):
    if input_file is None or not input_file.is_file():
        raise RuntimeError(f"Missing Pythia HepMC input for proton pairs: {input_file}")

    try:
        from pyHepMC3 import HepMC3
    except ImportError as exc:
        raise RuntimeError("pyHepMC3 is required to read Pythia HepMC files; run `source setup_env.sh`") from exc

    selected_positions = {int(event_index): index for index, event_index in enumerate(selected_event_indices)}
    last_selected = int(selected_event_indices[-1])
    left = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)
    right = np.full(selected_event_indices.shape, np.nan, dtype=np.float64)

    reader = HepMC3.ReaderAscii(str(input_file))
    try:
        event_index = 0
        while not reader.failed() and event_index <= last_selected:
            event = HepMC3.GenEvent()
            reader.read_event(event)
            if reader.failed():
                break

            output_index = selected_positions.get(event_index)
            if output_index is not None:
                beam_pos = None
                beam_neg = None
                left_energy = None
                left_abs_pz = -1.0
                right_energy = None
                right_abs_pz = -1.0
                for particle in event.particles():
                    if particle.pid() != 2212:
                        continue
                    momentum = particle.momentum()
                    pz = momentum.pz()
                    energy = momentum.e()
                    status = particle.status()
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

                if beam_pos is not None and beam_neg is not None and left_energy is not None and right_energy is not None:
                    left[output_index] = (beam_neg - left_energy) / beam_neg
                    right[output_index] = (beam_pos - right_energy) / beam_pos

            event_index += 1
    finally:
        reader.close()

    mx, yx, valid = proton_observables(np, left, right, sqrt_s)
    return {
        "xi_left": left,
        "xi_right": right,
        "mx": mx,
        "yx": yx,
        "valid": valid,
        "source": "pythia",
    }


def passes_pps(np, xi, xi_ranges):
    passed = np.zeros(np.asarray(xi).shape, dtype=bool)
    for _station, xi_min, xi_max in xi_ranges:
        passed |= (xi >= xi_min) & (xi < xi_max)
    return passed


def add_protons(np, table, protons, pps_config, rng):
    pps = protons["valid"] & passes_pps(np, protons["xi_left"], pps_config["xi_ranges"])
    pps &= passes_pps(np, protons["xi_right"], pps_config["xi_ranges"])

    xi_left_smeared = np.full(protons["xi_left"].shape, np.nan, dtype=np.float64)
    xi_right_smeared = np.full(protons["xi_right"].shape, np.nan, dtype=np.float64)
    if np.any(pps):
        xi_res = pps_config["xi_res"]
        if xi_res > 0.0:
            xi_left_smeared[pps] = rng.normal(protons["xi_left"][pps], xi_res)
            xi_right_smeared[pps] = rng.normal(protons["xi_right"][pps], xi_res)
        else:
            xi_left_smeared[pps] = protons["xi_left"][pps]
            xi_right_smeared[pps] = protons["xi_right"][pps]

    mx_smeared, yx_smeared, smeared_valid = proton_observables(
        np, xi_left_smeared, xi_right_smeared, pps_config["sqrt_s"]
    )
    table.update(
        {
            "xi_left": protons["xi_left"],
            "xi_right": protons["xi_right"],
            "mx": protons["mx"],
            "yx": protons["yx"],
            "xi_left_smeared": xi_left_smeared,
            "xi_right_smeared": xi_right_smeared,
            "mx_smeared": mx_smeared,
            "yx_smeared": yx_smeared,
            "valid_pp": protons["valid"],
            "pps": pps,
            "smeared_valid": smeared_valid,
            "proton_source": protons["source"],
        }
    )
    return table


def load_event_table(ak, np, uproot, delphes_file, pythia_file, tree_name, collection, pps_config, rng):
    print(f"Reading Delphes jets: {delphes_file}")
    jets = load_jets(ak, uproot, delphes_file, tree_name, collection)
    table = build_dijets(ak, np, jets, delphes_file, collection)
    table["n_generated"] = jets["n_generated"]
    table["n_selected"] = int(table["event_indices"].size)

    print(f"Reading Pythia protons: {pythia_file}")
    protons = parse_hepmc_protons(np, pythia_file, table["event_indices"], pps_config["sqrt_s"])
    return add_protons(np, table, protons, pps_config, rng)


def make_selection(
    np,
    table,
    require_valid_pp=False,
    require_pps=False,
    require_smeared_valid=False,
    min_jet_pt=None,
    dijet_mass_range=None,
    pp_mass_smeared_range=None,
):
    mask = np.ones(table["dijet_mass"].shape, dtype=bool)
    if require_valid_pp:
        mask &= table["valid_pp"]
    if require_pps:
        mask &= table["pps"]
    if require_smeared_valid:
        mask &= table["smeared_valid"]
    if min_jet_pt is not None:
        mask &= (table["jet1_pt"] >= min_jet_pt) & (table["jet2_pt"] >= min_jet_pt)
    if dijet_mass_range is not None:
        lo, hi = dijet_mass_range
        mask &= (table["dijet_mass"] >= lo) & (table["dijet_mass"] <= hi)
    if pp_mass_smeared_range is not None:
        lo, hi = pp_mass_smeared_range
        mask &= (table["mx_smeared"] >= lo) & (table["mx_smeared"] <= hi)
    return mask


def finite_values(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(np, datasets):
    finite = [finite_values(np, dataset) for dataset in datasets]
    finite = [dataset for dataset in finite if dataset.size]
    if not finite:
        return None
    data = np.concatenate(finite)
    lo = float(np.min(data))
    hi = float(np.max(data))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
        return lo - pad, hi + pad
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def concatenate_observables(np, observable_sets):
    keys = observable_sets[0].keys()
    return {key: np.concatenate([observables[key] for observables in observable_sets]) for key in keys}


def load_pair_observables(ak, np, uproot, pair, args, pps_config, rng, build_observables):
    delphes_file, pythia_file = pair
    if not delphes_file.is_file():
        raise RuntimeError(f"missing Delphes file: {delphes_file}")
    table = load_event_table(
        ak,
        np,
        uproot,
        delphes_file,
        pythia_file,
        args.tree,
        args.collection,
        pps_config,
        rng,
    )
    return {
        "observables": build_observables(np, table),
        "n_generated": int(table["n_generated"]),
        "n_selected": int(table["n_selected"]),
        "n_valid_pp": int(np.sum(table["valid_pp"])),
        "n_pps": int(np.sum(table["pps"])),
        "n_smeared_valid": int(np.sum(table["smeared_valid"])),
    }


def load_pair_observables_parallel(ak, np, uproot, pair_index, pair, args, pps_config, build_observables):
    rng = np.random.default_rng(pps_config["seed"] + pair_index)
    return load_pair_observables(ak, np, uproot, pair, args, pps_config, rng, build_observables)


def read_samples(ak, np, uproot, processes, parameters, pps_config, args, process_order, build_observables):
    samples = []
    skipped = []
    rng = np.random.default_rng(pps_config["seed"])
    workers = int(args.workers)
    for process_name in selected_processes(processes, process_order, args.include_light_qcd):
        process = processes[process_name]
        source_flavor = process_flavor(process)
        tag = tag_weight(parameters, args.flavor, source_flavor)
        try:
            campaign, pairs = resolve_input_pairs(process_name, args.campaign)
        except RuntimeError as exc:
            skipped.append((process_name, str(exc)))
            continue
        if args.max_files is not None:
            pairs = pairs[: args.max_files]

        results = []
        if workers == 1:
            for pair in pairs:
                try:
                    results.append(
                        load_pair_observables(ak, np, uproot, pair, args, pps_config, rng, build_observables)
                    )
                except RuntimeError as exc:
                    skipped.append((process_name, str(exc)))
        else:
            process_results = [None] * len(pairs)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        load_pair_observables_parallel,
                        ak,
                        np,
                        uproot,
                        index,
                        pair,
                        args,
                        pps_config,
                        build_observables,
                    ): index
                    for index, pair in enumerate(pairs)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        process_results[index] = future.result()
                    except RuntimeError as exc:
                        skipped.append((process_name, str(exc)))
            results = [result for result in process_results if result is not None]

        observable_sets = [result["observables"] for result in results]
        n_generated = 0
        n_selected = 0
        n_valid_pp = 0
        n_pps = 0
        n_smeared_valid = 0
        n_files = len(results)
        for result in results:
            n_generated += result["n_generated"]
            n_selected += result["n_selected"]
            n_valid_pp += result["n_valid_pp"]
            n_pps += result["n_pps"]
            n_smeared_valid += result["n_smeared_valid"]

        if not observable_sets:
            continue
        if n_generated <= 0:
            raise RuntimeError(f"{process_name}_{campaign} has zero generated events")

        event_weight = float(process["xsec_fb"]) * LUMI_FB * tag / float(n_generated)
        samples.append(
            {
                "name": process_name,
                "campaign": campaign,
                "observables": concatenate_observables(np, observable_sets),
                "event_weight": event_weight,
                "tag_weight": tag,
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
            f"pps={n_pps}, smeared_valid={n_smeared_valid}, tag_weight={tag:.6g}, "
            f"event_weight={event_weight:.6g}"
        )

    for process_name, reason in skipped:
        print(f"Warning: skipping {process_name}: {reason}")
    if not samples:
        raise RuntimeError("No usable inputs found for the selected processes")
    return samples


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
        data = finite_values(np, sample["observables"][variable])
        counts, edges = np.histogram(data, bins=bins, range=value_range)
        rows.append(
            {
                "sample": sample,
                "counts": counts.astype(np.float64),
                "entries": int(np.sum(counts)),
                "total_values": int(data.size),
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


def print_range_yields(histograms):
    lo, hi = histograms["value_range"]
    total = 0.0
    print(f"Yields for {histograms['variable']} within range [{lo:.6g}, {hi:.6g}]:")
    for row in histograms["rows"]:
        sample = row["sample"]
        in_range = row["entries"]
        range_yield = sample["event_weight"] * in_range
        total += range_yield
        percentage = 100 * in_range / row["total_values"] if row["total_values"] else 0.0
        print(
            f"  {sample['name']}_{sample['campaign']}: in_range={int(in_range)}, "
            f"percentage={percentage:.2f}%, range_yield={range_yield:.6g}"
        )
    print(f"  total range_yield={total:.6g}")


def plot_histograms(np, plt, histograms, output_dir, log_y, stacked, labels, colors, normalized=False):
    options = histograms["options"]
    value_range = histograms["value_range"]
    edges = histograms["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    plot_rows = [row for row in histograms["rows"] if row["entries"]]

    y_values = []
    y_errors = []
    plot_labels = []
    for row in plot_rows:
        sample = row["sample"]
        counts = row["counts"]
        if normalized:
            y = counts / row["entries"]
            yerr = np.sqrt(counts) / row["entries"]
            plot_labels.append(labels.get(sample["name"], sample["name"]))
        else:
            y = counts * sample["event_weight"]
            yerr = np.sqrt(counts) * sample["event_weight"]
            sample_yield = sample["event_weight"] * row["entries"]
            plot_labels.append(f"{labels.get(sample['name'], sample['name'])} ({sample_yield:.4g})")
        y_values.append(y)
        y_errors.append(yerr)
    plot_colors = [colors.get(row["sample"]["name"], None) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    artists = []
    if stacked and not normalized:
        bottom = np.zeros_like(y_values[0])
        for y, label, color in zip(y_values, plot_labels, plot_colors):
            artists.append(
                ax.bar(
                    edges[:-1],
                    y,
                    width=widths,
                    bottom=bottom,
                    align="edge",
                    label=label,
                    color=color,
                    alpha=0.75,
                    linewidth=0.8,
                    edgecolor="#333333",
                )
            )
            bottom += y
    else:
        for y, yerr, label, color in zip(y_values, y_errors, plot_labels, plot_colors):
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
    ax.set_ylabel("Normalized events / bin" if normalized else options.get("ylabel", "Expected events / bin"))
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


def write_plots(np, plt, samples, plot_variables, output_dir, bins, log_y, stacked, labels, colors):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for variable, options in plot_variables.items():
        histograms = make_histograms(np, samples, variable, options, bins)
        if histograms is None:
            continue
        if options["range"] is not None:
            print_range_yields(histograms)
        if plot_histograms(np, plt, histograms, output_dir, log_y, stacked, labels, colors):
            written += 1
        if plot_histograms(np, plt, histograms, output_dir, log_y, stacked, labels, colors, normalized=True):
            written += 1
    return written
