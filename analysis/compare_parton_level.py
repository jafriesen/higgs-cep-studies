#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pylhe
import vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_utils import discover_event_files, load_yaml, resolve_path  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_campaign_root,
    generation_config,
    generation_process_config,
    generation_stage_root,
)


LUMI_FB = 3000.0
PPS_MASS_RANGE_GEV = (90.0, 160.0)
COLORS = {
    "superchic": ("#0072B2", "#56B4E9", "#009E73", "#CC79A7"),
    "fpmc": ("#D55E00", "#E69F00", "#F0E442", "#999999"),
}

PLOT_VARIABLES = {
    "leading_parton_pt": ("Leading j $p_T$ [GeV]", None),
    "subleading_parton_pt": ("Subleading j $p_T$ [GeV]", None),
    "leading_parton_eta": ("Leading j $\\eta$", None),
    "subleading_parton_eta": ("Subleading j $\\eta$", None),
    "leading_parton_phi": ("Leading j $\\phi$", (-math.pi, math.pi)),
    "subleading_parton_phi": ("Subleading j $\\phi$", (-math.pi, math.pi)),
    "dijet_mass": ("Dijet mass [GeV]", (90,160)),
    "dijet_pt": ("Dijet $p_T$ [GeV]", None),
    "dijet_eta": ("Dijet $\\eta$", None),
    "dijet_phi": ("Dijet $\\phi$", (-math.pi, math.pi)),
    "dijet_delta_phi": ("Dijet $|\\Delta\\phi|$", (0, math.pi)),
    "dijet_delta_eta": ("Dijet $|\\Delta\\eta|$", None),
    "dijet_delta_r": ("Dijet $\\Delta R$", None),
    "proton_neg_xi": ("Negative-$p_z$ proton $\\xi$", None),
    "proton_neg_pt": ("Negative-$p_z$ proton $p_T$ [GeV]", None),
    "proton_neg_eta": ("Negative-$p_z$ proton $\\eta$", None),
    "proton_neg_phi": ("Negative-$p_z$ proton $\\phi$", (-math.pi, math.pi)),
    "proton_pos_xi": ("Positive-$p_z$ proton $\\xi$", (0,0.2)),
    "proton_pos_pt": ("Positive-$p_z$ proton $p_T$ [GeV]", None),
    "proton_pos_eta": ("Positive-$p_z$ proton $\\eta$", None),
    "proton_pos_phi": ("Positive-$p_z$ proton $\\phi$", (-math.pi, math.pi)),
    "proton_pair_mass": ("$M_X$ [GeV]", (90,160)),
    "proton_pair_rapidity": ("$y_X$", (-2,2)),
    "proton_pair_delta_phi": ("Proton pair $|\\Delta\\phi|$", (0, math.pi)),
    "proton_pair_delta_eta": ("Proton pair $|\\Delta\\eta|$", None),
    "proton_pair_delta_r": ("Proton pair $\\Delta R$", None),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SuperChic and FPMC parton-level LHE samples."
    )
    parser.add_argument("process", help="Process name from the generator process configs")
    parser.add_argument(
        "--superchic",
        nargs="+",
        default=None,
        metavar="CAMPAIGN",
        help="SuperChic main campaign(s); defaults to the configured campaign",
    )
    parser.add_argument(
        "--fpmc",
        nargs="+",
        default=None,
        metavar="CAMPAIGN",
        help="FPMC main campaign(s); defaults to the configured campaign",
    )
    parser.add_argument(
        "--fpmc-parton-campaign",
        default=None,
        help="FPMC parton-Pythia subcampaign; specifying it forces use of that layer.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files per sample")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum total events to read per sample",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: analysis/output/compare_parton_level/<process>)",
    )
    parser.add_argument(
        "--pps-config",
        default="analysis/scripts/new/config.yaml",
        help="YAML file containing beam.sqrt_s_gev and pps.xi_ranges",
    )
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins")
    parser.add_argument("--log-y", action="store_true", help="Use logarithmic y axes")
    return parser.parse_args()


def validate_args(args):
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")


def load_pps_config(path):
    config = load_yaml(path)
    sqrt_s = float((config.get("beam") or {}).get("sqrt_s_gev", 0.0))
    if sqrt_s <= 0.0:
        raise RuntimeError(f"{path} must define a positive beam.sqrt_s_gev")

    xi_ranges = []
    for station, bounds in ((config.get("pps") or {}).get("xi_ranges") or {}).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise RuntimeError(f"Invalid PPS xi range for station {station}: {bounds}")
        xi_ranges.append((str(station), low, high))
    if not xi_ranges:
        raise RuntimeError(f"{path} must define at least one pps.xi_ranges entry")
    return {"sqrt_s": sqrt_s, "xi_ranges": xi_ranges}


def generator_event_dir(generator, process, campaign):
    generator_cfg = generation_config(generator)
    campaign_name, _ = generation_campaign_config(generator, process, campaign)
    return (
        generation_campaign_root(generator, process, campaign_name)
        / generator_cfg["generation_dir"]
        / "evrecs"
    ), campaign_name


def parton_ids(jet_type):
    if jet_type == "b":
        return {5, -5}
    if jet_type == "c":
        return {4, -4}
    if jet_type == "q":
        return {-3, -2, -1, 1, 2, 3}
    if jet_type == "g":
        return {21}
    raise RuntimeError(f"Unsupported jet_type '{jet_type}'")


def event_partons(event, allowed_ids):
    return [
        particle
        for particle in event.particles
        if particle.status == 1 and particle.id in allowed_ids
    ]


def first_event_has_partons(files, allowed_ids):
    for filename in files:
        lhe = pylhe.LHEFile.fromfile(
            filename, generator=True, with_attributes=False
        )
        for event in lhe.events:
            return len(event_partons(event, allowed_ids)) == 2
    return False


def resolve_samples(args):
    print(f"Resolving generator inputs for process {args.process}...", flush=True)
    superchic_cfg = generation_process_config("superchic", args.process)
    fpmc_cfg = generation_process_config("fpmc", args.process)
    superchic_ids = parton_ids(superchic_cfg.get("jet_type"))
    fpmc_ids = parton_ids(fpmc_cfg.get("jet_type"))

    samples = []
    for index, campaign in enumerate(args.superchic or [None]):
        event_dir, campaign_name = generator_event_dir(
            "superchic", args.process, campaign
        )
        samples.append({
            "key": "superchic",
            "label": f"SuperChic {campaign_name}",
            "files": discover_event_files(event_dir, args.max_files),
            "process_config": superchic_cfg,
            "allowed_ids": superchic_ids,
            "color": COLORS["superchic"][index % len(COLORS["superchic"])],
        })

    for index, campaign in enumerate(args.fpmc or [None]):
        event_dir, campaign_name = generator_event_dir("fpmc", args.process, campaign)
        files = discover_event_files(event_dir, args.max_files)
        use_parton_layer = args.fpmc_parton_campaign is not None
        if not use_parton_layer:
            print(
                f"Checking whether raw FPMC {campaign_name} events contain the "
                "requested partons...",
                flush=True,
            )
            use_parton_layer = not first_event_has_partons(files, fpmc_ids)

        if use_parton_layer:
            stage_root = generation_stage_root(
                "fpmc",
                args.process,
                campaign_name,
                "parton-pythia",
                args.fpmc_parton_campaign,
            )
            files = discover_event_files(stage_root / "lhe", args.max_files)
            label = f"FPMC+Pythia {campaign_name}/{stage_root.name}"
        else:
            label = f"FPMC {campaign_name}"

        samples.append({
            "key": "fpmc",
            "label": label,
            "files": files,
            "process_config": fpmc_cfg,
            "allowed_ids": fpmc_ids,
            "color": COLORS["fpmc"][index % len(COLORS["fpmc"])],
        })

    for sample in samples:
        print(
            f"  {sample['label']}: {len(sample['files'])} file(s) under "
            f"{sample['files'][0].parent}",
            flush=True,
        )
    return samples


def p4(particle):
    return vector.obj(
        px=particle.px,
        py=particle.py,
        pz=particle.pz,
        E=particle.e,
    )


def abs_delta_phi(phi1, phi2):
    return abs(math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2)))


def event_observables(event, init_info, allowed_ids, pps_config, location):
    partons = event_partons(event, allowed_ids)
    if len(partons) != 2:
        raise RuntimeError(
            f"{location} has {len(partons)} configured final-state partons; expected exactly 2"
        )

    protons = [
        particle
        for particle in event.particles
        if particle.status == 1 and particle.id == 2212
    ]
    negative = [particle for particle in protons if particle.pz < 0.0]
    positive = [particle for particle in protons if particle.pz > 0.0]
    if len(negative) != 1 or len(positive) != 1:
        raise RuntimeError(
            f"{location} has {len(negative)} negative-pz and {len(positive)} "
            "positive-pz final-state protons; expected one per side"
        )

    energy_neg = float(init_info.energyB)
    energy_pos = float(init_info.energyA)
    if energy_neg <= 0.0 or energy_pos <= 0.0:
        raise RuntimeError(f"{location} has non-positive LHE beam energy")

    leading, subleading = sorted((p4(partons[0]), p4(partons[1])), key=lambda v: v.pt, reverse=True)
    dijet = leading + subleading
    proton_neg = p4(negative[0])
    proton_pos = p4(positive[0])
    dijet_delta_phi = abs_delta_phi(leading.phi, subleading.phi)
    dijet_delta_eta = abs(leading.eta - subleading.eta)
    proton_pair_delta_phi = abs_delta_phi(proton_neg.phi, proton_pos.phi)
    proton_pair_delta_eta = abs(proton_neg.eta - proton_pos.eta)
    xi_neg = (energy_neg - negative[0].e) / energy_neg
    xi_pos = (energy_pos - positive[0].e) / energy_pos
    if xi_neg <= 0.0 or xi_pos <= 0.0:
        raise RuntimeError(
            f"{location} has non-positive proton momentum loss: "
            f"xi_neg={xi_neg:.6g}, xi_pos={xi_pos:.6g}"
        )
    proton_pair_mass = math.sqrt(xi_neg * xi_pos) * pps_config["sqrt_s"]

    return {
        "leading_parton_pt": leading.pt,
        "subleading_parton_pt": subleading.pt,
        "leading_parton_eta": leading.eta,
        "subleading_parton_eta": subleading.eta,
        "leading_parton_phi": leading.phi,
        "subleading_parton_phi": subleading.phi,
        "dijet_mass": dijet.mass,
        "dijet_pt": dijet.pt,
        "dijet_eta": dijet.eta,
        "dijet_phi": dijet.phi,
        "dijet_delta_phi": dijet_delta_phi,
        "dijet_delta_eta": dijet_delta_eta,
        "dijet_delta_r": math.hypot(dijet_delta_eta, dijet_delta_phi),
        "proton_neg_xi": xi_neg,
        "proton_neg_pt": proton_neg.pt,
        "proton_neg_eta": proton_neg.eta,
        "proton_neg_phi": proton_neg.phi,
        "proton_pos_xi": xi_pos,
        "proton_pos_pt": proton_pos.pt,
        "proton_pos_eta": proton_pos.eta,
        "proton_pos_phi": proton_pos.phi,
        "proton_pair_mass": proton_pair_mass,
        "proton_pair_rapidity": 0.5 * math.log(xi_pos / xi_neg),
        "proton_pair_delta_phi": proton_pair_delta_phi,
        "proton_pair_delta_eta": proton_pair_delta_eta,
        "proton_pair_delta_r": math.hypot(
            proton_pair_delta_eta,
            proton_pair_delta_phi,
        ),
    }


def in_pps_acceptance(xi, xi_ranges):
    accepted = np.zeros(xi.shape, dtype=bool)
    for _station, low, high in xi_ranges:
        accepted |= (xi >= low) & (xi < high)
    return accepted


def cumulative_selections(observables, pps_config):
    mass_min, mass_max = PPS_MASS_RANGE_GEV
    cuts = (
        (
            f"{mass_min:g} <= M_X <= {mass_max:g} GeV",
            (observables["proton_pair_mass"] >= mass_min)
            & (observables["proton_pair_mass"] <= mass_max),
        ),
        (
            "0.002 < xi1 < 0.2",
            (observables["proton_neg_xi"] > 0.002) & (observables["proton_neg_xi"] < 0.2)
        ),
        (
            "0.002 < xi2 < 0.2",
            (observables["proton_pos_xi"] > 0.002) & (observables["proton_pos_xi"] < 0.2)
        ),
        ( 
            "|y_X| < 2.4",
            (observables["proton_pair_rapidity"] < 2.4) & (observables["proton_pair_rapidity"] > -2.4)
        ),
        (
            "pt1 > 20 GeV",
            (observables["leading_parton_pt"] > 20.0)
        ),
        (   
            "pt2 > 20 GeV",
            (observables["subleading_parton_pt"] > 20.0)
        ),
        (
            "|eta1| < 3",
            (observables["leading_parton_eta"] < 3.0) & (observables["leading_parton_eta"] > -3.0)
        ),
        (
            "|eta2| < 3",
            (observables["subleading_parton_eta"] < 3.0) & (observables["subleading_parton_eta"] > -3.0)
        ),
        (
            "x1 in PPS",
            in_pps_acceptance(observables["proton_neg_xi"], pps_config["xi_ranges"]),
        ),
        (
            "x2 in PPS",
            in_pps_acceptance(observables["proton_pos_xi"], pps_config["xi_ranges"]),
        ),
    )

    cumulative = np.ones(len(observables["proton_pair_mass"]), dtype=bool)
    selections = {"before_pps": cumulative.copy()}
    for name, cut in cuts:
        cumulative &= cut
        selections[name] = cumulative.copy()
    selections["after_pps"] = cumulative
    return selections


def print_cutflow(sample):
    selections = sample["selections"]
    weights = sample["weights"]
    total_events = sample["n_events"]
    total_yield = float(np.sum(weights))
    previous_events = total_events
    previous_yield = total_yield

    print("  Cumulative cutflow:")
    for name, mask in selections.items():
        if name == "after_pps":
            continue
        events = int(np.count_nonzero(mask))
        event_step = events / previous_events if previous_events else 0.0
        event_total = events / total_events if total_events else 0.0
        selected_yield = float(np.sum(weights[mask]))
        yield_step = selected_yield / previous_yield if previous_yield else 0.0
        yield_total = selected_yield / total_yield if total_yield else 0.0
        effective_xsec_fb = yield_total * sample["xsec_fb"]
        print(
            f"    {name:22s} events={events:8d} "
            f"event_eff(step/total)={event_step:7.3%}/{event_total:7.3%} "
            f"yield={selected_yield:.8g} xsec={effective_xsec_fb:.8g} fb "
            f"yield_eff(step/total)={yield_step:7.3%}/{yield_total:7.3%}"
        )
        previous_events = events
        previous_yield = selected_yield


def file_init(filename):
    return pylhe.LHEFile.fromfile(
        filename, generator=True, with_attributes=False
    ).init


def sample_cross_section_fb(process_config, inits, sample_label):
    configured = process_config.get("xsec_fb")
    if configured is not None:
        xsec_fb = float(configured)
        source = "config"
    else:
        xsecs_pb = [sum(float(proc.xSection) for proc in init.procInfo) for init in inits]
        if not xsecs_pb or not all(xsec > 0.0 for xsec in xsecs_pb):
            raise RuntimeError(f"{sample_label} has no positive configured or LHE cross section")
        xsec_fb = 1000.0 * float(np.mean(xsecs_pb))
        source = "LHE"
    if xsec_fb <= 0.0:
        raise RuntimeError(f"{sample_label} cross section must be > 0")
    return xsec_fb, source


def load_sample(sample, pps_config, max_events):
    limit = f", stopping after {max_events} events" if max_events is not None else ""
    print(
        f"Loading {sample['label']} from {len(sample['files'])} file(s){limit}...",
        flush=True,
    )
    global_weight = float(sample["process_config"].get("weight", 1.0))

    columns = {variable: [] for variable in PLOT_VARIABLES}
    raw_weights = []
    inits = []
    stop = False
    for file_index, filename in enumerate(sample["files"], start=1):
        print(
            f"  Reading file {file_index}/{len(sample['files'])}: {filename.name}",
            flush=True,
        )
        lhe = pylhe.LHEFile.fromfile(
            filename, generator=True, with_attributes=False
        )
        init = lhe.init
        inits.append(init)
        for file_event_index, event in enumerate(lhe.events):
            if max_events is not None and len(raw_weights) >= max_events:
                stop = True
                break
            location = f"{filename}, event {file_event_index}"
            observables = event_observables(
                event,
                init.initInfo,
                sample["allowed_ids"],
                pps_config,
                location,
            )
            for variable, values in columns.items():
                values.append(observables[variable])
            raw_weights.append(float(event.eventinfo.weight))
        if stop:
            break

    if sample["process_config"].get("xsec_fb") is None:
        inits.extend(
            file_init(filename) for filename in sample["files"][len(inits):]
        )
    xsec_fb, xsec_source = sample_cross_section_fb(
        sample["process_config"], inits, sample["label"]
    )

    if not raw_weights:
        raise RuntimeError(f"{sample['label']} has no events")
    total_weight = float(np.sum(raw_weights))
    if not math.isfinite(total_weight) or total_weight == 0.0:
        raise RuntimeError(f"{sample['label']} has zero or non-finite total event weight")

    expected_total = xsec_fb * global_weight * LUMI_FB
    weights = np.asarray(raw_weights, dtype=np.float64) * expected_total / total_weight
    observables = {
        variable: np.asarray(values, dtype=np.float64)
        for variable, values in columns.items()
    }
    selections = cumulative_selections(observables, pps_config)
    pps = selections["after_pps"]
    sample.update(
        {
            "observables": observables,
            "selections": selections,
            "weights": weights,
            "xsec_fb": xsec_fb,
            "xsec_source": xsec_source,
            "global_weight": global_weight,
            "n_events": len(raw_weights),
        }
    )
    print(
        f"{sample['label']}: files={len(sample['files'])}, events={len(raw_weights)}, "
        f"xsec={xsec_fb:.8g} fb ({xsec_source}), total_event_weight={total_weight:.8g}, "
        f"before_pps={np.sum(weights):.8g}, after_pps={np.sum(weights[pps]):.8g}, "
        f"pps={np.count_nonzero(pps)}/{len(raw_weights)}"
    )
    print_cutflow(sample)
    return sample


def histogram_range(samples, variable, fixed_range):
    if fixed_range is not None:
        return fixed_range
    values = []
    for sample in samples:
        finite = sample["observables"][variable]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)
    if not values:
        raise RuntimeError(f"No finite values for {variable}")
    combined = np.concatenate(values)
    low = float(np.min(combined))
    high = float(np.max(combined))
    if low == high:
        padding = 0.05 * abs(low) if low else 1.0
    else:
        padding = 0.02 * (high - low)
    return low - padding, high + padding


def plot_variable(
    samples,
    variable,
    xlabel,
    value_range,
    bins,
    selection_name,
    output_dir,
    log_y,
    normalized=False,
):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    positive_y = []
    for sample in samples:
        mask = sample["selections"][selection_name]
        values = sample["observables"][variable][mask]
        weights = sample["weights"][mask]
        finite = np.isfinite(values) & np.isfinite(weights)
        values = values[finite]
        weights = weights[finite]
        counts, edges = np.histogram(values, bins=bins, range=value_range, weights=weights)
        variances, _ = np.histogram(
            values, bins=bins, range=value_range, weights=weights * weights
        )
        errors = np.sqrt(variances)
        if normalized:
            integral = float(np.sum(counts))
            if integral != 0.0:
                counts = counts / integral
                errors = errors / abs(integral)
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = sample["color"]
        sample_yield = float(np.sum(weights))
        ax.stairs(
            counts,
            edges,
            color=color,
            linewidth=1.5,
            label=f"{sample['label']} ({sample_yield:.4g})",
        )
        nonzero = errors > 0.0
        ax.errorbar(
            centers[nonzero],
            counts[nonzero],
            yerr=errors[nonzero],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=1.5,
        )
        positive_y.extend(counts[counts > 0.0])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Unit-normalized events / bin" if normalized else "Expected events / bin")
    ax.set_xlim(value_range)
    if log_y and positive_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=float(np.min(positive_y)) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = output_dir / f"{variable}_{selection_name}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def write_plots(samples, output_dir, bins, log_y):
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = output_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Creating {4 * len(PLOT_VARIABLES)} plots in {output_dir}...",
        flush=True,
    )
    written = 0
    for variable, (xlabel, fixed_range) in PLOT_VARIABLES.items():
        value_range = histogram_range(samples, variable, fixed_range)
        for selection_name in ("before_pps", "after_pps"):
            plot_variable(
                samples,
                variable,
                xlabel,
                value_range,
                bins,
                selection_name,
                output_dir,
                log_y,
            )
            written += 1
            plot_variable(
                samples,
                variable,
                xlabel,
                value_range,
                bins,
                selection_name,
                normalized_dir,
                log_y,
                normalized=True,
            )
            written += 1
    return written


def main():
    args = parse_args()
    validate_args(args)
    pps_path = resolve_path(args.pps_config, base=ROOT)
    print(f"Loading PPS acceptance from {pps_path}", flush=True)
    pps_config = load_pps_config(pps_path)
    print(
        f"PPS selection includes {PPS_MASS_RANGE_GEV[0]:g} <= M_X <= "
        f"{PPS_MASS_RANGE_GEV[1]:g} GeV",
        flush=True,
    )
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else ROOT / "analysis" / "output" / "compare_parton_level" / args.process
    )

    samples = [
        load_sample(sample, pps_config, args.max_events)
        for sample in resolve_samples(args)
    ]
    written = write_plots(samples, output_dir, args.bins, args.log_y)
    print(f"Wrote {written} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
