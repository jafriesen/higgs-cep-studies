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

from common.config_utils import discover_event_files, resolve_path  # noqa: E402
from common.path_helper import (  # noqa: E402
    generation_campaign_config,
    generation_campaign_root,
    generation_config,
    generation_process_config,
    generation_stage_root,
)


LUMI_FB = 3000.0

PLOT_VARIABLES = {
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
        "range": (95, 145),
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
        description="Plot weighted generator parton-level dijet observables."
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
        help="Output plot directory. Defaults to analysis/output/partons/<generator>/<process>.",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum LHE files to load")
    parser.add_argument("--max-events", type=int, default=None, help="Maximum total events to load")
    return parser.parse_args()


def validate_args(args):
    if args.bins <= 0:
        raise RuntimeError("--bins must be > 0")
    if args.max_files is not None and args.max_files <= 0:
        raise RuntimeError("--max-files must be > 0")
    if args.max_events is not None and args.max_events <= 0:
        raise RuntimeError("--max-events must be > 0")


def default_output_dir(generator, process):
    return ROOT / "analysis" / "output" / "partons" / generator / process


def process_parton_ids(process_config):
    jet_type = process_config.get("jet_type")
    if jet_type == "b":
        return {5, -5}
    if jet_type == "c":
        return {4, -4}
    if jet_type == "q":
        return {-3, -2, -1, 1, 2, 3}
    if jet_type == "g":
        return {21}
    raise RuntimeError(f"Unsupported jet_type '{jet_type}'")


def generator_event_dir(generator, process_name, campaign):
    campaign_name, _ = generation_campaign_config(generator, process_name, campaign)
    event_dir = (
        generation_campaign_root(generator, process_name, campaign_name)
        / generation_config(generator)["generation_dir"]
        / "evrecs"
    )
    return campaign_name, event_dir


def p4(particle):
    return vector.obj(px=particle.px, py=particle.py, pz=particle.pz, E=particle.e)


def event_partons(event, allowed_ids):
    return [
        particle
        for particle in event.particles
        if particle.status == 1 and particle.id in allowed_ids
    ]


def first_event_has_partons(files, allowed_ids):
    for filename in files:
        lhe = pylhe.LHEFile.fromfile(filename, generator=True, with_attributes=False)
        for event in lhe.events:
            return len(event_partons(event, allowed_ids)) == 2
    return False


def resolve_files(generator, process_name, campaign, allowed_ids, max_files):
    campaign_name, event_dir = generator_event_dir(generator, process_name, campaign)
    files = discover_event_files(event_dir, max_files)
    label_suffix = campaign_name

    if generator == "fpmc" and not first_event_has_partons(files, allowed_ids):
        stage_root = generation_stage_root(
            generator,
            process_name,
            campaign_name,
            "parton-pythia",
        )
        files = discover_event_files(stage_root / "lhe", max_files)
        label_suffix = f"{campaign_name}/{stage_root.name}"

    return campaign_name, label_suffix, files


def file_init(filename):
    return pylhe.LHEFile.fromfile(filename, generator=True, with_attributes=False).init


def sample_cross_section_fb(process_config, inits, sample_label):
    configured = process_config.get("xsec_fb")
    if configured is not None:
        xsec_fb = float(configured)
        source = "config"
    else:
        xsecs_pb = [
            sum(float(proc.xSection) for proc in init.procInfo) for init in inits
        ]
        if not xsecs_pb or not all(xsec > 0.0 for xsec in xsecs_pb):
            raise RuntimeError(
                f"{sample_label} has no positive configured or LHE cross section"
            )
        xsec_fb = 1000.0 * float(np.mean(xsecs_pb))
        source = "LHE"
    if xsec_fb <= 0.0:
        raise RuntimeError(f"{sample_label} cross section must be > 0")
    return xsec_fb, source


def empty_columns():
    return {variable: [] for variable in PLOT_VARIABLES}


def append_event(columns, leading, subleading, dijet):
    columns["leading_jet_pt"].append(leading.pt)
    columns["subleading_jet_pt"].append(subleading.pt)
    columns["leading_jet_pt_over_mjj"].append(leading.pt / dijet.mass if dijet.mass > 0.0 else math.nan)
    columns["subleading_jet_pt_over_mjj"].append(subleading.pt / dijet.mass if dijet.mass > 0.0 else math.nan)
    columns["leading_jet_eta"].append(leading.eta)
    columns["subleading_jet_eta"].append(subleading.eta)
    columns["leading_jet_phi"].append(leading.phi)
    columns["subleading_jet_phi"].append(subleading.phi)
    columns["dijet_mass"].append(dijet.mass)
    columns["dijet_pt"].append(dijet.pt)
    columns["dijet_eta"].append(dijet.eta)
    columns["dijet_rapidity"].append(dijet.rapidity)
    columns["dijet_phi"].append(dijet.phi)


def append_weight(columns, weight):
    for variable in columns:
        columns[variable].append(weight)


def sample_label(generator, process_name, label_suffix):
    return f"{generator} {process_name} {label_suffix}"


def load_sample(generator, process_name, process_config, campaign_name, label_suffix, files, max_events):
    label = sample_label(generator, process_name, label_suffix)
    allowed_ids = process_parton_ids(process_config)
    columns = empty_columns()
    weight_columns = empty_columns()
    raw_weights = []
    inits = []
    stop = False

    print(f"Loading {label} from {len(files)} file(s)...", flush=True)
    for file_index, filename in enumerate(files, start=1):
        print(f"  Reading file {file_index}/{len(files)}: {filename.name}", flush=True)
        lhe = pylhe.LHEFile.fromfile(filename, generator=True, with_attributes=False)
        init = lhe.init
        inits.append(init)
        for event_index, event in enumerate(lhe.events):
            if max_events is not None and len(raw_weights) >= max_events:
                stop = True
                break
            partons = event_partons(event, allowed_ids)
            if len(partons) != 2:
                raise RuntimeError(
                    f"{filename}, event {event_index} has {len(partons)} "
                    "final-state partons; expected 2"
                )

            leading, subleading = sorted(
                (p4(partons[0]), p4(partons[1])),
                key=lambda v: v.pt,
                reverse=True,
            )
            # print(
            #     f"    Event {event_index}: "
            #     f"leading pT={leading.pt:.3f}, subleading pT={subleading.pt:.3f}, ",
            #     flush=True,
            # )
            dijet = leading + subleading
            raw_weight = float(event.eventinfo.weight)
            raw_weights.append(raw_weight)
            append_event(columns, leading, subleading, dijet)
            append_weight(weight_columns, raw_weight)
        if stop:
            break

    if process_config.get("xsec_fb") is None:
        inits.extend(file_init(filename) for filename in files[len(inits):])
    xsec_fb, xsec_source = sample_cross_section_fb(process_config, inits, label)
    if not raw_weights:
        raise RuntimeError(f"{label} has no events")
    total_weight = float(np.sum(raw_weights))
    if not math.isfinite(total_weight) or total_weight == 0.0:
        raise RuntimeError(f"{label} has zero or non-finite total event weight")

    scale = xsec_fb * LUMI_FB * float(process_config.get("weight", 1.0)) / total_weight
    observables = {
        variable: np.asarray(values, dtype=np.float64)
        for variable, values in columns.items()
    }
    weights = {
        variable: np.asarray(values, dtype=np.float64) * scale
        for variable, values in weight_columns.items()
    }
    print(
        f"{label}: files={len(files)}, events={len(raw_weights)}, "
        f"xsec={xsec_fb:.8g} fb ({xsec_source}), "
        f"yield={np.sum(weights['dijet_mass']):.8g}"
    )
    return {
        "name": f"{generator}:{process_name}",
        "campaign": campaign_name,
        "label": label,
        "color": COLORS[generator],
        "observables": observables,
        "weights": weights,
    }


def finite_values(values, weights):
    finite = np.isfinite(values) & np.isfinite(weights)
    return values[finite], weights[finite]


def histogram_range(samples, variable, fixed_range):
    if fixed_range is not None:
        return fixed_range
    values = []
    for sample in samples:
        data, _weights = finite_values(
            sample["observables"][variable],
            sample["weights"][variable],
        )
        if data.size:
            values.append(data)
    if not values:
        return None
    combined = np.concatenate(values)
    low = float(np.min(combined))
    high = float(np.max(combined))
    if low == high:
        padding = 0.05 * abs(low) if low else 1.0
    else:
        padding = 0.02 * (high - low)
    return low - padding, high + padding


def normalized_output_name(output_name):
    path = Path(output_name)
    return f"{path.stem}_normalized{path.suffix}"


def plot_variable(samples, variable, options, output_dir, bins, log_y, stacked, normalized=False):
    value_range = histogram_range(samples, variable, options["range"])
    if value_range is None:
        print(f"Warning: no finite values for {variable}; skipping")
        return False

    hist_bins = options["bins"] or bins
    rows = []
    edges = None
    for sample in samples:
        values, weights = finite_values(
            sample["observables"][variable],
            sample["weights"][variable],
        )
        counts, edges = np.histogram(
            values,
            bins=hist_bins,
            range=value_range,
            weights=weights,
        )
        variances, _ = np.histogram(
            values,
            bins=hist_bins,
            range=value_range,
            weights=weights * weights,
        )
        if normalized:
            integral = float(np.sum(counts))
            if integral != 0.0:
                counts = counts / integral
                variances = variances / (integral * integral)
        rows.append({"sample": sample, "counts": counts, "errors": np.sqrt(variances)})

    if not any(np.any(row["counts"]) for row in rows):
        print(f"Warning: no entries in range for {variable}; skipping")
        return False

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    positive_y = []
    if stacked and not normalized:
        bottom = np.zeros_like(rows[0]["counts"])
        for row in rows:
            counts = row["counts"]
            sample = row["sample"]
            ax.bar(
                edges[:-1],
                counts,
                width=widths,
                bottom=bottom,
                align="edge",
                label=f"{sample['label']} ({np.sum(counts):.4g})",
                color=sample["color"],
                alpha=0.75,
                linewidth=0.8,
                edgecolor="#333333",
            )
            bottom += counts
            positive_y.extend(counts[counts > 0.0])
    else:
        for row in rows:
            counts = row["counts"]
            errors = row["errors"]
            sample = row["sample"]
            label = sample["label"] if normalized else f"{sample['label']} ({np.sum(counts):.4g})"
            color = sample["color"]
            ax.stairs(counts, edges, color=color, linewidth=1.5, label=label)
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

    ax.set_xlabel(options["xlabel"])
    ax.set_ylabel("Normalized events / bin" if normalized else "Expected events / bin")
    ax.set_xlim(value_range)
    if log_y and positive_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=float(np.min(positive_y)) * 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_name = normalized_output_name(options["output"]) if normalized else options["output"]
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    return True


def write_plots(samples, plot_variables, output_dir, bins, log_y, stacked):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for variable, options in plot_variables.items():
        if plot_variable(samples, variable, options, output_dir, bins, log_y, stacked):
            written += 1
        if plot_variable(samples, variable, options, output_dir, bins, log_y, stacked, normalized=True):
            written += 1
    return written


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
    process_config = generation_process_config(args.generator, args.process)
    allowed_ids = process_parton_ids(process_config)
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else default_output_dir(args.generator, args.process)
    )

    campaign_name, label_suffix, files = resolve_files(
        args.generator,
        args.process,
        args.campaign,
        allowed_ids,
        args.max_files,
    )
    samples = [
        load_sample(
            args.generator,
            args.process,
            process_config,
            campaign_name,
            label_suffix,
            files,
            args.max_events,
        )
    ]
    written = write_plots(
        samples,
        process_plot_variables(process_config),
        output_dir,
        args.bins,
        args.log_y,
        args.stacked,
    )
    print(f"Wrote {written} plot(s)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
