#!/usr/bin/env python3
"""Plot saved JetPUPPI calibration and validation results."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_INPUT = HERE / "output/stage2_3"
PROFILE_FIELDS = ("leading_pt", "subleading_pt", "dijet_mass")
RESIDUAL_FIELDS = ("mass_ratio", "delta_pt", "delta_px", "delta_py")
JET_ACTIVITY_SELECTIONS = ("truth_matched", "two_hardest")
JET_ACTIVITY_FIELDS = (
    "leading_pt",
    "subleading_pt",
    "all_pt",
    "dijet_ht_fraction",
)
COLORS = {
    "Hbb": "#009E73",
    "QCDbb": "#CC79A7",
    "HardQCD": "#E69F00",
    "gen": "#0072B2",
    "raw": "#777777",
    "corrected": "#D55E00",
    "smeared": "#56B4E9",
    "expected": "#000000",
}

sys.path.insert(0, str(HERE))
import jet_calibration as calibration  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot destination; defaults to --input-dir",
    )
    return parser.parse_args()


def ensure_runtime():
    if os.environ.get("HIGGS_CEP_JET_PLOTTING_ENV") == "1":
        return
    setup = REPO / "setup_env.sh"
    command = "\n".join(
        (
            f"source {shlex.quote(str(setup))}",
            "export HIGGS_CEP_JET_PLOTTING_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        )
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


def eta_axes(plt, eta_edges):
    """Lay out one panel per absolute-eta bin."""
    count = len(eta_edges) - 1
    columns = min(3, count)
    rows = -(-count // columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4.0 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = list(axes.flat)
    for axis in flat[count:]:
        axis.set_visible(False)
    return figure, flat[:count]


def plot_key(fsr_state, source_sample, target_sample, kind, *parts):
    return "__".join(
        (fsr_state, f"{source_sample}_map_on_{target_sample}", kind, *parts)
    )


def expected_plot_keys(summary):
    keys = set()
    for fsr_state, targets in summary["validation"].items():
        for target_sample, target in targets.items():
            for source_sample, source in target["source_maps"].items():
                if source.get("event_closure") is None:
                    continue
                for variant in ("gen", "raw", "corrected", "smeared"):
                    for field in PROFILE_FIELDS:
                        keys.add(
                            plot_key(
                                fsr_state,
                                source_sample,
                                target_sample,
                                "profile",
                                variant,
                                field,
                            )
                        )
                for variant in ("raw", "corrected", "smeared"):
                    for field in RESIDUAL_FIELDS:
                        keys.add(
                            plot_key(
                                fsr_state,
                                source_sample,
                                target_sample,
                                "residual",
                                variant,
                                field,
                            )
                        )
                if source.get("jet_activity") is not None:
                    for selection in JET_ACTIVITY_SELECTIONS:
                        for variant in ("gen", "raw", "corrected", "smeared"):
                            for field in JET_ACTIVITY_FIELDS:
                                keys.add(
                                    plot_key(
                                        fsr_state,
                                        source_sample,
                                        target_sample,
                                        "jet_activity",
                                        selection,
                                        variant,
                                        field,
                                    )
                                )
    return keys


def save_plot(plt, figure, output_path):
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved plot: {output_path}", flush=True)


def add_legend(axis, **kwargs):
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(**kwargs)


def sample_color(sample, index):
    return COLORS.get(sample, f"C{index}")


def plot_correction_maps(
    np, plt, fsr_state, maps, eta_edges, raw_pt_support, output_path
):
    figure, axes = eta_axes(plt, eta_edges)
    raw_pt = np.geomspace(raw_pt_support[0], np.nextafter(raw_pt_support[1], 0.0), 500)
    for eta_index, axis in enumerate(axes):
        for sample_index, (sample, correction_map) in enumerate(maps.items()):
            color = sample_color(sample, sample_index)
            nodes = [
                node
                for node in correction_map["eta_bins"][eta_index]["nodes"]
                if node["usable"]
            ]
            if not nodes:
                continue
            eta = np.full(raw_pt.shape, 0.5 * sum(eta_edges[eta_index : eta_index + 2]))
            factors, valid = calibration.correction_factors(
                raw_pt, eta, correction_map
            )
            axis.plot(
                raw_pt[valid],
                factors[valid],
                label=sample,
                color=color,
            )
            x = np.asarray([node["raw_reco_pt_median"] for node in nodes])
            y = np.asarray([node["correction_factor"] for node in nodes])
            intervals = np.asarray([node["correction_ci68"] for node in nodes])
            axis.errorbar(
                x,
                y,
                yerr=np.vstack(
                    (
                        np.maximum(0.0, y - intervals[:, 0]),
                        np.maximum(0.0, intervals[:, 1] - y),
                    )
                ),
                fmt="o",
                color=color,
            )
        eta_min, eta_max = eta_edges[eta_index : eta_index + 2]
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"{eta_min:g} <= |eta| < {eta_max:g}")
        axis.set_xlabel(r"Raw JetPUPPI $p_T$ [GeV]")
        axis.set_ylabel("Correction factor")
        axis.set_xscale("log")
        axis.set_xlim(*raw_pt_support)
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False)
    figure.suptitle(f"{fsr_state}: continuous JetPUPPI correction maps")
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def plot_correction_nodes(
    np, plt, fsr_state, maps, eta_edges, gen_pt_edges, output_path
):
    figure, axes = eta_axes(plt, eta_edges)
    for eta_index, axis in enumerate(axes):
        for sample_index, sample in enumerate(maps):
            color = sample_color(sample, sample_index)
            nodes = [
                node
                for node in maps[sample]["eta_bins"][eta_index]["nodes"]
                if node["usable"]
            ]
            if not nodes:
                continue
            x = np.asarray([node["gen_pt_median"] for node in nodes])
            y = np.asarray([node["correction_factor"] for node in nodes])
            intervals = np.asarray([node["correction_ci68"] for node in nodes])
            axis.hlines(
                y,
                [node["gen_pt_min"] for node in nodes],
                [node["gen_pt_max"] for node in nodes],
                color=color,
                linewidth=2.0,
            )
            axis.errorbar(
                x,
                y,
                yerr=np.vstack(
                    (
                        np.maximum(0.0, y - intervals[:, 0]),
                        np.maximum(0.0, intervals[:, 1] - y),
                    )
                ),
                fmt="o",
                label=sample,
                color=color,
            )
        eta_min, eta_max = eta_edges[eta_index : eta_index + 2]
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"{eta_min:g} <= |eta| < {eta_max:g}")
        axis.set_xlabel(r"Generator $p_T$ bin [GeV]")
        axis.set_ylabel("Node correction factor")
        axis.set_xscale("log")
        axis.set_xlim(gen_pt_edges[0], gen_pt_edges[-1])
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False)
    figure.suptitle(f"{fsr_state}: calibration nodes")
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def selected_rows(rows, eta_index, pt_range=None):
    return [
        row
        for row in rows
        if row["eta_bin"] == eta_index
        and row["gen_pt_median"] is not None
        and (
            pt_range is None
            or pt_range[0] <= row["gen_pt_median"] <= pt_range[1]
        )
    ]


def selected_raw_pt_rows(rows, eta_index):
    return [
        row
        for row in rows
        if row["eta_bin"] == eta_index and row["raw_pt_median"] is not None
    ]


def plot_response_closure(np, plt, rows, eta_edges, pt_range, title, output_path):
    figure, axes = eta_axes(plt, eta_edges)
    for eta_index, axis in enumerate(axes):
        selected = selected_rows(rows, eta_index, pt_range)
        for key, label, color in (
            ("raw", "Raw PUPPI (supported)", COLORS["raw"]),
            ("corrected", "Corrected PUPPI", COLORS["corrected"]),
            ("l1_smeared", "L1-smeared GenJet", COLORS["smeared"]),
        ):
            points = [row for row in selected if row[key]["median"] is not None]
            if not points:
                continue
            x = np.asarray([row["gen_pt_median"] for row in points])
            y = np.asarray([row[key]["median"] for row in points])
            if key == "corrected":
                intervals = np.asarray([row[key]["median_ci68"] for row in points])
                axis.errorbar(
                    x,
                    y,
                    yerr=np.vstack(
                        (
                            np.maximum(0.0, y - intervals[:, 0]),
                            np.maximum(0.0, intervals[:, 1] - y),
                        )
                    ),
                    marker="o",
                    label=label,
                    color=color,
                )
            else:
                axis.plot(x, y, marker="o", label=label, color=color)
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        eta_min, eta_max = eta_edges[eta_index : eta_index + 2]
        axis.set_title(f"{eta_min:g} <= |eta| < {eta_max:g}")
        axis.set_xscale("log")
        axis.set_xlabel(r"GenJet $p_T$ [GeV]")
        axis.set_ylabel("Median response")
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def plot_raw_pt_response_closure(np, plt, rows, eta_edges, title, output_path):
    """Plot response closure in the common raw-pT diagnostic bins."""
    figure, axes = eta_axes(plt, eta_edges)
    for eta_index, axis in enumerate(axes):
        selected = selected_raw_pt_rows(rows, eta_index)
        for key, label, color in (
            ("raw", "Raw PUPPI", COLORS["raw"]),
            ("corrected", "Corrected PUPPI", COLORS["corrected"]),
        ):
            points = [row for row in selected if row[key]["median"] is not None]
            if not points:
                continue
            x = np.asarray([row["raw_pt_median"] for row in points])
            y = np.asarray([row[key]["median"] for row in points])
            intervals = np.asarray([row[key]["median_ci68"] for row in points])
            axis.hlines(
                y,
                [row["raw_pt_min"] for row in points],
                [row["raw_pt_max"] for row in points],
                color=color,
                linewidth=2.0,
            )
            axis.errorbar(
                x,
                y,
                yerr=np.vstack(
                    (
                        np.maximum(0.0, y - intervals[:, 0]),
                        np.maximum(0.0, intervals[:, 1] - y),
                    )
                ),
                fmt="o",
                label=label,
                color=color,
            )
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        eta_min, eta_max = eta_edges[eta_index : eta_index + 2]
        axis.set_title(f"{eta_min:g} <= |eta| < {eta_max:g}")
        axis.set_xscale("log")
        axis.set_xlabel(r"Raw JetPUPPI $p_T$ bin [GeV]")
        axis.set_ylabel("Median response")
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False, fontsize=8)
    figure.suptitle(f"{title}: closure in common raw-$p_T$ bins")
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def plot_resolution(np, plt, rows, eta_edges, pt_range, metric, title, output_path):
    figure, axes = eta_axes(plt, eta_edges)
    for eta_index, axis in enumerate(axes):
        selected = selected_rows(rows, eta_index, pt_range)
        for key, label, color in (
            ("raw", "Raw PUPPI (supported)", COLORS["raw"]),
            ("corrected", "Corrected PUPPI", COLORS["corrected"]),
            ("l1_smeared", "L1-smeared GenJet", COLORS["smeared"]),
        ):
            points = [row for row in selected if row[key][metric] is not None]
            if not points:
                continue
            x = np.asarray([row["gen_pt_median"] for row in points])
            y = np.asarray([row[key][metric] for row in points])
            if key == "corrected" and metric == "central68_resolution":
                intervals = np.asarray(
                    [row[key]["central68_resolution_ci68"] for row in points]
                )
                axis.errorbar(
                    x,
                    y,
                    yerr=np.vstack(
                        (
                            np.maximum(0.0, y - intervals[:, 0]),
                            np.maximum(0.0, intervals[:, 1] - y),
                        )
                    ),
                    marker="o",
                    label=label,
                    color=color,
                )
            else:
                axis.plot(x, y, marker="o", label=label, color=color)
        expected = [
            row for row in selected if row["l1_expected_relative_sigma"] is not None
        ]
        axis.plot(
            [row["gen_pt_median"] for row in expected],
            [row["l1_expected_relative_sigma"] for row in expected],
            linestyle="--",
            label=r"L1 $a+b/p_T$ relative sigma",
            color=COLORS["expected"],
        )
        eta_min, eta_max = eta_edges[eta_index : eta_index + 2]
        axis.set_title(f"{eta_min:g} <= |eta| < {eta_max:g}")
        axis.set_xscale("log")
        axis.set_xlabel(r"GenJet $p_T$ [GeV]")
        axis.set_ylabel(
            "Central-68% / median" if metric == "central68_resolution" else "RMS / mean"
        )
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False, fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def finite_array(np, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def positive_edges(np, series):
    populated = [finite_array(np, values) for values in series if len(values)]
    populated = [values for values in populated if values.size]
    upper = max(10.0, 1.05 * float(np.quantile(np.concatenate(populated), 0.995)))
    return np.linspace(0.0, upper, 61)


def plot_event_closure(np, plt, profile, title, output_path):
    fields = (
        ("leading_pt", r"Leading jet $p_T$ [GeV]"),
        ("subleading_pt", r"Subleading jet $p_T$ [GeV]"),
        ("dijet_mass", r"Dijet mass [GeV]"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (field, xlabel) in zip(axes, fields):
        edges = positive_edges(np, [profile[key][field] for key in profile])
        for variant, label in (
            ("gen", "GenJet"),
            ("raw", "Raw PUPPI"),
            ("corrected", "Corrected PUPPI"),
            ("smeared", "L1-smeared GenJet"),
        ):
            values = finite_array(np, profile[variant][field])
            if values.size:
                axis.hist(
                    values,
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    color=COLORS[variant],
                    label=f"{label} (N={values.size})",
                )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Normalized entries")
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False, fontsize=7)
    figure.suptitle(f"{title}: matched hard-b pair")
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def plot_jet_activity_pt(np, plt, profile, selection, title, output_path):
    all_label = (
        r"All truth-matched jet $p_T$ [GeV]"
        if selection == "truth_matched"
        else r"All jet $p_T$ [GeV]"
    )
    fields = (
        ("leading_pt", r"Leading jet $p_T$ [GeV]"),
        ("subleading_pt", r"Subleading jet $p_T$ [GeV]"),
        ("all_pt", all_label),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (field, xlabel) in zip(axes, fields):
        values_by_variant = [profile[key][field] for key in profile]
        populated = [finite_array(np, values) for values in values_by_variant]
        populated = [values for values in populated if values.size]
        upper = max(
            30.0,
            1.05 * float(np.quantile(np.concatenate(populated), 0.995)),
        )
        edges = np.linspace(20.0, upper, 61)
        for variant, label in (
            ("gen", "GenJet"),
            ("raw", "Raw PUPPI"),
            ("corrected", "Corrected PUPPI"),
            ("smeared", "L1-smeared GenJet"),
        ):
            values = finite_array(np, profile[variant][field])
            if values.size:
                axis.hist(
                    values,
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    color=COLORS[variant],
                    label=f"{label} (N={values.size})",
                )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Normalized entries")
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False, fontsize=7)
    selection_label = (
        "truth-matched hard-flavor jets"
        if selection == "truth_matched"
        else "two hardest jets"
    )
    figure.suptitle(
        rf"{title}: {selection_label}, $p_T>20$ GeV and $|\eta|<2.4$"
    )
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def plot_dijet_ht_fraction(np, plt, profile, selection, title, output_path):
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    edges = np.linspace(0.0, 1.0, 51)
    for variant, label in (
        ("gen", "GenJet"),
        ("raw", "Raw PUPPI"),
        ("corrected", "Corrected PUPPI"),
        ("smeared", "L1-smeared GenJet"),
    ):
        values = finite_array(np, profile[variant]["dijet_ht_fraction"])
        if values.size:
            axis.hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=COLORS[variant],
                label=f"{label} (N={values.size})",
            )
    axis.set_xlabel(r"$(p_{T,1}+p_{T,2})/H_T$")
    axis.set_ylabel("Normalized entries")
    axis.set_xlim(0.0, 1.02)
    axis.grid(alpha=0.2)
    add_legend(axis, frameon=False, fontsize=8)
    selection_label = (
        "truth-matched hard-flavor pair"
        if selection == "truth_matched"
        else "two hardest jets"
    )
    figure.suptitle(
        rf"{title}: {selection_label}" "\n"
        rf"$H_T=\sum p_T$ for jets with $p_T>20$ GeV and $|\eta|<2.4$"
    )
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def symmetric_edges(np, series):
    values = [finite_array(np, item) for item in series]
    values = [item for item in values if item.size]
    extent = max(1.0, float(np.quantile(np.abs(np.concatenate(values)), 0.995)))
    return np.linspace(-1.05 * extent, 1.05 * extent, 61)


def plot_dijet_residuals(np, plt, residuals, title, output_path):
    fields = (
        ("mass_ratio", r"$m_{jj}^{reco}/m_{jj}^{gen}$"),
        ("delta_pt", r"$p_{T,jj}^{reco}-p_{T,jj}^{gen}$ [GeV]"),
        ("delta_px", r"$p_{x,jj}^{reco}-p_{x,jj}^{gen}$ [GeV]"),
        ("delta_py", r"$p_{y,jj}^{reco}-p_{y,jj}^{gen}$ [GeV]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (field, xlabel) in zip(axes.flat, fields):
        if field == "mass_ratio":
            edges = np.linspace(0.0, 2.5, 61)
            reference = 1.0
        else:
            edges = symmetric_edges(np, [residuals[key][field] for key in residuals])
            reference = 0.0
        for variant, label in (
            ("raw", "Raw PUPPI (supported)"),
            ("corrected", "Corrected PUPPI"),
            ("smeared", "L1-smeared GenJet"),
        ):
            values = finite_array(np, residuals[variant][field])
            if values.size:
                axis.hist(
                    values,
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    color=COLORS[variant],
                    label=f"{label} (N={values.size})",
                )
        axis.axvline(reference, color="black", linestyle=":", linewidth=1.0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Normalized entries")
        axis.grid(alpha=0.2)
        add_legend(axis, frameon=False, fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    save_plot(plt, figure, output_path)


def profile_data(data, fsr_state, source_sample, target_sample):
    return {
        variant: {
            field: data[
                plot_key(
                    fsr_state,
                    source_sample,
                    target_sample,
                    "profile",
                    variant,
                    field,
                )
            ]
            for field in PROFILE_FIELDS
        }
        for variant in ("gen", "raw", "corrected", "smeared")
    }


def residual_data(data, fsr_state, source_sample, target_sample):
    return {
        variant: {
            field: data[
                plot_key(
                    fsr_state,
                    source_sample,
                    target_sample,
                    "residual",
                    variant,
                    field,
                )
            ]
            for field in RESIDUAL_FIELDS
        }
        for variant in ("raw", "corrected", "smeared")
    }


def jet_activity_data(data, fsr_state, source_sample, target_sample, selection):
    return {
        variant: {
            field: data[
                plot_key(
                    fsr_state,
                    source_sample,
                    target_sample,
                    "jet_activity",
                    selection,
                    variant,
                    field,
                )
            ]
            for field in JET_ACTIVITY_FIELDS
        }
        for variant in ("gen", "raw", "corrected", "smeared")
    }


def run(args):
    import matplotlib
    import numpy as np
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    corrections_path = input_dir / "corrections.yaml"
    summary_path = input_dir / "summary.yaml"
    plot_data_path = input_dir / "plot_data.npz"
    corrections = yaml.safe_load(corrections_path.read_text(encoding="utf-8"))
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    if corrections.get("schema_version") != calibration.SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported correction schema {corrections.get('schema_version')}; "
            f"expected {calibration.SCHEMA_VERSION}"
        )
    expected_keys = expected_plot_keys(summary)
    with np.load(plot_data_path, allow_pickle=False) as saved:
        missing = sorted(expected_keys - set(saved.files))
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} plot arrays in {plot_data_path}; first is {missing[0]}"
            )
        data = {key: np.asarray(saved[key]) for key in expected_keys}

    configuration = summary["configuration"]
    eta_edges = configuration["abs_eta_edges"]
    validation_eta_edges = configuration["validation_abs_eta_edges"]
    closure_pt_range = (
        configuration["calibration_gen_pt_edges"][0],
        configuration["calibration_gen_pt_edges"][-1],
    )
    calibration_edges = configuration["calibration_gen_pt_edges"]
    raw_pt_support = configuration["raw_pt_support"]
    plot_count = 0
    for fsr_state, maps in corrections["maps"].items():
        state_dir = output_dir / fsr_state
        state_dir.mkdir(parents=True, exist_ok=True)
        plot_correction_maps(
            np,
            plt,
            fsr_state,
            maps,
            eta_edges,
            raw_pt_support,
            state_dir / "correction_maps.png",
        )
        plot_correction_nodes(
            np,
            plt,
            fsr_state,
            maps,
            eta_edges,
            calibration_edges,
            state_dir / "correction_nodes.png",
        )
        plot_count += 2
        for target_sample, target in summary["validation"][fsr_state].items():
            for source_sample, source in target["source_maps"].items():
                comparison_dir = state_dir / f"{source_sample}_map_on_{target_sample}"
                comparison_dir.mkdir(parents=True, exist_ok=True)
                title = f"{fsr_state}: {source_sample} map on {target_sample}"
                rows = source["response_bins"]
                plot_response_closure(
                    np,
                    plt,
                    rows,
                    validation_eta_edges,
                    closure_pt_range,
                    title,
                    comparison_dir / "response_closure.png",
                )
                plot_raw_pt_response_closure(
                    np,
                    plt,
                    source["raw_pt_response_bins"],
                    eta_edges,
                    title,
                    comparison_dir / "response_closure_raw_pt.png",
                )
                plot_resolution(
                    np,
                    plt,
                    rows,
                    validation_eta_edges,
                    closure_pt_range,
                    "central68_resolution",
                    title,
                    comparison_dir / "resolution_central68.png",
                )
                plot_resolution(
                    np,
                    plt,
                    rows,
                    validation_eta_edges,
                    closure_pt_range,
                    "rms_over_mean",
                    title,
                    comparison_dir / "resolution_rms.png",
                )
                plot_count += 4
                if source.get("event_closure") is not None:
                    plot_event_closure(
                        np,
                        plt,
                        profile_data(data, fsr_state, source_sample, target_sample),
                        title,
                        comparison_dir / "event_closure.png",
                    )
                    plot_dijet_residuals(
                        np,
                        plt,
                        residual_data(data, fsr_state, source_sample, target_sample),
                        title,
                        comparison_dir / "matched_dijet_closure.png",
                    )
                    plot_count += 2
                if source.get("jet_activity") is not None:
                    for selection in JET_ACTIVITY_SELECTIONS:
                        activity = jet_activity_data(
                            data,
                            fsr_state,
                            source_sample,
                            target_sample,
                            selection,
                        )
                        plot_jet_activity_pt(
                            np,
                            plt,
                            activity,
                            selection,
                            title,
                            comparison_dir / f"jet_pt_{selection}.png",
                        )
                        plot_dijet_ht_fraction(
                            np,
                            plt,
                            activity,
                            selection,
                            title,
                            comparison_dir / f"dijet_ht_fraction_{selection}.png",
                        )
                        plot_count += 2
    print(f"Wrote {plot_count} plots beneath {output_dir}")


def main():
    run(parse_args())


if __name__ == "__main__":
    ensure_runtime()
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
