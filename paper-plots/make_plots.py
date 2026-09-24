#!/usr/bin/env python3
"""Render the dedicated mplhep figures for the Higgs CEP paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.interpolate import PchipInterpolator
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper-plots/output"
TIMING_DATA = ROOT / "paper-plots/data/trigger-timing"
TIMINGS_PS = np.asarray((1, 3, 5, 8, 10, 15, 20, 30), dtype=float)
REFERENCE_CROSS_SECTION_FB = 0.4

COLORS = {
    "signal": "#3f90da",
    "exclusive": "#ffa90e",
    "inclusive": "#bd1f01",
    "photon": "#832db6",
    "resonant": "#00a6a6",
    "timing3": "#e76300",
    "timing10": "#3f90da",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style():
    hep.style.use(hep.style.plothist)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def plot_header(ax):
    ax.text(
        0.5,
        1.015,
        r"$\mathcal{L}_{\mathrm{int}} = 3\,\mathrm{ab}^{-1},\quad "
        r"\sqrt{s}=14\,\mathrm{TeV}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )


def save_figure(figure, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.pdf")
    figure.savefig(output_dir / f"{name}.png", dpi=220)
    plt.close(figure)


def timing_report_path(kind: str, timing: int) -> Path:
    if timing in (3, 10):
        if kind == "minbias":
            return ROOT / f"output/minbias/minbias_trigger_rate_{timing}ps_31mhz.json"
        return (
            ROOT
            / "output/trigger/hardqcd_pthat10_150k_v1"
            / f"dijet_rate_20_20_{timing}ps.json"
        )
    return TIMING_DATA / f"{kind}_{timing}ps.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_timing_scan():
    series = {
        "proton": {"rate": [], "error": []},
        "dijet": {"rate": [], "error": []},
        "rapidity": {"rate": [], "error": []},
    }
    sources = []
    for value in TIMINGS_PS:
        timing = int(value)
        minbias_path = timing_report_path("minbias", timing)
        dijet_path = timing_report_path("dijet", timing)
        minbias = load_json(minbias_path)
        dijet = load_json(dijet_path)
        proton = minbias["criteria"]["pps_mass_z"]
        central = dijet["criteria"]["pps_mass_z__dy_none"]
        rapidity = dijet["criteria"]["pps_mass_z__dy_0.25"]
        for destination, result, error_key in (
            (series["proton"], proton, "rate_error_hz"),
            (series["dijet"], central, "rate_total_error_hz"),
            (series["rapidity"], rapidity, "rate_total_error_hz"),
        ):
            destination["rate"].append(result["rate_hz"] / 1.0e3)
            destination["error"].append(result[error_key] / 1.0e3)
        sources.extend((str(minbias_path), str(dijet_path)))
    for values in series.values():
        values["rate"] = np.asarray(values["rate"])
        values["error"] = np.asarray(values["error"])
    return series, sources


def plot_trigger_timing(output_dir: Path):
    series, sources = load_timing_scan()
    figure, ax = plt.subplots(figsize=(5.2, 4.5))
    dense = np.linspace(TIMINGS_PS[0], TIMINGS_PS[-1], 500)
    definitions = (
        ("proton", "PPS mass + vertex", COLORS["signal"], "o"),
        ("dijet", r"+ $p_T>20/20$ GeV dijet", COLORS["inclusive"], "s"),
        ("rapidity", r"+ $|y_X-y_{jj}|<0.25$", "#009e73", "^"),
    )
    for key, label, color, marker in definitions:
        values = series[key]
        interpolation = PchipInterpolator(TIMINGS_PS, values["rate"])
        ax.plot(dense, interpolation(dense), color=color)
        ax.errorbar(
            TIMINGS_PS,
            values["rate"],
            yerr=values["error"],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=5.5,
            capsize=2.5,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(TIMINGS_PS)
    ax.set_xlabel("Single-proton time resolution [ps]")
    ax.set_ylabel("Trigger rate [kHz]")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="lower right",
        fontsize=8.8,
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.7,
    )
    plot_header(ax)
    figure.tight_layout()
    save_figure(figure, output_dir, "trigger_timing_scan")
    return {
        "timing_ps": TIMINGS_PS.astype(int).tolist(),
        "series_khz": {
            key: {
                "rate": values["rate"].tolist(),
                "error": values["error"].tolist(),
            }
            for key, values in series.items()
        },
        "sources": sources,
        "interpolation": "shape-preserving piecewise cubic (PCHIP)",
    }


def hbb_feature_histograms():
    data_dir = ROOT / "analysis/MVA_new/data"
    with (data_dir / "metadata.yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    names = list(metadata["features"])
    matrix = np.load(data_dir / "x.npy", mmap_mode="r")
    classes = np.load(data_dir / "class.npy", mmap_mode="r")
    weights = np.load(data_dir / "physical_weight.npy", mmap_mode="r")
    specifications = {
        "tracks": {
            "index": names.index("n_tracks_interjet"),
            "edges": np.arange(-0.5, 30.5, 1.0),
        },
        "delta_eta": {
            "index": names.index("delta_eta_jj"),
            "edges": np.linspace(0.0, 4.5, 31),
        },
    }
    histograms = {
        feature: {class_id: np.zeros(len(spec["edges"]) - 1) for class_id in range(3)}
        for feature, spec in specifications.items()
    }
    chunk_size = 500_000
    for start in range(0, len(classes), chunk_size):
        stop = min(start + chunk_size, len(classes))
        chunk_class = np.asarray(classes[start:stop])
        chunk_weight = np.asarray(weights[start:stop])
        for feature, spec in specifications.items():
            values = np.asarray(matrix[start:stop, spec["index"]])
            if feature == "tracks":
                values = np.minimum(values, spec["edges"][-1] - 1.0e-6)
            for class_id in range(3):
                selected = chunk_class == class_id
                if np.any(selected):
                    histograms[feature][class_id] += np.histogram(
                        values[selected],
                        bins=spec["edges"],
                        weights=chunk_weight[selected],
                    )[0]
    return specifications, histograms


def leading_jet_pt_histograms(edges):
    """Leading-jet pT is not in the reduced dataset; rebuild it from the full
    feature cache as jet1_pt_over_mjj * dijet_mass.  The reduced dataset is the
    cache without the dropped MadGraph campaign, in cache order, so its
    restitched physical weights apply row by row."""
    data_dir = ROOT / "analysis/MVA_new/data"
    with (data_dir / "metadata.yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    cache_dir = Path(metadata["source_cache"])
    with (cache_dir / "metadata.yaml").open(encoding="utf-8") as handle:
        names = list(yaml.safe_load(handle)["features"])
    cache_classes = np.asarray(np.load(cache_dir / "class.npy", mmap_mode="r"))
    campaigns = np.load(cache_dir / "source_campaign.npy", mmap_mode="r")
    matrix = np.load(cache_dir / "x.npy", mmap_mode="r")
    classes = np.asarray(np.load(data_dir / "class.npy", mmap_mode="r"))
    weights = np.load(data_dir / "physical_weight.npy", mmap_mode="r")
    ratio_index = names.index("jet1_pt_over_mjj")
    mass_index = names.index("dijet_mass")
    histograms = {class_id: np.zeros(len(edges) - 1) for class_id in range(3)}
    chunk_size = 500_000
    offset = 0
    for start in range(0, len(cache_classes), chunk_size):
        stop = min(start + chunk_size, len(cache_classes))
        chunk_class = cache_classes[start:stop]
        keep = ~(
            (chunk_class == 2)
            & (np.asarray(campaigns[start:stop]) == metadata["dropped_campaign"])
        )
        kept = int(np.sum(keep))
        if not np.array_equal(chunk_class[keep], classes[offset : offset + kept]):
            raise RuntimeError("Cache rows do not line up with the reduced dataset")
        pt = np.asarray(matrix[start:stop, ratio_index], dtype=float)[keep] * np.asarray(
            matrix[start:stop, mass_index], dtype=float
        )[keep]
        pt = np.minimum(pt, edges[-1] - 1.0e-6)
        chunk_weight = np.asarray(weights[offset : offset + kept])
        for class_id in range(3):
            selected = chunk_class[keep] == class_id
            histograms[class_id] += np.histogram(
                pt[selected], bins=edges, weights=chunk_weight[selected]
            )[0]
        offset += kept
    if offset != len(classes):
        raise RuntimeError("Cache rows do not line up with the reduced dataset")
    return histograms


def delta_y_histograms():
    """Written by paper-plots/delta_y_histograms.py, before the |Delta y| cut."""
    with np.load(ROOT / "paper-plots/data/delta_y_histograms.npz") as source:
        edges = np.asarray(source["edges"], dtype=float)
        histograms = np.asarray(source["histograms"], dtype=float)
    return edges, {class_id: histograms[class_id] for class_id in range(3)}


def normalized(histogram):
    total = float(np.sum(histogram))
    if total <= 0.0:
        raise RuntimeError("Cannot normalize an empty histogram")
    return histogram / total


def draw_feature_histogram(ax, histograms, edges, classes):
    for class_id, label, color, linestyle in classes:
        hep.histplot(
            normalized(histograms[class_id]),
            bins=edges,
            ax=ax,
            histtype="step",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )


def plot_feature(output_dir, name, histograms, edges, classes, xlabel, xlim, legend_loc, fontsize, log):
    figure, ax = plt.subplots(figsize=(5.0, 5.0))
    draw_feature_histogram(ax, histograms, edges, classes)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of events / bin")
    ax.grid(True, which="both" if log else "major", alpha=0.2)
    ax.legend(
        frameon=False,
        loc=legend_loc,
        fontsize=fontsize,
        labelspacing=0.3,
        handlelength=1.7,
    )
    plot_header(ax)
    figure.tight_layout()
    save_figure(figure, output_dir, name)


def plot_discriminants(output_dir: Path):
    specifications, histograms = hbb_feature_histograms()
    physics_classes = (
        (0, r"CEP $H\to b\bar b$", COLORS["signal"], "-"),
        (1, r"CEP $b\bar b$", COLORS["exclusive"], "--"),
        (2, r"$b\bar b + pp$", COLORS["inclusive"], ":"),
    )

    plot_feature(
        output_dir,
        "interjet_track_multiplicity",
        histograms["tracks"],
        specifications["tracks"]["edges"],
        physics_classes,
        "Interjet track multiplicity",
        (-0.5, 20.5),
        "upper right",
        8.5,
        log=True,
    )
    delta_y_edges, delta_y = delta_y_histograms()
    plot_feature(
        output_dir,
        "proton_dijet_delta_y",
        delta_y,
        delta_y_edges,
        physics_classes,
        r"$y_X-y_{jj}$",
        (-0.3, 0.3),
        "upper right",
        8.5,
        log=False,
    )
    plot_feature(
        output_dir,
        "dijet_delta_eta",
        histograms["delta_eta"],
        specifications["delta_eta"]["edges"],
        physics_classes,
        r"Dijet $|\Delta\eta_{jj}|$",
        (0.0, 4.5),
        "upper right",
        10.0,
        log=False,
    )
    pt_edges = np.arange(15.0, 150.1, 2.5)
    plot_feature(
        output_dir,
        "leading_jet_pt",
        leading_jet_pt_histograms(pt_edges),
        pt_edges,
        physics_classes,
        r"Leading jet $p_T$ [GeV]",
        (15.0, 90.0),
        "upper right",
        10.0,
        log=False,
    )


def mass_components(channel: str, array: np.ndarray):
    if channel == "hbb":
        return (
            (array[0], r"CEP $H\to b\bar b$", COLORS["signal"], "-"),
            (array[1], r"CEP $b\bar b$", COLORS["exclusive"], "--"),
            (array[2], r"$b\bar b + pp$", COLORS["inclusive"], ":"),
        )
    return (
        (array[0], r"CEP $H\to c\bar c$", COLORS["signal"], "-"),
        (array[1] + array[2], r"CEP $c\bar c$, $b\bar b$", COLORS["exclusive"], "--"),
        (array[3], r"$\gamma\gamma\to c\bar c$", COLORS["photon"], "-."),
        (array[4] + array[5], r"$c\bar c$, $b\bar b + pp$", COLORS["inclusive"], ":"),
        (array[6], r"CEP $H\to b\bar b$", COLORS["resonant"], "-"),
    )


def plot_mass_spectrum(output_dir: Path, channel: str, stage: str):
    analysis_dir = "MVA_new" if channel == "hbb" else "MVA_hcc"
    path = ROOT / f"analysis/{analysis_dir}/results/report_data.npz"
    with np.load(path, allow_pickle=False) as source:
        edges = np.asarray(source["mass_bins"], dtype=float)
        values = np.asarray(source[f"{stage}_mass"], dtype=float)
    figure, ax = plt.subplots(figsize=(5.0, 5.0))
    for histogram, label, color, linestyle in mass_components(channel, values):
        hep.histplot(
            histogram,
            bins=edges,
            ax=ax,
            histtype="step",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Proton missing mass $M_X$ [GeV]")
    ax.set_ylabel("Expected events / GeV")
    ax.grid(True, which="both", alpha=0.2)
    if channel == "hcc" and stage == "preselection":
        legend_location = "upper left"
        legend_anchor = (0.02, 0.79)
    elif channel == "hcc" and stage == "selected":
        legend_location = "center left"
        legend_anchor = (0.02, 0.53)
    elif stage == "selected":
        legend_location = "lower left"
        legend_anchor = None
    else:
        legend_location = "center right"
        legend_anchor = None
    ax.legend(
        frameon=channel == "hcc",
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc=legend_location,
        bbox_to_anchor=legend_anchor,
        fontsize=7.6 if channel == "hcc" else 8.2,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.7,
    )
    title = "Preselection" if stage == "preselection" else "Single MVA score cut"
    ax.text(
        0.97,
        0.88,
        title,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    plot_header(ax)
    figure.tight_layout()
    save_figure(figure, output_dir, f"{channel}_mass_{stage}")


def timing_factors():
    reports = {
        timing: load_json(timing_report_path("dijet", timing)) for timing in (3, 10)
    }
    key = "pps_mass_z__dy_0.25"
    baseline_rate = reports[10]["criteria"][key]["rate_hz"]
    baseline_signal = reports[10]["signal"]["criteria"][key]["efficiency"]
    return {
        timing: {
            "accidental": reports[timing]["criteria"][key]["rate_hz"]
            / baseline_rate,
            "matched": reports[timing]["signal"]["criteria"][key]["efficiency"]
            / baseline_signal,
        }
        for timing in (3, 10)
    }


def significance(component_mass: np.ndarray) -> float:
    signal = component_mass[0]
    background = component_mass[1:].sum(axis=0)
    total = signal + background
    return float(
        np.sqrt(
            np.sum(
                np.divide(
                    signal * signal,
                    total,
                    out=np.zeros_like(signal),
                    where=total > 0.0,
                )
            )
        )
    )


def sensitivity_scan(channel: str, factors):
    if channel == "hbb":
        path = ROOT / "analysis/MVA_new/results/report_data.npz"
        survival_scaled = (0, 1)
        real_protons = (0, 1)
    else:
        path = ROOT / "analysis/MVA_hcc/results/report_data.npz"
        survival_scaled = (0, 1, 2, 6)
        real_protons = (0, 1, 2, 3, 6)
    with np.load(path, allow_pickle=False) as source:
        nominal = np.asarray(source["selected_mass"], dtype=float)
    cross_sections = np.geomspace(0.1, 10.0, 300)
    curves = {}
    for timing in (10, 3):
        timed = nominal.copy()
        accidental = tuple(index for index in range(len(timed)) if index not in real_protons)
        timed[list(real_protons)] *= factors[timing]["matched"]
        timed[list(accidental)] *= factors[timing]["accidental"]
        values = []
        for cross_section in cross_sections:
            varied = timed.copy()
            varied[list(survival_scaled)] *= (
                cross_section / REFERENCE_CROSS_SECTION_FB
            )
            values.append(significance(varied))
        curves[timing] = np.asarray(values)
    return cross_sections, curves


def plot_sensitivity(output_dir: Path, channel: str, factors):
    cross_sections, curves = sensitivity_scan(channel, factors)
    figure, ax = plt.subplots(figsize=(5.0, 5.0))
    for timing, color in ((10, COLORS["timing10"]), (3, COLORS["timing3"])):
        ax.plot(
            cross_sections,
            curves[timing],
            color=color,
            label=f"{timing} ps proton timing",
        )
    ax.axvline(REFERENCE_CROSS_SECTION_FB, color="0.35", linestyle=":", linewidth=1.5)
    if channel == "hbb":
        ax.axhline(3.0, color="0.65", linestyle="--", linewidth=1.0)
        ax.axhline(5.0, color="0.65", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_xticks((0.1, 0.2, 0.4, 1.0, 2.0, 5.0, 10.0))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Total $\sigma(pp\to pHp)$ [fb]")
    ax.set_ylabel(r"Expected sensitivity $Z$")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9.5,
        labelspacing=0.3,
        handlelength=1.7,
    )
    channel_label = r"$H\to b\bar b$" if channel == "hbb" else r"$H\to c\bar c$"
    ax.text(
        0.96,
        0.86,
        channel_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    plot_header(ax)
    figure.tight_layout()
    save_figure(figure, output_dir, f"{channel}_sensitivity")
    return {
        "cross_section_fb": cross_sections.tolist(),
        "curves": {str(timing): curves[timing].tolist() for timing in (10, 3)},
        "reference_significance": {
            str(timing): float(
                np.interp(REFERENCE_CROSS_SECTION_FB, cross_sections, curves[timing])
            )
            for timing in (10, 3)
        },
    }


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    configure_style()
    summary = {"trigger_timing": plot_trigger_timing(output_dir)}
    plot_discriminants(output_dir)
    for channel in ("hbb", "hcc"):
        for stage in ("preselection", "selected"):
            plot_mass_spectrum(output_dir, channel, stage)
    factors = timing_factors()
    summary["timing_factors"] = factors
    summary["sensitivity"] = {
        channel: plot_sensitivity(output_dir, channel, factors)
        for channel in ("hbb", "hcc")
    }
    with (output_dir / "plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote paper figures to {output_dir}")


if __name__ == "__main__":
    main()
