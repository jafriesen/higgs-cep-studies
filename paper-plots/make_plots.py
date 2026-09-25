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

# H(bb) inputs from the mva/ framework (nominal includes mistagged charm).
HBB_DATASET = ROOT / "mva/Hbb/data/fsr_mtd_eight_class_gg_cc"
HBB_RESULTS = ROOT / "mva/Hbb/results"
HBB_SEEDS = (12345, 20260101, 20260202, 20260303, 20260404)
HBB_REPRESENTATIVE_SEED = 12345  # closest to the 10 ps five-seed mean Z
HBB_SETUPS = {
    10: "fsr_mtd_binary_7p1ps_allrows_gg_cc_g256_s{seed}",
    3: "fsr_mtd_binary_3ps_7p1ps_allrows_gg_cc_g256_s{seed}",
    "exclusive": "fsr_mtd_binary_7p1ps_allrows_exclusive_only_gg_cc_g256_s{seed}",
}
HBB_FEATURE_COMPONENTS = ("Hbb_fsr", "QCDbb_fsr", "QCDbb_madgraph_fsr")

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
    # no creation date, so an unchanged figure is byte-identical when regenerated
    figure.savefig(output_dir / f"{name}.pdf", metadata={"CreationDate": None})
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
    """Preselection shapes from the nominal H(bb) dataset.  Accidental-proton rows
    carry their expected pair intensity (pair_band_intensity) on top of the
    central-event weight, as in the training."""
    with (HBB_DATASET / "metadata.yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    names = list(metadata["central_features"])
    ids = {item["name"]: item["id"] for item in metadata["components"]}
    pooled = {item["id"] for item in metadata["components"] if not item["real_protons"]}
    matrix = np.load(HBB_DATASET / "x.npy", mmap_mode="r")
    component = np.load(HBB_DATASET / "component.npy", mmap_mode="r")
    physical = np.load(HBB_DATASET / "physical_weight.npy", mmap_mode="r")
    band = np.load(HBB_DATASET / "pair_band_intensity.npy", mmap_mode="r")
    specifications = {
        "tracks": {"index": names.index("n_tracks_interjet"), "edges": np.arange(-0.5, 30.5, 1.0)},
        "delta_eta": {"index": names.index("delta_eta_jj"), "edges": np.linspace(0.0, 4.5, 31)},
        "leading_pt": {"index": names.index("jet1_pt"), "edges": np.arange(15.0, 150.1, 2.5)},
    }
    histograms = {
        feature: {class_id: np.zeros(len(spec["edges"]) - 1) for class_id in range(len(HBB_FEATURE_COMPONENTS))}
        for feature, spec in specifications.items()
    }
    chunk_size = 500_000
    for start in range(0, len(component), chunk_size):
        stop = min(start + chunk_size, len(component))
        chunk_component = np.asarray(component[start:stop])
        chunk_weight = np.asarray(physical[start:stop]) * np.where(
            np.isin(chunk_component, list(pooled)), np.asarray(band[start:stop]), 1.0
        )
        for feature, spec in specifications.items():
            values = np.asarray(matrix[start:stop, spec["index"]], dtype=float)
            values = np.minimum(values, spec["edges"][-1] - 1.0e-6)
            for class_id, name in enumerate(HBB_FEATURE_COMPONENTS):
                selected = chunk_component == ids[name]
                if np.any(selected):
                    histograms[feature][class_id] += np.histogram(
                        values[selected], bins=spec["edges"], weights=chunk_weight[selected]
                    )[0]
    return specifications, histograms


def delta_y_histograms():
    """Written by paper-plots/delta_y_histograms.py, before the |Delta y| cut."""
    with np.load(ROOT / "paper-plots/data/delta_y_histograms.npz") as source:
        edges = np.asarray(source["edges"], dtype=float)
        histograms = np.asarray(source["histograms"], dtype=float)
        components = tuple(str(name) for name in source["components"])
    if components != HBB_FEATURE_COMPONENTS:
        raise RuntimeError(f"delta_y_histograms.npz holds {components}; rerun delta_y_histograms.py")
    return edges, {class_id: histograms[class_id] for class_id in range(len(components))}


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
        (0, r"$H\to b\bar b$", COLORS["signal"], "-"),
        (1, r"Exclusive $gg\to b\bar b$", COLORS["exclusive"], "--"),
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
    plot_feature(
        output_dir,
        "leading_jet_pt",
        histograms["leading_pt"],
        specifications["leading_pt"]["edges"],
        physics_classes,
        r"Leading jet $p_T$ [GeV]",
        (15.0, 90.0),
        "upper right",
        10.0,
        log=False,
    )


def mass_components(channel: str, array: np.ndarray, components=None):
    if channel == "hbb":
        exclusive = [i for i, item in enumerate(components) if item["real_protons"] and not item["signal"]]
        inclusive = [i for i, item in enumerate(components) if not item["real_protons"]]
        return (
            (array[0], r"$H\to b\bar b$", COLORS["signal"], "-"),
            (array[exclusive].sum(axis=0), "Exclusive backgrounds", COLORS["exclusive"], "--"),
            (array[inclusive].sum(axis=0), r"$b\bar b$, $c\bar c + pp$", COLORS["inclusive"], ":"),
        )
    return (
        (array[0], r"CEP $H\to c\bar c$", COLORS["signal"], "-"),
        (array[1] + array[2], r"CEP $c\bar c$, $b\bar b$", COLORS["exclusive"], "--"),
        (array[3], r"$\gamma\gamma\to c\bar c$", COLORS["photon"], "-."),
        (array[4] + array[5], r"$c\bar c$, $b\bar b + pp$", COLORS["inclusive"], ":"),
        (array[6], r"CEP $H\to b\bar b$", COLORS["resonant"], "-"),
    )


def plot_mass_spectrum(output_dir: Path, channel: str, stage: str):
    components = None
    if channel == "hbb":
        result = HBB_RESULTS / HBB_SETUPS[10].format(seed=HBB_REPRESENTATIVE_SEED)
        path = result / "report_data.npz"
        with (result / "report.yaml").open(encoding="utf-8") as handle:
            components = yaml.safe_load(handle)["components"]
    else:
        path = ROOT / "analysis/MVA_hcc/results/report_data.npz"
    with np.load(path, allow_pickle=False) as source:
        edges = np.asarray(source["mass_bins"], dtype=float)
        values = np.asarray(source[f"{stage}_mass"], dtype=float)
    figure, ax = plt.subplots(figsize=(5.0, 5.0))
    for histogram, label, color, linestyle in mass_components(channel, values, components):
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
    if channel != "hcc":
        raise ValueError("H(bb) uses plot_hbb_sensitivity")
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


def hbb_scan_curve(setup: str, cross_sections):
    """Five-seed mean of Z versus the total CEP Higgs cross section.

    Survival-scaled components (signal and QCD exclusive backgrounds) scale with
    the cross section; photon-exchange and accidental-proton backgrounds stay
    fixed.  At each point the best stored score threshold is re-selected among
    those with the training's MadGraph support (all of them if there is no
    MadGraph component), as in mva/common/scans.py."""
    curves = []
    for seed in HBB_SEEDS:
        result = HBB_RESULTS / HBB_SETUPS[setup].format(seed=seed)
        with (result / "report.yaml").open(encoding="utf-8") as handle:
            components = yaml.safe_load(handle)["components"]
        with np.load(result / "report_data.npz", allow_pickle=False) as source:
            scan_mass = np.asarray(source["scan_component_mass"], dtype=float)
            effective = np.asarray(source["scan_madgraph_effective"], dtype=float)
            floor = float(source["support_floor"])
        valid = np.flatnonzero(effective >= floor)
        if valid.size == 0:
            valid = np.arange(effective.size)
        scaled = np.array([bool(item["survival_scaled"]) for item in components])
        values = []
        for cross_section in cross_sections:
            factor = np.where(scaled, cross_section / REFERENCE_CROSS_SECTION_FB, 1.0)
            varied = scan_mass[valid] * factor[np.newaxis, :, np.newaxis]
            values.append(max(significance(point) for point in varied))
        curves.append(values)
    return np.mean(np.asarray(curves), axis=0)


def plot_hbb_sensitivity(output_dir: Path, with_exclusive_limit: bool):
    cross_sections = np.geomspace(0.1, 10.0, 300)
    curves = {setup: hbb_scan_curve(setup, cross_sections) for setup in (10, 3, "exclusive")}
    figure, ax = plt.subplots(figsize=(5.0, 5.0))
    if with_exclusive_limit:
        ax.plot(cross_sections, curves["exclusive"], color="0.6", linewidth=2.0,
                label="No inclusive background")
    for timing, color in ((10, COLORS["timing10"]), (3, COLORS["timing3"])):
        ax.plot(cross_sections, curves[timing], color=color, label=f"{timing} ps proton timing")
    ax.axvline(REFERENCE_CROSS_SECTION_FB, color="0.35", linestyle=":", linewidth=1.5)
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
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelspacing=0.3, handlelength=1.7)
    ax.text(0.96, 0.86, r"$H\to b\bar b$", transform=ax.transAxes, ha="right", va="top", fontsize=12)
    plot_header(ax)
    figure.tight_layout()
    name = "hbb_sensitivity_exclusive_limit" if with_exclusive_limit else "hbb_sensitivity"
    save_figure(figure, output_dir, name)
    return {
        "cross_section_fb": cross_sections.tolist(),
        "seeds": list(HBB_SEEDS),
        "curves": {str(setup): curve.tolist() for setup, curve in curves.items()},
        "reference_significance": {
            str(setup): float(np.interp(REFERENCE_CROSS_SECTION_FB, cross_sections, curve))
            for setup, curve in curves.items()
        },
    }


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
        "hbb": plot_hbb_sensitivity(output_dir, with_exclusive_limit=False),
        "hcc": plot_sensitivity(output_dir, "hcc", factors),
    }
    plot_hbb_sensitivity(output_dir, with_exclusive_limit=True)
    with (output_dir / "plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote paper figures to {output_dir}")


if __name__ == "__main__":
    main()
