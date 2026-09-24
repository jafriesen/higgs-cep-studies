#!/usr/bin/env python3
"""Report all-campaign MadGraph stitching and physical group weights."""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--reference-cache", default=None)
    return parser.parse_args()


def effective_size(weights):
    denominator = np.sum(weights * weights)
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0 else 0.0


def aggregate_groups(cache_dir):
    classes = np.load(cache_dir / "class.npy", mmap_mode="r")
    mask = np.asarray(classes) == 2
    groups = np.asarray(np.load(cache_dir / "group_id.npy", mmap_mode="r")[mask])
    weights = np.asarray(
        np.load(cache_dir / "physical_weight.npy", mmap_mode="r")[mask],
        dtype=np.float64,
    )
    campaigns = np.asarray(
        np.load(cache_dir / "source_campaign.npy", mmap_mode="r")[mask]
    )
    coverage_path = cache_dir / "coverage_mask.npy"
    coverage = (
        np.asarray(np.load(coverage_path, mmap_mode="r")[mask])
        if coverage_path.is_file()
        else np.zeros(groups.size, dtype=np.uint8)
    )
    order = np.argsort(groups, kind="stable")
    groups, weights = groups[order], weights[order]
    campaigns, coverage = campaigns[order], coverage[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    return {
        "weight": np.add.reduceat(weights, starts),
        "campaign": campaigns[starts],
        "coverage": coverage[starts],
        "row_weight": weights,
        "row_campaign": campaigns,
        "row_coverage": coverage,
    }


def describe(weights):
    quantiles = np.quantile(weights, [0.1, 0.5, 0.9, 0.99])
    return {
        "groups": int(weights.size),
        "yield": float(np.sum(weights)),
        "neff": effective_size(weights),
        "mean_weight": float(np.mean(weights)),
        "p10_weight": float(quantiles[0]),
        "median_weight": float(quantiles[1]),
        "p90_weight": float(quantiles[2]),
        "p99_weight": float(quantiles[3]),
        "max_weight": float(np.max(weights)),
    }


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    data = aggregate_groups(cache_dir)
    campaigns = list(metadata["madgraph_campaigns"])
    summary = {
        "campaign_order_for_coverage_bits": campaigns,
        "all_madgraph": describe(data["weight"]),
        "all_madgraph_rows": describe(data["row_weight"]),
        "by_source_campaign": {},
        "by_coverage": {},
        "rows_by_source_campaign": {},
        "rows_by_coverage": {},
        "effective_mc_luminosity_fb_inv": {
            name: metadata["madgraph_stitching"][name][
                "effective_mc_luminosity_fb_inv"
            ]
            for name in campaigns
        },
    }
    summary["all_campaign_core_central_weight_fb"] = float(
        1.0 / sum(summary["effective_mc_luminosity_fb_inv"].values())
    )
    for campaign in campaigns:
        selected = data["campaign"] == campaign
        summary["by_source_campaign"][campaign] = describe(data["weight"][selected])
        selected_rows = data["row_campaign"] == campaign
        summary["rows_by_source_campaign"][campaign] = describe(
            data["row_weight"][selected_rows]
        )
    for value in np.unique(data["coverage"]):
        selected = data["coverage"] == value
        names = [
            campaign for bit, campaign in enumerate(campaigns)
            if int(value) & (1 << bit)
        ]
        label = "+".join(names) if names else "not_recorded"
        summary["by_coverage"][label] = {
            "bitmask": int(value),
            **describe(data["weight"][selected]),
        }
    for value in np.unique(data["row_coverage"]):
        selected = data["row_coverage"] == value
        names = [
            campaign for bit, campaign in enumerate(campaigns)
            if int(value) & (1 << bit)
        ]
        label = "+".join(names) if names else "not_recorded"
        summary["rows_by_coverage"][label] = {
            "bitmask": int(value),
            **describe(data["row_weight"][selected]),
        }
    if args.reference_cache:
        reference_cache = Path(args.reference_cache)
        reference = aggregate_groups(reference_cache)
        with open(reference_cache / "metadata.yaml", encoding="utf-8") as handle:
            reference_metadata = yaml.safe_load(handle)
        old = describe(reference["weight"])
        summary["reference_cache"] = str(args.reference_cache)
        summary["reference_all_madgraph"] = old
        difference = summary["all_madgraph"]["yield"] - old["yield"]
        uncertainty = np.sqrt(
            np.sum(data["weight"] ** 2) + np.sum(reference["weight"] ** 2)
        )
        summary["yield_closure"] = {
            "difference": float(difference),
            "relative_difference": float(difference / old["yield"]),
            "combined_mc_uncertainty": float(uncertainty),
            "pull": float(difference / uncertainty),
            "allowed_absolute_difference": float(max(0.01 * old["yield"], 3.0 * uncertainty)),
            "passes": bool(abs(difference) <= max(0.01 * old["yield"], 3.0 * uncertainty)),
        }
        reference_luminosities = [
            campaign["effective_mc_luminosity_fb_inv"]
            for campaign in reference_metadata["madgraph_stitching"].values()
        ]
        old_core_weight = 1.0 / sum(reference_luminosities)
        new_core_weight = summary["all_campaign_core_central_weight_fb"]
        summary["core_central_weight_comparison"] = {
            "reference_weight_fb": float(old_core_weight),
            "all_campaign_weight_fb": float(new_core_weight),
            "all_over_reference": float(new_core_weight / old_core_weight),
            "reduction_percent": float(100.0 * (1.0 - new_core_weight / old_core_weight)),
        }

    with open(result_dir / "stitching_weight_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    with open(
        result_dir / "stitching_weight_summary.csv", "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "region", "groups", "yield", "neff", "mean_weight",
                        "median_weight", "p90_weight", "p99_weight", "max_weight"],
        )
        writer.writeheader()
        for kind, regions in (
            ("source", summary["by_source_campaign"]),
            ("coverage", summary["by_coverage"]),
            ("row_source", summary["rows_by_source_campaign"]),
            ("row_coverage", summary["rows_by_coverage"]),
        ):
            for region, values in regions.items():
                writer.writerow({
                    "kind": kind,
                    "region": region,
                    **{name: values[name] for name in writer.fieldnames[2:]},
                })

    positive = np.concatenate((
        data["weight"][data["weight"] > 0],
        data["row_weight"][data["row_weight"] > 0],
    ))
    bins = np.geomspace(np.min(positive), np.max(positive), 80)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), sharex=True)
    for row, (weight_key, campaign_key, coverage_key, prefix) in enumerate((
        ("row_weight", "row_campaign", "row_coverage", "Candidate-row"),
        ("weight", "campaign", "coverage", "Hard-event group"),
    )):
        for campaign in campaigns:
            selected = data[campaign_key] == campaign
            axes[row, 0].hist(
                data[weight_key][selected], bins=bins, histtype="step", density=True,
                linewidth=1.6, label=campaign,
            )
        for value in np.unique(data[coverage_key]):
            selected = data[coverage_key] == value
            names = [
                campaign for bit, campaign in enumerate(campaigns)
                if int(value) & (1 << bit)
            ]
            axes[row, 1].hist(
                data[weight_key][selected], bins=bins, histtype="step", density=True,
                linewidth=1.4, label="+".join(names) if names else "not recorded",
            )
        axes[row, 0].set_title(f"{prefix}: source campaign")
        axes[row, 1].set_title(f"{prefix}: coverage region")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Physical weight")
        axis.set_ylabel("Density")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(result_dir / "stitching_group_weights.png", dpi=180)
    plt.close(fig)

    histograms = metadata["madgraph_stitching"]
    variables = (
        ("minimum_parton_pt_gev", r"minimum hard-$b$ $p_T$ [GeV]"),
        ("maximum_abs_parton_eta", r"maximum hard-$b$ $|\eta|$"),
        ("parton_dijet_mass_gev", r"hard $m_{bb}$ [GeV]"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for axis, (key, label) in zip(axes, variables):
        for campaign in campaigns:
            values = histograms[campaign]["truth_histograms"][key]
            axis.stairs(
                values["cross_section_fb"], values["edges"],
                linewidth=1.5, label=campaign,
            )
        axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Stitched cross section / bin [fb]")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "stitching_truth_phase_space.png", dpi=180)
    plt.close(fig)
    print(f"Wrote stitching diagnostics to {result_dir}")


if __name__ == "__main__":
    main()
