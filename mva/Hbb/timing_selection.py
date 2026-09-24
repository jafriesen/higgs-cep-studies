#!/usr/bin/env python3
"""Compare the vertex-timing likelihood with explicit vertex cuts for one result.

Needs a result whose report_data.npz stores ``base_mass`` (the BDT-only score x
mass histograms before the vertex-likelihood convolution) and
``madgraph_base_above_squared``.

Sequential variants apply a fixed vertex requirement and then cut on the
BDT-only score, re-selecting the best stored threshold under the training's
MadGraph support floor.  The requirement is a uniform factor per vertex
hypothesis (matched for real protons, unrelated for accidental pairs), so the
yields and the MadGraph effective count are exact: the factor cancels in the
count, which is that of the BDT-only selection.

The analysis likelihood (vertex log-LR added to the BDT score) is taken from the
stored scan, exactly as the training selected it.  Its composition by vertex
region is estimated by convolving ``base_mass`` bin by bin; that convolution
shifts pre-binned score centres, so it reproduces the stored scan only to
~0.3% in yield (reported).

Likelihood variants change the vertex log-LR table (finer binning, or a
pre-cut on q with the likelihood rebinned inside it) and keep it added to the
BDT score.  Yields use the same bin-centre rule as the analysis convolution.  A
varied table has no stored MadGraph effective count; it is replaced by a
rigorous lower bound (Minkowski): sqrt(sum w^2) <= sum_b p_b sqrt(Q(t - s_b)),
where Q is the stored per-event BDT-only sum of squares.  The bound is checked
against the exact count of the analysis table.

Vertex regions are cuts on the analysis quadratic form q (the log-LR quadratic
in dz and dct), placed by their matched-pair efficiency with the same
quadrature as minbias.vertex_likelihood_table.  "z only" drops the time
information and cuts |dz| < 2 sigma_z.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from minbias.vertex import (  # noqa: E402
    _quadratic_quantiles,
    _quadratic_scales,
    vertex_likelihood_table,
)
from mva.common.dataset import write_yaml  # noqa: E402
from mva.common.training import _significance, _vertex_convolution  # noqa: E402

TWO_SIGMA = math.erf(2.0 / math.sqrt(2.0))
SCAN = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95, TWO_SIGMA, 0.97, 0.98, 0.99, 0.9973)


def load_result(result_dir):
    report = yaml.safe_load((result_dir / "report.yaml").read_text(encoding="utf-8"))
    with np.load(result_dir / "report_data.npz", allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    missing = {"base_mass", "madgraph_base_above_squared"} - set(data)
    if missing:
        raise SystemExit(
            f"{result_dir} predates the base_mass output (missing {sorted(missing)}); rerun training"
        )
    settings = yaml.safe_load(
        Path(report["orchestration"]["manifest"]).read_text(encoding="utf-8")
    )["settings"]
    return report, data, settings


def region_probabilities(arguments, table, matched_efficiency):
    """(matched, unrelated) probabilities for q below the cut keeping that matched fraction."""
    scales = _quadratic_scales(table["parameters"], "matched")
    edge = float(
        _quadratic_quantiles(np.array([matched_efficiency]), scales, table["quadrature_order"])[0]
    )
    region = vertex_likelihood_table(quadratic_edges=[0.0, edge, np.inf], **arguments)
    return float(region["matched_probability"][0]), float(region["unrelated_probability"][0])


def z_only_probabilities(parameters):
    ratio = parameters["matched_sigma_z_cm"] / parameters["unrelated_sigma_z_cm"]
    return TWO_SIGMA, math.erf(2.0 * ratio / math.sqrt(2.0))


def selection_row(name, names, final_mass, significance, threshold, effective, matched, unrelated):
    final = final_mass.sum(axis=1)
    return {
        "variant": name,
        "matched_vertex_efficiency": matched,
        "unrelated_vertex_efficiency": unrelated,
        "threshold": float(threshold),
        "significance": float(significance),
        "madgraph_effective_central_events": float(effective),
        "final_yields": {n: float(v) for n, v in zip(names, final)},
        "signal": float(final[0]),
        "background": float(final[1:].sum()),
    }


def sequential(name, matched, unrelated, context):
    factor = np.array([matched if h == "matched" else unrelated if h == "unrelated" else 1.0
                       for h in context["hypotheses"]])
    scan_mass = factor[np.newaxis, :, np.newaxis] * context["base_scan"]
    significance = _significance(scan_mass)
    effective = context["base_effective"]
    valid = np.flatnonzero(effective >= context["floor"])
    best = valid[np.argmax(significance[valid])]
    row = selection_row(name, context["names"], scan_mass[best], significance[best],
                        context["thresholds"][best], effective[best], matched, unrelated)
    row["at_support_floor"] = bool(best == valid.max())
    return row


def rebinned_table(arguments, edges, cut_edge):
    """Log-LR table on the given interior edges below cut_edge; q > cut_edge rejected."""
    kept = [float(edge) for edge in edges if 0.0 < edge < cut_edge]
    table = vertex_likelihood_table(quadratic_edges=[0.0, *kept, cut_edge, np.inf], **arguments)
    table = {key: (np.array(value, dtype=float) if key != "parameters" else value) for key, value in table.items()}
    for key in ("matched_probability", "unrelated_probability"):
        table[key][-1] = 0.0
    return table


def likelihood_variant(name, table, context):
    """Vertex log-LR added to the BDT-only score, with the effective count lower-bounded."""
    if "none" in context["hypotheses"]:
        raise SystemExit("likelihood variants assume every component has a vertex hypothesis")
    cumulative = context["cumulative"]
    padded = np.concatenate([cumulative, np.zeros_like(cumulative[:, :1])], axis=1)
    squares = np.r_[context["base_above_squared"], 0.0]
    edges = context["thresholds"]
    pooled = context["pooled"]
    scan_mass = np.zeros((edges.size, cumulative.shape[0], cumulative.shape[2]))
    numerator = np.zeros(edges.size)
    root = np.zeros(edges.size)
    for bin_index, shift in enumerate(np.asarray(table["log_likelihood_ratio"], dtype=float)):
        probability = np.array([table[f"{h}_probability"][bin_index] for h in context["hypotheses"]], dtype=float)
        if not probability.any():
            continue
        # a BDT-only bin passes when its centre plus the shift clears the threshold edge
        first = np.searchsorted(context["centres"], edges - shift, side="left")
        above = np.transpose(padded[:, first, :], (1, 0, 2))
        scan_mass += probability[np.newaxis, :, np.newaxis] * above
        unrelated = float(np.max(probability[pooled]))
        numerator += unrelated * above[:, pooled].sum(axis=(1, 2))
        root += unrelated * np.sqrt(squares[first])
    effective = np.divide(numerator**2, root**2, out=np.zeros_like(numerator), where=root > 0.0)
    significance = _significance(scan_mass)
    valid = np.flatnonzero(effective >= context["floor"])
    best = valid[np.argmax(significance[valid])]
    matched = float(np.sum(table["matched_probability"]))
    unrelated = float(np.sum(table["unrelated_probability"]))
    row = selection_row(name, context["names"], scan_mass[best], significance[best],
                        edges[best], effective[best], matched, unrelated)
    row["at_support_floor"] = bool(best == valid.max())
    row["effective_count_is_lower_bound"] = True
    return row, effective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    report, data, settings = load_result(result_dir)

    components = report["components"]
    names = [item["name"] for item in components]
    pooled = np.array([not item["real_protons"] for item in components])
    edges = data["score_edges"]
    thresholds = data["scan_thresholds"]
    scan_bins = np.searchsorted(edges, thresholds)
    floor = float(data["support_floor"])
    base = data["base_mass"]
    arguments = {
        "beam_sigma_z_cm": settings["beam_sigma_z_cm"],
        "single_arm_time_resolution_ps": settings["pps_time_ps"],
        "pv_z_resolution_cm": settings["pv_z_resolution_cm"],
        "pv_time_resolution_ps": settings["pv_time_ps"],
    }
    table = vertex_likelihood_table(bins=settings["vertex_bins"], **arguments)
    stored_table = report["vertex"]["table"]
    for key in ("matched_probability", "unrelated_probability", "log_likelihood_ratio"):
        if not np.allclose(table[key], stored_table[key], rtol=0, atol=1e-12):
            raise SystemExit(f"Rebuilt vertex table disagrees with the stored one ({key})")
    preselection_closure = float(
        np.abs(base.sum(axis=(1, 2)) - data["preselection_mass"].sum(axis=1)).max()
        / data["preselection_mass"].sum(axis=1).max()
    )
    if preselection_closure > 1e-9:
        raise SystemExit(f"base_mass totals disagree with the preselection ({preselection_closure:.1e})")

    cumulative = np.cumsum(base[:, ::-1, :], axis=1)[:, ::-1, :]
    base_scan = np.transpose(cumulative[:, scan_bins], (1, 0, 2))
    mg_above = base_scan[:, pooled].sum(axis=(1, 2))
    squares = data["madgraph_base_above_squared"][scan_bins]
    context = {
        "names": names,
        "hypotheses": [item["vertex_hypothesis"] for item in components],
        "base_scan": base_scan,
        "base_effective": np.divide(mg_above**2, squares, out=np.zeros_like(mg_above), where=squares > 0.0),
        "thresholds": thresholds,
        "floor": floor,
        "cumulative": cumulative,
        "centres": 0.5 * (edges[:-1] + edges[1:]),
        "base_above_squared": data["madgraph_base_above_squared"],
        "pooled": pooled,
    }

    # the analysis selection, exactly as the training stored it
    operating = int(data["operating_index"])
    analysis = selection_row(
        "vertex log-LR added to BDT score (analysis)", names, data["scan_component_mass"][operating],
        _significance(data["scan_component_mass"])[operating], thresholds[operating],
        data["scan_madgraph_effective"][operating], 1.0, 1.0,
    )
    analysis["at_support_floor"] = bool(
        operating == np.flatnonzero(data["scan_madgraph_effective"] >= floor).max()
    )

    # composition of the analysis selection by vertex region (bin-centre convolution)
    convolved = _vertex_convolution(base, components, table, edges)
    convolved_scan = np.transpose(np.cumsum(convolved[:, ::-1, :], axis=1)[:, ::-1, :][:, scan_bins], (1, 0, 2))
    closure = float(np.max(np.abs(convolved_scan - data["scan_component_mass"])) / np.max(data["scan_component_mass"]))
    inner_table = {key: np.array(value, dtype=float) for key, value in table.items() if key != "parameters"}
    for key in ("matched_probability", "unrelated_probability"):
        inner_table[key][-1] = 0.0  # keep only the vertex-compatible core, q < 87.5% matched quantile
    inner = _vertex_convolution(base, components, inner_table, edges)
    inner_final = np.cumsum(inner[:, ::-1, :], axis=1)[:, ::-1, :][:, scan_bins[operating]].sum(axis=1)
    total_final = convolved_scan[operating].sum(axis=1)
    composition = {
        name: {"vertex_compatible_core_fraction": float(i / t) if t > 0 else None}
        for name, i, t in zip(names, inner_final, total_final)
    }

    parameters = table["parameters"]
    rows = [
        sequential("no vertex requirement", 1.0, 1.0, context),
        sequential("z only, |dz| < 2 sigma", *z_only_probabilities(parameters), context),
        sequential("z+t, 2 sigma (95.45%)", *region_probabilities(arguments, table, TWO_SIGMA), context),
        sequential("z+t, core (87.5%)", *region_probabilities(arguments, table, 0.875), context),
        sequential("z+t, 3 sigma (99.73%)", *region_probabilities(arguments, table, 0.9973), context),
    ]
    scan = [sequential(f"z+t {value:.4f}", *region_probabilities(arguments, table, value), context) for value in SCAN]
    best = max(scan, key=lambda row: row["significance"])
    rows += [dict(best, variant=f"best sequential cut ({best['matched_vertex_efficiency']:.3f})"), analysis]

    scales = _quadratic_scales(table["parameters"], "matched")
    def edge_at(efficiency):
        return float(_quadratic_quantiles(np.array([efficiency]), scales, table["quadrature_order"])[0])
    two, three = edge_at(TWO_SIGMA), edge_at(0.9973)
    interior = table["quadratic_edges"][1:-1]
    fine = {n: vertex_likelihood_table(bins=n, **arguments) for n in (16, 32)}
    variant_tables = [
        ("8 bins (analysis table, bin-centre)", table),
        ("8 bins, 2 sigma pre-cut, same bins", rebinned_table(arguments, interior, float(table["quadratic_edges"][-2]))),
        ("2 sigma pre-cut, rebinned inside", rebinned_table(arguments, interior, two)),
        ("3 sigma pre-cut, rebinned inside", rebinned_table(arguments, interior, three)),
        ("16 bins", fine[16]),
        ("16 bins, 2 sigma pre-cut", rebinned_table(arguments, fine[16]["quadratic_edges"][1:-1], two)),
        ("32 bins", fine[32]),
        ("32 bins, 2 sigma pre-cut", rebinned_table(arguments, fine[32]["quadratic_edges"][1:-1], two)),
    ]
    likelihood_rows = []
    for name, variant_table in variant_tables:
        row, effective = likelihood_variant(name, variant_table, context)
        likelihood_rows.append(row)
        if variant_table is table:
            bound_check = {
                "exact_at_operating_point": float(data["scan_madgraph_effective"][operating]),
                "lower_bound_at_operating_point": float(effective[operating]),
            }

    print(f"closures: base_mass vs preselection {preselection_closure:.1e}; "
          f"bin-centre convolution vs stored scan {closure:.1e} (composition only)")
    print(f"{'variant':44s}{'matched':>8}{'unrelated':>11}{'thr':>8}{'Z':>8}{'S':>8}{'B':>10}{'effN':>7}")
    for row in rows:
        mark = "  floor" if row["at_support_floor"] else ""
        print(f"{row['variant']:44s}{row['matched_vertex_efficiency']:8.4f}{row['unrelated_vertex_efficiency']:11.5f}"
              f"{row['threshold']:8.3f}{row['significance']:8.4f}{row['signal']:8.2f}{row['background']:10.2f}"
              f"{row['madgraph_effective_central_events']:7.1f}{mark}")
    print(f"\nlikelihood variants (effective count is a lower bound; at the analysis point "
          f"bound {bound_check['lower_bound_at_operating_point']:.1f} vs exact {bound_check['exact_at_operating_point']:.1f})")
    for row in likelihood_rows:
        mark = "  floor" if row["at_support_floor"] else ""
        print(f"{row['variant']:44s}{row['matched_vertex_efficiency']:8.4f}{row['unrelated_vertex_efficiency']:11.5f}"
              f"{row['threshold']:8.3f}{row['significance']:8.4f}{row['signal']:8.2f}{row['background']:10.2f}"
              f"{row['madgraph_effective_central_events']:7.1f}{mark}")
    print("\nsequential scan, matched efficiency -> Z:")
    print("  " + "  ".join(f"{r['matched_vertex_efficiency']:.3f}:{r['significance']:.3f}" for r in scan))
    print("\nanalysis selection, fraction of each final yield inside the vertex-compatible core:")
    print("  " + "  ".join(f"{n}:{c['vertex_compatible_core_fraction']:.3f}" for n, c in composition.items()))

    write_yaml(
        result_dir / "timing_selection.yaml",
        {
            "interpretation": __doc__.strip().splitlines()[0],
            "support_floor": floor,
            "closures": {"base_mass_vs_preselection": preselection_closure,
                         "bin_centre_convolution_vs_stored_scan": closure},
            "variants": rows,
            "sequential_scan": scan,
            "best_sequential": best,
            "analysis_composition_by_vertex_region": composition,
            "likelihood_variants": likelihood_rows,
            "effective_count_bound_check": bound_check,
        },
    )
    print(f"\nWrote {result_dir / 'timing_selection.yaml'}")


if __name__ == "__main__":
    main()
