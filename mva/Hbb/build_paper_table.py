#!/usr/bin/env python3
"""Assemble the H(bb) efficiency table from the cutflow recount and a timing study.

Rows follow the paper's order: expected events and cross section after the
central dijet preselection, then sequential efficiencies for the tagged proton
pair in the mass window, the vertex requirement, the proton-dijet rapidity
match, double b tagging and the BDT, then the final expected events.

Inputs are all machine-readable:
  --counts     pre_mva_counts.yaml from build_cutflow.py (central and real-proton stages)
  --dataset    the prepared dataset the training used (tag factors, and the
               analytic accidental-pair factors for the MadGraph component)
  --timing     timing_selection.yaml from timing_selection.py
  --variant    "analysis" (default): final yields of the analysis selection, with
               the vertex row set to the vertex-compatible core.  Allowed only if
               timing_selection.yaml shows every final yield lies inside the core,
               in which case the selection factorizes exactly.  Otherwise the name
               of a sequential variant, which supplies both rows.
  --full-phase-space COMPONENT=COUNTS
               take that component's dijet-preselection row and window
               efficiency from another recount (e.g. QCDgg v03 without the
               generator M_X cut); later rows still come from --counts

For the accidental-proton component the proton rows are expected pair counts
per central event, not event efficiencies: the mass-window row is
mu^2 * P(window | interaction pair) (exact for Poisson pileup) and the
rapidity row is the dataset's band-weighted fraction divided by it.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

LUMINOSITY_FB = 3000.0
LABELS = {
    "Hbb_fsr": r"$H\to b\bar b$",
    "QCDbb_fsr": r"CEP $b\bar b$",
    "QCDgg_fsr": r"CEP $gg$",
    "QEDbb_fsr": r"QED $b\bar b$",
    "QCDbb_madgraph_fsr": r"$b\bar b + pp$",
}


def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def central_and_protons(counts):
    """Per component: generated and post-dijet yields, and real-proton stage yields."""
    out = {}
    for group in counts["groups"]:
        scale = group["xsec_fb"] * LUMINOSITY_FB / group["counts"]["generated"]
        entry = out.setdefault(
            group["component"],
            {"real_protons": group["real_protons"], "generated": 0.0, "dijet": 0.0,
             "window": 0.0, "rapidity": 0.0},
        )
        entry["generated"] += group["xsec_fb"] * LUMINOSITY_FB
        entry["dijet"] += group["counts"]["dijet_mass"] * scale
        entry["window"] += group["counts"]["mass_window"] * scale
        entry["rapidity"] += group["counts"]["rapidity"] * scale
    return out


def accidental_factors(dataset_dir, metadata):
    component = np.load(dataset_dir / "component.npy")
    physical = np.load(dataset_dir / "physical_weight.npy")
    band = np.load(dataset_dir / "pair_band_intensity.npy")
    protons = metadata["protons"]
    window = protons["pileup_mu"] ** 2 * protons["full_mass_window_intensity_per_interaction_pair"]
    factors = {}
    for item in metadata["components"]:
        if item["real_protons"]:
            continue
        rows = component == item["id"]
        combined = float(np.sum(physical[rows] * band[rows]) / np.sum(physical[rows]))
        factors[item["name"]] = {"window": float(window), "rapidity": combined / float(window)}
    return factors


def fmt(value, efficiency=False):
    """Two significant figures for efficiencies, three for yields and cross sections."""
    if value is None:
        return "--"
    if value == 0:
        return "0"
    digits = 2 if efficiency else 3
    if 10**digits <= abs(value) < 1e4:
        return str(int(float(f"{value:.{digits}g}")))
    if 1e-2 <= abs(value) < 10**digits:
        return f"{value:#.{digits}g}".rstrip(".")
    exponent = int(np.floor(np.log10(abs(value))))
    return f"{value / 10**exponent:#.{digits}g}".rstrip(".") + f"e{exponent}"


def tex_number(text):
    if "e" in text and text not in ("--",):
        mantissa, exponent = text.split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"${text}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--variant", default="analysis", help='"analysis" or a sequential variant name')
    parser.add_argument("--full-phase-space", action="append", default=[], metavar="COMPONENT=COUNTS")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    counts = load(args.counts)
    metadata = load(args.dataset / "metadata.yaml")
    timing = load(args.timing)
    variants = {row["variant"]: row for row in timing["variants"]}
    if args.variant == "analysis":
        final_source = next(row for row in timing["variants"] if "(analysis)" in row["variant"])
        core = next(row for row in timing["variants"] if row["variant"].startswith("z+t, core"))
        inside = [c["vertex_compatible_core_fraction"] for c in timing["analysis_composition_by_vertex_region"].values()]
        if min(inside) < 0.9999:
            raise SystemExit(f"Analysis selection does not factorize at the core (min fraction {min(inside):.4f})")
        variant = dict(final_source, matched_vertex_efficiency=core["matched_vertex_efficiency"],
                       unrelated_vertex_efficiency=core["unrelated_vertex_efficiency"])
    elif args.variant in variants:
        variant = variants[args.variant]
    else:
        raise SystemExit(f"Unknown variant {args.variant!r}; have {sorted(variants)}")

    stages = central_and_protons(counts)
    overrides = {}
    for item in args.full_phase_space:
        component, path = item.split("=", 1)
        overrides[component] = central_and_protons(load(path))[component]
    accidental = accidental_factors(args.dataset, metadata)
    presel = metadata["physical_yields_per_component"]
    rows = {}
    for item in metadata["components"]:
        name = item["name"]
        stage = stages[name]
        tag = float(item["tag_factor"])
        if item["real_protons"]:
            window = stage["window"] / stage["dijet"]
            rapidity = stage["rapidity"] / stage["window"]
            if name in overrides:
                # dijet row and window efficiency from the full-phase-space sample;
                # the product up to the window must agree with the restricted sample
                full = overrides[name]
                stitch = full["window"] / stage["window"] - 1.0
                window = full["window"] / full["dijet"]
                stage = dict(stage, dijet=full["dijet"], generated=full["generated"], window=full["window"])
                print(f"{name}: dijet row and window efficiency from full phase space; "
                      f"events in window differ by {stitch:+.1%} between samples")
        else:
            window = accidental[name]["window"]
            rapidity = accidental[name]["rapidity"]
        vertex = variant["matched_vertex_efficiency"] if item["real_protons"] else variant["unrelated_vertex_efficiency"]
        # closure on the sample the training used: recounted real-proton events after
        # the rapidity cut, or the analytic chain for accidental pairs
        own = stages[name]
        rebuilt = own["rapidity"] * tag if item["real_protons"] else own["dijet"] * window * rapidity * tag
        final = variant["final_yields"][name]
        rows[name] = {
            "generated_events": stage["generated"],
            "dijet_cross_section_fb": stage["dijet"] / LUMINOSITY_FB,
            "dijet_events": stage["dijet"],
            "tagged_pp_window": window,
            "vertex": vertex,
            "rapidity": rapidity,
            "b_tagging": tag,
            "bdt": final / (presel[name] * vertex),
            "final_events": final,
            "preselection_check": rebuilt / presel[name] - 1.0,
        }
    worst = max(abs(row["preselection_check"]) for row in rows.values())
    if worst > 2e-3:
        raise SystemExit(f"Cutflow product does not reproduce the dataset preselection (worst {worst:.2e})")

    names = [item["name"] for item in metadata["components"]]
    signal = names[0]
    layout = [
        ("Cross section after dijet preselection [fb]", "dijet_cross_section_fb", False),
        (r"Expected events after dijet preselection", "dijet_events", False),
        (r"Tagged $pp$, $117<M_X<133\,\mathrm{GeV}$", "tagged_pp_window", True),
        ("Vertex requirement", "vertex", True),
        (r"$|y_X-y_{jj}|<0.2$", "rapidity", True),
        (r"$b$-tagging", "b_tagging", True),
        ("BDT", "bdt", True),
        ("Expected events after all selections", "final_events", False),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    tex = [r"\begin{tabular}{l" + "r" * len(names) + "}", r"\toprule",
           " & " + " & ".join(LABELS.get(n, n) for n in names) + r" \\", r"\midrule"]
    for label, key, efficiency in layout:
        cells = [fmt(rows[n][key], efficiency) for n in names]
        markdown.append(f"| {label} | " + " | ".join(cells) + " |")
        tex.append(f"{label} & " + " & ".join(tex_number(c) for c in cells) + r" \\")
        if key in ("dijet_events", "bdt"):
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    background = sum(rows[n]["final_events"] for n in names[1:])
    summary = {
        "variant": variant["variant"],
        "threshold": variant["threshold"],
        "significance": variant["significance"],
        "madgraph_effective_central_events": variant["madgraph_effective_central_events"],
        "signal": rows[signal]["final_events"],
        "background": background,
        "signal_dijet_efficiency_on_generated": rows[signal]["dijet_events"] / rows[signal]["generated_events"],
        "worst_preselection_closure": worst,
        "rows": rows,
        "sources": {"counts": str(args.counts.resolve()), "dataset": str(args.dataset.resolve()),
                    "timing": str(args.timing.resolve())},
    }
    (args.output_dir / "hbb_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (args.output_dir / "hbb_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (args.output_dir / "hbb_table.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print("\n".join(markdown))
    print(
        f"\n{variant['variant']}: Z = {variant['significance']:.3f}, S = {summary['signal']:.2f}, "
        f"B = {background:.1f}, effN = {variant['madgraph_effective_central_events']:.1f}"
    )
    print(f"signal dijet preselection efficiency on generated events: {summary['signal_dijet_efficiency_on_generated']:.3f}")
    print(f"preselection closure (cutflow product vs dataset): worst {worst:.1e}")
    print(f"Wrote {args.output_dir}/hbb_table.{{md,tex,yaml}}")


if __name__ == "__main__":
    main()
