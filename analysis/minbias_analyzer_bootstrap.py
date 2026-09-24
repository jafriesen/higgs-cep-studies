#!/usr/bin/env python3
"""Build a weighted global proton-pair pool from a min-bias campaign.

Bootstrap variant of minbias_analyzer.py. Instead of partitioning the
interactions into a fixed set of bunch crossings and pairing protons only
within each BX, this script pools every PPS-accepted forward proton across all
interactions and forms *all* cross-arm combinations that fall in the double-tag
mass window, each carrying the weight it would have in a random Poisson(mu) BX.
That extracts far more of the combinatorial information in a finite min-bias
library than the ~N_interactions/mu distinct BX allow, and it assigns the
correct (larger) weight to correlated same-interaction pairs.

Weight model. A BX is Poisson(mu) interactions drawn from the N_int-interaction
library, so any single interaction is present with expected count p = mu/N_int.
  - cross pair (left proton from interaction i, right from j != i): both parents
    present => weight p**2.
  - same-interaction pair (one diffractive interaction emitting an accepted
    proton on each side): weight p.
The expected number of window pairs per BX is
  lambda = N_cross * p**2 + N_same * p = sum of all pair weights.
Taking "the BX has a pair" as ">= 1 window pair exists" (we do not care how many;
one is picked if any exist), the fraction of BX carrying a proton pair --
bx_pair_acceptance, the weight that multiplies the MadGraph background yield when
these fake pairs are assigned to central events -- is obtained by directly
Poisson-sampling BX: draw Poisson(mu) interactions and test for a window pair.
That Poisson-sampled value is the authoritative one. The closed form
  bx_pair_acceptance ~= 1 - exp(-lambda)
is also reported, but it assumes the window-pair count per BX is Poisson, i.e.
that the pairs are approximately independent; that holds only when window pairs
are sparse (a small fraction of proton combinations land in the window, as the
narrow PPS mass window ensures) and breaks when the window is dense enough that
many pairs share the same few protons. The two agree in the sparse regime and
the Poisson-sampled number is used when they do not.

Stored pool. Every same-interaction pair is stored (they are rare); the cross
pairs are stored in full, or uniformly subsampled to --max-pairs. Each stored
pair carries a weight column whose sum over the pool equals lambda regardless of
subsampling, so sampling pairs in proportion to weight reproduces the correct
cross/same mixture and (xi_left, xi_right) shape.

Run under the analysis environment:
  source setup_env.sh
  python3 analysis/minbias_analyzer_bootstrap.py --campaign <name>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_minbias_campaign, resolve_path  # noqa: E402
from analysis.minbias_analyzer import (  # noqa: E402
    PROTON_COLUMNS,
    discover_inputs,
    load_pps_config,
    parquet_event_info,
    station_memberships,
)

OUTPUT_NAME = "proton_pairs.parquet"
METADATA_NAME = "metadata.json"
PARQUET_COMPRESSION = "snappy"
DEFAULT_MASS_WINDOW_GEV = (117.0, 133.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--campaign", default=None,
        help="Min-bias campaign. Defaults to config.yaml minbias.default_campaign.",
    )
    parser.add_argument("--mu", type=float, default=200.0, help="Mean interactions per BX")
    parser.add_argument(
        "--mass-window", type=float, nargs=2, default=list(DEFAULT_MASS_WINDOW_GEV),
        metavar=("LO", "HI"), help="Double-tag mx window in GeV (default 117 133).",
    )
    parser.add_argument(
        "--n-bx", type=int, default=200000,
        help="Number of BX to Poisson-sample when cross-checking the acceptance.",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=20_000_000,
        help="Cap on stored cross pairs; the pool is uniformly subsampled above this.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Defaults to random.seed from --pps-config.",
    )
    parser.add_argument(
        "--pps-config", default="analysis/scripts/new/config.yaml",
        help="YAML file defining beam.sqrt_s_gev, pps.xi_ranges, and random.seed.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Maximum input Parquet files")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to output/minbias/<campaign>/pairs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    return parser.parse_args()


def read_accepted_protons(file_infos, pps):
    """Global PPS-accepted forward protons across every interaction.

    Returns xi, side (-1 negative/left, +1 positive/right), and a global
    interaction id, plus the total interaction count (all interactions, whether
    or not they yielded an accepted proton).
    """
    beam_energy = pps["sqrt_s"] / 2.0
    xi_parts, side_parts, interaction_parts = [], [], []
    global_offset = 0
    for info in file_infos:
        filters = [
            ("pdg_id", "=", 2212),
            ("is_final", "=", True),
            ("pz", "!=", 0.0),
        ]
        table = pq.read_table(info["path"], columns=list(PROTON_COLUMNS), filters=filters)
        if table.num_rows:
            event_ids = np.asarray(table["event_id"], dtype=np.int64)
            pz = np.asarray(table["pz"], dtype=np.float64)
            energy = np.asarray(table["E"], dtype=np.float64)
            xi = (beam_energy - energy) / beam_energy
            valid = np.isfinite(xi) & (xi > 0.0)
            accepted, _stations = station_memberships(xi, pps["xi_ranges"])
            selected = np.nonzero(valid & accepted)[0]
            if selected.size:
                xi_parts.append(xi[selected])
                side_parts.append(np.where(pz[selected] < 0.0, -1, 1).astype(np.int8))
                interaction_parts.append(
                    global_offset + event_ids[selected] - info["event_min"]
                )
        global_offset += info["n_events"]

    if not xi_parts:
        raise RuntimeError("No PPS-accepted forward protons found in the campaign")
    return {
        "xi": np.concatenate(xi_parts),
        "side": np.concatenate(side_parts),
        "interaction": np.concatenate(interaction_parts).astype(np.int64),
        "total_interactions": global_offset,
    }


def group_by_interaction(interaction, xi):
    """Map interaction id -> array of that interaction's xi values (one side)."""
    order = np.argsort(interaction, kind="stable")
    ids_sorted = interaction[order]
    xi_sorted = xi[order]
    uniq, starts = np.unique(ids_sorted, return_index=True)
    ends = np.r_[starts[1:], ids_sorted.size]
    return {int(u): xi_sorted[s:e] for u, s, e in zip(uniq, starts, ends)}


def window_bounds(mass_window, sqrt_s):
    """xi_left * xi_right bounds for mx = sqrt(xi_l xi_r) sqrt_s in the window."""
    lo, hi = mass_window
    return (lo / sqrt_s) ** 2, (hi / sqrt_s) ** 2


def same_interaction_pairs(left_map, right_map, prod_lo, prod_hi):
    """Window pairs whose two protons come from one diffractive interaction."""
    xi_left, xi_right = [], []
    for interaction, left_xi in left_map.items():
        right_xi = right_map.get(interaction)
        if right_xi is None:
            continue
        l = np.repeat(left_xi, right_xi.size)
        r = np.tile(right_xi, left_xi.size)
        product = l * r
        keep = (product >= prod_lo) & (product <= prod_hi)
        if np.any(keep):
            xi_left.append(l[keep])
            xi_right.append(r[keep])
    if not xi_left:
        return np.empty(0), np.empty(0)
    return np.concatenate(xi_left), np.concatenate(xi_right)


def cross_pair_counts(xi_left, xi_right, prod_lo, prod_hi):
    """Per-left-proton count of right partners in the window, plus sort order.

    Counts include same-interaction partners; the (small) same count is
    subtracted separately by the caller.
    """
    order = np.argsort(xi_right)
    xr_sorted = xi_right[order]
    lo_edge = prod_lo / xi_left
    hi_edge = prod_hi / xi_left
    start = np.searchsorted(xr_sorted, lo_edge, side="left")
    stop = np.searchsorted(xr_sorted, hi_edge, side="right")
    return order, xr_sorted, start, stop, (stop - start)


def materialize_cross_pairs(
    xi_left, left_interaction, order, xr_sorted, right_interaction,
    start, counts, total_pairs, n_cross, max_pairs, rng,
):
    """All (or a uniform subsample of) cross pairs as (xi_left, xi_right) arrays."""
    if n_cross <= 0:
        return np.empty(0), np.empty(0)
    right_interaction_sorted = right_interaction[order]
    if n_cross <= max_pairs:
        # enumerate every window combination, then drop same-interaction ones
        left_idx = np.repeat(np.arange(xi_left.size), counts)
        group_start = np.repeat(np.cumsum(counts) - counts, counts)
        within = np.arange(total_pairs) - group_start
        sorted_pos = np.repeat(start, counts) + within
        cross = left_interaction[left_idx] != right_interaction_sorted[sorted_pos]
        return xi_left[left_idx][cross], xr_sorted[sorted_pos][cross]

    # subsample: pick a left proton in proportion to its partner count, then a
    # uniform partner within its window slice; drop the rare same-interaction hit
    probabilities = counts / counts.sum()
    picks = rng.choice(xi_left.size, size=max_pairs, replace=True, p=probabilities)
    offsets = np.floor(rng.random(max_pairs) * counts[picks]).astype(np.int64)
    sorted_pos = start[picks] + offsets
    cross = left_interaction[picks] != right_interaction_sorted[sorted_pos]
    return xi_left[picks][cross], xr_sorted[sorted_pos][cross]


def poisson_sample_acceptance(left_map, right_map, total_interactions, mu, prod_lo, prod_hi, n_bx, rng):
    """Fraction of Poisson(mu) BX containing >= 1 window pair (cross-check)."""
    relevant = np.union1d(
        np.fromiter(left_map.keys(), dtype=np.int64, count=len(left_map)),
        np.fromiter(right_map.keys(), dtype=np.int64, count=len(right_map)),
    )
    if relevant.size == 0:
        return 0.0
    left_list = [left_map.get(int(d), np.empty(0)) for d in relevant]
    right_list = [right_map.get(int(d), np.empty(0)) for d in relevant]
    lam_relevant = mu * relevant.size / total_interactions

    hits = 0
    for _ in range(n_bx):
        m = rng.poisson(lam_relevant)
        if m == 0:
            continue
        chosen = rng.integers(0, relevant.size, size=m)
        left_xi = np.concatenate([left_list[c] for c in chosen])
        right_xi = np.concatenate([right_list[c] for c in chosen])
        if left_xi.size == 0 or right_xi.size == 0:
            continue
        right_sorted = np.sort(right_xi)
        lo_edge = prod_lo / left_xi
        hi_edge = prod_hi / left_xi
        start = np.searchsorted(right_sorted, lo_edge, side="left")
        stop = np.searchsorted(right_sorted, hi_edge, side="right")
        if np.any(stop > start):
            hits += 1
    return hits / n_bx


def write_outputs(table, metadata, output_dir, overwrite):
    output_file = output_dir / OUTPUT_NAME
    metadata_file = output_dir / METADATA_NAME
    existing = [path for path in (output_file, metadata_file) if path.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Output already exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_file, compression=PARQUET_COMPRESSION)
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output_file, metadata_file


def main():
    args = parse_args()
    if args.mu <= 0.0:
        raise RuntimeError("--mu must be > 0")
    if args.mass_window[0] >= args.mass_window[1]:
        raise RuntimeError("--mass-window LO must be < HI")
    if args.n_bx <= 0 or args.max_pairs <= 0:
        raise RuntimeError("--n-bx and --max-pairs must be > 0")

    campaign_dir, campaign = resolve_minbias_campaign(args.campaign)
    pps_path = resolve_path(args.pps_config, base=ROOT)
    pps = load_pps_config(pps_path)
    seed = args.seed if args.seed is not None else pps["seed"]
    rng = np.random.default_rng(seed)
    files = discover_inputs(campaign_dir, args.max_files)
    file_infos = [parquet_event_info(path) for path in files]

    protons = read_accepted_protons(file_infos, pps)
    n_int = protons["total_interactions"]
    # PPS xi-resolution smearing, applied once per accepted proton after the
    # (truth-xi) PPS acceptance, matching run_dijet_mva.build_minbias_pair_pool
    # so MadGraph fake pairs get the same reco resolution as the SuperChic protons.
    xi_res = float((load_yaml(pps_path).get("pps") or {}).get("xi_res", 0.0))
    if xi_res > 0.0:
        smeared = rng.normal(protons["xi"], xi_res)
        keep = smeared > 0.0
        protons = {
            "xi": smeared[keep],
            "side": protons["side"][keep],
            "interaction": protons["interaction"][keep],
            "total_interactions": n_int,
        }
    left = protons["side"] < 0
    right = protons["side"] > 0
    xi_left_all = protons["xi"][left]
    xi_right_all = protons["xi"][right]
    left_interaction = protons["interaction"][left]
    right_interaction = protons["interaction"][right]
    n_left, n_right = xi_left_all.size, xi_right_all.size
    if n_left == 0 or n_right == 0:
        raise RuntimeError(f"Need accepted protons on both arms (left={n_left}, right={n_right})")

    prod_lo, prod_hi = window_bounds(args.mass_window, pps["sqrt_s"])
    left_map = group_by_interaction(left_interaction, xi_left_all)
    right_map = group_by_interaction(right_interaction, xi_right_all)

    order, xr_sorted, start, stop, counts = cross_pair_counts(
        xi_left_all, xi_right_all, prod_lo, prod_hi
    )
    total_pairs = int(counts.sum())
    same_xi_left, same_xi_right = same_interaction_pairs(left_map, right_map, prod_lo, prod_hi)
    n_same = same_xi_left.size
    n_cross = total_pairs - n_same
    if n_cross < 0:
        raise RuntimeError("Same-interaction pairs exceed total pairs (internal inconsistency)")

    p_int = args.mu / n_int
    cross_weight = p_int * p_int
    same_weight = p_int
    lam = n_cross * cross_weight + n_same * same_weight
    acceptance_analytic = 1.0 - np.exp(-lam)

    cross_xi_left, cross_xi_right = materialize_cross_pairs(
        xi_left_all, left_interaction, order, xr_sorted, right_interaction,
        start, counts, total_pairs, n_cross, args.max_pairs, rng,
    )
    n_cross_stored = cross_xi_left.size
    # weight per stored cross pair carries the full cross population when
    # subsampled, so the pool weights still sum to lambda.
    stored_cross_weight = (
        cross_weight * (n_cross / n_cross_stored) if n_cross_stored else 0.0
    )

    pool_xi_left = np.concatenate([cross_xi_left, same_xi_left])
    pool_xi_right = np.concatenate([cross_xi_right, same_xi_right])
    pool_weight = np.concatenate([
        np.full(n_cross_stored, stored_cross_weight),
        np.full(n_same, same_weight),
    ])
    pool_same = np.concatenate([
        np.zeros(n_cross_stored, dtype=bool),
        np.ones(n_same, dtype=bool),
    ])
    pool_mx = np.sqrt(pool_xi_left * pool_xi_right) * pps["sqrt_s"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pool_yx = 0.5 * np.log(pool_xi_right / pool_xi_left)

    acceptance_poisson = poisson_sample_acceptance(
        left_map, right_map, n_int, args.mu, prod_lo, prod_hi, args.n_bx, rng
    )

    table = pa.table({
        "xi_left": pool_xi_left,
        "xi_right": pool_xi_right,
        "mx": pool_mx,
        "yx": pool_yx,
        "weight": pool_weight,
        "same_interaction": pool_same,
    })
    output_dir = (
        resolve_path(args.output_dir, base=ROOT) if args.output_dir else campaign_dir / "pairs"
    )
    metadata = {
        "schema_version": 1,
        "campaign": campaign,
        "input_files": [str(path) for path in files],
        "mu": float(args.mu),
        "seed": int(seed),
        "sqrt_s_gev": float(pps["sqrt_s"]),
        "xi_resolution": xi_res,
        "pps_xi_ranges": {station: [low, high] for station, low, high in pps["xi_ranges"]},
        "mass_window_gev": [float(args.mass_window[0]), float(args.mass_window[1])],
        "total_interactions": int(n_int),
        "n_left_accepted": int(n_left),
        "n_right_accepted": int(n_right),
        "n_double_proton_pairs": int(n_same),
        "n_window_pairs_total": int(total_pairs),
        "n_cross_pairs": int(n_cross),
        "n_cross_pairs_stored": int(n_cross_stored),
        "per_interaction_presence_p": float(p_int),
        "cross_pair_weight": float(cross_weight),
        "same_pair_weight": float(same_weight),
        "stored_cross_pair_weight": float(stored_cross_weight),
        "lambda_expected_pairs_per_bx": float(lam),
        "bx_pair_acceptance_analytic": float(acceptance_analytic),
        "bx_pair_acceptance_poisson": float(acceptance_poisson),
        "n_bx_sampled": int(args.n_bx),
        "parquet_compression": PARQUET_COMPRESSION,
    }
    output_file, metadata_file = write_outputs(table, metadata, output_dir, args.overwrite)

    print(
        f"Interactions={n_int}, accepted protons left={n_left} right={n_right}; "
        f"window pairs={total_pairs} (cross={n_cross}, same-interaction={n_same})",
        flush=True,
    )
    print(
        f"Per-pair weights: cross=(mu/N_int)^2={cross_weight:.6e}, "
        f"same-interaction=mu/N_int={same_weight:.6e}",
        flush=True,
    )
    print(f"lambda (expected window pairs per BX) = {lam:.6e}", flush=True)
    print(
        "Computed BX proton-pair weight (fraction of BX with a pair): "
        f"Poisson-sampled ({args.n_bx} BX) = {acceptance_poisson:.6e}  [authoritative]; "
        f"analytic 1-exp(-lambda) = {acceptance_analytic:.6e}  [sparse-window approx]",
        flush=True,
    )
    print(f"Stored pool: {pool_weight.size} pairs (sum of weights = {pool_weight.sum():.6e})", flush=True)
    print(f"Wrote {output_file}")
    print(f"Wrote {metadata_file}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
