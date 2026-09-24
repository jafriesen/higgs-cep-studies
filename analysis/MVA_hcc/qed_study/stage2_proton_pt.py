#!/usr/bin/env python3
"""Stage 2: truth-level proton pT as a discriminant against the QED continuum.

gamma-gamma exchange leaves the proton with pT set by the elastic form factor;
QCD (pomeron) exchange leaves a substantially harder proton. The PPS model in
analysis/scripts/new/config.yaml carries only xi acceptance and xi resolution --
there is no |t| acceptance or resolution -- so this stage is deliberately
truth-level: it measures how much separation exists before asking whether a
detector could record it, and reports the resolution that would be required.

MadGraph is absent by construction: the min-bias pair pool
(proton_pairs.parquet) stores xi_left, xi_right, mx, yx and weight only, so a
combinatorial proton pair has no pT to sample.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))

from analysis.MVA import run_dijet_mva as pps_lib  # noqa: E402
from analysis.MVA import run_dijet_mva_multiclass_protons as source  # noqa: E402
from analysis.MVA_hcc.common import COMPONENT_SPECS  # noqa: E402
from analysis.MVA_hcc.prepare_dataset import resolve_superchic_files  # noqa: E402
from analysis.MVA_hcc.qed_study.common import (  # noqa: E402
    DEFAULT_OUTPUT, assign_folds, held_out_threshold_z, mass_binned_z, mass_histogram,
    scan_threshold_z, separation, write_yaml,
)
from analysis.MVA_hcc.qed_study.stage1_qed_variables import (  # noqa: E402
    background_efficiency, oof_binary,
)
from common.config_utils import resolve_path  # noqa: E402

REPO = SCRIPT_DIR.parents[2]
COMPONENTS = ("Hcc", "QCDcc_superchic", "QEDcc_superchic")
PROCESS = {"Hcc": "Hcc", "QCDcc_superchic": "QCDcc", "QEDcc_superchic": "QEDcc"}
SMEAR_GEV = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)
MASS_WINDOW_GEV = pps_lib.MASS_WINDOW_GEV
MAX_ABS_RAPIDITY_DIFFERENCE = pps_lib.MAX_ABS_RAPIDITY_DIFFERENCE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "stage2"))
    parser.add_argument("--pps-config", default="analysis/scripts/new/config.yaml")
    parser.add_argument("--tree", default="Delphes")
    parser.add_argument("--collection", default="JetPUPPI")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def parse_lhe_protons(input_file, selected_event_indices, sqrt_s):
    """Outgoing-proton xi and transverse momentum from a SuperChic LHE record.

    Extends prepare_dataset.parse_lhe_proton_xi, which reads only pz (field 8)
    and E (field 9), with px and py (fields 6 and 7).
    """
    shape = selected_event_indices.shape
    xi_left = np.full(shape, np.nan)
    xi_right = np.full(shape, np.nan)
    pt_left = np.full(shape, np.nan)
    pt_right = np.full(shape, np.nan)
    phi_left = np.full(shape, np.nan)
    phi_right = np.full(shape, np.nan)
    if selected_event_indices.size == 0:
        return xi_left, xi_right, pt_left, pt_right, phi_left, phi_right

    positions = {int(event): index for index, event in enumerate(selected_event_indices)}
    last = int(selected_event_indices[-1])
    beam_energy = sqrt_s / 2.0
    event_index = -1
    output_index = None
    left = right = None
    left_abs_pz = right_abs_pz = -1.0

    def store():
        if output_index is None:
            return
        for side, record in (("left", left), ("right", right)):
            if record is None:
                continue
            px, py, _pz, energy = record
            target_xi = xi_left if side == "left" else xi_right
            target_pt = pt_left if side == "left" else pt_right
            target_phi = phi_left if side == "left" else phi_right
            target_xi[output_index] = (beam_energy - energy) / beam_energy
            target_pt[output_index] = np.hypot(px, py)
            target_phi[output_index] = np.arctan2(py, px)

    with open(input_file, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("<event"):
                event_index += 1
                if event_index > last:
                    break
                output_index = positions.get(event_index)
                left = right = None
                left_abs_pz = right_abs_pz = -1.0
            elif stripped.startswith("</event"):
                store()
                output_index = None
            elif output_index is not None:
                fields = stripped.split()
                if len(fields) < 10:
                    continue
                try:
                    pid, status = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                if pid != 2212 or status != 1:
                    continue
                px, py = float(fields[6]), float(fields[7])
                pz, energy = float(fields[8]), float(fields[9])
                if pz < 0.0 and abs(pz) > left_abs_pz:
                    left, left_abs_pz = (px, py, pz, energy), abs(pz)
                elif pz > 0.0 and abs(pz) > right_abs_pz:
                    right, right_abs_pz = (px, py, pz, energy), abs(pz)
    return xi_left, xi_right, pt_left, pt_right, phi_left, phi_right


def piece(arguments):
    """One ROOT file: the same selection prepare_dataset applies, plus proton pT."""
    path, proton_file, tree_name, collection, pps, seed = arguments
    central = source.central_features(path, tree_name, collection)
    if central.get("empty"):
        return None
    xi_left, xi_right, pt_left, pt_right, phi_left, phi_right = parse_lhe_protons(
        proton_file, central["event_indices"], pps["sqrt_s"]
    )
    passed, _left, _right, mx, yx = source.real_proton_pass(
        np, xi_left, xi_right, pps, np.random.default_rng(seed)
    )
    keep = (
        passed
        & (np.abs(yx - central["dijet_rapidity"]) < MAX_ABS_RAPIDITY_DIFFERENCE)
        & (mx >= MASS_WINDOW_GEV[0])
        & (mx <= MASS_WINDOW_GEV[1])
    )
    names = list(source.FEATURE_NAMES)
    matrix = np.column_stack([central["matrix"], yx - central["dijet_rapidity"]])
    return {
        "mx": mx[keep],
        "delta_eta_jj": matrix[keep, names.index("delta_eta_jj")],
        "pt_left": pt_left[keep],
        "pt_right": pt_right[keep],
        "phi_left": phi_left[keep],
        "phi_right": phi_right[keep],
        "n_generated": central["n_generated"],
    }


def load_component(name, args, pps):
    spec = {"process": PROCESS[name]}
    campaign, _sub, _dir, files, kind, proton_files = resolve_superchic_files(spec, args.max_files)
    if kind != "lhe":
        raise RuntimeError(f"{name} has {kind} proton records; this stage needs LHE px/py")
    tasks = [
        # Seed offset keyed to the global COMPONENT_SPECS position, so this stage
        # selects exactly the events prepare_dataset.py selects.
        (path, proton_file, args.tree, args.collection, pps,
         args.seed + [spec["name"] for spec in COMPONENT_SPECS].index(name) * 10000 + index)
        for index, (path, proton_file) in enumerate(zip(files, proton_files))
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        pieces = [item for item in executor.map(piece, tasks) if item is not None]
    merged = {
        key: np.concatenate([item[key] for item in pieces])
        for key in ("mx", "delta_eta_jj", "pt_left", "pt_right", "phi_left", "phi_right")
    }
    merged["delta_phi_pp"] = wrapped_delta_phi(merged["phi_left"], merged["phi_right"])
    merged["n_generated"] = sum(item["n_generated"] for item in pieces)
    merged["campaign"] = campaign
    print(f"{name}: {merged['mx'].size:,} selected events from {merged['n_generated']:,} generated",
          flush=True)
    return merged


def wrapped_delta_phi(left, right):
    difference = left - right
    return np.abs(np.arctan2(np.sin(difference), np.cos(difference)))


def smear_proton(pt, phi, sigma, rng):
    """Gaussian smearing of (px, py), returning the smeared (pT, phi).

    pT and the azimuth come from the same measurement, so they have to be
    smeared together: a resolution good enough for one is good enough for both.
    """
    if sigma <= 0.0:
        return pt, phi
    px = pt * np.cos(phi) + rng.normal(0.0, sigma, pt.size)
    py = pt * np.sin(phi) + rng.normal(0.0, sigma, pt.size)
    return np.hypot(px, py), np.arctan2(py, px)


def smeared_block(sample, sigma, rng):
    """A copy of the observable block with both protons smeared consistently."""
    pt_left, phi_left = smear_proton(sample["pt_left"], sample["phi_left"], sigma, rng)
    pt_right, phi_right = smear_proton(sample["pt_right"], sample["phi_right"], sigma, rng)
    return {
        "mx": sample["mx"],
        "delta_eta_jj": sample["delta_eta_jj"],
        "pt_left": pt_left,
        "pt_right": pt_right,
        "delta_phi_pp": wrapped_delta_phi(phi_left, phi_right),
    }


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pps = pps_lib.load_pps_config(resolve_path(args.pps_config, base=REPO))
    samples = {name: load_component(name, args, pps) for name in COMPONENTS}

    # Physical weights: cross section per selected event, matching prepare_dataset.
    from analysis.cross_sections import generator_cross_section_fb, generator_weight
    from analysis.MVA_hcc.common import read_parameters, tag_factor
    parameters = read_parameters()
    weights = {}
    for name, sample in samples.items():
        xsec_fb, _src = generator_cross_section_fb("superchic", PROCESS[name], sample["campaign"])
        per_event = (
            xsec_fb * generator_weight("superchic", PROCESS[name])
            * pps_lib.LUMI_FB * tag_factor(parameters, "cc") / sample["n_generated"]
        )
        weights[name] = np.full(sample["mx"].size, per_event)
        print(f"  {name}: yield {per_event * sample['mx'].size:.4f}", flush=True)

    signal, qed, qcd = samples["Hcc"], samples["QEDcc_superchic"], samples["QCDcc_superchic"]
    rng = np.random.default_rng(args.seed)

    # ---- truth-level separation, and how it degrades with pT resolution
    observables = {}
    for label, background, background_weight in (
        ("QEDcc_superchic", qed, weights["QEDcc_superchic"]),
        ("QCDcc_superchic", qcd, weights["QCDcc_superchic"]),
    ):
        block = {}
        for column in ("pt_left", "pt_right", "delta_phi_pp", "delta_eta_jj"):
            block[column] = separation(signal[column], weights["Hcc"],
                                       background[column], background_weight)
        signal_mean = np.mean(0.5 * (signal["pt_left"] + signal["pt_right"]))
        background_mean = np.mean(0.5 * (background["pt_left"] + background["pt_right"]))
        block["mean_proton_pt_signal_gev"] = float(signal_mean)
        block["mean_proton_pt_background_gev"] = float(background_mean)
        block["pt_pair_resolution_scan"] = {}
        block["delta_phi_resolution_scan"] = {}
        for sigma in SMEAR_GEV:
            smeared_signal = smeared_block(signal, sigma, rng)
            smeared_background = smeared_block(background, sigma, rng)
            block["pt_pair_resolution_scan"][sigma] = separation(
                0.5 * (smeared_signal["pt_left"] + smeared_signal["pt_right"]), weights["Hcc"],
                0.5 * (smeared_background["pt_left"] + smeared_background["pt_right"]),
                background_weight,
            )
            block["delta_phi_resolution_scan"][sigma] = separation(
                smeared_signal["delta_phi_pp"], weights["Hcc"],
                smeared_background["delta_phi_pp"], background_weight,
            )
        observables[label] = block

    print("\nobservable                        vs QEDcc   vs QCDcc_SC")
    for column in ("pt_left", "pt_right", "delta_phi_pp", "delta_eta_jj"):
        print(f"  {column:30s} {observables['QEDcc_superchic'][column]:8.4f} "
              f"{observables['QCDcc_superchic'][column]:12.4f}")
    print("\nmean proton pT (GeV): Hcc "
          f"{observables['QEDcc_superchic']['mean_proton_pt_signal_gev']:.4f}  QEDcc "
          f"{observables['QEDcc_superchic']['mean_proton_pt_background_gev']:.4f}  QCDcc_SC "
          f"{observables['QCDcc_superchic']['mean_proton_pt_background_gev']:.4f}")
    print("\nresolution scan (separation of mean pT / of delta_phi_pp):")
    print(f"  {'sigma [GeV]':>12s} {'pT vs QED':>10s} {'dphi vs QED':>12s} "
          f"{'pT vs QCD':>10s} {'dphi vs QCD':>12s}")
    for sigma in SMEAR_GEV:
        against_qed = observables["QEDcc_superchic"]
        against_qcd = observables["QCDcc_superchic"]
        print(f"  {sigma:12.2f} {against_qed['pt_pair_resolution_scan'][sigma]:10.4f} "
              f"{against_qed['delta_phi_resolution_scan'][sigma]:12.4f} "
              f"{against_qcd['pt_pair_resolution_scan'][sigma]:10.4f} "
              f"{against_qcd['delta_phi_resolution_scan'][sigma]:12.4f}")

    # ---- what it buys on top of the production angle
    gains = {}
    for label, background, background_weight in (
        ("QEDcc_superchic", qed, weights["QEDcc_superchic"]),
        ("QCDcc_superchic+QEDcc_superchic", None, None),
    ):
        if background is None:
            background = {key: np.r_[qed[key], qcd[key]]
                          for key in ("mx", "delta_eta_jj", "pt_left", "pt_right",
                                      "phi_left", "phi_right")}
            background_weight = np.r_[weights["QEDcc_superchic"], weights["QCDcc_superchic"]]
        mx = np.r_[signal["mx"], background["mx"]]
        weight = np.r_[weights["Hcc"], background_weight]
        labels = np.r_[np.ones(signal["mx"].size, np.int32),
                       np.zeros(background["mx"].size, np.int32)]
        is_signal = labels == 1
        groups = np.arange(labels.size)
        base_z = mass_binned_z(mass_histogram(mx, weight, is_signal),
                               mass_histogram(mx, weight, ~is_signal))
        block = {"preselection_z": base_z}
        cases = [("angle_only", ("delta_eta_jj",), 0.0)]
        for sigma in (0.0, 0.05, 0.1):
            tag = "truth" if sigma == 0.0 else f"{int(sigma * 1000)}mev"
            cases += [
                (f"proton_pt_only_{tag}", ("pt_left", "pt_right"), sigma),
                (f"angle_plus_proton_pt_{tag}", ("delta_eta_jj", "pt_left", "pt_right"), sigma),
                (f"angle_plus_proton_pt_and_dphi_{tag}",
                 ("delta_eta_jj", "pt_left", "pt_right", "delta_phi_pp"), sigma),
            ]
        for name, columns, sigma in cases:
            smeared_signal = smeared_block(signal, sigma, rng)
            smeared_background = smeared_block(background, sigma, rng)
            matrix = np.column_stack([np.r_[smeared_signal[column], smeared_background[column]]
                                      for column in columns]).astype(np.float32)
            score = oof_binary(matrix, labels, weight, groups, args.seed, 400)
            _t, _v, cut, best = scan_threshold_z(score, mx, weight, is_signal)
            held_out = held_out_threshold_z(
                score, mx, weight, is_signal, assign_folds(groups, args.seed + 1)
            )
            block[name] = {
                "best_z": best,
                "held_out_z": held_out,
                "smear_sigma_gev": sigma,
                "gain_over_preselection": best / base_z,
                "background_efficiency_at_signal_efficiency":
                    background_efficiency(score, is_signal, weight, (0.2, 0.3, 0.5, 0.7)),
            }
            print(f"  [{label:32s}] {name:30s} Z={best:.4f} "
                  f"held_out={held_out:.4f} (x{held_out / base_z:.3f})", flush=True)
        gains[label] = block

    payload = {
        "description": "Truth-level proton pT versus the exclusive continuum",
        "components": {name: {"selected": int(sample["mx"].size),
                              "generated": int(sample["n_generated"]),
                              "campaign": sample["campaign"]}
                       for name, sample in samples.items()},
        "separation": observables,
        "mass_binned_significance": gains,
        "smear_sigma_gev": list(SMEAR_GEV),
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "caveats": [
            "Truth-level only: the PPS configuration models xi acceptance and xi "
            "resolution, with no |t| acceptance and no |t| resolution.",
            "MadGraph is excluded because the min-bias pair pool stores no proton pT.",
            "|t| ~ pT^2 for elastic scattering; pT is reported directly.",
            "pT and the proton azimuth are smeared together, since both come from "
            "the same measurement.",
        ],
    }
    write_yaml(output_dir / "stage2_report.yaml", payload)
    np.savez_compressed(
        output_dir / "stage2_arrays.npz",
        **{f"{name}_{key}": samples[name][key]
           for name in COMPONENTS
           for key in ("mx", "delta_eta_jj", "pt_left", "pt_right",
                       "phi_left", "phi_right", "delta_phi_pp")},
        **{f"{name}_weight": weights[name] for name in COMPONENTS},
    )
    print(f"\nWrote {output_dir} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
