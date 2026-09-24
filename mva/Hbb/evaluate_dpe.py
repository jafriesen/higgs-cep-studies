#!/usr/bin/env python3
"""Evaluate hadronized FPMC DPE samples with the frozen nominal H(bb) BDT."""

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml
from sklearn.linear_model import LogisticRegression
from uproot.source.file import MemmapSource
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from common.jet_calibration import load_correction_map  # noqa: E402
from minbias import vertex_likelihood_table  # noqa: E402
from mva.common.features import FEATURE_NAMES, central_features  # noqa: E402
from mva.common.protons import (  # noqa: E402
    load_pps_config,
    parse_lhe_protons,
    real_proton_pass,
)
from mva.common.training import calibrated_probabilities  # noqa: E402
from mva.common.weights import plugin_score  # noqa: E402


SAMPLES = {
    "bb": {
        "process": "DPEbbHiggsPPS",
        "campaign": "DPEbbHadrY_eval__v02",
        "pythia_tag": "DPEbb_Herwig__v02",
        "delphes_tag": "DPEbb_Herwig_200PU_MTD__v02",
        "truth_pid_abs": 5,
        "correction_source": "QCDbb_superchic",
        "tag_factor": 0.85**2,
    },
    "cc": {
        "process": "DPEccHiggsPPS",
        "campaign": "DPEccHadrY_eval__v02",
        "pythia_tag": "DPEcc_Herwig__v02",
        "delphes_tag": "DPEcc_Herwig_200PU_MTD__v02",
        "truth_pid_abs": 4,
        "correction_source": "QCDcc_superchic",
        "tag_factor": 0.10**2,
    },
    "jj": {
        "process": "DPEjjHiggsPPS",
        "campaign": "DPEjjHadrY_eval__v01",
        "pythia_tag": "DPEjj_Herwig__v01",
        "delphes_tag": "DPEjj_Herwig_200PU_MTD__v01",
        "truth_pid_abs": 21,
        "correction_source": "QCDgg",
        "tag_factor": 0.01**2,
    },
}


def file_number(path):
    match = re.search(r"_(\d+)\.root$", path.name)
    if match is None:
        raise ValueError(f"Cannot identify file number from {path}")
    return int(match.group(1))


def final_fpmc_cross_section(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Cross section\[pb\]=\s*([0-9.Ee+-]+)", text)
    if not matches:
        raise RuntimeError(f"No final FPMC cross section in {path}")
    return float(matches[-1])


def load_calibrator(path):
    payload = np.load(path)
    calibrator = LogisticRegression()
    calibrator.classes_ = payload["classes"]
    calibrator.coef_ = payload["coef"]
    calibrator.intercept_ = payload["intercept"]
    calibrator.n_features_in_ = calibrator.coef_.shape[1]
    return calibrator


def load_models(model_dir):
    output = []
    for fold in (0, 1):
        model = XGBClassifier()
        model.load_model(model_dir / f"fold_{fold}_main_model.json")
        calibrator = load_calibrator(model_dir / f"fold_{fold}_main_calibrator.npz")
        output.append((model, calibrator))
    return output


def event_numbers(path):
    with uproot.open(path, handler=MemmapSource) as root_file:
        values = root_file["Delphes"]["Event.Number"].array(library="ak")
    return ak.to_numpy(ak.firsts(values)).astype(np.int64)


def make_matrix(piece, requested, proton_yx):
    central = {
        name: piece["matrix"][:, index] for index, name in enumerate(FEATURE_NAMES)
    }
    delta_y = proton_yx - piece["dijet_rapidity"]
    columns = []
    for name in requested:
        if name == "yx_minus_dijet_rapidity":
            columns.append(delta_y)
        elif name == "jet1_mass_estimator":
            y_x = piece["dijet_rapidity"] + delta_y
            columns.append(
                2.0 * central["jet1_mt"] * np.cosh(central["jet1_rapidity"] - y_x)
            )
        else:
            columns.append(central[name])
    return np.asarray(np.column_stack(columns), dtype=np.float32)


def fold_pass_probability(matrix, models, kappa, threshold, timing):
    per_fold = []
    base_scores = []
    for model, calibrator in models:
        probabilities = calibrated_probabilities(model, calibrator, matrix)
        base_score = plugin_score(probabilities, [kappa])
        pass_probability = np.zeros(base_score.size, dtype=np.float64)
        for probability, shift in zip(
            timing["matched_probability"], timing["log_likelihood_ratio"]
        ):
            pass_probability += probability * (base_score >= threshold - shift)
        per_fold.append(pass_probability)
        base_scores.append(base_score)
    return np.mean(per_fold, axis=0), np.asarray(base_scores)


def extract_piece(task):
    root_path, truth_pid_abs, correction_source = task
    correction = load_correction_map(
        REPO / "jet-energy/output/fsr_mtd_recovered/corrections.yaml",
        "FSR",
        correction_source,
    )
    return central_features(
        root_path,
        "Delphes",
        "JetPUPPI",
        truth_pid_abs=truth_pid_abs,
        jet_mode="leading",
        correction_map=correction,
        track_min_pt=0.5,
        max_abs_jet_eta=3.0,
    )


def evaluate_sample(name, spec, args, features, kappa, threshold, models, pps, timing):
    base = REPO / "output-fpmc" / spec["process"] / spec["campaign"]
    root_dir = base / "sim-Delphes" / spec["delphes_tag"] / "root"
    lhe_dir = base / "gen-FPMC" / "evrecs"
    log_dir = base / "gen-FPMC" / "logs"
    roots = sorted(root_dir.glob("*.root"), key=file_number)
    if not roots:
        raise RuntimeError(f"No Delphes files found in {root_dir}")
    tasks = [
        (path, spec["truth_pid_abs"], spec["correction_source"]) for path in roots
    ]
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)), mp_context=get_context("spawn")
    ) as executor:
        pieces = list(executor.map(extract_piece, tasks))
    counts = {
        "simulated": 0,
        "two_jets": 0,
        "delta_phi": 0,
        "dijet_mass": 0,
        "pps_pair": 0,
        "proton_mass": 0,
        "rapidity_match": 0,
    }
    bdt_pass = 0.0
    base_scores = []
    selected_events = 0
    for root_path, piece in zip(roots, pieces):
        index = file_number(root_path)
        lhe_path = lhe_dir / f"FPMC_{spec['process']}_{spec['campaign']}_{index}.lhe"
        counts["simulated"] += int(piece["n_generated"])
        for key, value in piece["cutflow"].items():
            counts[key] += int(value)
        if piece.get("empty"):
            continue
        numbers = event_numbers(root_path)[piece["event_indices"]]
        source_indices = numbers - 1
        if np.any(source_indices < 0) or np.any(source_indices >= args.events_per_file):
            raise RuntimeError(f"Invalid source event numbers in {root_path}: {numbers}")
        protons = parse_lhe_protons(lhe_path, source_indices, pps["sqrt_s"])
        accepted, _xl, _xr, mx, yx = real_proton_pass(
            protons["xi_left"],
            protons["xi_right"],
            pps,
            np.random.default_rng(args.seed + index),
        )
        mass = accepted & (mx >= args.mass_window[0]) & (mx <= args.mass_window[1])
        rapidity = mass & (
            np.abs(yx - piece["dijet_rapidity"]) < args.max_delta_y
        )
        counts["pps_pair"] += int(np.sum(accepted))
        counts["proton_mass"] += int(np.sum(mass))
        counts["rapidity_match"] += int(np.sum(rapidity))
        if not np.any(rapidity):
            continue
        matrix = make_matrix(piece, features, yx)
        pass_probability, scores = fold_pass_probability(
            matrix[rapidity], models, kappa, threshold, timing
        )
        bdt_pass += float(np.sum(pass_probability))
        base_scores.append(scores)
        selected_events += int(np.sum(rapidity))
    counts["bdt_timing"] = bdt_pass
    denominator = counts["simulated"]
    efficiencies = {
        key: float(value / denominator) if denominator else 0.0
        for key, value in counts.items()
        if key != "simulated"
    }
    score_array = np.concatenate(base_scores, axis=1) if base_scores else np.empty((2, 0))
    cross_sections_pb = [
        final_fpmc_cross_section(
            log_dir / f"run_{spec['process']}_{spec['campaign']}_{file_number(path)}.log"
        )
        for path in roots
    ]
    cross_section_pb = float(np.mean(cross_sections_pb))
    final_efficiency = efficiencies["bdt_timing"]
    return {
        "sample": name,
        "files": len(roots),
        "counts": counts,
        "efficiencies": efficiencies,
        "selected_for_bdt": selected_events,
        "base_score_quantiles": (
            np.quantile(score_array, [0.1, 0.5, 0.9]).tolist()
            if score_array.size
            else []
        ),
        "base_score_max": float(np.max(score_array)) if score_array.size else None,
        "maximum_combined_score": (
            float(np.max(score_array) + np.max(timing["log_likelihood_ratio"]))
            if score_array.size
            else None
        ),
        "tag_factor": spec["tag_factor"],
        "cross_section_pb": cross_section_pb,
        "cross_section_file_rms_pb": float(np.std(cross_sections_pb, ddof=1)),
        "expected_yield_3000fb": (
            cross_section_pb * 1000.0 * 3000.0 * spec["tag_factor"] * final_efficiency
        ),
        "final_efficiency_stat_error": (
            float(np.sqrt(final_efficiency * (1.0 - final_efficiency) / denominator))
            if denominator
            else 0.0
        ),
        "correction_source": spec["correction_source"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO / "mva/Hbb/results/fsr_mtd_binary_7p1ps_allrows_gg_v04_g256_s12345",
    )
    parser.add_argument("--output", type=Path, default=REPO / "mva/Hbb/results/dpe_frozen_bdt/evaluation.yaml")
    parser.add_argument("--events-per-file", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mass-window", type=float, nargs=2, default=(117.0, 133.0))
    parser.add_argument("--max-delta-y", type=float, default=0.2)
    args = parser.parse_args()

    report = yaml.safe_load((args.model_dir / "report.yaml").read_text())
    threshold = float(report["single_cut_operating_point"]["threshold"])
    kappa = float(report["kappas"][0])
    features = list(report["features"])
    models = load_models(args.model_dir)
    pps = load_pps_config(REPO / "analysis/scripts/new/config.yaml")
    timing = vertex_likelihood_table(
        bins=8,
        beam_sigma_z_cm=5.7,
        single_arm_time_resolution_ps=10.0,
        pv_z_resolution_cm=0.001,
        pv_time_resolution_ps=7.1,
    )
    results = [
        evaluate_sample(name, spec, args, features, kappa, threshold, models, pps, timing)
        for name, spec in SAMPLES.items()
    ]
    by_name = {item["sample"]: item for item in results}
    inclusive_xsec = by_name["jj"]["cross_section_pb"]
    light_xsec = max(
        0.0,
        inclusive_xsec - by_name["bb"]["cross_section_pb"] - by_name["cc"]["cross_section_pb"],
    )
    by_name["jj"]["inclusive_cross_section_pb"] = inclusive_xsec
    by_name["jj"]["cross_section_pb"] = light_xsec
    by_name["jj"]["expected_yield_3000fb"] *= light_xsec / inclusive_xsec
    by_name["jj"]["normalization_note"] = (
        "Dedicated bb and cc cross sections are subtracted from inclusive DPEjj. "
        "The rejection efficiency still uses the inclusive sample as a light-jet proxy."
    )
    payload = {
        "model_dir": str(args.model_dir),
        "features": features,
        "score_threshold": threshold,
        "kappa": kappa,
        "mass_window_gev": list(args.mass_window),
        "max_abs_rapidity_difference": args.max_delta_y,
        "timing": {"pps_single_arm_ps": 10.0, "pv_ps": 7.1, "hypothesis": "matched"},
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
