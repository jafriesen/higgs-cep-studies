#!/usr/bin/env python3
"""Build the missing trigger reports for the paper timing-resolution scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minbias.artifact import DEFAULT_CONFIG, load_config, write_json
from minbias.flux import Acceptance, ProtonFlux
from trigger.dijet_rate import (
    load_hardqcd_campaign,
    load_jet_correction,
    study_dijet_trigger_rates,
)
from trigger.minbias_rate import study_trigger_rates

DATA_DIR = ROOT / "paper-plots/data/trigger-timing"
TIMINGS_PS = (1, 3, 5, 8, 10, 15, 20, 30)
ESTABLISHED_TIMINGS = (3, 10)
PPS_ARTIFACT = ROOT / "output/minbias/minbias_inelastic_100m_v1/protons.parquet"
HARDQCD_CAMPAIGN = ROOT / "output/trigger/hardqcd_pthat10_150k_v1"
JET_CORRECTIONS = ROOT / "jet-energy/output/stage2_3/corrections.yaml"
BX_FREQUENCY_HZ = 31.0e6
NEW_MINBIAS_BX = 100_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild paper-plots reports even when a validated cache exists.",
    )
    return parser.parse_args()


def established_path(kind: str, timing: int) -> Path:
    if kind == "minbias":
        return ROOT / f"output/minbias/minbias_trigger_rate_{timing}ps_31mhz.json"
    return HARDQCD_CAMPAIGN / f"dijet_rate_20_20_{timing}ps.json"


def generated_path(kind: str, timing: int) -> Path:
    return DATA_DIR / f"{kind}_{timing}ps.json"


def report_path(kind: str, timing: int) -> Path:
    if timing in ESTABLISHED_TIMINGS:
        return established_path(kind, timing)
    return generated_path(kind, timing)


def valid_report(path: Path, kind: str, timing: int) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError):
        return False
    configuration = report.get("configuration", {})
    timing_key = (
        "single_arm_time_resolution_ps"
        if kind == "minbias"
        else "pps_time_resolution_ps"
    )
    expected_bx = (
        1_000_000
        if kind == "minbias" and timing in ESTABLISHED_TIMINGS
        else 100_000
    )
    return (
        float(configuration.get(timing_key, -1.0)) == float(timing)
        and int(configuration.get("n_bx", -1)) == expected_bx
        and float(configuration.get("bx_frequency_hz", -1.0)) == BX_FREQUENCY_HZ
    )


def missing_timings(kind: str, force: bool) -> list[int]:
    missing = []
    for timing in TIMINGS_PS:
        path = report_path(kind, timing)
        if timing in ESTABLISHED_TIMINGS:
            if not valid_report(path, kind, timing):
                raise RuntimeError(f"Established {kind} report is invalid: {path}")
            continue
        if force or not valid_report(path, kind, timing):
            missing.append(timing)
    return missing


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing_minbias = missing_timings("minbias", args.force)
    missing_dijet = missing_timings("dijet", args.force)
    if not missing_minbias and not missing_dijet:
        print("All timing-scan reports are already complete.")
        return

    config = load_config(DEFAULT_CONFIG)
    flux = ProtonFlux.load(PPS_ARTIFACT)
    acceptance = Acceptance(config["xi_windows"])

    for timing in missing_minbias:
        print(f"Running minimum-bias timing point: {timing} ps", flush=True)
        report = study_trigger_rates(
            flux,
            acceptance,
            n_bx=NEW_MINBIAS_BX,
            xi_resolution=config["xi_resolution"],
            single_arm_time_resolution_ps=timing,
            bx_frequency_hz=BX_FREQUENCY_HZ,
            seed=config["seed"],
            progress=True,
        )
        write_json(generated_path("minbias", timing), report)

    if missing_dijet:
        correction_map, correction_info = load_jet_correction(JET_CORRECTIONS)
        hard = load_hardqcd_campaign(
            HARDQCD_CAMPAIGN,
            correction_map=correction_map,
            correction_info=correction_info,
            leading_pt_min=20.0,
            subleading_pt_min=20.0,
            eta_max=2.4,
            progress=True,
        )
        for timing in missing_dijet:
            print(f"Running dijet timing point: {timing} ps", flush=True)
            report = study_dijet_trigger_rates(
                hard,
                flux,
                acceptance,
                n_bx=100_000,
                xi_resolution=config["xi_resolution"],
                single_arm_time_resolution_ps=timing,
                bx_frequency_hz=BX_FREQUENCY_HZ,
                seed=12345,
                n_bootstrap=200,
                progress=True,
            )
            write_json(generated_path("dijet", timing), report)

    print("Completed trigger timing scan:")
    for timing in TIMINGS_PS:
        print(f"  {timing:>2} ps: {report_path('minbias', timing)}")
        print(f"         {report_path('dijet', timing)}")


if __name__ == "__main__":
    main()
