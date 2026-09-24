#!/usr/bin/env python3
"""Prepare and optionally submit a sharded HardQCD Condor campaign."""

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

from minbias.artifact import sha256, utc_timestamp, write_json
from trigger.hardqcd_campaign import MANIFEST_SCHEMA_VERSION
from trigger.produce_hardqcd import DEFAULT_CARD, ROOT


DEFAULT_CAMPAIGN = ROOT / "output/trigger/hardqcd_pthat10_150k_v1"
MAX_PYTHIA_SEED = 900_000_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--events", type=int, default=150_000)
    parser.add_argument("--derivation-events", type=int, default=50_000)
    parser.add_argument("--events-per-job", type=int, default=5_000)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--ecm-gev", type=float, default=14_000.0)
    parser.add_argument("--pthat-min-gev", type=float, default=10.0)
    parser.add_argument("--tune", type=int, default=14)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--request-memory-mb", type=int, default=4096)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def _job_counts(events, events_per_job):
    return [
        min(events_per_job, events - start)
        for start in range(0, events, events_per_job)
    ]


def render_runner():
    return f"""#!/usr/bin/env bash
set -euo pipefail

JOB_INDEX="${{1:?missing job index}}"
EVENTS="${{2:?missing event count}}"
SEED="${{3:?missing seed}}"
SHARD_DIR="${{4:?missing shard directory}}"
ECM_GEV="${{5:?missing collision energy}}"
PTHAT_MIN_GEV="${{6:?missing pTHat minimum}}"
TUNE="${{7:?missing tune}}"
CARD="${{8:?missing Delphes card}}"

cd {shlex.quote(str(ROOT))}
source setup_env.sh
python3 -u -m trigger.produce_hardqcd \\
  --campaign-dir "$SHARD_DIR" \\
  --events "$EVENTS" \\
  --seed "$SEED" \\
  --ecm-gev "$ECM_GEV" \\
  --pthat-min-gev "$PTHAT_MIN_GEV" \\
  --tune "$TUNE" \\
  --card "$CARD" \\
  --discard-hepmc
echo "Completed HardQCD job $JOB_INDEX"
"""


def prepare(args):
    if args.events <= 0 or args.events_per_job <= 0:
        raise RuntimeError("--events and --events-per-job must be positive")
    if not 0 < args.derivation_events < args.events:
        raise RuntimeError("--derivation-events must be between zero and --events")
    if args.ecm_gev <= 0.0 or args.pthat_min_gev < 0.0:
        raise RuntimeError("collision energy must be positive and pTHat nonnegative")
    if args.request_memory_mb <= 0:
        raise RuntimeError("--request-memory-mb must be positive")
    card = args.card.resolve()
    if not card.is_file():
        raise RuntimeError(f"Delphes card not found: {card}")
    campaign = args.campaign_dir.resolve()
    if campaign.exists():
        raise RuntimeError(f"Campaign directory already exists: {campaign}")

    rate_events = args.events - args.derivation_events
    job_specs = []
    for role, events in (
        ("derivation", args.derivation_events),
        ("rate_validation", rate_events),
    ):
        for count in _job_counts(events, args.events_per_job):
            index = len(job_specs)
            seed = args.seed_base + index
            job_specs.append(
                {
                    "index": index,
                    "role": role,
                    "events": count,
                    "seed": seed,
                    "campaign_dir": str(campaign / "shards" / f"job_{index:05d}"),
                }
            )
    seeds = [job["seed"] for job in job_specs]
    if args.seed_base < 1 or seeds[-1] > MAX_PYTHIA_SEED:
        raise RuntimeError("Requested seeds are outside Pythia's valid range")

    condor_dir = campaign / "condor"
    condor_dir.mkdir(parents=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign": campaign.name,
        "events": args.events,
        "derivation_events": args.derivation_events,
        "rate_validation_events": rate_events,
        "events_per_job": args.events_per_job,
        "jobs_requested": len(job_specs),
        "configuration": {
            "sqrt_s_gev": float(args.ecm_gev),
            "pthat_min_gev": float(args.pthat_min_gev),
            "tune_pp": int(args.tune),
            "card": str(card),
            "card_sha256": sha256(card),
        },
        "created_at": utc_timestamp(),
        "jobs": job_specs,
    }
    manifest_path = campaign / "manifest.json"
    write_json(manifest_path, manifest)

    queue_path = condor_dir / "queue.txt"
    queue_path.write_text(
        "\n".join(
            f"{job['index']} {job['events']} {job['seed']} {job['campaign_dir']}"
            for job in job_specs
        )
        + "\n",
        encoding="utf-8",
    )
    runner_path = condor_dir / "run_job.sh"
    runner_path.write_text(render_runner(), encoding="utf-8")
    runner_path.chmod(0o755)
    submit_path = condor_dir / "submit.sub"
    arguments = (
        "$(JOB_INDEX) $(EVENTS) $(SEED) $(SHARD_DIR) "
        f"{args.ecm_gev} {args.pthat_min_gev} {args.tune} {card}"
    )
    submit_path.write_text(
        f"""universe = vanilla
executable = {runner_path}
arguments = {arguments}
output = {condor_dir}/job_$(JOB_INDEX).out
error = {condor_dir}/job_$(JOB_INDEX).err
log = {condor_dir}/cluster.log
request_memory = {args.request_memory_mb}
request_cpus = 1
getenv = True
queue JOB_INDEX, EVENTS, SEED, SHARD_DIR from {queue_path}
""",
        encoding="utf-8",
    )
    return campaign, manifest_path, submit_path


def main():
    args = parse_args()
    try:
        campaign, manifest, submit_file = prepare(args)
        print(f"Prepared {manifest}")
        print(f"Condor submit file: {submit_file}")
        if args.submit:
            if shutil.which("condor_submit") is None:
                raise RuntimeError("condor_submit is not available")
            result = subprocess.run(
                ["condor_submit", str(submit_file)],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"condor_submit failed: {result.stderr.strip()}")
            (campaign / "submission.txt").write_text(result.stdout, encoding="utf-8")
            print(result.stdout.strip())
    except RuntimeError as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
