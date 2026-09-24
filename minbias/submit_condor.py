#!/usr/bin/env python3
"""Prepare and optionally submit a standalone minbias Condor campaign."""

import argparse
import shutil
import subprocess
from pathlib import Path

from minbias.artifact import DEFAULT_CONFIG, REPO_ROOT, utc_timestamp, write_json


MAX_PYTHIA_SEED = 900_000_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--request-memory-mb", type=int, default=2048)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def render_runner(config, manifest_path):
    setup = REPO_ROOT / "env" / "setup_pythia.sh"
    return f"""#!/usr/bin/env bash
set -euo pipefail

JOB_INDEX="${{1:?missing job index}}"
EVENTS="${{2:?missing events}}"
SEED="${{3:?missing seed}}"
OUTPUT="${{4:?missing output}}"

cd {shlex_quote(str(REPO_ROOT))}
source {shlex_quote(str(setup))}
python3 -m minbias.generate_protons \\
  --events "$EVENTS" \\
  --seed "$SEED" \\
  --config {shlex_quote(str(config))} \\
  --output "$OUTPUT"
echo "Completed job $JOB_INDEX from {manifest_path}"
"""


def shlex_quote(value):
    import shlex

    return shlex.quote(value)


def prepare(args):
    if args.events <= 0 or args.jobs <= 0 or args.events < args.jobs:
        raise RuntimeError("--events and --jobs must be positive, with at least one event per job")
    if args.request_memory_mb <= 0:
        raise RuntimeError("--request-memory-mb must be positive")
    seeds = [args.seed_base + index for index in range(args.jobs)]
    if args.seed_base < 1 or seeds[-1] > MAX_PYTHIA_SEED or len(seeds) != len(set(seeds)):
        raise RuntimeError("Requested seeds are duplicate or outside Pythia's valid range")
    campaign_dir = Path(args.campaign_dir).resolve()
    if campaign_dir.exists():
        raise RuntimeError(f"Campaign directory already exists: {campaign_dir}")
    config = Path(args.config).resolve()
    if not config.is_file():
        raise RuntimeError(f"Missing configuration: {config}")

    condor_dir = campaign_dir / "condor"
    shard_dir = campaign_dir / "shards"
    condor_dir.mkdir(parents=True)
    shard_dir.mkdir()
    base, remainder = divmod(args.events, args.jobs)
    jobs = []
    for index in range(args.jobs):
        count = base + (1 if index >= args.jobs - remainder and remainder else 0)
        jobs.append(
            {
                "index": index,
                "events": count,
                "seed": seeds[index],
                "output": str(shard_dir / f"protons_job_{index:05d}.parquet"),
            }
        )
    manifest = {
        "schema_version": 1,
        "campaign": args.campaign,
        "events": args.events,
        "jobs_requested": args.jobs,
        "seed_base": args.seed_base,
        "config": str(config),
        "created_at": utc_timestamp(),
        "jobs": jobs,
    }
    manifest_path = campaign_dir / "manifest.json"
    write_json(manifest_path, manifest)

    queue = condor_dir / "queue.txt"
    queue.write_text(
        "\n".join(
            f"{job['index']} {job['events']} {job['seed']} {job['output']}" for job in jobs
        )
        + "\n",
        encoding="utf-8",
    )
    runner = condor_dir / "run_job.sh"
    runner.write_text(render_runner(config, manifest_path), encoding="utf-8")
    runner.chmod(0o755)
    submit = condor_dir / "submit.sub"
    submit.write_text(
        f"""universe = vanilla
executable = {runner}
arguments = $(JOB_INDEX) $(EVENTS) $(SEED) $(OUTPUT)
output = {condor_dir}/job_$(JOB_INDEX).out
error = {condor_dir}/job_$(JOB_INDEX).err
log = {condor_dir}/cluster.log
request_memory = {args.request_memory_mb}
request_cpus = 1
getenv = True
queue JOB_INDEX, EVENTS, SEED, OUTPUT from {queue}
""",
        encoding="utf-8",
    )
    return campaign_dir, manifest_path, submit


def main():
    args = parse_args()
    try:
        campaign_dir, manifest, submit_file = prepare(args)
        print(f"Prepared {manifest}")
        print(f"Condor submit file: {submit_file}")
        if args.submit:
            if shutil.which("condor_submit") is None:
                raise RuntimeError("condor_submit is not available")
            result = subprocess.run(
                ["condor_submit", str(submit_file)], text=True, capture_output=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(f"condor_submit failed: {result.stderr.strip()}")
            (campaign_dir / "submission.txt").write_text(result.stdout, encoding="utf-8")
            print(result.stdout.strip())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
