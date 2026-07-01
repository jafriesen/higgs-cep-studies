#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Write and optionally submit generic HTCondor artifacts."
    )
    parser.add_argument("--condor-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--payload", type=Path, default=None)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--request-memory", type=int, default=2048)
    parser.add_argument("--request-cpus", type=int, default=1)
    parser.add_argument("--max-idle", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.jobs <= 0:
        parser.error("--jobs must be a positive integer")
    if not args.executable.is_file():
        parser.error(f"executable not found: {args.executable}")
    if args.payload is not None and not args.payload.is_file():
        parser.error(f"payload not found: {args.payload}")

    args.condor_dir.mkdir(parents=True, exist_ok=True)
    queue_file = args.condor_dir / "queue_items.txt"
    submit_file = args.condor_dir / "submit.sub"
    queue_file.write_text(
        "".join(f"{job}\n" for job in range(1, args.jobs + 1)),
        encoding="utf-8",
    )
    lines = [
        "universe = vanilla",
        f"executable = {args.executable}",
        "arguments = $(JOB_INDEX)",
    ]
    if args.payload is not None:
        lines.append(f"transfer_input_files = {args.payload}")
    lines.extend(
        [
            "should_transfer_files = YES",
            "when_to_transfer_output = ON_EXIT",
            'transfer_output_files = ""',
            f"output = {args.condor_dir}/job_$(JOB_INDEX).out",
            f"error = {args.condor_dir}/job_$(JOB_INDEX).err",
            f"log = {args.condor_dir}/cluster.log",
            "stream_output = False",
            "stream_error = False",
            f"max_idle = {args.max_idle}",
            f"request_memory = {args.request_memory}",
            f"request_cpus = {args.request_cpus}",
            "getenv = True",
            f"queue JOB_INDEX from {queue_file}",
            "",
        ]
    )
    submit_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"Condor submit file: {submit_file}")
    if args.dry_run:
        print("Dry run requested, not submitting.")
        return
    if shutil.which("condor_submit") is None:
        parser.error("condor_submit not found in PATH")
    subprocess.run(["condor_submit", str(submit_file)], check=True)


if __name__ == "__main__":
    main()
