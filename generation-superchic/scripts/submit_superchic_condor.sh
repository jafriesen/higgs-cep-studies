#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_superchic_condor.sh [--jobs N] [--process PROCESS] [--nev EVENTS] \
    [--card DAT_FILE] [--seed-base BASE] [--campaign NAME] \
    [--output-base DIR] [--request-memory MB] [--dry-run]

Examples:
  ./submit_superchic_condor.sh --process h_bb --nev 10000 --jobs 20
  ./submit_superchic_condor.sh --process qcd_bb --nev 20000 --jobs 50 --seed-base 5000
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="$STUDY_DIR/generation-superchic"

PROCESS="h_bb"
NEVT=200000
JOBS=200
CARD=""
SEED_BASE=""
CAMPAIGN=""
OUTPUT_BASE=""
REQUEST_MEMORY=2048
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --nev|--events) NEVT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --card|--template) CARD="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --request-memory) REQUEST_MEMORY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$JOBS" ]] || usage

if (( JOBS <= 0 || NEVT <= 0 || REQUEST_MEMORY <= 0 )); then
  echo "ERROR: --jobs, --nev, and --request-memory must be > 0" >&2
  exit 1
fi
if [[ -n "$SEED_BASE" ]] && (( SEED_BASE < 0 )); then
  echo "ERROR: --seed-base must be >= 0" >&2
  exit 1
fi

[[ -n "$OUTPUT_BASE" ]] || OUTPUT_BASE="$ROOT_DIR/output"
[[ "$OUTPUT_BASE" == /* ]] || OUTPUT_BASE="$STUDY_DIR/$OUTPUT_BASE"
[[ -z "$CARD" || "$CARD" == /* ]] || CARD="$STUDY_DIR/$CARD"

if [[ -z "$CAMPAIGN" ]]; then
  CAMPAIGN="superchic_${PROCESS}_nev${NEVT}_j${JOBS}_$(date +%Y%m%d_%H%M%S)"
fi

RUN_OUTPUT_BASE="$OUTPUT_BASE/$PROCESS"
CAMPAIGN_DIR="$RUN_OUTPUT_BASE/$CAMPAIGN"
CONDOR_DIR="$CAMPAIGN_DIR/condor"
mkdir -p "$CONDOR_DIR"

RUN_SCRIPT="$SCRIPT_DIR/run_superchic.sh"
[[ -x "$RUN_SCRIPT" ]] || chmod +x "$RUN_SCRIPT"

SUB_FILE="$CONDOR_DIR/submit.sub"
META_FILE="$CONDOR_DIR/campaign_args.txt"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
CARD_ARG=""
[[ -z "$CARD" ]] || CARD_ARG="--card $CARD"
SEED_ARG=""
QUEUE_VARS="JOB_INDEX"

cat > "$META_FILE" <<EOF
campaign=$CAMPAIGN
process=$PROCESS
nev=$NEVT
jobs=$JOBS
card=${CARD:-<auto>}
seed_base=${SEED_BASE:-<runner-default>}
output_base=$OUTPUT_BASE
run_output_base=$RUN_OUTPUT_BASE
request_memory=$REQUEST_MEMORY
EOF

if [[ -n "$SEED_BASE" ]]; then
  SEED_ARG="--seed \$(SEED)"
  QUEUE_VARS="JOB_INDEX, SEED"
  : > "$QUEUE_FILE"
  for (( j = 1; j <= JOBS; j++ )); do
    echo "$j $((SEED_BASE + j - 1))" >> "$QUEUE_FILE"
  done
else
  for (( j = 1; j <= JOBS; j++ )); do
    echo "$j"
  done > "$QUEUE_FILE"
fi

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $RUN_SCRIPT
arguments = --process $PROCESS --nev $NEVT $CARD_ARG $SEED_ARG --out-tag $CAMPAIGN --output-base $RUN_OUTPUT_BASE --job \$(JOB_INDEX)
output = $CONDOR_DIR/job_\$(JOB_INDEX).out
error = $CONDOR_DIR/job_\$(JOB_INDEX).err
log = $CONDOR_DIR/cluster.log
stream_output = True
stream_error = True
request_memory = $REQUEST_MEMORY
request_cpus = 1
getenv = True
queue $QUEUE_VARS from $QUEUE_FILE
EOF

echo "Campaign directory: $CAMPAIGN_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Process: $PROCESS"
echo "Events/job: $NEVT"
echo "Jobs: $JOBS"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

command -v condor_submit >/dev/null 2>&1 || {
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
}
condor_submit "$SUB_FILE"
