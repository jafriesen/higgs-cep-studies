#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_superchic_signal_condor.sh --jobs N [--process h_bb|qcd_bb] [--nev EVENTS] \
    [--card DAT_FILE] [--seed-base BASE] [--tag-prefix PREFIX] [--campaign NAME] \
    [--output-base DIR] [--request-memory MB] [--dry-run]

Examples:
  ./submit_superchic_signal_condor.sh --process h_bb --nev 10000 --jobs 20
  ./submit_superchic_signal_condor.sh --process qcd_bb --nev 20000 --jobs 50 --seed-base 5000
USAGE
  exit 1
}

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$RUN_SCRIPT_DIR/../.." && pwd)"

PROCESS="h_bb"
NEVT=10000
TOTAL_JOBS=""
CARD=""
SEED_BASE=""
TAG_PREFIX=""
CAMPAIGN=""
OUTPUT_BASE="$STUDY_DIR/signal-generation/output"
REQUEST_MEMORY=2048
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process)
      PROCESS="$2"
      shift 2
      ;;
    --nev|--events)
      NEVT="$2"
      shift 2
      ;;
    --jobs)
      TOTAL_JOBS="$2"
      shift 2
      ;;
    --card|--template)
      CARD="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
      ;;
    --tag-prefix)
      TAG_PREFIX="$2"
      shift 2
      ;;
    --campaign)
      CAMPAIGN="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --request-memory)
      REQUEST_MEMORY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "$TOTAL_JOBS" ]]; then
  usage
fi

case "$PROCESS" in
  h_bb|qcd_bb)
    ;;
  *)
    echo "ERROR: --process must be h_bb or qcd_bb (got: $PROCESS)" >&2
    exit 1
    ;;
esac

if (( TOTAL_JOBS <= 0 )); then
  echo "ERROR: --jobs must be > 0" >&2
  exit 1
fi
if (( NEVT <= 0 )); then
  echo "ERROR: --nev must be > 0" >&2
  exit 1
fi
if [[ -n "$SEED_BASE" ]] && (( SEED_BASE < 0 )); then
  echo "ERROR: --seed-base must be >= 0" >&2
  exit 1
fi
if (( REQUEST_MEMORY <= 0 )); then
  echo "ERROR: --request-memory must be > 0" >&2
  exit 1
fi

if [[ "$OUTPUT_BASE" != /* ]]; then
  OUTPUT_BASE="$STUDY_DIR/$OUTPUT_BASE"
fi

if [[ -n "$CARD" && "$CARD" != /* ]]; then
  CARD="$STUDY_DIR/$CARD"
fi

if [[ -z "$CAMPAIGN" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  CAMPAIGN="superchic_${PROCESS}_nev${NEVT}_j${TOTAL_JOBS}_${TS}"
fi

if [[ -z "$TAG_PREFIX" ]]; then
  TAG_PREFIX="$CAMPAIGN"
fi

CAMPAIGN_DIR="$OUTPUT_BASE/$PROCESS/$CAMPAIGN"
CONDOR_DIR="$CAMPAIGN_DIR/condor"
mkdir -p "$CONDOR_DIR"

BATCH_SCRIPT="$RUN_SCRIPT_DIR/run_superchic_signal.sh"
if [[ ! -x "$BATCH_SCRIPT" ]]; then
  chmod +x "$BATCH_SCRIPT"
fi

SUB_FILE="$CONDOR_DIR/submit.sub"
META_FILE="$CONDOR_DIR/campaign_args.txt"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"

CARD_ARG=""
if [[ -n "$CARD" ]]; then
  CARD_ARG="--card $CARD"
fi

cat > "$META_FILE" <<EOF
campaign=$CAMPAIGN
process=$PROCESS
nev=$NEVT
jobs=$TOTAL_JOBS
card=${CARD:-<auto>}
seed_base=${SEED_BASE:-<runner-default>}
tag_prefix=$TAG_PREFIX
output_base=$OUTPUT_BASE
request_memory=$REQUEST_MEMORY
EOF

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $BATCH_SCRIPT
output = $CONDOR_DIR/job_\$(JOB_INDEX).out
error = $CONDOR_DIR/job_\$(JOB_INDEX).err
log = $CONDOR_DIR/cluster.log
stream_output = True
stream_error = True
request_memory = $REQUEST_MEMORY
request_cpus = 1
getenv = True
EOF

: > "$QUEUE_FILE"

if [[ -n "$SEED_BASE" ]]; then
  for (( j = 1; j <= TOTAL_JOBS; j++ )); do
    seed=$(( SEED_BASE + j - 1 ))
    echo "$j $seed" >> "$QUEUE_FILE"
  done

  cat >> "$SUB_FILE" <<EOF
arguments = --process $PROCESS --nev $NEVT $CARD_ARG --seed \$(SEED) --out-tag ${TAG_PREFIX}_\$(JOB_INDEX) --output-base $OUTPUT_BASE \$(JOB_INDEX)
queue JOB_INDEX, SEED from $QUEUE_FILE
EOF
else
  for (( j = 1; j <= TOTAL_JOBS; j++ )); do
    echo "$j" >> "$QUEUE_FILE"
  done

  cat >> "$SUB_FILE" <<EOF
arguments = --process $PROCESS --nev $NEVT $CARD_ARG --out-tag ${TAG_PREFIX}_\$(JOB_INDEX) --output-base $OUTPUT_BASE \$(JOB_INDEX)
queue JOB_INDEX from $QUEUE_FILE
EOF
fi

echo "Campaign directory: $CAMPAIGN_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Process: $PROCESS"
echo "Events/job: $NEVT"
echo "Jobs: $TOTAL_JOBS"
echo "Tag prefix: $TAG_PREFIX"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
fi

condor_submit "$SUB_FILE"
