#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_madgraph.sh --process PROCESS --campaign CAMPAIGN [--nev EVENTS]
    [--seed SEED] [--job JOB_INDEX] [--init] [--overwrite] [--dry-run]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
METADATA_WRITER="$STUDY_DIR/common/write_metadata.py"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"
INIT_SCRIPT="$SCRIPT_DIR/prepare_madgraph_gridpack.sh"
ORIGINAL_ARGS=("$@")

PROCESS=""
CAMPAIGN=""
NEVT=1000
SEED=""
JOB_INDEX=""
RUN_INIT=false
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --nev|--events) NEVT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --job) JOB_INDEX="$2"; shift 2 ;;
    --init) RUN_INIT=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$PROCESS" ]] || { echo "ERROR: --process is required." >&2; usage; }
[[ -n "$CAMPAIGN" ]] || { echo "ERROR: --campaign is required." >&2; usage; }
[[ "$NEVT" =~ ^[0-9]+$ ]] && (( NEVT > 0 )) || {
  echo "ERROR: --nev must be a positive integer." >&2
  exit 1
}
if [[ -n "$JOB_INDEX" ]] &&
   { ! [[ "$JOB_INDEX" =~ ^[0-9]+$ ]] || (( JOB_INDEX <= 0 )); }; then
  echo "ERROR: --job must be a positive integer." >&2
  exit 1
fi
if [[ -n "$JOB_INDEX" ]]; then
  [[ -n "$SEED" ]] || SEED=$((1001 + JOB_INDEX - 1))
else
  [[ -n "$SEED" ]] || SEED=1001
fi
[[ "$SEED" =~ ^[0-9]+$ ]] && (( SEED > 0 && SEED <= 904866561 )) || {
  echo "ERROR: --seed must be between 1 and 904866561." >&2
  exit 1
}

source "$STUDY_DIR/env/setup_madgraph.sh"
eval "$(python3 "$PATH_HELPER" generation-env \
  --generator madgraph --process "$PROCESS" --campaign "$CAMPAIGN")"

JOB_TAG="$CAMPAIGN"
[[ -n "$JOB_INDEX" ]] && JOB_TAG="${CAMPAIGN}_${JOB_INDEX}"
LHE_OUTPUT="$EVENT_RECORDS_DIR/MadGraph_${PROCESS}_${JOB_TAG}.lhe"
CARD_OUTPUT="$CARDS_DIR/card_${PROCESS}_${JOB_TAG}.dat"
LOG_OUTPUT="$LOGS_DIR/run_${PROCESS}_${JOB_TAG}.log"
GRIDPACK="${HIGGS_CEP_MADGRAPH_GRIDPACK:-$GENERATION_ROOT/init/gridpack.tar.gz}"
INIT_KEY_FILE="$GENERATION_ROOT/init/init_key.txt"
INIT_RUN_CARD="$GENERATION_ROOT/init/run_card.dat"
EXPECTED_KEY="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode key \
    --madgraph-version "$MADGRAPH_VERSION"
)"
PROCESS_COMMAND="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode process
)"

if [[ "$DRY_RUN" == true ]]; then
  echo "Process: $PROCESS"
  echo "Campaign: $CAMPAIGN"
  [[ -n "$JOB_INDEX" ]] && echo "Job index: $JOB_INDEX"
  echo "Events: $NEVT"
  echo "Seed: $SEED"
  echo "MadGraph version: $MADGRAPH_VERSION"
  echo "Gridpack: $GRIDPACK"
  echo "LHE output: $LHE_OUTPUT"
  echo "Card output: $CARD_OUTPUT"
  echo "Log output: $LOG_OUTPUT"
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" \
    --process-dir PROCESS_DIR
  exit 0
fi

if [[ "$RUN_INIT" == true ]]; then
  "$INIT_SCRIPT" --process "$PROCESS" --campaign "$CAMPAIGN"
fi
if [[ ! -s "$GRIDPACK" || ! -s "$INIT_RUN_CARD" || ! -f "$INIT_KEY_FILE" ||
      "$(cat "$INIT_KEY_FILE" 2>/dev/null || true)" != "$EXPECTED_KEY" ]]; then
  echo "ERROR: matching MadGraph gridpack is not available." >&2
  echo "Pass --init or run prepare_madgraph_gridpack.sh first." >&2
  exit 1
fi

if [[ "$OVERWRITE" != true ]]; then
  for output in "$LHE_OUTPUT" "$CARD_OUTPUT" "$LOG_OUTPUT"; do
    if [[ -e "$output" ]]; then
      echo "ERROR: run output already exists: $output" >&2
      echo "Use --overwrite to replace this run." >&2
      exit 1
    fi
  done
else
  rm -f "$LHE_OUTPUT" "$CARD_OUTPUT" "$LOG_OUTPUT"
fi

mkdir -p "$EVENT_RECORDS_DIR" "$CARDS_DIR" "$LOGS_DIR"
printf -v COMMAND '%q ' "$0" "${ORIGINAL_ARGS[@]}"
METADATA_ARGS=(
  --output "$METADATA_FILE"
  --string-field "generator=madgraph"
  --string-field "process=$PROCESS"
  --string-field "campaign=$CAMPAIGN"
  --string-field "process_command=$PROCESS_COMMAND"
  --string-field "mode=run"
  --field "events=$NEVT"
  --field "seed=$SEED"
  --string-field "gridpack_key=$EXPECTED_KEY"
  --string-field "madgraph_version=$MADGRAPH_VERSION"
  --string-field "command=${COMMAND% }"
  --string-field "created_at=$(date -Iseconds)"
  --string-field "runtime_source=$LCG_VIEW"
)
CARD_METADATA="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode metadata \
    --run-card "$INIT_RUN_CARD"
)"
while IFS=$'\t' read -r FIELD_TYPE FIELD_VALUE; do
  case "$FIELD_TYPE" in
    field|string-field) METADATA_ARGS+=("--$FIELD_TYPE" "$FIELD_VALUE") ;;
    *) echo "ERROR: invalid card metadata field type: $FIELD_TYPE" >&2; exit 1 ;;
  esac
done <<< "$CARD_METADATA"
[[ -n "$JOB_INDEX" ]] && METADATA_ARGS+=(--field "job_index=$JOB_INDEX")
if [[ "${HIGGS_CEP_CONDOR_JOB:-0}" != "1" ]]; then
  python3 "$METADATA_WRITER" "${METADATA_ARGS[@]}"
fi

WORK_ROOT="${_CONDOR_SCRATCH_DIR:-$STUDY_DIR/generation-madgraph/workspace}"
mkdir -p "$WORK_ROOT"
RUN_DIR="$(mktemp -d "$WORK_ROOT/run_${PROCESS}_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT
tar -xzf "$GRIDPACK" -C "$RUN_DIR"

{
  echo "Process: $PROCESS"
  echo "Campaign: $CAMPAIGN"
  [[ -n "$JOB_INDEX" ]] && echo "Job index: $JOB_INDEX"
  echo "Events: $NEVT"
  echo "Seed: $SEED"
  echo "MadGraph version: $MADGRAPH_VERSION"
  echo "Gridpack key: $EXPECTED_KEY"
} > "$LOG_OUTPUT"

echo "Running MadGraph gridpack in $RUN_DIR" | tee -a "$LOG_OUTPUT"
if ! (
  cd "$RUN_DIR"
  ./run.sh "$NEVT" "$SEED"
) 2>&1 | tee -a "$LOG_OUTPUT"; then
  echo "ERROR: MadGraph gridpack failed; see $LOG_OUTPUT" >&2
  exit 1
fi

if [[ ! -s "$RUN_DIR/events.lhe.gz" ]]; then
  echo "ERROR: MadGraph did not produce events.lhe.gz" | tee -a "$LOG_OUTPUT" >&2
  exit 1
fi
gzip -cd "$RUN_DIR/events.lhe.gz" > "$LHE_OUTPUT"

EVENT_COUNT="$(grep -c '<event>' "$LHE_OUTPUT")"
if [[ "$EVENT_COUNT" -ne "$NEVT" ]]; then
  echo "ERROR: expected $NEVT events, found $EVENT_COUNT" | tee -a "$LOG_OUTPUT" >&2
  exit 1
fi

python3 "$CARD_GENERATOR" \
  --process "$PROCESS" --campaign "$CAMPAIGN" --mode lhe-card \
  --lhe "$LHE_OUTPUT" --output "$CARD_OUTPUT"

echo "Saved LHE output: $LHE_OUTPUT" | tee -a "$LOG_OUTPUT"
echo "Saved run card: $CARD_OUTPUT" | tee -a "$LOG_OUTPUT"
