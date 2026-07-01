#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_fpmc.sh --process PROCESS --campaign CAMPAIGN [--nev EVENTS]
    [--seed SEED] [--job JOB_INDEX] [--overwrite] [--dry-run]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
METADATA_WRITER="$STUDY_DIR/common/write_metadata.py"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"
FPMC_SETUP="$STUDY_DIR/env/setup_fpmc.sh"
ORIGINAL_ARGS=("$@")

PROCESS=""
CAMPAIGN=""
NEVT=1000
SEED=""
JOB_INDEX=""
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --nev|--events) NEVT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --job) JOB_INDEX="$2"; shift 2 ;;
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
  [[ -n "$SEED" ]] || SEED=$((33799 + JOB_INDEX - 1))
else
  [[ -n "$SEED" ]] || SEED=33799
fi
[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --seed must be a non-negative integer." >&2
  exit 1
}

eval "$(python3 "$PATH_HELPER" generation-env \
  --generator fpmc --process "$PROCESS" --campaign "$CAMPAIGN")"

JOB_TAG="${PROCESS}_${CAMPAIGN}"
[[ -n "$JOB_INDEX" ]] && JOB_TAG="${JOB_TAG}_${JOB_INDEX}"
LHE_OUTPUT="$EVENT_RECORDS_DIR/FPMC_${JOB_TAG}.lhe"
CARD_OUTPUT="$CARDS_DIR/card_${JOB_TAG}.dat"
LOG_OUTPUT="$LOGS_DIR/run_${JOB_TAG}.log"

if [[ "$DRY_RUN" == true ]]; then
  echo "Process: $PROCESS"
  echo "Campaign: $CAMPAIGN"
  [[ -n "$JOB_INDEX" ]] && echo "Job index: $JOB_INDEX"
  echo "Seed: $SEED"
  echo "LHE output: $LHE_OUTPUT"
  echo "Card output: $CARD_OUTPUT"
  echo "Log output: $LOG_OUTPUT"
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --nev "$NEVT" --seed "$SEED"
  exit 0
fi

if [[ "$OVERWRITE" != true ]]; then
  for output in "$LHE_OUTPUT" "$CARD_OUTPUT" "$LOG_OUTPUT"; do
    if [[ -e "$output" ]]; then
      echo "ERROR: run output already exists: $output" >&2
      echo "Use --overwrite to replace this run." >&2
      exit 1
    fi
  done
fi

mkdir -p "$EVENT_RECORDS_DIR" "$CARDS_DIR" "$LOGS_DIR"
python3 "$CARD_GENERATOR" \
  --process "$PROCESS" --campaign "$CAMPAIGN" --nev "$NEVT" --seed "$SEED" \
  --output "$CARD_OUTPUT"

printf -v COMMAND '%q ' "$0" "${ORIGINAL_ARGS[@]}"
METADATA_ARGS=(
  --output "$METADATA_FILE" \
  --string-field "generator=fpmc" \
  --string-field "process=$PROCESS" \
  --string-field "campaign=$CAMPAIGN" \
  --string-field "mode=run" \
  --field "events=$NEVT" \
  --field "seed=$SEED" \
  --string-field "command=${COMMAND% }" \
  --string-field "created_at=$(date -Iseconds)" \
  --string-field "runtime_source=${HIGGS_CEP_FPMC_DIR:-$STUDY_DIR/../fpmc}"
)
[[ -n "$JOB_INDEX" ]] && METADATA_ARGS+=(--field "job_index=$JOB_INDEX")
if [[ "${HIGGS_CEP_CONDOR_JOB:-0}" != "1" ]]; then
  python3 "$METADATA_WRITER" "${METADATA_ARGS[@]}"
fi

FPMC_DIR_VALUE="${HIGGS_CEP_FPMC_DIR:-}"
FPMC_BUILD_DIR_VALUE="${HIGGS_CEP_FPMC_BUILD_DIR:-}"
FPMC_EXE_VALUE="${HIGGS_CEP_FPMC_EXE:-}"
FPMC_GCC_SETUP_VALUE="${HIGGS_CEP_FPMC_GCC_SETUP:-}"

FPMC_DIR_DEFAULT="$(cd "$STUDY_DIR/.." && pwd)/fpmc"
RUN_BUILD_DIR="${FPMC_BUILD_DIR_VALUE:-${FPMC_DIR_VALUE:-$FPMC_DIR_DEFAULT}/build}"
if [[ ! -d "$RUN_BUILD_DIR" ]]; then
  echo "ERROR: FPMC build directory not found: $RUN_BUILD_DIR" >&2
  exit 1
fi

RUN_ROOT="${_CONDOR_SCRATCH_DIR:-$RUN_BUILD_DIR}"
mkdir -p "$RUN_ROOT"
RUN_DIR="$(mktemp -d "$RUN_ROOT/higgs_cep_fpmc_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT
if [[ -d "$RUN_BUILD_DIR/External" ]]; then
  ln -s "$RUN_BUILD_DIR/External" "$RUN_DIR/External"
fi

{
  echo "Process: $PROCESS"
  echo "Campaign: $CAMPAIGN"
  [[ -n "$JOB_INDEX" ]] && echo "Job index: $JOB_INDEX"
  echo "Events: $NEVT"
  echo "Seed: $SEED"
  echo "Card: $CARD_OUTPUT"
  echo "LHE output: $LHE_OUTPUT"
} > "$LOG_OUTPUT"

CLEAN_ENV=(
  env -i
  "HOME=${HOME:-/tmp}"
  "USER=${USER:-unknown}"
  "PATH=/usr/bin:/bin"
)
[[ -n "${TMPDIR:-}" ]] && CLEAN_ENV+=("TMPDIR=$TMPDIR")
[[ -n "$FPMC_DIR_VALUE" ]] && CLEAN_ENV+=("HIGGS_CEP_FPMC_DIR=$FPMC_DIR_VALUE")
[[ -n "$FPMC_BUILD_DIR_VALUE" ]] &&
  CLEAN_ENV+=("HIGGS_CEP_FPMC_BUILD_DIR=$FPMC_BUILD_DIR_VALUE")
[[ -n "$FPMC_EXE_VALUE" ]] && CLEAN_ENV+=("HIGGS_CEP_FPMC_EXE=$FPMC_EXE_VALUE")
[[ -n "$FPMC_GCC_SETUP_VALUE" ]] &&
  CLEAN_ENV+=("HIGGS_CEP_FPMC_GCC_SETUP=$FPMC_GCC_SETUP_VALUE")

echo "Running FPMC in $RUN_DIR" | tee -a "$LOG_OUTPUT"
if ! "${CLEAN_ENV[@]}" bash --noprofile --norc -c '
  set -euo pipefail
  source "$1"
  cd "$2"
  "$FPMC_EXE" < "$3"
' _ "$FPMC_SETUP" "$RUN_DIR" "$CARD_OUTPUT" 2>&1 | tee -a "$LOG_OUTPUT"; then
  echo "ERROR: FPMC failed; see $LOG_OUTPUT" >&2
  exit 1
fi

if [[ ! -s "$RUN_DIR/FPMC.lhe" ]]; then
  echo "ERROR: FPMC did not produce a nonempty FPMC.lhe" | tee -a "$LOG_OUTPUT" >&2
  exit 1
fi

cp -f "$RUN_DIR/FPMC.lhe" "$LHE_OUTPUT"
echo "Saved LHE output: $LHE_OUTPUT" | tee -a "$LOG_OUTPUT"
