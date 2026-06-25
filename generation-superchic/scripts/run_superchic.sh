#!/usr/bin/env bash

set -o pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_superchic.sh \
    [--process <process_name>] [--card <dat_file>] \
    [--nev <events>] [--seed <seed>] [--out-tag <tag>] \
    [--output-dir <dir>] [--job <job_index>] [--init]

Examples:
  ./run_superchic.sh
  ./run_superchic.sh --process Hbb --nev 1000 --seed 1001 --out-tag Hbb__test --job 1 --init
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="$STUDY_DIR/generation-superchic"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
INIT_SCRIPT="$SCRIPT_DIR/prepare_superchic_init.sh"

source "$STUDY_DIR/setup_env.sh"

PROCESS="Hbb"
CARD=""
NEVT=100
SEED=""
CAMPAIGN_TAG=""
OUTPUT_DIR=""
JOB_INDEX=""
RUN_INIT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --card|--template) CARD="$2"; shift 2 ;;
    --nev|--events) NEVT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out-tag|--tag) CAMPAIGN_TAG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --job) JOB_INDEX="$2"; shift 2 ;;
    --init) RUN_INIT=true; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$CAMPAIGN_TAG" ]] || CAMPAIGN_TAG="${PROCESS}__test"

if [[ -n "$JOB_INDEX" ]]; then
  [[ -n "$SEED" ]] || SEED=$((1000 + JOB_INDEX))
  JOB_TAG="${CAMPAIGN_TAG}_${JOB_INDEX}"
else
  [[ -n "$SEED" ]] || SEED=1001
  JOB_TAG="$CAMPAIGN_TAG"
fi

PATH_ARGS=(--process "$PROCESS" --campaign "$CAMPAIGN_TAG")
if [[ -n "$OUTPUT_DIR" ]]; then
  PATH_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
PATH_ENV="$(python3 "$PATH_HELPER" superchic-env "${PATH_ARGS[@]}")" || exit 1
eval "$PATH_ENV"

mkdir -p "$SUPERCHIC_LOGS_DIR"
LOG="$SUPERCHIC_LOGS_DIR/run_${JOB_TAG}.log"
: > "$LOG"

log_step() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

log_step "Starting SuperChic run"
log_step "Process: $PROCESS"
log_step "Campaign: $CAMPAIGN"
log_step "Job tag: $JOB_TAG"
log_step "SuperChic output dir: $SUPERCHIC_ROOT"
log_step "Log file: $LOG"

if [[ -z "$CARD" ]]; then
  CARD="$ROOT_DIR/cards/$PROCESS.DAT"
fi

if [[ ! -f "$CARD" ]]; then
  log_step "ERROR: card not found for process $PROCESS"
  log_step "Tried: $CARD"
  exit 1
fi
log_step "Using card: $CARD"

SUPERCHIC_EXE=""
for candidate in \
  "$SUPERCHIC_DIR/install/bin/superchic" \
  "$SUPERCHIC_DIR/build/bin/superchic"; do
  if [[ -x "$candidate" ]]; then
    SUPERCHIC_EXE="$candidate"
    break
  fi
done
if [[ -z "$SUPERCHIC_EXE" ]]; then
  log_step "ERROR: SuperChic executable not found under $SUPERCHIC_DIR"
  log_step "Expected $SUPERCHIC_DIR/install/bin/superchic or $SUPERCHIC_DIR/build/bin/superchic"
  exit 1
fi
log_step "Using superchic: $SUPERCHIC_EXE"

WORK_ROOT="$ROOT_DIR/workspace"
mkdir -p \
  "$WORK_ROOT" \
  "$SUPERCHIC_LOGS_DIR" \
  "$SUPERCHIC_CARDS_DIR" \
  "$SUPERCHIC_EVRECS_DIR" \
  "$SUPERCHIC_OUTPUT_DIR"

RUN_DIR="$(mktemp -d "$WORK_ROOT/run_${PROCESS}_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT
log_step "Work dir: $RUN_DIR"

CARD_LOCAL="$RUN_DIR/job.DAT"
JOB_TAG_QUOTED="'$JOB_TAG'"

# SuperChic reads several *_card.dat files via lha_read.f from ./Cards or parent dirs.
# Stage the full Cards directory in each run directory.
SUPERCHIC_CARDS_DIR_CANDIDATES=(
  "$SUPERCHIC_DIR/Cards"
  "$SUPERCHIC_DIR/install/share/doc/SuperChic/Cards"
  "$SUPERCHIC_DIR/build/share/doc/SuperChic/Cards"
)
SUPERCHIC_CARDS_DIR=""
for candidate in "${SUPERCHIC_CARDS_DIR_CANDIDATES[@]}"; do
  if [[ -d "$candidate" ]]; then
    SUPERCHIC_CARDS_DIR="$candidate"
    break
  fi
done
if [[ -z "$SUPERCHIC_CARDS_DIR" ]]; then
  log_step "ERROR: SuperChic Cards directory not found for runtime"
  log_step "Tried: \$SUPERCHIC_DIR/Cards and install/build share/doc locations"
  exit 1
fi
log_step "Staging SuperChic runtime cards from $SUPERCHIC_CARDS_DIR"
mkdir -p "$RUN_DIR/Cards"
cp -f "$SUPERCHIC_CARDS_DIR"/* "$RUN_DIR/Cards/"

log_step "Writing job card: $CARD_LOCAL"
awk -v outtg="$JOB_TAG_QUOTED" -v iseed="$SEED" -v nev="$NEVT" '
  /\[outtg\]/ { print outtg "          ! [outtg]"; next }
  /\[iseed\]/ { print iseed "           ! [iseed] : Random number seed (integer > 0)"; next }
  /\[nev\]/   { print nev "           ! [nev] : Number of events ( < 1000000 recommended)"; next }
  { print }
' "$CARD" > "$CARD_LOCAL"

log_step "Reading init parameters from job card"

INIT_INTAG=$(awk '/\[intag\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$CARD_LOCAL")
INIT_SCREEN_FILE="screening${INIT_INTAG}.dat"

INIT_INPUTS_DIR="$SUPERCHIC_ROOT/init/inputs"
PREPARE_INIT_COMMAND=("$INIT_SCRIPT" --process "$PROCESS" --out-tag "$CAMPAIGN_TAG" --card "$CARD")
if [[ -n "$OUTPUT_DIR" ]]; then
  PREPARE_INIT_COMMAND+=(--output-dir "$OUTPUT_DIR")
fi
printf -v PREPARE_INIT_DISPLAY '%q ' "${PREPARE_INIT_COMMAND[@]}"
PREPARE_INIT_DISPLAY="${PREPARE_INIT_DISPLAY% }"

if [[ "$RUN_INIT" == true ]]; then
  log_step "Preparing initialized inputs via $INIT_SCRIPT"
  if ! "${PREPARE_INIT_COMMAND[@]}" 2>&1 | tee -a "$LOG"; then
    log_step "ERROR: init preparation failed"
    exit 1
  fi
fi

if [[ ! -f "$INIT_INPUTS_DIR/$INIT_SCREEN_FILE" ]]; then
  log_step "ERROR: initialized SuperChic inputs missing: $INIT_INPUTS_DIR/$INIT_SCREEN_FILE"
  log_step "Run this before running jobs: $PREPARE_INIT_DISPLAY"
  log_step "Alternatively, rerun run_superchic.sh with --init."
  exit 1
fi

log_step "Using initialized inputs from $INIT_INPUTS_DIR"
cp -rf "$INIT_INPUTS_DIR" "$RUN_DIR/inputs"

log_step "Running superchic"
if ! (
  cd "$RUN_DIR"
  "$SUPERCHIC_EXE" < job.DAT
) 2>&1 | tee -a "$LOG"; then
  log_step "ERROR: superchic failed"
  exit 1
fi

if [[ -d "$RUN_DIR/evrecs" ]]; then
  log_step "Copying evrec files to $SUPERCHIC_EVRECS_DIR"
  cp -f "$RUN_DIR/evrecs"/* "$SUPERCHIC_EVRECS_DIR/" 2>/dev/null || true
else
  log_step "No evrecs directory produced"
fi

if [[ -d "$RUN_DIR/outputs" ]]; then
  log_step "Copying output files to $SUPERCHIC_OUTPUT_DIR"
  cp -f "$RUN_DIR/outputs"/* "$SUPERCHIC_OUTPUT_DIR/" 2>/dev/null || true
else
  log_step "No outputs directory produced"
fi

log_step "Saving resolved job card to $SUPERCHIC_CARDS_DIR/job_${JOB_TAG}.DAT"
cp -f "$CARD_LOCAL" "$SUPERCHIC_CARDS_DIR/job_${JOB_TAG}.DAT"
log_step "Finished SuperChic run"
