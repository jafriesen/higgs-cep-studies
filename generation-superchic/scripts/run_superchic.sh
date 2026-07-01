#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_superchic.sh --process PROCESS --campaign CAMPAIGN [--card DAT_FILE]
    [--nev EVENTS] [--seed SEED] [--job JOB_INDEX] [--init]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
METADATA_WRITER="$STUDY_DIR/common/write_metadata.py"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"
INIT_SCRIPT="$SCRIPT_DIR/prepare_superchic_init.sh"
DEFAULT_CARD="$STUDY_DIR/generation-superchic/cards/template.DAT"
ORIGINAL_ARGS=("$@")

PROCESS=""
CAMPAIGN=""
CARD=""
NEVT=100
SEED=""
JOB_INDEX=""
RUN_INIT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --card) CARD="$2"; shift 2 ;;
    --nev|--events) NEVT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --job) JOB_INDEX="$2"; shift 2 ;;
    --init) RUN_INIT=true; shift ;;
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
  JOB_TAG="${CAMPAIGN}_${JOB_INDEX}"
else
  [[ -n "$SEED" ]] || SEED=1001
  JOB_TAG="$CAMPAIGN"
fi
[[ "$SEED" =~ ^[0-9]+$ ]] && (( SEED > 0 )) || {
  echo "ERROR: --seed must be a positive integer." >&2
  exit 1
}

CARD="${CARD:-$DEFAULT_CARD}"
[[ "$CARD" == /* ]] || CARD="$STUDY_DIR/$CARD"
[[ -f "$CARD" ]] || { echo "ERROR: card not found: $CARD" >&2; exit 1; }

source "$STUDY_DIR/env/setup_superchic.sh"
eval "$(python3 "$PATH_HELPER" generation-env \
  --generator superchic --process "$PROCESS" --campaign "$CAMPAIGN")"

SUPERCHIC_OUTPUT_DIR="$GENERATION_ROOT/output"
INIT_INPUTS_DIR="$GENERATION_ROOT/init/inputs"
LOG="$LOGS_DIR/run_${JOB_TAG}.log"

mkdir -p "$LOGS_DIR" "$CARDS_DIR" "$EVENT_RECORDS_DIR" "$SUPERCHIC_OUTPUT_DIR"
: > "$LOG"

log_step() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

log_step "Starting SuperChic run"
log_step "Process: $PROCESS"
log_step "Campaign: $CAMPAIGN"
log_step "Job tag: $JOB_TAG"
log_step "Generation dir: $GENERATION_ROOT"
log_step "Using card template: $CARD"

SUPERCHIC_EXE=""
for candidate in \
  "$SUPERCHIC_DIR/install/bin/superchic" \
  "$SUPERCHIC_DIR/build/bin/superchic"; do
  if [[ -x "$candidate" ]]; then
    SUPERCHIC_EXE="$candidate"
    break
  fi
done
[[ -n "$SUPERCHIC_EXE" ]] || {
  log_step "ERROR: SuperChic executable not found under $SUPERCHIC_DIR"
  exit 1
}

WORK_ROOT="${_CONDOR_SCRATCH_DIR:-$STUDY_DIR/generation-superchic/workspace}"
mkdir -p "$WORK_ROOT"
RUN_DIR="$(mktemp -d "$WORK_ROOT/run_${PROCESS}_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT

RUNTIME_CARDS_DIR=""
for candidate in \
  "$SUPERCHIC_DIR/Cards" \
  "$SUPERCHIC_DIR/install/share/doc/SuperChic/Cards" \
  "$SUPERCHIC_DIR/build/share/doc/SuperChic/Cards"; do
  if [[ -d "$candidate" ]]; then
    RUNTIME_CARDS_DIR="$candidate"
    break
  fi
done
[[ -n "$RUNTIME_CARDS_DIR" ]] || {
  log_step "ERROR: SuperChic runtime Cards directory not found"
  exit 1
}
mkdir -p "$RUN_DIR/Cards"
cp -f "$RUNTIME_CARDS_DIR"/* "$RUN_DIR/Cards/"

CARD_LOCAL="$RUN_DIR/job.DAT"
python3 "$CARD_GENERATOR" \
  --template "$CARD" --process "$PROCESS" --nev "$NEVT" --seed "$SEED" \
  --out-tag "$JOB_TAG" --output "$CARD_LOCAL"

INIT_INTAG=$(awk '/\[intag\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$CARD_LOCAL")
INIT_SCREEN_FILE="screening${INIT_INTAG}.dat"
INIT_RTS=$(awk '/\[rts\]/ {print $1; exit}' "$CARD_LOCAL")
INIT_ISURV=$(awk '/\[isurv\]/ {print $1; exit}' "$CARD_LOCAL")
INIT_PDFNAME=$(awk '/\[PDFname\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$CARD_LOCAL")
INIT_PDFMEMBER=$(awk '/\[PDFmember\]/ {print $1; exit}' "$CARD_LOCAL")
INIT_KEY="$(printf '%s|%s|%s|%s|%s\n' \
  "$INIT_RTS" "$INIT_ISURV" "$INIT_INTAG" "$INIT_PDFNAME" "$INIT_PDFMEMBER" |
  sha1sum | awk '{print $1}')"
INIT_ARGS=(--process "$PROCESS" --campaign "$CAMPAIGN" --card "$CARD")
if [[ "$RUN_INIT" == true ]]; then
  log_step "Preparing initialized inputs"
  "${INIT_SCRIPT}" "${INIT_ARGS[@]}" 2>&1 | tee -a "$LOG"
fi
if [[ ! -f "$INIT_INPUTS_DIR/$INIT_SCREEN_FILE" ||
      ! -f "$GENERATION_ROOT/init/init_key.txt" ||
      "$(cat "$GENERATION_ROOT/init/init_key.txt" 2>/dev/null || true)" != "$INIT_KEY" ]]; then
  log_step "ERROR: matching initialized inputs are not available"
  log_step "Run prepare_superchic_init.sh or pass --init."
  exit 1
fi
ln -s "$INIT_INPUTS_DIR" "$RUN_DIR/inputs"

printf -v COMMAND '%q ' "$0" "${ORIGINAL_ARGS[@]}"
METADATA_ARGS=(
  --output "$METADATA_FILE" \
  --string-field "generator=superchic" \
  --string-field "process=$PROCESS" \
  --string-field "campaign=$CAMPAIGN" \
  --string-field "mode=run" \
  --field "events=$NEVT" \
  --field "seed=$SEED" \
  --string-field "card=$CARD" \
  --string-field "command=${COMMAND% }" \
  --string-field "created_at=$(date -Iseconds)" \
  --string-field "runtime_source=$SUPERCHIC_DIR"
)
[[ -n "$JOB_INDEX" ]] && METADATA_ARGS+=(--field "job_index=$JOB_INDEX")
if [[ "${HIGGS_CEP_CONDOR_JOB:-0}" != "1" ]]; then
  python3 "$METADATA_WRITER" "${METADATA_ARGS[@]}"
fi

log_step "Running SuperChic"
if ! (
  cd "$RUN_DIR"
  "$SUPERCHIC_EXE" < job.DAT
) 2>&1 | tee -a "$LOG"; then
  log_step "ERROR: SuperChic failed"
  exit 1
fi

if [[ -d "$RUN_DIR/evrecs" ]]; then
  cp -f "$RUN_DIR/evrecs"/* "$EVENT_RECORDS_DIR/" 2>/dev/null || true
fi
if [[ -d "$RUN_DIR/outputs" ]]; then
  cp -f "$RUN_DIR/outputs"/* "$SUPERCHIC_OUTPUT_DIR/" 2>/dev/null || true
fi
cp -f "$CARD_LOCAL" "$CARDS_DIR/job_${JOB_TAG}.DAT"
log_step "Finished SuperChic run"
