#!/usr/bin/env bash

set -o pipefail

usage() {
  cat <<'USAGE'
Usage:
  prepare_superchic_init.sh \
    [--process <process_name>] [--card <dat_file>] \
    [--out-tag <tag>] [--output-dir <dir>]

Examples:
  ./prepare_superchic_init.sh
  ./prepare_superchic_init.sh --process Hbb --out-tag Hbb__test
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="$STUDY_DIR/generation-superchic"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"

source "$STUDY_DIR/setup_env.sh"

PROCESS="Hbb"
CARD=""
CAMPAIGN_TAG=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --card|--template) CARD="$2"; shift 2 ;;
    --out-tag|--tag) CAMPAIGN_TAG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$CAMPAIGN_TAG" ]] || CAMPAIGN_TAG="${PROCESS}__test"

PATH_ARGS=(--process "$PROCESS" --campaign "$CAMPAIGN_TAG")
if [[ -n "$OUTPUT_DIR" ]]; then
  PATH_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
PATH_ENV="$(python3 "$PATH_HELPER" superchic-env "${PATH_ARGS[@]}")" || exit 1
eval "$PATH_ENV"

INIT_ROOT="$SUPERCHIC_ROOT/init"
INIT_INPUTS_DIR="$INIT_ROOT/inputs"
INIT_CARD="$INIT_ROOT/init.DAT"
LOG="$SUPERCHIC_LOGS_DIR/init_${CAMPAIGN}.log"

mkdir -p "$INIT_ROOT" "$SUPERCHIC_LOGS_DIR" "$SUPERCHIC_CARDS_DIR"
: > "$LOG"

log_step() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

log_step "Starting SuperChic init"
log_step "Process: $PROCESS"
log_step "Campaign: $CAMPAIGN"
log_step "SuperChic dir: $SUPERCHIC_ROOT"
log_step "Init inputs dir: $INIT_INPUTS_DIR"
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

SUPERCHIC_INIT=""
for candidate in \
  "$SUPERCHIC_DIR/install/bin/init" \
  "$SUPERCHIC_DIR/build/bin/init"; do
  if [[ -x "$candidate" ]]; then
    SUPERCHIC_INIT="$candidate"
    break
  fi
done
if [[ -z "$SUPERCHIC_INIT" ]]; then
  log_step "ERROR: SuperChic init executable not found under $SUPERCHIC_DIR"
  log_step "Expected $SUPERCHIC_DIR/install/bin/init or $SUPERCHIC_DIR/build/bin/init"
  exit 1
fi
log_step "Using init: $SUPERCHIC_INIT"

WORK_ROOT="$ROOT_DIR/workspace"
mkdir -p "$WORK_ROOT"
RUN_DIR="$(mktemp -d "$WORK_ROOT/init_${PROCESS}_${CAMPAIGN}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT
log_step "Work dir: $RUN_DIR"

SUPERCHIC_CARDS_DIR_CANDIDATES=(
  "$SUPERCHIC_DIR/Cards"
  "$SUPERCHIC_DIR/install/share/doc/SuperChic/Cards"
  "$SUPERCHIC_DIR/build/share/doc/SuperChic/Cards"
)
SUPERCHIC_RUNTIME_CARDS_DIR=""
for candidate in "${SUPERCHIC_CARDS_DIR_CANDIDATES[@]}"; do
  if [[ -d "$candidate" ]]; then
    SUPERCHIC_RUNTIME_CARDS_DIR="$candidate"
    break
  fi
done
if [[ -z "$SUPERCHIC_RUNTIME_CARDS_DIR" ]]; then
  log_step "ERROR: SuperChic Cards directory not found for runtime"
  log_step "Tried: \$SUPERCHIC_DIR/Cards and install/build share/doc locations"
  exit 1
fi

log_step "Staging SuperChic runtime cards from $SUPERCHIC_RUNTIME_CARDS_DIR"
mkdir -p "$RUN_DIR/Cards" "$RUN_DIR/inputs"
cp -f "$SUPERCHIC_RUNTIME_CARDS_DIR"/* "$RUN_DIR/Cards/"

log_step "Writing init card: $INIT_CARD"
awk -v outtg="'${CAMPAIGN}_init'" -v iseed="1" -v nev="1" '
  /\[outtg\]/ { print outtg "          ! [outtg]"; next }
  /\[iseed\]/ { print iseed "           ! [iseed] : Random number seed (integer > 0)"; next }
  /\[nev\]/   { print nev "           ! [nev] : Number of events ( < 1000000 recommended)"; next }
  { print }
' "$CARD" > "$INIT_CARD"
cp -f "$INIT_CARD" "$RUN_DIR/job.DAT"

INIT_RTS=$(awk '/\[rts\]/ {print $1; exit}' "$INIT_CARD")
INIT_ISURV=$(awk '/\[isurv\]/ {print $1; exit}' "$INIT_CARD")
INIT_INTAG=$(awk '/\[intag\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$INIT_CARD")
INIT_PDFNAME=$(awk '/\[PDFname\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$INIT_CARD")
INIT_PDFMEMBER=$(awk '/\[PDFmember\]/ {print $1; exit}' "$INIT_CARD")
INIT_SCREEN_FILE="screening${INIT_INTAG}.dat"
INIT_KEY="$(printf '%s|%s|%s|%s|%s\n' "$INIT_RTS" "$INIT_ISURV" "$INIT_INTAG" "$INIT_PDFNAME" "$INIT_PDFMEMBER" | sha1sum | awk '{print $1}')"

log_step "Init key: $INIT_KEY"
log_step "Expected screen file: $INIT_SCREEN_FILE"

if [[ -f "$INIT_INPUTS_DIR/$INIT_SCREEN_FILE" ]]; then
  log_step "Initialized inputs already exist; skipping init"
  log_step "Finished SuperChic init"
  exit 0
fi

log_step "Running init"
if ! (
  cd "$RUN_DIR"
  "$SUPERCHIC_INIT" < job.DAT
) 2>&1 | tee -a "$LOG"; then
  log_step "ERROR: init failed"
  exit 1
fi

if [[ ! -f "$RUN_DIR/inputs/$INIT_SCREEN_FILE" ]]; then
  log_step "ERROR: init did not produce expected file inputs/$INIT_SCREEN_FILE in $RUN_DIR"
  exit 1
fi

log_step "Installing initialized inputs in $INIT_INPUTS_DIR"
rm -rf "$INIT_ROOT/inputs.tmp"
cp -rf "$RUN_DIR/inputs" "$INIT_ROOT/inputs.tmp"
rm -rf "$INIT_INPUTS_DIR"
mv "$INIT_ROOT/inputs.tmp" "$INIT_INPUTS_DIR"
printf '%s\n' "$INIT_KEY" > "$INIT_ROOT/init_key.txt"

log_step "Finished SuperChic init"
