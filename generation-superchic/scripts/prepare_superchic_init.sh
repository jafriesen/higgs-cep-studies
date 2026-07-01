#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  prepare_superchic_init.sh --process PROCESS --campaign CAMPAIGN
    [--card DAT_FILE]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"
DEFAULT_CARD="$STUDY_DIR/generation-superchic/cards/template.DAT"

PROCESS=""
CAMPAIGN=""
CARD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --card) CARD="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$PROCESS" ]] || { echo "ERROR: --process is required." >&2; usage; }
[[ -n "$CAMPAIGN" ]] || { echo "ERROR: --campaign is required." >&2; usage; }
CARD="${CARD:-$DEFAULT_CARD}"
[[ "$CARD" == /* ]] || CARD="$STUDY_DIR/$CARD"
[[ -f "$CARD" ]] || { echo "ERROR: card not found: $CARD" >&2; exit 1; }

source "$STUDY_DIR/env/setup_superchic.sh"
eval "$(python3 "$PATH_HELPER" generation-env \
  --generator superchic --process "$PROCESS" --campaign "$CAMPAIGN")"

INIT_ROOT="$GENERATION_ROOT/init"
INIT_INPUTS_DIR="$INIT_ROOT/inputs"
INIT_CARD="$INIT_ROOT/init.DAT"
LOG="$LOGS_DIR/init_${CAMPAIGN}.log"

mkdir -p "$INIT_ROOT" "$LOGS_DIR" "$CARDS_DIR"
: > "$LOG"

log_step() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

log_step "Starting SuperChic init"
log_step "Process: $PROCESS"
log_step "Campaign: $CAMPAIGN"
log_step "Generation dir: $GENERATION_ROOT"
log_step "Using card template: $CARD"

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
  exit 1
fi

WORK_ROOT="$STUDY_DIR/generation-superchic/workspace"
mkdir -p "$WORK_ROOT"
RUN_DIR="$(mktemp -d "$WORK_ROOT/init_${PROCESS}_${CAMPAIGN}_XXXXXX")"
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
if [[ -z "$RUNTIME_CARDS_DIR" ]]; then
  log_step "ERROR: SuperChic runtime Cards directory not found"
  exit 1
fi

mkdir -p "$RUN_DIR/Cards" "$RUN_DIR/inputs"
cp -f "$RUNTIME_CARDS_DIR"/* "$RUN_DIR/Cards/"
python3 "$CARD_GENERATOR" \
  --template "$CARD" --process "$PROCESS" --nev 1 --seed 1 \
  --out-tag "${CAMPAIGN}_init" --output "$INIT_CARD"
cp -f "$INIT_CARD" "$RUN_DIR/job.DAT"

INIT_RTS=$(awk '/\[rts\]/ {print $1; exit}' "$INIT_CARD")
INIT_ISURV=$(awk '/\[isurv\]/ {print $1; exit}' "$INIT_CARD")
INIT_INTAG=$(awk '/\[intag\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$INIT_CARD")
INIT_PDFNAME=$(awk '/\[PDFname\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$INIT_CARD")
INIT_PDFMEMBER=$(awk '/\[PDFmember\]/ {print $1; exit}' "$INIT_CARD")
INIT_SCREEN_FILE="screening${INIT_INTAG}.dat"
INIT_KEY="$(printf '%s|%s|%s|%s|%s\n' \
  "$INIT_RTS" "$INIT_ISURV" "$INIT_INTAG" "$INIT_PDFNAME" "$INIT_PDFMEMBER" |
  sha1sum | awk '{print $1}')"

if [[ -f "$INIT_INPUTS_DIR/$INIT_SCREEN_FILE" &&
      -f "$INIT_ROOT/init_key.txt" &&
      "$(cat "$INIT_ROOT/init_key.txt")" == "$INIT_KEY" ]]; then
  log_step "Matching initialized inputs already exist; skipping init"
  exit 0
fi

log_step "Running init for key $INIT_KEY"
if ! (
  cd "$RUN_DIR"
  "$SUPERCHIC_INIT" < job.DAT
) 2>&1 | tee -a "$LOG"; then
  log_step "ERROR: init failed"
  exit 1
fi

if [[ ! -f "$RUN_DIR/inputs/$INIT_SCREEN_FILE" ]]; then
  log_step "ERROR: init did not produce inputs/$INIT_SCREEN_FILE"
  exit 1
fi

rm -rf "$INIT_ROOT/inputs.tmp"
cp -rf "$RUN_DIR/inputs" "$INIT_ROOT/inputs.tmp"
rm -rf "$INIT_INPUTS_DIR"
mv "$INIT_ROOT/inputs.tmp" "$INIT_INPUTS_DIR"
printf '%s\n' "$INIT_KEY" > "$INIT_ROOT/init_key.txt"
log_step "Finished SuperChic init"
