#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  prepare_madgraph_gridpack.sh --process PROCESS --campaign CAMPAIGN
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"

PROCESS=""
CAMPAIGN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$PROCESS" ]] || { echo "ERROR: --process is required." >&2; usage; }
[[ -n "$CAMPAIGN" ]] || { echo "ERROR: --campaign is required." >&2; usage; }

source "$STUDY_DIR/env/setup_madgraph.sh"
eval "$(python3 "$PATH_HELPER" generation-env \
  --generator madgraph --process "$PROCESS" --campaign "$CAMPAIGN")"

INIT_DIR="$GENERATION_ROOT/init"
GRIDPACK="$INIT_DIR/gridpack.tar.gz"
INIT_KEY_FILE="$INIT_DIR/init_key.txt"
LOG="$LOGS_DIR/init_${CAMPAIGN}.log"
EXPECTED_KEY="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode key \
    --madgraph-version "$MADGRAPH_VERSION"
)"

mkdir -p "$INIT_DIR" "$LOGS_DIR"
: > "$LOG"

log_step() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

if [[ -s "$GRIDPACK" && -f "$INIT_KEY_FILE" &&
      "$(cat "$INIT_KEY_FILE")" == "$EXPECTED_KEY" ]]; then
  log_step "Matching MadGraph gridpack already exists; skipping initialization"
  exit 0
fi

WORK_ROOT="$STUDY_DIR/generation-madgraph/workspace"
mkdir -p "$WORK_ROOT"
RUN_DIR="$(mktemp -d "$WORK_ROOT/init_${PROCESS}_${CAMPAIGN}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT
PROCESS_DIR="$RUN_DIR/process"
INIT_CARD="$RUN_DIR/init_card.dat"

python3 "$CARD_GENERATOR" \
  --process "$PROCESS" --campaign "$CAMPAIGN" \
  --process-dir "$PROCESS_DIR" --output "$INIT_CARD"

log_step "Starting MadGraph gridpack initialization"
log_step "Process: $PROCESS"
log_step "Campaign: $CAMPAIGN"
log_step "MadGraph version: $MADGRAPH_VERSION"
log_step "Initialization key: $EXPECTED_KEY"

if ! (
  cd "$RUN_DIR"
  "$MADGRAPH_EXE" "$INIT_CARD"
) 2>&1 | tee -a "$LOG"; then
  log_step "ERROR: MadGraph gridpack initialization failed"
  exit 1
fi

GENERATED_GRIDPACK="$PROCESS_DIR/run_01_gridpack.tar.gz"
for required in \
  "$GENERATED_GRIDPACK" \
  "$PROCESS_DIR/Cards/proc_card_mg5.dat" \
  "$PROCESS_DIR/Cards/run_card.dat"; do
  if [[ ! -s "$required" ]]; then
    log_step "ERROR: expected MadGraph output not found: $required"
    exit 1
  fi
done

cp -f "$GENERATED_GRIDPACK" "$INIT_DIR/gridpack.tar.gz.tmp"
cp -f "$INIT_CARD" "$INIT_DIR/init_card.dat.tmp"
cp -f "$PROCESS_DIR/Cards/proc_card_mg5.dat" "$INIT_DIR/proc_card.dat.tmp"
cp -f "$PROCESS_DIR/Cards/run_card.dat" "$INIT_DIR/run_card.dat.tmp"
mv -f "$INIT_DIR/gridpack.tar.gz.tmp" "$GRIDPACK"
mv -f "$INIT_DIR/init_card.dat.tmp" "$INIT_DIR/init_card.dat"
mv -f "$INIT_DIR/proc_card.dat.tmp" "$INIT_DIR/proc_card.dat"
mv -f "$INIT_DIR/run_card.dat.tmp" "$INIT_DIR/run_card.dat"
printf '%s\n' "$EXPECTED_KEY" > "$INIT_KEY_FILE"

log_step "Saved gridpack: $GRIDPACK"
log_step "Finished MadGraph gridpack initialization"
