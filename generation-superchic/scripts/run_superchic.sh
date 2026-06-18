#!/usr/bin/env bash

usage() {
  cat <<'USAGE'
Usage:
  run_superchic.sh \
    [--process <process_name>] [--card <dat_file>] \
    [--nev <events>] [--seed <seed>] [--out-tag <tag>] \
    [--output-base <dir>] [--job <job_index>]

Examples:
  ./run_superchic.sh
  ./run_superchic.sh --process h_cc --nev 10000 --seed 1001 --out-tag hbb_001 --job 1
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="$STUDY_DIR/generation-superchic"

source "$STUDY_DIR/setup_env.sh"

PROCESS="h_bb"
CARD=""
NEVT=100
SEED=""
CAMPAIGN_TAG=""
OUTPUT_BASE=""
JOB_INDEX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --card|--template) CARD="$2"; shift 2 ;;
    --nev|--events) NEVT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out-tag|--tag) CAMPAIGN_TAG="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --job) JOB_INDEX="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$OUTPUT_BASE" ]] || OUTPUT_BASE="$ROOT_DIR/output/$PROCESS"
[[ -n "$CAMPAIGN_TAG" ]] || CAMPAIGN_TAG="$PROCESS"

if [[ -n "$JOB_INDEX" ]]; then
  [[ -n "$SEED" ]] || SEED=$((1000 + JOB_INDEX))
  JOB_TAG="${CAMPAIGN_TAG}_${JOB_INDEX}"
else
  [[ -n "$SEED" ]] || SEED=1001
  JOB_TAG="$CAMPAIGN_TAG"
fi

if [[ -z "$CARD" ]]; then
  CARD_CANDIDATES=(
    "$ROOT_DIR/cards/$PROCESS.DAT"
    "$SUPERCHIC_DIR/bin/$PROCESS/$PROCESS.DAT"
    "$SUPERCHIC_DIR/Cards/$PROCESS.DAT"
  )
  for candidate in "${CARD_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
      CARD="$candidate"
      break
    fi
  done
fi

if [[ -n "$CARD" && "$CARD" != /* ]]; then
  CARD="$STUDY_DIR/$CARD"
fi

if [[ ! -f "$CARD" ]]; then
  echo "ERROR: card not found for process $PROCESS." >&2
  echo "Tried: ${CARD_CANDIDATES[*]}" >&2
  exit 1
fi

if ! command -v superchic >/dev/null 2>&1; then
  echo "ERROR: superchic is not in PATH. Check SuperChic build and setup_env.sh." >&2
  exit 1
fi
if ! command -v init >/dev/null 2>&1; then
  echo "ERROR: init is not in PATH. Check SuperChic build and setup_env.sh." >&2
  exit 1
fi

WORK_ROOT="$ROOT_DIR/workspace"
OUT_ROOT="$OUTPUT_BASE/$CAMPAIGN_TAG"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$WORK_ROOT" "$LOG_ROOT" "$OUT_ROOT/cards"

RUN_DIR="$(mktemp -d "$WORK_ROOT/run_${PROCESS}_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT

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
  echo "ERROR: SuperChic Cards directory not found for runtime." >&2
  echo "Tried: \$SUPERCHIC_DIR/Cards and install/build share/doc locations." >&2
  exit 1
fi
mkdir -p "$RUN_DIR/Cards"
cp -f "$SUPERCHIC_CARDS_DIR"/* "$RUN_DIR/Cards/"

awk -v outtg="$JOB_TAG_QUOTED" -v iseed="$SEED" -v nev="$NEVT" '
  /\[outtg\]/ { print outtg "          ! [outtg]"; next }
  /\[iseed\]/ { print iseed "           ! [iseed] : Random number seed (integer > 0)"; next }
  /\[nev\]/   { print nev "           ! [nev] : Number of events ( < 1000000 recommended)"; next }
  { print }
' "$CARD" > "$CARD_LOCAL"

LOG="$LOG_ROOT/run_${JOB_TAG}.log"
{
  echo "=== run_superchic: preparing init inputs ==="
} > "$LOG"

INIT_RTS=$(awk '/\[rts\]/ {print $1; exit}' "$CARD_LOCAL")
INIT_ISURV=$(awk '/\[isurv\]/ {print $1; exit}' "$CARD_LOCAL")
INIT_INTAG=$(awk '/\[intag\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$CARD_LOCAL")
INIT_PDFNAME=$(awk '/\[PDFname\]/ {gsub(/'\''/, "", $1); print $1; exit}' "$CARD_LOCAL")
INIT_PDFMEMBER=$(awk '/\[PDFmember\]/ {print $1; exit}' "$CARD_LOCAL")
INIT_SCREEN_FILE="screening${INIT_INTAG}.dat"

INIT_KEY="$(printf '%s|%s|%s|%s|%s\n' "$INIT_RTS" "$INIT_ISURV" "$INIT_INTAG" "$INIT_PDFNAME" "$INIT_PDFMEMBER" | sha1sum | awk '{print $1}')"
INIT_CACHE_ROOT="$WORK_ROOT/init_cache"
INIT_CACHE_DIR="$INIT_CACHE_ROOT/$INIT_KEY"

mkdir -p "$INIT_CACHE_ROOT"

if [[ -f "$INIT_CACHE_DIR/inputs/$INIT_SCREEN_FILE" ]]; then
  echo "Using cached init inputs for intag='$INIT_INTAG'."
  echo "Using cached init inputs: $INIT_CACHE_DIR/inputs" >> "$LOG"
  cp -rf "$INIT_CACHE_DIR/inputs" "$RUN_DIR/inputs"
else
  echo "Running SuperChic init for intag='$INIT_INTAG' (first time may take several minutes)..."
  echo "No cached init inputs found. Running init for key $INIT_KEY..." >> "$LOG"
  (
    cd "$RUN_DIR"
    init < job.DAT
  ) >> "$LOG" 2>&1

  if [[ ! -f "$RUN_DIR/inputs/$INIT_SCREEN_FILE" ]]; then
    echo "ERROR: init did not produce expected file inputs/$INIT_SCREEN_FILE in $RUN_DIR" >> "$LOG"
    exit 1
  fi

  mkdir -p "$INIT_CACHE_DIR"
  cp -rf "$RUN_DIR/inputs" "$INIT_CACHE_DIR/inputs"
fi

echo "=== run_superchic: running superchic ===" >> "$LOG"
(
  cd "$RUN_DIR"
  superchic < job.DAT
) >> "$LOG" 2>&1

if [[ -d "$RUN_DIR/evrecs" ]]; then
  mkdir -p "$OUT_ROOT/evrecs"
  cp -f "$RUN_DIR/evrecs"/* "$OUT_ROOT/evrecs/" 2>/dev/null || true
fi

if [[ -d "$RUN_DIR/outputs" ]]; then
  mkdir -p "$OUT_ROOT/outputs"
  cp -f "$RUN_DIR/outputs"/* "$OUT_ROOT/outputs/" 2>/dev/null || true
fi

cp -f "$CARD_LOCAL" "$OUT_ROOT/cards/job_${JOB_TAG}.DAT"

echo "Process: $PROCESS"
echo "Card source: $CARD"
echo "Output dir: $OUT_ROOT"
echo "Log file:   $LOG"
