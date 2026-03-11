#!/usr/bin/env bash

usage() {
  cat <<'USAGE'
Usage:
  run_superchic_signal.sh [--process h_bb|qcd_bb] [--card <dat_file>] \
    [--nev <events>] [--seed <seed>] [--out-tag <tag>] \
    [--output-base <dir>] <job_index>

Examples:
  ./run_superchic_signal.sh --process h_bb --nev 10000 --seed 1001 --out-tag hbb_001 1
  ./run_superchic_signal.sh --process qcd_bb --nev 10000 --out-tag qcd_001 1
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$STUDY_DIR/setup_env.sh"

PROCESS="h_bb"
CARD=""
NEVT=10000
SEED=""
OUT_TAG=""
OUTPUT_BASE="$STUDY_DIR/signal-generation/output"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process)
      PROCESS="$2"
      shift 2
      ;;
    --card|--template)
      CARD="$2"
      shift 2
      ;;
    --nev|--events)
      NEVT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --out-tag|--tag)
      OUT_TAG="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
fi

JOB_INDEX="$1"

case "$PROCESS" in
  h_bb|qcd_bb)
    ;;
  *)
    echo "ERROR: --process must be h_bb or qcd_bb (got: $PROCESS)" >&2
    exit 1
    ;;
esac

if [[ -z "$SEED" ]]; then
  SEED=$((1000 + JOB_INDEX))
fi

if [[ -z "$OUT_TAG" ]]; then
  OUT_TAG="${PROCESS}_${JOB_INDEX}"
fi

if [[ "$OUTPUT_BASE" != /* ]]; then
  OUTPUT_BASE="$STUDY_DIR/$OUTPUT_BASE"
fi

if [[ -z "$CARD" ]]; then
  CARD_CANDIDATES=()
  if [[ "$PROCESS" == "h_bb" ]]; then
    CARD_CANDIDATES=(
      "$STUDY_DIR/signal-generation/templates/h_bb_template.DAT"
      "$SUPERCHIC_DIR/bin/h_bb/h_bb.DAT"
      "$SUPERCHIC_DIR/Cards/h_bb.DAT"
    )
  else
    CARD_CANDIDATES=(
      "$STUDY_DIR/signal-generation/templates/qcd_bb_template.DAT"
      "$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb_template.DAT"
      "$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb.DAT"
      "$SUPERCHIC_DIR/Cards/qcd_bb.DAT"
    )
  fi

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
  if [[ "$PROCESS" == "h_bb" ]]; then
    echo "ERROR: card not found for process h_bb." >&2
    echo "Tried: signal-generation/templates/h_bb_template.DAT, \$SUPERCHIC_DIR/bin/h_bb/h_bb.DAT, \$SUPERCHIC_DIR/Cards/h_bb.DAT" >&2
  else
    echo "ERROR: card not found for process qcd_bb." >&2
    echo "Tried: signal-generation/templates/qcd_bb_template.DAT, \$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb_template.DAT, \$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb.DAT, \$SUPERCHIC_DIR/Cards/qcd_bb.DAT" >&2
  fi
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

WORK_ROOT="$STUDY_DIR/signal-generation/workspace"
OUT_ROOT="$OUTPUT_BASE/$PROCESS/$OUT_TAG"
LOG_ROOT="$STUDY_DIR/signal-generation/logs/$PROCESS"
mkdir -p "$WORK_ROOT" "$OUT_ROOT" "$LOG_ROOT"

RUN_DIR="$(mktemp -d "$WORK_ROOT/run_${PROCESS}_${OUT_TAG}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT

CARD_LOCAL="$RUN_DIR/job.DAT"
OUT_TAG_QUOTED="'$OUT_TAG'"

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

awk -v outtg="$OUT_TAG_QUOTED" -v iseed="$SEED" -v nev="$NEVT" '
  /\[outtg\]/ { print outtg "          ! [outtg]"; next }
  /\[iseed\]/ { print iseed "           ! [iseed] : Random number seed (integer > 0)"; next }
  /\[nev\]/   { print nev "           ! [nev] : Number of events ( < 1000000 recommended)"; next }
  { print }
' "$CARD" > "$CARD_LOCAL"

LOG="$LOG_ROOT/run_${OUT_TAG}.log"
{
  echo "=== run_superchic_signal: preparing init inputs ==="
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

echo "=== run_superchic_signal: running superchic ===" >> "$LOG"
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

cp -f "$CARD_LOCAL" "$OUT_ROOT/job_${OUT_TAG}.DAT"

echo "Process: $PROCESS"
echo "Card source: $CARD"
echo "Output dir: $OUT_ROOT"
echo "Log file:   $LOG"
