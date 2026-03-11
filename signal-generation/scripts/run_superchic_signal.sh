#!/usr/bin/env bash
set -euo pipefail

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
  if [[ "$PROCESS" == "h_bb" ]]; then
    CARD="$SUPERCHIC_DIR/bin/h_bb/h_bb.DAT"
  else
    if [[ -f "$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb_template.DAT" ]]; then
      CARD="$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb_template.DAT"
    else
      CARD="$SUPERCHIC_DIR/bin/qcd_bb/qcd_bb.DAT"
    fi
  fi
fi

if [[ ! -f "$CARD" ]]; then
  echo "ERROR: card not found: $CARD" >&2
  exit 1
fi

if ! command -v superchic >/dev/null 2>&1; then
  echo "ERROR: superchic is not in PATH. Check SuperChic build and setup_env.sh." >&2
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

awk -v outtg="$OUT_TAG_QUOTED" -v iseed="$SEED" -v nev="$NEVT" '
  /\[outtg\]/ { print outtg "          ! [outtg]"; next }
  /\[iseed\]/ { print iseed "           ! [iseed] : Random number seed (integer > 0)"; next }
  /\[nev\]/   { print nev "           ! [nev] : Number of events ( < 1000000 recommended)"; next }
  { print }
' "$CARD" > "$CARD_LOCAL"

LOG="$LOG_ROOT/run_${OUT_TAG}.log"
(
  cd "$RUN_DIR"
  superchic < job.DAT
) > "$LOG" 2>&1

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
