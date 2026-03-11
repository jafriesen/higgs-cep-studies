#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_superchic_signal.sh [--template <dat_file>] [--nev <events>] [--seed <seed>] [--out-tag <tag>] <job_index>

Examples:
  ./run_superchic_signal.sh --nev 10000 --seed 1001 --out-tag hbb_001 1
  ./run_superchic_signal.sh --template ../templates/qcd_bb_template.DAT --out-tag qcd_001 1
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$STUDY_DIR/setup_env.sh"

TEMPLATE="$SCRIPT_DIR/../templates/h_bb_template.DAT"
NEVT=10000
SEED=""
OUT_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --template|--card)
      TEMPLATE="$2"
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

if [[ -z "$OUT_TAG" ]]; then
  OUT_TAG="$JOB_INDEX"
fi

if [[ -z "$SEED" ]]; then
  SEED=$((1000 + JOB_INDEX))
fi

if [[ "$TEMPLATE" != /* ]]; then
  if [[ -f "$SCRIPT_DIR/$TEMPLATE" ]]; then
    TEMPLATE="$SCRIPT_DIR/$TEMPLATE"
  elif [[ -f "$SCRIPT_DIR/../$TEMPLATE" ]]; then
    TEMPLATE="$SCRIPT_DIR/../$TEMPLATE"
  elif [[ -f "$SCRIPT_DIR/../templates/$TEMPLATE" ]]; then
    TEMPLATE="$SCRIPT_DIR/../templates/$TEMPLATE"
  fi
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

WORK_ROOT="$STUDY_DIR/signal-generation/workspace"
OUT_ROOT="$STUDY_DIR/signal-generation/output/$OUT_TAG"
LOG_ROOT="$STUDY_DIR/signal-generation/logs"
mkdir -p "$WORK_ROOT" "$OUT_ROOT" "$LOG_ROOT"

RUN_DIR="$(mktemp -d "$WORK_ROOT/run_${OUT_TAG}_XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT
cp "$TEMPLATE" "$RUN_DIR/job.DAT"

sed -i "s/@SEED@/$SEED/g" "$RUN_DIR/job.DAT"
sed -i "s/@NEVT@/$NEVT/g" "$RUN_DIR/job.DAT"
sed -i "s/@OUT@/$OUT_TAG/g" "$RUN_DIR/job.DAT"

if [[ ! -x "$(command -v superchic)" ]]; then
  echo "ERROR: superchic not found in PATH. Source setup_env.sh and check SuperChic installation." >&2
  exit 1
fi

LOG="$LOG_ROOT/run_${OUT_TAG}.log"
(
  cd "$RUN_DIR"
  superchic < "job.DAT"
) > "$LOG" 2>&1

if [[ -d "$RUN_DIR/evrecs" ]]; then
  cp -f "$RUN_DIR/evrecs"/* "$OUT_ROOT/"
fi

echo "Generated output copied to: $OUT_ROOT"
echo "Log file: $LOG"
