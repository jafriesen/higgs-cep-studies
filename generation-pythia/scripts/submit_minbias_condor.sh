#!/usr/bin/env bash

set -euo pipefail

REQUEST_MEMORY=4096

usage() {
  cat <<'USAGE'
Usage:
  submit_minbias_condor.sh (--events TOTAL | --events-per-job EVENTS) --jobs N \
    [--campaign NAME] [--seed-base BASE] [--e-cm ECM_GEV] \
    [--processes "SoftQCD:all"] [--output-dir DIR] [--overwrite] [--dry-run]

Examples:
  ./submit_minbias_condor.sh --events 100000 --jobs 100 --campaign minbias_v01
  ./submit_minbias_condor.sh --events-per-job 1000 --jobs 20 --seed-base 5000 --dry-run
USAGE
  exit 1
}

quote_args() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf '%q' "$arg")")
  done
  printf '%s' "${quoted[*]}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GEN_SCRIPT="$STUDY_DIR/generation-pythia/scripts/generate_minbias.py"
ORIGINAL_ARGS=("$@")

TOTAL_EVENTS=""
JOBS=""
EVENTS_PER_JOB=""
CAMPAIGN="minbias__test"
SEED_BASE=""
E_CM=14000.0
PROCESSES="SoftQCD:all"
OUTPUT_DIR=""
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --events) TOTAL_EVENTS="$2"; shift 2 ;;
    --events-per-job) EVENTS_PER_JOB="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --campaign|--out-tag|--tag) CAMPAIGN="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --e-cm) E_CM="$2"; shift 2 ;;
    --processes) PROCESSES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --overwrite) OVERWRITE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

if [[ -z "$JOBS" ]]; then
  usage
fi
if [[ -n "$TOTAL_EVENTS" && -n "$EVENTS_PER_JOB" ]]; then
  echo "ERROR: pass only one of --events or --events-per-job." >&2
  exit 1
fi
if [[ -z "$TOTAL_EVENTS" && -z "$EVENTS_PER_JOB" ]]; then
  echo "ERROR: pass either --events or --events-per-job." >&2
  exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS <= 0 )); then
  echo "ERROR: --jobs must be a positive integer." >&2
  exit 1
fi
if [[ -n "$TOTAL_EVENTS" ]] && { ! [[ "$TOTAL_EVENTS" =~ ^[0-9]+$ ]] || (( TOTAL_EVENTS <= 0 )); }; then
  echo "ERROR: --events must be a positive integer." >&2
  exit 1
fi
if [[ -n "$EVENTS_PER_JOB" ]] && { ! [[ "$EVENTS_PER_JOB" =~ ^[0-9]+$ ]] || (( EVENTS_PER_JOB <= 0 )); }; then
  echo "ERROR: --events-per-job must be a positive integer." >&2
  exit 1
fi
if [[ -n "$SEED_BASE" ]] && { ! [[ "$SEED_BASE" =~ ^[0-9]+$ ]]; }; then
  echo "ERROR: --seed-base must be a non-negative integer." >&2
  exit 1
fi
if [[ -z "$CAMPAIGN" ]]; then
  echo "ERROR: --campaign must not be empty." >&2
  exit 1
fi
if [[ ! -f "$GEN_SCRIPT" ]]; then
  echo "ERROR: generator not found: $GEN_SCRIPT" >&2
  exit 1
fi

source "$STUDY_DIR/env/setup_pythia.sh"

if [[ -n "$OUTPUT_DIR" ]]; then
  if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$STUDY_DIR/$OUTPUT_DIR"
  fi
  CAMPAIGN_DIR="$OUTPUT_DIR/$CAMPAIGN"
else
  CAMPAIGN_DIR="$(python3 - "$STUDY_DIR" "$CAMPAIGN" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1])))
from common.config_utils import resolve_minbias_campaign

campaign_dir, _ = resolve_minbias_campaign(sys.argv[2])
print(campaign_dir)
PY
)"
fi

CONDOR_DIR="$CAMPAIGN_DIR/condor"
PARQUET_DIR="$CAMPAIGN_DIR/parquet"
SUB_FILE="$CONDOR_DIR/submit.sub"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
META_FILE="$CONDOR_DIR/campaign_args.txt"

if [[ -d "$CONDOR_DIR" && "$OVERWRITE" != true ]]; then
  echo "ERROR: submit destination already exists: $CONDOR_DIR" >&2
  echo "Use --overwrite to recreate Condor submit artifacts." >&2
  exit 1
fi

if [[ "$OVERWRITE" == true ]]; then
  rm -rf "$CONDOR_DIR"
fi

mkdir -p "$CONDOR_DIR" "$PARQUET_DIR"

if [[ -n "$EVENTS_PER_JOB" ]]; then
  TOTAL_EVENTS=$(( EVENTS_PER_JOB * JOBS ))
fi

if (( TOTAL_EVENTS < JOBS )); then
  echo "ERROR: --events must be at least --jobs so every job has events to generate." >&2
  exit 1
fi

cat > "$QUEUE_FILE" <<EOF
EOF
BASE_EVENTS=$(( TOTAL_EVENTS / JOBS ))
REMAINDER=$(( TOTAL_EVENTS % JOBS ))
SPLIT_BOUNDARY=$(( JOBS - REMAINDER ))
for (( job = 1; job <= JOBS; job++ )); do
  if [[ -n "$EVENTS_PER_JOB" ]]; then
    JOB_EVENTS="$EVENTS_PER_JOB"
  elif (( job <= SPLIT_BOUNDARY )); then
    JOB_EVENTS="$BASE_EVENTS"
  else
    JOB_EVENTS=$(( BASE_EVENTS + 1 ))
  fi

  if [[ -n "$SEED_BASE" ]]; then
    JOB_SEED=$(( SEED_BASE + job - 1 ))
  else
    JOB_SEED="-"
  fi
  echo "$job $JOB_EVENTS $JOB_SEED" >> "$QUEUE_FILE"
done

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

JOB_INDEX="\${1:?missing JOB_INDEX}"
EVENTS="\${2:?missing EVENTS}"
SEED="\${3:-}"

STUDY_DIR=$(printf '%q' "$STUDY_DIR")
PARQUET_DIR=$(printf '%q' "$PARQUET_DIR")
CAMPAIGN=$(printf '%q' "$CAMPAIGN")
E_CM=$(printf '%q' "$E_CM")
PROCESSES=$(printf '%q' "$PROCESSES")

echo "Job start timestamp: \$(date -Iseconds)"
echo "Batch host: \$(hostname)"
echo "Batch scratch: \$PWD"
echo "Job index: \$JOB_INDEX"
echo "Events: \$EVENTS"
echo "Seed: \${SEED:-<generator default>}"

cd "\$STUDY_DIR"
source env/setup_pythia.sh

python3 -c "import pythia8mc, pyarrow"

mkdir -p "\$PARQUET_DIR"
OUTPUT="\$PARQUET_DIR/\${CAMPAIGN}_job_\${JOB_INDEX}.parquet"
GEN_ARGS=(
  --events "\$EVENTS"
  --campaign "\$CAMPAIGN"
  --e-cm "\$E_CM"
  --processes "\$PROCESSES"
  --output "\$OUTPUT"
)
if [[ -n "\$SEED" && "\$SEED" != "-" ]]; then
  GEN_ARGS+=(--seed "\$SEED")
fi

echo "Run command: python3 -u generation-pythia/scripts/generate_minbias.py \${GEN_ARGS[*]}"
python3 -u generation-pythia/scripts/generate_minbias.py "\${GEN_ARGS[@]}"
EOF
chmod +x "$JOB_SCRIPT"

cat > "$META_FILE" <<EOF
created_at=$(date -Iseconds)
command=$(quote_args "$0" "${ORIGINAL_ARGS[@]}")
campaign=$CAMPAIGN
events=$TOTAL_EVENTS
jobs=$JOBS
events_per_job=${EVENTS_PER_JOB:-<split-total>}
seed_base=${SEED_BASE:-<generator-default>}
e_cm=$E_CM
processes=$PROCESSES
output_dir=${OUTPUT_DIR:-<config default>}
campaign_dir=$CAMPAIGN_DIR
parquet_dir=$PARQUET_DIR
request_memory=$REQUEST_MEMORY
overwrite=$OVERWRITE
EOF

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $JOB_SCRIPT
arguments = \$(JOB_INDEX) \$(EVENTS) \$(SEED)
output = $CONDOR_DIR/job_\$(JOB_INDEX).out
error = $CONDOR_DIR/job_\$(JOB_INDEX).err
log = $CONDOR_DIR/cluster.log
stream_output = False
stream_error = False
request_memory = $REQUEST_MEMORY
request_cpus = 1
getenv = True
queue JOB_INDEX, EVENTS, SEED from $QUEUE_FILE
EOF

echo "Campaign directory: $CAMPAIGN_DIR"
echo "Parquet directory: $PARQUET_DIR"
echo "Condor directory: $CONDOR_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Events: $TOTAL_EVENTS"
echo "Jobs: $JOBS"
echo "Split summary: base=${BASE_EVENTS}, remainder=${REMAINDER} (extra events to last ${REMAINDER} jobs)"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

command -v condor_submit >/dev/null 2>&1 || {
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
}
condor_submit "$SUB_FILE"
