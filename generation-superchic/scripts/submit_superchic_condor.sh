#!/usr/bin/env bash

set -euo pipefail

REQUEST_MEMORY=2048

usage() {
  cat <<'USAGE'
Usage:
  submit_superchic_condor.sh [--jobs N] [--process PROCESS] [--campaign NAME] \
    [--nev-per-job EVENTS] [--seed-base BASE] [--card DAT_FILE] \
    [--output-dir DIR] [--init] [--overwrite] [--dry-run]

Examples:
  ./submit_superchic_condor.sh --process Hbb --campaign Hbb__test --nev-per-job 10000 --jobs 20 --init
  ./submit_superchic_condor.sh --process QCDbb --campaign QCDbb__test --nev-per-job 20000 --jobs 50 --seed-base 5000
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
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
RUN_SCRIPT="$STUDY_DIR/generation-superchic/scripts/run_superchic.sh"
INIT_SCRIPT="$STUDY_DIR/generation-superchic/scripts/prepare_superchic_init.sh"
ORIGINAL_ARGS=("$@")

PROCESS="Hbb"
JOBS=100
NEV_PER_JOB=2000
CARD=""
SEED_BASE=""
CAMPAIGN=""
OUTPUT_DIR=""
RUN_INIT=false
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --campaign|--out-tag|--tag) CAMPAIGN="$2"; shift 2 ;;
    --nev-per-job) NEV_PER_JOB="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --card|--template) CARD="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --init) RUN_INIT=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS <= 0 )); then
  echo "ERROR: --jobs must be a positive integer." >&2
  exit 1
fi
if ! [[ "$NEV_PER_JOB" =~ ^[0-9]+$ ]] || (( NEV_PER_JOB <= 0 )); then
  echo "ERROR: --nev-per-job must be a positive integer." >&2
  exit 1
fi
if [[ -n "$SEED_BASE" ]] && { ! [[ "$SEED_BASE" =~ ^[0-9]+$ ]]; }; then
  echo "ERROR: --seed-base must be a non-negative integer." >&2
  exit 1
fi

[[ -n "$CAMPAIGN" ]] || CAMPAIGN="${PROCESS}__test"
if [[ -z "$CARD" ]]; then
  CARD="$STUDY_DIR/generation-superchic/cards/$PROCESS.DAT"
elif [[ "$CARD" != /* ]]; then
  CARD="$STUDY_DIR/$CARD"
fi
if [[ ! -f "$CARD" ]]; then
  echo "ERROR: card not found for process $PROCESS: $CARD" >&2
  exit 1
fi

source "$STUDY_DIR/setup_env.sh"

PATH_ARGS=(--process "$PROCESS" --campaign "$CAMPAIGN")
RUN_OUTPUT_ARG=()
if [[ -n "$OUTPUT_DIR" ]]; then
  PATH_ARGS+=(--output-dir "$OUTPUT_DIR")
  RUN_OUTPUT_ARG=(--output-dir "$OUTPUT_DIR")
fi
PATH_ENV="$(python3 "$PATH_HELPER" superchic-env "${PATH_ARGS[@]}")" || exit 1
eval "$PATH_ENV"

CONDOR_DIR="$SUPERCHIC_ROOT/condor"
SUB_FILE="$CONDOR_DIR/submit.sub"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
META_FILE="$CONDOR_DIR/campaign_args.txt"
PAYLOAD_TAR="$CONDOR_DIR/payload.tar.gz"

if [[ -d "$CONDOR_DIR" && "$OVERWRITE" != true ]]; then
  echo "ERROR: submit destination already exists: $CONDOR_DIR" >&2
  echo "Use --overwrite to clear existing non-init SuperChic outputs and recreate submit artifacts." >&2
  exit 1
fi

if [[ "$OVERWRITE" == true ]]; then
  echo "Overwrite requested: clearing existing non-init SuperChic outputs under $SUPERCHIC_ROOT"
  echo "Preserving initialized inputs under $SUPERCHIC_ROOT/init"
  rm -rf "$CONDOR_DIR" "$SUPERCHIC_EVRECS_DIR" "$SUPERCHIC_OUTPUT_DIR" "$SUPERCHIC_CARDS_DIR"
  if [[ -d "$SUPERCHIC_LOGS_DIR" ]]; then
    find "$SUPERCHIC_LOGS_DIR" -maxdepth 1 -type f -name 'run_*.log' -delete
  fi
fi

mkdir -p "$CONDOR_DIR" "$SUPERCHIC_EVRECS_DIR" "$SUPERCHIC_OUTPUT_DIR" "$SUPERCHIC_CARDS_DIR" "$SUPERCHIC_LOGS_DIR"

if [[ "$RUN_INIT" == true ]]; then
  INIT_ARGS=(--process "$PROCESS" --out-tag "$CAMPAIGN" --card "$CARD")
  if [[ -n "$OUTPUT_DIR" ]]; then
    INIT_ARGS+=(--output-dir "$OUTPUT_DIR")
  fi
  echo "Preparing SuperChic init:"
  echo "  $(quote_args "$INIT_SCRIPT" "${INIT_ARGS[@]}")"
  "$INIT_SCRIPT" "${INIT_ARGS[@]}"
fi

INIT_INPUTS_DIR="$SUPERCHIC_ROOT/init/inputs"
if [[ ! -d "$INIT_INPUTS_DIR" ]]; then
  echo "ERROR: initialized SuperChic inputs missing: $INIT_INPUTS_DIR" >&2
  echo "Run $INIT_SCRIPT --process $PROCESS --out-tag $CAMPAIGN --card $CARD before submitting, or pass --init." >&2
  exit 1
fi

PAYLOAD_TMP="$(mktemp -d "${TMPDIR:-/tmp}/higgs_cep_payload_XXXXXX")"
trap 'rm -rf "$PAYLOAD_TMP"' EXIT
PAYLOAD_ROOT="$PAYLOAD_TMP/payload"
PAYLOAD_STUDY_DIR="$PAYLOAD_ROOT/higgs-cep-studies"
PAYLOAD_SUPERCHIC_DIR="$PAYLOAD_ROOT/SuperChic"

mkdir -p \
  "$PAYLOAD_STUDY_DIR/common" \
  "$PAYLOAD_STUDY_DIR/generation-superchic/scripts" \
  "$PAYLOAD_STUDY_DIR/generation-superchic/cards" \
  "$PAYLOAD_SUPERCHIC_DIR/install/bin" \
  "$PAYLOAD_SUPERCHIC_DIR/install/share" \
  "$PAYLOAD_SUPERCHIC_DIR/Cards" \
  "$PAYLOAD_SUPERCHIC_DIR/SF" \
  "$PAYLOAD_ROOT/init"

cp -f "$STUDY_DIR/setup_env.sh" "$PAYLOAD_STUDY_DIR/setup_env.sh"
cp -f "$STUDY_DIR/config.yaml" "$PAYLOAD_STUDY_DIR/config.yaml"
cp -f "$STUDY_DIR/processes.yaml" "$PAYLOAD_STUDY_DIR/processes.yaml"
cp -f "$STUDY_DIR/common/"*.py "$PAYLOAD_STUDY_DIR/common/"
cp -f "$RUN_SCRIPT" "$PAYLOAD_STUDY_DIR/generation-superchic/scripts/"
cp -f "$INIT_SCRIPT" "$PAYLOAD_STUDY_DIR/generation-superchic/scripts/"
cp -f "$CARD" "$PAYLOAD_STUDY_DIR/generation-superchic/cards/$PROCESS.DAT"

cp -f "$SUPERCHIC_BIN_DIR/superchic" "$PAYLOAD_SUPERCHIC_DIR/install/bin/"
if [[ -x "$SUPERCHIC_BIN_DIR/init" ]]; then
  cp -f "$SUPERCHIC_BIN_DIR/init" "$PAYLOAD_SUPERCHIC_DIR/install/bin/"
fi
if [[ -d "$SUPERCHIC_DATA_PATH" ]]; then
  cp -a "$SUPERCHIC_DATA_PATH" "$PAYLOAD_SUPERCHIC_DIR/install/share/"
  rm -rf "$PAYLOAD_SUPERCHIC_DIR/install/share/SuperChic/SF"
fi
if [[ -d "$SUPERCHIC_DIR/Cards" ]]; then
  cp -a "$SUPERCHIC_DIR/Cards/." "$PAYLOAD_SUPERCHIC_DIR/Cards/"
fi
if [[ -d "$SUPERCHIC_DIR/SF" ]]; then
  cp -a "$SUPERCHIC_DIR/SF/." "$PAYLOAD_SUPERCHIC_DIR/SF/"
fi
cp -a "$INIT_INPUTS_DIR" "$PAYLOAD_ROOT/init/inputs"

tar -C "$PAYLOAD_TMP" -czf "$PAYLOAD_TAR" payload

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

JOB_INDEX="\${1:?missing JOB_INDEX}"
SEED="\${2:-}"

SUBMIT_TIME="$(date -Iseconds)"
JOB_START_TIME="\$(date -Iseconds)"
FINAL_SUPERCHIC_ROOT=$(printf '%q' "$SUPERCHIC_ROOT")
FINAL_EVRECS_DIR=$(printf '%q' "$SUPERCHIC_EVRECS_DIR")
FINAL_OUTPUT_DIR=$(printf '%q' "$SUPERCHIC_OUTPUT_DIR")
FINAL_CARDS_DIR=$(printf '%q' "$SUPERCHIC_CARDS_DIR")
FINAL_LOGS_DIR=$(printf '%q' "$SUPERCHIC_LOGS_DIR")

echo "Submit timestamp: \$SUBMIT_TIME"
echo "Job start timestamp: \$JOB_START_TIME"
echo "Batch host: \$(hostname)"
echo "Batch scratch: \$PWD"
echo "Job index: \$JOB_INDEX"
echo "Seed: \${SEED:-<runner default>}"

tar -xzf payload.tar.gz

LOCAL_ROOT="\$PWD/payload"
LOCAL_STUDY_DIR="\$LOCAL_ROOT/higgs-cep-studies"
LOCAL_OUTPUT_DIR="\$PWD/local_output/$PROCESS"
LOCAL_SUPERCHIC_ROOT="\$LOCAL_OUTPUT_DIR/$CAMPAIGN/SuperChic"

mkdir -p "\$LOCAL_SUPERCHIC_ROOT/init"
cp -a "\$LOCAL_ROOT/init/inputs" "\$LOCAL_SUPERCHIC_ROOT/init/inputs"
cd "\$LOCAL_STUDY_DIR"

RUN_ARGS=(--process "$PROCESS" --nev "$NEV_PER_JOB" --out-tag "$CAMPAIGN" --output-dir "\$LOCAL_OUTPUT_DIR" --job "\$JOB_INDEX")
if [[ -n "\$SEED" ]]; then
  RUN_ARGS+=(--seed "\$SEED")
fi

echo "Run command: ./generation-superchic/scripts/run_superchic.sh \${RUN_ARGS[*]}"
HIGGS_CEP_SUPERCHIC_DIR="\$LOCAL_ROOT/SuperChic" ./generation-superchic/scripts/run_superchic.sh "\${RUN_ARGS[@]}"

mkdir -p "\$FINAL_EVRECS_DIR" "\$FINAL_OUTPUT_DIR" "\$FINAL_CARDS_DIR" "\$FINAL_LOGS_DIR"
if [[ -d "\$LOCAL_SUPERCHIC_ROOT/evrecs" ]]; then
  cp -f "\$LOCAL_SUPERCHIC_ROOT/evrecs/"* "\$FINAL_EVRECS_DIR/" 2>/dev/null || true
fi
if [[ -d "\$LOCAL_SUPERCHIC_ROOT/output" ]]; then
  cp -f "\$LOCAL_SUPERCHIC_ROOT/output/"* "\$FINAL_OUTPUT_DIR/" 2>/dev/null || true
fi
if [[ -d "\$LOCAL_SUPERCHIC_ROOT/cards" ]]; then
  cp -f "\$LOCAL_SUPERCHIC_ROOT/cards/"* "\$FINAL_CARDS_DIR/" 2>/dev/null || true
fi
if [[ -d "\$LOCAL_SUPERCHIC_ROOT/logs" ]]; then
  cp -f "\$LOCAL_SUPERCHIC_ROOT/logs/"* "\$FINAL_LOGS_DIR/" 2>/dev/null || true
fi

echo "Copied final outputs to: \$FINAL_SUPERCHIC_ROOT"
EOF
chmod +x "$JOB_SCRIPT"

cat > "$META_FILE" <<EOF
created_at=$(date -Iseconds)
command=$(quote_args "$0" "${ORIGINAL_ARGS[@]}")
process=$PROCESS
campaign=$CAMPAIGN
jobs=$JOBS
nev_per_job=$NEV_PER_JOB
card=$CARD
seed_base=${SEED_BASE:-<runner-default>}
output_dir=${OUTPUT_DIR:-<config default>}
request_memory=$REQUEST_MEMORY
overwrite=$OVERWRITE
superchic_dir=$SUPERCHIC_DIR
superchic_bin_dir=$SUPERCHIC_BIN_DIR
superchic_install_dir=$SUPERCHIC_INSTALL_DIR
payload=$PAYLOAD_TAR
init_inputs=$INIT_INPUTS_DIR
EOF

if [[ -n "$SEED_BASE" ]]; then
  : > "$QUEUE_FILE"
  for (( j = 1; j <= JOBS; j++ )); do
    echo "$j $((SEED_BASE + j - 1))" >> "$QUEUE_FILE"
  done
  QUEUE_LINE="queue JOB_INDEX, SEED from $QUEUE_FILE"
  JOB_ARGUMENTS="\$(JOB_INDEX) \$(SEED)"
else
  : > "$QUEUE_FILE"
  for (( j = 1; j <= JOBS; j++ )); do
    echo "$j" >> "$QUEUE_FILE"
  done
  QUEUE_LINE="queue JOB_INDEX from $QUEUE_FILE"
  JOB_ARGUMENTS="\$(JOB_INDEX)"
fi

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $JOB_SCRIPT
arguments = $JOB_ARGUMENTS
transfer_input_files = $PAYLOAD_TAR
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
output = $CONDOR_DIR/job_\$(JOB_INDEX).out
error = $CONDOR_DIR/job_\$(JOB_INDEX).err
log = $CONDOR_DIR/cluster.log
stream_output = False
stream_error = False
max_idle = 50
request_memory = $REQUEST_MEMORY
request_cpus = 1
getenv = True
$QUEUE_LINE
EOF

echo "Campaign directory: $CAMPAIGN_ROOT"
echo "SuperChic directory: $SUPERCHIC_ROOT"
echo "SuperChic install source: $SUPERCHIC_DIR"
echo "Condor directory: $CONDOR_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Payload: $PAYLOAD_TAR"
echo "Process: $PROCESS"
echo "Campaign: $CAMPAIGN"
echo "Events/job: $NEV_PER_JOB"
echo "Jobs: $JOBS"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

command -v condor_submit >/dev/null 2>&1 || {
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
}
condor_submit "$SUB_FILE"
