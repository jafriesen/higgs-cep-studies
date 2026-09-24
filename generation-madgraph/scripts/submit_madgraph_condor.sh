#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_madgraph_condor.sh --process PROCESS --campaign CAMPAIGN
    [--nev-per-job EVENTS] [--jobs N] [--init] [--overwrite] [--dry-run]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
METADATA_WRITER="$STUDY_DIR/common/write_metadata.py"
CONDOR_HELPER="$STUDY_DIR/common/submit_condor.py"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"
RUN_SCRIPT="$SCRIPT_DIR/run_madgraph.sh"
INIT_SCRIPT="$SCRIPT_DIR/prepare_madgraph_gridpack.sh"
ORIGINAL_ARGS=("$@")

PROCESS=""
CAMPAIGN=""
JOBS=100
NEV_PER_JOB=2000
RUN_INIT=false
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --nev-per-job) NEV_PER_JOB="$2"; shift 2 ;;
    --init) RUN_INIT=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$PROCESS" ]] || { echo "ERROR: --process is required." >&2; usage; }
[[ -n "$CAMPAIGN" ]] || { echo "ERROR: --campaign is required." >&2; usage; }
[[ "$JOBS" =~ ^[0-9]+$ ]] && (( JOBS > 0 )) || {
  echo "ERROR: --jobs must be a positive integer." >&2
  exit 1
}
[[ "$NEV_PER_JOB" =~ ^[0-9]+$ ]] && (( NEV_PER_JOB > 0 )) || {
  echo "ERROR: --nev-per-job must be a positive integer." >&2
  exit 1
}

source "$STUDY_DIR/env/setup_madgraph.sh"
eval "$(python3 "$PATH_HELPER" generation-env \
  --generator madgraph --process "$PROCESS" --campaign "$CAMPAIGN")"

if [[ "$RUN_INIT" == true ]]; then
  "$INIT_SCRIPT" --process "$PROCESS" --campaign "$CAMPAIGN"
fi

GRIDPACK="$GENERATION_ROOT/init/gridpack.tar.gz"
INIT_KEY_FILE="$GENERATION_ROOT/init/init_key.txt"
INIT_RUN_CARD="$GENERATION_ROOT/init/run_card.dat"
EXPECTED_KEY="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode key \
    --madgraph-version "$MADGRAPH_VERSION"
)"
PROCESS_COMMAND="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode process
)"
if [[ ! -s "$GRIDPACK" || ! -s "$INIT_RUN_CARD" || ! -f "$INIT_KEY_FILE" ||
      "$(cat "$INIT_KEY_FILE" 2>/dev/null || true)" != "$EXPECTED_KEY" ]]; then
  echo "ERROR: matching MadGraph gridpack is not available." >&2
  echo "Pass --init or run prepare_madgraph_gridpack.sh first." >&2
  exit 1
fi

if [[ -d "$CONDOR_DIR" && "$OVERWRITE" != true ]]; then
  echo "ERROR: submit destination already exists: $CONDOR_DIR" >&2
  echo "Use --overwrite to replace non-initialization generation outputs." >&2
  exit 1
fi
if [[ "$OVERWRITE" != true ]]; then
  for (( job = 1; job <= JOBS; job++ )); do
    tag="${CAMPAIGN}_${job}"
    for output in \
      "$EVENT_RECORDS_DIR/MadGraph_${PROCESS}_${tag}.lhe" \
      "$CARDS_DIR/card_${PROCESS}_${tag}.dat" \
      "$LOGS_DIR/run_${PROCESS}_${tag}.log"; do
      [[ ! -e "$output" ]] || {
        echo "ERROR: expected job output already exists: $output" >&2
        exit 1
      }
    done
  done
else
  rm -rf "$CONDOR_DIR" "$EVENT_RECORDS_DIR" "$CARDS_DIR"
  if [[ -d "$LOGS_DIR" ]]; then
    find "$LOGS_DIR" -maxdepth 1 -type f -name 'run_*.log' -delete
  fi
fi

mkdir -p "$CONDOR_DIR" "$EVENT_RECORDS_DIR" "$CARDS_DIR" "$LOGS_DIR"
JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
GRIDPACK_NAME="$(basename "$GRIDPACK")"

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
JOB_INDEX="\${1:?missing JOB_INDEX}"
export HIGGS_CEP_CONDOR_JOB=1
export HIGGS_CEP_MADGRAPH_GRIDPACK="\${_CONDOR_SCRATCH_DIR:?}/$GRIDPACK_NAME"
exec $(printf '%q' "$RUN_SCRIPT") \
  --process $(printf '%q' "$PROCESS") \
  --campaign $(printf '%q' "$CAMPAIGN") \
  --nev "$NEV_PER_JOB" \
  --job "\$JOB_INDEX"
EOF
chmod +x "$JOB_SCRIPT"

printf -v COMMAND '%q ' "$0" "${ORIGINAL_ARGS[@]}"
METADATA_ARGS=(
  --output "$METADATA_FILE" \
  --string-field "generator=madgraph" \
  --string-field "process=$PROCESS" \
  --string-field "campaign=$CAMPAIGN" \
  --string-field "process_command=$PROCESS_COMMAND" \
  --string-field "mode=condor" \
  --field "jobs=$JOBS" \
  --field "events_per_job=$NEV_PER_JOB" \
  --field "seed_start=1001" \
  --string-field "gridpack_key=$EXPECTED_KEY" \
  --string-field "madgraph_version=$MADGRAPH_VERSION" \
  --field "init=$RUN_INIT" \
  --field "dry_run=$DRY_RUN" \
  --field "overwrite=$OVERWRITE" \
  --string-field "command=${COMMAND% }" \
  --string-field "created_at=$(date -Iseconds)" \
  --string-field "runtime_source=$LCG_VIEW"
)
CARD_METADATA="$(
  python3 "$CARD_GENERATOR" \
    --process "$PROCESS" --campaign "$CAMPAIGN" --mode metadata \
    --run-card "$INIT_RUN_CARD"
)"
while IFS=$'\t' read -r FIELD_TYPE FIELD_VALUE; do
  case "$FIELD_TYPE" in
    field|string-field) METADATA_ARGS+=("--$FIELD_TYPE" "$FIELD_VALUE") ;;
    *) echo "ERROR: invalid card metadata field type: $FIELD_TYPE" >&2; exit 1 ;;
  esac
done <<< "$CARD_METADATA"
python3 "$METADATA_WRITER" "${METADATA_ARGS[@]}"

echo "Generation directory: $GENERATION_ROOT"
CONDOR_ARGS=(
  --condor-dir "$CONDOR_DIR"
  --executable "$JOB_SCRIPT"
  --payload "$GRIDPACK"
  --jobs "$JOBS"
)
[[ "$DRY_RUN" == true ]] && CONDOR_ARGS+=(--dry-run)
python3 "$CONDOR_HELPER" "${CONDOR_ARGS[@]}"
