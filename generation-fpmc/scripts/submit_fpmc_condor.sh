#!/usr/bin/env bash

set -euo pipefail

if [[ "${HIGGS_CEP_FPMC_ENV_LOADED:-0}" == "1" &&
      "${HIGGS_CEP_FPMC_CONDOR_CLEAN:-0}" != "1" ]]; then
  exec /usr/bin/env \
    -u LD_LIBRARY_PATH -u COMPILER_PATH -u CC -u CXX -u FC \
    -u HIGGS_CEP_FPMC_ENV_LOADED \
    HIGGS_CEP_FPMC_CONDOR_CLEAN=1 /bin/bash "$0" "$@"
fi

usage() {
  cat <<'USAGE'
Usage:
  submit_fpmc_condor.sh --process PROCESS --campaign CAMPAIGN
    [--nev-per-job EVENTS] [--jobs N] [--overwrite] [--dry-run]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
METADATA_WRITER="$STUDY_DIR/common/write_metadata.py"
CONDOR_HELPER="$STUDY_DIR/common/submit_condor.py"
RUN_SCRIPT="$SCRIPT_DIR/run_fpmc.sh"
ORIGINAL_ARGS=("$@")

PROCESS=""
CAMPAIGN=""
JOBS=100
NEV_PER_JOB=2000
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --nev-per-job) NEV_PER_JOB="$2"; shift 2 ;;
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

eval "$(python3 "$PATH_HELPER" generation-env \
  --generator fpmc --process "$PROCESS" --campaign "$CAMPAIGN")"

FPMC_DIR="${HIGGS_CEP_FPMC_DIR:-$(cd "$STUDY_DIR/.." && pwd)/fpmc}"
FPMC_BUILD_DIR="${HIGGS_CEP_FPMC_BUILD_DIR:-$FPMC_DIR/build}"
FPMC_EXE="${HIGGS_CEP_FPMC_EXE:-$FPMC_BUILD_DIR/fpmc-lhe}"
FPMC_GCC_SETUP="${HIGGS_CEP_FPMC_GCC_SETUP:-/cvmfs/sft.cern.ch/lcg/external/gcc/6.1.0/x86_64-slc6/setup.sh}"
FPMC_LIB="$FPMC_BUILD_DIR/libFpmc.so"
COMPHEP_LIB="$FPMC_BUILD_DIR/Fpmc/External/comphep_interface/libcomphep.so"
FPMC_EXTERNAL="$FPMC_BUILD_DIR/External"

for required in "$FPMC_EXE" "$FPMC_LIB" "$COMPHEP_LIB" "$FPMC_GCC_SETUP"; do
  [[ -f "$required" ]] || {
    echo "ERROR: required FPMC runtime file not found: $required" >&2
    exit 1
  }
done
[[ -d "$FPMC_EXTERNAL" ]] || {
  echo "ERROR: FPMC runtime data not found: $FPMC_EXTERNAL" >&2
  exit 1
}

if [[ -d "$CONDOR_DIR" && "$OVERWRITE" != true ]]; then
  echo "ERROR: submit destination already exists: $CONDOR_DIR" >&2
  echo "Use --overwrite to replace this campaign's generation outputs." >&2
  exit 1
fi
if [[ "$OVERWRITE" != true ]]; then
  for (( job = 1; job <= JOBS; job++ )); do
    tag="${PROCESS}_${CAMPAIGN}_${job}"
    for output in \
      "$EVENT_RECORDS_DIR/FPMC_${tag}.lhe" \
      "$CARDS_DIR/card_${tag}.dat" \
      "$LOGS_DIR/run_${tag}.log"; do
      [[ ! -e "$output" ]] || {
        echo "ERROR: expected job output already exists: $output" >&2
        exit 1
      }
    done
  done
fi
if [[ "$OVERWRITE" == true ]]; then
  rm -rf "$CONDOR_DIR" "$EVENT_RECORDS_DIR" "$CARDS_DIR" "$LOGS_DIR"
fi
mkdir -p "$CONDOR_DIR" "$EVENT_RECORDS_DIR" "$CARDS_DIR" "$LOGS_DIR"

JOB_SCRIPT="$CONDOR_DIR/run_job.sh"

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
JOB_INDEX="\${1:?missing JOB_INDEX}"
export HIGGS_CEP_CONDOR_JOB=1
export HIGGS_CEP_FPMC_DIR=$(printf '%q' "$FPMC_DIR")
export HIGGS_CEP_FPMC_BUILD_DIR=$(printf '%q' "$FPMC_BUILD_DIR")
export HIGGS_CEP_FPMC_EXE=$(printf '%q' "$FPMC_EXE")
export HIGGS_CEP_FPMC_GCC_SETUP=$(printf '%q' "$FPMC_GCC_SETUP")
exec $(printf '%q' "$RUN_SCRIPT") \
  --process $(printf '%q' "$PROCESS") \
  --campaign $(printf '%q' "$CAMPAIGN") \
  --nev "$NEV_PER_JOB" \
  --job "\$JOB_INDEX"
EOF
chmod +x "$JOB_SCRIPT"

printf -v COMMAND '%q ' "$0" "${ORIGINAL_ARGS[@]}"
python3 "$METADATA_WRITER" \
  --output "$METADATA_FILE" \
  --string-field "generator=fpmc" \
  --string-field "process=$PROCESS" \
  --string-field "campaign=$CAMPAIGN" \
  --string-field "mode=condor" \
  --field "jobs=$JOBS" \
  --field "events_per_job=$NEV_PER_JOB" \
  --field "seed_start=33799" \
  --field "dry_run=$DRY_RUN" \
  --field "overwrite=$OVERWRITE" \
  --string-field "command=${COMMAND% }" \
  --string-field "created_at=$(date -Iseconds)" \
  --string-field "runtime_source=$FPMC_DIR"

echo "Generation directory: $GENERATION_ROOT"
CONDOR_ARGS=(
  --condor-dir "$CONDOR_DIR"
  --executable "$JOB_SCRIPT"
  --jobs "$JOBS"
)
[[ "$DRY_RUN" == true ]] && CONDOR_ARGS+=(--dry-run)
python3 "$CONDOR_HELPER" "${CONDOR_ARGS[@]}"
