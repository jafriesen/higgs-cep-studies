#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_processing_condor.sh [--pythia|--delphes|--both] [options]

Stages:
  --pythia                 Run generation-pythia/scripts/run_processes_pythia.py.
  --delphes                Run sim/scripts/run_processes_delphes.py.
  --both                   Run Pythia first, then Delphes. Default if no stage is set.

Shared processing options:
  --process PROCESS        Process name to run. May be repeated.
  --campaign CAMPAIGN      Campaign key to use for selected processes.
  --overwrite              Overwrite existing outputs.
  --dry-run                Print processing commands and do not submit to Condor.

Pythia options:
  --max-events N           Optional event cap per output file.
  --max-files N            Optional SuperChic input file cap per process.
  --seed SEED              Optional Pythia seed.
  --verbose                Pass verbose mode to the Pythia bridge.

Delphes options:
  --card CARD              Delphes card path.

Condor options:
  --tag NAME               Name for sim/condor/NAME. Defaults to processing_TIMESTAMP.
  --request-memory MB      Requested memory in MB. Default: 4096.
  --request-cpus N         Requested CPUs. Default: 1.

Examples:
  ./submit_processing_condor.sh --both --process h_bb --dry-run
  ./submit_processing_condor.sh --pythia --max-events 10 --dry-run
  ./submit_processing_condor.sh --delphes --card /path/to/card.tcl --dry-run
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
STUDY_DIR="$SCRIPT_DIR"

RUN_PYTHIA=false
RUN_DELPHES=false
STAGE_SET=false
TAG=""
REQUEST_MEMORY=4096
REQUEST_CPUS=1
DRY_RUN=false

PYTHIA_ARGS=()
DELPHES_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pythia)
      RUN_PYTHIA=true
      STAGE_SET=true
      shift
      ;;
    --delphes)
      RUN_DELPHES=true
      STAGE_SET=true
      shift
      ;;
    --both)
      RUN_PYTHIA=true
      RUN_DELPHES=true
      STAGE_SET=true
      shift
      ;;
    --process)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--process" "$2")
      DELPHES_ARGS+=("--process" "$2")
      shift 2
      ;;
    --campaign)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--campaign" "$2")
      DELPHES_ARGS+=("--campaign" "$2")
      shift 2
      ;;
    --overwrite)
      PYTHIA_ARGS+=("--overwrite")
      DELPHES_ARGS+=("--overwrite")
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      PYTHIA_ARGS+=("--dry-run")
      DELPHES_ARGS+=("--dry-run")
      shift
      ;;
    --max-events)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--max-events" "$2")
      shift 2
      ;;
    --max-files)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--max-files" "$2")
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--seed" "$2")
      shift 2
      ;;
    --verbose)
      PYTHIA_ARGS+=("--verbose")
      shift
      ;;
    --card)
      [[ $# -ge 2 ]] || usage
      DELPHES_ARGS+=("--card" "$2")
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || usage
      TAG="$2"
      shift 2
      ;;
    --request-memory)
      [[ $# -ge 2 ]] || usage
      REQUEST_MEMORY="$2"
      shift 2
      ;;
    --request-cpus)
      [[ $# -ge 2 ]] || usage
      REQUEST_CPUS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ "$STAGE_SET" == false ]]; then
  RUN_PYTHIA=true
  RUN_DELPHES=true
fi

if [[ "$RUN_PYTHIA" == false && "$RUN_DELPHES" == false ]]; then
  echo "ERROR: at least one of --pythia, --delphes, or --both must be selected." >&2
  exit 1
fi
if ! [[ "$REQUEST_MEMORY" =~ ^[0-9]+$ ]] || (( REQUEST_MEMORY <= 0 )); then
  echo "ERROR: --request-memory must be a positive integer." >&2
  exit 1
fi
if ! [[ "$REQUEST_CPUS" =~ ^[0-9]+$ ]] || (( REQUEST_CPUS <= 0 )); then
  echo "ERROR: --request-cpus must be a positive integer." >&2
  exit 1
fi

if [[ -z "$TAG" ]]; then
  TAG="processing_$(date +%Y%m%d_%H%M%S)"
fi

CONDOR_DIR="$STUDY_DIR/sim/condor/$TAG"
mkdir -p "$CONDOR_DIR"

JOB_SCRIPT="$CONDOR_DIR/run_processing_job.sh"
SUB_FILE="$CONDOR_DIR/submit.sub"
META_FILE="$CONDOR_DIR/campaign_args.txt"

PYTHIA_COMMAND=""
DELPHES_COMMAND=""
if [[ "$RUN_PYTHIA" == true ]]; then
  PYTHIA_COMMAND="python3 generation-pythia/scripts/run_processes_pythia.py $(quote_args "${PYTHIA_ARGS[@]}")"
fi
if [[ "$RUN_DELPHES" == true ]]; then
  DELPHES_COMMAND="python3 sim/scripts/run_processes_delphes.py $(quote_args "${DELPHES_ARGS[@]}")"
fi

cat > "$META_FILE" <<EOF
tag=$TAG
run_pythia=$RUN_PYTHIA
run_delphes=$RUN_DELPHES
request_memory=$REQUEST_MEMORY
request_cpus=$REQUEST_CPUS
pythia_args=$(quote_args "${PYTHIA_ARGS[@]}")
delphes_args=$(quote_args "${DELPHES_ARGS[@]}")
EOF

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

cd "$STUDY_DIR"
source setup_env.sh
if command -v cmsenv >/dev/null 2>&1; then
  cmsenv
fi

EOF

if [[ "$RUN_PYTHIA" == true ]]; then
  cat >> "$JOB_SCRIPT" <<EOF
echo "$PYTHIA_COMMAND"
$PYTHIA_COMMAND

EOF
fi

if [[ "$RUN_DELPHES" == true ]]; then
  cat >> "$JOB_SCRIPT" <<EOF
echo "$DELPHES_COMMAND"
$DELPHES_COMMAND

EOF
fi

chmod +x "$JOB_SCRIPT"

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $JOB_SCRIPT
output = $CONDOR_DIR/processing.out
error = $CONDOR_DIR/processing.err
log = $CONDOR_DIR/cluster.log
stream_output = True
stream_error = True
request_memory = $REQUEST_MEMORY
request_cpus = $REQUEST_CPUS
getenv = True
queue 1
EOF

echo "Condor directory: $CONDOR_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Job script: $JOB_SCRIPT"
echo "Run Pythia: $RUN_PYTHIA"
echo "Run Delphes: $RUN_DELPHES"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

command -v condor_submit >/dev/null 2>&1 || {
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
}
condor_submit "$SUB_FILE"
