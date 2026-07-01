#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_processing_condor.sh [--pythia|--delphes|--both] [options]

Stages:
  --pythia                 Run generation-pythia/scripts/process_superchic.py.
  --delphes                Run sim/scripts/run_processes_delphes.py.
  --both                   Run Pythia first, then Delphes. Default if no stage is set.

Shared processing options:
  --process PROCESS        Process name to run. May be repeated. Defaults to all processes.
  --campaign CAMPAIGN      Main campaign name.
  --overwrite              Forward overwrite to selected processing scripts.
  --dry-run                Write Condor artifacts and do not submit.

Tag options:
  --pythia-tag NAME        Pythia subcampaign tag. Also used as Delphes input tag.
  --delphes-tag NAME       Delphes subcampaign tag.

Pythia options:
  --max-events N           Optional event cap per output file.
  --max-files N            Optional SuperChic input file cap per process.
  --seed SEED              Optional Pythia seed.
  --verbose                Pass verbose mode to the Pythia bridge.

Delphes options:
  --card CARD              Delphes card path.

Condor options:
  --condor-tag NAME        Name for sim/condor/NAME. Defaults to processing_TIMESTAMP.
  --request-memory MB      Requested memory in MB. Default: 4096.
  --request-cpus N         Requested CPUs. Default: 1.

Examples:
  ./submit_processing_condor.sh --both --process Hbb --campaign Hbb__test --pythia-tag PythiaTest --delphes-tag DelphesTest --dry-run
  ./submit_processing_condor.sh --pythia --process Hbb --max-events 10 --dry-run
  ./submit_processing_condor.sh --delphes --process Hbb --pythia-tag PythiaTest --delphes-tag DelphesTest --dry-run
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
ORIGINAL_ARGS=("$@")

RUN_PYTHIA=false
RUN_DELPHES=false
STAGE_SET=false
CONDOR_TAG=""
REQUEST_MEMORY=4096
REQUEST_CPUS=1
DRY_RUN=false

PROCESSES=()
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
      PROCESSES+=("$2")
      shift 2
      ;;
    --campaign)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--campaign" "$2")
      DELPHES_ARGS+=("--campaign" "$2")
      shift 2
      ;;
    --pythia-tag)
      [[ $# -ge 2 ]] || usage
      PYTHIA_ARGS+=("--tag" "$2")
      DELPHES_ARGS+=("--pythia-tag" "$2")
      shift 2
      ;;
    --delphes-tag)
      [[ $# -ge 2 ]] || usage
      DELPHES_ARGS+=("--tag" "$2")
      shift 2
      ;;
    --overwrite)
      PYTHIA_ARGS+=("--overwrite")
      DELPHES_ARGS+=("--overwrite")
      shift
      ;;
    --dry-run)
      DRY_RUN=true
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
    --condor-tag)
      [[ $# -ge 2 ]] || usage
      CONDOR_TAG="$2"
      shift 2
      ;;
    --tag)
      echo "ERROR: --tag is ambiguous here; use --condor-tag, --pythia-tag, or --delphes-tag." >&2
      usage
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

if [[ ${#PROCESSES[@]} -eq 0 ]]; then
  mapfile -t PROCESSES < <(
    cd "$STUDY_DIR"
    python3 - <<'PY'
from common.config_utils import load_yaml
for process in load_yaml("processes.yaml"):
    print(process)
PY
  )
fi
if [[ ${#PROCESSES[@]} -eq 0 ]]; then
  echo "ERROR: no processes selected." >&2
  exit 1
fi

if [[ -z "$CONDOR_TAG" ]]; then
  CONDOR_TAG="processing_$(date +%Y%m%d_%H%M%S)"
fi

CONDOR_DIR="$STUDY_DIR/sim/condor/$CONDOR_TAG"
mkdir -p "$CONDOR_DIR"

JOB_SCRIPT="$CONDOR_DIR/run_processing_job.sh"
SUB_FILE="$CONDOR_DIR/submit.sub"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
META_FILE="$CONDOR_DIR/campaign_args.txt"

: > "$QUEUE_FILE"
for process in "${PROCESSES[@]}"; do
  printf '%s\n' "$process" >> "$QUEUE_FILE"
done

PYTHIA_ARGS_QUOTED="$(quote_args "${PYTHIA_ARGS[@]}")"
DELPHES_ARGS_QUOTED="$(quote_args "${DELPHES_ARGS[@]}")"

cat > "$META_FILE" <<EOF
created_at=$(date -Iseconds)
command=$(quote_args "$0" "${ORIGINAL_ARGS[@]}")
condor_tag=$CONDOR_TAG
run_pythia=$RUN_PYTHIA
run_delphes=$RUN_DELPHES
processes=$(quote_args "${PROCESSES[@]}")
request_memory=$REQUEST_MEMORY
request_cpus=$REQUEST_CPUS
pythia_args=$PYTHIA_ARGS_QUOTED
delphes_args=$DELPHES_ARGS_QUOTED
EOF

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

PROCESS="\${1:?missing PROCESS}"

cd $(printf '%q' "$STUDY_DIR")
if command -v cmsenv >/dev/null 2>&1; then
  cmsenv
fi

echo "Batch host: \$(hostname)"
echo "Batch scratch: \$PWD"
echo "Process: \$PROCESS"

EOF

if [[ "$RUN_PYTHIA" == true ]]; then
  cat >> "$JOB_SCRIPT" <<EOF
source env/setup_pythia.sh
PYTHIA_CMD=(python3 generation-pythia/scripts/process_superchic.py --process "\$PROCESS" $PYTHIA_ARGS_QUOTED)
echo "\${PYTHIA_CMD[*]}"
"\${PYTHIA_CMD[@]}"

EOF
fi

if [[ "$RUN_DELPHES" == true ]]; then
  cat >> "$JOB_SCRIPT" <<EOF
source env/setup_delphes.sh
DELPHES_CMD=(python3 sim/scripts/run_processes_delphes.py --process "\$PROCESS" $DELPHES_ARGS_QUOTED)
echo "\${DELPHES_CMD[*]}"
"\${DELPHES_CMD[@]}"

EOF
fi

chmod +x "$JOB_SCRIPT"

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $JOB_SCRIPT
arguments = \$(PROCESS)
output = $CONDOR_DIR/job_\$(PROCESS).out
error = $CONDOR_DIR/job_\$(PROCESS).err
log = $CONDOR_DIR/cluster.log
stream_output = True
stream_error = True
request_memory = $REQUEST_MEMORY
request_cpus = $REQUEST_CPUS
getenv = True
queue PROCESS from $QUEUE_FILE
EOF

echo "Condor directory: $CONDOR_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Queue file: $QUEUE_FILE"
echo "Job script: $JOB_SCRIPT"
echo "Run Pythia: $RUN_PYTHIA"
echo "Run Delphes: $RUN_DELPHES"
echo "Processes: ${PROCESSES[*]}"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

command -v condor_submit >/dev/null 2>&1 || {
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
}
condor_submit "$SUB_FILE"
