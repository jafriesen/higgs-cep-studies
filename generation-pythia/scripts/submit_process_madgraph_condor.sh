#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_process_madgraph_condor.sh [--process PROCESS] [--campaign CAMPAIGN]
    [--tag TAG] [--max-events N] [--max-files N] [--seed SEED]
    [--overwrite] [--verbose] [--condor-tag TAG]
    [--request-memory MB] [--dry-run]

One job is submitted for each LHE input file. If --process is omitted, files
from every process in processes-madgraph.yaml are queued. --process may be
repeated.

Example:
  ./generation-pythia/scripts/submit_process_madgraph_condor.sh \
    --process QCDbb --campaign QCDbb__v02 --max-events 100 --dry-run
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
PROCESS_SCRIPT="$SCRIPT_DIR/process_madgraph.py"

PROCESSES=()
PROCESS_ARGS=()
CAMPAIGN=""
MAX_FILES=""
CONDOR_TAG=""
REQUEST_MEMORY=4096
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process)
      [[ $# -ge 2 ]] || usage
      PROCESSES+=("$2")
      shift 2
      ;;
    --campaign)
      [[ $# -ge 2 ]] || usage
      CAMPAIGN="$2"
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --max-files)
      [[ $# -ge 2 ]] || usage
      MAX_FILES="$2"
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --tag|--max-events|--seed)
      [[ $# -ge 2 ]] || usage
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --overwrite|--verbose)
      PROCESS_ARGS+=("$1")
      shift
      ;;
    --condor-tag)
      [[ $# -ge 2 ]] || usage
      CONDOR_TAG="$2"
      shift 2
      ;;
    --request-memory)
      [[ $# -ge 2 ]] || usage
      REQUEST_MEMORY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
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

if ! [[ "$REQUEST_MEMORY" =~ ^[0-9]+$ ]] || (( REQUEST_MEMORY <= 0 )); then
  echo "ERROR: --request-memory must be a positive integer." >&2
  exit 1
fi
if [[ -n "$MAX_FILES" ]] && { ! [[ "$MAX_FILES" =~ ^[0-9]+$ ]] || (( MAX_FILES <= 0 )); }; then
  echo "ERROR: --max-files must be a positive integer." >&2
  exit 1
fi
if [[ ! -f "$PROCESS_SCRIPT" ]]; then
  echo "ERROR: processing script not found: $PROCESS_SCRIPT" >&2
  exit 1
fi

if [[ ${#PROCESSES[@]} -eq 0 ]]; then
  mapfile -t PROCESSES < <(
    cd "$STUDY_DIR"
    python3 - <<'PY'
from common.path_helper import generation_processes

for process in generation_processes("madgraph"):
    print(process)
PY
  )
fi
if [[ ${#PROCESSES[@]} -eq 0 ]]; then
  echo "ERROR: no MadGraph processes selected." >&2
  exit 1
fi

if [[ -z "$CONDOR_TAG" ]]; then
  CONDOR_TAG="madgraph_pythia_$(date +%Y%m%d_%H%M%S)"
fi

CONDOR_DIR="$STUDY_DIR/generation-pythia/condor/$CONDOR_TAG"
JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
SUB_FILE="$CONDOR_DIR/submit.sub"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
mkdir -p "$CONDOR_DIR"

python3 - "$PROCESS_SCRIPT" "$CAMPAIGN" "$MAX_FILES" "${PROCESSES[@]}" > "$QUEUE_FILE" <<'PY'
import importlib.util
import sys

script, requested_campaign, requested_max_files, *requested_processes = sys.argv[1:]
spec = importlib.util.spec_from_file_location("process_madgraph", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

processes = module.generation_processes("madgraph")
for process_name in module.selected_processes(processes, requested_processes):
    process_config = processes[process_name] or {}
    campaign_name, _ = module.generation_campaign_config(
        "madgraph", process_name, requested_campaign or None
    )
    max_files = (
        int(requested_max_files)
        if requested_max_files
        else module.configured_max_files(process_config)
    )
    input_dir = module.event_records_dir(process_name, campaign_name)
    input_files = module.discover_lhe_files(input_dir, max_files=max_files)
    for file_index in range(1, len(input_files) + 1):
        print(process_name, file_index)
PY
if [[ ! -s "$QUEUE_FILE" ]]; then
  echo "ERROR: no MadGraph LHE files found to queue." >&2
  exit 1
fi
PROCESS_ARGS_QUOTED="$(quote_args "${PROCESS_ARGS[@]}")"

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

PROCESS="\${1:?missing PROCESS}"
FILE_INDEX="\${2:?missing FILE_INDEX}"

cd $(printf '%q' "$STUDY_DIR")
source env/setup_pythia.sh

COMMAND=(python3 -u generation-pythia/scripts/process_madgraph.py --process "\$PROCESS" --file-index "\$FILE_INDEX" $PROCESS_ARGS_QUOTED)
echo "\${COMMAND[*]}"
"\${COMMAND[@]}"
EOF
chmod +x "$JOB_SCRIPT"

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $JOB_SCRIPT
arguments = \$(PROCESS) \$(FILE_INDEX)
output = $CONDOR_DIR/job_\$(PROCESS)_\$(FILE_INDEX).out
error = $CONDOR_DIR/job_\$(PROCESS)_\$(FILE_INDEX).err
log = $CONDOR_DIR/cluster.log
request_memory = $REQUEST_MEMORY
request_cpus = 1
getenv = True
queue PROCESS, FILE_INDEX from $QUEUE_FILE
EOF

echo "Condor directory: $CONDOR_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Processes: ${PROCESSES[*]}"
echo "Jobs: $(wc -l < "$QUEUE_FILE")"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

command -v condor_submit >/dev/null 2>&1 || {
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
}
condor_submit "$SUB_FILE"
