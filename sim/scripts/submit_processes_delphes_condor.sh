#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_processes_delphes_condor.sh [--generator superchic|fpmc|madgraph]
    [--process PROCESS] [--campaign CAMPAIGN] [--pythia-tag TAG]
    [--delphes-tag TAG] [--max-files N] [--card CARD] [--overwrite]
    [--condor-tag TAG] [--request-memory MB] [--dry-run]

One job is submitted for each HepMC input file. If --process is omitted, files
from every configured process for the selected generator are queued.

Example:
  ./sim/scripts/submit_processes_delphes_condor.sh \
    --generator madgraph --process QCDbb --campaign QCDbb__v02 --dry-run
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
PROCESS_SCRIPT="$SCRIPT_DIR/run_processes_delphes.py"

GENERATOR="fpmc"
PROCESSES=()
PROCESS_ARGS=()
CAMPAIGN=""
PYTHIA_TAG=""
MAX_FILES=""
CONDOR_TAG=""
REQUEST_MEMORY=4096
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generator)
      [[ $# -ge 2 ]] || usage
      GENERATOR="$2"
      shift 2
      ;;
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
    --pythia-tag)
      [[ $# -ge 2 ]] || usage
      PYTHIA_TAG="$2"
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --max-files)
      [[ $# -ge 2 ]] || usage
      MAX_FILES="$2"
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --tag|--delphes-tag|--card)
      [[ $# -ge 2 ]] || usage
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --overwrite)
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

if [[ "$GENERATOR" != "superchic" && "$GENERATOR" != "fpmc" && "$GENERATOR" != "madgraph" ]]; then
  echo "ERROR: --generator must be superchic, fpmc, or madgraph." >&2
  exit 1
fi
if ! [[ "$REQUEST_MEMORY" =~ ^[0-9]+$ ]] || (( REQUEST_MEMORY <= 0 )); then
  echo "ERROR: --request-memory must be a positive integer." >&2
  exit 1
fi
if [[ -n "$MAX_FILES" ]] && { ! [[ "$MAX_FILES" =~ ^[0-9]+$ ]] || (( MAX_FILES <= 0 )); }; then
  echo "ERROR: --max-files must be a positive integer." >&2
  exit 1
fi
if [[ ! -f "$PROCESS_SCRIPT" ]]; then
  echo "ERROR: Delphes processing script not found: $PROCESS_SCRIPT" >&2
  exit 1
fi

if [[ ${#PROCESSES[@]} -eq 0 ]]; then
  mapfile -t PROCESSES < <(
    cd "$STUDY_DIR"
    python3 - "$GENERATOR" <<'PY'
import sys

from common.path_helper import generation_processes

for process in generation_processes(sys.argv[1]):
    print(process)
PY
  )
fi
if [[ ${#PROCESSES[@]} -eq 0 ]]; then
  echo "ERROR: no processes selected." >&2
  exit 1
fi

if [[ -z "$CONDOR_TAG" ]]; then
  CONDOR_TAG="delphes_${GENERATOR}_$(date +%Y%m%d_%H%M%S)"
fi

CONDOR_DIR="$STUDY_DIR/sim/condor/$CONDOR_TAG"
JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
SUB_FILE="$CONDOR_DIR/submit.sub"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
mkdir -p "$CONDOR_DIR"

python3 - "$PROCESS_SCRIPT" "$GENERATOR" "$CAMPAIGN" "$PYTHIA_TAG" "$MAX_FILES" "${PROCESSES[@]}" > "$QUEUE_FILE" <<'PY'
import importlib.util
import sys

script, generator, requested_campaign, requested_pythia_tag, requested_max_files, *requested_processes = sys.argv[1:]
spec = importlib.util.spec_from_file_location("run_processes_delphes", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

processes = module.generation_processes(generator)
for process_name in module.selected_processes(processes, requested_processes):
    process_cfg = processes[process_name] or {}
    campaign_name, _ = module.generation_campaign_config(
        generator, process_name, requested_campaign or None
    )
    pythia_tag = requested_pythia_tag or module.default_subcampaign(
        process_cfg, "hadr-pythia"
    )
    input_dir = module.input_hepmc_dir(
        generator, process_name, campaign_name, pythia_tag
    )
    max_files = int(requested_max_files) if requested_max_files else None
    input_files = module.discover_hepmc_files(input_dir, max_files=max_files)
    for file_index in range(1, len(input_files) + 1):
        print(process_name, file_index)
PY
if [[ ! -s "$QUEUE_FILE" ]]; then
  echo "ERROR: no HepMC files found to queue." >&2
  exit 1
fi
PROCESS_ARGS_QUOTED="$(quote_args "${PROCESS_ARGS[@]}")"

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

PROCESS="\${1:?missing PROCESS}"
FILE_INDEX="\${2:?missing FILE_INDEX}"

cd $(printf '%q' "$STUDY_DIR")
source env/setup_delphes.sh

COMMAND=(python3 -u sim/scripts/run_processes_delphes.py --generator $(printf '%q' "$GENERATOR") --process "\$PROCESS" --file-index "\$FILE_INDEX" $PROCESS_ARGS_QUOTED)
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
echo "Generator: $GENERATOR"
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
