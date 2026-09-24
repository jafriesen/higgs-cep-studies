#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_processes_delphes_pythia8_condor.sh --generator superchic|madgraph
    [--process PROCESS] [--campaign CAMPAIGN] [--delphes-tag TAG]
    [--cmnd-template PATH] [--fsr on|off] [--seed SEED]
    [--max-events N] [--events-per-job N] [--max-files N]
    [--card CARD] [--overwrite] [--condor-tag TAG] [--request-memory MB]
    [--dry-run]

By default, one job is submitted for each generator LHE input file. With
--events-per-job, the --max-events cap for each file is split across jobs.
Each job runs DelphesPythia8 (Pythia hadronization in-process, no HepMC
intermediate). --events-per-job requires --max-events.
If --process is omitted, files from every configured process for the
selected generator are queued.

Example:
  ./sim/scripts/submit_processes_delphes_pythia8_condor.sh \
    --generator madgraph --process QCDbb --delphes-tag QCDbb_DPy8__v01 \
    --max-events 10000 --events-per-job 2000 --dry-run
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
PROCESS_SCRIPT="$SCRIPT_DIR/run_processes_delphes_pythia8.py"

GENERATOR=""
PROCESSES=()
PROCESS_ARGS=()
CAMPAIGN=""
MAX_FILES=""
MAX_EVENTS=""
EVENTS_PER_JOB=""
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
    --max-files)
      [[ $# -ge 2 ]] || usage
      MAX_FILES="$2"
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --max-events)
      [[ $# -ge 2 ]] || usage
      MAX_EVENTS="$2"
      shift 2
      ;;
    --events-per-job)
      [[ $# -ge 2 ]] || usage
      EVENTS_PER_JOB="$2"
      shift 2
      ;;
    --tag|--delphes-tag|--card|--cmnd-template|--fsr|--seed|--filter-activity-max)
      [[ $# -ge 2 ]] || usage
      PROCESS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --overwrite|--filter)
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

if [[ "$GENERATOR" != "superchic" && "$GENERATOR" != "madgraph" ]]; then
  echo "ERROR: --generator must be superchic or madgraph." >&2
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
if [[ -n "$MAX_EVENTS" ]] && { ! [[ "$MAX_EVENTS" =~ ^[0-9]+$ ]] || (( MAX_EVENTS <= 0 )); }; then
  echo "ERROR: --max-events must be a positive integer." >&2
  exit 1
fi
if [[ -n "$EVENTS_PER_JOB" ]] && { ! [[ "$EVENTS_PER_JOB" =~ ^[0-9]+$ ]] || (( EVENTS_PER_JOB <= 0 )); }; then
  echo "ERROR: --events-per-job must be a positive integer." >&2
  exit 1
fi
if [[ -n "$EVENTS_PER_JOB" && -z "$MAX_EVENTS" ]]; then
  echo "ERROR: --events-per-job requires --max-events." >&2
  exit 1
fi
if [[ -n "$MAX_EVENTS" && -z "$EVENTS_PER_JOB" ]]; then
  PROCESS_ARGS+=(--max-events "$MAX_EVENTS")
fi
if [[ ! -f "$PROCESS_SCRIPT" ]]; then
  echo "ERROR: DelphesPythia8 processing script not found: $PROCESS_SCRIPT" >&2
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
  CONDOR_TAG="delphes_py8_${GENERATOR}_$(date +%Y%m%d_%H%M%S)"
fi

CONDOR_DIR="$STUDY_DIR/sim/condor/$CONDOR_TAG"
JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
SUB_FILE="$CONDOR_DIR/submit.sub"
QUEUE_FILE="$CONDOR_DIR/queue_items.txt"
mkdir -p "$CONDOR_DIR"

python3 - "$PROCESS_SCRIPT" "$GENERATOR" "$CAMPAIGN" "$MAX_FILES" "$MAX_EVENTS" "$EVENTS_PER_JOB" "${PROCESSES[@]}" > "$QUEUE_FILE" <<'PY'
import importlib.util
import sys

(
    script,
    generator,
    requested_campaign,
    requested_max_files,
    requested_max_events,
    requested_events_per_job,
    *requested_processes,
) = sys.argv[1:]
spec = importlib.util.spec_from_file_location("run_processes_delphes_pythia8", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

processes = module.generation_processes(generator)
for process_name in module.base.selected_processes(processes, requested_processes):
    campaign_name, _ = module.generation_campaign_config(
        generator, process_name, requested_campaign or None
    )
    input_dir = module.gen_input_dir(generator, process_name, campaign_name)
    max_files = int(requested_max_files) if requested_max_files else None
    input_files = module.discover_event_files(input_dir, max_files=max_files)
    chunks = (
        module.event_chunks(int(requested_max_events), int(requested_events_per_job))
        if requested_events_per_job
        else [(0, 0, 0)]
    )
    for file_index in range(1, len(input_files) + 1):
        for part_index, event_offset, event_count in chunks:
            print(process_name, file_index, part_index, event_offset, event_count)
PY
if [[ ! -s "$QUEUE_FILE" ]]; then
  echo "ERROR: no generator input files found to queue." >&2
  exit 1
fi
PROCESS_ARGS_QUOTED="$(quote_args "${PROCESS_ARGS[@]}")"

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash

set -euo pipefail

PROCESS="\${1:?missing PROCESS}"
FILE_INDEX="\${2:?missing FILE_INDEX}"
PART_INDEX="\${3:?missing PART_INDEX}"
EVENT_OFFSET="\${4:?missing EVENT_OFFSET}"
EVENT_COUNT="\${5:?missing EVENT_COUNT}"

cd $(printf '%q' "$STUDY_DIR")
source env/setup_delphes.sh

PART_ARGS=()
if (( PART_INDEX > 0 )); then
  PART_ARGS=(--part-index "\$PART_INDEX" --skip-events "\$EVENT_OFFSET" --max-events "\$EVENT_COUNT")
fi

COMMAND=(python3 -u sim/scripts/run_processes_delphes_pythia8.py --generator $(printf '%q' "$GENERATOR") --process "\$PROCESS" --file-index "\$FILE_INDEX" $PROCESS_ARGS_QUOTED "\${PART_ARGS[@]}")
echo "\${COMMAND[*]}"
"\${COMMAND[@]}"
EOF
chmod +x "$JOB_SCRIPT"

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $JOB_SCRIPT
arguments = \$(PROCESS) \$(FILE_INDEX) \$(PART_INDEX) \$(EVENT_OFFSET) \$(EVENT_COUNT)
output = $CONDOR_DIR/job_\$(PROCESS)_\$(FILE_INDEX)_\$(PART_INDEX).out
error = $CONDOR_DIR/job_\$(PROCESS)_\$(FILE_INDEX)_\$(PART_INDEX).err
log = $CONDOR_DIR/cluster.log
request_memory = $REQUEST_MEMORY
request_cpus = 1
getenv = True
queue PROCESS, FILE_INDEX, PART_INDEX, EVENT_OFFSET, EVENT_COUNT from $QUEUE_FILE
EOF

echo "Condor directory: $CONDOR_DIR"
echo "Condor submit file: $SUB_FILE"
echo "Generator: $GENERATOR"
echo "Processes: ${PROCESSES[*]}"
if [[ -n "$EVENTS_PER_JOB" ]]; then
  echo "Event split: max-events=$MAX_EVENTS, events-per-job=$EVENTS_PER_JOB"
fi
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
