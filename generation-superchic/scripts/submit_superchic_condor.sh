#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_superchic_condor.sh --process PROCESS --campaign CAMPAIGN
    [--nev-per-job EVENTS] [--jobs N] [--job-offset N] [--card DAT_FILE]
    [--init] [--overwrite] [--dry-run]
    [--survival-model MODEL] [--no-soft-survival]
    [--mass-min GEV] [--mass-max GEV]
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_HELPER="$STUDY_DIR/common/path_helper.py"
METADATA_WRITER="$STUDY_DIR/common/write_metadata.py"
CONDOR_HELPER="$STUDY_DIR/common/submit_condor.py"
RUN_SCRIPT="$SCRIPT_DIR/run_superchic.sh"
INIT_SCRIPT="$SCRIPT_DIR/prepare_superchic_init.sh"
CARD_GENERATOR="$SCRIPT_DIR/generate_card.py"
DEFAULT_CARD="$STUDY_DIR/generation-superchic/cards/template.DAT"
ORIGINAL_ARGS=("$@")

PROCESS=""
CAMPAIGN=""
JOBS=100
JOB_OFFSET=0
NEV_PER_JOB=2000
CARD=""
RUN_INIT=false
OVERWRITE=false
DRY_RUN=false
SURVIVAL_MODEL=""
NO_SOFT_SURVIVAL=false
MASS_MIN=""
MASS_MAX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --process) PROCESS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --job-offset) JOB_OFFSET="$2"; shift 2 ;;
    --nev-per-job) NEV_PER_JOB="$2"; shift 2 ;;
    --card) CARD="$2"; shift 2 ;;
    --init) RUN_INIT=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --survival-model) SURVIVAL_MODEL="$2"; shift 2 ;;
    --no-soft-survival) NO_SOFT_SURVIVAL=true; shift ;;
    --mass-min) MASS_MIN="$2"; shift 2 ;;
    --mass-max) MASS_MAX="$2"; shift 2 ;;
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
[[ "$JOB_OFFSET" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --job-offset must be a non-negative integer." >&2
  exit 1
}
if (( JOB_OFFSET > 0 )) && [[ "$OVERWRITE" == true ]]; then
  echo "ERROR: --job-offset cannot be combined with --overwrite." >&2
  echo "       --overwrite deletes the event records of the earlier waves." >&2
  exit 1
fi
if [[ -n "$SURVIVAL_MODEL" ]] &&
   { ! [[ "$SURVIVAL_MODEL" =~ ^[1-4]$ ]]; }; then
  echo "ERROR: --survival-model must be one of 1, 2, 3, or 4." >&2
  exit 1
fi
CARD="${CARD:-$DEFAULT_CARD}"
[[ "$CARD" == /* ]] || CARD="$STUDY_DIR/$CARD"
[[ -f "$CARD" ]] || { echo "ERROR: card not found: $CARD" >&2; exit 1; }
CARD_ARGS=()
[[ -n "$SURVIVAL_MODEL" ]] && CARD_ARGS+=(--survival-model "$SURVIVAL_MODEL")
[[ "$NO_SOFT_SURVIVAL" == true ]] && CARD_ARGS+=(--no-soft-survival)
[[ -n "$MASS_MIN" ]] && CARD_ARGS+=(--mass-min "$MASS_MIN")
[[ -n "$MASS_MAX" ]] && CARD_ARGS+=(--mass-max "$MASS_MAX")
if [[ -n "$MASS_MIN" && -n "$MASS_MAX" ]]; then
  awk -v low="$MASS_MIN" -v high="$MASS_MAX" 'BEGIN { exit !(low < high) }' || {
    echo "ERROR: --mass-min must be below --mass-max" >&2
    exit 1
  }
fi

source "$STUDY_DIR/env/setup_superchic.sh"
eval "$(python3 "$PATH_HELPER" generation-env \
  --generator superchic --process "$PROCESS" --campaign "$CAMPAIGN")"

SUPERCHIC_OUTPUT_DIR="$GENERATION_ROOT/output"
INIT_INPUTS_DIR="$GENERATION_ROOT/init/inputs"

# A job index sets that job's seed, so a later wave asks for offset indices to get
# new events. Such a wave keeps its own condor and metadata files, leaving the
# records of the earlier waves in place.
if (( JOB_OFFSET > 0 )); then
  WAVE_SUFFIX="_jobs$(( JOB_OFFSET + 1 ))-$(( JOB_OFFSET + JOBS ))"
  CONDOR_DIR="${CONDOR_DIR}${WAVE_SUFFIX}"
  METADATA_FILE="$GENERATION_ROOT/metadata${WAVE_SUFFIX}.yaml"
fi

if [[ -d "$CONDOR_DIR" && "$OVERWRITE" != true ]]; then
  echo "ERROR: submit destination already exists: $CONDOR_DIR" >&2
  echo "Use --overwrite to replace non-init generation outputs." >&2
  exit 1
fi
if [[ "$OVERWRITE" != true ]]; then
  for (( job = JOB_OFFSET + 1; job <= JOB_OFFSET + JOBS; job++ )); do
    tag="${CAMPAIGN}_${job}"
    for output in \
      "$EVENT_RECORDS_DIR/evrec${tag}.dat" \
      "$SUPERCHIC_OUTPUT_DIR/output${tag}.dat" \
      "$CARDS_DIR/job_${tag}.DAT" \
      "$LOGS_DIR/run_${tag}.log"; do
      [[ ! -e "$output" ]] || {
        echo "ERROR: expected job output already exists: $output" >&2
        exit 1
      }
    done
  done
fi
if [[ "$OVERWRITE" == true ]]; then
  rm -rf "$CONDOR_DIR" "$EVENT_RECORDS_DIR" "$SUPERCHIC_OUTPUT_DIR" "$CARDS_DIR"
  if [[ -d "$LOGS_DIR" ]]; then
    find "$LOGS_DIR" -maxdepth 1 -type f -name 'run_*.log' -delete
  fi
fi
INIT_ARGS=(--process "$PROCESS" --campaign "$CAMPAIGN" --card "$CARD")
[[ -n "$SURVIVAL_MODEL" ]] && INIT_ARGS+=(--survival-model "$SURVIVAL_MODEL")
if [[ "$RUN_INIT" == true && "$NO_SOFT_SURVIVAL" != true ]]; then
  "$INIT_SCRIPT" "${INIT_ARGS[@]}"
fi
INIT_CARD_CONTENT="$(
  python3 "$CARD_GENERATOR" \
    --template "$CARD" --process "$PROCESS" --nev 1 --seed 1 \
    --out-tag "${CAMPAIGN}_init" \
    "${CARD_ARGS[@]}"
)"
INIT_RTS=$(printf '%s\n' "$INIT_CARD_CONTENT" | awk '/\[rts\]/ {print $1; exit}')
INIT_ISURV=$(printf '%s\n' "$INIT_CARD_CONTENT" | awk '/\[isurv\]/ {print $1; exit}')
INIT_INTAG=$(printf '%s\n' "$INIT_CARD_CONTENT" | awk '/\[intag\]/ {gsub(/'\''/, "", $1); print $1; exit}')
INIT_PDFNAME=$(printf '%s\n' "$INIT_CARD_CONTENT" | awk '/\[PDFname\]/ {gsub(/'\''/, "", $1); print $1; exit}')
INIT_PDFMEMBER=$(printf '%s\n' "$INIT_CARD_CONTENT" | awk '/\[PDFmember\]/ {print $1; exit}')
SOFT_SURVIVAL_VALUE=$(printf '%s\n' "$INIT_CARD_CONTENT" | awk '/\[sfaci\]/ {print $1; exit}')
SOFT_SURVIVAL_ENABLED=true
[[ "$SOFT_SURVIVAL_VALUE" == ".false." ]] && SOFT_SURVIVAL_ENABLED=false
INIT_SCREEN_FILE="screening${INIT_INTAG}.dat"
INIT_KEY="$(printf '%s|%s|%s|%s|%s\n' \
  "$INIT_RTS" "$INIT_ISURV" "$INIT_INTAG" "$INIT_PDFNAME" "$INIT_PDFMEMBER" |
  sha1sum | awk '{print $1}')"
if [[ "$NO_SOFT_SURVIVAL" != true ]]; then
  if [[ ! -f "$INIT_INPUTS_DIR/$INIT_SCREEN_FILE" ||
        ! -f "$GENERATION_ROOT/init/init_key.txt" ||
        "$(cat "$GENERATION_ROOT/init/init_key.txt" 2>/dev/null || true)" != "$INIT_KEY" ]]; then
    echo "ERROR: matching initialized SuperChic inputs are not available." >&2
    echo "Pass --init or run prepare_superchic_init.sh first." >&2
    exit 1
  fi
fi
mkdir -p \
  "$CONDOR_DIR" "$EVENT_RECORDS_DIR" "$SUPERCHIC_OUTPUT_DIR" "$CARDS_DIR" "$LOGS_DIR"

JOB_SCRIPT="$CONDOR_DIR/run_job.sh"
WORKER_CARD_ARGS=""
for arg in "${CARD_ARGS[@]}"; do
  printf -v QUOTED_ARG '%q' "$arg"
  WORKER_CARD_ARGS+=" \\
  $QUOTED_ARG"
done

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
JOB_INDEX="\${1:?missing JOB_INDEX}"
export HIGGS_CEP_CONDOR_JOB=1
export HIGGS_CEP_SUPERCHIC_DIR=$(printf '%q' "$SUPERCHIC_DIR")
export HIGGS_CEP_SUPERCHIC_INSTALL_DIR=$(printf '%q' "$SUPERCHIC_INSTALL_DIR")
export HIGGS_CEP_SUPERCHIC_BUILD_DIR=$(printf '%q' "$SUPERCHIC_BUILD_DIR")
exec $(printf '%q' "$RUN_SCRIPT") \
  --process $(printf '%q' "$PROCESS") \
  --campaign $(printf '%q' "$CAMPAIGN") \
  --card $(printf '%q' "$CARD") \
  --nev "$NEV_PER_JOB" \
  --job "\$JOB_INDEX"$WORKER_CARD_ARGS
EOF
chmod +x "$JOB_SCRIPT"

printf -v COMMAND '%q ' "$0" "${ORIGINAL_ARGS[@]}"
MASS_METADATA_ARGS=()
[[ -n "$MASS_MIN" ]] && MASS_METADATA_ARGS+=(--field "mass_min_gev=$MASS_MIN")
[[ -n "$MASS_MAX" ]] && MASS_METADATA_ARGS+=(--field "mass_max_gev=$MASS_MAX")
python3 "$METADATA_WRITER" \
  --output "$METADATA_FILE" \
  --string-field "generator=superchic" \
  --string-field "process=$PROCESS" \
  --string-field "campaign=$CAMPAIGN" \
  --string-field "mode=condor" \
  --field "jobs=$JOBS" \
  --field "events_per_job=$NEV_PER_JOB" \
  --field "seed_start=$(( 1001 + JOB_OFFSET ))" \
  --field "job_offset=$JOB_OFFSET" \
  --field "init=$RUN_INIT" \
  --field "dry_run=$DRY_RUN" \
  --field "overwrite=$OVERWRITE" \
  --field "survival_model=$INIT_ISURV" \
  --field "soft_survival=$SOFT_SURVIVAL_ENABLED" \
  "${MASS_METADATA_ARGS[@]}" \
  --string-field "card=$CARD" \
  --string-field "command=${COMMAND% }" \
  --string-field "created_at=$(date -Iseconds)" \
  --string-field "runtime_source=$SUPERCHIC_DIR"

echo "Generation directory: $GENERATION_ROOT"
CONDOR_ARGS=(
  --condor-dir "$CONDOR_DIR"
  --executable "$JOB_SCRIPT"
  --jobs "$JOBS"
  --job-offset "$JOB_OFFSET"
)
[[ "$DRY_RUN" == true ]] && CONDOR_ARGS+=(--dry-run)
python3 "$CONDOR_HELPER" "${CONDOR_ARGS[@]}"
