#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_pythia8_minbias_condor.sh --total-bx TOTAL --jobs N_JOBS [--mu MU] \
    [--mu-mode fixed|poisson] [--e-cm ECM_GEV] [--processes "SoftQCD:all"] \
    [--seed-base BASE] [--campaign NAME] [--output-base DIR] [--store-tracks] \
    [--dry-run] [--no-root]

Examples:
  ./submit_pythia8_minbias_condor.sh --total-bx 10000 --jobs 50 --mu 200 --mu-mode poisson
  ./submit_pythia8_minbias_condor.sh --total-bx 10000 --jobs 50 --campaign mb_scan_a
USAGE
  exit 1
}

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$RUN_SCRIPT_DIR/../.." && pwd)"
CMSSW_BASE_DEFAULT="$(cd "$STUDY_DIR/../.." && pwd)"

TOTAL_BX=""
TOTAL_JOBS=""
MU=200
MU_MODE="fixed"
E_CM=14000.0
PROCESSES="SoftQCD:all"
SEED_BASE=1000
CAMPAIGN=""
OUTPUT_BASE="$STUDY_DIR/bkg-generation/output"
DRY_RUN=false
BUILD_ROOT=true
STORE_TRACKS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-bx)
      TOTAL_BX="$2"
      shift 2
      ;;
    --jobs)
      TOTAL_JOBS="$2"
      shift 2
      ;;
    --mu)
      MU="$2"
      shift 2
      ;;
    --mu-mode)
      MU_MODE="$2"
      shift 2
      ;;
    --e-cm)
      E_CM="$2"
      shift 2
      ;;
    --processes)
      PROCESSES="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
      ;;
    --campaign)
      CAMPAIGN="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-root)
      BUILD_ROOT=false
      shift
      ;;
    --store-tracks)
      STORE_TRACKS=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "$TOTAL_BX" || -z "$TOTAL_JOBS" ]]; then
  usage
fi

if (( TOTAL_BX <= 0 )); then
  echo "ERROR: --total-bx must be > 0" >&2
  exit 1
fi
if (( TOTAL_JOBS <= 0 )); then
  echo "ERROR: --jobs must be > 0" >&2
  exit 1
fi
if (( MU <= 0 )); then
  echo "ERROR: --mu must be > 0" >&2
  exit 1
fi
if [[ "$MU_MODE" != "fixed" && "$MU_MODE" != "poisson" ]]; then
  echo "ERROR: --mu-mode must be fixed or poisson" >&2
  exit 1
fi
if (( SEED_BASE < 0 )); then
  echo "ERROR: --seed-base must be >= 0" >&2
  exit 1
fi

if [[ "$OUTPUT_BASE" != /* ]]; then
  OUTPUT_BASE="$STUDY_DIR/$OUTPUT_BASE"
fi

if [[ -z "$CAMPAIGN" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  CAMPAIGN="minbias_nbx${TOTAL_BX}_j${TOTAL_JOBS}_mu${MU}_${MU_MODE}_${TS}"
fi

CAMPAIGN_DIR="$OUTPUT_BASE/$CAMPAIGN"
CONDOR_DIR="$CAMPAIGN_DIR/condor"
mkdir -p "$CONDOR_DIR"

BATCH_SCRIPT="$RUN_SCRIPT_DIR/run_pythia8_minbias_batch_job.sh"
if [[ ! -x "$BATCH_SCRIPT" ]]; then
  chmod +x "$BATCH_SCRIPT"
fi

SUB_FILE="$CONDOR_DIR/submit.sub"
META_FILE="$CONDOR_DIR/campaign_args.txt"
NO_ROOT_ARG=""
if [[ "$BUILD_ROOT" == false ]]; then
  NO_ROOT_ARG="--no-root"
fi
STORE_TRACKS_ARG=""
if [[ "$STORE_TRACKS" == true ]]; then
  STORE_TRACKS_ARG="--store-tracks"
fi

cat > "$META_FILE" <<EOF
campaign=$CAMPAIGN
total_bx=$TOTAL_BX
jobs=$TOTAL_JOBS
mu=$MU
mu_mode=$MU_MODE
e_cm=$E_CM
processes=$PROCESSES
seed_base=$SEED_BASE
output_base=$OUTPUT_BASE
build_root=$BUILD_ROOT
store_tracks=$STORE_TRACKS
EOF

cat > "$SUB_FILE" <<EOF
universe = vanilla
executable = $BATCH_SCRIPT
arguments = --job-index \$(Process) --total-bx $TOTAL_BX --jobs $TOTAL_JOBS --mu $MU --mu-mode $MU_MODE --e-cm $E_CM --processes $PROCESSES --seed-base $SEED_BASE --campaign $CAMPAIGN --output-base $OUTPUT_BASE $NO_ROOT_ARG $STORE_TRACKS_ARG
output = $CONDOR_DIR/job_\$(Process).out
error = $CONDOR_DIR/job_\$(Process).err
log = $CONDOR_DIR/cluster.log
stream_output = True
stream_error = True
request_memory = 2048
request_cpus = 1
getenv = True
environment = "SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12};CMSSW_BASE=${CMSSW_BASE:-$CMSSW_BASE_DEFAULT}"
queue $TOTAL_JOBS
EOF

echo "Campaign directory: $CAMPAIGN_DIR"
echo "Condor submit file: $SUB_FILE"

BASE_BX=$(( TOTAL_BX / TOTAL_JOBS ))
REMAINDER=$(( TOTAL_BX % TOTAL_JOBS ))
SPLIT_BOUNDARY=$(( TOTAL_JOBS - REMAINDER ))
echo "Split summary: base=${BASE_BX}, remainder=${REMAINDER} (extra BX to last ${REMAINDER} jobs)"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run requested, not submitting."
  exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
  echo "ERROR: condor_submit not found in PATH." >&2
  exit 1
fi

condor_submit "$SUB_FILE"
