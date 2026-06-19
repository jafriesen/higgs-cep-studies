#!/usr/bin/env bash

set -euo pipefail

run_cmd() {
  printf "+ "
  printf "%q " "$@"
  echo
  "$@"
}

usage() {
  cat <<'USAGE'
Usage:
  run_pythia8_minbias_batch_job.sh --job-index IDX --total-bx TOTAL --jobs N_JOBS \
    [--mu MU] [--mu-mode fixed|poisson] [--e-cm ECM_GEV] [--processes "SoftQCD:all"] \
    [--campaign NAME] [--output-base DIR] [--seed-base BASE] [--store-tracks] [--no-root]

Notes:
  - BX split is balanced with remainder assigned to the last jobs.
  - Job seed is computed as (seed_base + job_index).
USAGE
  exit 1
}

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$RUN_SCRIPT_DIR/../.." && pwd)"

CMSSW_BASE="${CMSSW_BASE:-$(cd "$STUDY_DIR/../.." && pwd)}"
SCRAM_ARCH="${SCRAM_ARCH:-el9_amd64_gcc12}"

echo "+ cd $RUN_SCRIPT_DIR"
cd "$RUN_SCRIPT_DIR"

export HIGGS_CEP_SETUP_STRICT=1
echo "+ source $STUDY_DIR/setup_env.sh"
source "$STUDY_DIR/setup_env.sh"

if ! run_cmd python3 -c "import pythia8mc" >/dev/null 2>&1; then
  echo "ERROR: Python module 'pythia8mc' is not available after environment setup." >&2
  echo "Check CMSSW runtime and SCRAM_ARCH on the worker node." >&2
  exit 1
fi

unset PYTHIA8DATA

JOB_INDEX=""
TOTAL_BX=""
TOTAL_JOBS=""
MU=200
MU_MODE="fixed"
E_CM=14000.0
PROCESSES="SoftQCD:all"
CAMPAIGN=""
OUTPUT_BASE="$STUDY_DIR/bkg-generation/output"
SEED_BASE=1000
BUILD_ROOT=true
STORE_TRACKS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-index)
      JOB_INDEX="$2"
      shift 2
      ;;
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
    --campaign)
      CAMPAIGN="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
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

if [[ -z "$JOB_INDEX" || -z "$TOTAL_BX" || -z "$TOTAL_JOBS" || -z "$CAMPAIGN" ]]; then
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
if (( JOB_INDEX < 0 || JOB_INDEX >= TOTAL_JOBS )); then
  echo "ERROR: --job-index must be in [0, jobs-1]" >&2
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

BASE_BX=$(( TOTAL_BX / TOTAL_JOBS ))
REMAINDER=$(( TOTAL_BX % TOTAL_JOBS ))
JOB_SEED=$(( SEED_BASE + JOB_INDEX ))

SPLIT_BOUNDARY=$(( TOTAL_JOBS - REMAINDER ))
if (( JOB_INDEX < SPLIT_BOUNDARY )); then
  N_BX_JOB=$BASE_BX
  BX_OFFSET=$(( JOB_INDEX * BASE_BX ))
else
  N_BX_JOB=$(( BASE_BX + 1 ))
  BX_OFFSET=$(( SPLIT_BOUNDARY * BASE_BX + (JOB_INDEX - SPLIT_BOUNDARY) * (BASE_BX + 1) ))
fi

if (( N_BX_JOB <= 0 )); then
  echo "Skipping job $JOB_INDEX because this partition has 0 BX." >&2
  exit 0
fi

CAMPAIGN_DIR="$OUTPUT_BASE/$CAMPAIGN"
JOB_DIR="$CAMPAIGN_DIR"
mkdir -p "$JOB_DIR"

LABEL="${CAMPAIGN}_job_${JOB_INDEX}_bx${BX_OFFSET}_n${N_BX_JOB}"
NPZ_FILE="$JOB_DIR/${LABEL}.npz"
ROOT_FILE="$JOB_DIR/${LABEL}_pairs.root"

echo "[job ${JOB_INDEX}] campaign=${CAMPAIGN} total_bx=${TOTAL_BX} jobs=${TOTAL_JOBS} mu=${MU} mu_mode=${MU_MODE}"
echo "[job ${JOB_INDEX}] bx_offset=${BX_OFFSET} n_bx=${N_BX_JOB} seed=${JOB_SEED}"
echo "[job ${JOB_INDEX}] output npz: ${NPZ_FILE}"

GEN_ARGS=(
  --n-bx "$N_BX_JOB"
  --mu "$MU"
  --mu-mode "$MU_MODE"
  --e-cm "$E_CM"
  --processes "$PROCESSES"
  --seed "$JOB_SEED"
  --bx-offset "$BX_OFFSET"
  --output "$NPZ_FILE"
)

if [[ "$STORE_TRACKS" == true ]]; then
  GEN_ARGS+=(--store-tracks)
fi

run_cmd python3 -u "$RUN_SCRIPT_DIR/generate_minbias_protons.py" "${GEN_ARGS[@]}"

if [[ "$BUILD_ROOT" == true ]]; then
  echo "[job ${JOB_INDEX}] output root: ${ROOT_FILE}"
  if run_cmd python3 -c "import uproot" >/dev/null 2>&1; then
    run_cmd python3 -X faulthandler -u "$RUN_SCRIPT_DIR/build_minbias_pairs.py" \
      --input "$NPZ_FILE" \
      --all-bx \
      --root-out "$ROOT_FILE"
  else
    echo "WARNING: uproot is not available; skipping ROOT output for job ${JOB_INDEX}" >&2
    {
      echo "ROOT build skipped because uproot is not installed"
      echo "install_hint=python3 -m pip install --user uproot awkward"
      echo "npz=${NPZ_FILE}"
      echo "intended_root=${ROOT_FILE}"
      echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${ROOT_FILE}.FAILED.txt"
  fi
fi

echo "[job ${JOB_INDEX}] done"
