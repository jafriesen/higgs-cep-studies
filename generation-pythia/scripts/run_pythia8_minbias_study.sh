#!/usr/bin/env bash

usage() {
  cat <<'USAGE'
Usage:
  run_pythia8_minbias_study.sh [--n-bx N_BX] [--mu MU] [--mu-mode fixed|poisson] \
    [--e-cm ECM_GEV] [--processes "SoftQCD:all"] [--label NAME] \
    [--seed SEED] [--bx-offset BX_OFFSET] [--store-tracks] [--no-root]

Example:
  ./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --mu-mode poisson --label minbias
USAGE
  exit 1
}

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$RUN_SCRIPT_DIR/../.." && pwd)"

source "$STUDY_DIR/setup_env.sh"

unset PYTHIA8DATA

N_BX=1000
MU=200
MU_MODE="fixed"
E_CM=14000.0
PROCESSES="SoftQCD:all"
LABEL="minbias"
SEED=""
BX_OFFSET=0
BUILD_ROOT=true
STORE_TRACKS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-bx)
      N_BX="$2"
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
    --label)
      LABEL="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --bx-offset)
      BX_OFFSET="$2"
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
      break
      ;;
  esac
 done

OUT_DIR="$STUDY_DIR/bkg-generation/output/$LABEL"
mkdir -p "$OUT_DIR"
NPZ_FILE="$OUT_DIR/$LABEL.npz"
ROOT_FILE="$OUT_DIR/${LABEL}_pairs.root"

GEN_ARGS=(
  --n-bx "$N_BX"
  --mu "$MU"
  --mu-mode "$MU_MODE"
  --e-cm "$E_CM"
  --processes "$PROCESSES"
  --bx-offset "$BX_OFFSET"
  --output "$NPZ_FILE"
)

if [[ -n "$SEED" ]]; then
  GEN_ARGS+=(--seed "$SEED")
fi
if [[ "$STORE_TRACKS" == true ]]; then
  GEN_ARGS+=(--store-tracks)
fi

python3 "$RUN_SCRIPT_DIR/generate_minbias_protons.py" "${GEN_ARGS[@]}"

if [[ "$BUILD_ROOT" == true ]]; then
  python3 "$RUN_SCRIPT_DIR/build_minbias_pairs.py" \
    --input "$NPZ_FILE" \
    --all-bx \
    --root-out "$ROOT_FILE"
fi

echo "Generated min-bias npz: $NPZ_FILE"
if [[ "$BUILD_ROOT" == true ]]; then
  echo "Generated min-bias pair ROOT: $ROOT_FILE"
fi
