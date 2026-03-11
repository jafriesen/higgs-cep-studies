#!/usr/bin/env bash

usage() {
  cat <<'USAGE'
Usage:
  run_pythia8_minbias_study.sh [--n-bx N_BX] [--mu MU] [--e-cm ECM_GEV] \
    [--processes "SoftQCD:all"] [--label NAME] [--no-root]

Example:
  ./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --label minbias
USAGE
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$STUDY_DIR/setup_env.sh"

N_BX=1000
MU=200
E_CM=14000.0
PROCESSES="SoftQCD:all"
LABEL="minbias"
BUILD_ROOT=true

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
    --no-root)
      BUILD_ROOT=false
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

python3 "$SCRIPT_DIR/generate_minbias_protons.py" \
  --n-bx "$N_BX" \
  --mu "$MU" \
  --e-cm "$E_CM" \
  --processes "$PROCESSES" \
  --output "$NPZ_FILE"

if [[ "$BUILD_ROOT" == true ]]; then
  python3 "$SCRIPT_DIR/build_minbias_pairs.py" \
    --input "$NPZ_FILE" \
    --all-bx \
    --root-out "$ROOT_FILE"
fi

echo "Generated min-bias npz: $NPZ_FILE"
if [[ "$BUILD_ROOT" == true ]]; then
  echo "Generated min-bias pair ROOT: $ROOT_FILE"
fi
