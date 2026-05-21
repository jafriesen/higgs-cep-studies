#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  rebuild_minbias_pairs_from_npz.sh [campaign-identification options] [control options] [-- <build_minbias_pairs args>]

Campaign identification options (match submit_pythia8_minbias_condor.sh):
  --campaign NAME              Campaign folder name under output-base.
  --total-bx TOTAL             Total BX used at submission time.
  --jobs N_JOBS                Number of jobs used at submission time.
  --mu MU                      Mu used at submission time (default: 200).
  --output-base DIR            Output base used at submission time.

If --campaign is not given, the script searches for a unique directory matching:
  minbias_nbx<TOTAL>_j<JOBS>_mu<MU>_*
under output-base.

Control options:
  --pairs-output-dir DIR       Write rebuilt *_pairs.root files here instead of campaign dir.
  --dry-run                    Print commands without executing.
  --no-all-bx                  Do not add --all-bx by default.
  -h, --help                   Show this help.

Pass-through:
  Any args after '--' are forwarded to build_minbias_pairs.py.
  Do not pass --input/-i or --root-out in forwarded args; those are managed by this script.

Examples:
  ./rebuild_minbias_pairs_from_npz.sh --campaign minbias_mu200

  ./rebuild_minbias_pairs_from_npz.sh --total-bx 30000 --jobs 30 --mu 200 \
    --pairs-output-dir /tmp/minbias_pairs_rebuild

  ./rebuild_minbias_pairs_from_npz.sh --campaign minbias_mu200 -- --bx 0
USAGE
  exit 1
}

run_cmd() {
  printf "+ "
  printf "%q " "$@"
  echo
  "$@"
}

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$RUN_SCRIPT_DIR/../.." && pwd)"

CAMPAIGN=""
TOTAL_BX=""
TOTAL_JOBS=""
MU=200
OUTPUT_BASE="$STUDY_DIR/bkg-generation/output"
PAIRS_OUTPUT_DIR=""
DRY_RUN=false
USE_ALL_BX=true

FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign)
      CAMPAIGN="$2"
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
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --pairs-output-dir)
      PAIRS_OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-all-bx)
      USE_ALL_BX=false
      shift
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ "$OUTPUT_BASE" != /* ]]; then
  OUTPUT_BASE="$STUDY_DIR/$OUTPUT_BASE"
fi

if [[ -z "$CAMPAIGN" ]]; then
  if [[ -z "$TOTAL_BX" || -z "$TOTAL_JOBS" ]]; then
    echo "ERROR: provide --campaign, or provide both --total-bx and --jobs to auto-find campaign." >&2
    exit 1
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

  pattern="$OUTPUT_BASE/minbias_nbx${TOTAL_BX}_j${TOTAL_JOBS}_mu${MU}_*"
  mapfile -t matches < <(compgen -G "$pattern" || true)

  if (( ${#matches[@]} == 0 )); then
    echo "ERROR: no campaign directory matched pattern: $pattern" >&2
    exit 1
  fi
  if (( ${#matches[@]} > 1 )); then
    echo "ERROR: multiple campaign directories matched. Use --campaign explicitly." >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
  fi

  CAMPAIGN_DIR="${matches[0]}"
  CAMPAIGN="$(basename "$CAMPAIGN_DIR")"
else
  CAMPAIGN_DIR="$OUTPUT_BASE/$CAMPAIGN"
fi

if [[ ! -d "$CAMPAIGN_DIR" ]]; then
  echo "ERROR: campaign directory not found: $CAMPAIGN_DIR" >&2
  exit 1
fi

if [[ -n "$PAIRS_OUTPUT_DIR" ]]; then
  if [[ "$PAIRS_OUTPUT_DIR" != /* ]]; then
    PAIRS_OUTPUT_DIR="$STUDY_DIR/$PAIRS_OUTPUT_DIR"
  fi
  mkdir -p "$PAIRS_OUTPUT_DIR"
else
  PAIRS_OUTPUT_DIR="$CAMPAIGN_DIR"
fi

for arg in "${FORWARD_ARGS[@]}"; do
  case "$arg" in
    -i|--input|--root-out)
      echo "ERROR: do not pass $arg after '--'; this script sets input/output per file." >&2
      exit 1
      ;;
  esac
done

mapfile -t NPZ_FILES < <(find "$CAMPAIGN_DIR" -maxdepth 1 -type f -name "*.npz" | sort -V)
if (( ${#NPZ_FILES[@]} == 0 )); then
  echo "ERROR: no .npz files found in $CAMPAIGN_DIR" >&2
  exit 1
fi

echo "Campaign directory: $CAMPAIGN_DIR"
echo "NPZ files found: ${#NPZ_FILES[@]}"
echo "Pairs output dir: $PAIRS_OUTPUT_DIR"
echo "Default --all-bx: $USE_ALL_BX"

ok=0
for npz in "${NPZ_FILES[@]}"; do
  base="$(basename "$npz" .npz)"
  root_out="$PAIRS_OUTPUT_DIR/${base}_pairs.root"

  cmd=(python3 "$RUN_SCRIPT_DIR/build_minbias_pairs.py" --input "$npz" --root-out "$root_out")
  if [[ "$USE_ALL_BX" == true ]]; then
    cmd+=(--all-bx)
  fi
  if (( ${#FORWARD_ARGS[@]} > 0 )); then
    cmd+=("${FORWARD_ARGS[@]}")
  fi

  if [[ "$DRY_RUN" == true ]]; then
    printf "+ "
    printf "%q " "${cmd[@]}"
    echo
  else
    run_cmd "${cmd[@]}"
  fi

  ok=$((ok + 1))
done

echo "Done. Processed ${ok} NPZ files."
