#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_superchic_pythia.sh --input LHE_OR_DIR --output OUTPUT.hepmc \
    [--max-events N] [--max-files N] [--seed SEED] [--verbose]

Example:
  ./run_superchic_pythia.sh \
    --input ../../generation-superchic/output/h_cc/superchic_h_cc_nev100000_j100_20260618_102604/evrecs \
    --output /tmp/h_cc_pythia.hepmc \
    --max-events 100
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
HELPER_SRC="$SCRIPT_DIR/superchic_lhe_to_hepmc.cc"
BUILD_DIR="${TMPDIR:-/tmp}/higgs_cep_pythia_${USER:-user}"
HELPER_BIN="$BUILD_DIR/superchic_lhe_to_hepmc"

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

source "$STUDY_DIR/setup_env.sh"

tool_value() {
  local tool="$1"
  local key="$2"
  scram tool info "$tool" 2>/dev/null | awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1); exit}'
}

compile_helper() {
  local pythia_include pythia_libdir pythia_libs
  local hepmc_include hepmc_libdir hepmc_libs

  pythia_include="$(tool_value pythia8 INCLUDE)"
  pythia_libdir="$(tool_value pythia8 LIBDIR)"
  pythia_libs="$(tool_value pythia8 LIB)"
  hepmc_include="$(tool_value hepmc3 INCLUDE)"
  hepmc_libdir="$(tool_value hepmc3 LIBDIR)"
  hepmc_libs="$(tool_value hepmc3 LIB)"

  if [[ -z "$pythia_include" || -z "$pythia_libdir" || -z "$pythia_libs" ]]; then
    echo "ERROR: could not find pythia8 tool settings from scram." >&2
    exit 1
  fi
  if [[ -z "$hepmc_include" || -z "$hepmc_libdir" || -z "$hepmc_libs" ]]; then
    echo "ERROR: could not find hepmc3 tool settings from scram." >&2
    exit 1
  fi

  mkdir -p "$BUILD_DIR"

  local link_args=()
  local lib
  for lib in $pythia_libs $hepmc_libs; do
    link_args+=("-l$lib")
  done

  g++ -std=c++17 -O2 -Wall -Wextra \
    "$HELPER_SRC" \
    -I"$pythia_include" -I"$hepmc_include" \
    -L"$pythia_libdir" -L"$hepmc_libdir" \
    -Wl,-rpath,"$pythia_libdir" -Wl,-rpath,"$hepmc_libdir" \
    "${link_args[@]}" \
    -o "$HELPER_BIN"
}

if [[ ! -x "$HELPER_BIN" || "$HELPER_SRC" -nt "$HELPER_BIN" ]]; then
  compile_helper
fi

exec "$HELPER_BIN" "$@"
