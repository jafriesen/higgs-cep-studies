#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_minbias.sh [--events N] [--e-cm ECM_GEV] [--processes "SoftQCD:all"] \
    [--seed SEED] [--campaign NAME|--out-tag NAME] [--output OUTPUT.hepmc] \
    [--verbose]

Examples:
  ./run_minbias.sh --events 1000 --campaign minbias_test --seed 1
  ./run_minbias.sh --events 1000 --output /tmp/minbias.hepmc --seed 1
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
HELPER_SRC="$SCRIPT_DIR/generate_minbias.cc"
BUILD_DIR="${TMPDIR:-/tmp}/higgs_cep_pythia_${USER:-user}"
HELPER_BIN="$BUILD_DIR/generate_minbias"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "$STUDY_DIR/setup_env.sh"

tool_value() {
  local tool="$1"
  local key="$2"
  if ! command -v scram >/dev/null 2>&1; then
    return 0
  fi
  scram tool info "$tool" 2>/dev/null | awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1); exit}'
}

find_header_dir() {
  local header="$1"
  local paths="${2:-}"
  local path

  IFS=':' read -r -a path_entries <<< "$paths"
  for path in "${path_entries[@]}"; do
    if [[ -f "$path/$header" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
}

find_lib_dir() {
  local library="$1"
  local paths="${2:-}"
  local path

  IFS=':' read -r -a path_entries <<< "$paths"
  for path in "${path_entries[@]}"; do
    if [[ -f "$path/$library" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
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
    if [[ -z "${PYTHIA8:-}" ]]; then
      echo "ERROR: could not find Pythia8 from scram or PYTHIA8." >&2
      exit 1
    fi
    pythia_include="$PYTHIA8/include"
    pythia_libdir="$PYTHIA8/lib"
    pythia_libs="pythia8"
  fi

  if [[ -z "$hepmc_include" ]]; then
    hepmc_include="$(find_header_dir HepMC3/GenEvent.h "${CPLUS_INCLUDE_PATH:-}:${C_INCLUDE_PATH:-}:${ROOT_INCLUDE_PATH:-}")"
  fi
  if [[ -z "$hepmc_libdir" ]]; then
    hepmc_libdir="$(find_lib_dir libHepMC3.so "${LD_LIBRARY_PATH:-}")"
  fi
  if [[ -z "$hepmc_libs" ]]; then
    hepmc_libs="HepMC3"
  fi
  if [[ -z "$hepmc_include" || -z "$hepmc_libdir" ]]; then
    echo "ERROR: could not find HepMC3 headers or library from scram or the LCG environment." >&2
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
