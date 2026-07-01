#!/usr/bin/env bash

_higgs_cep_fpmc_main() {
  local script_dir status had_nounset
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "$script_dir/setup_common.sh" || return 1

  if [[ -n "${HIGGS_CEP_MODERN_ENV_LOADED:-}" ]]; then
    echo "ERROR: a modern LCG environment is already loaded." >&2
    echo "Use a fresh shell before loading the FPMC environment." >&2
    return 1
  fi

  export FPMC_DIR="${HIGGS_CEP_FPMC_DIR:-$REPO_ROOT_DIR/fpmc}"
  export FPMC_BUILD_DIR="${HIGGS_CEP_FPMC_BUILD_DIR:-$FPMC_DIR/build}"
  export FPMC_EXE="${HIGGS_CEP_FPMC_EXE:-$FPMC_BUILD_DIR/fpmc-lhe}"
  export FPMC_GCC_SETUP="${HIGGS_CEP_FPMC_GCC_SETUP:-/cvmfs/sft.cern.ch/lcg/external/gcc/6.1.0/x86_64-slc6/setup.sh}"

  if [[ ! -f "$FPMC_GCC_SETUP" ]]; then
    echo "ERROR: FPMC GCC setup script not found: $FPMC_GCC_SETUP" >&2
    return 1
  fi
  if [[ ! -x "$FPMC_EXE" ]]; then
    echo "ERROR: FPMC executable not found or not executable: $FPMC_EXE" >&2
    return 1
  fi

  case "$-" in
    *u*) had_nounset=1; set +u ;;
    *) had_nounset=0 ;;
  esac
  status=0
  source "$FPMC_GCC_SETUP" || status=$?
  if [[ "$had_nounset" -eq 1 ]]; then
    set -u
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: failed to source $FPMC_GCC_SETUP" >&2
    return "$status"
  fi

  _higgs_cep_prepend_path PATH "$FPMC_BUILD_DIR"
  _higgs_cep_prepend_path LD_LIBRARY_PATH "$FPMC_BUILD_DIR"
  _higgs_cep_prepend_path \
    LD_LIBRARY_PATH "$FPMC_BUILD_DIR/Fpmc/External/comphep_interface"
  export HIGGS_CEP_FPMC_ENV_LOADED=1
}

_higgs_cep_fpmc_main
