#!/usr/bin/env bash

_higgs_cep_modern_main() {
  local script_dir status had_nounset
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "$script_dir/setup_common.sh" || return 1

  if [[ "${HIGGS_CEP_FPMC_ENV_LOADED:-0}" == "1" ]]; then
    echo "ERROR: FPMC's legacy environment is already loaded." >&2
    echo "Use a fresh shell before loading a modern tool environment." >&2
    return 1
  fi

  export LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh}"
  if [[ "${HIGGS_CEP_MODERN_ENV_LOADED:-}" == "$LCG_VIEW" ]]; then
    return 0
  fi
  if [[ ! -f "$LCG_VIEW" ]]; then
    echo "ERROR: LCG setup script not found: $LCG_VIEW" >&2
    return 1
  fi

  case "$-" in
    *u*) had_nounset=1; set +u ;;
    *) had_nounset=0 ;;
  esac
  status=0
  source "$LCG_VIEW" || status=$?
  if [[ "$had_nounset" -eq 1 ]]; then
    set -u
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: failed to source $LCG_VIEW" >&2
    return "$status"
  fi

  export HIGGS_CEP_MODERN_ENV_LOADED="$LCG_VIEW"
}

_higgs_cep_modern_main
