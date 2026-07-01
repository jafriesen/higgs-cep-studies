#!/usr/bin/env bash

_higgs_cep_pythia_main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "$script_dir/setup_modern.sh" || return 1

  if [[ -z "${PYTHIA8:-}" || ! -d "$PYTHIA8" ]]; then
    echo "ERROR: PYTHIA8 was not provided by the LCG view: $LCG_VIEW" >&2
    return 1
  fi
}

_higgs_cep_pythia_main
