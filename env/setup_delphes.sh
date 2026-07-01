#!/usr/bin/env bash

_higgs_cep_delphes_main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "$script_dir/setup_modern.sh" || return 1

  export DELPHES_DIR="${HIGGS_CEP_DELPHES_DIR:-$REPO_ROOT_DIR/delphes}"
  if [[ ! -d "$DELPHES_DIR" ]]; then
    echo "ERROR: Delphes directory not found: $DELPHES_DIR" >&2
    return 1
  fi
  _higgs_cep_prepend_path PATH "$DELPHES_DIR"
  _higgs_cep_prepend_path LD_LIBRARY_PATH "$DELPHES_DIR"
}

_higgs_cep_delphes_main
