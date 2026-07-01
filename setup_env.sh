#!/usr/bin/env bash

_higgs_cep_setup_main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  source "$script_dir/env/setup_superchic.sh" || return 1
  export DELPHES_DIR="${HIGGS_CEP_DELPHES_DIR:-$REPO_ROOT_DIR/delphes}"
  if [[ -d "$DELPHES_DIR" ]]; then
    source "$script_dir/env/setup_delphes.sh" || return 1
  fi

  printf 'Setup complete.\n'
  printf 'LCG view: %s\n' "$LCG_VIEW"
  printf 'SuperChic dir: %s\n' "$SUPERCHIC_DIR"
  printf 'Delphes dir: %s\n' "$DELPHES_DIR"
  printf 'Repo dir: %s\n' "$HIGGS_CEP_STUDIES_DIR"
}

if _higgs_cep_setup_main; then
  _higgs_cep_setup_status=0
else
  _higgs_cep_setup_status=$?
fi

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  if [[ "$_higgs_cep_setup_status" -ne 0 ]]; then
    echo "setup_env.sh: setup failed (status=$_higgs_cep_setup_status), shell kept alive." >&2
  fi
  return 0
fi

if [[ "$_higgs_cep_setup_status" -ne 0 ]]; then
  echo "setup_env.sh: setup failed (status=$_higgs_cep_setup_status)." >&2
  if [[ "${HIGGS_CEP_SETUP_STRICT:-0}" != "1" ]]; then
    echo "setup_env.sh: returning exit code 0 to avoid IDE terminal termination." >&2
    exit 0
  fi
fi
exit "$_higgs_cep_setup_status"
