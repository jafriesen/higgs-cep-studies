#!/usr/bin/env bash

_higgs_cep_setup_main() {
  _prepend_env_path() {
    local var_name="$1"
    local path_to_add="$2"
    local current_value="${!var_name:-}"

    case ":${current_value}:" in
      *":${path_to_add}:"*) return 0 ;;
    esac

    if [[ -n "$current_value" ]]; then
      export "${var_name}=${path_to_add}:${current_value}"
    else
      export "${var_name}=${path_to_add}"
    fi
  }

  _remove_env_path() {
    local var_name="$1"
    local path_to_remove="$2"
    local current_value="${!var_name:-}"
    local new_value=""
    local entry

    [[ -n "$path_to_remove" ]] || return 0
    IFS=':' read -r -a _higgs_cep_path_entries <<< "$current_value"
    for entry in "${_higgs_cep_path_entries[@]}"; do
      [[ "$entry" == "$path_to_remove" ]] && continue
      if [[ -n "$new_value" ]]; then
        new_value="${new_value}:${entry}"
      else
        new_value="$entry"
      fi
    done
    export "${var_name}=${new_value}"
  }

  OLD_SUPERCHIC_DIR="${SUPERCHIC_DIR:-}"
  OLD_SUPERCHIC_INSTALL_DIR="${SUPERCHIC_INSTALL_DIR:-}"
  OLD_SUPERCHIC_BUILD_DIR="${SUPERCHIC_BUILD_DIR:-}"
  OLD_SUPERCHIC_BIN_DIR="${SUPERCHIC_BIN_DIR:-}"
  OLD_DELPHES_DIR="${DELPHES_DIR:-}"
  if [[ -n "$OLD_SUPERCHIC_DIR" ]]; then
    OLD_SUPERCHIC_INSTALL_DIR="${OLD_SUPERCHIC_INSTALL_DIR:-$OLD_SUPERCHIC_DIR/install}"
    OLD_SUPERCHIC_BUILD_DIR="${OLD_SUPERCHIC_BUILD_DIR:-$OLD_SUPERCHIC_DIR/build}"
  fi

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$SCRIPT_DIR"
  REPO_ROOT_DIR="$(cd "$REPO_DIR/../" && pwd)"

  SUPERCHIC_DIR="${HIGGS_CEP_SUPERCHIC_DIR:-$REPO_ROOT_DIR/SuperChic}"
  SUPERCHIC_INSTALL_DIR="${HIGGS_CEP_SUPERCHIC_INSTALL_DIR:-$SUPERCHIC_DIR/install}"
  SUPERCHIC_BUILD_DIR="${HIGGS_CEP_SUPERCHIC_BUILD_DIR:-$SUPERCHIC_DIR/build}"

  DELPHES_DIR="${HIGGS_CEP_DELPHES_DIR:-$REPO_ROOT_DIR/delphes}"
  LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh}"

  _remove_env_path PATH "$OLD_SUPERCHIC_INSTALL_DIR/bin"
  _remove_env_path PATH "$OLD_SUPERCHIC_BUILD_DIR/bin"
  _remove_env_path PATH "$OLD_SUPERCHIC_BIN_DIR"
  _remove_env_path PATH "$OLD_DELPHES_DIR"
  _remove_env_path LD_LIBRARY_PATH "$OLD_SUPERCHIC_INSTALL_DIR/lib"
  _remove_env_path LD_LIBRARY_PATH "$OLD_SUPERCHIC_INSTALL_DIR/lib64"
  _remove_env_path LD_LIBRARY_PATH "$OLD_SUPERCHIC_BUILD_DIR/lib"
  _remove_env_path LD_LIBRARY_PATH "$OLD_SUPERCHIC_BUILD_DIR/lib64"
  _remove_env_path LD_LIBRARY_PATH "$OLD_DELPHES_DIR"
  _remove_env_path LHAPDF_DATA_PATH "$OLD_SUPERCHIC_INSTALL_DIR/share/SuperChic/SF"
  _remove_env_path LHAPDF_DATA_PATH "$OLD_SUPERCHIC_BUILD_DIR/unpacked/SF"
  _remove_env_path LHAPDF_DATA_PATH "$OLD_SUPERCHIC_DIR/SF"

  if [[ ! -f "$LCG_VIEW" ]]; then
    echo "ERROR: LCG setup script not found: $LCG_VIEW" >&2
    echo "Load this environment on a CVMFS-enabled node or set LCG_VIEW." >&2
    return 1
  fi

  case "$-" in
    *u*) _higgs_cep_had_nounset=1; set +u ;;
    *) _higgs_cep_had_nounset=0 ;;
  esac
  _higgs_cep_lcg_status=0
  source "$LCG_VIEW" || _higgs_cep_lcg_status=$?
  if [[ "$_higgs_cep_had_nounset" -eq 1 ]]; then
    set -u
  fi
  if [[ "$_higgs_cep_lcg_status" -ne 0 ]]; then
    echo "ERROR: failed to source $LCG_VIEW" >&2
    return 1
  fi

  if [[ ! -d "$SUPERCHIC_DIR" ]]; then
    echo "ERROR: SuperChic directory not found: $SUPERCHIC_DIR" >&2
    echo "Clone it with:" >&2
    echo "  cd $REPO_ROOT_DIR && git clone https://github.com/LucianHL/SuperChic.git" >&2
    return 1
  fi

  SUPERCHIC_BIN_DIR=""
  if [[ -x "$SUPERCHIC_INSTALL_DIR/bin/superchic" ]]; then
    SUPERCHIC_BIN_DIR="$SUPERCHIC_INSTALL_DIR/bin"
    export SUPERCHIC_DATA_PATH="$SUPERCHIC_INSTALL_DIR/share/SuperChic"
  elif [[ -x "$SUPERCHIC_BUILD_DIR/bin/superchic" ]]; then
    SUPERCHIC_BIN_DIR="$SUPERCHIC_BUILD_DIR/bin"
    export SUPERCHIC_DATA_PATH="$SUPERCHIC_BUILD_DIR/share/SuperChic"
  else
    echo "ERROR: SuperChic runtime not found under $SUPERCHIC_DIR" >&2
    echo "Expected one of:" >&2
    echo "  $SUPERCHIC_INSTALL_DIR/bin/superchic" >&2
    echo "  $SUPERCHIC_BUILD_DIR/bin/superchic" >&2
    echo "Build SuperChic first, e.g. cmake -S . -B build ... && cmake --build build && cmake --install build" >&2
    return 1
  fi

  _prepend_env_path PATH "$SUPERCHIC_BIN_DIR"

  for libdir in \
    "$SUPERCHIC_INSTALL_DIR/lib" \
    "$SUPERCHIC_INSTALL_DIR/lib64" \
    "$SUPERCHIC_BUILD_DIR/lib" \
    "$SUPERCHIC_BUILD_DIR/lib64"; do
    if [[ -d "$libdir" ]]; then
      _prepend_env_path LD_LIBRARY_PATH "$libdir"
    fi
  done

  if command -v lhapdf-config >/dev/null 2>&1; then
    _prepend_env_path LHAPDF_DATA_PATH "$(lhapdf-config --datadir)"
  fi

  # SuperChic uses its own SF_* LHAPDF grids (e.g. SF_MSHT20qed_nnlo) during init.
  # Ensure these local grids are visible in LHAPDF_DATA_PATH.
  for sfdir in \
    "$SUPERCHIC_INSTALL_DIR/share/SuperChic/SF" \
    "$SUPERCHIC_BUILD_DIR/unpacked/SF" \
    "$SUPERCHIC_DIR/SF"; do
    if [[ -d "$sfdir" ]]; then
      _prepend_env_path LHAPDF_DATA_PATH "$sfdir"
    fi
  done

  export HIGGS_CEP_STUDIES_DIR="$REPO_DIR"
  export REPO_ROOT_DIR
  export LCG_VIEW
  export SUPERCHIC_DIR
  export SUPERCHIC_INSTALL_DIR
  export SUPERCHIC_BUILD_DIR
  export SUPERCHIC_BIN_DIR
  export DELPHES_DIR

  if [[ -d "$DELPHES_DIR" ]]; then
    _prepend_env_path PATH "$DELPHES_DIR"
    _prepend_env_path LD_LIBRARY_PATH "$DELPHES_DIR"
  fi

  printf 'Setup complete.\n'
  printf 'LCG view: %s\n' "$LCG_VIEW"
  printf 'SuperChic dir: %s\n' "$SUPERCHIC_DIR"
  printf 'Delphes dir: %s\n' "$DELPHES_DIR"
  printf 'Repo dir: %s\n' "$REPO_DIR"
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
