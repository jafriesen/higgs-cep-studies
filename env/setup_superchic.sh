#!/usr/bin/env bash

_higgs_cep_superchic_main() {
  local script_dir candidate libdir sfdir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "$script_dir/setup_modern.sh" || return 1

  export SUPERCHIC_DIR="${HIGGS_CEP_SUPERCHIC_DIR:-$REPO_ROOT_DIR/SuperChic}"
  export SUPERCHIC_INSTALL_DIR="${HIGGS_CEP_SUPERCHIC_INSTALL_DIR:-$SUPERCHIC_DIR/install}"
  export SUPERCHIC_BUILD_DIR="${HIGGS_CEP_SUPERCHIC_BUILD_DIR:-$SUPERCHIC_DIR/build}"
  export SUPERCHIC_BIN_DIR=""

  for candidate in "$SUPERCHIC_INSTALL_DIR/bin" "$SUPERCHIC_BUILD_DIR/bin"; do
    if [[ -x "$candidate/superchic" ]]; then
      export SUPERCHIC_BIN_DIR="$candidate"
      break
    fi
  done
  if [[ -z "$SUPERCHIC_BIN_DIR" ]]; then
    echo "ERROR: SuperChic runtime not found under $SUPERCHIC_DIR" >&2
    return 1
  fi

  _higgs_cep_prepend_path PATH "$SUPERCHIC_BIN_DIR"
  for libdir in \
    "$SUPERCHIC_INSTALL_DIR/lib" \
    "$SUPERCHIC_INSTALL_DIR/lib64" \
    "$SUPERCHIC_BUILD_DIR/lib" \
    "$SUPERCHIC_BUILD_DIR/lib64"; do
    if [[ -d "$libdir" ]]; then
      _higgs_cep_prepend_path LD_LIBRARY_PATH "$libdir"
    fi
  done

  if [[ "$SUPERCHIC_BIN_DIR" == "$SUPERCHIC_INSTALL_DIR/bin" ]]; then
    export SUPERCHIC_DATA_PATH="$SUPERCHIC_INSTALL_DIR/share/SuperChic"
  else
    export SUPERCHIC_DATA_PATH="$SUPERCHIC_BUILD_DIR/share/SuperChic"
  fi

  if command -v lhapdf-config >/dev/null 2>&1; then
    _higgs_cep_prepend_path LHAPDF_DATA_PATH "$(lhapdf-config --datadir)"
  fi
  for sfdir in \
    "$SUPERCHIC_INSTALL_DIR/share/SuperChic/SF" \
    "$SUPERCHIC_BUILD_DIR/unpacked/SF" \
    "$SUPERCHIC_DIR/SF"; do
    if [[ -d "$sfdir" ]]; then
      _higgs_cep_prepend_path LHAPDF_DATA_PATH "$sfdir"
    fi
  done
}

_higgs_cep_superchic_main
