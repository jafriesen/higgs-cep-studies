#!/usr/bin/env bash

_higgs_cep_madgraph_main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "$script_dir/setup_modern.sh" || return 1

  export MADGRAPH_EXE="${HIGGS_CEP_MADGRAPH_EXE:-$(command -v mg5_aMC || true)}"
  if [[ -z "$MADGRAPH_EXE" || ! -x "$MADGRAPH_EXE" ]]; then
    echo "ERROR: mg5_aMC was not found in the LCG environment." >&2
    return 1
  fi

  export MADGRAPH_DIR="$(cd "$(dirname "$(readlink -f "$MADGRAPH_EXE")")/.." && pwd)"
  if [[ ! -f "$MADGRAPH_DIR/VERSION" ]]; then
    echo "ERROR: MadGraph VERSION file not found under $MADGRAPH_DIR." >&2
    return 1
  fi
  export MADGRAPH_VERSION="$(
    awk -F= '/^version/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' \
      "$MADGRAPH_DIR/VERSION"
  )"

  if ! command -v lhapdf-config >/dev/null 2>&1; then
    echo "ERROR: lhapdf-config was not found in the LCG environment." >&2
    return 1
  fi
}

_higgs_cep_madgraph_main
