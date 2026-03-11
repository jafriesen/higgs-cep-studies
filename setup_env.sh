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

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="${HIGGS_CEP_STUDIES_DIR:-$SCRIPT_DIR}"

  CMSSW_ROOT_DIR="${CMSSW_ROOT_DIR:-$(cd "$REPO_DIR/../.." && pwd)}"
  CMSSW_PROJECT_DIR="${CMSSW_PROJECT_DIR:-$CMSSW_ROOT_DIR/src}"
  SUPERCHIC_DIR="${SUPERCHIC_DIR:-$CMSSW_PROJECT_DIR/SuperChic}"
  SUPERCHIC_INSTALL_DIR="${SUPERCHIC_INSTALL_DIR:-$SUPERCHIC_DIR/install}"
  SUPERCHIC_BUILD_DIR="${SUPERCHIC_BUILD_DIR:-$SUPERCHIC_DIR/build}"

  if [[ ! -f /cvmfs/cms.cern.ch/cmsset_default.sh ]]; then
    echo "ERROR: /cvmfs/cms.cern.ch/cmsset_default.sh not found." >&2
    echo "Load this environment on a CMS/cvmfs-enabled node or with CVMFS mounted." >&2
    return 1
  fi

  # Keep cmsset_default.sh compatible with shells using nounset by predefining
  # variables it dereferences directly.
  : "${VO_CMS_SW_DIR:=}"
  : "${SCRAM_ARCH:=}"
  : "${CMS_PATH:=}"
  : "${SITECONFIG_PATH:=}"
  : "${CVSROOT:=}"
  : "${OSG_APP:=}"

  if ! source /cvmfs/cms.cern.ch/cmsset_default.sh; then
    echo "ERROR: failed to source /cvmfs/cms.cern.ch/cmsset_default.sh" >&2
    return 1
  fi

  if [[ ! -d "$CMSSW_PROJECT_DIR" ]]; then
    echo "ERROR: CMSSW source directory not found: $CMSSW_PROJECT_DIR" >&2
    return 1
  fi

  _scram_env="$(cd "$CMSSW_PROJECT_DIR" && scram runtime -sh 2>/dev/null)"
  if [[ -z "$_scram_env" ]]; then
    echo "ERROR: failed to load scram runtime from $CMSSW_PROJECT_DIR" >&2
    return 1
  fi
  eval "$_scram_env"

  if [[ ! -d "$SUPERCHIC_DIR" ]]; then
    echo "ERROR: SuperChic directory not found: $SUPERCHIC_DIR" >&2
    echo "Clone it with:" >&2
    echo "  cd $CMSSW_PROJECT_DIR && git clone https://github.com/LucianHL/SuperChic.git" >&2
    return 1
  fi

  if [[ -f "$SUPERCHIC_DIR/env_setup.sh" ]]; then
    if ! source "$SUPERCHIC_DIR/env_setup.sh"; then
      echo "ERROR: failed to source $SUPERCHIC_DIR/env_setup.sh" >&2
      return 1
    fi
  else
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
  fi

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
  export CMSSW_RELEASE_DIR="$CMSSW_ROOT_DIR"
  export CMSSW_PROJECT_DIR
  export SUPERCHIC_DIR
  export HIGGS_SIGNAL_DIR="$REPO_DIR/signal-generation"
  export HIGGS_BKG_DIR="$REPO_DIR/bkg-generation"
  export HIGGS_ANALYSIS_DIR="$REPO_DIR/analysis"

  CMSSW_REL=$(basename "$CMSSW_ROOT_DIR")
  printf 'Setup complete for %s.\n' "$CMSSW_REL"
  printf 'SuperChic dir: %s\n' "$SUPERCHIC_DIR"
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
