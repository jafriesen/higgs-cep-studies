#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${HIGGS_CEP_STUDIES_DIR:-$SCRIPT_DIR}"

CMSSW_ROOT_DIR="${CMSSW_ROOT_DIR:-$(cd "$REPO_DIR/../.." && pwd)}"
CMSSW_PROJECT_DIR="${CMSSW_PROJECT_DIR:-$CMSSW_ROOT_DIR/src}"
SUPERCHIC_DIR="${SUPERCHIC_DIR:-$CMSSW_PROJECT_DIR/SuperChic}"

if [[ ! -f /cvmfs/cms.cern.ch/cmsset_default.sh ]]; then
  echo "ERROR: /cvmfs/cms.cern.ch/cmsset_default.sh not found." >&2
  echo "Load this environment on a CMS/cvmfs-enabled node or with CVMFS mounted." >&2
  exit 1
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh

if [[ ! -d "$CMSSW_PROJECT_DIR" ]]; then
  echo "ERROR: CMSSW source directory not found: $CMSSW_PROJECT_DIR" >&2
  exit 1
fi

cd "$CMSSW_PROJECT_DIR"
eval "$(scram runtime -sh)"

if [[ ! -d "$SUPERCHIC_DIR" ]]; then
  echo "ERROR: SuperChic directory not found: $SUPERCHIC_DIR" >&2
  exit 1
fi

if [[ -f "$SUPERCHIC_DIR/env_setup.sh" ]]; then
  source "$SUPERCHIC_DIR/env_setup.sh"
else
  echo "ERROR: SuperChic env_setup.sh not found: $SUPERCHIC_DIR/env_setup.sh" >&2
  exit 1
fi

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
