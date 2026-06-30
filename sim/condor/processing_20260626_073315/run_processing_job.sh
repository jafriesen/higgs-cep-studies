#!/usr/bin/env bash

set -euo pipefail

PROCESS="${1:?missing PROCESS}"

cd /home/jfriesen/higgs-cep-studies/higgs-cep-studies
source setup_env.sh
if command -v cmsenv >/dev/null 2>&1; then
  cmsenv
fi

echo "Batch host: $(hostname)"
echo "Batch scratch: $PWD"
echo "Process: $PROCESS"

PYTHIA_CMD=(python3 generation-pythia/scripts/process_superchic.py --process "$PROCESS" )
echo "${PYTHIA_CMD[*]}"
"${PYTHIA_CMD[@]}"

DELPHES_CMD=(python3 sim/scripts/run_processes_delphes.py --process "$PROCESS" )
echo "${DELPHES_CMD[*]}"
"${DELPHES_CMD[@]}"

