#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$STUDY_DIR/signal-generation/scripts/run_superchic_signal.sh" \
  --process qcd_bb \
  --output-base "$STUDY_DIR/bkg-generation/output" \
  "$@"
