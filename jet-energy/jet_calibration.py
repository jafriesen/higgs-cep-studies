#!/usr/bin/env python3
"""Compatibility import for the shared JetPUPPI calibration API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.jet_calibration import *  # noqa: F401,F403
