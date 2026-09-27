"""Source and installed-package path defaults for PCD generation."""

import os
from pathlib import Path

from ...configuration import PROJECT_ROOT


SOURCE_PCD = Path(os.environ.get(
    "LEE_MANIPULATOR_SOURCE_PCD", str(PROJECT_ROOT / "path" / "data" / "AI_Glass_Front_0.010mm.pcd"),
)).expanduser()
SOURCE_OUTPUT_ROOT = Path(os.environ.get(
    "LEE_MANIPULATOR_RUNS", str(PROJECT_ROOT / "runs"),
)).expanduser() / "source_generation"
