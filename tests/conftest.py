from __future__ import annotations

import sys
from pathlib import Path

# Compute the repo root and src path for local test imports.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Insert src/ once so tests import the in-repo package.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
