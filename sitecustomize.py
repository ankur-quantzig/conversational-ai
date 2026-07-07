from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
for path in (ROOT / "backend", ROOT / "engine"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
