from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
__path__ = [
    str(ROOT / "backend" / "app"),
    str(ROOT / "engine" / "app"),
    str(Path(__file__).resolve().parent),
]
