from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return ROOT


def data_root() -> Path:
    configured = (
        os.getenv("INSIGHT_DATA_ROOT")
        or os.getenv("DATABRICKS_DATA_VOLUME")
        or os.getenv("DATA_ROOT")
    )
    if configured:
        return Path(configured).expanduser()
    return ROOT / "data"


def output_dir(*parts: str) -> Path:
    return ROOT.joinpath("output", *parts)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
