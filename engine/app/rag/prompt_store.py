from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@lru_cache(maxsize=1)
def _load_prompt_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_text(prompt_type: str, task: str, field: str, **values: Any) -> str:
    if prompt_type not in {"static", "dynamic"}:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    path = PROMPT_DIR / f"{prompt_type}_prompts.json"
    prompts = _load_prompt_file(path)
    template = prompts[task][field]
    return str(template).format_map(_SafeFormatDict(values)).strip()


def clear_prompt_cache() -> None:
    _load_prompt_file.cache_clear()
