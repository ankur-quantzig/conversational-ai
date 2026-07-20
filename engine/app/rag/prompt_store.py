from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Any, Literal


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_ROLES = ("system", "human")
REQUIRED_PROMPT_TASKS = frozenset(
    {
        "rag_answer",
        "follow_up_questions",
        "conversation_question",
        "question_preparation",
        "response_diagram",
        "quality_enrichment",
    }
)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@lru_cache(maxsize=2)
def _load_prompt_file(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt catalog must be a JSON object: {path}")
    return {str(key): str(value) for key, value in raw.items()}


def prompt_text(role: Literal["system", "human"], task: str, **values: Any) -> str:
    if role not in PROMPT_ROLES:
        raise ValueError(f"Unknown prompt role: {role}")
    path = PROMPT_DIR / f"{role}_prompts.json"
    prompts = _load_prompt_file(path)
    if task not in prompts:
        raise KeyError(f"Missing {role} prompt for task: {task}")
    template = prompts[task]
    return str(template).format_map(_SafeFormatDict(values)).strip()


def validate_prompt_catalogs() -> None:
    catalogs = {
        role: _load_prompt_file(PROMPT_DIR / f"{role}_prompts.json")
        for role in PROMPT_ROLES
    }
    for role, prompts in catalogs.items():
        tasks = set(prompts)
        if tasks != REQUIRED_PROMPT_TASKS:
            missing = sorted(REQUIRED_PROMPT_TASKS - tasks)
            unexpected = sorted(tasks - REQUIRED_PROMPT_TASKS)
            raise ValueError(
                f"{role} prompt catalog mismatch; missing={missing}, unexpected={unexpected}"
            )
        for task, template in prompts.items():
            if not template.strip():
                raise ValueError(f"Empty {role} prompt for task: {task}")
            if "DO:" not in template or "DO NOT:" not in template:
                raise ValueError(
                    f"{role} prompt must contain explicit DO and DO NOT guidance: {task}"
                )
            for _, field_name, _, _ in Formatter().parse(template):
                if field_name and not field_name.replace("_", "").isalnum():
                    raise ValueError(
                        f"Unsafe placeholder {field_name!r} in {role} prompt: {task}"
                    )


def clear_prompt_cache() -> None:
    _load_prompt_file.cache_clear()
