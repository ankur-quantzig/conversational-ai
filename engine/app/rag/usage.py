from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_usage: ContextVar[dict[str, Any]] = ContextVar(
    "rag_model_usage",
    default={"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "models": []},
)


def reset_model_usage() -> None:
    _usage.set({"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "models": []})


def record_model_usage(payload: Any, *, model: str) -> None:
    usage = payload if isinstance(payload, dict) else {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    current = dict(_usage.get())
    models = list(current.get("models") or [])
    if model and model not in models:
        models.append(model)
    _usage.set(
        {
            "calls": int(current.get("calls") or 0) + 1,
            "input_tokens": int(current.get("input_tokens") or 0) + input_tokens,
            "output_tokens": int(current.get("output_tokens") or 0) + output_tokens,
            "total_tokens": int(current.get("total_tokens") or 0) + total_tokens,
            "models": models,
        }
    )


def model_usage_snapshot() -> dict[str, Any]:
    return dict(_usage.get())
