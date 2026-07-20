from __future__ import annotations

import json
from typing import Any

from app.clients.document_intelligence import env_value, load_dotenv_file


DEFAULT_RETRIEVAL_TOP_K = 6
MIN_RETRIEVAL_TOP_K = 1
MAX_RETRIEVAL_TOP_K = 10
DEFAULT_RATE_LIMIT_PER_MINUTE = 30
DEFAULT_BASIC_USER_QUESTION_LIMIT = 10
DEFAULT_POWER_USERS = {"ankurkumarj@quantzig.com"}
DEFAULT_BASIC_USERS = {
    "surajc@quantzig.com",
    "sidhus@quantzig.com",
    "akshatameemamshi@quantzig.com",
    "vikasgoyal@quantzig.com",
    "saiprasad@quantzig.com",
}


def app_env() -> str:
    load_dotenv_file()
    return (env_value("APP_ENV") or "local").lower()


def is_local_env() -> bool:
    return app_env() in {"local", "dev", "development", "test"}


def is_databricks_env() -> bool:
    return app_env() in {"databricks", "dbx"}


def cors_origins() -> list[str]:
    load_dotenv_file()
    raw_value = env_value("CORS_ORIGINS")
    if not raw_value:
        return ["http://127.0.0.1:5173", "http://localhost:5173", "http://localhost:8080"] if is_local_env() else []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def retrieval_top_k() -> int:
    load_dotenv_file()
    raw_value = env_value("RETRIEVAL_TOP_K")
    if not raw_value:
        return DEFAULT_RETRIEVAL_TOP_K
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_RETRIEVAL_TOP_K
    return max(MIN_RETRIEVAL_TOP_K, min(MAX_RETRIEVAL_TOP_K, value))


def rate_limit_per_minute() -> int:
    load_dotenv_file()
    raw_value = env_value("RATE_LIMIT_PER_MINUTE")
    if not raw_value:
        return DEFAULT_RATE_LIMIT_PER_MINUTE
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_RATE_LIMIT_PER_MINUTE


def basic_user_question_limit() -> int:
    load_dotenv_file()
    raw_value = env_value("BASIC_USER_QUESTION_LIMIT")
    if not raw_value:
        return DEFAULT_BASIC_USER_QUESTION_LIMIT
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_BASIC_USER_QUESTION_LIMIT


def power_users() -> set[str]:
    return _email_set("POWER_USERS", DEFAULT_POWER_USERS)


def basic_users() -> set[str]:
    return _email_set("BASIC_USERS", DEFAULT_BASIC_USERS)


def _email_set(env_name: str, default: set[str]) -> set[str]:
    load_dotenv_file()
    raw_value = env_value(env_name)
    if not raw_value:
        return set(default)
    return {item.strip().lower() for item in raw_value.split(",") if item.strip()}


def local_dev_email() -> str:
    """Optional override for the local-dev user's email (empty if unset)."""
    load_dotenv_file()
    return (env_value("LOCAL_DEV_EMAIL") or "").strip().lower()


def local_dev_name() -> str:
    """Optional override for the local-dev user's display name (empty if unset)."""
    load_dotenv_file()
    return (env_value("LOCAL_DEV_NAME") or "").strip()


def app_api_keys() -> dict[str, dict[str, Any]]:
    load_dotenv_file()
    raw_value = env_value("APP_API_KEYS")
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def llm_provider() -> str:
    load_dotenv_file()
    return (env_value("LLM_PROVIDER") or "openai").lower()


def databricks_host() -> str:
    load_dotenv_file()
    return (env_value("DATABRICKS_HOST") or env_value("DATABRICKS_WORKSPACE_URL") or "").rstrip("/")


def databricks_token() -> str:
    load_dotenv_file()
    for value in (
        env_value("DATABRICKS_TOKEN"),
        env_value("DATABRICKS_APP_TOKEN"),
        env_value("DATABRICKS_OAUTH_TOKEN"),
    ):
        cleaned = (value or "").strip()
        if cleaned and not cleaned.startswith("<"):
            return cleaned
    return ""


def databricks_chat_endpoint() -> str:
    load_dotenv_file()
    return env_value("DATABRICKS_CHAT_ENDPOINT") or "databricks-claude-sonnet-4"


def databricks_guardrail_endpoint() -> str:
    load_dotenv_file()
    return env_value("DATABRICKS_GUARDRAIL_ENDPOINT") or databricks_chat_endpoint()


def databricks_embedding_endpoint() -> str:
    load_dotenv_file()
    return env_value("DATABRICKS_EMBEDDING_ENDPOINT") or "databricks-bge-large-en"


def databricks_transcription_endpoint() -> str:
    load_dotenv_file()
    return env_value("DATABRICKS_TRANSCRIPTION_ENDPOINT") or "databricks-gemini-3-5-flash"


def databricks_vision_endpoint() -> str:
    load_dotenv_file()
    return env_value("DATABRICKS_VISION_ENDPOINT") or databricks_transcription_endpoint()


def quality_enrichment_provider() -> str:
    load_dotenv_file()
    return (env_value("QUALITY_ENRICHMENT_PROVIDER") or "auto").lower()


def quality_enrichment_model() -> str:
    load_dotenv_file()
    return env_value("QUALITY_ENRICHMENT_MODEL") or databricks_chat_endpoint()


def quality_enrichment_required() -> bool:
    load_dotenv_file()
    value = (env_value("QUALITY_ENRICHMENT_REQUIRED") or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def guardrail_model() -> str:
    load_dotenv_file()
    return env_value("OPENAI_GUARDRAIL_MODEL")


def guardrail_confidence_threshold() -> float:
    load_dotenv_file()
    raw_value = env_value("GUARDRAIL_CONFIDENCE_THRESHOLD")
    if not raw_value:
        return 0.9
    try:
        return max(0.0, min(1.0, float(raw_value)))
    except ValueError:
        return 0.9


def answer_confidence_threshold() -> float:
    load_dotenv_file()
    raw_value = env_value("ANSWER_CONFIDENCE_THRESHOLD")
    if not raw_value:
        return 0.8
    try:
        return max(0.0, min(1.0, float(raw_value)))
    except ValueError:
        return 0.8
