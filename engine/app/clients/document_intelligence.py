from __future__ import annotations

import os
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from app.utils.files import project_root


def env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def load_dotenv_file() -> None:
    env_path = project_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)
        if key == "OPANAI_API_KEY":
            os.environ.setdefault("OPENAI_API_KEY", value)


def client_from_env() -> DocumentIntelligenceClient:
    load_dotenv_file()
    endpoint = env_value("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "DOCUMENT_INTELLIGENCE_ENDPOINT", "AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = env_value("AZURE_DOCUMENT_INTELLIGENCE_KEY", "DOCUMENT_INTELLIGENCE_KEY", "AZURE_FORM_RECOGNIZER_KEY")
    if not endpoint or not key:
        raise RuntimeError("Missing Azure Document Intelligence endpoint/key in .env")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

