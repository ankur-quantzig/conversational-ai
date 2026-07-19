from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.config import databricks_chat_endpoint, databricks_host, databricks_token, databricks_transcription_endpoint


def runtime_host_token() -> tuple[str, str]:
    try:
        import IPython

        shell = IPython.get_ipython()
        dbutils = shell.user_ns.get("dbutils") if shell else None
        if dbutils:
            context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
            host = ""
            token = ""
            try:
                host = context.apiUrl().get()
            except Exception:
                try:
                    host = f"https://{context.browserHostName().get()}"
                except Exception:
                    host = ""
            try:
                token = context.apiToken().get()
            except Exception:
                token = ""
            return host.rstrip("/"), token
    except Exception:
        pass
    return "", ""


def serving_auth() -> tuple[str, str]:
    host = databricks_host()
    token = databricks_token()
    if host and token:
        return host, token
    runtime_host, runtime_token = runtime_host_token()
    return host or runtime_host, token or runtime_token


def invoke_endpoint(endpoint_name: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    host, token = serving_auth()
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required for Databricks model serving")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required for Databricks model serving")

    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Databricks endpoint `{endpoint_name}` failed with HTTP {exc.code}: {detail[:500]}") from exc


def invoke_chat_completions(payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    host, token = serving_auth()
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required for Databricks model serving")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required for Databricks model serving")

    url = f"{host}/ai-gateway/mlflow/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Databricks chat completions failed with HTTP {exc.code}: {detail[:500]}") from exc


def chat_completion(
    messages: list[dict[str, str]],
    endpoint: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> str:
    endpoint_name = endpoint or databricks_chat_endpoint()
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = invoke_endpoint(endpoint_name, payload, timeout=120)
    return extract_message_content(data)


def embeddings(texts: list[str], endpoint: str | None = None) -> list[list[float]]:
    endpoint_name = endpoint or "databricks-bge-large-en"
    payload = {"input": texts}
    data = invoke_endpoint(endpoint_name, payload, timeout=120)
    return extract_embeddings(data)


def transcribe_audio(
    audio_path: Path,
    endpoint: str | None = None,
    prompt: str | None = None,
    max_tokens: int = 4096,
    timeout: int = 300,
) -> str:
    endpoint_name = endpoint or databricks_transcription_endpoint()
    mime_type = audio_mime_type(audio_path)
    audio_b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("utf-8")
    transcription_prompt = prompt or (
        "Transcribe the spoken audio exactly. Do not summarize, translate, add explanations, "
        "or mention that you are an AI model. Return strict JSON with this shape: "
        '{"text":"full transcript text"}.'
    )
    payload = {
        "model": endpoint_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": transcription_prompt},
                    {"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{audio_b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    data = invoke_chat_completions(payload, timeout=timeout)
    return extract_message_content(data)


def audio_mime_type(audio_path: Path) -> str:
    if audio_path.suffix.lower() == ".mp3":
        return "audio/mp3"
    guessed, _ = mimetypes.guess_type(str(audio_path))
    return guessed or "audio/mpeg"


def extract_embeddings(data: dict[str, Any]) -> list[list[float]]:
    rows = data.get("data")
    if isinstance(rows, list):
        vectors = [row.get("embedding") for row in rows if isinstance(row, dict)]
        if vectors and all(isinstance(vector, list) for vector in vectors):
            return vectors

    predictions = data.get("predictions")
    if isinstance(predictions, list):
        if predictions and all(isinstance(vector, list) for vector in predictions):
            return predictions
        vectors = [row.get("embedding") for row in predictions if isinstance(row, dict)]
        if vectors and all(isinstance(vector, list) for vector in vectors):
            return vectors

    raise RuntimeError("Databricks embedding response did not include embeddings")


def extract_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    if isinstance(data.get("content"), str):
        return data["content"]
    if isinstance(data.get("predictions"), list) and data["predictions"]:
        first = data["predictions"][0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return extract_message_content(first)
    raise RuntimeError("Databricks model serving response did not include message content")
