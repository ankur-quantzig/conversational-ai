from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.config import databricks_chat_endpoint, databricks_host, databricks_token


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


def chat_completion(
    messages: list[dict[str, str]],
    endpoint: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> str:
    host, token = serving_auth()
    endpoint_name = endpoint or databricks_chat_endpoint()
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required for Databricks model serving")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required for Databricks model serving")

    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
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
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Databricks model serving failed with HTTP {exc.code}: {detail[:500]}") from exc

    return extract_message_content(data)


def embeddings(texts: list[str], endpoint: str | None = None) -> list[list[float]]:
    host, token = serving_auth()
    endpoint_name = endpoint or "databricks-bge-large-en"
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required for Databricks model serving")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required for Databricks model serving")

    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    payload = {"input": texts}
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
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Databricks embedding endpoint failed with HTTP {exc.code}: {detail[:500]}") from exc

    return extract_embeddings(data)


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
