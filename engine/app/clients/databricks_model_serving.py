from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import databricks_chat_endpoint, databricks_host, databricks_token, databricks_transcription_endpoint, databricks_vision_endpoint
from app.rag.usage import record_model_usage


# Cache for the OAuth token minted from Databricks Apps service-principal credentials.
_oauth_token_cache: dict[str, Any] = {"token": "", "expires_at": 0.0}
_circuit_state: dict[str, Any] = {"failures": 0, "opened_at": 0.0}
_circuit_lock = RLock()
MAX_RETRIES = 3
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RESET_SECONDS = 30


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


def oauth_token_from_app_credentials(host: str) -> str:
    """Mint a short-lived bearer token from Databricks Apps service-principal credentials.

    Databricks Apps inject DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET for the
    app's service principal. Exchange them for an OAuth token via the workspace OIDC
    token endpoint (client-credentials grant). This is how a non-notebook runtime
    (uvicorn in a Databricks App) authenticates to serving endpoints, since no PAT is
    configured and no dbutils runtime token is available.
    """
    client_id = (os.getenv("DATABRICKS_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("DATABRICKS_CLIENT_SECRET") or "").strip()
    if not host or not client_id or not client_secret:
        return ""

    now = time.time()
    cached = _oauth_token_cache
    if cached["token"] and cached["expires_at"] - 60 > now:
        return cached["token"]

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    body = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": "all-apis"}).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/oidc/v1/token",
        data=body,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""

    token = data.get("access_token", "")
    if token:
        cached["token"] = token
        cached["expires_at"] = now + float(data.get("expires_in", 3600))
    return token


def serving_auth() -> tuple[str, str]:
    host = databricks_host()
    token = databricks_token()
    if host and token:
        return host, token
    runtime_host, runtime_token = runtime_host_token()
    host = host or runtime_host
    token = token or runtime_token
    if host and not token:
        token = oauth_token_from_app_credentials(host)
    return host, token


def _circuit_allows_request() -> bool:
    with _circuit_lock:
        if _circuit_state["failures"] < CIRCUIT_FAILURE_THRESHOLD:
            return True
        if time.monotonic() - _circuit_state["opened_at"] >= CIRCUIT_RESET_SECONDS:
            _circuit_state["failures"] = 0
            _circuit_state["opened_at"] = 0.0
            return True
        return False


def _record_serving_success() -> None:
    with _circuit_lock:
        _circuit_state["failures"] = 0
        _circuit_state["opened_at"] = 0.0


def _record_serving_failure() -> None:
    with _circuit_lock:
        _circuit_state["failures"] += 1
        if _circuit_state["failures"] >= CIRCUIT_FAILURE_THRESHOLD:
            _circuit_state["opened_at"] = time.monotonic()


def _send_json_request(request_factory, timeout: int, operation: str) -> dict[str, Any]:
    if not _circuit_allows_request():
        raise RuntimeError(f"{operation} is temporarily unavailable after repeated provider failures")

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        request = request_factory()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            _record_serving_success()
            return data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"{operation} failed with HTTP {exc.code}: {detail[:500]}")
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if exc.code == 401:
                _oauth_token_cache["token"] = ""
                _oauth_token_cache["expires_at"] = 0.0
            if not retryable or attempt + 1 >= MAX_RETRIES:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = RuntimeError(f"{operation} failed: {type(exc).__name__}")
            if attempt + 1 >= MAX_RETRIES:
                break
        time.sleep(min(2.0, (0.25 * (2**attempt)) + random.uniform(0.0, 0.15)))

    _record_serving_failure()
    raise last_error or RuntimeError(f"{operation} failed")


def invoke_endpoint(endpoint_name: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    def request_factory():
        host, token = serving_auth()
        if not host:
            raise RuntimeError("DATABRICKS_HOST is required for Databricks model serving")
        if not token:
            raise RuntimeError("Databricks model-serving authentication is unavailable")
        return urllib.request.Request(
            f"{host}/serving-endpoints/{endpoint_name}/invocations",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )

    return _send_json_request(request_factory, timeout, f"Databricks endpoint `{endpoint_name}`")


def invoke_chat_completions(payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    def request_factory():
        host, token = serving_auth()
        if not host:
            raise RuntimeError("DATABRICKS_HOST is required for Databricks model serving")
        if not token:
            raise RuntimeError("Databricks model-serving authentication is unavailable")
        return urllib.request.Request(
            f"{host}/ai-gateway/mlflow/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )

    return _send_json_request(request_factory, timeout, "Databricks chat completions")


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
    record_model_usage(data.get("usage") or {}, model=endpoint_name)
    return extract_message_content(data)


def embeddings(texts: list[str], endpoint: str | None = None) -> list[list[float]]:
    endpoint_name = endpoint or "databricks-bge-large-en"
    payload = {"input": texts}
    data = invoke_endpoint(endpoint_name, payload, timeout=120)
    record_model_usage(data.get("usage") or {}, model=endpoint_name)
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


def analyze_image(
    image_path: Path,
    prompt: str,
    endpoint: str | None = None,
    max_tokens: int = 1400,
    timeout: int = 180,
) -> str:
    endpoint_name = endpoint or databricks_vision_endpoint()
    mime_type = image_mime_type(image_path)
    image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": endpoint_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
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


def image_mime_type(image_path: Path) -> str:
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if image_path.suffix.lower() == ".png":
        return "image/png"
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "image/jpeg"


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
