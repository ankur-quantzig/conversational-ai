from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.clients.databricks_model_serving import embeddings as databricks_embeddings
from app.clients.document_intelligence import env_value, load_dotenv_file
from app.config import databricks_embedding_endpoint, llm_provider
from app.services.chunk_document import load_chunks_jsonl, write_jsonl
from app.rag.usage import record_model_usage
from app.utils.files import output_dir


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_BATCH_SIZE = 64
DEFAULT_DATABRICKS_INPUT_MAX_CHARS = 1800


def embedding_config() -> tuple[str, int]:
    load_dotenv_file()
    if llm_provider() == "databricks":
        return databricks_embedding_endpoint(), 0
    model = env_value("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    dimensions = int(env_value("OPENAI_EMBEDDING_DIMENSIONS") or DEFAULT_EMBEDDING_DIMENSIONS)
    return model, dimensions


def embed_texts(texts: list[str], model: str, dimensions: int, batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    load_dotenv_file()
    if llm_provider() == "databricks":
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(databricks_embeddings(texts[start : start + batch_size], endpoint=model))
        return vectors

    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"), max_retries=2, timeout=30)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch, dimensions=dimensions)
        usage = getattr(response, "usage", None)
        record_model_usage(
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            },
            model=model,
        )
        vectors.extend([item.embedding for item in response.data])
    return vectors


def embed_chunks(chunks: list[dict[str, Any]], model: str | None = None, dimensions: int | None = None) -> list[dict[str, Any]]:
    configured_model, configured_dimensions = embedding_config()
    model = model or configured_model
    dimensions = dimensions or configured_dimensions
    max_chars = int(
        os.getenv("EMBEDDING_INPUT_MAX_CHARS")
        or (DEFAULT_DATABRICKS_INPUT_MAX_CHARS if llm_provider() == "databricks" else 0)
    )
    texts = [
        str(chunk["content"])[:max_chars] if max_chars > 0 else str(chunk["content"])
        for chunk in chunks
    ]
    vectors = embed_texts(texts, model=model, dimensions=dimensions)
    embedded = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        item = dict(chunk)
        item["embedding"] = vector
        item["embedding_model"] = model
        item["embedding_dimensions"] = dimensions
        embedded.append(item)
    return embedded


def embed_chunks_file(chunk_path: Path, model: str | None = None, dimensions: int | None = None) -> tuple[list[dict[str, Any]], Path]:
    chunks = load_chunks_jsonl(chunk_path)
    embedded = embed_chunks(chunks, model=model, dimensions=dimensions)
    output_path = output_dir("embeddings") / chunk_path.name.replace("-chunks.jsonl", "-embedded.jsonl")
    write_jsonl(output_path, embedded)
    summary_path = output_dir("embeddings") / chunk_path.name.replace("-chunks.jsonl", "-embedding-summary.json")
    summary = {
        "chunk_count": len(embedded),
        "embedding_model": embedded[0].get("embedding_model") if embedded else model,
        "embedding_dimensions": embedded[0].get("embedding_dimensions") if embedded else dimensions,
        "output": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return embedded, output_path
