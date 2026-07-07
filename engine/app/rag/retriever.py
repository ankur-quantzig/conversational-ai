from __future__ import annotations

import math
from typing import Any

from app.clients.lancedb_store import vector_search
from app.services.embed_chunks import embedding_config, embed_texts


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def local_vector_search(query: str, embedded_chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    model, dimensions = embedding_config()
    query_vector = embed_texts([query], model=model, dimensions=dimensions)[0]
    scored = []
    for chunk in embedded_chunks:
        vector = chunk.get("embedding")
        if not vector:
            continue
        scored.append((cosine_similarity(query_vector, vector), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "content": chunk["content"],
            "content_type": chunk.get("content_type"),
            "section": chunk.get("section"),
            "page_numbers": chunk.get("page_numbers", []),
            "score": score,
            "metadata": chunk.get("metadata", {}),
        }
        for score, chunk in scored[:top_k]
    ]


def lancedb_retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    model, dimensions = embedding_config()
    query_vector = embed_texts([query], model=model, dimensions=dimensions)[0]
    return vector_search(query_vector=query_vector, top_k=top_k)
