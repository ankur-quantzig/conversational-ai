from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from typing import Any


WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def estimate_context_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def content_fingerprint(source: dict[str, Any]) -> frozenset[str]:
    words = WORD_RE.findall(str(source.get("content") or "").lower())
    return frozenset(words[:400])


def content_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def select_context_sources(
    sources: list[dict[str, Any]],
    *,
    max_chunks: int,
    token_budget: int,
    per_document_limit: int = 3,
    duplicate_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    fingerprints: list[frozenset[str]] = []
    document_counts: defaultdict[str, int] = defaultdict(int)
    used_tokens = 0

    for source in sources:
        if len(selected) >= max_chunks:
            break
        doc_id = str(source.get("doc_id") or "unknown")
        if document_counts[doc_id] >= per_document_limit:
            continue
        fingerprint = content_fingerprint(source)
        is_structured_table = str(source.get("content_type") or "").lower() == "table"
        if not is_structured_table and any(
            content_similarity(fingerprint, previous) >= duplicate_threshold for previous in fingerprints
        ):
            continue
        source_tokens = estimate_context_tokens(str(source.get("content") or ""))
        if selected and used_tokens + source_tokens > token_budget:
            continue
        selected.append(source)
        fingerprints.append(fingerprint)
        document_counts[doc_id] += 1
        used_tokens += source_tokens

    if not selected and sources:
        selected = [sources[0]]
        used_tokens = estimate_context_tokens(str(sources[0].get("content") or ""))

    for source in selected:
        retrieval = dict(source.get("retrieval") or {})
        retrieval["context_tokens"] = estimate_context_tokens(str(source.get("content") or ""))
        source["retrieval"] = retrieval
    return selected


def context_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chunks": len(sources),
        "documents": len({str(source.get("doc_id") or "") for source in sources}),
        "tokens": sum(estimate_context_tokens(str(source.get("content") or "")) for source in sources),
    }


def expand_adjacent_sources(
    ranked_sources: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    *,
    max_neighbors: int = 1,
) -> list[dict[str, Any]]:
    by_document: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in corpus:
        by_document[str(chunk.get("doc_id") or "")].append(chunk)
    positions = {
        str(chunk.get("id") or ""): (doc_id, index)
        for doc_id, chunks in by_document.items()
        for index, chunk in enumerate(chunks)
    }

    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in ranked_sources:
        source_id = str(source.get("id") or "")
        if source_id and source_id not in seen:
            expanded.append(source)
            seen.add(source_id)
        position = positions.get(source_id)
        if not position:
            continue
        doc_id, index = position
        chunks = by_document[doc_id]
        for offset in range(1, max_neighbors + 1):
            for neighbor_index in (index - offset, index + offset):
                if neighbor_index < 0 or neighbor_index >= len(chunks):
                    continue
                neighbor = chunks[neighbor_index]
                neighbor_id = str(neighbor.get("id") or "")
                if not neighbor_id or neighbor_id in seen:
                    continue
                item = deepcopy(neighbor)
                metadata = item.get("metadata") or {}
                item["source_type"] = item.get("source_type") or metadata.get("source_type") or "document"
                item["title"] = source.get("title") or item.get("title")
                item["score"] = float(source.get("score") or 0.0) * 0.95
                item["retrieval"] = {"methods": ["adjacent_context"], "parent_source_id": source_id}
                expanded.append(item)
                seen.add(neighbor_id)
    return expanded
