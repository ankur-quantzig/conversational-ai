from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


RetrieveFunction = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    question: str
    category: str
    expected_doc_ids: tuple[str, ...] = field(default_factory=tuple)
    source_type: str | None = None
    doc_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalCase":
        case_id = str(payload.get("id") or "").strip()
        question = str(payload.get("question") or "").strip()
        category = str(payload.get("category") or "").strip()
        if not case_id or not question or not category:
            raise ValueError("Evaluation cases require non-empty id, question, and category.")
        expected_doc_ids = tuple(str(value).strip() for value in payload.get("expected_doc_ids") or [] if str(value).strip())
        if not expected_doc_ids:
            raise ValueError(f"Evaluation case {case_id!r} requires at least one expected_doc_id.")
        source_type = payload.get("source_type")
        if source_type not in (None, "document", "video"):
            raise ValueError(f"Evaluation case {case_id!r} has invalid source_type {source_type!r}.")
        return cls(
            id=case_id,
            question=question,
            category=category,
            expected_doc_ids=expected_doc_ids,
            source_type=source_type,
            doc_id=payload.get("doc_id"),
            tags=tuple(str(value) for value in payload.get("tags") or []),
        )


def load_cases(path: Path) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = RetrievalCase.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if case.id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate case id {case.id!r}.")
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}.")
    return cases


def evaluate_retrieval(
    cases: list[RetrievalCase],
    retrieve: RetrieveFunction,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    results = []
    total_reciprocal_rank = 0.0
    hits = 0
    full_coverage_hits = 0
    total_expected_recall = 0.0
    started_at = time.perf_counter()

    for case in cases:
        case_started_at = time.perf_counter()
        sources = retrieve(
            question=case.question,
            top_k=top_k,
            doc_id=case.doc_id,
            source_type=case.source_type,
        )
        ranked_doc_ids = [str(source.get("doc_id") or "") for source in sources]
        first_relevant_rank = next(
            (rank for rank, doc_id in enumerate(ranked_doc_ids, 1) if doc_id in case.expected_doc_ids),
            None,
        )
        reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        hit = first_relevant_rank is not None
        matched_doc_ids = set(ranked_doc_ids) & set(case.expected_doc_ids)
        expected_doc_recall = len(matched_doc_ids) / len(case.expected_doc_ids)
        full_coverage = expected_doc_recall == 1.0
        hits += int(hit)
        full_coverage_hits += int(full_coverage)
        total_expected_recall += expected_doc_recall
        total_reciprocal_rank += reciprocal_rank
        results.append(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "hit": hit,
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": round(reciprocal_rank, 4),
                "expected_doc_recall": round(expected_doc_recall, 4),
                "full_coverage": full_coverage,
                "latency_ms": round((time.perf_counter() - case_started_at) * 1000),
                "expected_doc_ids": list(case.expected_doc_ids),
                "retrieved_doc_ids": ranked_doc_ids,
            }
        )

    case_count = len(cases)
    return {
        "schema_version": "1.0",
        "case_count": case_count,
        "top_k": top_k,
        "hit_rate": round(hits / case_count, 4),
        "full_coverage_rate": round(full_coverage_hits / case_count, 4),
        "expected_doc_recall": round(total_expected_recall / case_count, 4),
        "mean_reciprocal_rank": round(total_reciprocal_rank / case_count, 4),
        "total_ms": round((time.perf_counter() - started_at) * 1000),
        "results": results,
    }
