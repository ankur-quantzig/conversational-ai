from __future__ import annotations

import re
from typing import Any

from app.rag.answer import AnswerCitation, RagAnswer


WORD_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,}", re.IGNORECASE)
STOP_WORDS = {
    "about", "and", "are", "for", "from", "into", "that", "the", "this",
    "was", "were", "with", "your",
}


def citation_matches_source(citation: AnswerCitation, source: dict[str, Any]) -> bool:
    source_type = str(source.get("source_type") or "document")
    source_pages = {int(page) for page in source.get("page_numbers") or []}
    citation_pages = {int(page) for page in citation.pages}
    metadata = source.get("metadata") or {}
    source_start = str(source.get("start_time_label") or metadata.get("start_time_label") or "")
    source_end = str(source.get("end_time_label") or metadata.get("end_time_label") or "")

    if source_type == "video":
        if source_start and citation.start_time_label != source_start:
            return False
        if source_end and citation.end_time_label != source_end:
            return False
        if citation.pages:
            return False
    else:
        if citation.start_time_label or citation.end_time_label:
            return False
        if source_pages and (not citation_pages or not citation_pages.issubset(source_pages)):
            return False

    quote_terms = {
        term.lower()
        for term in WORD_RE.findall(citation.quote)
        if term.lower() not in STOP_WORDS
    }
    if not quote_terms:
        return False
    source_text = str(source.get("content") or "").lower()
    return any(term in source_text for term in quote_terms)


def validate_citations(answer: RagAnswer, sources: list[dict[str, Any]]) -> tuple[list[AnswerCitation], list[dict[str, Any]]]:
    valid: list[AnswerCitation] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for citation in answer.citations:
        source_index = citation.source_index
        if not 1 <= source_index <= len(sources):
            rejected.append({"source_index": source_index, "reason": "unknown_source"})
            continue
        source = sources[source_index - 1]
        if not citation_matches_source(citation, source):
            rejected.append({"source_index": source_index, "reason": "metadata_or_evidence_mismatch"})
            continue
        key = (
            source_index,
            tuple(citation.pages),
            citation.start_time_label,
            citation.end_time_label,
            citation.quote.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        valid.append(citation)
    return valid, rejected


def calibrate_answer(answer: RagAnswer, sources: list[dict[str, Any]]) -> tuple[RagAnswer, dict[str, Any]]:
    valid, rejected = validate_citations(answer, sources)
    expected_evidence_count = max(1, min(2, len(sources)))
    supported_source_count = len({citation.source_index for citation in valid})
    citation_coverage = min(1.0, supported_source_count / expected_evidence_count)
    source_strength = min(1.0, len(sources) / 3.0)
    calibrated = (answer.confidence_score * 0.65) + (citation_coverage * 0.25) + (source_strength * 0.10)
    if not valid:
        calibrated = min(calibrated, 0.55)
    calibrated = max(0.0, min(1.0, calibrated))

    answer.citations = valid
    answer.confidence_score = round(calibrated, 4)
    answer.confidence = "high" if calibrated >= 0.85 else "medium" if calibrated >= 0.7 else "low"
    return answer, {
        "valid_citations": len(valid),
        "rejected_citations": rejected,
        "citation_coverage": round(citation_coverage, 4),
        "source_strength": round(source_strength, 4),
        "calibrated_confidence": answer.confidence_score,
    }
