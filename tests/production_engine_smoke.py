from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source(
    source_id: str,
    doc_id: str,
    content: str,
    *,
    source_type: str = "document",
    pages: list[int] | None = None,
    start: str = "",
    end: str = "",
    score: float = 1.0,
):
    return {
        "id": source_id,
        "doc_id": doc_id,
        "content": content,
        "source_type": source_type,
        "page_numbers": pages or [],
        "start_time_label": start,
        "end_time_label": end,
        "score": score,
    }


def main() -> None:
    from app.api.main import public_progress_payload
    import app.api.retrieval as retrieval_module
    from app.api.retrieval import cached_answer_key
    from app.evaluation.release import evaluate_release_gates
    from app.rag.answer import AnswerCitation, RagAnswer, deterministic_question_preparation
    from app.rag.context import select_context_sources
    from app.rag.grounding import calibrate_answer, validate_citations
    import app.clients.databricks_model_serving as serving
    from app.clients.lancedb_store import index_coverage_status
    from app.services.embed_chunks import DEFAULT_DATABRICKS_INPUT_MAX_CHARS

    for technical_stage in ("bm25_search", "vector_search", "deduplicating_chunks", "reranking_chunks"):
        payload = public_progress_payload({"stage": technical_stage, "request_id": "request-1"})
        rendered = f"{payload['stage']} {payload['message']}".lower()
        assert "bm25" not in rendered
        assert "vector" not in rendered
        assert "chunk" not in rendered
        assert "rerank" not in rendered

    assert index_coverage_status(321, 29)["consistent"] is False
    assert index_coverage_status(321, 321)["consistent"] is True
    assert DEFAULT_DATABRICKS_INPUT_MAX_CHARS > 0

    vague = deterministic_question_preparation("Tell me more")
    assert vague.status == "needs_clarification"
    clear = deterministic_question_preparation("What is multi-head attention?")
    assert clear.status == "ready"

    candidates = [
        source("1", "a", "Alpha evidence about architecture and attention.", pages=[1]),
        source("2", "a", "Alpha evidence about architecture and attention.", pages=[1]),
        source("3", "b", "Independent beta evidence about evaluation results.", pages=[2]),
        source("4", "c", "Independent gamma evidence about limitations.", pages=[3]),
    ]
    selected = select_context_sources(candidates, max_chunks=3, token_budget=1000, per_document_limit=2)
    assert [item["id"] for item in selected] == ["1", "3", "4"]

    answer = RagAnswer(
        heading="Supported Answer",
        answer="- Attention connects relevant sequence positions.",
        citations=[
            AnswerCitation(source_index=1, pages=[1], quote="architecture and attention"),
            AnswerCitation(source_index=99, pages=[], quote="invented evidence"),
        ],
        confidence="high",
        confidence_score=0.95,
        missing_information="",
    )
    valid, rejected = validate_citations(answer, candidates[:1])
    assert len(valid) == 1
    assert rejected == [{"source_index": 99, "reason": "unknown_source"}]
    calibrated, grounding = calibrate_answer(answer, candidates[:1])
    assert len(calibrated.citations) == 1
    assert grounding["valid_citations"] == 1
    assert calibrated.confidence_score >= 0.8

    video_answer = RagAnswer(
        heading="Video Answer",
        answer="- The workflow uses an external service.",
        citations=[AnswerCitation(source_index=1, pages=[], start_time_label="01:00", end_time_label="01:30", quote="external service")],
        confidence="high",
        confidence_score=0.9,
        missing_information="",
    )
    video_source = source("v1", "video", "The workflow uses an external service.", source_type="video", start="01:00", end="01:30")
    assert len(validate_citations(video_answer, [video_source])[0]) == 1

    assert cached_answer_key("q", 6, None, None, "tenant-a") != cached_answer_key("q", 6, None, None, "tenant-b")

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"ok": true}'

    retryable = urllib.error.HTTPError("https://example", 429, "rate limited", {}, io.BytesIO(b'{"error":"busy"}'))
    serving._circuit_state.update({"failures": 0, "opened_at": 0.0})
    with patch.object(serving.urllib.request, "urlopen", side_effect=[retryable, JsonResponse()]), patch.object(
        serving.time, "sleep", return_value=None
    ):
        assert serving._send_json_request(lambda: object(), 1, "test operation") == {"ok": True}

    original_similar = retrieval_module.generate_similar_questions
    original_bm25 = retrieval_module.retrieve_bm25_chunks
    original_vector = retrieval_module.lancedb_retrieve
    original_chunks = retrieval_module.load_chunks
    original_index_status = retrieval_module.vector_index_status
    original_allowlist = retrieval_module.source_doc_allowlist
    try:
        retrieval_module.generate_similar_questions = lambda question: [question]
        retrieval_module.retrieve_bm25_chunks = lambda **kwargs: [
            source("hybrid-1", "doc-a", "Architecture evidence supports the question.", pages=[1]),
            source("hybrid-2", "doc-b", "Evaluation evidence supports the question.", pages=[2]),
        ]
        retrieval_module.lancedb_retrieve = lambda *args, **kwargs: [
            source("hybrid-2", "doc-b", "Evaluation evidence supports the question.", pages=[2]),
            source("hybrid-3", "doc-c", "Limitations evidence supports the question.", pages=[3]),
        ]
        retrieval_module.load_chunks = lambda: []
        retrieval_module.vector_index_status = lambda chunks: {"consistent": True}
        retrieval_module.source_doc_allowlist = lambda: None
        stages = []
        hybrid = retrieval_module.retrieve_hybrid_sources(
            "What architecture and evaluation evidence is available?",
            top_k=3,
            progress=lambda stage, message, metadata: stages.append(stage),
        )
        assert {item["id"] for item in hybrid} == {"hybrid-1", "hybrid-2", "hybrid-3"}
        hybrid_two = next(item for item in hybrid if item["id"] == "hybrid-2")
        assert set(hybrid_two["retrieval"]["methods"]) == {"bm25", "vector"}
        assert "context_selection" in stages
    finally:
        retrieval_module.generate_similar_questions = original_similar
        retrieval_module.retrieve_bm25_chunks = original_bm25
        retrieval_module.lancedb_retrieve = original_vector
        retrieval_module.load_chunks = original_chunks
        retrieval_module.vector_index_status = original_index_status
        retrieval_module.source_doc_allowlist = original_allowlist

    gates = evaluate_release_gates({"hit_rate": 0.9, "mean_reciprocal_rank": 0.8, "expected_doc_recall": 0.8})
    assert gates["passed"] is True
    print("production engine smoke ok")


if __name__ == "__main__":
    main()
