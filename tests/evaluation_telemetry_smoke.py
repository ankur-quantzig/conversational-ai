from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.api.telemetry import PipelineTrace
    from app.evaluation.retrieval import evaluate_retrieval, load_cases

    cases = load_cases(ROOT / "evaluation" / "golden_questions.v1.jsonl")
    assert len(cases) >= 16
    assert any(case.source_type == "video" for case in cases)
    assert any(case.category == "multi-document" for case in cases)

    expected = cases[0].expected_doc_ids[0]

    def fake_retrieve(**kwargs):
        return [{"doc_id": "irrelevant"}, {"doc_id": expected}]

    report = evaluate_retrieval([cases[0]], fake_retrieve, top_k=10)
    assert report["hit_rate"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.5

    events = []
    trace = PipelineTrace(provider="test", trace_id="trace-1")
    trace.progress("embedding_query", "Embedding", downstream=lambda stage, message, metadata: events.append(stage))
    trace.progress("retrieving_chunks", "Retrieving")
    telemetry = trace.finish(mode="test", source_count=2)
    assert events == ["embedding_query"]
    assert telemetry["schema_version"] == "1.0"
    assert telemetry["trace_id"] == "trace-1"
    assert telemetry["source_count"] == 2
    assert [stage["stage"] for stage in telemetry["stages"]] == ["embedding_query", "retrieving_chunks"]
    print("evaluation telemetry smoke ok")


if __name__ == "__main__":
    main()
