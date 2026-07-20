from __future__ import annotations

from typing import Any


DEFAULT_GATES = {
    "retrieval_hit_rate": 0.85,
    "retrieval_mrr": 0.65,
    "expected_doc_recall": 0.75,
    "grounded_answer_rate": 0.90,
    "citation_precision": 0.95,
    "p95_latency_ms": 8000,
}


def evaluate_release_gates(
    retrieval_report: dict[str, Any],
    rag_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = [
        {
            "metric": "retrieval_hit_rate",
            "actual": float(retrieval_report.get("hit_rate") or 0.0),
            "target": DEFAULT_GATES["retrieval_hit_rate"],
            "passed": float(retrieval_report.get("hit_rate") or 0.0) >= DEFAULT_GATES["retrieval_hit_rate"],
        },
        {
            "metric": "expected_doc_recall",
            "actual": float(retrieval_report.get("expected_doc_recall") or 0.0),
            "target": DEFAULT_GATES["expected_doc_recall"],
            "passed": float(retrieval_report.get("expected_doc_recall") or 0.0) >= DEFAULT_GATES["expected_doc_recall"],
        },
        {
            "metric": "retrieval_mrr",
            "actual": float(retrieval_report.get("mean_reciprocal_rank") or 0.0),
            "target": DEFAULT_GATES["retrieval_mrr"],
            "passed": float(retrieval_report.get("mean_reciprocal_rank") or 0.0) >= DEFAULT_GATES["retrieval_mrr"],
        },
    ]
    if rag_report:
        checks.extend(
            [
                {
                    "metric": "grounded_answer_rate",
                    "actual": float(rag_report.get("grounded_answer_rate") or 0.0),
                    "target": DEFAULT_GATES["grounded_answer_rate"],
                    "passed": float(rag_report.get("grounded_answer_rate") or 0.0) >= DEFAULT_GATES["grounded_answer_rate"],
                },
                {
                    "metric": "citation_precision",
                    "actual": float(rag_report.get("citation_precision") or 0.0),
                    "target": DEFAULT_GATES["citation_precision"],
                    "passed": float(rag_report.get("citation_precision") or 0.0) >= DEFAULT_GATES["citation_precision"],
                },
                {
                    "metric": "p95_latency_ms",
                    "actual": int(rag_report.get("p95_latency_ms") or 0),
                    "target": DEFAULT_GATES["p95_latency_ms"],
                    "passed": int(rag_report.get("p95_latency_ms") or 0) <= DEFAULT_GATES["p95_latency_ms"],
                },
            ]
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}
