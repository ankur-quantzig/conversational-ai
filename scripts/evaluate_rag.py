from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.retrieval import answer_question
from app.evaluation.retrieval import load_cases


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation using the configured model provider.")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/golden_questions.v1.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases; 0 runs all cases.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/rag-report.json"))
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    results = []
    latencies = []
    valid_citations = 0
    rejected_citations = 0
    grounded_answers = 0

    for case in cases:
        started_at = time.perf_counter()
        try:
            response = answer_question(
                question=case.question,
                top_k=args.top_k,
                doc_id=case.doc_id,
                source_type=case.source_type,
                cache_namespace="evaluation",
            )
            error_type = ""
        except Exception as exc:
            response = {"mode": "error", "sources": [], "citations": [], "telemetry": {}}
            error_type = type(exc).__name__
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        latencies.append(latency_ms)
        grounding = response.get("grounding") or {}
        valid_count = int(grounding.get("valid_citations") or 0)
        rejected_count = len(grounding.get("rejected_citations") or [])
        valid_citations += valid_count
        rejected_citations += rejected_count
        grounded = response.get("mode") != "insufficient_evidence" and valid_count > 0
        grounded_answers += int(grounded)
        results.append(
            {
                "id": case.id,
                "mode": response.get("mode"),
                "grounded": grounded,
                "confidence": response.get("confidence"),
                "confidence_score": response.get("confidence_score"),
                "source_count": len(response.get("sources") or []),
                "valid_citations": valid_count,
                "rejected_citations": rejected_count,
                "latency_ms": latency_ms,
                "telemetry": response.get("telemetry") or {},
                "error_type": error_type,
            }
        )

    citation_total = valid_citations + rejected_citations
    report = {
        "schema_version": "1.0",
        "dataset": str(args.dataset),
        "case_count": len(cases),
        "grounded_answer_rate": round(grounded_answers / max(1, len(cases)), 4),
        "citation_precision": round(valid_citations / max(1, citation_total), 4),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
