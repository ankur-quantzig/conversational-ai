from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.retrieval import retrieve_bm25_chunks, retrieve_chunks
from app.evaluation.retrieval import evaluate_retrieval, load_cases


DEFAULT_DATASET = Path("evaluation/golden_questions.v1.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval against the versioned golden question set.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--method", choices=("bm25", "keyword"), default="bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    retrieve = retrieve_bm25_chunks if args.method == "bm25" else retrieve_chunks
    report = {
        "dataset": str(args.dataset),
        "method": args.method,
        **evaluate_retrieval(cases, retrieve, top_k=max(1, args.top_k)),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "method": report["method"],
                "case_count": report["case_count"],
                "top_k": report["top_k"],
                "hit_rate": report["hit_rate"],
                "full_coverage_rate": report["full_coverage_rate"],
                "expected_doc_recall": report["expected_doc_recall"],
                "mean_reciprocal_rank": report["mean_reciprocal_rank"],
                "total_ms": report["total_ms"],
                "output": str(args.output) if args.output else "",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
