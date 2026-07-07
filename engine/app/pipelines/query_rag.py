from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag.answer import answer_question
from app.rag.retriever import lancedb_retrieve, local_vector_search
from app.services.chunk_document import load_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("question")
    parser.add_argument("--embedded-jsonl", type=Path, help="Use local embedded chunks instead of LanceDB.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-answer", action="store_true")
    args = parser.parse_args()

    if args.embedded_jsonl:
        results = local_vector_search(args.question, load_chunks_jsonl(args.embedded_jsonl), top_k=args.top_k)
    else:
        results = lancedb_retrieve(args.question, top_k=args.top_k)
    payload = {"question": args.question, "results": results}
    if not args.no_answer:
        payload["answer"] = answer_question(args.question, results)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
