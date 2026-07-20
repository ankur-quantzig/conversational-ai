from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.release import evaluate_release_gates


def load_report(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RAG quality reports against production release gates.")
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--rag-report", type=Path)
    args = parser.parse_args()
    result = evaluate_release_gates(load_report(args.retrieval_report) or {}, load_report(args.rag_report))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
