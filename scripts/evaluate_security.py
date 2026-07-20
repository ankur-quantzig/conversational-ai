from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.security.guardrails import deterministic_query_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic query-security regression cases.")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/security_cases.v1.jsonl"))
    args = parser.parse_args()
    results = []
    for line_number, line in enumerate(args.dataset.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        actual = deterministic_query_check(case["question"]).is_attack
        results.append({"id": case["id"], "passed": actual == case["expected_attack"], "actual_attack": actual})
    report = {
        "case_count": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": [result for result in results if not result["passed"]],
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not report["failed"] else 1)


if __name__ == "__main__":
    main()
