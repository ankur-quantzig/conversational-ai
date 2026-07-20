from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class Result:
    ok: bool
    latency_ms: int
    status: int
    error: str = ""


def send_question(base_url: str, token: str, question: str, timeout: int) -> Result:
    payload = json.dumps({"question": question, "source_type": None}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat", data=payload, headers=headers, method="POST")
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            ok = response.status == 200 and bool(body.get("answer"))
            return Result(ok=ok, latency_ms=round((time.perf_counter() - started_at) * 1000), status=response.status)
    except urllib.error.HTTPError as exc:
        return Result(False, round((time.perf_counter() - started_at) * 1000), exc.code, f"HTTP {exc.code}")
    except Exception as exc:
        return Result(False, round((time.perf_counter() - started_at) * 1000), 0, type(exc).__name__)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded concurrent load test against the JSON chat endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--question", default="What are the key ideas in the selected sources?")
    args = parser.parse_args()
    token = os.getenv("LOAD_TEST_BEARER_TOKEN") or os.getenv("LOAD_TEST_API_KEY") or ""

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(send_question, args.base_url, token, args.question, args.timeout)
            for _ in range(max(1, args.requests))
        ]
        results = [future.result() for future in as_completed(futures)]

    latencies = [result.latency_ms for result in results]
    failures = [result for result in results if not result.ok]
    report = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "success_rate": round((len(results) - len(failures)) / len(results), 4),
        "latency_ms": {
            "mean": round(statistics.mean(latencies)),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "failures": [{"status": result.status, "error": result.error} for result in failures[:10]],
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["success_rate"] >= 0.98 else 1)


if __name__ == "__main__":
    main()
