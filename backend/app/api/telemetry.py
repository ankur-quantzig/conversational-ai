from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable


ProgressCallback = Callable[[str, str, dict[str, Any] | None], None]
_metrics_lock = RLock()
_recent_metrics: deque[dict[str, Any]] = deque(maxlen=500)


def record_pipeline_metric(telemetry: dict[str, Any]) -> None:
    with _metrics_lock:
        _recent_metrics.append(dict(telemetry))


def pipeline_metrics_snapshot() -> dict[str, Any]:
    with _metrics_lock:
        metrics = list(_recent_metrics)
    if not metrics:
        return {"requests": 0, "modes": {}, "cache_hits": 0, "fallbacks": 0, "errors": 0, "latency_ms": {}}
    latencies = sorted(int(metric.get("total_ms") or 0) for metric in metrics)

    def percentile(fraction: float) -> int:
        index = min(len(latencies) - 1, max(0, round((len(latencies) - 1) * fraction)))
        return latencies[index]

    return {
        "requests": len(metrics),
        "modes": dict(Counter(str(metric.get("mode") or "unknown") for metric in metrics)),
        "cache_hits": sum(bool(metric.get("cache_hit")) for metric in metrics),
        "fallbacks": sum(bool(metric.get("fallback_used")) for metric in metrics),
        "errors": sum(bool(metric.get("error_type")) for metric in metrics),
        "model_usage": {
            "calls": sum(int((metric.get("model_usage") or {}).get("calls") or 0) for metric in metrics),
            "total_tokens": sum(int((metric.get("model_usage") or {}).get("total_tokens") or 0) for metric in metrics),
        },
        "latency_ms": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": latencies[-1],
        },
    }


@dataclass
class PipelineTrace:
    provider: str
    trace_id: str = ""
    started_at: float = field(default_factory=time.perf_counter)
    _active_stage: str = ""
    _active_stage_started_at: float = 0.0
    _stages: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    fallback_used: bool = False

    def progress(
        self,
        stage: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        downstream: ProgressCallback | None = None,
    ) -> None:
        now = time.perf_counter()
        if self._active_stage:
            self._close_active_stage(now)
        self._active_stage = stage
        self._active_stage_started_at = now
        self.cache_hit = self.cache_hit or stage == "cache_hit"
        self.fallback_used = self.fallback_used or stage == "fallback_search"
        if downstream is not None:
            downstream(stage, message, metadata or {})

    def finish(
        self,
        *,
        mode: str,
        source_count: int,
        error_type: str = "",
        model_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finished_at = time.perf_counter()
        if self._active_stage:
            self._close_active_stage(finished_at)
        return {
            "schema_version": "1.0",
            "provider": self.provider,
            "trace_id": self.trace_id,
            "mode": mode,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "source_count": source_count,
            "total_ms": max(0, round((finished_at - self.started_at) * 1000)),
            "stages": list(self._stages),
            "error_type": error_type,
            "model_usage": model_usage or {},
        }

    def _close_active_stage(self, finished_at: float) -> None:
        self._stages.append(
            {
                "stage": self._active_stage,
                "duration_ms": max(0, round((finished_at - self._active_stage_started_at) * 1000)),
            }
        )
        self._active_stage = ""
        self._active_stage_started_at = 0.0
