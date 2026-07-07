from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, status

from app.config import rate_limit_per_minute


_REQUESTS: dict[str, deque[float]] = defaultdict(deque)


def enforce_rate_limit(key: str) -> None:
    limit = rate_limit_per_minute()
    now = time.time()
    window_start = now - 60
    requests = _REQUESTS[key]
    while requests and requests[0] < window_start:
        requests.popleft()
    if len(requests) >= limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    requests.append(now)
