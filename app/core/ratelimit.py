"""In-memory sliding-window rate limiter — no external dependencies."""

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

from app.core.config import settings


def _parse_rate(rate_str: str) -> tuple[int, float]:
    """Parse '60/minute' → (60, 60.0). Supported units: second, minute, hour, day."""
    parts = rate_str.split("/", 1)
    count = int(parts[0])
    unit = parts[1].strip().lower() if len(parts) > 1 else "minute"
    windows: dict[str, float] = {
        "second": 1.0,
        "minute": 60.0,
        "hour": 3600.0,
        "day": 86400.0,
    }
    window_s = windows.get(unit)
    if window_s is None:
        raise ValueError(f"Unknown rate-limit unit '{unit}'. Use: second, minute, hour, day.")
    return count, window_s


class _SlidingWindowLimiter:
    """Thread-safe sliding-window counter, keyed by client identifier."""

    def __init__(self, max_requests: int, window_s: float) -> None:
        self._max = max_requests
        self._window = window_s
        self._store: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            q = self._store[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max:
                return False
            q.append(now)
            return True


_max_req, _window_s = _parse_rate(settings.RATE_LIMIT)
_limiter = _SlidingWindowLimiter(max_requests=_max_req, window_s=_window_s)


async def rate_limit(request: Request) -> None:
    """FastAPI dependency: enforce per-IP sliding-window rate limit."""
    client_ip = request.client.host if request.client else "unknown"
    if not _limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Allowed: {settings.RATE_LIMIT}.",
            headers={"Retry-After": str(int(_window_s))},
        )
