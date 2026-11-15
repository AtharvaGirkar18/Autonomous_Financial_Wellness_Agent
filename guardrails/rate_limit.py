"""Rate limiting and call budgeting (abuse / cost / denial-of-wallet protection)."""
from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    """Sliding-window limiter. One instance per session/key."""

    def __init__(self, max_calls: int = 12, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._hits: deque[float] = deque()

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        while self._hits and now - self._hits[0] > self.window:
            self._hits.popleft()
        if len(self._hits) >= self.max_calls:
            return False
        self._hits.append(now)
        return True

    def retry_after(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if not self._hits:
            return 0.0
        return max(0.0, self.window - (now - self._hits[0]))
