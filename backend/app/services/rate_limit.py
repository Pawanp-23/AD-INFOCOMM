"""In-process token-bucket rate limiter (TRD SEC-06). Redis-backed in multi-instance deployments."""

import threading
import time

from fastapi import HTTPException, status


class TokenBucket:
    def __init__(self, capacity: int = 10, refill_per_s: float = 10 / 60) -> None:
        self.capacity, self.refill = capacity, refill_per_s
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens < 1:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts, try again shortly"
                )
            self._buckets[key] = (tokens - 1, now)


limiter = TokenBucket()
