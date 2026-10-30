"""In-process sliding-window rate limiter (standard library). One process, one window per client address; good enough for a local or
single-instance deployment. Behind several workers each has its own counters, so the effective limit is per worker."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque


class SlidingWindowLimiter:
    def __init__(self, window_s: float = 60.0, max_clients: int = 5000, clock=time.monotonic) -> None:
        self.window_s, self.max_clients, self._clock = window_s, max_clients, clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, client: str, limit: int) -> tuple[bool, float]:
        """Record an attempt. Returns (allowed, seconds_until_a_slot_frees). Denied attempts are not recorded."""
        now = self._clock()
        with self._lock:
            q = self._hits.get(client)
            if q is None:
                q = self._hits[client] = deque()
                while len(self._hits) > self.max_clients:         # bounded memory: forget the least recently active client
                    self._hits.popitem(last=False)
            self._hits.move_to_end(client)
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= limit:
                return False, max(1.0, self.window_s - (now - q[0]))
            q.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
