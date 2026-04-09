"""Retry with exponential backoff and circuit breaker."""

from __future__ import annotations

import asyncio
import time
from typing import Any


class CircuitBreaker:
    """Simple circuit breaker: closed → open (after N failures) → half-open (after timeout)."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._last_failure_time = 0.0
        self._state: str = "closed"  # closed | open | half-open

    @property
    def state(self) -> str:
        return self._state

    def can_execute(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._last_failure_time >= self._reset_timeout:
                self._state = "half-open"
                return True
            return False
        return False  # half-open: block until probe completes

    def on_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def on_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self._failure_threshold:
            self._state = "open"


async def retry_with_backoff(
    coro_factory: Any,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    retryable_check: Any = None,
) -> Any:
    """Execute an async callable with exponential backoff.

    Args:
        coro_factory: Async callable (called each attempt).
        max_retries: Max retry attempts (total attempts = max_retries + 1).
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.
        retryable_check: Optional callable(exception) -> bool.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if retryable_check and not retryable_check(e):
                raise
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]
