"""Mnemoverse SDK error types."""

from __future__ import annotations


class MnemoError(Exception):
    """Base error for all Mnemoverse API errors."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class MnemoAuthError(MnemoError):
    """Invalid or missing API key (401/403)."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status=401)


class MnemoRateLimitError(MnemoError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None) -> None:
        super().__init__(message, status=429)
        self.retry_after = retry_after


class MnemoUnavailableError(MnemoError):
    """Service unreachable — circuit breaker open, network error, or timeout."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message, status=None)
        self.__cause__ = cause
