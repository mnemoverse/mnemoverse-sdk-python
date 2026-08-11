"""Async Mnemoverse client using httpx."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import httpx

from mnemoverse._retry import CircuitBreaker, retry_with_backoff
from mnemoverse.errors import (
    MnemoAuthError,
    MnemoError,
    MnemoRateLimitError,
    MnemoUnavailableError,
)
from mnemoverse.types import (
    FeedbackResponse,
    HealthResponse,
    ReadResponse,
    RecentResponse,
    StatsResponse,
    WriteBatchResponse,
    WriteResponse,
)

_DEFAULT_BASE_URL = "https://core.mnemoverse.com"


def _isoformat(value: datetime | str) -> str:
    """Accept either a datetime or an already-formatted string.

    Callers reach for both: a datetime when they computed the boundary, a string
    when they are echoing a watermark the server gave them. Rejecting the string
    form would make round-tripping a cursor-adjacent value needlessly awkward.
    """
    return value.isoformat() if isinstance(value, datetime) else value


# FastAPI prefixes each error location with where it was found; keeping it would
# turn "content" into "body.content" for no gain to the reader.
_LOC_SOURCES = ("body", "query", "path", "header", "cookie")


def _error_field(loc: Any) -> str:
    """Name the offending field from a validation error's ``loc`` path."""
    if not isinstance(loc, list) or not loc:
        return ""
    parts = [str(p) for p in loc]
    if len(parts) > 1 and parts[0] in _LOC_SOURCES:
        parts = parts[1:]
    return ".".join(parts)


def _format_validation_errors(data: Any) -> str:
    """Pull the field names and the limits out of a core error body.

    Core answers a rejected request with a generic ``message`` — "Request
    validation failed" — and puts everything that identifies the problem in
    ``details.errors``: which field, and the number it exceeded
    (mnemoverse-core/src/mnemo/api/server.py:358-379). Reporting only the
    top-level message tells the caller that something was wrong but not what,
    which is the difference between a fixable error and a mystery.

    Returns an empty string for any body that is not shaped this way, so
    non-validation errors and other services' error formats fall through
    untouched.
    """
    details = data.get("details") if isinstance(data, dict) else None
    errors = details.get("errors") if isinstance(details, dict) else None
    if not isinstance(errors, list):
        return ""

    parts: list[str] = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        msg = str(err.get("msg") or "").strip()
        if not msg:
            continue
        field = _error_field(err.get("loc"))
        parts.append(f"{field}: {msg}" if field else msg)
    return "; ".join(parts)


class AsyncMnemoClient:
    """Async client for the Mnemoverse Memory API.

    Features:
    - Circuit breaker (5 failures → open → 30s half-open)
    - Timeout (10s default)
    - Retry with exponential backoff (3 attempts, rate-limit-aware)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._cb = CircuitBreaker(failure_threshold=5, reset_timeout=30.0)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "X-Api-Key": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncMnemoClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # --- Public API ---

    async def write(
        self,
        content: str,
        *,
        concepts: list[str] | None = None,
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        external_ref: str | None = None,
    ) -> WriteResponse:
        """Store a single memory atom."""
        body: dict[str, Any] = {"content": content, "domain": domain}
        if concepts:
            body["concepts"] = concepts
        if metadata:
            body["metadata"] = metadata
        if external_ref:
            body["external_ref"] = external_ref
        data = await self._request("POST", "/api/v1/memory/write", json=body)
        return WriteResponse.model_validate(data)

    async def write_batch(
        self,
        items: list[dict[str, Any]],
    ) -> WriteBatchResponse:
        """Store up to 500 atoms in one request."""
        data = await self._request("POST", "/api/v1/memory/write-batch", json={"items": items})
        return WriteBatchResponse.model_validate(data)

    async def read(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: str | None = None,
        min_relevance: float = 0.3,
        include_associations: bool = True,
        concepts: list[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        order_by: str | None = None,
        exclude_author: str | None = None,
    ) -> ReadResponse:
        """Query memory with semantic search + Hebbian expansion.

        ``since`` / ``until`` bound the result by creation time, inclusive at
        both ends; naive datetimes are read as UTC. ``order_by="recency"``
        re-sorts the matched set newest-first without changing which entries
        matched. ``exclude_author`` drops one author principal — the
        "everyone but me" read in a shared room.

        For "what happened lately" rather than "what do I know about X", use
        :meth:`recent`: search returns what MATCHES, so entries that exist but
        do not match are absent — correctly, but invisibly.
        """
        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "min_relevance": min_relevance,
            "include_associations": include_associations,
        }
        if domain:
            body["domain"] = domain
        if concepts:
            body["concepts"] = concepts
        if since is not None:
            body["since"] = _isoformat(since)
        if until is not None:
            body["until"] = _isoformat(until)
        if order_by is not None:
            body["order_by"] = order_by
        if exclude_author is not None:
            body["exclude_author"] = exclude_author
        data = await self._request("POST", "/api/v1/memory/read", json=body)
        return ReadResponse.model_validate(data)

    async def recent(
        self,
        *,
        domain: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        exclude_author: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> RecentResponse:
        """List the newest entries first — no query, complete by construction.

        The temporal complement of :meth:`read`. Nothing is ranked away, so this
        is what to use when you need to be sure you are seeing everything:
        resuming after a break, catching up on a shared room, or auditing.

        Paged by cursor rather than truncated. When more entries exist the
        response carries ``next_cursor``; passing it back continues the listing
        with no skips and no duplicates even while writes are landing, which
        LIMIT/OFFSET cannot guarantee. An empty feed is a normal empty list.
        """
        body: dict[str, Any] = {"limit": limit}
        if domain:
            body["domain"] = domain
        if since is not None:
            body["since"] = _isoformat(since)
        if until is not None:
            body["until"] = _isoformat(until)
        if exclude_author is not None:
            body["exclude_author"] = exclude_author
        if cursor:
            body["cursor"] = cursor
        data = await self._request("POST", "/api/v1/memory/recent", json=body)
        return RecentResponse.model_validate(data)

    async def feedback(
        self,
        atom_ids: list[UUID | str],
        outcome: float,
        *,
        concepts: list[str] | None = None,
        query_concepts: list[str] | None = None,
        domain: str = "general",
    ) -> FeedbackResponse:
        """Report outcome (success/failure) for memories."""
        body: dict[str, Any] = {
            "atom_ids": [str(aid) for aid in atom_ids],
            "outcome": outcome,
            "domain": domain,
        }
        if concepts:
            body["concepts"] = concepts
        if query_concepts:
            body["query_concepts"] = query_concepts
        data = await self._request("POST", "/api/v1/memory/feedback", json=body)
        return FeedbackResponse.model_validate(data)

    async def stats(self) -> StatsResponse:
        """Get memory statistics."""
        data = await self._request("GET", "/api/v1/memory/stats")
        return StatsResponse.model_validate(data)

    async def health(self) -> HealthResponse:
        """Check API health."""
        data = await self._request("GET", "/api/v1/health")
        return HealthResponse.model_validate(data)

    # --- Internal ---

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if not self._cb.can_execute():
            raise MnemoUnavailableError(
                f"Circuit breaker open (state: {self._cb.state})"
            )

        def is_retryable(e: Exception) -> bool:
            if isinstance(e, MnemoRateLimitError):
                return True
            if isinstance(e, MnemoError) and e.status and e.status >= 500:
                return True
            if isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
                return True
            return False

        async def attempt() -> Any:
            return await self._single_request(method, path, json)

        try:
            result = await retry_with_backoff(
                attempt,
                max_retries=self._max_retries,
                retryable_check=is_retryable,
            )
            self._cb.on_success()
            return result
        except (MnemoAuthError, MnemoError) as e:
            if isinstance(e, MnemoAuthError):
                raise
            self._cb.on_failure()
            raise

    async def _single_request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        client = self._get_client()
        try:
            response = await client.request(method, path, json=json)
        except httpx.TimeoutException as e:
            raise MnemoUnavailableError(f"Request timeout after {self._timeout}s", e)
        except httpx.ConnectError as e:
            raise MnemoUnavailableError(f"Connection error: {e}", e)

        if response.status_code == 401 or response.status_code == 403:
            raise MnemoAuthError(self._extract_detail(response))

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise MnemoRateLimitError(
                self._extract_detail(response),
                retry_after=float(retry_after) if retry_after else None,
            )

        if response.status_code >= 400:
            raise MnemoError(self._extract_detail(response), status=response.status_code)

        return response.json()

    @staticmethod
    def _extract_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
            if not isinstance(data, dict):
                return str(data)
        except Exception:
            return f"HTTP {response.status_code}"

        summary = data.get("detail") or data.get("message")
        specifics = _format_validation_errors(data)
        if summary and specifics:
            return f"{summary} ({specifics})"
        return str(summary or specifics or data)
