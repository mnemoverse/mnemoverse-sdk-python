"""Async Mnemoverse client using httpx."""

from __future__ import annotations

import os
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
_API_KEY_ENV_VAR = "MNEMOVERSE_API_KEY"


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve the API key: explicit argument wins, then the environment.

    An empty string is treated the same as ``None`` — a caller that
    interpolates a missing config value into an empty string should not
    silently pass "no api_key" checks.
    """
    if api_key:
        return api_key
    env_key = os.environ.get(_API_KEY_ENV_VAR)
    if env_key:
        return env_key
    raise ValueError(
        "No API key: pass api_key= or set the "
        f"{_API_KEY_ENV_VAR} environment variable"
    )


def _isoformat(value: datetime | str) -> str:
    """Accept either a datetime or an already-formatted string.

    Callers reach for both: a datetime when they computed the boundary, a string
    when they are echoing a watermark the server gave them. Rejecting the string
    form would make round-tripping a cursor-adjacent value needlessly awkward.
    """
    return value.isoformat() if isinstance(value, datetime) else value


def _counts_towards_breaker(exc: Exception) -> bool:
    """Whether an error is evidence that the service may be unhealthy.

    Retry policy and health policy are deliberately separate. Caller errors do
    not count; a permanent 5xx still does even when Core says retrying it would
    be futile. A non-retryable 429 is a permanent caller/quota rejection, while
    older retryable/unspecified 429 responses retain the historical behaviour.
    Transport-shaped errors have no status and continue to count.
    """
    if isinstance(exc, MnemoError) and exc.status is not None:
        if 400 <= exc.status < 500:
            return exc.status == 429 and exc.retryable is not False
        return exc.status >= 500
    return True


# FastAPI prefixes each error location with where it was found; keeping it would
# turn "content" into "body.content" for no gain to the reader.
_LOC_SOURCES = ("body", "query", "path", "header", "cookie")


def _error_field(loc: Any) -> str:
    """Name the offending field from a validation error's ``loc`` path."""
    if not isinstance(loc, list) or not loc:
        return ""
    parts = [str(part) for part in loc]
    if len(parts) > 1 and parts[0] in _LOC_SOURCES:
        parts = parts[1:]
    return ".".join(parts)


def _format_validation_errors(data: Any, summary: str) -> str:
    """Extract actionable validation details not already present in summary."""
    details = data.get("details") if isinstance(data, dict) else None
    errors = details.get("errors") if isinstance(details, dict) else None
    if not isinstance(errors, list):
        return ""

    parts: list[str] = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        message = str(error.get("msg") or "").strip()
        if not message:
            continue
        field = _error_field(error.get("loc"))
        rendered = f"{field}: {message}" if field else message
        # Core versions that enrich their top-level message may already carry
        # this exact detail (possibly with the source prefix, e.g. ``body.``).
        if rendered not in summary:
            parts.append(rendered)
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
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        """Create a client.

        ``api_key`` is optional: when it is ``None`` or empty, the key is
        read from the ``MNEMOVERSE_API_KEY`` environment variable. An
        explicit ``api_key`` always wins over the environment. Raises
        ``ValueError`` when neither is set.
        """
        self._api_key = _resolve_api_key(api_key)
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

    def _forget_client(self) -> None:
        """Drop the pooled HTTP client without closing it.

        A last resort for the sync wrapper: when the event loop the pool is
        bound to is going away and the pool could not be closed on it, keeping
        the reference would hand the next call a connection tied to a dead
        loop, which is exactly how 0.2.0 failed. Letting go of it costs at most
        one idle socket, which the OS reclaims; keeping it costs every later
        call.
        """
        self._client = None

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
        supersedes: list[UUID | str] | None = None,
    ) -> WriteResponse:
        """Store a single memory atom.

        ``supersedes`` marks this write as the correction for one or more
        earlier atoms (own-organization ids, at most 32): the new atom is
        stored and every listed one is marked superseded by it, in a single
        transaction — all of it or none. Rejected by the server (422) on
        :meth:`write_batch`, and alongside an ``xroom:`` domain.
        """
        body: dict[str, Any] = {"content": content, "domain": domain}
        if concepts:
            body["concepts"] = concepts
        if metadata:
            body["metadata"] = metadata
        if external_ref:
            body["external_ref"] = external_ref
        if supersedes:
            body["supersedes"] = [str(sid) for sid in supersedes]
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
            if isinstance(e, MnemoError) and e.retryable is not None:
                return e.retryable
            if isinstance(e, MnemoRateLimitError):
                return True
            if isinstance(e, MnemoError) and e.status and e.status >= 500:
                return True
            return isinstance(e, (httpx.ConnectError, httpx.TimeoutException))

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
        except MnemoError as e:
            if _counts_towards_breaker(e):
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

        if response.status_code < 400:
            return response.json()

        detail, retryable = self._extract_error(response)

        if response.status_code == 401 or response.status_code == 403:
            raise MnemoAuthError(detail)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise MnemoRateLimitError(
                detail,
                retry_after=float(retry_after) if retry_after else None,
                retryable=retryable,
            )

        raise MnemoError(detail, status=response.status_code, retryable=retryable)

    @staticmethod
    def _extract_error(response: httpx.Response) -> tuple[str, bool | None]:
        try:
            data = response.json()
            if not isinstance(data, dict):
                return str(data), None
        except ValueError:
            return f"HTTP {response.status_code}", None

        summary_value = data.get("detail") or data.get("message")
        summary = str(summary_value) if summary_value else ""
        specifics = _format_validation_errors(data, summary)
        if summary and specifics:
            message = f"{summary} ({specifics})"
        else:
            message = str(summary or specifics or data)
        retryable = data.get("retryable")
        return message, retryable if isinstance(retryable, bool) else None
