"""Async Mnemoverse client using httpx."""

from __future__ import annotations

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
    StatsResponse,
    WriteBatchResponse,
    WriteResponse,
)

_DEFAULT_BASE_URL = "https://api.mnemoverse.com"


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
    ) -> ReadResponse:
        """Query memory with semantic search + Hebbian expansion."""
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
        data = await self._request("POST", "/api/v1/memory/read", json=body)
        return ReadResponse.model_validate(data)

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
            if isinstance(data, dict):
                return str(data.get("detail") or data.get("message") or data)
            return str(data)
        except Exception:
            return f"HTTP {response.status_code}"
