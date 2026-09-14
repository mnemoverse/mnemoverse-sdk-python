"""Synchronous Mnemoverse client — wraps AsyncMnemoClient with asyncio.run()."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

from mnemoverse._async_client import AsyncMnemoClient
from mnemoverse.types import (
    FeedbackResponse,
    HealthResponse,
    ReadResponse,
    RecentResponse,
    StatsResponse,
    WriteBatchResponse,
    WriteResponse,
)

T = TypeVar("T")


class MnemoClient:
    """Synchronous client for the Mnemoverse Memory API.

    Wraps AsyncMnemoClient for use in scripts, notebooks, and sync applications.
    For async applications (FastAPI, Discord bots), use AsyncMnemoClient directly.

    Usage:
        # export MNEMOVERSE_API_KEY=mk_live_...
        client = MnemoClient()
        result = client.write("Caching reduces latency", concepts=["caching"])
        memories = client.read("how to reduce latency?")

    ``api_key`` is optional: when it is ``None`` or empty, the key is read
    from the ``MNEMOVERSE_API_KEY`` environment variable. An explicit
    ``api_key`` always wins over the environment. Raises ``ValueError`` when
    neither is set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://core.mnemoverse.com",
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self._async_client = AsyncMnemoClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Inside an existing event loop (e.g., Jupyter notebook)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def write(
        self,
        content: str,
        *,
        concepts: list[str] | None = None,
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        external_ref: str | None = None,
    ) -> WriteResponse:
        """Store a single memory atom."""
        return self._run(
            self._async_client.write(
                content, concepts=concepts, domain=domain,
                metadata=metadata, external_ref=external_ref,
            )
        )

    def write_batch(self, items: list[dict[str, Any]]) -> WriteBatchResponse:
        """Store up to 500 atoms in one request."""
        return self._run(self._async_client.write_batch(items))

    def read(
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
        both ends. ``order_by="recency"`` re-sorts the matched set newest-first
        without changing which entries matched. ``exclude_author`` drops one
        author principal.

        For "what happened lately" rather than "what do I know about X", use
        :meth:`recent` — search returns what MATCHES, so entries that exist but
        do not match are absent, correctly but invisibly.
        """
        return self._run(
            self._async_client.read(
                query, top_k=top_k, domain=domain,
                min_relevance=min_relevance,
                include_associations=include_associations,
                concepts=concepts,
                since=since, until=until,
                order_by=order_by, exclude_author=exclude_author,
            )
        )

    def recent(
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

        The temporal complement of :meth:`read`: nothing is ranked away, so use
        this when you need to be sure you are seeing everything. Paged by
        ``next_cursor``, which continues the listing with no skips and no
        duplicates even while writes are landing.
        """
        return self._run(
            self._async_client.recent(
                domain=domain, since=since, until=until,
                exclude_author=exclude_author, limit=limit, cursor=cursor,
            )
        )

    def feedback(
        self,
        atom_ids: list[UUID | str],
        outcome: float,
        *,
        concepts: list[str] | None = None,
        query_concepts: list[str] | None = None,
        domain: str = "general",
    ) -> FeedbackResponse:
        """Report outcome (success/failure) for memories."""
        return self._run(
            self._async_client.feedback(
                atom_ids, outcome, concepts=concepts,
                query_concepts=query_concepts, domain=domain,
            )
        )

    def stats(self) -> StatsResponse:
        """Get memory statistics."""
        return self._run(self._async_client.stats())

    def health(self) -> HealthResponse:
        """Check API health."""
        return self._run(self._async_client.health())

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._run(self._async_client.close())
