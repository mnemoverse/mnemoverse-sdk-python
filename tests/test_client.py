"""Tests for MnemoClient and AsyncMnemoClient."""

from __future__ import annotations

import pytest
import httpx
from pytest_httpx import HTTPXMock

from mnemoverse import AsyncMnemoClient, MnemoAuthError, MnemoError, MnemoRateLimitError


@pytest.fixture
def client():
    return AsyncMnemoClient(
        api_key="mk_test_abc123",
        base_url="https://test.api.mnemoverse.com",
        timeout=5.0,
        max_retries=0,  # no retries in tests
    )


async def test_write(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        json={
            "stored": True,
            "atom_id": "550e8400-e29b-41d4-a716-446655440000",
            "importance": 0.85,
            "reason": "novel insight",
        },
    )

    result = await client.write("test memory", concepts=["test"])

    assert result.stored is True
    assert str(result.atom_id) == "550e8400-e29b-41d4-a716-446655440000"
    assert result.importance == 0.85


async def test_read(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        json={
            "items": [
                {
                    "atom_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "test memory",
                    "relevance": 0.92,
                    "similarity": 0.87,
                    "valence": 0.5,
                    "importance": 0.85,
                    "source": "semantic",
                    "concepts": ["test"],
                    "domain": "general",
                    "metadata": {},
                }
            ],
            "episodic_hit": False,
            "query_concepts": ["test"],
            "expanded_concepts": ["test"],
            "search_time_ms": 12.5,
        },
    )

    result = await client.read("test query")

    assert len(result.items) == 1
    assert result.items[0].content == "test memory"
    assert result.search_time_ms == 12.5


async def test_feedback(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/feedback",
        json={"updated_count": 1, "avg_valence": 0.8, "coactivation_edges": 3},
    )

    result = await client.feedback(
        atom_ids=["550e8400-e29b-41d4-a716-446655440000"],
        outcome=1.0,
    )

    assert result.updated_count == 1
    assert result.avg_valence == 0.8


async def test_stats(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/stats",
        json={
            "total_atoms": 100,
            "episodes": 80,
            "prototypes": 15,
            "singletons": 5,
            "hebbian_edges": 250,
            "episodic_fingerprints": 10,
            "domains": ["general", "engineering"],
            "avg_valence": 0.3,
            "avg_importance": 0.6,
        },
    )

    result = await client.stats()

    assert result.total_atoms == 100
    assert "engineering" in result.domains


async def test_health(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/health",
        json={"status": "ok", "database": True, "version": "1.0.0"},
    )

    result = await client.health()

    assert result.status == "ok"
    assert result.database is True


async def test_auth_error(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        status_code=401,
        json={"detail": "Invalid API key"},
    )

    with pytest.raises(MnemoAuthError):
        await client.read("test")


async def test_rate_limit_error(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        status_code=429,
        json={"detail": "Rate limit exceeded"},
        headers={"Retry-After": "60"},
    )

    with pytest.raises(MnemoRateLimitError) as exc_info:
        await client.read("test")

    assert exc_info.value.retry_after == 60.0


async def test_recent_returns_the_feed_and_its_cursor(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """The queryless feed: complete by construction, paged by cursor."""
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/recent",
        json={
            "items": [
                {
                    "atom_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "deployed the feed",
                    "domain": "work",
                    "created_at": "2026-08-06T10:00:00Z",
                    "concepts": ["deploy"],
                    "provenance": {"principal": "alice", "is_external": False},
                },
                {
                    "atom_id": "550e8400-e29b-41d4-a716-446655440001",
                    "content": "an older note with no author",
                    "domain": "work",
                    "created_at": "2026-08-05T10:00:00Z",
                },
            ],
            "next_cursor": "b3BhcXVl",
        },
    )

    r = await client.recent(limit=2)

    assert [i.content for i in r.items] == ["deployed the feed", "an older note with no author"]
    assert r.next_cursor == "b3BhcXVl"
    # Provenance is optional per item — an entry written before authorship was
    # recorded must parse, not raise.
    assert r.items[0].provenance is not None
    assert r.items[0].provenance.principal == "alice"
    assert r.items[1].provenance is None
    assert r.items[0].created_at.year == 2026


async def test_recent_sends_only_the_filters_it_was_given(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """Absent filters must be absent from the body, not sent as nulls — the
    server treats a present-but-null differently from omitted in some places,
    and sending noise makes request logs harder to read."""
    import json as _json
    from datetime import datetime, timezone

    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/recent",
        json={"items": [], "next_cursor": None},
    )

    await client.recent(
        since=datetime(2026, 8, 1, tzinfo=timezone.utc),
        until="2026-08-06T00:00:00Z",
        exclude_author="alice",
    )

    sent = _json.loads(httpx_mock.get_requests()[-1].content)
    assert sent["since"] == "2026-08-01T00:00:00+00:00"
    # A string passes through untouched: callers echo watermarks the server gave
    # them, and forcing them to parse first would be gratuitous.
    assert sent["until"] == "2026-08-06T00:00:00Z"
    assert sent["exclude_author"] == "alice"
    assert "domain" not in sent
    assert "cursor" not in sent


async def test_read_forwards_the_temporal_params(client: AsyncMnemoClient, httpx_mock: HTTPXMock):
    """These shipped in the API in August and the SDK had never learned them, so
    a Python caller could not use half of what /memory/read offers."""
    import json as _json

    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        json={
            "items": [],
            "episodic_hit": False,
            "query_concepts": [],
            "expanded_concepts": [],
            "search_time_ms": 1.0,
        },
    )

    await client.read(
        "anything",
        since="2026-08-01T00:00:00Z",
        until="2026-08-06T00:00:00Z",
        order_by="recency",
        exclude_author="bob",
    )

    sent = _json.loads(httpx_mock.get_requests()[-1].content)
    assert sent["since"] == "2026-08-01T00:00:00Z"
    assert sent["until"] == "2026-08-06T00:00:00Z"
    assert sent["order_by"] == "recency"
    assert sent["exclude_author"] == "bob"


async def test_read_items_keep_created_at_and_provenance(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """The SDK was parsing read responses without these two fields, so a caller
    saw a memory with no timestamp and no author even when the API sent both."""
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        json={
            "items": [
                {
                    "atom_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "x",
                    "relevance": 0.9,
                    "similarity": 0.9,
                    "valence": 0.0,
                    "importance": 0.5,
                    "source": "semantic",
                    "concepts": [],
                    "domain": "general",
                    "created_at": "2026-08-06T10:00:00Z",
                    "provenance": {"principal": "alice"},
                }
            ],
            "episodic_hit": False,
            "query_concepts": [],
            "expanded_concepts": [],
            "search_time_ms": 1.0,
        },
    )

    r = await client.read("x")

    assert r.items[0].created_at is not None
    assert r.items[0].provenance is not None
    assert r.items[0].provenance.principal == "alice"


async def test_read_still_parses_a_response_without_the_new_fields(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """Compatibility pin: an older core, or a cached response, omits both — that
    must stay a parse, not an exception."""
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        json={
            "items": [
                {
                    "atom_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "x",
                    "relevance": 0.9,
                    "similarity": 0.9,
                    "valence": 0.0,
                    "importance": 0.5,
                    "source": "semantic",
                    "concepts": [],
                    "domain": "general",
                }
            ],
            "episodic_hit": False,
            "query_concepts": [],
            "expanded_concepts": [],
            "search_time_ms": 1.0,
        },
    )

    r = await client.read("x")

    assert r.items[0].created_at is None
    assert r.items[0].provenance is None


async def test_validation_error_names_the_field_and_the_limit(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """A caller who sent 10 001 characters must be told which field and which limit.

    Core's 400 carries no ``detail`` at all, and its ``message`` is the generic
    "Request validation failed" — the field name and the number live one level
    down, in ``details.errors[0]``. Reading only the top two keys threw the
    only useful part of the response away, so the user got a sentence that
    tells them nothing about what to change. Body copied from the shape core
    actually emits (mnemoverse-core/src/mnemo/api/server.py:358-379).
    """
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        status_code=400,
        json={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "requestId": "01KX77H1AX5E2457MDWRP1H72V",
            "retryable": False,
            "details": {
                "errors": [
                    {
                        "loc": ["body", "content"],
                        "msg": "String should have at most 10000 characters",
                        "type": "string_too_long",
                    }
                ]
            },
        },
    )

    with pytest.raises(MnemoError) as exc_info:
        await client.write("x" * 10_001)

    assert str(exc_info.value) == (
        "Request validation failed "
        "(content: String should have at most 10000 characters)"
    )


async def test_validation_error_reports_every_field_that_failed(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """One request can break several rules at once; naming only the first would
    send the caller round the loop again for the second."""
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        status_code=400,
        json={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "requestId": "01KX77H1AX5E2457MDWRP1H72V",
            "retryable": False,
            "details": {
                "errors": [
                    {
                        "loc": ["body", "content"],
                        "msg": "String should have at most 10000 characters",
                        "type": "string_too_long",
                    },
                    {
                        "loc": ["body", "domain"],
                        "msg": "String should have at most 100 characters",
                        "type": "string_too_long",
                    },
                ]
            },
        },
    )

    with pytest.raises(MnemoError) as exc_info:
        await client.write("x" * 10_001, domain="d" * 101)

    assert str(exc_info.value) == (
        "Request validation failed "
        "(content: String should have at most 10000 characters; "
        "domain: String should have at most 100 characters)"
    )
