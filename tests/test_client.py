"""Tests for MnemoClient and AsyncMnemoClient."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from mnemoverse import (
    AsyncMnemoClient,
    MnemoAuthError,
    MnemoClient,
    MnemoError,
    MnemoRateLimitError,
    MnemoUnavailableError,
)


@pytest.fixture
def client():
    return AsyncMnemoClient(
        api_key="mk_test_abc123",
        base_url="https://test.api.mnemoverse.com",
        timeout=5.0,
        max_retries=0,  # no retries in tests
    )


def test_explicit_api_key_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_env_key")
    client = AsyncMnemoClient(api_key="mk_explicit_key")

    assert client._api_key == "mk_explicit_key"


def test_api_key_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_env_key")
    client = AsyncMnemoClient()

    assert client._api_key == "mk_env_key"


def test_empty_string_api_key_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty string is not a key — it falls through to the environment
    exactly like ``None``, matching the constructor docstring."""
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_env_key")
    client = AsyncMnemoClient(api_key="")

    assert client._api_key == "mk_env_key"


def test_missing_api_key_and_environment_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MNEMOVERSE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="api_key"):
        AsyncMnemoClient()

    monkeypatch.delenv("MNEMOVERSE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MNEMOVERSE_API_KEY"):
        AsyncMnemoClient(api_key="")


def test_sync_client_also_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """MnemoClient forwards to AsyncMnemoClient, which resolves the key —
    the fallback must not be duplicated (or missed) in the sync wrapper."""
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_env_key")
    client = MnemoClient()

    assert client._async_client._api_key == "mk_env_key"


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


async def test_wire_non_retryable_rate_limit_is_not_retried(httpx_mock: HTTPXMock):
    """Core can use 429 for a permanent quota rejection, not just throttling.

    The wire-level ``retryable`` flag is the authority: retrying a permanent
    rejection wastes requests and turns the useful API error into a breaker
    failure.
    """
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/read",
        status_code=429,
        json={"message": "Quota exhausted", "retryable": False},
    )
    retrying_client = AsyncMnemoClient(
        api_key="mk_test_abc123",
        base_url="https://test.api.mnemoverse.com",
        max_retries=2,
    )

    try:
        with pytest.raises(MnemoRateLimitError) as exc_info:
            await retrying_client.read("test")
    finally:
        await retrying_client.close()

    assert exc_info.value.retryable is False
    assert len(httpx_mock.get_requests()) == 1


async def test_wire_non_retryable_rate_limits_do_not_open_breaker(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """Five permanent quota rejections must not suppress a valid sixth call."""
    for _ in range(5):
        httpx_mock.add_response(
            url="https://test.api.mnemoverse.com/api/v1/memory/write",
            status_code=429,
            json={"message": "Quota exhausted", "retryable": False},
        )
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        json={
            "stored": True,
            "atom_id": "550e8400-e29b-41d4-a716-446655440000",
            "importance": 0.85,
            "reason": "novel insight",
        },
    )

    for _ in range(5):
        with pytest.raises(MnemoRateLimitError):
            await client.write("rejected by quota")

    result = await client.write("accepted after quota changes")

    assert result.stored is True
    assert len(httpx_mock.get_requests()) == 6


async def test_rate_limit_without_wire_retryable_keeps_legacy_retry(
    httpx_mock: HTTPXMock,
):
    """Older Core responses omit the flag; status 429 must still retry."""
    for _ in range(2):
        httpx_mock.add_response(
            url="https://test.api.mnemoverse.com/api/v1/memory/read",
            status_code=429,
            json={"detail": "Rate limit exceeded"},
        )
    retrying_client = AsyncMnemoClient(
        api_key="mk_test_abc123",
        base_url="https://test.api.mnemoverse.com",
        max_retries=1,
    )

    try:
        with pytest.raises(MnemoRateLimitError) as exc_info:
            await retrying_client.read("test")
    finally:
        await retrying_client.close()

    assert exc_info.value.retryable is None
    assert len(httpx_mock.get_requests()) == 2


async def test_client_errors_do_not_open_the_circuit_breaker(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """Five rejected writes must not stop the sixth valid one from going out.

    A 400 says the request was wrong, not that the service is down. Counting it
    as a breaker failure reproduces the 2026-08-11 symptom exactly: after the
    fifth over-length write the SDK stops issuing HTTP for 30s, blames the
    service ("Circuit breaker open"), and leaves no server-side trace of the
    writes it swallowed. mnemoverse-chat/packages/core-sdk/src/client.ts:175-178
    already answers this the right way; this SDK answered it the other way.
    """
    over_length = {
        "code": "VALIDATION_ERROR",
        "message": "Request validation failed",
        "requestId": "req_probe",
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
    }
    for _ in range(5):
        httpx_mock.add_response(
            url="https://test.api.mnemoverse.com/api/v1/memory/write",
            status_code=400,
            json=over_length,
        )
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        json={
            "stored": True,
            "atom_id": "550e8400-e29b-41d4-a716-446655440000",
            "importance": 0.85,
            "reason": "novel insight",
        },
    )

    for _ in range(5):
        with pytest.raises(MnemoError):
            await client.write("x" * 10_001)

    result = await client.write("a perfectly good memory")

    assert result.stored is True
    # Six requests, not five: the sixth must actually reach the wire.
    assert len(httpx_mock.get_requests()) == 6


async def test_server_errors_still_open_the_circuit_breaker(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """The other side of the same rule, pinned so the fix above cannot overshoot.

    Green before the fix as well as after — it is a regression pin, not a proof.
    """
    for _ in range(5):
        httpx_mock.add_response(
            url="https://test.api.mnemoverse.com/api/v1/health",
            status_code=500,
            json={"code": "INTERNAL", "message": "boom", "retryable": True},
        )

    for _ in range(5):
        with pytest.raises(MnemoError):
            await client.health()

    with pytest.raises(MnemoUnavailableError, match="Circuit breaker open"):
        await client.health()

    assert len(httpx_mock.get_requests()) == 5


async def test_non_retryable_server_errors_still_open_the_circuit_breaker(
    httpx_mock: HTTPXMock,
):
    """A wire retry instruction does not redefine service health.

    A permanent 500 should not be retried, but five independent 500 responses
    are still evidence that Core is unhealthy and must open the breaker.
    """
    for _ in range(5):
        httpx_mock.add_response(
            url="https://test.api.mnemoverse.com/api/v1/health",
            status_code=500,
            json={"message": "Permanent server failure", "retryable": False},
        )
    retrying_client = AsyncMnemoClient(
        api_key="mk_test_abc123",
        base_url="https://test.api.mnemoverse.com",
        max_retries=2,
    )

    try:
        for _ in range(5):
            with pytest.raises(MnemoError) as exc_info:
                await retrying_client.health()
            assert exc_info.value.retryable is False

        with pytest.raises(MnemoUnavailableError, match="Circuit breaker open"):
            await retrying_client.health()
    finally:
        await retrying_client.close()

    # One request per call proves retryable=false still controls retries.
    assert len(httpx_mock.get_requests()) == 5


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
    """Expose the actionable validation detail instead of only its summary."""
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
    assert exc_info.value.retryable is False


async def test_validation_error_reports_every_field_that_failed(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """One response can carry several independent, actionable failures."""
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        status_code=400,
        json={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
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


async def test_enriched_validation_summary_is_not_duplicated(
    client: AsyncMnemoClient, httpx_mock: HTTPXMock
):
    """Newer Core versions may already copy the first detail into ``message``."""
    summary = (
        "Request validation failed: body.content: "
        "String should have at most 10000 characters"
    )
    httpx_mock.add_response(
        url="https://test.api.mnemoverse.com/api/v1/memory/write",
        status_code=400,
        json={
            "message": summary,
            "retryable": False,
            "details": {
                "errors": [
                    {
                        "loc": ["body", "content"],
                        "msg": "String should have at most 10000 characters",
                    }
                ]
            },
        },
    )

    with pytest.raises(MnemoError) as exc_info:
        await client.write("x" * 10_001)

    assert str(exc_info.value) == summary
