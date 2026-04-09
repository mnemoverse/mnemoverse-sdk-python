"""Tests for MnemoClient and AsyncMnemoClient."""

from __future__ import annotations

import pytest
import httpx
from pytest_httpx import HTTPXMock

from mnemoverse import AsyncMnemoClient, MnemoAuthError, MnemoRateLimitError


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
