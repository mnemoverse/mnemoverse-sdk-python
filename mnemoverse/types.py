"""Pydantic models matching mnemoverse-core REST API schemas.

Source of truth: mnemoverse-core/src/mnemo/api/schemas.py
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


# --- Write ---


class WriteResponse(BaseModel):
    stored: bool
    atom_id: UUID | None = None
    importance: float = 0.0
    reason: str = ""


class WriteBatchItemResult(BaseModel):
    index: int
    stored: bool
    atom_id: UUID | None = None
    importance: float = 0.0
    error: str | None = None


class WriteBatchResponse(BaseModel):
    total_count: int
    stored_count: int
    results: list[WriteBatchItemResult]


# --- Read ---


class Provenance(BaseModel):
    """Who recorded a memory, when the server was told.

    Omitted entirely for entries written before provenance existed, which is why
    every field is optional and the whole object is nullable on an item.
    """

    principal: str | None = None
    agent: str | None = None
    agent_name: str | None = None
    client_env: str | None = None
    is_external: bool | None = None


class MemoryItem(BaseModel):
    atom_id: UUID
    content: str
    relevance: float
    similarity: float
    valence: float
    importance: float
    source: str
    concepts: list[str]
    domain: str
    metadata: dict[str, Any] = {}
    # The server has returned both of these since the temporal read shipped; the
    # SDK was parsing responses without them, so callers saw a memory with no
    # timestamp and no author even when the API sent both.
    created_at: datetime | None = None
    provenance: Provenance | None = None


class RecentItem(BaseModel):
    """One entry of the queryless feed.

    Deliberately not a MemoryItem: the feed does no ranking, so relevance and
    similarity would be meaningless numbers rather than absent ones.
    """

    atom_id: UUID
    content: str
    domain: str
    created_at: datetime
    concepts: list[str] = []
    provenance: Provenance | None = None


class RecentResponse(BaseModel):
    items: list[RecentItem]
    next_cursor: str | None = None


class ReadResponse(BaseModel):
    items: list[MemoryItem]
    episodic_hit: bool
    query_concepts: list[str]
    expanded_concepts: list[str]
    search_time_ms: float


# --- Feedback ---


class FeedbackResponse(BaseModel):
    updated_count: int
    avg_valence: float
    coactivation_edges: int = 0


# --- Stats ---


class StatsResponse(BaseModel):
    total_atoms: int
    episodes: int
    prototypes: int
    singletons: int
    hebbian_edges: int
    episodic_fingerprints: int
    domains: list[str]
    avg_valence: float
    avg_importance: float


# --- Health ---


class HealthResponse(BaseModel):
    status: str
    database: bool
    version: str
