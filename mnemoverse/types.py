"""Pydantic models matching mnemoverse-core REST API schemas.

Source of truth: mnemoverse-core/src/mnemo/api/schemas.py
"""

from __future__ import annotations

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
