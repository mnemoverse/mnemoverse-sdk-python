"""Mnemoverse Python SDK — persistent memory for AI agents."""

from mnemoverse.client import MnemoClient
from mnemoverse._async_client import AsyncMnemoClient
from mnemoverse.errors import MnemoError, MnemoAuthError, MnemoRateLimitError, MnemoUnavailableError
from mnemoverse.types import (
    WriteResponse,
    WriteBatchResponse,
    WriteBatchItemResult,
    ReadResponse,
    RecentResponse,
    RecentItem,
    Provenance,
    MemoryItem,
    FeedbackResponse,
    StatsResponse,
    HealthResponse,
)

__version__ = "0.3.0"

__all__ = [
    "MnemoClient",
    "AsyncMnemoClient",
    "MnemoError",
    "MnemoAuthError",
    "MnemoRateLimitError",
    "MnemoUnavailableError",
    "WriteResponse",
    "WriteBatchResponse",
    "WriteBatchItemResult",
    "ReadResponse",
    "RecentResponse",
    "RecentItem",
    "Provenance",
    "MemoryItem",
    "FeedbackResponse",
    "StatsResponse",
    "HealthResponse",
]
