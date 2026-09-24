[![PyPI version](https://img.shields.io/pypi/v/mnemoverse.svg?color=blue)](https://pypi.org/project/mnemoverse/)
[![Python versions](https://img.shields.io/pypi/pyversions/mnemoverse.svg)](https://pypi.org/project/mnemoverse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Research: SLoD arXiv](https://img.shields.io/badge/Research-arXiv%3A2603.08965-b31b1b)](https://arxiv.org/abs/2603.08965)

# Mnemoverse Python SDK

Hosted AI agent memory that learns from outcomes, from Python. `mnemoverse` is a
sync and async client for the Mnemoverse REST API: write memories, search them
with a natural-language query, and report whether a recalled memory helped or
misled, which re-ranks what comes back next. Shared rooms (beta) use the same
calls: pass a room's `xroom:` address as `domain` to write to it or read it.

The SDK does not create, invite to or join rooms. Those are tools of the
[Mnemoverse MCP server](https://mnemoverse.com/docs/api/mcp-server), which also
connects Claude, Cursor and ChatGPT to the same memory.

## Installation

```bash
pip install mnemoverse
```

## Quick Start

```bash
export MNEMOVERSE_API_KEY=mk_live_YOUR_KEY
```

**Check the key in one command.** It reads the same variable the client reads (in PowerShell, write `$env:MNEMOVERSE_API_KEY` and type `curl.exe`):
```bash
curl -s -H "X-Api-Key: $MNEMOVERSE_API_KEY" https://core.mnemoverse.com/api/v1/memory/stats
```
| The API answers (as of 2026-09-23) | What it means |
|---|---|
| JSON that includes `"total_atoms"` | The key works. |
| `"message":"This API key is not recognized. Check that the whole key was copied, or create a new key at https://console.mnemoverse.com/dashboard/keys."` | The key is wrong or revoked. |
| `"message":"This API key is the placeholder from the documentation, not a real key. Create a key at https://console.mnemoverse.com/dashboard/keys and use it instead (MNEMOVERSE_API_KEY for the MCP server, the X-Api-Key header for REST)."` | `MNEMOVERSE_API_KEY` is still set to the placeholder from the export step above. |
| `"message":"Missing API key. Send X-Api-Key header."` | No key reached the API: the variable is empty or not set in this shell. |

```python
from mnemoverse import MnemoClient

client = MnemoClient()
# or pass it explicitly — an explicit api_key always wins over the environment:
# client = MnemoClient(api_key="mk_live_YOUR_KEY")

# Store a memory
result = client.write(
    "Retry with exponential backoff fixed the timeout issue",
    concepts=["retry", "backoff", "timeout"]
)

# Search by natural-language query
memories = client.read("how to handle timeouts?")

# Report outcome — the system learns what works
client.feedback(
    atom_ids=[item.atom_id for item in memories.items],
    outcome=1.0,
    query_concepts=memories.query_concepts
)
```

### Closing the client

`MnemoClient` holds a background event loop and a pooled HTTP connection, so
close it when a script is done. Use it as a context manager,
`with MnemoClient() as client:`, to close automatically on exit, or call
`client.close()` directly. A client left unclosed closes itself at
interpreter exit, which is a fallback rather than something to rely on.

## Async Client

```python
from mnemoverse import AsyncMnemoClient

async with AsyncMnemoClient(api_key="mk_live_YOUR_KEY") as client:
    result = await client.write("async memory", concepts=["async"])
    memories = await client.read("what about async?")
```

## Features

- **Circuit breaker** — 5 failures → open → 30s half-open → probe
- **Retry with backoff** — 3 attempts, rate-limit-aware
- **Sync + async** — `MnemoClient` for scripts, `AsyncMnemoClient` for FastAPI
- **Type-safe** — Pydantic models, full type hints

## Methods

| Method | Description |
|--------|-------------|
| `write(content, concepts, domain, metadata)` | Store a memory |
| `write_batch(items)` | Store up to 500 memories |
| `read(query, top_k, domain, since, until, order_by, exclude_author)` | Natural-language search — "what do I know about X" |
| `recent(domain, since, until, exclude_author, limit, cursor)` | Newest-first feed — "what happened lately" |
| `feedback(atom_ids, outcome)` | Report success/failure |
| `stats()` | Memory statistics |
| `health()` | API health check |

Every method exists on both `MnemoClient` (sync) and `AsyncMnemoClient` (async).

### Search or feed?

`read()` answers *what do I know about X* and ranks by relevance. `recent()`
answers *what happened lately* and is complete within one scope by
construction — nothing is skipped, which a ranked search cannot promise.
Reach for `recent()` to resume after a break or to catch up on a shared room.

```python
from mnemoverse import MnemoClient

client = MnemoClient(api_key="mk_live_...")

# Catch up on a shared room. Rooms are SEPARATE stores: pass the address as
# `domain`, or an unscoped feed will not cover them.
page = client.recent(domain="xroom:room_01ABC", since="2026-08-01T00:00:00Z", limit=20)
for item in page.items:
    print(item.created_at, item.content)

if page.next_cursor:
    page = client.recent(domain="xroom:room_01ABC", cursor=page.next_cursor)
```

Read items carry `created_at` and `provenance` (who wrote it, where from).

## Documentation

- [Getting Started](https://mnemoverse.com/docs/api/getting-started)
- [API Reference](https://mnemoverse.com/docs/api/reference)
- [Python SDK Docs](https://mnemoverse.com/docs/api/python-sdk)

## License

MIT
