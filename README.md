# Mnemoverse Python SDK

Persistent memory for AI agents. Not vector search — statistical learning.

## Installation

```bash
pip install mnemoverse
```

## Quick Start

```python
from mnemoverse import MnemoClient

client = MnemoClient(api_key="mk_live_YOUR_KEY")

# Store a memory
result = client.write(
    "Retry with exponential backoff fixed the timeout issue",
    concepts=["retry", "backoff", "timeout"]
)

# Query — Hebbian associations expand "timeout" → "retry", "backoff"
memories = client.read("how to handle timeouts?")

# Report outcome — the system learns what works
client.feedback(
    atom_ids=[item.atom_id for item in memories.items],
    outcome=1.0,
    query_concepts=memories.query_concepts
)
```

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
| `read(query, top_k, domain)` | Query with Hebbian expansion |
| `feedback(atom_ids, outcome)` | Report success/failure |
| `stats()` | Memory statistics |
| `health()` | API health check |

## Documentation

- [Getting Started](https://mnemoverse.com/docs/api/getting-started)
- [API Reference](https://mnemoverse.com/docs/api/reference)
- [Python SDK Docs](https://mnemoverse.com/docs/api/python-sdk)

## License

MIT
