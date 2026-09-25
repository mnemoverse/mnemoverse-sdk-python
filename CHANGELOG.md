# Changelog

All notable changes to `mnemoverse` (PyPI).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The
project is pre-1.0, so a MINOR bump may change behaviour.

**Versioning is per-package, not shared with the rest of Mnemoverse.**
`@mnemoverse/mcp-memory-server`, `@mnemoverse/memory-viz` and this SDK ship on
their own cadence and carry their own numbers; a shared number would force
empty releases and would claim a history a package does not have. What IS
aligned across clients is the *core contract* each one speaks — see
`mnemoverse-mcp-remote/contracts/` for the snapshot-with-source-SHA pattern
this repo should adopt next.

**What counts as a PATCH here.** A patch may change text — exception messages,
docstrings, README prose — and may fix behaviour that was plainly a defect,
provided no public name is added, removed, renamed or retyped, and no method
changes what it sends on the wire. Anything that adds a public class, method or
keyword argument is a MINOR, even pre-1.0.

## [Unreleased]

_Nothing yet._

## [0.3.2] — 2026-09-25

Package metadata and README text only. No public name, signature or request
changes, so this is a PATCH under the rule above.

### Changed

- **The PyPI summary and the README's opening describe the package the way the
  rest of Mnemoverse does:** hosted AI agent memory that learns from outcomes.
  The README now says what the SDK covers (write, search, feedback, the
  `graph()` read, and reading and writing a shared room by its `xroom:`
  address) and what it does not (creating, inviting to and joining rooms,
  which are MCP server tools).
  The opening line and a Quick Start comment that described the retrieval
  mechanism now say what the call does instead.
- **Keywords.** The two that named a retrieval mechanism are dropped; `mcp`,
  `agent-memory` and `feedback` are added.
- **The author email is hello@mnemoverse.com.**

## [0.3.1] — 2026-09-25

### Added

- **`graph()` / `AsyncMnemoClient.graph()`** — bounded read of the
  concept-association graph around caller-supplied `seeds`, matching
  `POST /memory/graph` (live since 2026-09-25). Distinct from `read()`'s
  `expanded_concepts` (names only, discarded weights): returns the actual
  association edges — weight, valence, count, `updated_at` — for a bounded
  neighbourhood, never the whole organization's graph. Bounds: `seeds` 1-20
  items (each ≤200 chars), `depth` 1-3 hops, `limit` 1-500 edges, `min_weight`
  ≥0. `domain="xroom:<room_id>"` reads a room's own store, same as `read()`;
  any other value is inert. New public types: `GraphResponse`, `GraphNode`,
  `GraphEdge`.

- **`supersedes` on `write()`** (sync and async). Pass a list of up to 32
  atom ids (own organization) to mark this write as the correction for them —
  the new atom is stored and every listed one is marked superseded by it, in
  one transaction, matching `POST /memory/write`'s contract. Omitted from the
  request body when not given. Rejected by the server (422) on `write_batch`
  either way; `write_batch` takes raw dicts and forwards them unchanged, so no
  SDK-side model needed updating there.

## [0.3.0] — 2026-09-16

0.2.0's synchronous client failed on every second call. This release fixes
it. If you are running 0.2.0, upgrade.

### Fixed

- **The synchronous client no longer fails on every second call.** In 0.2.0,
  `MnemoClient` ran each call in its own event loop (`asyncio.run`) and closed
  that loop on the way out, while the `httpx.AsyncClient` underneath was kept
  and reused. The pooled keep-alive connection stayed bound to the loop that
  had just been closed, so the next call raised
  `RuntimeError: Event loop is closed`; that failure disposed of the HTTP
  client, the call after it opened a fresh one and succeeded, and the pattern
  repeated. Any script doing two or more calls on one `MnemoClient` hit it,
  including the Quick Start. The client now creates **one** event loop, lazily,
  and keeps it until `close()`, so the connection pool survives between calls.
  `close()` and the new context-manager support (`with MnemoClient() as c:`)
  close the HTTP client and then the loop, in that order; a client that is
  never closed is closed at interpreter exit rather than warning about it. A
  client used from inside a running event loop (a notebook) moves onto a
  worker thread of its own and keeps one loop there too.
  The test suite was green throughout the bug because it mocked httpx at the
  transport layer: no socket, no pooled connection, nothing to strand. The
  suite now talks to a real HTTP/1.1 keep-alive server
  (`tests/keepalive_server.py`), and the regression test does three calls on
  one client, which fails against 0.2.0 and passes here.

- **A `MnemoClient` shared between threads no longer raises at random.** One
  event loop can be driven by one thread at a time, so a second thread calling
  into the same client got `RuntimeError: This event loop is already running`
  instead of its answer. It also depended on where the *first* call came from:
  a client first used from inside a running loop lives on a worker thread of
  its own and was never affected, so the same code worked in a notebook and
  failed in a script. Calls on a shared client are serialised now. That makes
  sharing safe rather than fast, because the calls queue; for throughput give
  each thread its own client, or use `AsyncMnemoClient`. A call that fails on
  the way in no longer leaves its coroutine un-awaited either, which used to
  print `coroutine ... was never awaited` on top of the real error, from an
  unrelated line.

- **An ordinary reference cycle no longer prints a traceback.** A `__del__`
  added earlier in this same unreleased work called `close()`, and closing
  means driving the event loop, which is not allowed inside a garbage
  collection pass: on Windows every collected cycle holding a client wrote
  `Error on reading from the event loop self pipe` and eleven lines of
  traceback. 0.2.0 had no `__del__` and printed nothing, so this was a
  regression introduced inside a fix. `__del__` is gone. The `atexit` hook
  still closes a client that was never closed, which is the case it was added
  for, and a client that is still alive at exit is exactly the case `atexit`
  can reach.

- **A worker event loop that cannot start now says so instead of hanging.**
  The client's loop thread was waited on with no timeout, so a failure before
  the loop signalled ready stopped an ordinary call forever, with no exception
  and no message. The wait is bounded at five seconds, and whatever the thread
  died of is re-raised on the caller's thread with the SDK named.

### Added

- **`api_key` is now optional on `MnemoClient` and `AsyncMnemoClient`.** When
  omitted or empty, the key is read from the `MNEMOVERSE_API_KEY` environment
  variable, matching the MCP server and the other Mnemoverse SDKs. An
  explicit `api_key` argument still always wins over the environment; a
  clear `ValueError` is raised when neither is set. No `.env` file loading
  was added.

- **`MnemoClient` supports `with MnemoClient() as client:`.** The context
  manager calls `close()` on exit, same as calling it directly.

## [0.2.0] — 2026-08-13

First release since 0.1.0. Everything below existed only on `main` until now,
which is its own kind of silence — the incident of 2026-08-11 was about fixes
that never reached the caller, and an unpublished package is the same failure
one layer out.

### Changed

- **The default `base_url` is now `https://core.mnemoverse.com`.** 0.1.0
  shipped `https://api.mnemoverse.com`, a different host that answers 401 —
  so a default-configured 0.1.0 client could not reach the service at all.
  **If you allowlist egress or run through a proxy, permit the new host.**
  Callers who passed `base_url` explicitly are unaffected.
- **A failure the service marks permanent is no longer retried.** The client
  honours a `retryable` flag in the error body instead of deciding from the
  status code alone, so a rejection the server calls final costs one request
  rather than three.

### Added

- **`MnemoError.retryable`** and the `retryable` keyword on `MnemoError` /
  `MnemoRateLimitError` — the server's own verdict on whether a retry could
  ever succeed, `None` when it did not say.

- **`recent()` / `AsyncMnemoClient.recent()`** — the newest-first feed, the
  read that answers "what happened lately" rather than "what do I know about
  X". Complete within one scope by construction, unlike a semantic search.
  With `since` / `until` / `limit` / `cursor` / `exclude_author`, and `domain`
  to page a shared room (rooms are separate stores; an unscoped feed never
  covers them).
- **Temporal parameters on `read()`** — `since`, `until`, `order_by`,
  `exclude_author`.
- **`created_at` and `provenance` on read items.** Both were on the wire and
  the client dropped them on the floor.
- New public types: `RecentResponse`, `RecentItem`, `Provenance`.

### Fixed

- **A rejected request no longer opens the circuit breaker.** A non-retryable
  4xx is the caller's problem, not evidence that the service is unhealthy.
  Before this, five over-length writes tripped the breaker and then blocked
  *valid* writes for 30 seconds — with an error naming the SERVICE, while the
  client stopped issuing HTTP at all, so no server-side trace of the failure
  existed either. Permanent 5xx still count toward breaker health; a
  non-retryable 429 (a quota rejection, i.e. a caller fact) does not.
  Found by code read during the 2026-08-11 incident review
  (`_async_client.py:194-198`) and pinned by tests in `tests/test_client.py`.
- **Validation errors name the field and the limit.** `_extract_detail` read
  only `detail` / `message`; core's 400 has no `detail` and its `message` is
  the generic "Request validation failed", while the field name and the number
  live in `details.errors[0]`. A caller who sent 10,001 characters was told
  only that validation failed. Every failing field is now reported.

## [0.1.0]

Initial release: `read`, `write`, `feedback`, `stats`, `health`, sync and async
clients, retry with a circuit breaker.
