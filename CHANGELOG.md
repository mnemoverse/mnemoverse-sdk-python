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
