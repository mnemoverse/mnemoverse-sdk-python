"""A real HTTP/1.1 keep-alive server standing in for core, for the tests.

Why a server and not a transport mock. Until 0.2.1 this suite mocked httpx at
the transport layer, so no socket was ever opened and no connection was ever
reused. That is exactly the blind spot that let the sync client ship broken:
``MnemoClient`` ran every call in a throwaway event loop while caching one
``httpx.AsyncClient`` across calls, so the pooled keep-alive connection stayed
bound to a loop that had been closed and every second call raised
``RuntimeError: Event loop is closed``. A test double with no connection to
keep alive cannot see that, and the suite was green through the whole life of
the bug.

So the rule this file exists to enforce: **any HTTP double for this SDK keeps
connections alive.** ``protocol_version = "HTTP/1.1"`` plus an honest
``Content-Length`` on every response, and ``connections`` counted so a test can
assert the reuse really happened rather than trusting that it did.

``connections`` alone is not enough for one question, though, and the gap let a
second bug hide. It counts opens and never comes back down, so a socket that
was closed properly and one that was abandoned still open look identical to it.
That is exactly the difference the handover path turns on: releasing a pooled
connection on the loop that owns it, rather than walking away from it. Hence
``live_connections`` and :meth:`MockCore.live_settles_at`, which count opens
AND closes, and which a test uses to say "one socket is still open, not two".
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ATOM_ID = "550e8400-e29b-41d4-a716-446655440000"

# What an endpoint answers when a test has not queued anything specific. Enough
# for lifecycle tests ("three calls in a row must all succeed") to exercise real
# request shapes without every one of them restating a payload.
DEFAULT_PAYLOADS: dict[str, Any] = {
    "/api/v1/health": {"status": "ok", "database": True, "version": "1.0.0"},
    "/api/v1/memory/write": {
        "stored": True,
        "atom_id": ATOM_ID,
        "importance": 0.85,
        "reason": "novel insight",
    },
    "/api/v1/memory/write-batch": {
        "total_count": 1,
        "stored_count": 1,
        "results": [{"index": 0, "stored": True, "atom_id": ATOM_ID, "importance": 0.5}],
    },
    "/api/v1/memory/read": {
        "items": [
            {
                "atom_id": ATOM_ID,
                "content": "a memory that survived the round trip",
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
    "/api/v1/memory/recent": {"items": [], "next_cursor": None},
    "/api/v1/memory/feedback": {
        "updated_count": 1,
        "avg_valence": 0.8,
        "coactivation_edges": 3,
    },
    "/api/v1/memory/stats": {
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
}


@dataclass
class RecordedRequest:
    """One request as the server saw it."""

    method: str
    path: str
    headers: dict[str, str]  # names lowercased; use header() rather than indexing
    content: bytes

    def header(self, name: str) -> str | None:
        """Look a header up without caring about its case, as HTTP requires."""
        return self.headers.get(name.lower())

    def json(self) -> Any:
        """The request body, parsed. Every request this SDK sends is JSON."""
        return json.loads(self.content)


@dataclass
class QueuedResponse:
    """One response a test asked for, consumed in the order it was queued."""

    status: int = 200
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    path: str | None = None


class MockCore:
    """A local core, on a real socket, speaking HTTP/1.1 with keep-alive."""

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.connections = 0
        self.live_connections = 0
        self.problems: list[str] = []
        self._queue: list[QueuedResponse] = []
        self._lock = threading.Lock()

        core = self

        class Handler(BaseHTTPRequestHandler):
            # The whole point of this harness. HTTP/1.0 (BaseHTTPRequestHandler's
            # default) closes after every response, which hides every bug that
            # lives in a reused connection.
            protocol_version = "HTTP/1.1"
            # Do not leave a handler thread blocked on a client that went away.
            timeout = 5.0

            def setup(self) -> None:  # one call per TCP connection
                with core._lock:
                    core.connections += 1
                    core.live_connections += 1
                super().setup()

            def finish(self) -> None:  # one call per TCP connection, on the way out
                try:
                    super().finish()
                finally:
                    with core._lock:
                        core.live_connections -= 1

            def do_GET(self) -> None:
                core._answer(self)

            def do_POST(self) -> None:
                core._answer(self)

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # pytest output stays about the tests

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # poll_interval is how long stop() waits for the accept loop to notice
        # it. The default 0.5s is per test, and there are dozens of tests.
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self._thread.start()

    # --- what a test drives ---

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def respond(
        self,
        *,
        json: Any = None,
        status: int = 200,
        headers: dict[str, str] | None = None,
        path: str | None = None,
    ) -> None:
        """Queue one response. ``path`` asserts which endpoint must ask for it."""
        with self._lock:
            self._queue.append(
                QueuedResponse(status=status, payload=json, headers=headers or {}, path=path)
            )

    @property
    def pending(self) -> int:
        """Queued responses nobody asked for. A drained queue is part of a pass."""
        with self._lock:
            return len(self._queue)

    def live_settles_at(self, expected: int, timeout: float = 2.0) -> int:
        """Wait for the number of OPEN connections to reach ``expected``.

        ``connections`` counts opens and never goes down, so it cannot tell a
        socket that was closed from one that was abandoned still open, which is
        the difference between releasing a pooled connection on its own loop
        and walking away from it. This can: a handler thread decrements on the
        way out of its connection.

        The wait is why this is a method and not an attribute. A handler learns
        its peer is gone when its next read comes back empty, which is a moment
        after the client let the socket go, so a bare read races the scheduler.
        Returns the count actually observed, so a failing assertion reports the
        real number rather than a timeout.

        **The default timeout has to stay well under ``Handler.timeout``**,
        which is 5s. A handler whose peer went quiet gives up on its own after
        that and runs ``finish()``, so an abandoned socket eventually stops
        being counted as live no matter what the client did. Wait that long and
        this method reports a tidy number the client never earned, and the
        assertion that leaned on it is decorative. Two seconds against five is
        the margin, and the real thing being measured takes milliseconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                live = self.live_connections
            if live == expected or time.monotonic() >= deadline:
                return live
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    # --- what the handler calls ---

    def _problem(self, description: str) -> None:
        """Record a server-side complaint. Handler threads run concurrently."""
        with self._lock:
            self.problems.append(description)

    def _answer(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        # Read the body whole, always: a half-read body desynchronises a
        # connection that is about to be reused.
        body = handler.rfile.read(length) if length else b""
        path = handler.path

        with self._lock:
            self.requests.append(
                RecordedRequest(
                    method=handler.command,
                    path=path,
                    headers={k.lower(): v for k, v in handler.headers.items()},
                    content=body,
                )
            )
            queued = self._queue.pop(0) if self._queue else None

        if queued is not None:
            if queued.path is not None and queued.path != path:
                self._problem(f"response queued for {queued.path} was requested by {path}")
            status, payload, extra = queued.status, queued.payload, queued.headers
        elif path in DEFAULT_PAYLOADS:
            status, payload, extra = 200, DEFAULT_PAYLOADS[path], {}
        else:
            self._problem(f"no queued response and no default for {path}")
            status, payload, extra = 404, {"detail": f"no route {path}"}, {}

        encoded = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        # Content-Length, not connection-close framing: this is what lets the
        # client hand the connection back to its pool instead of dropping it.
        handler.send_header("Content-Length", str(len(encoded)))
        for name, value in extra.items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(encoded)
