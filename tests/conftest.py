"""Fixtures shared by the suite: something real to talk to."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.keepalive_server import MockCore


@pytest.fixture
def core() -> Iterator[MockCore]:
    """A local core on a real socket, speaking HTTP/1.1 with keep-alive.

    Strict on the way out, the way the transport mock it replaces was: a queued
    response nobody asked for, or a request nobody prepared for, fails the test
    that left it behind.
    """
    server = MockCore()
    try:
        yield server
        assert server.problems == [], f"the server complained: {server.problems}"
        assert server.pending == 0, f"{server.pending} queued response(s) nobody requested"
    finally:
        server.stop()
