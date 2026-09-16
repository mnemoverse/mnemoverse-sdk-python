"""The sync client must survive its own second call.

Until this fix it did not. ``MnemoClient._run`` called ``asyncio.run`` once per
call (a fresh event loop, closed on the way out) while ``AsyncMnemoClient``
cached one ``httpx.AsyncClient`` across calls. The pooled keep-alive connection
stayed bound to the loop that had just been closed, so the second call raised
``RuntimeError: Event loop is closed``; the failure closed the HTTP client, and
the third call succeeded on a fresh one. Every second call, for the whole life
of 0.2.0.

These tests need a server that keeps connections alive, which is why they talk
to ``tests.keepalive_server`` instead of a transport mock. With nothing pooled
there is nothing to strand, and a transport mock stays green against this bug.
That is how it shipped.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

from mnemoverse import AsyncMnemoClient, MnemoClient
from tests.keepalive_server import MockCore

API_KEY = "mk_test_abc123"


def _sync_client(core: MockCore) -> MnemoClient:
    return MnemoClient(api_key=API_KEY, base_url=core.url, timeout=5.0, max_retries=0)


def _async_client(core: MockCore) -> AsyncMnemoClient:
    return AsyncMnemoClient(api_key=API_KEY, base_url=core.url, timeout=5.0, max_retries=0)


def test_three_health_calls_on_one_sync_client(core: MockCore) -> None:
    """The bug at its smallest: the second call used to raise."""
    client = _sync_client(core)
    try:
        assert [client.health().status for _ in range(3)] == ["ok", "ok", "ok"]
    finally:
        client.close()

    assert len(core.requests) == 3
    # The harness earns its keep here: one TCP connection for three calls means
    # the connection really was kept alive, so the test could have seen the bug.
    assert core.connections == 1


def test_three_writes_on_one_sync_client(core: MockCore) -> None:
    """A second method, with a request body, on the same reused connection."""
    client = _sync_client(core)
    try:
        for i in range(3):
            assert client.write(f"memory {i}", concepts=["test"]).stored is True
    finally:
        client.close()

    assert [r.path for r in core.requests] == ["/api/v1/memory/write"] * 3
    assert core.connections == 1


def test_three_reads_on_one_sync_client(core: MockCore) -> None:
    """And a third, so the fix is not pinned to one code path."""
    client = _sync_client(core)
    try:
        for _ in range(3):
            result = client.read("what survived")
            assert result.items[0].content == "a memory that survived the round trip"
    finally:
        client.close()

    assert core.connections == 1


def test_the_sync_client_keeps_working_after_close(core: MockCore) -> None:
    """``close`` ends an epoch, not the client: the next call opens a fresh
    loop and a fresh connection. Documented in the class docstring, so pinned."""
    client = _sync_client(core)
    assert client.health().status == "ok"
    client.close()
    assert client.health().status == "ok"
    client.close()

    # Two epochs, two connections, and the second one proves the first was
    # really closed rather than silently reused.
    assert core.connections == 2
    assert len(core.requests) == 2


def test_closing_twice_is_not_an_error(core: MockCore) -> None:
    client = _sync_client(core)
    client.health()
    client.close()
    client.close()


def test_the_sync_client_is_a_context_manager(core: MockCore) -> None:
    with _sync_client(core) as client:
        assert [client.health().status for _ in range(3)] == ["ok", "ok", "ok"]

    assert core.connections == 1


def test_async_client_does_three_calls_inside_one_asyncio_run(core: MockCore) -> None:
    """The async client was never broken. Pinned so a fix to the sync wrapper
    cannot quietly cost the async one its connection reuse."""

    async def three() -> list[str]:
        client = _async_client(core)
        try:
            return [(await client.health()).status for _ in range(3)]
        finally:
            await client.close()

    assert asyncio.run(three()) == ["ok", "ok", "ok"]
    assert core.connections == 1


async def test_the_sync_client_works_from_inside_a_running_loop(core: MockCore) -> None:
    """The notebook shape: sync calls made from code that is already async.

    The client cannot drive a loop on this thread, so it moves onto a worker
    thread of its own and stays there: one loop, one pool, all three calls.
    """
    client = _sync_client(core)
    try:
        assert [client.health().status for _ in range(3)] == ["ok", "ok", "ok"]
    finally:
        client.close()

    assert core.connections == 1


def test_a_client_used_from_sync_code_and_then_from_async_code(core: MockCore) -> None:
    """The one handover that needs care.

    The first call binds the pool to a caller-driven loop; the call from inside
    ``asyncio.run`` cannot use that loop, so the client closes it, moves to its
    worker thread, and carries on. The last call proves it stays there.
    """
    client = _sync_client(core)

    async def from_async() -> str:
        return client.health().status

    try:
        assert client.health().status == "ok"
        assert asyncio.run(from_async()) == "ok"
        assert client.health().status == "ok"
    finally:
        client.close()

    assert len(core.requests) == 3
    # Two epochs: the caller-driven loop, then the worker thread's.
    assert core.connections == 2


def test_a_client_that_is_never_closed_exits_quietly(core: MockCore) -> None:
    """A client holding an open loop must not turn the exit into a warning.

    Nobody calls ``close`` in a five-line script, and "unclosed event loop" at
    interpreter exit arrives when there is nothing left to do about it. The
    ``atexit`` hook is what makes the silence, and a subprocess is the only
    honest place to check it.
    """
    script = (
        "import warnings\n"
        "from mnemoverse import MnemoClient\n"
        "warnings.simplefilter('error', ResourceWarning)\n"
        f"client = MnemoClient(api_key={API_KEY!r}, base_url={core.url!r}, max_retries=0)\n"
        "print(client.health().status)\n"
        "print(client.health().status)\n"
    )
    finished = subprocess.run(
        [sys.executable, "-W", "error::ResourceWarning", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.split() == ["ok", "ok"]
    assert finished.stderr == ""
