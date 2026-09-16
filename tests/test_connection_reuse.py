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
import os
import subprocess
import sys
import threading
import time
from inspect import CORO_CLOSED, getcoroutinestate

import pytest

import mnemoverse.client as client_module
from mnemoverse import AsyncMnemoClient, MnemoClient
from tests.keepalive_server import MockCore

API_KEY = "mk_test_abc123"

# Interpreter debug switches that make Python itself write to stderr. A test
# asserting that WE are silent must not inherit them from whoever ran pytest.
_CHATTY_ENV = frozenset(
    {
        "PYTHONASYNCIODEBUG",
        "PYTHONDEVMODE",
        "PYTHONTRACEMALLOC",
        "PYTHONVERBOSE",
        "PYTHONWARNINGS",
    }
)


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

    Closing the old pool needs a loop driven from a thread that is not the
    caller's, because the caller is inside ``asyncio.run`` and cannot drive a
    second loop. That is what ``_drive_on_a_borrowed_thread`` is for, and
    ``live_connections`` is what makes the difference visible: skip the
    borrowed thread and the first socket is not closed, it is abandoned open.
    ``connections`` cannot see that, because two connections were opened
    either way.

    The test holds the first pool for the same reason. Abandon it and CPython
    refcounting closes the socket a moment later anyway, so "the client
    released it" and "the interpreter tidied up after the client" look
    identical from the server. One reference is enough to tell them apart, and
    telling them apart is the whole assertion.
    """
    client = _sync_client(core)

    async def from_async() -> str:
        return client.health().status

    first_pool = None
    try:
        assert client.health().status == "ok"
        first_pool = client._async_client._client
        assert first_pool is not None and not first_pool.is_closed

        assert asyncio.run(from_async()) == "ok"

        # Closed on its own loop, not dropped on the floor.
        assert first_pool.is_closed, "the first pool was let go rather than closed"
        # And the server agrees: two opened, one still open.
        assert core.live_settles_at(1) == 1, "the first loop's socket was abandoned open"

        assert client.health().status == "ok"
    finally:
        client.close()

    assert len(core.requests) == 3
    # Two epochs: the caller-driven loop, then the worker thread's.
    assert core.connections == 2
    # And close() ended the second epoch as well: nothing is left open.
    assert core.live_settles_at(0) == 0


def test_a_client_that_is_never_closed_exits_quietly(core: MockCore) -> None:
    """A client holding an open loop must not turn the exit into a warning.

    Nobody calls ``close`` in a five-line script, and "unclosed event loop" at
    interpreter exit arrives when there is nothing left to do about it. The
    ``atexit`` hook is what makes the silence, and a subprocess is the only
    honest place to check it.

    The child's environment is pruned rather than inherited whole. This test
    asserts an empty stderr, and ``PYTHONASYNCIODEBUG=1`` in the shell that
    started pytest makes asyncio narrate on stderr for reasons that have
    nothing to do with this SDK. A test that fails because of how the developer
    configured their terminal is reporting on the terminal.
    """
    script = (
        "import warnings\n"
        "from mnemoverse import MnemoClient\n"
        "warnings.simplefilter('error', ResourceWarning)\n"
        f"client = MnemoClient(api_key={API_KEY!r}, base_url={core.url!r}, max_retries=0)\n"
        "print(client.health().status)\n"
        "print(client.health().status)\n"
    )
    quiet_env = {k: v for k, v in os.environ.items() if k not in _CHATTY_ENV}
    finished = subprocess.run(
        [sys.executable, "-W", "error::ResourceWarning", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=quiet_env,
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.split() == ["ok", "ok"]
    assert finished.stderr == ""


def test_a_client_collected_by_the_garbage_collector_is_silent(core: MockCore) -> None:
    """An ordinary reference cycle must not turn into a traceback.

    A ``__del__`` that called ``close()`` ran inside the collection pass and
    drove the event loop from there, which is not allowed: on Windows it wrote
    "Error on reading from the event loop self pipe" and eleven lines of
    traceback every time a plain parent/child cycle holding a client was
    collected. 0.2.0 printed nothing, so it was a regression, and the ``atexit``
    hook above already covers the case ``__del__`` was added for.

    Default warning filters on purpose: this is about what an ordinary user
    sees. A client that is collected without ``close()`` did leak its loop, and
    Python reports that as a ``ResourceWarning``, which is off by default and
    which the user asked for if they turned it on. A traceback is not a
    warning, and nobody asked for it.
    """
    script = (
        "import gc\n"
        "from mnemoverse import MnemoClient\n"
        "class Agent:\n"
        "    def __init__(self, url):\n"
        f"        self.client = MnemoClient(api_key={API_KEY!r}, base_url=url, max_retries=0)\n"
        "        self.on_done = self._done  # bound method -> self: an ordinary cycle\n"
        "    def _done(self):\n"
        "        pass\n"
        "def go():\n"
        f"    print(Agent({core.url!r}).client.health().status)\n"
        "go()\n"
        "gc.collect()\n"
        "print('collected')\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k not in _CHATTY_ENV},
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.split() == ["ok", "collected"]
    assert finished.stderr == ""


def _hammer(client: MnemoClient, threads: int, each: int) -> tuple[list[str], list[str]]:
    """Call ``health`` from several threads at once; report answers and failures."""
    answers: list[str] = []
    failures: list[str] = []
    guard = threading.Lock()

    def caller() -> None:
        for _ in range(each):
            try:
                status = client.health().status
            except BaseException as exc:  # the failure IS the finding here
                with guard:
                    failures.append(f"{type(exc).__name__}: {exc}")
                return
            with guard:
                answers.append(status)

    workers = [threading.Thread(target=caller) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert not any(worker.is_alive() for worker in workers), "a caller thread never finished"
    return answers, failures


def test_one_client_shared_between_threads_from_sync_code(core: MockCore) -> None:
    """Four threads, one client, every call answered.

    An event loop cannot be entered twice, so a client whose loop is driven by
    whichever thread calls used to answer one thread and raise ``RuntimeError:
    This event loop is already running`` at the others. Sharing a client
    between threads is not the recommended shape, but "raises at random"
    is not an acceptable way to say so, and the failure depended on where the
    FIRST call came from: the same client reached first from inside a running
    loop lives on a worker thread and was always fine (the test below). Calls
    are serialised now, so both shapes behave the same way.
    """
    client = _sync_client(core)
    try:
        answers, failures = _hammer(client, threads=4, each=15)
        assert failures == []
        assert answers == ["ok"] * 60
        # Serialised, not parallel: one loop, one pool, one connection for all
        # sixty calls. This is the cost of sharing, and it is the documented one.
        assert core.connections == 1
    finally:
        client.close()


async def test_one_client_shared_between_threads_from_async_code(core: MockCore) -> None:
    """The same, for a client that lives on its own worker thread.

    This shape already worked, through ``run_coroutine_threadsafe``. Pinned so
    the two shapes cannot drift apart again.
    """
    client = _sync_client(core)
    try:
        assert client.health().status == "ok"  # binds the client to its worker thread
        answers, failures = _hammer(client, threads=4, each=15)
        assert failures == []
        assert answers == ["ok"] * 60
    finally:
        client.close()


def test_a_call_that_never_reaches_a_loop_closes_its_coroutine(
    core: MockCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that fails on the way in must not leave a coroutine behind.

    ``client.health()`` builds the coroutine at the call site and then hands it
    over. If the hand-over raises, nobody ever awaits it, and Python reports
    that later from wherever the collection happened to run: "coroutine
    AsyncMnemoClient.health was never awaited", pointing at a line that has
    nothing to do with it. That is what the shared-client failure printed on
    top of its real error, and the guard outlives that bug.

    Asserted on the coroutine rather than on the warning, because the warning
    fires whenever the object is finalised and the assertion should not depend
    on when that is.
    """
    client = _sync_client(core)

    def refuse(self: MnemoClient) -> object:
        raise RuntimeError("injected: the runner could not be acquired")

    monkeypatch.setattr(client_module.MnemoClient, "_acquire_runner", refuse)

    coro = client._async_client.health()
    with pytest.raises(RuntimeError, match="injected"):
        client._run(coro)

    assert getcoroutinestate(coro) == CORO_CLOSED


def test_a_worker_loop_that_cannot_start_raises_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside the worker thread has to reach the caller.

    ``_LoopThread`` waits for its loop to signal that it is running. Waiting
    with no bound means any failure in there, an OS refusing the self-pipe
    being the realistic one, stops an ordinary ``client.read()`` dead: no
    exception, no message, no traceback, forever. The wait is bounded now and
    the reason the thread died is re-raised on the caller's thread.
    """
    real_new_event_loop = asyncio.new_event_loop
    sabotaged_one: list[int] = []

    class WillNotRun:
        """A loop that cannot be served, standing in for one that fails to start."""

        def __init__(self, inner: asyncio.AbstractEventLoop) -> None:
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name: str) -> object:
            return getattr(object.__getattribute__(self, "_inner"), name)

    def new_event_loop() -> object:
        inner = real_new_event_loop()
        if sabotaged_one:
            return inner
        sabotaged_one.append(1)
        return WillNotRun(inner)

    monkeypatch.setattr(asyncio, "new_event_loop", new_event_loop)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="mnemoverse"):
        client_module._LoopThread()
    # The bound is 5s; anything under half a minute proves it is not unbounded,
    # without making the assertion about how fast this machine happens to be.
    assert time.monotonic() - started < 30.0


def test_a_worker_loop_that_never_answers_at_all_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the bound itself, for a thread that neither starts nor dies.

    The test above covers the failure that announces itself: something in
    ``_serve`` raises, and the constructor re-raises it. This one covers the
    failure that says nothing, a thread that is simply wedged, where the only
    thing standing between the caller and a permanent hang is the timeout. The
    shipped bound is shortened here so the suite does not spend five seconds
    proving it exists.

    Built on a thread of its own, deliberately: a bug here is an unbounded
    wait, and a test that reproduces it by hanging tells nobody anything. This
    way it fails.
    """
    monkeypatch.setattr(client_module, "_LOOP_THREAD_START_TIMEOUT", 0.2)
    monkeypatch.setattr(
        client_module._LoopThread, "_serve", lambda self, ready: time.sleep(5.0)
    )
    outcome: list[str] = []

    def build() -> None:
        try:
            client_module._LoopThread()
        except BaseException as exc:
            outcome.append(f"{type(exc).__name__}: {exc}")
        else:
            outcome.append("constructed")

    builder = threading.Thread(target=build, daemon=True)
    builder.start()
    builder.join(timeout=20.0)

    assert not builder.is_alive(), "the constructor is still waiting for a loop that never came"
    assert outcome[0].startswith("RuntimeError"), outcome
    assert "mnemoverse" in outcome[0]
