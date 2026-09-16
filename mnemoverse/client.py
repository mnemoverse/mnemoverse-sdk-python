"""Synchronous Mnemoverse client: one event loop, held open for the client's life."""

from __future__ import annotations

import asyncio
import atexit
import threading
import weakref
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

from mnemoverse._async_client import AsyncMnemoClient
from mnemoverse.types import (
    FeedbackResponse,
    HealthResponse,
    ReadResponse,
    RecentResponse,
    StatsResponse,
    WriteBatchResponse,
    WriteResponse,
)

T = TypeVar("T")


def _inside_a_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _drive_on_a_borrowed_thread(
    loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, T]
) -> T:
    """Run one coroutine on ``loop`` from a thread that is not the caller's.

    A loop may be driven by any thread that is not already driving one of its
    own; it does not have to be the thread that created it. That is what lets a
    caller who is inside somebody else's running loop still finish work on
    ours, which is what :meth:`MnemoClient.close` needs: a socket has to be
    released on the loop it belongs to, not abandoned there.
    """
    result: list[T] = []
    failure: list[BaseException] = []

    def drive() -> None:
        try:
            result.append(loop.run_until_complete(coro))
        except BaseException as exc:  # re-raised on the caller's thread below
            failure.append(exc)

    thread = threading.Thread(target=drive, name="mnemoverse-sync-handover", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


class _OwnLoop:
    """One event loop, created by this client and driven by the calling thread.

    The normal case: a script, a CLI, a test. ``run_until_complete`` returns the
    loop to the caller between calls without closing it, so whatever the HTTP
    stack parked on that loop (a pooled keep-alive connection, most of all) is
    still valid on the next call.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        if _inside_a_running_loop():
            # A thread that is already running a loop cannot drive ours, so
            # borrow one for this single coroutine. Only the handover path
            # below gets here: ordinary calls have moved to _LoopThread by
            # then, and this is how the sockets on THIS loop still get closed
            # on it rather than abandoned open.
            return _drive_on_a_borrowed_thread(self._loop, coro)
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        try:
            if not _inside_a_running_loop():
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        finally:
            self._loop.close()


class _LoopThread:
    """One event loop, owned by a daemon thread of this client's own.

    Used when the caller is already inside a running event loop (a Jupyter
    notebook, or sync code reached from async code): a loop cannot be driven
    from a thread that is already running one. The thread lives as long as the
    client does, so the loop the connection pool is bound to does not go away
    between calls.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            args=(ready,),
            name="mnemoverse-sync-client",
            daemon=True,
        )
        self._thread.start()
        ready.wait()

    def _serve(self, ready: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(ready.set)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        try:
            # Bounded: a thread that already died must not hang the caller.
            future = asyncio.run_coroutine_threadsafe(
                self._loop.shutdown_asyncgens(), self._loop
            )
            future.result(timeout=5.0)
        except Exception:
            pass
        finally:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass  # already stopped or closed: nothing left to ask it
            self._thread.join(timeout=5.0)
            self._loop.close()


# Clients that were never closed. Closing them at interpreter exit keeps an
# unclosed client from printing "unclosed event loop" / "unclosed transport"
# warnings at a moment when nobody can act on them.
_LIVE_CLIENTS: weakref.WeakSet[MnemoClient] = weakref.WeakSet()


@atexit.register
def _close_live_clients() -> None:
    for client in list(_LIVE_CLIENTS):
        try:
            client.close()
        except Exception:
            pass


class MnemoClient:
    """Synchronous client for the Mnemoverse Memory API.

    Wraps AsyncMnemoClient for use in scripts, notebooks, and sync applications.
    For async applications (FastAPI, Discord bots), use AsyncMnemoClient directly.

    Usage:
        # export MNEMOVERSE_API_KEY=mk_live_...
        client = MnemoClient()
        result = client.write("Caching reduces latency", concepts=["caching"])
        memories = client.read("how to reduce latency?")

    ``api_key`` is optional: when it is ``None`` or empty, the key is read
    from the ``MNEMOVERSE_API_KEY`` environment variable. An explicit
    ``api_key`` always wins over the environment. Raises ``ValueError`` when
    neither is set.

    Event loop, and why it is worth a paragraph: the client creates ONE event
    loop the first time it is used and keeps it until :meth:`close`. Every call
    runs on that loop, so the connection pool underneath survives between calls
    and keep-alive works. Up to 0.2.0 each call ran ``asyncio.run()``, a fresh
    loop closed on the way out, while the ``httpx.AsyncClient`` was cached
    across calls: its pooled connection stayed bound to the closed loop and
    every second call died with ``RuntimeError: Event loop is closed``.

    Closing is optional but cheap, and a context manager does it for you::

        with MnemoClient() as client:
            client.write("something worth keeping")

    A client that is never closed is closed at interpreter exit instead, with
    no warning. After :meth:`close` the client is still usable: the next call
    opens a fresh loop and a fresh connection.

    Inside a running event loop (a notebook cell, or sync code reached from
    async code) the loop cannot be the caller's, so the client moves onto a
    daemon thread of its own and stays there for the rest of its life. Still
    one loop, still one pool, just not this thread's.

    One client, one caller at a time: the loop is driven by whichever thread
    calls, so a single ``MnemoClient`` shared between threads is not supported.
    Give each thread its own, or use ``AsyncMnemoClient``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://core.mnemoverse.com",
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self._async_client = AsyncMnemoClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._runner: _OwnLoop | _LoopThread | None = None
        self._lock = threading.Lock()
        _LIVE_CLIENTS.add(self)

    def _acquire_runner(self) -> _OwnLoop | _LoopThread:
        with self._lock:
            runner = self._runner
            if isinstance(runner, _LoopThread):
                # Already living on our own thread. Stay there: a pooled
                # connection must not be handed from one loop to another.
                return runner
            if _inside_a_running_loop():
                if runner is not None:
                    # This client has been driven from plain sync code before,
                    # so its connection belongs to a loop we are about to stop
                    # using. Close it there, then let it reopen on the thread.
                    self._runner = None
                    try:
                        self._shutdown(runner)
                    except Exception:
                        # The old loop is closed either way, and the pool has
                        # been let go. A noisy goodbye must not fail the call
                        # the caller actually made.
                        pass
                worker = _LoopThread()
                self._runner = worker
                return worker
            if runner is None:
                runner = _OwnLoop()
                self._runner = runner
            return runner

    def _shutdown(self, runner: _OwnLoop | _LoopThread) -> None:
        """Close the HTTP client on the loop that owns its connections, then
        close the loop. Order matters: the other way round leaks a socket.

        The loop is closed even if closing the HTTP client failed, because a
        loop nobody closes is the warning at interpreter exit this whole class
        exists to avoid.
        """
        try:
            runner.run(self._async_client.close())
        except Exception:
            # The pool could not be closed on its own loop and that loop is
            # going away anyway. Let go of it rather than hand the next call a
            # connection bound to a dead loop, which is the 0.2.0 failure.
            self._async_client._forget_client()
            raise
        finally:
            runner.close()

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return self._acquire_runner().run(coro)

    def write(
        self,
        content: str,
        *,
        concepts: list[str] | None = None,
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        external_ref: str | None = None,
    ) -> WriteResponse:
        """Store a single memory atom."""
        return self._run(
            self._async_client.write(
                content, concepts=concepts, domain=domain,
                metadata=metadata, external_ref=external_ref,
            )
        )

    def write_batch(self, items: list[dict[str, Any]]) -> WriteBatchResponse:
        """Store up to 500 atoms in one request."""
        return self._run(self._async_client.write_batch(items))

    def read(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: str | None = None,
        min_relevance: float = 0.3,
        include_associations: bool = True,
        concepts: list[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        order_by: str | None = None,
        exclude_author: str | None = None,
    ) -> ReadResponse:
        """Query memory with semantic search + Hebbian expansion.

        ``since`` / ``until`` bound the result by creation time, inclusive at
        both ends. ``order_by="recency"`` re-sorts the matched set newest-first
        without changing which entries matched. ``exclude_author`` drops one
        author principal.

        For "what happened lately" rather than "what do I know about X", use
        :meth:`recent` — search returns what MATCHES, so entries that exist but
        do not match are absent, correctly but invisibly.
        """
        return self._run(
            self._async_client.read(
                query, top_k=top_k, domain=domain,
                min_relevance=min_relevance,
                include_associations=include_associations,
                concepts=concepts,
                since=since, until=until,
                order_by=order_by, exclude_author=exclude_author,
            )
        )

    def recent(
        self,
        *,
        domain: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        exclude_author: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> RecentResponse:
        """List the newest entries first — no query, complete by construction.

        The temporal complement of :meth:`read`: nothing is ranked away, so use
        this when you need to be sure you are seeing everything. Paged by
        ``next_cursor``, which continues the listing with no skips and no
        duplicates even while writes are landing.
        """
        return self._run(
            self._async_client.recent(
                domain=domain, since=since, until=until,
                exclude_author=exclude_author, limit=limit, cursor=cursor,
            )
        )

    def feedback(
        self,
        atom_ids: list[UUID | str],
        outcome: float,
        *,
        concepts: list[str] | None = None,
        query_concepts: list[str] | None = None,
        domain: str = "general",
    ) -> FeedbackResponse:
        """Report outcome (success/failure) for memories."""
        return self._run(
            self._async_client.feedback(
                atom_ids, outcome, concepts=concepts,
                query_concepts=query_concepts, domain=domain,
            )
        )

    def stats(self) -> StatsResponse:
        """Get memory statistics."""
        return self._run(self._async_client.stats())

    def health(self) -> HealthResponse:
        """Check API health."""
        return self._run(self._async_client.health())

    def close(self) -> None:
        """Close the HTTP client and the event loop this client owns.

        Safe to call twice, and safe not to call at all. The client stays
        usable afterwards: the next call opens a fresh loop and connection.
        """
        with self._lock:
            runner, self._runner = self._runner, None
        if runner is None:
            return
        self._shutdown(runner)

    def __enter__(self) -> MnemoClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Last resort for a client that was neither closed nor alive at exit.
        # Nothing here may raise: __del__ runs at times the caller cannot see.
        try:
            self.close()
        except Exception:
            pass
