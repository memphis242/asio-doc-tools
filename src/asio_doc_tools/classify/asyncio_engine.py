"""The default engine: every batch is a task on one asyncio event loop, sending its
requests through the async Anthropic client and storing its results itself as soon as
they settle.

Everything but scheduling, waiting, cancellation, deadlines, and Ctrl-C comes from the
shared modules, as in the threaded reference: the per-batch protocol (batch.py), the
budget, the retry rules, the store, and the run's messages. What differs, and why it is
the default (bench/RESULTS.md, "Cancellation"): a request in flight is an `await` the
loop can cancel, which closes its connection within milliseconds, so the hard limit and
a second Ctrl-C end the run cleanly (every `finally` runs, every store write commits)
instead of abandoning threads blocked in socket reads and exiting the process under them.

Design choices:

- One thread. Every batch task, the SIGINT handler, the hard-limit timer, and every
  store write run on the loop thread, so the budget's admit() (which never suspends) is
  atomic and there is never more than one writer to the store.
- Store writes are synchronous, on the loop thread, rather than `asyncio.to_thread`: a
  write takes milliseconds, sqlite connections belong to one thread, and a thread could
  not be interrupted mid-write anyway. What could block, waiting for another process's
  lock, waits no longer than the time left before the hard limit (run.store_lock_wait_s),
  and falls back to the pending directory, so the loop is never held past it.
- SIGINT. asyncio.run installs its own SIGINT handler, which cancels the whole run on the
  first Ctrl-C; that is right until the first batch starts, when nothing is in flight.
  From then on `loop.add_signal_handler` replaces it with this engine's two stages, and
  every exit path removes it again; `run_batches` puts back whatever handler was in
  place before the run, since asyncio.run only restores one it installed itself.
- The hard limit is a `loop.call_at` timer that cancels every batch task. A cancelled
  task's request settles as possibly billed, the batch protocol hands back what the batch
  classified before it (a split's first half), and the task stores that before ending.
"""

import asyncio
import contextlib
import signal
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..diag import AsioDocsError, note, spend
from ..versions import Version
from . import batch, budget, run
from .batch import BatchOutcome, BatchProtocol
from .budget import RunBudget, StopFlag
from .store import Classification, Recorder

ClientFactory = Callable[[], Any]


def default_client_factory() -> Any:
    """The async client every request of this engine goes through, used inside
    `async with`. The SDK's own retries are off: each retry is made by the batch protocol,
    so the run budget admits and counts it. The timeouts are explained in the "Run budget"
    comment in budget.py."""
    import anthropic
    import httpx

    client = anthropic.AsyncAnthropic(
        max_retries=0, timeout=httpx.Timeout(budget.REQUEST_TIMEOUT_S, connect=budget.CONNECT_TIMEOUT_S)
    )
    run.check_credentials(client)
    return client


class _Ending(Enum):
    DONE = "done"
    INTERRUPTED = "interrupted"
    HARD_LIMIT = "hard limit"


@dataclass(frozen=True, slots=True)
class _Result:
    ending: _Ending
    message: str = ""


class _Run:
    """One run on the event loop: its batch tasks, and how they are stopped."""

    def __init__(
        self,
        plan: run.Plan,
        run_budget: RunBudget,
        recorder: Recorder,
        texts: Mapping[str, str],
        client_factory: ClientFactory,
    ) -> None:
        self._plan: Final = plan
        self._budget: Final = run_budget
        self._stop: Final[StopFlag] = run_budget.stop
        self._recorder: Final = recorder
        self._texts: Final = texts
        self._client_factory: Final = client_factory
        self._report_usage: Final = run.UsageReport(run_budget)
        self._slots: asyncio.Semaphore | None = None
        self._stopped: asyncio.Event | None = None
        self._client: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._queued: set[asyncio.Task[None]] = set()  # tasks still waiting for a slot
        self._interrupts = 0
        self._abandoned: int | None = None  # requests in flight when the second Ctrl-C came
        self._hard_limit_in_flight: int | None = None  # requests in flight at the hard limit

    async def main(self) -> _Result:
        loop = asyncio.get_running_loop()
        # The hard limit is set on the budget's clock and run on the loop's: they are one clock.
        assert abs(loop.time() - self._budget.clock()) < 1.0
        self._slots = asyncio.Semaphore(budget.MAX_IN_FLIGHT)
        self._stopped = asyncio.Event()

        def on_stop() -> None:
            loop.call_soon_threadsafe(self._on_stop)

        self._stop.on_trip(on_stop)
        async with self._client_factory() as client:
            self._client = client
            spend(budget.preview_line(self._plan.payload_bytes, self._plan.entries, self._plan.caps))
            loop.add_signal_handler(signal.SIGINT, self._on_sigint)
            hard_limit = loop.call_at(run.hard_limit_at(self._budget), self._on_hard_limit)
            try:
                async with asyncio.TaskGroup() as group:
                    for batch_keys in self._plan.batches:
                        task = group.create_task(self._batch(batch_keys))
                        self._tasks.append(task)
                        self._queued.add(task)
            finally:
                hard_limit.cancel()
                loop.remove_signal_handler(signal.SIGINT)
                self._report_usage()
        return self._result()

    def _result(self) -> _Result:
        assert all(task.done() for task in self._tasks)
        assert self._budget.usage().in_flight_requests == 0, "every attempt is settled once its task is done"
        progress = self._recorder.progress(self._plan.entries)
        if self._hard_limit_in_flight is not None:
            return _Result(_Ending.HARD_LIMIT, run.hard_limit_message(self._hard_limit_in_flight, progress))
        if self._abandoned is not None:
            spend(run.abandoned_line(self._abandoned, progress))
            return _Result(_Ending.INTERRUPTED)
        if self._interrupts:
            note(run.interrupted_line(progress))
            return _Result(_Ending.INTERRUPTED)
        return _Result(_Ending.DONE)

    async def _batch(self, batch_keys: Sequence[str]) -> None:
        """One batch's task: waits for a slot, runs the batch protocol, and stores what it
        classified, even when the task is cancelled midway."""
        assert self._slots is not None
        task = asyncio.current_task()
        assert task is not None
        try:
            async with self._slots:
                self._queued.discard(task)
                outcome, cancelled = await self._drive(
                    batch.classify_batch(self._budget, batch_keys, self._texts)
                )
        finally:
            self._queued.discard(task)
        self._recorder.record(outcome.classified, outcome.failure)
        if cancelled:
            raise asyncio.CancelledError

    async def _drive(self, protocol: BatchProtocol) -> tuple[BatchOutcome, bool]:
        """Runs a batch's protocol to its end on the loop; returns its outcome, and whether
        the task was cancelled meanwhile. A cancellation lands at the `await` of a request
        or a retry's wait: the protocol settles the request as possibly billed and hands
        back what the batch classified before it."""
        cancelled = False
        try:
            effect = next(protocol)
            while True:
                if cancelled:
                    effect = protocol.throw(batch.Cancelled())
                    continue
                try:
                    match effect:
                        case batch.Call(kwargs=kwargs):
                            reply = await self._client.messages.create(**kwargs)
                        case batch.Sleep(seconds=seconds):
                            await self._stop_or_timeout(seconds)
                            reply = None
                except asyncio.CancelledError:
                    cancelled = True
                    # No other request is admitted once a task is cancelled, whatever did it.
                    self._stop.trip(AsioDocsError("the run was cancelled"))
                    effect = protocol.throw(batch.Cancelled())
                except BaseException as e:
                    effect = protocol.throw(e)
                else:
                    effect = protocol.send(reply)
        except StopIteration as finished:
            return finished.value, cancelled

    async def _stop_or_timeout(self, seconds: float) -> None:
        assert self._stopped is not None
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self._stopped.wait()

    def _in_flight(self) -> int:
        return self._budget.usage().in_flight_requests

    def _cancel_unfinished(self) -> None:
        for task in self._tasks:
            task.cancel()  # a no-op on a task already done

    def _on_stop(self) -> None:
        """The stop flag was set: retry waits end at once, and, after an error (not a
        Ctrl-C or the hard limit, which say so themselves), the run says it is stopping."""
        assert self._stopped is not None
        self._stopped.set()
        if self._interrupts == 0 and self._hard_limit_in_flight is None and (in_flight := self._in_flight()):
            note(run.stopping_early_line(in_flight))

    def _on_sigint(self) -> None:
        """First Ctrl-C: no new request is sent, the batches waiting for a slot are dropped,
        and the run waits for the requests in flight, already paid for, and stores them.
        Second Ctrl-C: the requests in flight are cancelled too."""
        self._interrupts += 1
        if self._interrupts == 1:
            self._stop.trip(AsioDocsError("interrupted"))
            for task in tuple(self._queued):
                task.cancel()  # has not sent anything
            if in_flight := self._in_flight():
                spend(run.waiting_line(in_flight, interrupted=True))
        elif self._abandoned is None:
            self._abandoned = self._in_flight()
            self._cancel_unfinished()

    def _on_hard_limit(self) -> None:
        self._hard_limit_in_flight = self._in_flight()
        self._stop.trip(
            AsioDocsError(f"stopped at this run's {budget.hard_limit_s():.0f} s wall-clock limit")
        )
        self._cancel_unfinished()


def run_batches(
    conn: sqlite3.Connection,
    store_path: Path,
    pending: Sequence[str],
    texts: dict[str, str],
    release_by_key: Mapping[str, Version],
    store: dict[str, Classification],
    client_factory: ClientFactory | None,
) -> None:
    """Classifies the `pending` keys into `store` (and the store file) within one run budget,
    at most budget.MAX_IN_FLIGHT batches at a time on one event loop.

    Must be called from the main thread, which alone can take signals: the run handles
    Ctrl-C itself. Raises KeyboardInterrupt after a Ctrl-C (exit status 130) and
    AsioDocsError at the hard limit or when the run stopped early (status 1).
    """
    assert threading.current_thread() is threading.main_thread()
    plan = run.plan_run(conn, store_path, pending, texts)
    stop = StopFlag()
    run_budget = RunBudget(plan.caps, stop)
    recorder = Recorder(
        conn, store_path, store, texts, release_by_key, stop, lambda: run.store_lock_wait_s(run_budget)
    )
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        result = asyncio.run(
            _Run(plan, run_budget, recorder, texts, client_factory or default_client_factory).main()
        )
    finally:
        if previous_sigint is not None and signal.getsignal(signal.SIGINT) is not previous_sigint:
            signal.signal(signal.SIGINT, previous_sigint)
    match result.ending:
        case _Ending.INTERRUPTED:
            raise KeyboardInterrupt
        case _Ending.HARD_LIMIT:
            raise AsioDocsError(result.message)
        case _Ending.DONE:
            run.finish(recorder, stop, plan.entries)
