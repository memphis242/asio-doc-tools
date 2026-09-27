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
- A batch task raises nothing but CancelledError, so an unexpected exception in one batch
  (a bug) never makes the TaskGroup cancel the others and their paid requests: it stops
  the run like a batch's failure, what it concerns is saved for the next run, and it is
  raised once every batch has ended, as the threaded reference does.
- Teardown is bounded. The only threads this path creates are the loop's default
  executor's: `loop.getaddrinfo` (anyio's DNS lookups for httpcore) and `asyncio.to_thread`
  (the SDK's platform detection and credential refresh). The loop gets
  `_DaemonThreadExecutor` instead, whose jobs run on daemon threads and whose shutdown
  waits at most 2 s, so a job stuck in the kernel holds neither asyncio.run's shutdown
  (300 s for a ThreadPoolExecutor) nor the interpreter's exit. As a last resort, after a
  hard limit or a Ctrl-C, a non-daemon thread the run started and that is still alive 4 s
  after every batch ended makes the run flush its output and exit the process, with the
  same status (1 or 130), after storing and reporting as usual.
"""

import asyncio
import concurrent.futures
import contextlib
import inspect
import os
import signal
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..diag import AsioDocsError, error, note, spend, warn
from ..versions import Version
from . import batch, budget, run
from .batch import BatchOutcome, BatchProtocol
from .budget import RunBudget, StopFlag
from .store import Classification, Recorder

ClientFactory = Callable[[], Any]

# The loop's default executor waits this long for its jobs when the run ends; a run that
# was stopped exits the process this long after every batch ended if a thread it started
# is still alive (see the module docstring).
_EXECUTOR_SHUTDOWN_GRACE_S: Final = 2.0
_TEARDOWN_GRACE_S: Final = 4.0
assert _EXECUTOR_SHUTDOWN_GRACE_S < _TEARDOWN_GRACE_S <= 5.0

# The process exit of the last resort above; tests replace it.
_exit_process: Callable[[int], Any] = os._exit


class _DaemonThreadExecutor(concurrent.futures.ThreadPoolExecutor):
    """The event loop's default executor: each job (a DNS lookup, the SDK's platform
    detection or credential refresh) runs on its own daemon thread, and shutdown waits at
    most `grace_s` for them. A ThreadPoolExecutor, since asyncio accepts nothing else, but
    none of its own (non-daemon) worker threads is ever started."""

    def __init__(self, grace_s: float) -> None:
        super().__init__(max_workers=1)
        self._grace_s: Final = grace_s
        self._jobs_lock: Final = threading.Lock()
        self._running: set[threading.Thread] = set()
        self._closed = False

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> "concurrent.futures.Future[Any]":
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def job() -> None:
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as e:
                        future.set_exception(e)
                    else:
                        future.set_result(result)
            finally:
                with self._jobs_lock:
                    self._running.discard(threading.current_thread())

        thread = threading.Thread(target=job, name="asio-docs executor", daemon=True)
        with self._jobs_lock:
            if self._closed:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._running.add(thread)
        thread.start()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._jobs_lock:
            self._closed = True
            running = tuple(self._running)
        if wait:
            deadline = time.monotonic() + self._grace_s
            for thread in running:
                thread.join(max(0.0, deadline - time.monotonic()))


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
    batches_ended_at: float = 0.0  # time.monotonic() when every batch task had ended


def _end_early(protocol: BatchProtocol) -> BatchOutcome:
    """Ends a batch protocol its driver could not run to the end, with the run's stop flag
    set: a request left in flight settles as possibly billed, and the protocol hands back
    what it classified before it."""
    match inspect.getgeneratorstate(protocol):
        case inspect.GEN_CREATED | inspect.GEN_CLOSED:
            protocol.close()
            return BatchOutcome({})
    try:
        # With the stop flag set, nothing more is admitted: the protocol ends within a few
        # effects, each answered with another cancellation.
        for _ in range(8):
            protocol.throw(batch.Cancelled())
    except StopIteration as finished:
        return finished.value
    raise AssertionError("a batch protocol went on after being cancelled, with the run stopping")


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
        self._batches_ended_at = 0.0

    @property
    def urgent(self) -> bool:
        """Whether the run is being cut short (a second Ctrl-C, or the hard limit), so a
        store write should not wait for another process's lock at all."""
        return self._abandoned is not None or self._hard_limit_in_flight is not None

    async def main(self) -> _Result:
        loop = asyncio.get_running_loop()
        # The hard limit is set on the budget's clock and run on the loop's: they are one clock.
        assert abs(loop.time() - self._budget.clock()) < 1.0
        loop.set_default_executor(_DaemonThreadExecutor(_EXECUTOR_SHUTDOWN_GRACE_S))
        self._slots = asyncio.Semaphore(budget.MAX_IN_FLIGHT)
        self._stopped = asyncio.Event()

        def on_stop() -> None:
            loop.call_soon_threadsafe(self._on_stop)

        self._stop.on_trip(on_stop)
        handling_sigint = False
        try:
            async with self._client_factory() as client:
                self._client = client
                spend(budget.preview_line(self._plan.payload_bytes, self._plan.entries, self._plan.caps))
                loop.add_signal_handler(signal.SIGINT, self._on_sigint)
                handling_sigint = True
                hard_limit = loop.call_at(run.hard_limit_at(self._budget), self._on_hard_limit)
                try:
                    async with asyncio.TaskGroup() as group:
                        for batch_keys in self._plan.batches:
                            task = group.create_task(self._batch(batch_keys))
                            self._tasks.append(task)
                            self._queued.add(task)
                finally:
                    self._batches_ended_at = time.monotonic()
                    hard_limit.cancel()
                    self._report_usage()
                result = self._result()
        finally:
            # Only now, with usage reported and the client closed: until here a Ctrl-C is
            # this engine's to handle, and cannot cut the report short.
            if handling_sigint:
                loop.remove_signal_handler(signal.SIGINT)
        return result

    def _result(self) -> _Result:
        assert all(task.done() for task in self._tasks)
        assert self._budget.usage().in_flight_requests == 0, "every attempt is settled once its task is done"
        progress = self._recorder.progress(self._plan.entries)
        ended_at = self._batches_ended_at
        if self._hard_limit_in_flight is not None:
            message = run.hard_limit_message(self._hard_limit_in_flight, progress)
            return _Result(_Ending.HARD_LIMIT, message, ended_at)
        if self._abandoned is not None:
            spend(run.abandoned_line(self._abandoned, progress))
            return _Result(_Ending.INTERRUPTED, batches_ended_at=ended_at)
        if self._interrupts:
            note(run.interrupted_line(progress))
            return _Result(_Ending.INTERRUPTED, batches_ended_at=ended_at)
        return _Result(_Ending.DONE, batches_ended_at=ended_at)

    async def _batch(self, batch_keys: Sequence[str]) -> None:
        """One batch's task: waits for a slot, runs the batch protocol, and records what it
        classified, even when the task is cancelled midway.

        Raises nothing but CancelledError (see the module docstring): an unexpected
        exception while driving or recording is handed to _failed instead."""
        assert self._slots is not None
        task = asyncio.current_task()
        assert task is not None
        protocol = batch.classify_batch(self._budget, batch_keys, self._texts)
        outcome: BatchOutcome | None = None
        cancelled = False
        try:
            try:
                async with self._slots:
                    self._queued.discard(task)
                    outcome, cancelled = await self._drive(protocol)
            finally:
                self._queued.discard(task)
            self._recorder.record(outcome.classified, outcome.failure)
        except Exception as e:
            self._failed(e, protocol, outcome)
        if cancelled:
            raise asyncio.CancelledError

    def _failed(self, failure: Exception, protocol: BatchProtocol, outcome: BatchOutcome | None) -> None:
        """An unexpected exception in a batch's task: the run stops as for a batch's own
        failure, `failure` is kept to be raised when every batch has ended, and the batch's
        results, recorded or not, are saved for the next run (importing them twice is
        harmless)."""
        self._stop.trip(AsioDocsError(f"a classification worker failed unexpectedly: {failure!r}"))
        self._recorder.keep_failure(failure)
        self._recorder.salvage((outcome if outcome is not None else _end_early(protocol)).classified)

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
        Second Ctrl-C: the requests in flight are cancelled too. Once every batch has
        ended (the client closing), there is nothing left to stop, and a Ctrl-C does nothing."""
        if self._batches_ended_at:
            return
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
    runs: list[_Run] = []  # the run, once made: its lock wait depends on how it is ending

    def store_lock_wait_s() -> float:
        return 0.0 if runs and runs[0].urgent else run.store_lock_wait_s(run_budget)

    recorder = Recorder(conn, store_path, store, texts, release_by_key, stop, store_lock_wait_s)
    runs.append(_Run(plan, run_budget, recorder, texts, client_factory or default_client_factory))
    threads_before = frozenset(threading.enumerate())
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        result = asyncio.run(runs[0].main())
    finally:
        if previous_sigint is not None and signal.getsignal(signal.SIGINT) is not previous_sigint:
            signal.signal(signal.SIGINT, previous_sigint)
    match result.ending:
        case _Ending.INTERRUPTED:
            _exit_unless_threads_end(threads_before, result.batches_ended_at, status=130, message=None)
            raise KeyboardInterrupt
        case _Ending.HARD_LIMIT:
            _exit_unless_threads_end(
                threads_before, result.batches_ended_at, status=1, message=result.message
            )
            raise AsioDocsError(result.message)
        case _Ending.DONE:
            run.finish(recorder, stop, plan.entries)


def _exit_unless_threads_end(
    threads_before: frozenset[threading.Thread], batches_ended_at: float, *, status: int, message: str | None
) -> None:
    """The last resort of a stopped run (see the module docstring): a non-daemon thread the
    run started would hold the interpreter's exit for as long as it lives, so one still
    alive _TEARDOWN_GRACE_S after every batch ended makes the process exit at once, with
    everything already stored and reported. `message` is the run's error, printed first,
    since the caller will not get to print it."""
    started = [t for t in threading.enumerate() if t not in threads_before and not t.daemon and t.is_alive()]
    deadline = batches_ended_at + _TEARDOWN_GRACE_S
    for thread in started:
        thread.join(max(0.0, deadline - time.monotonic()))
    stuck = [thread.name for thread in started if thread.is_alive()]
    if not stuck:
        return
    try:
        if message is not None:
            error(message)
        warn(f"exiting at once: {', '.join(stuck)} did not end within {_TEARDOWN_GRACE_S:.0f} s of the run")
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        _exit_process(status)
