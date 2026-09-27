"""The threaded reference engine: batches on a thread pool, each blocking in the sync
Anthropic client, results stored by the main thread (see README.md here).

Everything but scheduling, waiting, cancellation, deadlines, and Ctrl-C comes from the
shared modules, as in the asyncio engine: the per-batch protocol (batch.py), the budget,
the retry rules, the store, and the run's messages.
"""

import contextlib
import os
import signal
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

from ...diag import AsioDocsError, error, note, spend, warn
from ...versions import Version
from .. import batch, budget, run
from ..batch import BatchOutcome, BatchProtocol
from ..budget import RunBudget, StopFlag
from ..store import Classification, Recorder

if TYPE_CHECKING:
    from concurrent.futures import Future

ClientFactory = Callable[[], Any]


def default_client_factory() -> Any:
    """The sync client every request of this engine goes through. The SDK's own retries are
    off: each retry is made by the batch protocol, so the run budget admits and counts it.
    The timeouts are explained in the "Run budget" comment in budget.py."""
    import anthropic

    client = anthropic.Anthropic(
        max_retries=0, timeout=anthropic.Timeout(budget.REQUEST_TIMEOUT_S, connect=budget.CONNECT_TIMEOUT_S)
    )
    run.check_credentials(client)
    return client


@dataclass(frozen=True, slots=True)
class _RunContext:
    """What every worker of one run shares."""

    client: Any
    budget: RunBudget
    texts: Mapping[str, str]

    @property
    def stop(self) -> StopFlag:
        return self.budget.stop


def _drive(ctx: _RunContext, protocol: BatchProtocol) -> BatchOutcome:
    """Runs a batch's protocol to its end on this (worker) thread: a request blocks in the
    sync client until it answers or times out, and nothing can interrupt it; a retry's
    wait blocks on the stop flag."""
    try:
        effect = next(protocol)
        while True:
            try:
                match effect:
                    case batch.Call(kwargs=kwargs):
                        reply = ctx.client.messages.create(**kwargs)
                    case batch.Sleep(seconds=seconds):
                        ctx.stop.wait(seconds)
                        reply = None
            except BaseException as e:
                effect = protocol.throw(e)
            else:
                effect = protocol.send(reply)
    except StopIteration as finished:
        return finished.value


def _classify_batch(ctx: _RunContext, batch_keys: Sequence[str]) -> BatchOutcome:
    """Worker entry point. Never raises: see batch.classify_batch."""
    return _drive(ctx, batch.classify_batch(ctx.budget, batch_keys, ctx.texts))


class _Harvester:
    """Hands each finished batch's future to the recorder, on the main thread (which owns
    the sqlite connection), exactly once.

    Never raises for a batch's own failure (see store.Recorder).
    """

    def __init__(self, recorder: Recorder) -> None:
        self._recorder: Final = recorder
        self._harvested: "set[Future[BatchOutcome]]" = set()

    def __call__(self, future: "Future[BatchOutcome]") -> None:
        assert future.done()
        if future in self._harvested:
            return
        if future.cancelled():
            self._harvested.add(future)
            return
        outcome = self._outcome(future)
        self._recorder.record(outcome.classified, outcome.failure)
        # Marked only now: an interrupt during the store write rolls its transaction back
        # and leaves this future to be harvested again.
        self._harvested.add(future)

    def save_for_later(self, futures: Sequence["Future[BatchOutcome]"]) -> None:
        """For a run being abandoned: saves the results of every finished batch not yet
        harvested to the pending directory, without touching the store, whose lock could
        make it wait. The next run imports them."""
        finished = tuple(
            future
            for future in futures
            if future.done() and not future.cancelled() and future not in self._harvested
        )
        self._recorder.save_for_later(tuple(self._outcome(future).classified for future in finished))
        self._harvested.update(finished)

    @staticmethod
    def _outcome(future: "Future[BatchOutcome]") -> BatchOutcome:
        try:
            return future.result()
        except BaseException as e:  # _classify_batch never raises; this keeps a bug in it visible
            return BatchOutcome({}, failure=e)


_Harvest = Callable[["Future[BatchOutcome]"], None]


@contextlib.contextmanager
def _sigint_ignored() -> Iterator[None]:
    """Ignores Ctrl-C (on the main thread, the only one a signal handler can be set from).

    Replacing the handler is what protects the main thread: blocking SIGINT on it with
    pthread_sigmask does not, since the kernel then delivers it to a worker thread and
    Python still raises KeyboardInterrupt on the main thread.
    """
    assert threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous if previous is not None else signal.SIG_DFL)


@dataclass(frozen=True, slots=True)
class _Waiting:
    """How the main thread waits for a run's batches, and gives up on them.

    Every wait ends at the hard limit (`hard_limit_at`, on `clock`), and no batch is
    harvested after it: the run is abandoned instead. Whatever finished is saved for the
    next run, usage is reported, and the process exits.
    """

    harvest: _Harvest
    save_for_later: Callable[[Sequence["Future[BatchOutcome]"]], None]
    stop: StopFlag
    hard_limit_at: float
    clock: Callable[[], float]
    progress: Callable[[], str]
    report_usage: Callable[[], None]
    exit_process: Callable[[int], NoReturn]

    def await_batches(self, futures: Sequence["Future[BatchOutcome]"]) -> None:
        """Harvests each batch as it finishes. Once the stop flag is set, cancels the
        batches not yet started and waits only for the ones in flight."""
        from concurrent.futures import as_completed

        stopping = False
        try:
            for future in as_completed(futures, timeout=self._seconds_to_hard_limit()):
                self._harvest_unless_hard_limit(future, futures)
                if self.stop.is_set() and not stopping:
                    stopping = True
                    for queued in futures:
                        queued.cancel()  # a no-op on a batch already running or done
                    in_flight = sum(1 for f in futures if not f.done())
                    if in_flight:
                        note(run.stopping_early_line(in_flight))
        except TimeoutError:
            self._hard_limit_passed(futures)

    def stop_and_harvest(
        self, futures: Sequence["Future[BatchOutcome]"], *, interrupted: bool
    ) -> Exception | None:
        """After a Ctrl-C (or an unexpected failure) has set the stop flag: cancels the
        batches not yet started, then waits for the ones in flight and harvests every
        finished batch, so their paid-for results are stored. Each in-flight batch ends
        within one attempt's timeout, since the stop flag refuses its further attempts.
        Harvests every batch even when one harvest fails, and returns the first such
        failure.

        A Ctrl-C while waiting abandons the requests in flight, and so does the hard limit.
        """
        from concurrent.futures import as_completed

        assert self.stop.is_set()
        first_error: Exception | None = None

        def harvest_one(future: "Future[BatchOutcome]") -> None:
            nonlocal first_error
            try:
                self._harvest_unless_hard_limit(future, futures)
            except Exception as e:
                if first_error is None:
                    first_error = e

        try:
            for future in futures:
                future.cancel()
            in_flight = [future for future in futures if not future.done()]
            if in_flight:
                spend(run.waiting_line(len(in_flight), interrupted=interrupted))
            for future in futures:
                if future.done() and not future.cancelled():
                    harvest_one(future)
            for future in as_completed(in_flight, timeout=self._seconds_to_hard_limit()):
                harvest_one(future)
        except KeyboardInterrupt:
            self.abandon(futures, status=130, message=None)
        except TimeoutError:
            self._hard_limit_passed(futures)
        return first_error

    def abandon(
        self, futures: Sequence["Future[BatchOutcome]"], *, status: int, message: str | None
    ) -> NoReturn:
        """Stops waiting and exits the process at once with `status`: a normal exit would
        wait for the (not daemonic) worker threads, which nothing can interrupt while a
        request is in flight.

        Before exiting it saves every finished batch not harvested yet to the pending
        directory (never to the store, whose lock could make it wait), prints `message` as
        an error or, without one, a note that the requests in flight were abandoned, and
        reports usage, counting what is still in flight as possibly billed. Ctrl-C is
        ignored meanwhile, so it cannot cut the save short.
        """
        with _sigint_ignored():
            try:
                in_flight = sum(1 for future in futures if not future.done())
                try:
                    self.save_for_later(futures)
                except Exception as e:
                    warn(f"could not save the finished batches while abandoning the run: {e!r}")
                if message is not None:
                    error(message)
                else:
                    spend(run.abandoned_line(in_flight, self.progress()))
                self.report_usage()
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                self.exit_process(status)

    def _harvest_unless_hard_limit(
        self, future: "Future[BatchOutcome]", futures: Sequence["Future[BatchOutcome]"]
    ) -> None:
        # as_completed yields every future already finished before checking its timeout.
        if self._seconds_to_hard_limit() <= 0:
            self._hard_limit_passed(futures)
        self.harvest(future)

    def _seconds_to_hard_limit(self) -> float:
        return max(0.0, self.hard_limit_at - self.clock())

    def _hard_limit_passed(self, futures: Sequence["Future[BatchOutcome]"]) -> NoReturn:
        in_flight = sum(1 for future in futures if not future.done())
        self.abandon(futures, status=1, message=run.hard_limit_message(in_flight, self.progress()))


# The process exit used when a run is abandoned (see _Waiting.abandon); tests replace it.
_exit_process: Callable[[int], NoReturn] = os._exit


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
    on a pool of budget.MAX_IN_FLIGHT threads."""
    # Imported only when there is something to send: the import costs every cached run
    # (and every other command) tens of milliseconds.
    from concurrent.futures import ThreadPoolExecutor

    assert threading.current_thread() is threading.main_thread()
    plan = run.plan_run(conn, store_path, pending, texts)
    client = (client_factory or default_client_factory)()

    stop = StopFlag()
    run_budget = RunBudget(plan.caps, stop)
    ctx = _RunContext(client=client, budget=run_budget, texts=texts)
    recorder = Recorder(
        conn, store_path, store, texts, release_by_key, stop, lambda: run.store_lock_wait_s(run_budget)
    )
    harvest = _Harvester(recorder)
    report_usage = run.UsageReport(run_budget)
    waiting = _Waiting(
        harvest=harvest,
        save_for_later=harvest.save_for_later,
        stop=stop,
        hard_limit_at=run.hard_limit_at(run_budget),
        clock=run_budget.clock,
        progress=lambda: recorder.progress(plan.entries),
        report_usage=report_usage,
        exit_process=_exit_process,
    )
    spend(budget.preview_line(plan.payload_bytes, plan.entries, plan.caps))
    try:
        # Workers take no batch until every batch is submitted: a Ctrl-C landing mid-submit
        # (possibly between a submit and its append) then cannot leave a request in flight
        # whose future `futures` is missing, since no request has been sent yet.
        all_submitted = threading.Event()
        pool = ThreadPoolExecutor(
            max_workers=min(budget.MAX_IN_FLIGHT, len(plan.batches)), initializer=all_submitted.wait
        )
        futures: "list[Future[BatchOutcome]]" = []
        try:
            for batch_keys in plan.batches:
                futures.append(pool.submit(_classify_batch, ctx, batch_keys))
            all_submitted.set()
            waiting.await_batches(futures)
        # Not SystemExit and the like: those end the process, and draining would only delay it.
        except (KeyboardInterrupt, Exception) as e:
            interrupted = isinstance(e, KeyboardInterrupt)
            stop.trip(
                AsioDocsError("interrupted" if interrupted else f"stopped by an unexpected error: {e!r}")
            )
            all_submitted.set()  # the workers find the stop flag set and send nothing
            harvest_error = waiting.stop_and_harvest(futures, interrupted=interrupted)
            if interrupted:
                note(run.interrupted_line(recorder.progress(plan.entries)))
            if harvest_error is not None:
                e.add_note(f"storing the batches in flight afterwards also failed: {harvest_error!r}")
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        assert run_budget.usage().in_flight_requests == 0, "every attempt is settled once its batch is done"
    finally:
        report_usage()
    run.finish(recorder, stop, plan.entries)
