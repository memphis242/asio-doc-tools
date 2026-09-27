"""The threaded reference engine's own mechanics: its two-stage Ctrl-C, its hard limit
(abandon to the pending directory and exit the process), and how its main thread harvests."""

import signal
import sqlite3
import threading
import time
from concurrent.futures import Future

import classify_support as cs
import pytest

from asio_doc_tools import classify
from asio_doc_tools.classify import batch, budget, prompt, store
from asio_doc_tools.classify.threaded_reference import engine
from asio_doc_tools.diag import AsioDocsError
from classify_support import Delay, Hook, Raise


class _Exited(BaseException):
    """Raised by the fake exit function in place of ending the process (a BaseException,
    like SystemExit, so the run's own error handling lets it through)."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def _fake_exit(status: int):
    raise _Exited(status)


def _run(classify_items, client, path):
    return classify.classify(classify_items, engine="threads", client_factory=lambda: client, store_path=path)


def _one_per_batch(monkeypatch, workers: int) -> None:
    monkeypatch.setattr(budget, "BATCH_SIZE", 1)
    monkeypatch.setattr(budget, "MAX_IN_FLIGHT", workers)


def _two_concurrent_calls(first_result, second_result, *, second_delay_s: float = 0.05, on_both_started=None):
    """Two script items guaranteed to be inside create() at the same time: the first waits
    until the second has started, runs `on_both_started`, then produces `first_result`;
    the second produces `second_result` after `second_delay_s`."""
    second_started = threading.Event()

    def first():
        assert second_started.wait(timeout=5.0)
        if on_both_started is not None:
            on_both_started()
        return first_result

    def second():
        second_started.set()
        time.sleep(second_delay_s)
        return second_result

    return [first, second]


# ---------------------------------------------------------------------------
# a whole run
# ---------------------------------------------------------------------------


def test_ctrl_c_keeps_in_flight_results_and_sends_nothing_more(tmp_path, monkeypatch, capsys) -> None:
    _one_per_batch(monkeypatch, 2)
    path = cs.store_path(tmp_path)
    client = cs.ScriptedClient(
        _two_concurrent_calls(
            Delay(0.05, cs.good_response()),
            cs.good_response(),
            second_delay_s=0.1,
            on_both_started=cs.interrupt_main_thread,
        )
        + [cs.good_response()] * 3
    )
    with pytest.raises(KeyboardInterrupt):
        _run(cs.items(5), client, path)
    assert len(client.calls) == 2
    assert len(classify.stored_keys(path)) == 2  # both in-flight results were kept
    err = capsys.readouterr().err
    assert "Ctrl-C again" in err and "used 2 requests" in err
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_worker_interrupt_does_not_wait_for_queued_batches(tmp_path, monkeypatch) -> None:
    _one_per_batch(monkeypatch, 1)
    path = cs.store_path(tmp_path)
    four = cs.items(4)
    never_spent = 2.0  # those batches are never sent (asserted below)
    client = cs.ScriptedClient(
        [
            cs.good_response(),
            Raise(KeyboardInterrupt()),
            Delay(never_spent, cs.good_response()),
            Delay(never_spent, cs.good_response()),
        ]
    )
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run(four, client, path)
    assert time.monotonic() - started < never_spent
    assert len(client.calls) == 2
    assert classify.cache_key(four[0].entry) in classify.stored_keys(path)


def test_every_paid_result_is_stored_before_the_first_worker_failure_is_raised(tmp_path, monkeypatch) -> None:
    # Three requests in flight at once; each finishes only after the main thread has
    # harvested the one before, so "first" is well defined: two raise, the last one is a
    # billed reply.
    _one_per_batch(monkeypatch, 3)
    harvested = threading.Semaphore(0)
    real_harvest = engine._Harvester.__call__

    def counting_harvest(self, future):
        real_harvest(self, future)
        harvested.release()

    monkeypatch.setattr(engine._Harvester, "__call__", counting_harvest)
    all_started = threading.Barrier(3)
    turn = [threading.Event() for _ in range(3)]
    turn[0].set()

    def call(index: int, result):
        def run():
            all_started.wait(timeout=5.0)
            assert turn[index].wait(timeout=5.0)
            if index + 1 < len(turn):
                threading.Thread(
                    target=lambda: harvested.acquire(timeout=5.0) and turn[index + 1].set()
                ).start()
            return Raise(result) if isinstance(result, BaseException) else result

        return run

    results = [RuntimeError("first failure"), RuntimeError("second failure"), cs.good_response()]
    path = cs.store_path(tmp_path)
    client = cs.ScriptedClient([call(i, result) for i, result in enumerate(results)])
    with pytest.raises(RuntimeError, match="first failure"):
        _run(cs.items(3), client, path)
    assert len(client.calls) == 3
    assert len(classify.stored_keys(path)) == 1  # the billed reply that finished last


def test_the_hard_limit_abandons_a_request_that_never_ends_and_exits(tmp_path, monkeypatch, capsys) -> None:
    _one_per_batch(monkeypatch, 2)
    monkeypatch.setattr(budget, "hard_limit_s", lambda: 1.0)
    released = threading.Event()  # set by the exit, so the stuck worker thread can end

    def exit_after_releasing(status: int):
        released.set()
        raise _Exited(status)

    def dripping_reply():
        assert released.wait(timeout=10.0)
        return cs.good_response()

    monkeypatch.setattr(engine, "_exit_process", exit_after_releasing)
    path = cs.store_path(tmp_path)
    client = cs.ScriptedClient([cs.good_response(), dripping_reply])
    started = time.monotonic()
    with pytest.raises(_Exited) as excinfo:
        _run(cs.items(2), client, path)
    assert excinfo.value.status == 1
    assert 1.0 <= time.monotonic() - started < 5.0
    assert len(classify.stored_keys(path)) == 1  # the reply that finished was stored
    err = capsys.readouterr().err
    assert "wall-clock limit" in err and "1 request still in flight" in err
    usage = next(line for line in err.splitlines() if "classification used" in line)
    assert "possibly billed" in usage


# ---------------------------------------------------------------------------
# how the main thread waits
# ---------------------------------------------------------------------------


def _waiting(
    harvest, *, save_for_later=lambda futures: None, stop=None, report_usage=lambda: None
) -> engine._Waiting:
    if stop is None:
        stop = budget.StopFlag()
        stop.trip(AsioDocsError("interrupted"))
    return engine._Waiting(
        harvest=harvest,
        save_for_later=save_for_later,
        stop=stop,
        hard_limit_at=time.monotonic() + 10.0,
        clock=time.monotonic,
        progress=lambda: "progress",
        report_usage=report_usage,
        exit_process=_fake_exit,
    )


def test_first_ctrl_c_cancels_queued_batches_and_harvests_in_flight_ones(capsys) -> None:
    done: Future = Future()
    done.set_result(batch.BatchOutcome({}))
    in_flight: Future = Future()
    assert in_flight.set_running_or_notify_cancel()
    queued: Future = Future()
    harvested: list[Future] = []

    def harvest(future: Future) -> None:
        harvested.append(future)
        if future is done:  # stop_and_harvest has counted what is in flight by now
            threading.Timer(0.02, lambda: in_flight.set_result(batch.BatchOutcome({}))).start()

    error = _waiting(
        harvest, report_usage=lambda: pytest.fail("usage is reported by the caller on this path")
    ).stop_and_harvest([done, in_flight, queued], interrupted=True)

    assert error is None
    assert queued.cancelled()
    assert harvested == [done, in_flight]
    assert "Ctrl-C again" in capsys.readouterr().err


def test_a_failing_harvest_does_not_stop_the_others_and_the_first_error_is_kept() -> None:
    futures: list[Future] = []
    for _ in range(3):
        future: Future = Future()
        future.set_result(batch.BatchOutcome({}))
        futures.append(future)
    harvested: list[Future] = []

    def harvest(future: Future) -> None:
        harvested.append(future)
        if future is not futures[2]:
            raise ValueError(f"harvest {futures.index(future)} failed")

    error = _waiting(harvest).stop_and_harvest(futures, interrupted=False)

    assert harvested == futures
    assert str(error) == "harvest 0 failed"


def test_second_ctrl_c_abandons_in_flight_requests_and_exits_at_once(capsys) -> None:
    done: Future = Future()
    done.set_result(batch.BatchOutcome({}))
    in_flight: Future = Future()
    assert in_flight.set_running_or_notify_cancel()
    harvested: list[Future] = []
    saved: list[list[Future]] = []
    reported: list[bool] = []

    def harvest(future: Future) -> None:
        harvested.append(future)
        threading.Timer(0.02, cs.interrupt_main_thread).start()  # arrives while the drain waits

    try:
        with pytest.raises(_Exited) as excinfo:
            _waiting(
                harvest,
                save_for_later=lambda futures: saved.append(list(futures)),
                report_usage=lambda: reported.append(True),
            ).stop_and_harvest([done, in_flight], interrupted=True)
    finally:
        in_flight.set_result(batch.BatchOutcome({}))
    assert excinfo.value.status == 130
    assert reported == [True]
    assert harvested == [done]  # stored while waiting; the abandon then saves whatever is left
    assert saved == [[done, in_flight]]
    assert "abandoned 1 in-flight request" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# abandoning a run
# ---------------------------------------------------------------------------


def _finished_batch(entry_item: classify.ClassifyItem, category: str = "fixed") -> Future:
    future: Future = Future()
    future.set_result(
        batch.BatchOutcome(
            {classify.cache_key(entry_item.entry): prompt.RawResult(prompt.Category(category), False, "")}
        )
    )
    return future


def _harvester(path, conn, classify_items, stop) -> engine._Harvester:
    texts = {classify.cache_key(it.entry): it.entry.full_text() for it in classify_items}
    releases = {classify.cache_key(it.entry): it.release for it in classify_items}
    return engine._Harvester(
        store.Recorder(conn, path, {}, texts, releases, stop, lambda: store.STORE_LOCK_WAIT_S)
    )


def _hold_store_lock(path) -> sqlite3.Connection:
    other = sqlite3.connect(path, timeout=0.0, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    other.execute("PRAGMA user_version = 1")
    return other


def test_abandoning_a_run_never_waits_on_a_locked_store(tmp_path, capsys) -> None:
    path = cs.store_path(tmp_path)
    conn = store.connect(path)
    other = _hold_store_lock(path)
    entry_item = cs.item("Fixed a bug that finished just before the hard limit.")
    try:
        stop = budget.StopFlag()
        harvest = _harvester(path, conn, [entry_item], stop)
        waiting = _waiting(harvest, save_for_later=harvest.save_for_later, stop=stop)
        started = time.monotonic()
        with pytest.raises(_Exited) as excinfo:
            waiting.abandon([_finished_batch(entry_item)], status=1, message="abandoned by the test")
        assert time.monotonic() - started < 1.0  # the store's lock wait is 10 s
    finally:
        other.rollback()
        other.close()
        conn.close()
    assert excinfo.value.status == 1
    cs.import_pending(path)  # the next run
    assert classify.stored_keys(path) == cs.keys_of([entry_item])


def test_ctrl_c_during_abandon_does_not_cut_the_save_short(tmp_path, monkeypatch) -> None:
    path = cs.store_path(tmp_path)
    conn = store.connect(path)
    entry_item = cs.item("Fixed a bug saved while the user presses Ctrl-C.")
    real_write_all = store._write_all

    def interrupted_write(fd: int, data: bytes) -> None:
        cs.interrupt_main_thread()
        time.sleep(0.05)  # the signal arrives while the save is under way
        real_write_all(fd, data)

    monkeypatch.setattr(store, "_write_all", interrupted_write)
    stop = budget.StopFlag()
    harvest = _harvester(path, conn, [entry_item], stop)
    waiting = _waiting(harvest, save_for_later=harvest.save_for_later, stop=stop)
    try:
        with pytest.raises(_Exited) as excinfo:
            waiting.abandon([_finished_batch(entry_item)], status=130, message=None)
    finally:
        conn.close()
    assert excinfo.value.status == 130
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored
    monkeypatch.setattr(store, "_write_all", real_write_all)
    cs.import_pending(path)
    assert classify.stored_keys(path) == cs.keys_of([entry_item])


def test_the_reference_runs_max_in_flight_requests_at_once_on_as_many_threads(tmp_path, monkeypatch) -> None:
    _one_per_batch(monkeypatch, budget.MAX_IN_FLIGHT)
    seen: set[int] = set()
    lock = threading.Lock()

    def note_thread():
        with lock:
            seen.add(threading.get_ident())

    # The first 32 calls each wait until all 32 have started: 32 threads at once.
    script = [Hook(note_thread, cs.WaitForCalls(32, cs.good_response())) for _ in range(32)]
    script += [Hook(note_thread, cs.good_response()) for _ in range(8)]
    _run(cs.items(40), cs.ScriptedClient(script), cs.store_path(tmp_path))
    assert len(seen) == budget.MAX_IN_FLIGHT == 32
