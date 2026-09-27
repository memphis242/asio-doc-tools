"""The asyncio engine's own mechanics: its two-stage Ctrl-C (real SIGINTs), its hard limit,
and that cancelling requests in flight leaves no task or connection behind."""

import asyncio
import signal
import socket
import threading
import time

import anthropic
import classify_support as cs
import httpx
import pytest

from asio_doc_tools import classify
from asio_doc_tools.classify import budget
from asio_doc_tools.diag import AsioDocsError
from classify_support import Delay, Hook, Raise, WaitForCalls


def _run(classify_items, client, path):
    return classify.classify(classify_items, engine="asyncio", client_factory=lambda: client, store_path=path)


def _one_per_batch(monkeypatch, in_flight: int) -> None:
    monkeypatch.setattr(budget, "BATCH_SIZE", 1)
    monkeypatch.setattr(budget, "MAX_IN_FLIGHT", in_flight)


def _interrupt_twice() -> None:
    """A first Ctrl-C now, and a second one 50 ms later, while the run waits."""
    cs.interrupt_main_thread()
    asyncio.get_running_loop().call_later(0.05, cs.interrupt_main_thread)


def _schedule_interrupt() -> None:
    asyncio.get_running_loop().call_later(0.05, cs.interrupt_main_thread)


def _no_tasks_or_calls_left(client: cs.AsyncScriptedClient) -> None:
    assert client.messages.in_progress == 0
    assert (client.entered, client.closed) == (1, 1)  # the client was closed on the way out
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# ---------------------------------------------------------------------------
# Ctrl-C
# ---------------------------------------------------------------------------


def test_first_ctrl_c_waits_for_requests_in_flight_stores_them_and_sends_nothing_more(
    tmp_path, monkeypatch, capsys
) -> None:
    _one_per_batch(monkeypatch, 2)
    path = cs.store_path(tmp_path)
    client = cs.AsyncScriptedClient(
        [
            WaitForCalls(2, Hook(cs.interrupt_main_thread, Delay(0.1, cs.good_response()))),
            WaitForCalls(2, Delay(0.15, cs.good_response())),
        ]
        + [cs.good_response()] * 3
    )
    with pytest.raises(KeyboardInterrupt):
        _run(cs.items(5), client, path)
    assert len(client.calls) == 2
    assert len(classify.stored_keys(path)) == 2  # both in-flight results were kept
    assert client.messages.cancelled == 0
    err = capsys.readouterr().err
    assert "waiting for 2 in-flight requests" in err and "Ctrl-C again" in err
    assert "used 2 requests" in err and "interrupted: 2 of 5 entries" in err
    _no_tasks_or_calls_left(client)


def test_first_ctrl_c_drops_the_batches_waiting_for_a_slot(tmp_path, monkeypatch) -> None:
    _one_per_batch(monkeypatch, 1)
    path = cs.store_path(tmp_path)
    client = cs.AsyncScriptedClient(
        [Hook(cs.interrupt_main_thread, Delay(0.1, cs.good_response()))] + [cs.good_response()] * 2
    )
    with pytest.raises(KeyboardInterrupt):
        _run(cs.items(3), client, path)
    assert len(client.calls) == 1
    assert len(classify.stored_keys(path)) == 1
    _no_tasks_or_calls_left(client)


def test_first_ctrl_c_ends_a_retry_wait_at_once(tmp_path, capsys) -> None:
    client = cs.AsyncScriptedClient(
        [Hook(_schedule_interrupt, Raise(cs.status_error(429, {"retry-after": "30"})))]
    )
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run(cs.items(1), client, cs.store_path(tmp_path))
    assert time.monotonic() - started < 5.0
    assert len(client.calls) == 1  # the retry was never sent
    _no_tasks_or_calls_left(client)


def test_second_ctrl_c_cancels_the_requests_in_flight_and_keeps_what_finished(
    tmp_path, monkeypatch, capsys
) -> None:
    _one_per_batch(monkeypatch, 3)
    path = cs.store_path(tmp_path)
    threads_before = threading.active_count()
    client = cs.AsyncScriptedClient(
        [
            cs.good_response(),  # finishes before any Ctrl-C
            Hook(_interrupt_twice, Delay(10.0, cs.good_response())),
            Delay(10.0, cs.good_response()),
        ]
    )
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run(cs.items(3), client, path)
    assert time.monotonic() - started < 3.0  # not the 10 s the requests would take
    assert len(classify.stored_keys(path)) == 1
    assert client.messages.cancelled == 2
    err = capsys.readouterr().err
    assert "abandoned 2 in-flight requests" in err
    usage = next(line for line in err.splitlines() if "classification used" in line)
    assert "possibly billed by 2 requests" in usage
    _no_tasks_or_calls_left(client)
    assert threading.active_count() == threads_before  # no thread left behind either


# ---------------------------------------------------------------------------
# the hard limit
# ---------------------------------------------------------------------------


def test_the_hard_limit_cancels_the_requests_in_flight_and_keeps_what_finished(
    tmp_path, monkeypatch, capsys
) -> None:
    _one_per_batch(monkeypatch, 2)
    monkeypatch.setattr(budget, "hard_limit_s", lambda: 0.3)
    path = cs.store_path(tmp_path)
    client = cs.AsyncScriptedClient([cs.good_response(), Delay(10.0, cs.good_response())])
    started = time.monotonic()
    with pytest.raises(AsioDocsError, match="wall-clock limit") as excinfo:
        _run(cs.items(2), client, path)
    assert 0.3 <= time.monotonic() - started < 3.0
    assert "1 request still in flight" in str(excinfo.value) and "1 of 2 entries" in str(excinfo.value)
    assert len(classify.stored_keys(path)) == 1
    assert client.messages.cancelled == 1
    usage = next(line for line in capsys.readouterr().err.splitlines() if "classification used" in line)
    assert "possibly billed by 1 request" in usage
    _no_tasks_or_calls_left(client)


def test_the_hard_limit_during_a_split_keeps_the_half_that_finished(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(budget, "hard_limit_s", lambda: 0.3)
    path = cs.store_path(tmp_path)
    first, second = cs.item("Fixed the first entry."), cs.item("Fixed the second entry.")
    client = cs.AsyncScriptedClient(
        [cs.max_tokens_response(), cs.good_response(), Delay(10.0, cs.good_response())]
    )
    with pytest.raises(AsioDocsError, match="wall-clock limit"):
        _run([first, second], client, path)
    assert classify.stored_keys(path) == cs.keys_of([first])
    _no_tasks_or_calls_left(client)


class _SilentServer:
    """A local HTTP server that reads each request and never answers. Once `interrupt_after`
    requests have arrived it presses Ctrl-C twice (50 ms apart), and it notes when the
    client closes each connection."""

    def __init__(self, interrupt_after: int) -> None:
        self._sock = socket.create_server(("127.0.0.1", 0))
        self._sock.settimeout(0.05)
        self.port = self._sock.getsockname()[1]
        self._interrupt_after = interrupt_after
        self.requests = 0
        self.interrupted_at: float | None = None
        self.closed_at: list[float] = []
        self._lock = threading.Lock()
        self._running = True
        self._accepting = threading.Thread(target=self._accept, daemon=True)
        self._accepting.start()

    def _accept(self) -> None:
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            data = conn.recv(65536)  # the request; no reply ever comes
            with self._lock:
                self.requests += 1
                interrupt = self.requests == self._interrupt_after
            if interrupt:
                self.interrupted_at = time.monotonic()
                cs.interrupt_main_thread()
                time.sleep(0.05)
                cs.interrupt_main_thread()
            while data:
                data = conn.recv(65536)
        with self._lock:
            self.closed_at.append(time.monotonic())

    def close(self) -> None:
        self._running = False
        self._accepting.join(timeout=5.0)  # notices within 50 ms
        self._sock.close()


def test_cancelling_real_requests_in_flight_closes_their_connections_at_once(tmp_path, monkeypatch) -> None:
    _one_per_batch(monkeypatch, 2)
    server = _SilentServer(interrupt_after=2)
    try:

        def client_factory():
            return anthropic.AsyncAnthropic(
                api_key="test-key-not-used",
                base_url=f"http://127.0.0.1:{server.port}",
                max_retries=0,
                timeout=httpx.Timeout(budget.REQUEST_TIMEOUT_S, connect=budget.CONNECT_TIMEOUT_S),
            )

        with pytest.raises(KeyboardInterrupt):
            classify.classify(
                cs.items(2),
                engine="asyncio",
                client_factory=client_factory,
                store_path=cs.store_path(tmp_path),
            )
        returned = time.monotonic()
        deadline = returned + 2.0
        while len(server.closed_at) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.requests == 2
        assert len(server.closed_at) == 2  # both connections were closed by the client
        assert server.interrupted_at is not None
        assert max(server.closed_at) - server.interrupted_at < 2.0  # at the second Ctrl-C, not at a timeout
    finally:
        server.close()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# ---------------------------------------------------------------------------
# failures and signal handling
# ---------------------------------------------------------------------------


def test_every_paid_result_is_stored_before_the_first_task_failure_is_raised(tmp_path, monkeypatch) -> None:
    # On the loop, timers fire in deadline order: the failures come 0 and 50 ms in, the
    # billed reply 100 ms in, and all three are in flight together.
    _one_per_batch(monkeypatch, 3)
    path = cs.store_path(tmp_path)
    client = cs.AsyncScriptedClient(
        [
            WaitForCalls(3, Raise(RuntimeError("first failure"))),
            WaitForCalls(3, Delay(0.05, Raise(RuntimeError("second failure")))),
            WaitForCalls(3, Delay(0.1, cs.good_response())),
        ]
    )
    with pytest.raises(RuntimeError, match="first failure"):
        _run(cs.items(3), client, path)
    assert len(client.calls) == 3
    assert len(classify.stored_keys(path)) == 1


def test_the_sigint_handler_in_place_before_a_run_is_put_back_after_it(tmp_path) -> None:
    def custom_handler(signum, frame) -> None:
        pass

    previous = signal.signal(signal.SIGINT, custom_handler)
    try:
        _run(cs.items(1), cs.AsyncScriptedClient([cs.good_response()]), cs.store_path(tmp_path))
        assert signal.getsignal(signal.SIGINT) is custom_handler
        result, _ = cs.Engine("asyncio").run(
            cs.items(1, start=1), [cs.refusal_response()], cs.store_path(tmp_path)
        )
        assert isinstance(result, AsioDocsError)
        assert signal.getsignal(signal.SIGINT) is custom_handler
    finally:
        signal.signal(signal.SIGINT, previous)


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


class _InterruptedWhileClosing(cs.AsyncScriptedClient):
    async def __aexit__(self, *exc_info) -> None:
        cs.interrupt_main_thread()
        await asyncio.sleep(0.05)  # the Ctrl-C arrives while the client closes
        await super().__aexit__(*exc_info)


def test_a_ctrl_c_while_the_client_closes_cuts_nothing_short(tmp_path, capsys) -> None:
    client = _InterruptedWhileClosing([cs.good_response()])
    result = _run(cs.items(1), client, cs.store_path(tmp_path))
    assert len(result) == 1
    assert "used 1 request" in capsys.readouterr().err
    _no_tasks_or_calls_left(client)


class _Exited(BaseException):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def _fake_exit(status: int):
    raise _Exited(status)


@pytest.fixture
def short_teardown(monkeypatch):
    """A 0.3 s hard limit, and teardown graces short enough for a test."""
    from asio_doc_tools.classify import asyncio_engine

    monkeypatch.setattr(budget, "hard_limit_s", lambda: 0.3)
    monkeypatch.setattr(asyncio_engine, "_EXECUTOR_SHUTDOWN_GRACE_S", 0.3)
    monkeypatch.setattr(asyncio_engine, "_TEARDOWN_GRACE_S", 0.6)
    monkeypatch.setattr(asyncio_engine, "_exit_process", _fake_exit)
    return 0.3 + 0.6  # the most a stopped run may take, before slack


@pytest.fixture
def stuck() -> threading.Event:
    """Something that never returns until the test ends."""
    release = threading.Event()
    yield release
    release.set()


def test_a_dns_lookup_that_never_answers_does_not_hold_the_run(
    tmp_path, monkeypatch, short_teardown, stuck
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: stuck.wait())

    def resolve():
        return asyncio.get_running_loop().getaddrinfo("api.invalid", 443)

    client = cs.AsyncScriptedClient([Hook(resolve, cs.good_response())])
    started = time.monotonic()
    with pytest.raises(AsioDocsError, match="wall-clock limit"):
        _run(cs.items(1), client, cs.store_path(tmp_path))
    assert time.monotonic() - started < short_teardown + 1.0
    lookups = [t for t in threading.enumerate() if t.name == "asio-docs executor"]
    assert lookups and all(t.daemon for t in lookups)  # still stuck, but cannot hold the exit
    _no_tasks_or_calls_left(client)


def test_a_helper_job_that_never_returns_does_not_hold_the_run(tmp_path, short_teardown, stuck) -> None:
    def helper():  # as the SDK runs its platform detection: asyncio.to_thread
        return asyncio.to_thread(stuck.wait)

    client = cs.AsyncScriptedClient([Hook(helper, cs.good_response())])
    started = time.monotonic()
    with pytest.raises(AsioDocsError, match="wall-clock limit"):
        _run(cs.items(1), client, cs.store_path(tmp_path))
    assert time.monotonic() - started < short_teardown + 1.0
    _no_tasks_or_calls_left(client)


@pytest.mark.parametrize(("how", "status"), [("hard-limit", 1), ("second-ctrl-c", 130)])
def test_a_thread_the_run_started_that_never_ends_makes_it_exit_after_reporting(
    tmp_path, monkeypatch, capsys, short_teardown, stuck, how, status
) -> None:
    if how == "second-ctrl-c":
        monkeypatch.setattr(budget, "hard_limit_s", lambda: 30.0)

    def start_stuck_thread():  # a non-daemon thread, as a ThreadPoolExecutor's are
        threading.Thread(target=stuck.wait, name="stuck worker").start()
        if how == "second-ctrl-c":
            _interrupt_twice()

    path = cs.store_path(tmp_path)
    client = cs.AsyncScriptedClient(
        [cs.good_response(), Hook(start_stuck_thread, Delay(10.0, cs.good_response()))]
    )
    _one_per_batch(monkeypatch, 2)
    started = time.monotonic()
    with pytest.raises(_Exited) as excinfo:
        _run(cs.items(2), client, path)
    assert excinfo.value.status == status
    assert time.monotonic() - started < short_teardown + 1.5
    assert len(classify.stored_keys(path)) == 1  # stored before exiting
    err = capsys.readouterr().err
    assert "classification used 2 requests" in err  # reported before exiting
    assert "stuck worker did not end" in err
    if how == "hard-limit":
        assert "wall-clock limit" in err  # the run's error, printed since nothing else will


def test_the_process_exits_within_the_bound_while_a_dns_lookup_hangs(tmp_path) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    child = tmp_path / "child.py"
    child.write_text(
        """
import asyncio, socket, sys, threading, time
from pathlib import Path
from classify_support import AsyncScriptedClient, Hook, items
from asio_doc_tools import classify
from asio_doc_tools.classify import budget
from asio_doc_tools.diag import AsioDocsError

socket.getaddrinfo = lambda *args, **kwargs: threading.Event().wait()  # never answers
budget.hard_limit_s = lambda: 0.5

def resolve():
    return asyncio.get_running_loop().getaddrinfo("api.invalid", 443)

client = AsyncScriptedClient([Hook(resolve, None)])
print(time.monotonic(), flush=True)
try:
    classify.classify(items(1), engine="asyncio", client_factory=lambda: client, store_path=Path(sys.argv[1]))
except AsioDocsError as e:
    print(time.monotonic(), flush=True)
    print("stopped:", e, flush=True)
"""
    )
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}
    process = subprocess.run(
        [sys.executable, str(child), str(cs.store_path(tmp_path))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    ended = time.monotonic()
    assert process.returncode == 0, process.stderr
    started_at, returned_at, stopped = process.stdout.splitlines()
    assert stopped.startswith("stopped:") and "wall-clock limit" in stopped
    # classify() itself: the 0.5 s hard limit, then at most the 2 s executor grace.
    assert float(returned_at) - float(started_at) < 0.5 + 2.0 + 1.5
    # And the process exits at all: the stuck lookup is a daemon thread, which the
    # interpreter does not join (a non-daemon one would have held it until the timeout
    # above). How long exiting takes is the interpreter's, and grows under load.
    assert ended - float(returned_at) < 20.0
