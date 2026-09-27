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
