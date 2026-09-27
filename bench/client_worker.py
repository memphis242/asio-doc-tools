"""Runs one benchmark cell against the mock server, in a process of its own.

The harness starts one of these per cell so that each cell's CPU time, peak
RSS, thread count, and open file descriptors belong to that cell alone.

Client modes:
  threads         sync `anthropic.Anthropic`, one shared client, requests submitted to a
                  `concurrent.futures.ThreadPoolExecutor(max_workers=C)` (the classifier's design)
  asyncio         `anthropic.AsyncAnthropic`, one shared client, N tasks in an
                  `asyncio.TaskGroup`, at most C inside an `asyncio.Semaphore(C)` at a time
  asyncio-uvloop  the asyncio mode on uvloop's (libuv-based) event loop instead of asyncio's own
  raw-asyncio     no SDK: C persistent keep-alive connections written with asyncio streams;
                  the floor the event loop itself can reach (and a load check of the mock server)

Scenarios:
  load    N requests at concurrency C; writes one JSON result (see `_load_result`)
  cancel  N requests at concurrency C, cancelled after --cancel-after-s by --cancel-method;
          writes one JSON event per line as they happen, so a hard exit still leaves a record

Per request it records four timestamps on the shared monotonic clock: t0 (the
caller starts the SDK call), the mock's `x-mock-recv` (full request arrived)
and `x-mock-send` (response written), and t1 (the SDK call returned a parsed
Message). `t1 - t0 - (send - recv)` is the client-side overhead: everything that
is not the simulated model latency.
"""

import argparse
import asyncio
import inspect
import itertools
import json
import math
import os
import resource
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO
from urllib.parse import urlsplit

from common import (
    HDR_DELAY,
    HDR_JITTER_SIGMA,
    HDR_LATENCY_MS,
    HDR_RECV,
    HDR_RESPONSE_BYTES,
    HDR_SEND,
    MOCK_API_KEY,
    MOCK_MODEL,
    proc_status,
)

MODES: Final = ("threads", "asyncio", "asyncio-uvloop", "raw-asyncio")
CANCEL_METHODS: Final = (
    "threads-shutdown-wait",
    "threads-shutdown-nowait",
    "threads-close-client",
    "threads-socket-shutdown",
    "threads-sigint",
    "threads-os-exit",
    "asyncio-cancel",
    "asyncio-timeout",
    "asyncio-sigint",
)
_WORDS: Final = "request payload text for the concurrency benchmark "
_MAX_TOKENS: Final = 1024


@dataclass(frozen=True, slots=True)
class CellConfig:
    base_url: str
    concurrency: int
    requests: int
    latency_ms: float
    jitter_sigma: float
    prompt_bytes: int
    response_bytes: int
    timeout_s: float

    def mock_headers(self) -> dict[str, str]:
        return {
            HDR_LATENCY_MS: f"{self.latency_ms:g}",
            HDR_JITTER_SIGMA: f"{self.jitter_sigma:g}",
            HDR_RESPONSE_BYTES: str(self.response_bytes),
        }

    def prompt(self) -> str:
        return (_WORDS * (self.prompt_bytes // len(_WORDS) + 1))[: self.prompt_bytes]

    def message_params(self) -> dict[str, Any]:
        return {
            "model": MOCK_MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": self.prompt()}],
        }


@dataclass(frozen=True, slots=True)
class Sample:
    """One request as the caller saw it; server timestamps are NaN when it failed."""

    t0: float
    t1: float
    server_recv: float
    server_send: float
    delay: float
    error: str | None

    @classmethod
    def ok(cls, t0: float, t1: float, headers: Mapping[str, str]) -> "Sample":
        return cls(t0, t1, float(headers[HDR_RECV]), float(headers[HDR_SEND]), float(headers[HDR_DELAY]), None)

    @classmethod
    def failed(cls, t0: float, t1: float, error: BaseException) -> "Sample":
        """Names the exception and, when it wraps one (the SDK's APIConnectionError does), its cause."""
        nan = float("nan")
        name = f"{type(error).__module__}.{type(error).__name__}"
        if error.__cause__ is not None:
            name += f"({type(error.__cause__).__module__}.{type(error.__cause__).__name__})"
        return cls(t0, t1, nan, nan, nan, name)


# ---------------------------------------------------------------------------
# Clients


def _sync_client(cfg: CellConfig) -> Any:
    import anthropic

    return anthropic.Anthropic(
        base_url=cfg.base_url,
        api_key=MOCK_API_KEY,
        max_retries=0,
        timeout=cfg.timeout_s,
        default_headers=cfg.mock_headers(),
    )


def _async_client(cfg: CellConfig) -> Any:
    import anthropic

    return anthropic.AsyncAnthropic(
        base_url=cfg.base_url,
        api_key=MOCK_API_KEY,
        max_retries=0,
        timeout=cfg.timeout_s,
        default_headers=cfg.mock_headers(),
    )


def _sync_request(client: Any, params: dict[str, Any]) -> Sample:
    t0 = time.monotonic()
    try:
        raw = client.messages.with_raw_response.create(**params)
        raw.parse()
    except Exception as e:
        return Sample.failed(t0, time.monotonic(), e)
    return Sample.ok(t0, time.monotonic(), raw.headers)


async def _async_request(client: Any, params: dict[str, Any]) -> Sample:
    t0 = time.monotonic()
    try:
        raw = await client.messages.with_raw_response.create(**params)
        # The 0.x SDK's raw response parses synchronously; later releases return an awaitable.
        parsed = raw.parse()
        if inspect.isawaitable(parsed):
            await parsed
    except Exception as e:
        return Sample.failed(t0, time.monotonic(), e)
    return Sample.ok(t0, time.monotonic(), raw.headers)


def _run_async(main: Callable[[], Any], mode: str) -> Any:
    if mode == "asyncio-uvloop":
        import uvloop

        return asyncio.run(main(), loop_factory=uvloop.new_event_loop)
    return asyncio.run(main())


# ---------------------------------------------------------------------------
# Load scenario


class Window:
    """The measured part of a cell: wall time, CPU time, and resident memory around the load."""

    def __init__(self) -> None:
        self.t_start = self.t_end = math.nan
        self.cpu_start = self.cpu_end = (0.0, 0.0)
        self.rss_before_kb = 0

    def begin(self) -> None:
        self.rss_before_kb = proc_status().get("VmRSS", 0)
        self.cpu_start = _cpu_times()
        self.t_start = time.monotonic()

    def end(self) -> None:
        self.t_end = time.monotonic()
        self.cpu_end = _cpu_times()


def _cpu_times() -> tuple[float, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


def _warm_up_sync(client: Any, params: dict[str, Any]) -> None:
    """One unmeasured request, so one-time lazy initialization in the SDK stays out of the cell."""
    client.messages.with_raw_response.create(**params, extra_headers={HDR_LATENCY_MS: "0"}).parse()


async def _warm_up_async(client: Any, params: dict[str, Any]) -> None:
    parsed = (await client.messages.with_raw_response.create(**params, extra_headers={HDR_LATENCY_MS: "0"})).parse()
    if inspect.isawaitable(parsed):
        await parsed


def _load_threads(cfg: CellConfig, window: Window) -> list[Sample]:
    client = _sync_client(cfg)
    params = cfg.message_params()
    try:
        _warm_up_sync(client, params)
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            window.begin()
            futures = [pool.submit(_sync_request, client, params) for _ in range(cfg.requests)]
            samples = [future.result() for future in futures]
            window.end()
        return samples
    finally:
        client.close()


def _load_asyncio(cfg: CellConfig, window: Window, mode: str) -> list[Sample]:
    params = cfg.message_params()

    async def main() -> list[Sample]:
        async with _async_client(cfg) as client:
            await _warm_up_async(client, params)
            gate = asyncio.Semaphore(cfg.concurrency)

            async def one() -> Sample:
                async with gate:
                    return await _async_request(client, params)

            window.begin()
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(one()) for _ in range(cfg.requests)]
            window.end()
            return [task.result() for task in tasks]

    return _run_async(main, mode)


def _raw_request_bytes(cfg: CellConfig) -> tuple[str, int, bytes]:
    parts = urlsplit(cfg.base_url)
    assert parts.hostname is not None and parts.port is not None
    body = json.dumps(cfg.message_params()).encode()
    head_lines = [
        "POST /v1/messages HTTP/1.1",
        f"host: {parts.hostname}:{parts.port}",
        "content-type: application/json",
        "accept: application/json",
        f"content-length: {len(body)}",
        *(f"{name}: {value}" for name, value in cfg.mock_headers().items()),
    ]
    return parts.hostname, parts.port, ("\r\n".join(head_lines) + "\r\n\r\n").encode() + body


async def _raw_exchange(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, request: bytes
) -> dict[str, str]:
    writer.write(request)
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    if not lines[0].startswith("HTTP/1.1 200"):
        raise ConnectionError(f"unexpected status line {lines[0]!r}")
    headers = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    body = await reader.readexactly(int(headers["content-length"]))
    json.loads(body)
    return headers


def _load_raw(cfg: CellConfig, window: Window) -> list[Sample]:
    host, port, request = _raw_request_bytes(cfg)
    order = itertools.count()

    async def worker() -> list[Sample]:
        samples: list[Sample] = []
        reader, writer = await asyncio.open_connection(host, port)
        try:
            while next(order) < cfg.requests:
                t0 = time.monotonic()
                try:
                    headers = await _raw_exchange(reader, writer, request)
                except (OSError, asyncio.IncompleteReadError, ValueError, KeyError) as e:
                    samples.append(Sample.failed(t0, time.monotonic(), e))
                    break
                samples.append(Sample.ok(t0, time.monotonic(), headers))
        finally:
            writer.close()
        return samples

    async def main() -> list[Sample]:
        # No warm-up: this path has no lazy initialization, and its connections are
        # opened inside the window like the SDK modes' are.
        window.begin()
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(worker()) for _ in range(min(cfg.concurrency, cfg.requests))]
        window.end()
        return [sample for task in tasks for sample in task.result()]

    return asyncio.run(main())


def _load_result(cfg: CellConfig, mode: str) -> dict[str, Any]:
    runners: dict[str, Callable[[CellConfig, Window], list[Sample]]] = {
        "threads": _load_threads,
        "asyncio": lambda c, w: _load_asyncio(c, w, "asyncio"),
        "asyncio-uvloop": lambda c, w: _load_asyncio(c, w, "asyncio-uvloop"),
        "raw-asyncio": _load_raw,
    }
    window = Window()
    samples = runners[mode](cfg, window)
    status = proc_status()
    assert len(samples) == cfg.requests, (len(samples), cfg.requests)
    assert window.t_end >= window.t_start
    return {
        "mode": mode,
        "config": {
            "concurrency": cfg.concurrency,
            "requests": cfg.requests,
            "latency_ms": cfg.latency_ms,
            "jitter_sigma": cfg.jitter_sigma,
            "prompt_bytes": cfg.prompt_bytes,
            "response_bytes": cfg.response_bytes,
        },
        "t_start": window.t_start,
        "t_end": window.t_end,
        "wall_s": window.t_end - window.t_start,
        "cpu_user_s": window.cpu_end[0] - window.cpu_start[0],
        "cpu_sys_s": window.cpu_end[1] - window.cpu_start[1],
        "rss_before_kb": window.rss_before_kb,
        "rss_after_kb": status.get("VmRSS", 0),
        "rss_peak_kb": status.get("VmHWM", 0),
        "samples": {
            "t0": [s.t0 for s in samples],
            "t1": [s.t1 for s in samples],
            "server_recv": [s.server_recv for s in samples],
            "server_send": [s.server_send for s in samples],
            "delay": [s.delay for s in samples],
            "error": [s.error for s in samples],
        },
    }


# ---------------------------------------------------------------------------
# Cancellation scenario


class EventLog:
    """Appends one JSON object per line and flushes at once, so os._exit loses nothing."""

    def __init__(self, path: Path) -> None:
        self._file: TextIO = path.open("a", buffering=1)
        self._lock = threading.Lock()

    def emit(self, event: str, *, at: float | None = None, **fields: Any) -> None:
        """Records `event` as happening now, or at the monotonic time `at`."""
        record = {"event": event, "t": time.monotonic() if at is None else at, **fields}
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()


def _announce_load_start() -> None:
    """Tells the harness (reading stdout) that the requests are being submitted now."""
    print(f"LOAD_START {time.monotonic():.6f}", flush=True)


def _shutdown_client_sockets(client: Any) -> int:
    """shutdown(SHUT_RDWR) on every pooled socket, which wakes a thread blocked in recv().

    Reaches into httpx/httpcore internals (the SDK exposes no such control), so this is
    a demonstration of the mechanism, not something to ship.
    """
    count = 0
    pool = client._client._transport._pool
    for connection in list(pool.connections):
        inner = getattr(connection, "_connection", None)
        stream = getattr(inner, "_network_stream", None)
        sock = getattr(stream, "_sock", None)
        if isinstance(sock, socket.socket):
            try:
                sock.shutdown(socket.SHUT_RDWR)
                count += 1
            except OSError:
                pass
    return count


def _cancel_threads(cfg: CellConfig, method: str, cancel_after_s: float, log: EventLog) -> None:
    client = _sync_client(cfg)
    params = cfg.message_params()
    finished = {"ok": 0, "failed": 0}
    finished_lock = threading.Lock()

    def one() -> Sample:
        sample = _sync_request(client, params)
        with finished_lock:
            finished["ok" if sample.error is None else "failed"] += 1
            total = finished["ok"] + finished["failed"]
        log.emit("request_done", ok=sample.error is None, error=sample.error, finished=total)
        return sample

    pool = ThreadPoolExecutor(max_workers=cfg.concurrency)
    _announce_load_start()
    log.emit("load_start")
    futures: list[Future[Sample]] = [pool.submit(one) for _ in range(cfg.requests)]

    if method == "threads-sigint":
        # The classifier's shape: the caller waits in as_completed, a Ctrl-C (SIGINT from the
        # harness) raises KeyboardInterrupt there, and the handler cancels queued work without
        # waiting for the requests in flight, then re-raises.
        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            log.emit("cancel_requested", method=method)
            pool.shutdown(wait=False, cancel_futures=True)
            log.emit("regain", cancelled=sum(f.cancelled() for f in futures))
            _register_exit_log(log)
            raise
        return

    time.sleep(cancel_after_s)
    log.emit("cancel_requested", method=method)
    match method:
        case "threads-shutdown-wait":
            pool.shutdown(wait=True, cancel_futures=True)
        case "threads-shutdown-nowait":
            pool.shutdown(wait=False, cancel_futures=True)
        case "threads-close-client":
            pool.shutdown(wait=False, cancel_futures=True)
            client.close()
        case "threads-socket-shutdown":
            pool.shutdown(wait=False, cancel_futures=True)
            log.emit("sockets_shut_down", count=_shutdown_client_sockets(client))
        case "threads-os-exit":
            log.emit("regain", cancelled=0)
            os._exit(130)
        case _:
            raise AssertionError(f"not a threads cancel method: {method}")
    log.emit("regain", cancelled=sum(f.cancelled() for f in futures))
    _register_exit_log(log)


def _register_exit_log(log: EventLog) -> None:
    import atexit

    # atexit handlers run after the interpreter has joined the executor's worker threads,
    # so this marks when a normal exit could actually complete.
    atexit.register(lambda: log.emit("atexit", threads=threading.active_count()))


def _cancel_asyncio(cfg: CellConfig, method: str, cancel_after_s: float, log: EventLog) -> None:
    params = cfg.message_params()

    async def main() -> None:
        async with _async_client(cfg) as client:
            gate = asyncio.Semaphore(cfg.concurrency)
            finished = 0

            async def one() -> Sample:
                nonlocal finished
                async with gate:
                    sample = await _async_request(client, params)
                finished += 1
                log.emit("request_done", ok=sample.error is None, error=sample.error, finished=finished)
                return sample

            _announce_load_start()
            log.emit("load_start")
            match method:
                case "asyncio-cancel":
                    tasks = [asyncio.create_task(one()) for _ in range(cfg.requests)]
                    await asyncio.sleep(cancel_after_s)
                    log.emit("cancel_requested", method=method)
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    log.emit("regain", cancelled=sum(t.cancelled() for t in tasks))
                case "asyncio-timeout":
                    # The deadline itself is the cancellation request. asyncio's loop clock is
                    # time.monotonic(), the clock every event is stamped with.
                    deadline = asyncio.get_running_loop().time() + cancel_after_s
                    try:
                        async with asyncio.timeout_at(deadline):
                            async with asyncio.TaskGroup() as group:
                                for _ in range(cfg.requests):
                                    group.create_task(one())
                    except TimeoutError:
                        log.emit("cancel_requested", at=deadline, method=method)
                        log.emit("regain", cancelled=cfg.requests - finished)
                case "asyncio-sigint":
                    try:
                        async with asyncio.TaskGroup() as group:
                            for _ in range(cfg.requests):
                                group.create_task(one())
                    except asyncio.CancelledError:
                        log.emit("regain", cancelled=cfg.requests - finished)
                        raise
                case _:
                    raise AssertionError(f"not an asyncio cancel method: {method}")
        log.emit("client_closed")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.emit("keyboard_interrupt")
    log.emit("run_returned")


def _cancel(cfg: CellConfig, method: str, cancel_after_s: float, out: Path) -> None:
    log = EventLog(out)
    log.emit("start", method=method, concurrency=cfg.concurrency, requests=cfg.requests)
    if method.startswith("threads-"):
        _cancel_threads(cfg, method, cancel_after_s, log)
    else:
        _cancel_asyncio(cfg, method, cancel_after_s, log)
    log.emit("main_returned")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("load", "cancel"), required=True)
    parser.add_argument("--mode", choices=MODES, default="threads", help="client mode (load scenario)")
    parser.add_argument("--cancel-method", choices=CANCEL_METHODS, help="cancel scenario only")
    parser.add_argument("--cancel-after-s", type=float, default=0.5)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--latency-ms", type=float, default=1000.0)
    parser.add_argument("--jitter-sigma", type=float, default=0.0)
    parser.add_argument("--prompt-bytes", type=int, default=32)
    parser.add_argument("--response-bytes", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.concurrency < 1 or args.requests < 1:
        parser.error("--concurrency and --requests must be at least 1")
    if args.scenario == "cancel" and args.cancel_method is None:
        parser.error("--scenario cancel needs --cancel-method")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    # A SIGINT from the harness must reach Python's default handler (KeyboardInterrupt),
    # even when the harness itself was started with SIGINT ignored.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    cfg = CellConfig(
        base_url=args.base_url,
        concurrency=args.concurrency,
        requests=args.requests,
        latency_ms=args.latency_ms,
        jitter_sigma=args.jitter_sigma,
        prompt_bytes=args.prompt_bytes,
        response_bytes=args.response_bytes,
        timeout_s=args.timeout_s,
    )
    # Import the SDK before any measurement window: it costs about a second of CPU.
    import anthropic  # noqa: F401

    if args.scenario == "load":
        result = _load_result(cfg, args.mode)
        args.out.write_text(json.dumps(result))
    else:
        _cancel(cfg, args.cancel_method, args.cancel_after_s, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
