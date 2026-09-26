"""A local HTTP/1.1 mock of the Claude Messages API, for client-side concurrency benchmarks.

It implements just enough of `POST /v1/messages` for the Anthropic Python SDK to
parse a reply: a valid non-streaming Messages JSON body with usage. The reply is
delayed by a simulated model latency, fixed or log-normally jittered, which each
request selects through `x-mock-*` headers (see common.py).

The server is built for high concurrency on one thread: an `asyncio.Protocol`
per connection (no coroutine per request) and a loop timer (`call_later`) per
pending reply, so a thousand sleeping requests cost a thousand timer-heap
entries and nothing else. Because the transport keeps reading while a reply is
pending, a client that closes its connection mid-request is noticed at once
(`connection_lost`), which the cancellation scenario relies on.

Control endpoints:
  GET  /_mock/stats[?close_times=1]  counters since the last reset, as JSON
  POST /_mock/reset                  zero the counters and reseed the jitter RNG

Run: python3 mock_server.py [--port 0] [--loop auto|asyncio|uvloop]
Prints "LISTENING <port> <loop>" on stdout once it accepts connections.
"""

import argparse
import asyncio
import collections
import json
import math
import random
import resource
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Final, cast

from common import (
    HDR_DELAY,
    HDR_JITTER_SIGMA,
    HDR_LATENCY_MS,
    HDR_RECV,
    HDR_RESPONSE_BYTES,
    HDR_SEND,
    Summary,
)

_MAX_HEADER_BYTES: Final = 64 * 1024
_MAX_BODY_BYTES: Final = 64 * 1024 * 1024
_MAX_LATENCY_S: Final = 600.0
_MAX_RESPONSE_BYTES: Final = 16 * 1024 * 1024
_LAG_PROBE_INTERVAL_S: Final = 0.005
_JITTER_SEED: Final = 20260926
_FILLER: Final = "The quick brown fox jumps over the lazy dog. "


@dataclass(slots=True)
class Stats:
    """Counters since the last reset. Only touched from the event loop thread."""

    reset_at: float = field(default_factory=time.monotonic)
    cpu_at_reset: float = field(default_factory=lambda: _cpu_seconds())
    connections_accepted: int = 0
    open_connections: int = 0
    peak_open_connections: int = 0
    # Connections that carried at least one /v1/messages request (not the harness's
    # /_mock/* control connections); close_times covers only these.
    api_connections: int = 0
    open_api_connections: int = 0
    requests: int = 0
    responses: int = 0
    inflight: int = 0
    peak_inflight: int = 0
    cancelled_inflight: int = 0  # connection closed by the client while its reply was pending
    bad_requests: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    timer_lateness_s: list[float] = field(default_factory=list)
    loop_lag_s: collections.deque[float] = field(default_factory=lambda: collections.deque(maxlen=400_000))
    close_times: list[float] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)

    def to_json(self, include_close_times: bool) -> dict[str, object]:
        now = time.monotonic()
        wall = now - self.reset_at
        cpu = _cpu_seconds() - self.cpu_at_reset
        data: dict[str, object] = {
            "now": now,
            "wall_s": wall,
            "cpu_s": cpu,
            "cpu_util": cpu / wall if wall > 0 else math.nan,
            "connections_accepted": self.connections_accepted,
            "open_connections": self.open_connections,
            "peak_open_connections": self.peak_open_connections,
            "api_connections": self.api_connections,
            "open_api_connections": self.open_api_connections,
            "requests": self.requests,
            "responses": self.responses,
            "inflight": self.inflight,
            "peak_inflight": self.peak_inflight,
            "cancelled_inflight": self.cancelled_inflight,
            "bad_requests": self.bad_requests,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "timer_lateness_s": Summary.of(self.timer_lateness_s).to_json(),
            "loop_lag_s": Summary.of(list(self.loop_lag_s)).to_json(),
        }
        if include_close_times:
            data["close_times"] = list(self.close_times)
            data["response_times"] = list(self.response_times)
        return data


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


@dataclass(frozen=True, slots=True)
class ReplyShape:
    latency_s: float
    jitter_sigma: float
    response_bytes: int

    def delay(self, rng: random.Random) -> float:
        if self.jitter_sigma <= 0.0:
            return self.latency_s
        # Log-normal around the configured median: a long right tail, like real API latency.
        return min(_MAX_LATENCY_S, self.latency_s * math.exp(rng.gauss(0.0, self.jitter_sigma)))


class BadRequest(Exception):
    pass


class MockServer:
    def __init__(self, default_latency_s: float) -> None:
        self.default_latency_s: Final = default_latency_s
        self.stats = Stats()
        self.rng = random.Random(_JITTER_SEED)
        self._texts: dict[int, str] = {}
        self._message_ids = 0

    def reset(self) -> None:
        self.stats = Stats()
        self.rng = random.Random(_JITTER_SEED)

    def reply_shape(self, headers: dict[str, str]) -> ReplyShape:
        try:
            latency_ms = float(headers.get(HDR_LATENCY_MS, self.default_latency_s * 1000.0))
            sigma = float(headers.get(HDR_JITTER_SIGMA, "0"))
            response_bytes = int(headers.get(HDR_RESPONSE_BYTES, "2"))
        except ValueError as e:
            raise BadRequest(f"malformed x-mock header: {e}") from e
        if not (0.0 <= latency_ms <= _MAX_LATENCY_S * 1000.0 and 0.0 <= sigma <= 3.0):
            raise BadRequest("x-mock latency or jitter out of range")
        if not 1 <= response_bytes <= _MAX_RESPONSE_BYTES:
            raise BadRequest("x-mock-response-bytes out of range")
        return ReplyShape(latency_ms / 1000.0, sigma, response_bytes)

    def reply_text(self, size: int) -> str:
        text = self._texts.get(size)
        if text is None:
            text = "OK" if size <= 2 else (_FILLER * (size // len(_FILLER) + 1))[:size]
            self._texts[size] = text
        return text

    def next_message_id(self) -> str:
        self._message_ids += 1
        return f"msg_mock_{self._message_ids:010d}"


def _response(status: str, body: bytes, extra_headers: list[tuple[str, str]], close: bool) -> bytes:
    lines = [f"HTTP/1.1 {status}", "content-type: application/json", f"content-length: {len(body)}"]
    lines.extend(f"{name}: {value}" for name, value in extra_headers)
    if close:
        lines.append("connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


class Connection(asyncio.Protocol):
    def __init__(self, server: MockServer) -> None:
        self._server = server
        self._transport: asyncio.Transport | None = None
        self._buffer = bytearray()
        self._pending: asyncio.TimerHandle | None = None
        self._close_after_reply = False
        self._carried_api_request = False

    # -- asyncio.Protocol callbacks

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        # uvloop's transports implement asyncio.Transport without subclassing it.
        self._transport = cast(asyncio.Transport, transport)
        stats = self._server.stats
        stats.connections_accepted += 1
        stats.open_connections += 1
        stats.peak_open_connections = max(stats.peak_open_connections, stats.open_connections)

    def data_received(self, data: bytes) -> None:
        self._server.stats.bytes_in += len(data)
        self._buffer += data
        self._process_buffer()

    def connection_lost(self, exc: Exception | None) -> None:
        stats = self._server.stats
        stats.open_connections -= 1
        if self._carried_api_request:
            stats.open_api_connections -= 1
            stats.close_times.append(time.monotonic())
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
            stats.inflight -= 1
            stats.cancelled_inflight += 1
        self._transport = None

    # -- request handling

    def _process_buffer(self) -> None:
        # One request at a time: a pipelined request waits in the buffer until the
        # pending reply has been written.
        while self._pending is None and self._transport is not None:
            header_end = self._buffer.find(b"\r\n\r\n")
            if header_end < 0:
                if len(self._buffer) > _MAX_HEADER_BYTES:
                    self._reject("431 Request Header Fields Too Large")
                return
            try:
                method, target, headers = _parse_head(bytes(self._buffer[:header_end]))
                length = _content_length(headers)
            except BadRequest as e:
                self._reject("400 Bad Request", str(e))
                return
            total = header_end + 4 + length
            if len(self._buffer) < total:
                return
            body = bytes(self._buffer[header_end + 4 : total])
            del self._buffer[:total]
            self._close_after_reply = headers.get("connection", "").lower() == "close"
            self._dispatch(method, target, headers, body)

    def _dispatch(self, method: str, target: str, headers: dict[str, str], body: bytes) -> None:
        path, _, query = target.partition("?")
        match (method, path):
            case ("POST", "/v1/messages"):
                self._start_message(headers, body)
            case ("GET", "/_mock/stats"):
                include = "close_times=1" in query.split("&")
                self._send_json("200 OK", self._server.stats.to_json(include))
            case ("POST", "/_mock/reset"):
                self._server.reset()
                self._send_json("200 OK", {"reset": True})
            case _:
                self._send_json("404 Not Found", _api_error("not_found_error", f"no route {method} {path}"))

    def _start_message(self, headers: dict[str, str], body: bytes) -> None:
        t_recv = time.monotonic()
        try:
            shape = self._server.reply_shape(headers)
        except BadRequest as e:
            self._reject("400 Bad Request", str(e))
            return
        stats = self._server.stats
        if not self._carried_api_request:
            self._carried_api_request = True
            stats.api_connections += 1
            stats.open_api_connections += 1
        stats.requests += 1
        stats.inflight += 1
        stats.peak_inflight = max(stats.peak_inflight, stats.inflight)
        delay = shape.delay(self._server.rng)
        loop = asyncio.get_running_loop()
        self._pending = loop.call_later(delay, self._finish_message, t_recv, delay, len(body), shape.response_bytes)

    def _finish_message(self, t_recv: float, delay: float, request_bytes: int, response_bytes: int) -> None:
        stats = self._server.stats
        t_fire = time.monotonic()
        self._pending = None
        stats.inflight -= 1
        stats.timer_lateness_s.append(t_fire - (t_recv + delay))
        if self._transport is None:
            return
        text = self._server.reply_text(response_bytes)
        message = {
            "id": self._server.next_message_id(),
            "type": "message",
            "role": "assistant",
            "model": "mock-model",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": max(1, request_bytes // 4), "output_tokens": max(1, len(text) // 4)},
        }
        body = json.dumps(message, separators=(",", ":")).encode()
        t_send = time.monotonic()
        extra = [
            ("request-id", f"req_mock_{stats.responses:010d}"),
            (HDR_RECV, f"{t_recv:.6f}"),
            (HDR_SEND, f"{t_send:.6f}"),
            (HDR_DELAY, f"{delay:.6f}"),
        ]
        self._write(_response("200 OK", body, extra, self._close_after_reply))
        stats.responses += 1
        stats.response_times.append(t_send)
        if self._close_after_reply:
            self._close()
        else:
            self._process_buffer()

    # -- output

    def _send_json(self, status: str, data: object) -> None:
        self._write(_response(status, json.dumps(data).encode(), [], self._close_after_reply))
        if self._close_after_reply:
            self._close()

    def _reject(self, status: str, message: str = "") -> None:
        self._server.stats.bad_requests += 1
        body = json.dumps(_api_error("invalid_request_error", message or status)).encode()
        self._write(_response(status, body, [], close=True))
        self._close()

    def _write(self, data: bytes) -> None:
        if self._transport is not None:
            self._server.stats.bytes_out += len(data)
            self._transport.write(data)

    def _close(self) -> None:
        if self._transport is not None:
            self._transport.close()


def _api_error(kind: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def _parse_head(head: bytes) -> tuple[str, str, dict[str, str]]:
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise BadRequest(f"malformed request line {lines[0]!r}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if not sep:
            raise BadRequest(f"malformed header line {line!r}")
        headers[name.strip().lower()] = value.strip()
    return parts[0], parts[1], headers


def _content_length(headers: dict[str, str]) -> int:
    if "transfer-encoding" in headers:
        raise BadRequest("chunked request bodies are not supported")
    try:
        length = int(headers.get("content-length", "0"))
    except ValueError as e:
        raise BadRequest("malformed content-length") from e
    if not 0 <= length <= _MAX_BODY_BYTES:
        raise BadRequest("content-length out of range")
    return length


async def _probe_loop_lag(server: MockServer) -> None:
    """Records how late a short sleep wakes up: the event loop's scheduling delay."""
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(_LAG_PROBE_INTERVAL_S)
        server.stats.loop_lag_s.append(loop.time() - start - _LAG_PROBE_INTERVAL_S)


async def serve(host: str, port: int, default_latency_s: float, loop_name: str) -> None:
    server = MockServer(default_latency_s)
    loop = asyncio.get_running_loop()
    listener = await loop.create_server(
        lambda: Connection(server), host, port, backlog=4096, reuse_address=True
    )
    bound_port = listener.sockets[0].getsockname()[1]
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    probe = asyncio.create_task(_probe_loop_lag(server))
    print(f"LISTENING {bound_port} {loop_name}", flush=True)
    async with listener:
        await stop.wait()
    probe.cancel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port (printed on stdout)")
    parser.add_argument("--latency-ms", type=float, default=1000.0, help="default when a request sets none")
    parser.add_argument("--loop", choices=("auto", "asyncio", "uvloop"), default="auto")
    args = parser.parse_args(argv)

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    # uvloop's add_signal_handler calls the deprecated asyncio.iscoroutinefunction.
    warnings.filterwarnings("ignore", message=".*iscoroutinefunction.*", category=DeprecationWarning)
    loop_name = "asyncio"
    loop_factory = None
    if args.loop in ("auto", "uvloop"):
        try:
            import uvloop
        except ImportError:
            if args.loop == "uvloop":
                print("mock_server: uvloop is not installed", file=sys.stderr)
                return 2
        else:
            loop_name = "uvloop"
            loop_factory = uvloop.new_event_loop
    asyncio.run(serve(args.host, args.port, args.latency_ms / 1000.0, loop_name), loop_factory=loop_factory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
