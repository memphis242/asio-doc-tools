"""The local (mock API) scenarios: the concurrency sweeps, request sizes, and cancellation.

Every cell runs in a fresh `client_worker.py` process against one long-lived
mock server process. While a cell runs, this process samples the worker's
/proc entry (thread count, open file descriptors) from the outside, so the
measurement adds no threads or work to the process being measured.
"""

import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from client_worker import CANCEL_METHODS
from common import BENCH_DIR, BenchError, open_fd_count, proc_status, write_json

_WORKER: Final = BENCH_DIR / "client_worker.py"
_SERVER: Final = BENCH_DIR / "mock_server.py"
_SAMPLE_INTERVAL_S: Final = 0.01
_FD_SAMPLE_EVERY: Final = 5  # sample open fds on every 5th tick (listing /proc/<pid>/fd costs more)
_CELL_TIMEOUT_S: Final = 600.0
_SERVER_START_TIMEOUT_S: Final = 10.0

CONCURRENCY_LEVELS: Final = (1, 4, 8, 16, 64, 256, 1024)
# N = max(16, 4C), capped so the slowest cell (C=1024, where the SDK client is CPU-bound)
# stays near a minute.
_MAX_REQUESTS: Final = 2048
CANCEL_CONCURRENCY: Final = 64
CANCEL_REQUESTS: Final = 128
CANCEL_LATENCY_MS: Final = 10_000.0
CANCEL_AFTER_S: Final = 0.5
_SIGINT_METHODS: Final = frozenset({"threads-sigint", "asyncio-sigint"})


@dataclass(frozen=True, slots=True)
class Size:
    name: str
    prompt_bytes: int
    response_bytes: int


SIZES: Final = (
    Size("tiny", 32, 2),
    Size("classifier-like", 6 * 1024, 4 * 1024),
    Size("large", 256 * 1024, 64 * 1024),
)
DEFAULT_SIZE: Final = SIZES[0]


@dataclass(frozen=True, slots=True)
class LoadCell:
    scenario: str
    mode: str
    concurrency: int
    requests: int
    latency_ms: float
    jitter_sigma: float
    size: Size
    repeat: int

    @property
    def cell_id(self) -> str:
        return f"{self.scenario}-{self.size.name}-{self.mode}-c{self.concurrency}-r{self.repeat}"


@dataclass(frozen=True, slots=True)
class SweepSpec:
    scenario: str
    latency_ms: float
    jitter_sigma: float
    modes: tuple[str, ...]
    concurrency: tuple[int, ...]
    sizes: tuple[Size, ...] = (DEFAULT_SIZE,)
    max_requests: int = _MAX_REQUESTS


SWEEPS: Final = {
    "sweep-1s": SweepSpec("sweep-1s", 1000.0, 0.0, ("threads", "asyncio", "raw-asyncio"), CONCURRENCY_LEVELS),
    "sweep-1s-jitter": SweepSpec("sweep-1s-jitter", 1000.0, 0.5, ("threads", "asyncio"), CONCURRENCY_LEVELS),
    "sweep-50ms": SweepSpec(
        "sweep-50ms", 50.0, 0.0, ("threads", "asyncio", "asyncio-uvloop", "raw-asyncio"), CONCURRENCY_LEVELS
    ),
    "sizes": SweepSpec("sizes", 200.0, 0.0, ("threads", "asyncio"), (16, 256), SIZES, max_requests=1024),
}
QUICK_CONCURRENCY: Final = (1, 16, 256)


def build_load_cells(spec: SweepSpec, repeats: int, quick: bool) -> list[LoadCell]:
    """Repeats are the outer loop and the mode order rotates per repeat, so slow drift in
    machine load spreads across modes instead of favoring whichever always runs first."""
    levels = tuple(c for c in spec.concurrency if c in QUICK_CONCURRENCY) if quick else spec.concurrency
    cells = []
    for repeat in range(repeats):
        rotation = repeat % len(spec.modes)
        modes = spec.modes[rotation:] + spec.modes[:rotation]
        for size in spec.sizes:
            for concurrency in levels:
                requests = min(max(16, 4 * concurrency), spec.max_requests)
                for mode in modes:
                    cells.append(
                        LoadCell(spec.scenario, mode, concurrency, requests, spec.latency_ms, spec.jitter_sigma, size, repeat)
                    )
    return cells


# ---------------------------------------------------------------------------
# The mock server process


class MockServerProcess:
    def __init__(self, process: subprocess.Popen[str], port: int, loop_name: str) -> None:
        self.process: Final = process
        self.port: Final = port
        self.loop_name: Final = loop_name

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _call(self, method: str, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, method=method, data=b"" if method == "POST" else None)
        with urllib.request.urlopen(request, timeout=10.0) as response:
            return json.loads(response.read())

    def reset(self) -> None:
        self._call("POST", "/_mock/reset")

    def stats(self, close_times: bool = False) -> dict[str, Any]:
        return self._call("GET", "/_mock/stats" + ("?close_times=1" if close_times else ""))

    def check_alive(self) -> None:
        if self.process.poll() is not None:
            raise BenchError(f"the mock server exited unexpectedly (status {self.process.returncode})")


@contextmanager
def mock_server(log_dir: Path) -> Iterator[MockServerProcess]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = (log_dir / "mock_server.stderr").open("w")
    process = subprocess.Popen(
        [sys.executable, str(_SERVER), "--port", "0", "--loop", "auto"],
        stdout=subprocess.PIPE,
        stderr=stderr_log,
        text=True,
        cwd=BENCH_DIR,
    )
    try:
        assert process.stdout is not None
        line = _readline_with_timeout(process, _SERVER_START_TIMEOUT_S)
        parts = line.split()
        if len(parts) != 3 or parts[0] != "LISTENING":
            raise BenchError(f"the mock server did not start (said {line!r}); see {log_dir / 'mock_server.stderr'}")
        yield MockServerProcess(process, int(parts[1]), parts[2])
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stderr_log.close()


def _readline_with_timeout(process: subprocess.Popen[str], timeout_s: float) -> str:
    assert process.stdout is not None
    result: list[str] = []
    reader = threading.Thread(target=lambda: result.append(process.stdout.readline()), daemon=True)  # type: ignore[union-attr]
    reader.start()
    reader.join(timeout_s)
    if not result:
        raise BenchError(f"no output from the process within {timeout_s:.0f}s")
    return result[0].strip()


# ---------------------------------------------------------------------------
# Running one worker process


@dataclass(slots=True)
class ProcessPeaks:
    threads: int = 0
    fds: int = 0
    rss_kb: int = 0


@dataclass(slots=True)
class WorkerRun:
    returncode: int
    exit_time: float
    peaks: ProcessPeaks
    stdout_lines: list[str] = field(default_factory=list)
    signal_times: list[float] = field(default_factory=list)
    timed_out: bool = False


def run_worker(
    args: Sequence[str], stderr_path: Path, *, sigint_after_start_s: float | None = None, timeout_s: float = _CELL_TIMEOUT_S
) -> WorkerRun:
    """Runs client_worker.py, sampling its /proc entry until it exits.

    With `sigint_after_start_s`, sends SIGINT that long after the worker prints its
    LOAD_START line (the cancellation scenario's Ctrl-C).
    """
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(_WORKER), *args], stdout=subprocess.PIPE, stderr=stderr, text=True, cwd=BENCH_DIR
        )
        lines: list[str] = []
        load_start: list[float] = []

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line.rstrip("\n"))
                if line.startswith("LOAD_START"):
                    load_start.append(float(line.split()[1]))

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        peaks = ProcessPeaks()
        signal_times: list[float] = []
        deadline = time.monotonic() + timeout_s
        timed_out = False
        for tick in itertools.count():
            if process.poll() is not None:
                break
            now = time.monotonic()
            if now > deadline:
                process.kill()
                timed_out = True
                process.wait()
                break
            status = proc_status(process.pid)
            peaks.threads = max(peaks.threads, status.get("Threads", 0))
            peaks.rss_kb = max(peaks.rss_kb, status.get("VmHWM", 0))
            if tick % _FD_SAMPLE_EVERY == 0:
                peaks.fds = max(peaks.fds, open_fd_count(process.pid))
            if sigint_after_start_s is not None and load_start and not signal_times:
                if now >= load_start[0] + sigint_after_start_s:
                    signal_times.append(time.monotonic())
                    process.send_signal(signal.SIGINT)
            time.sleep(_SAMPLE_INTERVAL_S if sigint_after_start_s is None else 0.002)
        exit_time = time.monotonic()
        reader.join(5.0)
    return WorkerRun(process.returncode, exit_time, peaks, lines, signal_times, timed_out)


# ---------------------------------------------------------------------------
# Scenario drivers


def run_load_cell(server: MockServerProcess, cell: LoadCell, out_dir: Path) -> dict[str, Any]:
    server.check_alive()
    server.reset()
    result_path = out_dir / f"{cell.cell_id}.worker.json"
    args = [
        "--scenario", "load",
        "--mode", cell.mode,
        "--base-url", server.base_url,
        "--concurrency", str(cell.concurrency),
        "--requests", str(cell.requests),
        "--latency-ms", f"{cell.latency_ms:g}",
        "--jitter-sigma", f"{cell.jitter_sigma:g}",
        "--prompt-bytes", str(cell.size.prompt_bytes),
        "--response-bytes", str(cell.size.response_bytes),
        "--out", str(result_path),
    ]  # fmt: skip
    run = run_worker(args, out_dir / f"{cell.cell_id}.stderr")
    server_stats = server.stats()
    record: dict[str, Any] = {
        "cell": {
            "scenario": cell.scenario,
            "mode": cell.mode,
            "concurrency": cell.concurrency,
            "requests": cell.requests,
            "latency_ms": cell.latency_ms,
            "jitter_sigma": cell.jitter_sigma,
            "size": cell.size.name,
            "prompt_bytes": cell.size.prompt_bytes,
            "response_bytes": cell.size.response_bytes,
            "repeat": cell.repeat,
        },
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "peaks": {"threads": run.peaks.threads, "fds": run.peaks.fds, "rss_kb": run.peaks.rss_kb},
        "server": server_stats,
        "server_loop": server.loop_name,
    }
    if run.returncode == 0 and result_path.is_file():
        record["worker"] = json.loads(result_path.read_text())
        result_path.unlink()
    write_json(out_dir / f"{cell.cell_id}.json", record)
    return record


def _wait_for_api_connections_closed(server: MockServerProcess, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        stats = server.stats(close_times=True)
        if stats["open_api_connections"] == 0 or time.monotonic() > deadline:
            return stats
        time.sleep(0.05)


def run_cancel_cell(server: MockServerProcess, method: str, repeat: int, out_dir: Path) -> dict[str, Any]:
    server.check_alive()
    server.reset()
    cell_id = f"cancel-{method}-r{repeat}"
    events_path = out_dir / f"{cell_id}.events.jsonl"
    events_path.unlink(missing_ok=True)
    args = [
        "--scenario", "cancel",
        "--cancel-method", method,
        "--cancel-after-s", f"{CANCEL_AFTER_S:g}",
        "--base-url", server.base_url,
        "--concurrency", str(CANCEL_CONCURRENCY),
        "--requests", str(CANCEL_REQUESTS),
        "--latency-ms", f"{CANCEL_LATENCY_MS:g}",
        "--out", str(events_path),
    ]  # fmt: skip
    sigint_after = CANCEL_AFTER_S if method in _SIGINT_METHODS else None
    run = run_worker(args, out_dir / f"{cell_id}.stderr", sigint_after_start_s=sigint_after, timeout_s=60.0)
    server_stats = _wait_for_api_connections_closed(server, timeout_s=15.0)
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.is_file() else []
    record = {
        "cell": {
            "scenario": "cancel",
            "method": method,
            "repeat": repeat,
            "concurrency": CANCEL_CONCURRENCY,
            "requests": CANCEL_REQUESTS,
            "latency_ms": CANCEL_LATENCY_MS,
            "cancel_after_s": CANCEL_AFTER_S,
        },
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "exit_time": run.exit_time,
        "signal_times": run.signal_times,
        "peaks": {"threads": run.peaks.threads, "fds": run.peaks.fds, "rss_kb": run.peaks.rss_kb},
        "events": events,
        "server": server_stats,
    }
    write_json(out_dir / f"{cell_id}.json", record)
    return record


def _describe_load(record: dict[str, Any]) -> str:
    cell = record["cell"]
    head = f"{cell['scenario']:>15} {cell['size']:>15} {cell['mode']:>14} C={cell['concurrency']:<5} r{cell['repeat']}"
    worker = record.get("worker")
    if worker is None:
        return f"{head}  FAILED (returncode {record['returncode']}, timed out: {record['timed_out']})"
    errors = sum(1 for e in worker["samples"]["error"] if e)
    ok = cell["requests"] - errors
    cpu = worker["cpu_user_s"] + worker["cpu_sys_s"]
    return (
        f"{head}  {worker['wall_s']:7.2f}s  {ok / worker['wall_s']:8.1f} req/s  cpu {cpu:6.2f}s  "
        f"threads {record['peaks']['threads']:>4}  errors {errors}"
    )


def _describe_cancel(record: dict[str, Any]) -> str:
    cell = record["cell"]
    times = {e["event"]: e["t"] for e in record["events"]}
    cancel = record["signal_times"][0] if record["signal_times"] else times.get("cancel_requested")
    if cancel is None:
        return f"{'cancel':>15} {cell['method']:>24} r{cell['repeat']}  no cancel recorded (returncode {record['returncode']})"
    regain = times.get("regain")
    regain_text = f"{regain - cancel:6.2f}s" if regain is not None else "   n/a"
    return (
        f"{'cancel':>15} {cell['method']:>24} r{cell['repeat']}  regain {regain_text}  "
        f"exit {record['exit_time'] - cancel:6.2f}s  returncode {record['returncode']}"
    )


def run_local(out_dir: Path, scenarios: Sequence[str], repeats: int, quick: bool) -> None:
    assert repeats >= 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with mock_server(out_dir) as server:
        print(f"mock server on port {server.port} ({server.loop_name}); results in {out_dir}", flush=True)
        for scenario in scenarios:
            scenario_dir = out_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            if scenario == "cancel":
                for repeat in range(repeats):
                    for method in CANCEL_METHODS:
                        print(_describe_cancel(run_cancel_cell(server, method, repeat, scenario_dir)), flush=True)
                continue
            spec = SWEEPS[scenario]
            for cell in build_load_cells(spec, repeats, quick):
                print(_describe_load(run_load_cell(server, cell, scenario_dir)), flush=True)
    write_json(out_dir / "environment.json", environment())


def environment() -> dict[str, Any]:
    import platform

    import anthropic
    import httpcore
    import httpx

    try:
        import uvloop

        uvloop_version = uvloop.__version__
    except ImportError:
        uvloop_version = None
    try:
        from anthropic import DefaultAioHttpClient

        DefaultAioHttpClient()
        aiohttp_backend = "available"
    except Exception as e:  # the SDK raises RuntimeError when the extra is not installed
        aiohttp_backend = f"unavailable: {e}"
    return {
        "date": time.strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "anthropic": anthropic.__version__,
        "httpx": httpx.__version__,
        "httpcore": httpcore.__version__,
        "uvloop": uvloop_version,
        "aiohttp_backend": aiohttp_backend,
        "nproc": os.cpu_count(),
        "cpu": _cpu_model(),
        "kernel": platform.release(),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"
