"""Renders the benchmark's result files as Markdown tables.

Across repeats, each cell shows the median; throughput and CPU cost also show
the range (min-max) so run-to-run spread on a shared machine stays visible.
"""

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from client_worker import CANCEL_METHODS
from common import Summary, median, read_json

_POOL_LIMIT: Final = 1000  # the SDK's default httpx max_connections
_SDK_MODES: Final = frozenset({"threads", "asyncio", "asyncio-uvloop"})
_MODE_ORDER: Final = ("threads", "asyncio", "asyncio-uvloop", "raw-asyncio")


def _fmt(value: float, digits: int = 1) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _med_range(values: Sequence[float], digits: int = 1) -> str:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return "-"
    if len(finite) == 1:
        return _fmt(finite[0], digits)
    return f"{_fmt(median(finite), digits)} ({_fmt(min(finite), digits)}-{_fmt(max(finite), digits)})"


def _bytes(count: int) -> str:
    return f"{count // 1024} KiB" if count >= 1024 and count % 1024 == 0 else f"{count} B"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load cells


class LoadCellResult:
    """Derived metrics of one load cell (one worker process run)."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.cell: Final[dict[str, Any]] = record["cell"]
        self.record: Final = record
        worker = record.get("worker")
        self.completed: Final = worker is not None
        if worker is None:
            return
        samples = worker["samples"]
        rows = list(zip(samples["t0"], samples["t1"], samples["server_recv"], samples["server_send"], samples["delay"], samples["error"], strict=True))
        ok = [r for r in rows if r[5] is None]
        self.errors: Final = len(rows) - len(ok)
        self.error_kinds: Final = sorted({r[5] for r in rows if r[5] is not None})
        self.wall_s: Final = worker["wall_s"]
        self.throughput: Final = len(ok) / self.wall_s
        self.overhead_ms: Final = Summary.of([(t1 - t0 - (send - recv)) * 1000 for t0, t1, recv, send, _, _ in ok])
        self.pre_ms: Final = Summary.of([(recv - t0) * 1000 for t0, _, recv, _, _, _ in ok])
        self.post_ms: Final = Summary.of([(t1 - send) * 1000 for _, t1, _, send, _, _ in ok])
        self.mean_delay_s: Final = sum(r[4] for r in ok) / len(ok) if ok else math.nan
        self.cpu_ms_per_req: Final = (worker["cpu_user_s"] + worker["cpu_sys_s"]) * 1000 / len(rows)
        self.cpu_util: Final = (worker["cpu_user_s"] + worker["cpu_sys_s"]) / self.wall_s
        self.rss_peak_mb: Final = worker["rss_peak_kb"] / 1024
        self.rss_growth_mb: Final = (worker["rss_peak_kb"] - worker["rss_before_kb"]) / 1024
        self.threads: Final = record["peaks"]["threads"]
        self.fds: Final = record["peaks"]["fds"]
        server = record["server"]
        self.connections: Final = server["api_connections"]
        self.server_lag_p99_ms: Final = server["loop_lag_s"]["p99"] * 1000
        self.server_lag_max_ms: Final = server["loop_lag_s"]["max"] * 1000
        self.server_late_p99_ms: Final = server["timer_lateness_s"]["p99"] * 1000
        self.server_cpu_util: Final = server["cpu_util"]

    def ideal_throughput(self) -> float:
        """Little's law with the simulated latency alone: N requests in ceil(N/C) rounds.

        Only meaningful for a fixed latency; the SDK modes cannot exceed the pool's 1000 connections.
        """
        if self.cell["jitter_sigma"] > 0:
            return math.nan
        concurrency = self.cell["concurrency"]
        if self.cell["mode"] in _SDK_MODES:
            concurrency = min(concurrency, _POOL_LIMIT)
        rounds = math.ceil(self.cell["requests"] / concurrency)
        return self.cell["requests"] / (rounds * self.cell["latency_ms"] / 1000)


def _load_results(directory: Path) -> list[LoadCellResult]:
    return [LoadCellResult(read_json(p)) for p in sorted(directory.glob("*.json"))]


def _group(results: Iterable[LoadCellResult], key: Callable[[LoadCellResult], Any]) -> dict[Any, list[LoadCellResult]]:
    groups: dict[Any, list[LoadCellResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return groups


def _mode_rank(mode: str) -> int:
    return _MODE_ORDER.index(mode) if mode in _MODE_ORDER else len(_MODE_ORDER)


def _short_error(kind: str) -> str:
    """'anthropic.APIConnectionError(httpx.ReadError)' -> 'APIConnectionError(ReadError)'."""
    return re.sub(r"[\w.]+\.(\w+)", r"\1", kind)


def _errors_cell(cells: Sequence[LoadCellResult]) -> str:
    total = sum(c.errors for c in cells)
    kinds = sorted({_short_error(k) for c in cells for k in c.error_kinds})
    return f"{total} {', '.join(kinds)}" if total else "0"


def sweep_table(directory: Path) -> str:
    results = _load_results(directory)
    groups = _group(results, lambda r: (r.cell["concurrency"], _mode_rank(r.cell["mode"]), r.cell["mode"]))
    rows = []
    for (concurrency, _, mode), cells in sorted(groups.items()):
        done = [c for c in cells if c.completed]
        requests = cells[0].cell["requests"]
        if not done:
            rows.append([str(concurrency), str(requests), mode, "failed"] + ["-"] * 8)
            continue
        ideal = done[0].ideal_throughput()
        throughputs = [c.throughput for c in done]
        rows.append([
            str(concurrency),
            str(requests),
            mode,
            _med_range(throughputs),
            _fmt(100 * median(throughputs) / ideal, 0) if not math.isnan(ideal) else "-",
            " / ".join(_fmt(median([getattr(c.overhead_ms, q) for c in done])) for q in ("p50", "p95", "p99")),
            _med_range([c.cpu_ms_per_req for c in done], 2),
            _fmt(median([c.cpu_util for c in done]) * 100, 0),
            f"{_fmt(median([c.rss_peak_mb for c in done]), 0)} (+{_fmt(median([c.rss_growth_mb for c in done]), 0)})",
            str(int(median([c.threads for c in done]))),
            str(int(median([c.fds for c in done]))),
            str(int(median([c.connections for c in done]))),
            _errors_cell(done),
        ])  # fmt: skip
    headers = [
        "C", "N", "mode", "req/s med (min-max)", "% of ideal", "client overhead p50 / p95 / p99 ms",
        "CPU ms/req", "CPU %", "peak RSS MB (+growth)", "threads", "fds", "TCP conns", "errors",
    ]  # fmt: skip
    return _table(headers, rows)


def overhead_split_table(directory: Path, levels: Sequence[int]) -> str:
    """Where the client overhead goes: before the request reaches the server, or after the reply leaves it."""
    results = [r for r in _load_results(directory) if r.completed and r.cell["concurrency"] in levels]
    groups = _group(results, lambda r: (r.cell["concurrency"], _mode_rank(r.cell["mode"]), r.cell["mode"]))
    rows = []
    for (concurrency, _, mode), cells in sorted(groups.items()):
        rows.append([
            str(concurrency),
            mode,
            f"{_fmt(median([c.pre_ms.p50 for c in cells]))} / {_fmt(median([c.pre_ms.p99 for c in cells]))}",
            f"{_fmt(median([c.post_ms.p50 for c in cells]))} / {_fmt(median([c.post_ms.p99 for c in cells]))}",
        ])  # fmt: skip
    return _table(["C", "mode", "request path p50 / p99 ms", "response path p50 / p99 ms"], rows)


def sizes_table(directory: Path) -> str:
    results = _load_results(directory)
    order = {"tiny": 0, "classifier-like": 1, "large": 2}
    groups = _group(
        results, lambda r: (order.get(r.cell["size"], 9), r.cell["size"], r.cell["concurrency"], _mode_rank(r.cell["mode"]), r.cell["mode"])
    )
    rows = []
    for (_, size, concurrency, _, mode), cells in sorted(groups.items()):
        done = [c for c in cells if c.completed]
        if not done:
            rows.append([size, "-", str(concurrency), mode, "failed", "-", "-", "-"])
            continue
        cell = done[0].cell
        rows.append([
            size,
            f"{_bytes(cell['prompt_bytes'])} / {_bytes(cell['response_bytes'])}",
            str(concurrency),
            mode,
            _med_range([c.throughput for c in done]),
            " / ".join(_fmt(median([getattr(c.overhead_ms, q) for c in done])) for q in ("p50", "p99")),
            _med_range([c.cpu_ms_per_req for c in done], 2),
            str(sum(c.errors for c in done)),
        ])  # fmt: skip
    headers = ["size", "prompt / reply", "C", "mode", "req/s med (min-max)", "client overhead p50 / p99 ms", "CPU ms/req", "errors"]
    return _table(headers, rows)


def server_health_table(directories: Sequence[Path]) -> str:
    rows = []
    for directory in directories:
        done = [r for r in _load_results(directory) if r.completed]
        if not done:
            continue
        rows.append([
            directory.name,
            str(len(done)),
            _fmt(max(r.server_lag_p99_ms for r in done), 2),
            _fmt(max(r.server_lag_max_ms for r in done), 1),
            _fmt(max(r.server_late_p99_ms for r in done), 2),
            _fmt(max(r.server_cpu_util for r in done) * 100, 0),
        ])  # fmt: skip
    headers = ["scenario", "cells", "worst loop lag p99 ms", "worst loop lag max ms", "worst reply-timer lateness p99 ms", "peak server CPU %"]
    return _table(headers, rows)


# ---------------------------------------------------------------------------
# Cancellation


def _cancel_metrics(record: dict[str, Any]) -> dict[str, float] | None:
    events = record["events"]
    times: dict[str, float] = {}
    for event in events:
        times.setdefault(event["event"], event["t"])
    if record["signal_times"]:
        cancel = record["signal_times"][0]
    elif "cancel_requested" in times:
        cancel = times["cancel_requested"]
    else:
        return None
    server = record["server"]
    closes = server.get("close_times", [])
    responses = server.get("response_times", [])
    completed_ok = sum(1 for e in events if e["event"] == "request_done" and e["ok"] and e["t"] > cancel)
    return {
        "regain": times["regain"] - cancel if "regain" in times else math.nan,
        "closed": max(closes) - cancel if closes else math.nan,
        "exit": record["exit_time"] - cancel,
        "replies_after": float(sum(1 for t in responses if t > cancel)),
        "cut": float(server["cancelled_inflight"]),
        "client_ok_after": float(completed_ok),
        "returncode": float(record["returncode"]),
    }


def cancel_table(directory: Path) -> str:
    records = [read_json(p) for p in sorted(directory.glob("*.json"))]
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    order: list[str] = []
    for record in records:
        method = record["cell"]["method"]
        if method not in order:
            order.append(method)
        metrics = _cancel_metrics(record)
        if metrics is not None:
            groups[method].append(metrics)
    rows = []
    for method in sorted(order, key=lambda m: CANCEL_METHODS.index(m) if m in CANCEL_METHODS else 99):
        metrics = groups.get(method, [])
        if not metrics:
            rows.append([method, "no cancel recorded"] + ["-"] * 5)
            continue
        rows.append([
            method,
            _med_range([m["regain"] for m in metrics], 2),
            _med_range([m["closed"] for m in metrics], 2),
            _med_range([m["exit"] for m in metrics], 2),
            _fmt(median([m["replies_after"] for m in metrics]), 0),
            _fmt(median([m["cut"] for m in metrics]), 0),
            ",".join(sorted({str(int(m["returncode"])) for m in metrics})),
        ])  # fmt: skip
    headers = [
        "method", "caller regains control (s)", "last API connection closed (s)", "process exited (s)",
        "replies the server still sent", "requests cut mid-flight", "exit status",
    ]  # fmt: skip
    return _table(headers, rows)


# ---------------------------------------------------------------------------
# Live


def _headroom(summary: dict[str, Any], kind: str) -> str:
    entry = summary["ratelimit"].get(kind)
    if not entry or math.isnan(entry.get("min_headroom", math.nan)):
        return "-"
    return f"{entry['min_headroom'] * 100:.1f}%"


def live_tiny_table(path: Path) -> str:
    data = read_json(path)
    rows = []
    for run in sorted(data["runs"], key=lambda r: (r["concurrency"], _mode_rank(r["mode"]))):
        latency = run["latency_s"]
        rows.append([
            str(run["concurrency"]),
            str(run["requests"]),
            run["mode"],
            _fmt(run["wall_s"], 2),
            _fmt(run["ok"] / run["wall_s"], 1),
            f"{_fmt(latency['p50'], 2)} / {_fmt(latency['p95'], 2)} / {_fmt(latency['max'], 2)}",
            str(run["outcomes"].get("http_429", 0)),
            str(run["requests"] - run["ok"]),
            " / ".join(_headroom(run, k) for k in ("requests", "input-tokens", "output-tokens")),
            f"${run['cost_usd']:.4f}",
        ])  # fmt: skip
    headers = [
        "C", "N", "mode", "wall s", "req/s", "latency p50 / p95 / max s", "429s", "not ok",
        "lowest remaining: requests / input tok / output tok", "cost",
    ]  # fmt: skip
    return _table(headers, rows)


def live_classify_table(path: Path) -> str:
    data = read_json(path)
    rows = []
    for run in sorted(data["runs"], key=lambda r: (r["concurrency"], _mode_rank(r["mode"]))):
        latency = run["latency_s"]
        out_per = run["output_tokens_per_request"]
        rows.append([
            str(run["concurrency"]),
            run["mode"],
            _fmt(run["wall_s"], 2),
            f"{_fmt(latency['p50'], 2)} / {_fmt(latency['p95'], 2)} / {_fmt(latency['max'], 2)}",
            f"{run['input_tokens']:,}",
            f"{run['output_tokens']:,}",
            f"{_fmt(out_per['p50'], 0)} / {_fmt(out_per['max'], 0)}",
            f"${run['cost_usd']:.4f}",
            ", ".join(f"{k}: {v}" for k, v in sorted(run["stop_reasons"].items())),
            f"{run['valid_replies']}/{run['requests']}",
            str(run["outcomes"].get("http_429", 0)),
        ])  # fmt: skip
    headers = [
        "C", "mode", "wall s", "latency p50 / p95 / max s", "input tokens", "output tokens",
        "output tok/request p50 / max", "cost", "stop reasons", "valid replies", "429s",
    ]  # fmt: skip
    return _table(headers, rows)


# ---------------------------------------------------------------------------


def render(out_dir: Path) -> str:
    local_dir = out_dir / "local"
    sections = []
    titles = {
        "sweep-1s": "Concurrency sweep, fixed 1 s simulated latency",
        "sweep-1s-jitter": "Concurrency sweep, 1 s median latency, log-normal jitter (sigma 0.5)",
        "sweep-50ms": "Concurrency sweep, fixed 50 ms simulated latency",
    }
    for scenario, title in titles.items():
        directory = local_dir / scenario
        if directory.is_dir():
            sections.append(f"### {title}\n\n{sweep_table(directory)}")
    if (local_dir / "sweep-1s").is_dir():
        sections.append(
            "### Client overhead split, fixed 1 s latency\n\n"
            + overhead_split_table(local_dir / "sweep-1s", (1, 16, 256, 1024))
        )
    if (local_dir / "sizes").is_dir():
        sections.append(f"### Request and reply sizes, fixed 200 ms latency\n\n{sizes_table(local_dir / 'sizes')}")
    load_dirs = [local_dir / s for s in ("sweep-1s", "sweep-1s-jitter", "sweep-50ms", "sizes") if (local_dir / s).is_dir()]
    if load_dirs:
        sections.append(f"### Mock server health\n\n{server_health_table(load_dirs)}")
    if (local_dir / "cancel").is_dir():
        sections.append(
            "### Cancellation (C=64 in flight + 64 queued, 10 s latency, cancel at 0.5 s; times from the cancel request)\n\n"
            + cancel_table(local_dir / "cancel")
        )
    live_dir = out_dir / "live"
    if (live_dir / "live-tiny.json").is_file():
        sections.append(f"### Live API, tiny requests\n\n{live_tiny_table(live_dir / 'live-tiny.json')}")
    if (live_dir / "live-classify.json").is_file():
        sections.append(f"### Live API, classification workload\n\n{live_classify_table(live_dir / 'live-classify.json')}")
    return "\n\n".join(sections) + "\n"

