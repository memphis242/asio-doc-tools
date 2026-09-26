"""Helpers shared by the benchmark harness, the client worker, and the mock server.

Every timestamp exchanged between processes is `time.monotonic()`, which on
Linux reads CLOCK_MONOTONIC: one system-wide clock, so a timestamp taken in the
mock server can be subtracted from one taken in a client process.
"""

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

BENCH_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = BENCH_DIR.parent
SRC_DIR: Final = REPO_ROOT / "src"

# Request headers the mock server reads to shape its reply. The client sets them
# as default headers, so every request of one benchmark cell carries the same ones.
HDR_LATENCY_MS: Final = "x-mock-latency-ms"
HDR_JITTER_SIGMA: Final = "x-mock-jitter-sigma"
HDR_RESPONSE_BYTES: Final = "x-mock-response-bytes"

# Response headers the mock server sets: monotonic timestamps for when the full
# request had arrived and when the response was written, and the simulated delay.
HDR_RECV: Final = "x-mock-recv"
HDR_SEND: Final = "x-mock-send"
HDR_DELAY: Final = "x-mock-delay"

MOCK_MODEL: Final = "claude-sonnet-5"
MOCK_API_KEY: Final = "mock-key-not-a-secret"


class BenchError(Exception):
    """A benchmark failure to report as-is (no traceback)."""


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]; NaN for an empty input."""
    assert 0.0 <= q <= 100.0
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


@dataclass(frozen=True, slots=True)
class Summary:
    count: int
    p50: float
    p95: float
    p99: float
    max: float
    mean: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "Summary":
        if not values:
            return cls(0, math.nan, math.nan, math.nan, math.nan, math.nan)
        return cls(
            count=len(values),
            p50=percentile(values, 50.0),
            p95=percentile(values, 95.0),
            p99=percentile(values, 99.0),
            max=max(values),
            mean=sum(values) / len(values),
        )

    def to_json(self) -> dict[str, float]:
        return {"count": self.count, "p50": self.p50, "p95": self.p95, "p99": self.p99, "max": self.max, "mean": self.mean}


def proc_status(pid: int | str = "self") -> dict[str, int]:
    """Threads and memory fields (kB) of /proc/<pid>/status; empty once the process is gone."""
    wanted = ("Threads", "VmRSS", "VmHWM")
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:  # gone, or a zombie whose entries are no longer readable
        return {}
    fields: dict[str, int] = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name in wanted:
            fields[name] = int(value.split()[0])
    return fields


def open_fd_count(pid: int | str = "self") -> int:
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:  # gone, or a zombie whose fd directory is no longer readable
        return 0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ratelimit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Only the rate-limit headers of a response: nothing identifying (no request or org id)."""
    kept = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered.startswith("anthropic-ratelimit-") or lowered == "retry-after":
            kept[lowered] = value
    return kept
