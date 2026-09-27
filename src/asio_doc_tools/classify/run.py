"""What every run does around its engine, the same for both: planning the batches and
caps, the checks before anything is sent or printed, the messages a run prints, and how
a run that stopped early ends.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..diag import AsioDocsError, spend
from . import budget
from .budget import Caps, RunBudget, StopFlag, quantity
from .prompt import batch_payload
from .store import STORE_LOCK_WAIT_S, Recorder, check_store_writable

RERUN_HINT: Final = "Rerunning continues where this run stopped: stored entries are never sent again."


@dataclass(frozen=True, slots=True)
class Plan:
    batches: tuple[tuple[str, ...], ...]
    payload_bytes: tuple[int, ...]  # of each batch's first request
    caps: Caps

    @property
    def entries(self) -> int:
        return sum(len(batch) for batch in self.batches)


def plan_run(
    conn: sqlite3.Connection, store_path: Path, pending: Sequence[str], texts: dict[str, str]
) -> Plan:
    """Plans the batches and their caps, and checks what must hold before anything is sent,
    or anything about sending printed: the run's worst case is under the ceiling, and its
    results can be stored."""
    assert pending
    size = budget.BATCH_SIZE
    batches = tuple(tuple(pending[i : i + size]) for i in range(0, len(pending), size))
    payload_bytes = tuple(len(batch_payload(batch, texts)[1].encode()) for batch in batches)
    caps = budget.caps_for_run(len(pending), tuple(budget.REQUEST_INPUT_BASE + n for n in payload_bytes))
    budget.check_cost_ceiling(caps)
    check_store_writable(conn, store_path)
    return Plan(batches=batches, payload_bytes=payload_bytes, caps=caps)


def check_credentials(client: Any) -> None:
    """Without credentials the SDK fails only at the first request, with a bare TypeError."""
    if not client.api_key and not client.auth_token and client.credentials is None:
        raise AsioDocsError(
            "no Anthropic API credentials found; set ANTHROPIC_API_KEY or run 'ant auth login'"
        )


def hard_limit_at(run_budget: RunBudget) -> float:
    """When, on the budget's clock, the run stops waiting for anything."""
    return run_budget.started_at + budget.hard_limit_s()


def store_lock_wait_s(run_budget: RunBudget) -> float:
    """How long a store write may wait for another process's lock: never past the hard limit."""
    return min(STORE_LOCK_WAIT_S, max(0.0, hard_limit_at(run_budget) - run_budget.clock()))


class UsageReport:
    """Prints the run's usage line, once, whichever way the run ends."""

    def __init__(self, run_budget: RunBudget) -> None:
        self._budget: Final = run_budget
        self._printed = False

    def __call__(self) -> None:
        if not self._printed:
            self._printed = True
            spend(budget.usage_line(self._budget.usage()))


def waiting_line(in_flight: int, *, interrupted: bool) -> str:
    again = "again " if interrupted else ""
    return (
        f"stopping: waiting for {quantity(in_flight, 'in-flight request')} so their results are kept; "
        f"press Ctrl-C {again}to abandon them"
    )


def stopping_early_line(in_flight: int) -> str:
    return f"stopping early: waiting for {quantity(in_flight, 'in-flight request')} so their results are kept"


def abandoned_line(in_flight: int, progress: str) -> str:
    return f"abandoned {quantity(in_flight, 'in-flight request')}, which may still be billed; {progress}"


def hard_limit_message(in_flight: int, progress: str) -> str:
    return (
        f"the run reached its {budget.hard_limit_s():.0f} s wall-clock limit and abandoned "
        f"{quantity(in_flight, 'request')} still in flight, which may still be billed; {progress}. "
        f"{RERUN_HINT}"
    )


def interrupted_line(progress: str) -> str:
    return f"interrupted: {progress}. {RERUN_HINT}"


def finish(recorder: Recorder, stop: StopFlag, total: int) -> None:
    """How a run that was not interrupted ends, once everything paid for is recorded:
    a batch's unexpected exception is a bug, raised with its traceback; otherwise the stop
    flag's reason, if any, becomes the run's error."""
    if recorder.first_failure is not None:
        raise recorder.first_failure
    reason = stop.reason
    if reason is not None:
        raise AsioDocsError(f"{reason}; {recorder.progress(total)}. {RERUN_HINT}") from reason
