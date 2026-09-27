"""Tests for the live-run spending cap. Run: python3 -m pytest bench/test_budget.py"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget import Budget, BudgetRefused, Pricing, SONNET_5_PRICING  # noqa: E402


def _usage(input_tokens: int, output_tokens: int, **extra: int) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, **extra)


def test_worst_case_prices_input_and_full_max_tokens() -> None:
    assert SONNET_5_PRICING.worst_case_usd(1_000_000, 1) == pytest.approx(2.0 + 10.0 / 1e6)
    assert SONNET_5_PRICING.worst_case_usd(1, 1_000_000) == pytest.approx(10.0 + 2.0 / 1e6)


def test_usage_cost_includes_cache_tokens() -> None:
    pricing = Pricing(input_per_mtok=2.0, output_per_mtok=10.0)
    usage = _usage(1_000_000, 0, cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    assert pricing.usage_usd(usage) == pytest.approx(2.0 + 2.5 + 0.2)


def test_refuses_a_reservation_past_the_cap_counting_in_flight_money() -> None:
    budget = Budget(cap_usd=0.01, request_cap=100, pricing=SONNET_5_PRICING)
    first = budget.reserve(est_input_tokens=1000, max_tokens=500)  # $0.007
    with pytest.raises(BudgetRefused):
        budget.reserve(est_input_tokens=1000, max_tokens=500)  # would make $0.014 committed
    budget.settle_usage(first, _usage(1000, 100))  # actually cost $0.003
    budget.reserve(est_input_tokens=1000, max_tokens=500)  # $0.003 + $0.007 fits


def test_refuses_past_the_request_cap_even_when_money_remains() -> None:
    budget = Budget(cap_usd=100.0, request_cap=2, pricing=SONNET_5_PRICING)
    for _ in range(2):
        budget.settle_unbilled(budget.reserve(10, 10))
    with pytest.raises(BudgetRefused):
        budget.reserve(10, 10)


def test_unknown_outcomes_are_charged_the_worst_case() -> None:
    budget = Budget(cap_usd=1.0, request_cap=10, pricing=SONNET_5_PRICING)
    reservation = budget.reserve(1000, 1000)
    budget.settle_unknown(reservation)
    snapshot = budget.snapshot()
    assert snapshot.spent_usd == pytest.approx(reservation.worst_usd)
    assert snapshot.unknown_outcome_usd == pytest.approx(reservation.worst_usd)
    assert snapshot.outstanding_usd == pytest.approx(0.0)


def test_a_response_costing_more_than_reserved_is_flagged() -> None:
    budget = Budget(cap_usd=1.0, request_cap=10, pricing=SONNET_5_PRICING)
    budget.settle_usage(budget.reserve(10, 10), _usage(10_000, 10))
    assert budget.snapshot().estimate_violations == 1


def test_concurrent_reservations_never_commit_past_the_cap() -> None:
    budget = Budget(cap_usd=0.05, request_cap=10_000, pricing=SONNET_5_PRICING)
    granted: list[object] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(200):
            try:
                reservation = budget.reserve(100, 100)  # $0.0012 each
            except BudgetRefused:
                return
            with lock:
                granted.append(reservation)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snapshot = budget.snapshot()
    assert snapshot.outstanding_usd <= 0.05 + 1e-12
    assert len(granted) == int(0.05 / SONNET_5_PRICING.worst_case_usd(100, 100))
