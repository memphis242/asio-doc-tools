"""A hard spending cap shared by every live Claude API request of one benchmark run.

Every request attempt reserves its worst-case cost before it is sent - its
estimated input tokens at the input price plus its full `max_tokens` at the
output price - and the reservation is refused if it would take reserved plus
already-spent money past the cap, or the attempt count past the request cap.
After the attempt, the reservation settles to what it actually cost:

  - a response: its reported usage;
  - a 4xx rejection (400, 401, 403, 404, 413, 429): nothing, the API does not bill it;
  - anything whose outcome is unknown (timeout, connection error, 5xx, other 4xx):
    the full worst case, since it may have been billed.

The SDK's own retries are disabled everywhere (`max_retries=0`), so every attempt
that reaches the network passes through `reserve` first. The input-token
estimate is the only estimated term; the live harness checks it against the
free token-counting endpoint for every distinct request before spending anything.
"""

import threading
from dataclasses import dataclass
from typing import Any, Final

UNBILLED_STATUSES: Final = frozenset({400, 401, 403, 404, 413, 429})


class BudgetRefused(Exception):
    """A request attempt that would exceed the spending or request cap."""


@dataclass(frozen=True, slots=True)
class Pricing:
    """US dollars per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.1

    def worst_case_usd(self, est_input_tokens: int, max_tokens: int) -> float:
        assert est_input_tokens > 0 and max_tokens > 0
        return (est_input_tokens * self.input_per_mtok + max_tokens * self.output_per_mtok) / 1e6

    def usage_usd(self, usage: Any) -> float:
        """Cost of a response's `usage` (an SDK Usage object)."""
        cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        input_equivalent = (
            usage.input_tokens + cache_write * self.cache_write_multiplier + cache_read * self.cache_read_multiplier
        )
        return (input_equivalent * self.input_per_mtok + usage.output_tokens * self.output_per_mtok) / 1e6


# Claude Sonnet 5, first-party API list prices.
SONNET_5_PRICING: Final = Pricing(input_per_mtok=2.0, output_per_mtok=10.0)


@dataclass(frozen=True, slots=True)
class Reservation:
    worst_usd: float
    est_input_tokens: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    cap_usd: float
    request_cap: int
    attempts: int
    spent_usd: float
    unknown_outcome_usd: float
    outstanding_usd: float
    estimate_violations: int


class Budget:
    """Thread-safe; its methods never block for long, so asyncio code may call them directly."""

    def __init__(self, cap_usd: float, request_cap: int, pricing: Pricing) -> None:
        assert cap_usd > 0.0 and request_cap > 0
        self._cap_usd: Final = cap_usd
        self._request_cap: Final = request_cap
        self._pricing: Final = pricing
        self._lock = threading.Lock()
        self._attempts = 0
        self._spent_usd = 0.0  # settled cost, including worst cases counted for unknown outcomes
        self._unknown_outcome_usd = 0.0  # the part of _spent_usd that is a worst case, not a measurement
        self._outstanding_usd = 0.0  # reserved by attempts still in flight
        self._estimate_violations = 0  # responses that cost more than their reservation

    @property
    def pricing(self) -> Pricing:
        return self._pricing

    def remaining_usd(self) -> float:
        with self._lock:
            return self._cap_usd - self._spent_usd - self._outstanding_usd

    def remaining_requests(self) -> int:
        with self._lock:
            return self._request_cap - self._attempts

    def reserve(self, est_input_tokens: int, max_tokens: int) -> Reservation:
        worst = self._pricing.worst_case_usd(est_input_tokens, max_tokens)
        with self._lock:
            if self._attempts >= self._request_cap:
                raise BudgetRefused(f"request cap of {self._request_cap} attempts reached")
            committed = self._spent_usd + self._outstanding_usd
            if committed + worst > self._cap_usd:
                raise BudgetRefused(
                    f"worst case ${worst:.4f} on top of ${committed:.4f} committed would exceed the "
                    f"${self._cap_usd:.2f} cap"
                )
            self._attempts += 1
            self._outstanding_usd += worst
        return Reservation(worst, est_input_tokens, max_tokens)

    def settle_usage(self, reservation: Reservation, usage: Any) -> float:
        cost = self._pricing.usage_usd(usage)
        self._settle(reservation, cost, unknown=False)
        return cost

    def settle_unbilled(self, reservation: Reservation) -> None:
        self._settle(reservation, 0.0, unknown=False)

    def settle_unknown(self, reservation: Reservation) -> None:
        self._settle(reservation, reservation.worst_usd, unknown=True)

    def _settle(self, reservation: Reservation, cost: float, *, unknown: bool) -> None:
        assert cost >= 0.0
        with self._lock:
            assert self._outstanding_usd >= reservation.worst_usd - 1e-12
            self._outstanding_usd = max(0.0, self._outstanding_usd - reservation.worst_usd)
            self._spent_usd += cost
            if unknown:
                self._unknown_outcome_usd += cost
            if cost > reservation.worst_usd + 1e-12:
                self._estimate_violations += 1

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                cap_usd=self._cap_usd,
                request_cap=self._request_cap,
                attempts=self._attempts,
                spent_usd=self._spent_usd,
                unknown_outcome_usd=self._unknown_outcome_usd,
                outstanding_usd=self._outstanding_usd,
                estimate_violations=self._estimate_violations,
            )
