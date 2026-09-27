"""A classification run's budget: its hard caps, the reservations every request attempt
makes against them, and the spend lines printed before and after the run.

Both engines admit every HTTP attempt through one `RunBudget`, so the bounds below hold
whichever engine runs.
"""

import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from ..diag import AsioDocsError
from .prompt import EFFORT, MODEL, SCHEMA, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Run budget
#
# Every HTTP attempt a classify() run makes (a batch's first send, a split half, a
# resend after an invalid reply, and every retry of any of these) is first admitted by
# the run's RunBudget, so the bounds below hold on every code path of either engine.
#
# Requests per batch. A request for n entries asks for max_tokens = 2,048 + 128 n
# (7,168 for a full batch of 40, 4,608 for a half of 20, 2,176 for one entry), at least
# 5x the normal output of about 32 tokens per entry, thinking included. A max_tokens
# stop therefore means runaway output, which smaller requests do not fix, so a batch is
# split at most once:
#
#   reply                                          logical sends for the batch
#   valid                                          1
#   invalid (bad JSON, missing or unknown ids)     2: the same request is resent once
#   max_tokens on the full batch                   1 + two halves of 1 or 2 each: at most 5
#   max_tokens on a half, a resend, or one entry   error
#   refusal, invalid twice, non-retryable error    error
#
# A logical send is at most 3 HTTP attempts (see retry.py): up to 2 retries, only after
# a 408, 409, 429, or 5xx status (529 included), a connection error, or a timeout,
# waiting 2 s then 4 s, or the server's retry-after, never more than 30 s. The SDK's own
# retries are off. So one batch makes at most 5 * 3 = 15 attempts, and any error stops
# the whole run.
#
# Input per request. An entry's text is sent whole up to 9,160 characters (twice the
# longest real entry) and cut with a visible marker beyond that. A request's input is
# bounded by the UTF-8 bytes of the system prompt (3,624), the JSON schema (447), and
# its payload, plus 1,024 for message framing and the hidden structured-output prompt:
# count_tokens measured 7 tokens for a one-token message and a constant 378 for the
# structured output, 385 in all, so 1,024 is over 2.5x that. This rests on a token
# never covering less than one byte of text, so tokens <= bytes; measured ratios were
# 0.32-0.48 tokens per byte for real payloads and 0.97 at most, for random punctuation.
#
# Caps per run, for N entries to classify in B = ceil(N / 40) batches whose planned
# requests have input bounds b_1..b_B:
#
#   requests       R = 2 B + 4
#   output tokens  T = (min(32, B) + 3) * 7,168 + 128 N
#   input tokens   I = (b_1 + ... + b_B) + (R - B) * max(b_i)
#   time           no attempt starts more than 300 s after the run started
#   wall clock     at 560 s the run stops: the asyncio engine cancels the requests
#                  still in flight, the threaded reference abandons them and exits;
#                  either way whatever finished is kept, usage is reported, and the
#                  exit status is 1
#
# Before sending, an attempt reserves its whole max_tokens and its input bound, and is
# refused unless, for output and for input alike, billed + possibly billed + reserved +
# its own reservation stays within the cap. A response turns its reservations into its
# actual usage; a failure that generated nothing (an HTTP error status, a connection that
# was never made) refunds them; a timeout, a connection dropped mid-request, or a
# cancelled request keeps them as possibly billed. So billed output <= T, billed input
# <= I, and requests <= R, whatever happens. Every request beyond the first per batch (a
# half, a resend, a retry) carries at most one batch's payload, which is why
# (R - B) * max(b_i) covers them. A response billed for more than its reservation (the
# API passing max_tokens, or text taking more tokens than bytes) stops the run, since the
# caps then prove nothing: only the requests already in flight finish.
#
# Why these numbers. Up to 32 requests are in flight, each reserving up to 7,168, so a
# run needs min(32, B) * 7,168 of headroom for reservations alone. On top of that, 128
# per entry is 4x the normal 32, and 3 more full-size max_tokens cover a max_tokens stop
# (up to 7,168 billed) plus the halves and the resend that follow it. A normal run peaks
# at about 32 N billed plus min(32, B) * 7,168 reserved, which is 96 N + 21,504 under T,
# and its input is well under the sum of its batches' bounds. R gives every batch a
# second request plus 4 spare, so even a one-batch run absorbs a split (3 requests), a
# resend (1), and 2 retries.
#
# Worst case, at Sonnet 5's $2 / $10 per million input / output tokens. A run whose
# worst case, T * $10/M + I * $2/M, is over $6.00 is refused before anything is sent (a
# page with far more or far longer entries than Asio's history would be): diffing a
# smaller range first costs less, and its results carry over.
#
#                       requests  output tokens  input tokens  cost at most
#   one batch of 28     6         32,256         58,464        $0.44
#   full history, 995   54        328,064        904,519       $5.09
#
# (for 1.38.1 -> 1.38.2, whose payload is 4,649 bytes; and for all 995 entries, whose 25
# payloads total 201,349 bytes, the largest 14,760). Even were every one of the full
# history's 25 batches as large as its largest, the worst case would be $5.43. A normal
# full-history run: 25 requests, about 32K output and 115K input tokens, about $0.55.
#
# Time. A request times out after 240 s without receiving a byte (a non-streaming reply
# arrives whole once generated, and a full-size 7,168 tokens takes about 57 s at the
# observed 126 tokens/s: about 4x margin), or after 10 s trying to connect. Retry waits
# end at the time limit, so the last attempt starts by 300 s and, unless its reply
# arrives a byte at a time, ends by 550 s. The hard limit at 560 s bounds even that: a
# store write waits for another process's lock only until it, and past it the asyncio
# engine cancels what is in flight (its connections close within milliseconds) while
# the threaded reference writes one pending file and exits the process, since a thread
# blocked in a socket read cannot be interrupted.
#
# Stopping. The first error, cap refusal, store-write failure, or Ctrl-C sets the run's
# stop flag, which every attempt checks first: from then on no request is sent, only the
# ones already in flight (at most 32) finish, and their results are stored.
# ---------------------------------------------------------------------------

# A request's latency grows with its batch (about 0.9 s + 0.24 s per entry at low
# effort) and does not grow with more requests in flight (bench/RESULTS.md), so every
# batch of even the full history (25 batches) can be in flight at once.
BATCH_SIZE: Final = 40
MAX_IN_FLIGHT: Final = 32

MAX_TOKENS_BASE: Final = 2048
MAX_TOKENS_PER_ENTRY: Final = 128
# The SDK refuses a non-streaming request whose max_tokens could take over 10 minutes to
# generate at its assumed 128,000 tokens per hour (`_calculate_nonstreaming_timeout`):
# 600 s * 128,000 / 3,600 s = 21,333. It only checks that under its default timeout,
# which the clients here replace, so the limit is kept here instead.
SDK_NONSTREAMING_MAX_TOKENS: Final = 21_333

_SYSTEM_PROMPT_BYTES: Final = len(SYSTEM_PROMPT.encode())
_SCHEMA_BYTES: Final = len(json.dumps(SCHEMA).encode())
# count_tokens, for claude-sonnet-5 with this request's shape: a user message "x" alone
# counts 7 tokens, and output_config's json_schema format adds 378 whatever the payload
# (thinking and effort add none).
_MEASURED_MESSAGE_FRAMING_TOKENS: Final = 7
_MEASURED_STRUCTURED_OUTPUT_TOKENS: Final = 378
_INPUT_FRAMING_ALLOWANCE: Final = 1_024
REQUEST_INPUT_BASE: Final = _SYSTEM_PROMPT_BYTES + _SCHEMA_BYTES + _INPUT_FRAMING_ALLOWANCE

REQUESTS_PER_BATCH_CAP: Final = 2
SPARE_REQUESTS_CAP: Final = 4
SPARE_FULL_SIZE_OUTPUTS_CAP: Final = 3
OUTPUT_TOKENS_PER_ENTRY_CAP: Final = 128
RUN_TIME_LIMIT_S: Final = 300.0
MAX_RUN_COST_USD: Final = 6.00

REQUEST_TIMEOUT_S: Final = 240.0
CONNECT_TIMEOUT_S: Final = 10.0
HARD_LIMIT_GRACE_S: Final = 10.0

# Observed at low effort, for the estimate printed before a run: about 32 output tokens
# per entry, thinking included; and count_tokens found every request's input to be
# 1,521 tokens plus about one token per 2.6-2.7 bytes of payload.
_EXPECTED_OUTPUT_TOKENS_PER_ENTRY: Final = 32
_EXPECTED_INPUT_TOKENS_PER_REQUEST: Final = 1_521
_EXPECTED_PAYLOAD_BYTES_PER_TOKEN: Final = 2.6
# Claude Sonnet 5 list prices in US dollars per million tokens: what the printed costs,
# and the $6.00 ceiling on a run's worst case, are computed with.
_INPUT_USD_PER_MTOK: Final = 2.0
_OUTPUT_USD_PER_MTOK: Final = 10.0

# The full history as of 1.38.2, for the checks below: 995 entries in 25 batches, whose
# payloads total 201,349 bytes, the largest 14,760.
_FULL_HISTORY_PAYLOAD_BYTES: Final = 201_349
_FULL_HISTORY_LARGEST_PAYLOAD_BYTES: Final = 14_760


def max_tokens_for(entry_count: int) -> int:
    """max_tokens for a request classifying `entry_count` entries."""
    assert entry_count >= 1
    max_tokens = MAX_TOKENS_BASE + MAX_TOKENS_PER_ENTRY * entry_count
    assert max_tokens <= SDK_NONSTREAMING_MAX_TOKENS
    return max_tokens


def input_bound(payload: str) -> int:
    """An upper bound on a request's input tokens: see "Input per request" above."""
    return REQUEST_INPUT_BASE + len(payload.encode())


def hard_limit_s() -> float:
    """How long after its start a run stops waiting for anything."""
    return RUN_TIME_LIMIT_S + CONNECT_TIMEOUT_S + REQUEST_TIMEOUT_S + HARD_LIMIT_GRACE_S


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _INPUT_USD_PER_MTOK + output_tokens * _OUTPUT_USD_PER_MTOK) / 1_000_000


@dataclass(frozen=True, slots=True)
class Caps:
    max_requests: int
    max_output_tokens: int
    max_input_tokens: int
    time_limit_s: float

    def max_cost_usd(self) -> float:
        """The most a run under these caps can be billed."""
        return cost_usd(self.max_input_tokens, self.max_output_tokens)


def run_input_cap(bounds_sum: int, bounds_max: int, batches: int, requests: int) -> int:
    assert requests >= batches >= 1
    assert bounds_max * batches >= bounds_sum >= bounds_max
    return bounds_sum + (requests - batches) * bounds_max


def caps_for_run(entries: int, batch_input_bounds: Sequence[int]) -> Caps:
    """The caps for `entries` entries in batches whose planned requests have these input bounds."""
    batches = len(batch_input_bounds)
    assert 1 <= batches <= entries
    assert all(bound > REQUEST_INPUT_BASE for bound in batch_input_bounds)
    max_requests = REQUESTS_PER_BATCH_CAP * batches + SPARE_REQUESTS_CAP
    return Caps(
        max_requests=max_requests,
        max_output_tokens=(min(MAX_IN_FLIGHT, batches) + SPARE_FULL_SIZE_OUTPUTS_CAP)
        * max_tokens_for(BATCH_SIZE)
        + OUTPUT_TOKENS_PER_ENTRY_CAP * entries,
        max_input_tokens=run_input_cap(
            sum(batch_input_bounds), max(batch_input_bounds), batches, max_requests
        ),
        time_limit_s=RUN_TIME_LIMIT_S,
    )


def check_cost_ceiling(caps: Caps) -> None:
    worst = caps.max_cost_usd()
    if worst > MAX_RUN_COST_USD:
        raise AsioDocsError(
            f"this run could cost up to ${worst:,.2f} (at most {caps.max_requests:,} requests, "
            f"{caps.max_output_tokens:,} output and {caps.max_input_tokens:,} input tokens), over "
            f"the ${MAX_RUN_COST_USD:.2f} limit per run, so nothing was sent; diff a smaller range "
            "of versions first: its results are stored, so a wider diff afterwards pays only for "
            "what is still missing"
        )


# The arithmetic in the "Run budget" comment above, checked at import time.
assert max_tokens_for(BATCH_SIZE) == 7_168 <= SDK_NONSTREAMING_MAX_TOKENS
assert max_tokens_for(BATCH_SIZE) >= 5 * _EXPECTED_OUTPUT_TOKENS_PER_ENTRY * BATCH_SIZE
assert OUTPUT_TOKENS_PER_ENTRY_CAP >= 4 * _EXPECTED_OUTPUT_TOKENS_PER_ENTRY
assert (_SYSTEM_PROMPT_BYTES, _SCHEMA_BYTES, REQUEST_INPUT_BASE) == (3_624, 447, 5_095)
assert _INPUT_FRAMING_ALLOWANCE >= 2 * (_MEASURED_MESSAGE_FRAMING_TOKENS + _MEASURED_STRUCTURED_OUTPUT_TOKENS)
# A one-batch run absorbs a split, a resend, and a retry: requests 1 + 2 + 1 + 1, and for
# output, the whole batch billed at max_tokens plus both halves and a resend reserved at once.
assert caps_for_run(BATCH_SIZE, (REQUEST_INPUT_BASE + 1,)).max_requests >= 5
assert (
    max_tokens_for(BATCH_SIZE) + 3 * max_tokens_for(BATCH_SIZE // 2)
    <= caps_for_run(BATCH_SIZE, (REQUEST_INPUT_BASE + 1,)).max_output_tokens
)
# One batch of 1.38.1 -> 1.38.2 (28 entries, a 4,649-byte payload).
assert caps_for_run(28, (REQUEST_INPUT_BASE + 4_649,)) == Caps(
    max_requests=6, max_output_tokens=32_256, max_input_tokens=58_464, time_limit_s=300.0
)
assert math.ceil(caps_for_run(28, (REQUEST_INPUT_BASE + 4_649,)).max_cost_usd() * 100) == 44
# The full history: every batch fits in flight at once, its exact worst case is $5.09,
# and even with every batch as large as its largest it stays under the ceiling.
assert MAX_IN_FLIGHT >= 25
_FULL_HISTORY_CAPS: Final = caps_for_run(
    995, (REQUEST_INPUT_BASE + _FULL_HISTORY_LARGEST_PAYLOAD_BYTES,) * 25
)
_FULL_HISTORY_INPUT_CAP: Final = run_input_cap(
    25 * REQUEST_INPUT_BASE + _FULL_HISTORY_PAYLOAD_BYTES,
    REQUEST_INPUT_BASE + _FULL_HISTORY_LARGEST_PAYLOAD_BYTES,
    25,
    _FULL_HISTORY_CAPS.max_requests,
)
assert (_FULL_HISTORY_CAPS.max_requests, _FULL_HISTORY_CAPS.max_output_tokens) == (54, 328_064)
assert _FULL_HISTORY_INPUT_CAP == 904_519
assert math.ceil(cost_usd(_FULL_HISTORY_INPUT_CAP, 328_064) * 100) == 509
assert math.ceil(_FULL_HISTORY_CAPS.max_cost_usd() * 100) == 543
assert _FULL_HISTORY_CAPS.max_cost_usd() < MAX_RUN_COST_USD
# A request times out after at least 4x the time it takes to generate its max_tokens at
# the observed 126 tokens/s, and the hard limit comes after every attempt's own timeouts.
assert REQUEST_TIMEOUT_S >= 4 * max_tokens_for(BATCH_SIZE) / 126
assert hard_limit_s() == 560.0 > RUN_TIME_LIMIT_S + CONNECT_TIMEOUT_S + REQUEST_TIMEOUT_S


def quantity(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural if plural is not None else singular + 's')}"


def _about_usd(amount: float) -> str:
    assert amount >= 0
    if amount < 0.001:
        return "under $0.001"
    return f"about ${amount:.3f}" if amount < 0.1 else f"about ${amount:.2f}"


def at_most_usd(amount: float) -> str:
    """`amount` rounded up to the cent, so the figure printed is never below the bound."""
    assert amount >= 0
    return f"${math.ceil(amount * 100) / 100:.2f}"


def _expected_input_tokens(payload_bytes: int) -> int:
    return _EXPECTED_INPUT_TOKENS_PER_REQUEST + round(payload_bytes / _EXPECTED_PAYLOAD_BYTES_PER_TOKEN)


class RunStopped(Exception):
    """An attempt refused because the run is stopping. Internal: the reason the run
    stopped is the stop flag's, and this never reaches the user."""


class StopFlag:
    """A run's stop flag and the reason it was first set (thread-safe)."""

    def __init__(self) -> None:
        self._event: Final = threading.Event()
        self._lock: Final = threading.Lock()
        self._reason: AsioDocsError | None = None
        self._listeners: list[Callable[[], None]] = []

    def trip(self, reason: AsioDocsError) -> None:
        """Sets the flag; the first reason given is the one kept. Listeners are called
        once, on the thread that sets it first."""
        with self._lock:
            first = self._reason is None
            if first:
                self._reason = reason
                self._event.set()
            listeners = tuple(self._listeners) if first else ()
            self._listeners.clear()
        for listener in listeners:
            listener()

    def on_trip(self, listener: Callable[[], None]) -> None:
        """Calls `listener` once when the flag is set: at once if it already is."""
        with self._lock:
            if self._reason is None:
                self._listeners.append(listener)
                return
        listener()

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> AsioDocsError | None:
        with self._lock:
            assert (self._reason is None) == (not self._event.is_set())
            return self._reason

    def wait(self, seconds: float) -> None:
        """Blocks the calling thread for `seconds`, or less if the flag is set meanwhile."""
        self._event.wait(max(0.0, seconds))


@dataclass(frozen=True, slots=True)
class Reservation:
    """One admitted attempt, holding its max_tokens and input bound until it is settled."""

    serial: int
    max_tokens: int
    input_bound: int


@dataclass(frozen=True, slots=True)
class Usage:
    requests: int  # HTTP attempts admitted, whatever their outcome
    input_tokens: int  # billed, as reported by responses
    output_tokens: int
    cut_off_requests: int  # attempts that failed, or were cancelled, after possibly being billed
    cut_off_input_tokens: int  # their reservations, counted as possibly billed
    cut_off_output_tokens: int
    in_flight_requests: int  # attempts admitted and not settled yet
    reserved_input_tokens: int  # their reservations
    reserved_output_tokens: int

    @property
    def possibly_billed_requests(self) -> int:
        return self.cut_off_requests + self.in_flight_requests

    @property
    def possibly_billed_input_tokens(self) -> int:
        return self.cut_off_input_tokens + self.reserved_input_tokens

    @property
    def possibly_billed_output_tokens(self) -> int:
        return self.cut_off_output_tokens + self.reserved_output_tokens

    def cost_usd(self) -> float:
        """What was billed, plus everything that may have been."""
        return cost_usd(
            self.input_tokens + self.possibly_billed_input_tokens,
            self.output_tokens + self.possibly_billed_output_tokens,
        )


class RunBudget:
    """A run's hard caps, consulted before every HTTP attempt (thread-safe).

    admit() either reserves an attempt's max_tokens and input bound or refuses the
    attempt; a refusal for a cap sets the stop flag with that cap as its reason. Every
    admitted attempt is then settled exactly once, by settle_billed() or settle_failed().
    admit() never blocks or suspends between its checks and its reservation (it takes an
    uncontended lock), so it is atomic for threads and for coroutines on one event loop.
    """

    def __init__(self, caps: Caps, stop: StopFlag, clock: Callable[[], float] = time.monotonic) -> None:
        self._caps: Final = caps
        self._stop: Final = stop
        self._clock: Final = clock
        self._started_at: Final = clock()
        self._deadline: Final = self._started_at + caps.time_limit_s
        self._lock: Final = threading.Lock()
        self._open: set[int] = set()  # serials of admitted, unsettled attempts
        self._requests = 0
        self._billed_input = 0
        self._billed_output = 0
        self._cut_off_requests = 0
        self._cut_off_input = 0
        self._cut_off_output = 0
        self._reserved_input = 0
        self._reserved_output = 0

    @property
    def caps(self) -> Caps:
        return self._caps

    @property
    def stop(self) -> StopFlag:
        return self._stop

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    @property
    def started_at(self) -> float:
        return self._started_at

    def admit(self, max_tokens: int, input_bound: int) -> Reservation:
        """Reserves one attempt, or raises RunStopped: when the stop flag is already set,
        or when this attempt would pass a cap (which sets the stop flag)."""
        assert max_tokens > 0 and input_bound > 0
        with self._lock:
            if self._stop.is_set():
                raise RunStopped
            refusal = self._refusal(max_tokens, input_bound)
            if refusal is None:
                self._requests += 1
                self._reserved_output += max_tokens
                self._reserved_input += input_bound
                self._open.add(self._requests)
                return Reservation(serial=self._requests, max_tokens=max_tokens, input_bound=input_bound)
        self._stop.trip(refusal)  # outside the lock: its listeners may call back in
        raise RunStopped

    def settle_billed(self, reservation: Reservation, *, input_tokens: int, output_tokens: int) -> None:
        """A response arrived: its reservations become the usage it reports.

        Usage over a reservation means the bound behind it (max_tokens for output, bytes >=
        tokens for input) does not hold, so the caps no longer prove anything: the stop
        flag is set, and only the requests already in flight finish. The response itself
        is paid for and kept.
        """
        overruns: list[AsioDocsError] = []
        with self._lock:
            self._release(reservation)
            self._billed_input += input_tokens
            self._billed_output += output_tokens
            for kind, billed, bound in (
                ("input", input_tokens, reservation.input_bound),
                ("output", output_tokens, reservation.max_tokens),
            ):
                if billed > bound:
                    overruns.append(
                        AsioDocsError(
                            f"the API billed {billed:,} {kind} tokens for a request whose computed upper "
                            f"bound was {bound:,}, so this run's hard caps no longer hold and it stopped; "
                            "please report this"
                        )
                    )
        for overrun in overruns:
            self._stop.trip(overrun)

    def settle_failed(self, reservation: Reservation, *, possibly_billed: bool) -> None:
        """No usable response: the reservations are refunded, or kept as possibly billed."""
        with self._lock:
            self._release(reservation)
            if possibly_billed:
                self._cut_off_requests += 1
                self._cut_off_input += reservation.input_bound
                self._cut_off_output += reservation.max_tokens

    def seconds_left(self) -> float:
        return max(0.0, self._deadline - self._clock())

    def usage(self) -> Usage:
        with self._lock:
            return Usage(
                requests=self._requests,
                input_tokens=self._billed_input,
                output_tokens=self._billed_output,
                cut_off_requests=self._cut_off_requests,
                cut_off_input_tokens=self._cut_off_input,
                cut_off_output_tokens=self._cut_off_output,
                in_flight_requests=len(self._open),
                reserved_input_tokens=self._reserved_input,
                reserved_output_tokens=self._reserved_output,
            )

    def _release(self, reservation: Reservation) -> None:
        assert reservation.serial in self._open, "an attempt was settled twice"
        self._open.remove(reservation.serial)
        self._reserved_output -= reservation.max_tokens
        self._reserved_input -= reservation.input_bound
        assert self._reserved_output >= 0 and self._reserved_input >= 0

    def _refusal(self, max_tokens: int, input_bound: int) -> AsioDocsError | None:
        caps = self._caps
        committed_output = self._billed_output + self._cut_off_output + self._reserved_output
        committed_input = self._billed_input + self._cut_off_input + self._reserved_input
        if self._clock() >= self._deadline:
            return AsioDocsError(
                f"stopped at this run's time limit of {caps.time_limit_s:.0f} s ({self._used()})"
            )
        if self._requests >= caps.max_requests:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_requests} API requests ({self._used()})"
            )
        if committed_output + max_tokens > caps.max_output_tokens:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_output_tokens:,} output tokens: the next "
                f"request could generate up to {max_tokens:,} on top of {committed_output:,} already "
                f"billed or reserved by requests in flight ({self._used()})"
            )
        if committed_input + input_bound > caps.max_input_tokens:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_input_tokens:,} input tokens: the next "
                f"request could take up to {input_bound:,} on top of {committed_input:,} already billed "
                f"or reserved by requests in flight ({self._used()})"
            )
        return None

    def _used(self) -> str:
        cut_off = (
            f", up to {self._cut_off_input:,} input and {self._cut_off_output:,} output more possibly "
            f"billed by {quantity(self._cut_off_requests, 'request')} cut off"
            if self._cut_off_requests
            else ""
        )
        return (
            f"{quantity(self._requests, 'request')} used, {self._billed_input:,} input and "
            f"{self._billed_output:,} output tokens billed{cut_off}"
        )


def preview_line(batch_payload_bytes: Sequence[int], entries: int, caps: Caps) -> str:
    batches = len(batch_payload_bytes)
    expected_cost = cost_usd(
        sum(_expected_input_tokens(size) for size in batch_payload_bytes),
        _EXPECTED_OUTPUT_TOKENS_PER_ENTRY * entries,
    )
    return (
        f"classifying {quantity(entries, 'entry', 'entries')} in {quantity(batches, 'batch', 'batches')} "
        f"with {MODEL} ({EFFORT} effort): expect {quantity(batches, 'request')}, "
        f"{_about_usd(expected_cost)}; hard caps {quantity(caps.max_requests, 'request')}, "
        f"{caps.max_output_tokens:,} output and {caps.max_input_tokens:,} input tokens, "
        f"{caps.time_limit_s:.0f} s ({hard_limit_s():.0f} s wall clock): at most "
        f"{at_most_usd(caps.max_cost_usd())}"
    )


def usage_line(usage: Usage) -> str:
    possibly = (
        f", plus up to {usage.possibly_billed_input_tokens:,} input and "
        f"{usage.possibly_billed_output_tokens:,} output tokens possibly billed by "
        f"{quantity(usage.possibly_billed_requests, 'request')} cut off or abandoned in flight"
        if usage.possibly_billed_requests
        else ""
    )
    return (
        f"classification used {quantity(usage.requests, 'request')}: {usage.input_tokens:,} input and "
        f"{usage.output_tokens:,} output tokens{possibly}, {_about_usd(usage.cost_usd())}"
    )
