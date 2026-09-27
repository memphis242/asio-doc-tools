"""The budget, the retry rules, and the batch protocol, driven directly (no engine, no time)."""

from typing import Any

import classify_support as cs
import httpx
import pytest

from asio_doc_tools.classify import batch, budget, prompt, retry, run
from asio_doc_tools.diag import AsioDocsError


def drive(protocol: Any, script: list[Any]) -> tuple[Any, list[Any]]:
    """Runs a batch-protocol generator against `script` (replies, or exceptions to throw
    into it, one per Call); returns its result and every effect it yielded."""
    effects: list[Any] = []
    try:
        effect = next(protocol)
        while True:
            effects.append(effect)
            match effect:
                case batch.Call():
                    reply = script.pop(0)
                    effect = (
                        protocol.throw(reply) if isinstance(reply, BaseException) else protocol.send(reply)
                    )
                case batch.Sleep():
                    effect = protocol.send(None)
    except StopIteration as finished:
        return finished.value, effects


def calls(effects: list[Any]) -> list[batch.Call]:
    return [effect for effect in effects if isinstance(effect, batch.Call)]


def run_budget(caps: budget.Caps | None = None, clock: Any = None) -> budget.RunBudget:
    stop = budget.StopFlag()
    return budget.RunBudget(caps or cs.one_batch_caps(), stop, **({"clock": clock} if clock else {}))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# one batch
# ---------------------------------------------------------------------------


def test_usage_accounting_includes_split_and_resent_responses() -> None:
    # One batch of 2: the whole batch hits max_tokens (split), the first half's reply is
    # invalid (resent), and both halves then succeed. Every response's usage counts.
    texts = {"left-key": "Fixed the left entry.", "right-key": "Fixed the right entry."}
    b = run_budget()
    outcome, effects = drive(
        batch.classify_batch(b, ["left-key", "right-key"], texts),
        [
            cs.max_tokens_response(),  # usage: 10 / 5
            cs.text_response([cs.good_item("e0"), cs.good_item("e0")], usage_=cs.usage(20, 7)),  # invalid
            cs.text_response([cs.good_item("e0")], usage_=cs.usage(11, 3)),  # resend succeeds
            cs.text_response([cs.good_item("e0")], usage_=cs.usage(13, 4)),  # second half succeeds
        ],
    )
    assert set(outcome.classified) == {"left-key", "right-key"}
    assert outcome.failure is None and not b.stop.is_set()
    assert [call.kwargs["max_tokens"] for call in calls(effects)] == [
        2048 + 256,
        2048 + 128,
        2048 + 128,
        2048 + 128,
    ]
    usage = b.usage()
    assert usage.requests == 4
    assert (usage.input_tokens, usage.output_tokens) == (10 + 20 + 11 + 13, 5 + 7 + 3 + 4)
    assert usage.in_flight_requests == 0


def test_the_worst_case_batch_makes_exactly_15_attempts() -> None:
    # Every logical send retried twice, the whole batch hitting max_tokens, and both halves
    # resent after an invalid reply: 5 logical sends of 3 attempts.
    caps = budget.Caps(max_requests=100, max_output_tokens=10**7, max_input_tokens=10**8, time_limit_s=60.0)
    b = run_budget(caps)
    texts = {f"k{i}": f"Entry {i}." for i in range(4)}
    rl = cs.rate_limited
    outcome, effects = drive(
        batch.classify_batch(b, list(texts), texts),
        [rl(), rl(), cs.max_tokens_response()]
        + [rl(), rl(), cs.invalid_response(), rl(), rl(), cs.good_response(2)] * 2,
    )
    assert len(outcome.classified) == 4
    assert len(calls(effects)) == 15
    assert b.usage().requests == 15


def test_a_failed_first_half_means_the_second_half_is_never_sent() -> None:
    b = run_budget()
    texts = {"a": "Fixed a.", "b": "Fixed b."}
    outcome, effects = drive(
        batch.classify_batch(b, ["a", "b"], texts), [cs.max_tokens_response(), cs.refusal_response()]
    )
    assert outcome.classified == {}
    assert len(calls(effects)) == 2
    assert "refused" in str(b.stop.reason)


def test_an_unexpected_exception_keeps_what_was_classified_before_it() -> None:
    b = run_budget()
    texts = {"a": "Fixed a.", "b": "Fixed b."}
    outcome, _ = drive(
        batch.classify_batch(b, ["a", "b"], texts),
        [cs.max_tokens_response(), cs.good_response(1), RuntimeError("x")],
    )
    assert set(outcome.classified) == {"a"}
    assert isinstance(outcome.failure, RuntimeError)
    assert b.usage().cut_off_requests == 1  # the request the exception interrupted may have been billed


def test_a_cancellation_keeps_what_was_classified_before_it_without_a_failure() -> None:
    b = run_budget()
    texts = {"a": "Fixed a.", "b": "Fixed b."}
    outcome, _ = drive(
        batch.classify_batch(b, ["a", "b"], texts),
        [cs.max_tokens_response(), cs.good_response(1), batch.Cancelled()],
    )
    assert set(outcome.classified) == {"a"} and outcome.failure is None
    usage = b.usage()
    assert (usage.cut_off_requests, usage.in_flight_requests) == (1, 0)  # kept as possibly billed


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------


def send(b: budget.RunBudget, script: list[Any]) -> tuple[Any, list[Any]]:
    """Drives one logical send; its result is the response, or the exception it raised."""
    protocol = batch._send(b, "{}", 1)
    try:
        return drive(protocol, script)
    except (AsioDocsError, budget.RunStopped) as e:
        return e, []


def test_retryable_errors_are_retried_up_to_3_attempts_each_counted() -> None:
    b = run_budget()
    result, _ = send(b, [cs.rate_limited() for _ in range(5)])
    assert (
        isinstance(result, AsioDocsError)
        and "rate limit" in str(result)
        and "after 3 attempts" in str(result)
    )
    assert b.usage().requests == 3


def test_a_retry_that_succeeds_returns_its_response() -> None:
    b = run_budget()
    response, effects = send(b, [cs.status_error(529), cs.good_response()])
    assert response.stop_reason == "end_turn"
    assert [type(effect) for effect in effects] == [batch.Call, batch.Sleep, batch.Call]
    assert b.usage().requests == 2


@pytest.mark.parametrize(
    "error",
    [
        cs.status_error(400),
        cs.status_error(401),
        cs.status_error(403),
        cs.status_error(404),
        cs.status_error(500, {"x-should-retry": "false"}),
    ],
    ids=["bad-request", "authentication", "permission", "not-found", "server-says-no-retry"],
)
def test_non_retryable_errors_are_not_retried(error) -> None:
    b = run_budget()
    result, _ = send(b, [error, cs.good_response()])
    assert isinstance(result, AsioDocsError)
    assert b.usage().requests == 1


@pytest.mark.parametrize(
    ("error", "possibly_billed"),
    [
        (cs.status_error(429), False),
        (cs.status_error(500), False),
        (cs.status_error(529), False),
        (cs.connection_error(httpx.ConnectError("connection refused")), False),
        (cs.timeout_error(httpx.ConnectTimeout("connect timed out")), False),
        (cs.timeout_error(httpx.ReadTimeout("read timed out")), True),
        (cs.connection_error(httpx.RemoteProtocolError("peer closed the connection mid-response")), True),
    ],
    ids=["429", "500", "529", "connect-refused", "connect-timeout", "read-timeout", "dropped-mid-response"],
)
def test_a_failed_attempt_refunds_or_keeps_its_reservation(error, possibly_billed) -> None:
    b = run_budget()
    send(b, [error, cs.text_response([cs.good_item("e0")], usage_=cs.usage(10, 5))])
    usage = b.usage()
    assert usage.requests == 2
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)
    assert (usage.in_flight_requests, usage.reserved_input_tokens, usage.reserved_output_tokens) == (0, 0, 0)
    assert usage.cut_off_requests == (1 if possibly_billed else 0)
    assert usage.cut_off_output_tokens == (budget.max_tokens_for(1) if possibly_billed else 0)
    assert usage.cut_off_input_tokens == (budget.input_bound("{}") if possibly_billed else 0)


def test_retry_delay_honors_retry_after_capped_at_30_s() -> None:
    assert retry.retry_delay_s(cs.status_error(429, {"retry-after-ms": "1500"}), 1) == 1.5
    assert retry.retry_delay_s(cs.status_error(429, {"retry-after": "7"}), 1) == 7.0
    assert retry.retry_delay_s(cs.status_error(429, {"retry-after": "600"}), 1) == 30.0
    assert retry.retry_delay_s(cs.status_error(429, {"retry-after": "soon"}), 1) == 2.0
    assert retry.retry_delay_s(cs.status_error(503), 2) == 4.0
    assert retry.retry_delay_s(cs.timeout_error(httpx.ReadTimeout("t")), 1) == 2.0


def test_a_retry_wait_never_lasts_past_the_time_limit() -> None:
    clock = FakeClock()
    b = run_budget(
        budget.Caps(max_requests=10, max_output_tokens=10**6, max_input_tokens=10**7, time_limit_s=5.0), clock
    )
    protocol = batch._send(b, "{}", 1)
    assert isinstance(next(protocol), batch.Call)
    clock.now += 4.0
    sleep = protocol.throw(cs.status_error(429, {"retry-after": "30"}))
    assert isinstance(sleep, batch.Sleep) and sleep.seconds == pytest.approx(1.0)
    clock.now += sleep.seconds
    with pytest.raises(budget.RunStopped):
        protocol.send(None)
    assert "time limit" in str(b.stop.reason)


# ---------------------------------------------------------------------------
# the run budget
# ---------------------------------------------------------------------------


def test_budget_refuses_attempts_after_the_time_limit() -> None:
    clock = FakeClock()
    b = run_budget(clock=clock)
    b.settle_billed(b.admit(100, 1_000), input_tokens=1, output_tokens=1)
    clock.now += budget.RUN_TIME_LIMIT_S
    with pytest.raises(budget.RunStopped):
        b.admit(100, 1_000)
    assert "time limit" in str(b.stop.reason)
    assert b.usage().requests == 1


def test_budget_output_cap_counts_reservations_of_requests_in_flight() -> None:
    b = run_budget(
        budget.Caps(max_requests=10, max_output_tokens=1_000, max_input_tokens=10**7, time_limit_s=60.0)
    )
    first = b.admit(600, 1_000)
    with pytest.raises(budget.RunStopped):
        b.admit(600, 1_000)  # 600 reserved + 600 > 1,000, although nothing is billed yet
    assert "output tokens" in str(b.stop.reason)
    b.settle_billed(first, input_tokens=10, output_tokens=40)
    assert b.usage() == budget.Usage(
        requests=1,
        input_tokens=10,
        output_tokens=40,
        cut_off_requests=0,
        cut_off_input_tokens=0,
        cut_off_output_tokens=0,
        in_flight_requests=0,
        reserved_input_tokens=0,
        reserved_output_tokens=0,
    )


def test_budget_input_cap_counts_reservations_of_requests_in_flight() -> None:
    b = run_budget(
        budget.Caps(max_requests=10, max_output_tokens=10**6, max_input_tokens=1_000, time_limit_s=60.0)
    )
    first = b.admit(10, 600)
    with pytest.raises(budget.RunStopped):
        b.admit(10, 600)
    assert "input tokens" in str(b.stop.reason)
    b.settle_billed(first, input_tokens=450, output_tokens=5)
    assert b.usage().input_tokens == 450


@pytest.mark.parametrize("kind", ["input", "output"])
def test_usage_over_a_reservation_sets_the_stop_flag_with_both_numbers(kind) -> None:
    b = run_budget()
    reservation = b.admit(100, 1_000)
    b.settle_billed(
        reservation,
        input_tokens=5_000 if kind == "input" else 10,
        output_tokens=500 if kind == "output" else 5,
    )
    assert f"billed {'5,000' if kind == 'input' else '500'} {kind} tokens" in str(b.stop.reason)
    assert f"bound was {'1,000' if kind == 'input' else '100'}" in str(b.stop.reason)


def test_budget_refuses_everything_once_the_stop_flag_is_set() -> None:
    b = run_budget()
    b.stop.trip(AsioDocsError("first reason"))
    b.stop.trip(AsioDocsError("second reason"))
    with pytest.raises(budget.RunStopped):
        b.admit(100, 1_000)
    assert str(b.stop.reason) == "first reason"
    assert b.usage().requests == 0


def test_stop_flag_listeners_run_once_when_it_is_first_set() -> None:
    stop = budget.StopFlag()
    heard: list[str] = []
    stop.on_trip(lambda: heard.append("before"))
    stop.trip(AsioDocsError("a"))
    stop.trip(AsioDocsError("b"))
    stop.on_trip(lambda: heard.append("after"))  # already set: called at once
    assert heard == ["before", "after"]


def test_the_request_input_bound_covers_the_actual_payload() -> None:
    ids, payload = prompt.batch_payload(["k"], {"k": "Fixed a bug."})
    assert ids == ("e0",)
    assert budget.input_bound(payload) == 3_624 + 447 + 1_024 + len(payload.encode())


def test_worst_case_of_the_full_history_is_under_the_ceiling() -> None:
    # 995 entries in 25 batches, every one as large as the largest real one (14,760 bytes).
    caps = budget.caps_for_run(995, (budget.REQUEST_INPUT_BASE + 14_760,) * 25)
    assert (caps.max_requests, caps.max_output_tokens) == (54, 328_064)
    assert caps.max_input_tokens == 54 * (5_095 + 14_760)
    assert caps.max_cost_usd() < budget.MAX_RUN_COST_USD == 6.0
    assert budget.MAX_IN_FLIGHT == 32


def test_the_usage_line_counts_abandoned_reservations_as_possibly_billed() -> None:
    usage = budget.Usage(
        requests=3,
        input_tokens=5_000,
        output_tokens=1_000,
        cut_off_requests=0,
        cut_off_input_tokens=0,
        cut_off_output_tokens=0,
        in_flight_requests=1,
        reserved_input_tokens=20_000,
        reserved_output_tokens=7_168,
    )
    line = budget.usage_line(usage)
    assert "20,000 input and 7,168 output tokens possibly billed" in line
    assert usage.cost_usd() == ((5_000 + 20_000) * 2 + (1_000 + 7_168) * 10) / 1_000_000


def test_the_usage_report_prints_again_when_an_interrupt_cut_it_short(monkeypatch) -> None:
    report = run.UsageReport(run_budget())
    printed: list[str] = []

    def spend_interrupted_once(message: str) -> None:
        printed.append(message)
        if len(printed) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(run, "spend", spend_interrupted_once)
    with pytest.raises(KeyboardInterrupt):
        report()
    report()
    report()
    assert len(printed) == 2  # the interrupted print, then the one that got through, once
