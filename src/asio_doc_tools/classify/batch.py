"""Classifying one batch: its requests, retries, split, and resend, written once for both
engines as a generator that does no I/O itself.

`classify_batch` yields an effect whenever it needs the outside world: `Call` (send this
request and send me the response, or throw me the exception it raised) or `Sleep` (wait
this long before a retry, less if the run is stopping, and send me None). Each engine
drives it its own way: the threaded reference with a blocking `messages.create` on a
worker thread, the asyncio engine with `await` on the event loop. So how a batch is
classified, budgeted, retried, and failed is the same in both, and the engines differ
only in how they wait, schedule, cancel, and stop.
"""

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ..diag import AsioDocsError
from . import budget, retry
from .budget import RunBudget, RunStopped
from .prompt import RawResult, batch_payload, check_refusal, diagnose_reply, request_kwargs, validate_reply


@dataclass(frozen=True, slots=True)
class Call:
    """Send one request: `client.messages.create(**kwargs)`, with the budget's reservation
    for it already made."""

    kwargs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Sleep:
    """Wait `seconds` before a retry, or less if the run's stop flag is set meanwhile."""

    seconds: float


Effect = Call | Sleep


class Cancelled(BaseException):
    """Thrown into a batch's generator when the asyncio engine cancels its task (at the hard
    limit, or on a second Ctrl-C). The generator settles the request in flight as possibly
    billed and returns what the batch classified before it, without counting a failure."""


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """What a batch (or part of one) classified, and the unexpected exception, if any,
    that ended it. Expected failures are the stop flag's reason instead."""

    classified: Mapping[str, RawResult]
    failure: BaseException | None = None


_NOTHING: Final = BatchOutcome({})

BatchProtocol = Generator[Effect, Any, BatchOutcome]


def classify_batch(
    run_budget: RunBudget, batch_keys: Sequence[str], texts: Mapping[str, str]
) -> BatchProtocol:
    """Classifies one batch, returning whatever it classified.

    Never raises. An expected failure (an API error, a refusal, an invalid reply twice, a
    budget refusal) sets the run's stop flag with the failure as the reason; anything else
    sets it too and comes back as the outcome's `failure`, for the engine to raise once
    everything paid for is recorded. Either way the outcome keeps what was classified
    before the failure (a split's first half).
    """
    assert batch_keys
    return (yield from _classify_part(run_budget, batch_keys, texts, allow_split=True))


def _classify_part(
    run_budget: RunBudget, keys: Sequence[str], texts: Mapping[str, str], *, allow_split: bool
) -> BatchProtocol:
    try:
        return (yield from _classify_keys(run_budget, keys, texts, allow_split=allow_split))
    except GeneratorExit:
        raise
    except (RunStopped, Cancelled):
        return _NOTHING
    except AsioDocsError as e:
        run_budget.stop.trip(e)
        return _NOTHING
    except BaseException as e:
        run_budget.stop.trip(AsioDocsError(f"a classification worker failed unexpectedly: {e!r}"))
        return BatchOutcome({}, failure=e)


def _classify_keys(
    run_budget: RunBudget, keys: Sequence[str], texts: Mapping[str, str], *, allow_split: bool
) -> BatchProtocol:
    """Classifies `keys` with one request, resent once after an invalid reply. When the
    reply hits max_tokens and `allow_split`, classifies the two halves instead, which do
    not split again (see the "Run budget" comment in budget.py)."""
    assert keys
    ids, payload = batch_payload(keys, texts)
    max_tokens = budget.max_tokens_for(len(keys))

    response = yield from _send(run_budget, payload, len(keys))
    if response.stop_reason == "max_tokens":
        if len(keys) == 1:
            raise AsioDocsError(
                f"the model hit its max_tokens limit ({max_tokens:,} output tokens) classifying a single "
                f"entry; entry text: {texts[keys[0]]!r}"
            )
        if not allow_split:
            raise AsioDocsError(
                f"the model hit its max_tokens limit ({max_tokens:,} output tokens) on half of a batch "
                f"({len(keys)} entries) after the whole batch hit it too"
            )
        mid = len(keys) // 2
        # Each half keeps its own result whatever happens to the other. After a failed
        # first half the stop flag is set, so the second half sends nothing.
        first = yield from _classify_part(run_budget, keys[:mid], texts, allow_split=False)
        second = yield from _classify_part(run_budget, keys[mid:], texts, allow_split=False)
        return BatchOutcome(
            {**first.classified, **second.classified},
            failure=first.failure if first.failure is not None else second.failure,
        )

    check_refusal(response)
    parsed = validate_reply(response, ids)
    if parsed is None:
        resent = yield from _send(run_budget, payload, len(keys))
        if resent.stop_reason == "max_tokens":
            raise AsioDocsError(
                f"the model hit its max_tokens limit ({max_tokens:,} output tokens) resending a request "
                f"of {len(keys)} entries after an invalid reply"
            )
        check_refusal(resent)
        parsed = validate_reply(resent, ids)
        if parsed is None:
            raise AsioDocsError(
                f"the model returned an invalid classification reply twice for a request of "
                f"{len(keys)} entries ({diagnose_reply(resent, ids)})"
            )
    return BatchOutcome({key: parsed[id_] for key, id_ in zip(keys, ids, strict=True)})


def _send(run_budget: RunBudget, payload: str, entry_count: int) -> Generator[Effect, Any, Any]:
    """One logical request: up to retry.MAX_ATTEMPTS HTTP attempts, each admitted by the
    budget, which reserves its max_tokens and its input bound first.

    Raises RunStopped when the budget refuses an attempt, and AsioDocsError when a failure
    is not retryable or outlasts the retries.
    """
    import anthropic

    max_tokens = budget.max_tokens_for(entry_count)
    input_bound = budget.input_bound(payload)
    for attempt in range(1, retry.MAX_ATTEMPTS + 1):
        reservation = run_budget.admit(max_tokens, input_bound)
        try:
            response = yield Call(request_kwargs(max_tokens, payload))
        except anthropic.APIError as e:
            failure = retry.api_failure(e)
            run_budget.settle_failed(reservation, possibly_billed=failure.possibly_billed)
            if not failure.retryable or attempt == retry.MAX_ATTEMPTS:
                attempts = f" (after {attempt} attempts)" if attempt > 1 else ""
                raise AsioDocsError(f"{failure.message}{attempts}") from e
            # Ends early when the run is stopping, and never lasts past its time limit;
            # the next admit() then refuses.
            yield Sleep(min(retry.retry_delay_s(e, attempt), run_budget.seconds_left()))
            continue
        except BaseException:
            # Not an API failure (a cancellation, an interrupt, or a bug): the request may
            # have been billed.
            run_budget.settle_failed(reservation, possibly_billed=True)
            raise
        usage = getattr(response, "usage", None)
        if usage is None:
            run_budget.settle_failed(reservation, possibly_billed=True)
        else:
            run_budget.settle_billed(
                reservation, input_tokens=retry.billed_input_tokens(usage), output_tokens=usage.output_tokens
            )
        return response
    raise AssertionError("unreachable: the last attempt returns or raises")
