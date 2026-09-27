"""How a failed request attempt is handled: whether it is retried, whether it may have
been billed, and how long to wait before the retry. Both engines use exactly these rules.

A logical send is at most MAX_ATTEMPTS HTTP attempts. Only a 408, 409, 429, or 5xx
status (529 included), a connection error, or a timeout is retried, and not when the
server answers `x-should-retry: false`. The wait is 2 s then 4 s, or the server's
retry-after, never more than 30 s. The SDK's own retries are off.
"""

import math
from dataclasses import dataclass
from typing import Any, Final

from .prompt import MODEL

MAX_ATTEMPTS: Final = 3
_RETRY_BASE_DELAY_S: Final = 2.0
_RETRY_MAX_DELAY_S: Final = 30.0
# Retried as well: every 5xx status (529, overloaded, included).
_RETRYABLE_STATUSES: Final = frozenset({408, 409, 429})

assert MAX_ATTEMPTS == 3 and _RETRY_BASE_DELAY_S * 2 ** (MAX_ATTEMPTS - 2) <= _RETRY_MAX_DELAY_S


@dataclass(frozen=True, slots=True)
class ApiFailure:
    """How to handle the exception the SDK raised for one attempt."""

    message: str
    retryable: bool
    possibly_billed: bool  # the attempt may have generated (and been billed for) output


def request_id(error: Exception) -> str:
    return getattr(error, "request_id", None) or "no request id"


def _cause_suffix(error: BaseException) -> str:
    return f" ({error.__cause__!r})" if error.__cause__ is not None else ""


def _status_is_retryable(error: Any) -> bool:
    # The server can say outright that retrying will not help.
    if error.response.headers.get("x-should-retry") == "false":
        return False
    return error.status_code in _RETRYABLE_STATUSES or error.status_code >= 500


def _status_message(error: Any) -> str:
    rid = request_id(error)
    match error.status_code:
        case 400:
            return f"Anthropic API rejected the request ({rid}): {error}"
        case 401:
            return (
                f"Anthropic API authentication failed ({rid}); set ANTHROPIC_API_KEY or run 'ant auth login'"
            )
        case 403:
            return f"Anthropic API denied permission ({rid}): {error}"
        case 404:
            return f"Anthropic API could not find model {MODEL!r} ({rid})"
        case 429:
            return f"Anthropic API rate limit exceeded ({rid}): {error}"
        case 529:
            return f"the Anthropic API is overloaded ({rid}): {error}"
        case status:
            return f"Anthropic API error (status {status}, request {rid}): {error}"


def api_failure(error: Exception) -> ApiFailure:
    """How to handle `error`, an exception the SDK (sync or async) raised for one attempt."""
    import anthropic
    import httpx

    match error:
        case anthropic.APITimeoutError():
            # Without a connection (or a free one in the pool) nothing was sent.
            never_sent = isinstance(error.__cause__, (httpx.ConnectTimeout, httpx.PoolTimeout))
            return ApiFailure(
                f"an Anthropic API request timed out{_cause_suffix(error)}",
                retryable=True,
                possibly_billed=not never_sent,
            )
        case anthropic.APIConnectionError():
            never_sent = isinstance(error.__cause__, httpx.ConnectError)
            return ApiFailure(
                f"could not reach the Anthropic API{_cause_suffix(error)}",
                retryable=True,
                possibly_billed=not never_sent,
            )
        case anthropic.APIStatusError():
            # An error status means the API rejected or abandoned the request: nothing billed.
            return ApiFailure(
                _status_message(error), retryable=_status_is_retryable(error), possibly_billed=False
            )
        case _:
            return ApiFailure(
                f"unexpected Anthropic API error: {error!r}", retryable=False, possibly_billed=True
            )


def retry_after_s(error: Exception) -> float | None:
    """The wait the server asked for before a retry, if it sent a usable one."""
    response = getattr(error, "response", None)
    if response is None:
        return None
    for header, seconds_per_unit in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = response.headers.get(header)
        if value is None:
            continue
        try:
            seconds = float(value) * seconds_per_unit
        except ValueError:
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def retry_delay_s(error: Exception, failed_attempt: int) -> float:
    """How long to wait after failed attempt number `failed_attempt` (from 1) of a send."""
    assert 1 <= failed_attempt < MAX_ATTEMPTS
    requested = retry_after_s(error)
    delay = requested if requested is not None else _RETRY_BASE_DELAY_S * 2 ** (failed_attempt - 1)
    return min(delay, _RETRY_MAX_DELAY_S)


def billed_input_tokens(usage: Any) -> int:
    # Cache reads and writes are billed input too, though requests here never ask for caching.
    cached = (getattr(usage, "cache_creation_input_tokens", None) or 0) + (
        getattr(usage, "cache_read_input_tokens", None) or 0
    )
    return int(usage.input_tokens) + int(cached)
