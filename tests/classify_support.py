"""Fakes and helpers for the classification tests (no network, no real API).

A fake client plays a script: one item per `messages.create` call, in call order. The
same script items work for both engines' clients: `ScriptedClient` (sync, for the
threaded reference) and `AsyncScriptedClient` (async, for the asyncio engine).
"""

import asyncio
import json
import signal
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx

from asio_doc_tools import classify
from asio_doc_tools.classify import budget, prompt, store
from asio_doc_tools.history import Entry
from asio_doc_tools.versions import Version

ENGINES = classify.ENGINES


# ---------------------------------------------------------------------------
# replies
# ---------------------------------------------------------------------------


def usage(input_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def text_response(items: list[dict], *, stop_reason: str = "end_turn", usage_=None) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps({"items": items}))],
        usage=usage_ or usage(),
        stop_details=None,
    )


def refusal_response() -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="refusal",
        content=[],
        usage=usage(),
        stop_details=SimpleNamespace(category="cyber", explanation="looked risky"),
    )


def max_tokens_response(output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="max_tokens", content=[], usage=usage(output_tokens=output_tokens), stop_details=None
    )


def good_item(id_: str, *, category: str = "fixed", breaking: bool = False, reason: str = "") -> dict:
    return {"id": id_, "category": category, "breaking": breaking, "breaking_reason": reason}


def good_items(count: int) -> list[dict]:
    return [good_item(f"e{i}") for i in range(count)]


def good_response(count: int = 1, **kwargs: Any) -> SimpleNamespace:
    return text_response(good_items(count), **kwargs)


def invalid_response() -> SimpleNamespace:
    return text_response([good_item("not-an-input-id")])


_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

_STATUS_ERRORS = {
    400: anthropic.BadRequestError,
    401: anthropic.AuthenticationError,
    403: anthropic.PermissionDeniedError,
    404: anthropic.NotFoundError,
    429: anthropic.RateLimitError,
}


def status_error(status: int, headers: dict[str, str] | None = None) -> anthropic.APIStatusError:
    response = httpx.Response(status, headers=headers or {}, request=_REQUEST)
    default_type = anthropic.InternalServerError if status >= 500 else anthropic.APIStatusError
    return _STATUS_ERRORS.get(status, default_type)(f"Error code: {status}", response=response, body=None)


def rate_limited() -> anthropic.APIStatusError:
    return status_error(429, {"retry-after-ms": "1"})


def connection_error(cause: Exception) -> anthropic.APIConnectionError:
    error = anthropic.APIConnectionError(request=_REQUEST)
    error.__cause__ = cause
    return error


def timeout_error(cause: Exception) -> anthropic.APITimeoutError:
    error = anthropic.APITimeoutError(request=_REQUEST)
    error.__cause__ = cause
    return error


# ---------------------------------------------------------------------------
# script items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Raise:
    """The call raises `error`."""

    error: BaseException


@dataclass(frozen=True)
class Echo:
    """A valid reply for exactly the entries the request sent."""

    usage_: SimpleNamespace | None = None


@dataclass(frozen=True)
class Delay:
    """The call takes `seconds` (sleeping, or awaiting), then plays `then`."""

    seconds: float
    then: Any


@dataclass(frozen=True)
class WaitForCalls:
    """The call waits until `count` calls have started (so they are all in flight at once),
    then plays `then`. Engine-neutral: it polls, sleeping or awaiting."""

    count: int
    then: Any


_WAIT_FOR_CALLS_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Hook:
    """The call runs `action()` (on the calling thread), then plays `then`."""

    action: Callable[[], Any]
    then: Any


def echo_reply(kwargs: dict, usage_: SimpleNamespace | None = None) -> SimpleNamespace:
    ids = [item["id"] for item in json.loads(kwargs["messages"][0]["content"])["items"]]
    return text_response(
        [good_item(id_) for id_ in ids], usage_=usage_ or usage(2_100 + 50 * len(ids), 32 * len(ids))
    )


class _Script:
    def __init__(self, script: Iterable[Any]) -> None:
        self._script = list(script)
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    def next_item(self, kwargs: dict) -> Any:
        with self._lock:
            self.calls.append(kwargs)
            if not self._script:
                raise AssertionError("fake client received more requests than its script has replies")
            return self._script.pop(0)


class ScriptedMessages(_Script):
    def create(self, **kwargs: Any) -> Any:
        return self._play(self.next_item(kwargs), kwargs)

    def _play(self, item: Any, kwargs: dict) -> Any:
        match item:
            case Raise(error=error):
                raise error
            case Echo(usage_=usage_):
                return echo_reply(kwargs, usage_)
            case Delay(seconds=seconds, then=then):
                time.sleep(seconds)
                return self._play(then, kwargs)
            case WaitForCalls(count=count, then=then):
                deadline = time.monotonic() + _WAIT_FOR_CALLS_TIMEOUT_S
                while len(self.calls) < count:
                    assert time.monotonic() < deadline, f"only {len(self.calls)} of {count} calls started"
                    time.sleep(0.001)
                return self._play(then, kwargs)
            case Hook(action=action, then=then):
                action()
                return self._play(then, kwargs)
            case _ if callable(item):
                return self._play(item(), kwargs)
        return item


class AsyncScriptedMessages(_Script):
    """Also counts calls in progress and calls cancelled, to check cancellation."""

    def __init__(self, script: Iterable[Any]) -> None:
        super().__init__(script)
        self.in_progress = 0
        self.cancelled = 0

    async def create(self, **kwargs: Any) -> Any:
        item = self.next_item(kwargs)
        self.in_progress += 1
        try:
            return await self._play(item, kwargs)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_progress -= 1

    async def _play(self, item: Any, kwargs: dict) -> Any:
        match item:
            case Raise(error=error):
                raise error
            case Echo(usage_=usage_):
                return echo_reply(kwargs, usage_)
            case Delay(seconds=seconds, then=then):
                await asyncio.sleep(seconds)
                return await self._play(then, kwargs)
            case WaitForCalls(count=count, then=then):
                deadline = time.monotonic() + _WAIT_FOR_CALLS_TIMEOUT_S
                while len(self.calls) < count:
                    assert time.monotonic() < deadline, f"only {len(self.calls)} of {count} calls started"
                    await asyncio.sleep(0.001)
                return await self._play(then, kwargs)
            case Hook(action=action, then=then):
                result = action()
                if asyncio.iscoroutine(result):
                    await result
                return await self._play(then, kwargs)
            case _ if callable(item):
                return await self._play(item(), kwargs)
        return item


class ScriptedClient:
    def __init__(self, script: Iterable[Any]) -> None:
        self.messages = ScriptedMessages(script)

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls


class AsyncScriptedClient:
    def __init__(self, script: Iterable[Any]) -> None:
        self.messages = AsyncScriptedMessages(script)
        self.entered = 0
        self.closed = 0

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls

    async def __aenter__(self) -> "AsyncScriptedClient":
        self.entered += 1
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.closed += 1


def scripted_client(engine: str, script: Iterable[Any]) -> ScriptedClient | AsyncScriptedClient:
    match engine:
        case "asyncio":
            return AsyncScriptedClient(script)
        case "threads":
            return ScriptedClient(script)
    raise AssertionError(engine)


# ---------------------------------------------------------------------------
# items, runs, and the store
# ---------------------------------------------------------------------------


def entry(text: str) -> Entry:
    return Entry(html=text, text=text, children=())


def item(text: str, version: str = "1.38.0") -> classify.ClassifyItem:
    return classify.ClassifyItem(release=Version.parse(version), entry=entry(text))


def items(count: int, start: int = 0) -> list[classify.ClassifyItem]:
    return [item(f"Entry number {i}.") for i in range(start, start + count)]


def store_path(tmp_path) -> Any:
    return tmp_path / "classifications.sqlite3"


def keys_of(classify_items: Iterable[classify.ClassifyItem]) -> set[str]:
    return {classify.cache_key(it.entry) for it in classify_items}


@dataclass
class Engine:
    """Runs classify() with one engine and a scripted fake client of the right kind."""

    name: str
    clients: list[Any] = field(default_factory=list)

    def client(self, script: Iterable[Any]) -> Any:
        client = scripted_client(self.name, script)
        self.clients.append(client)
        return client

    def classify(self, classify_items: Any, client: Any, path: Any) -> tuple[classify.Classification, ...]:
        return classify.classify(
            classify_items, engine=self.name, client_factory=lambda: client, store_path=path
        )

    def run(self, classify_items: Any, script: Iterable[Any], path: Any) -> tuple[Any, Any]:
        """(the result, or the exception raised; the client), for tests that check both."""
        client = self.client(script)
        try:
            return self.classify(classify_items, client, path), client
        except (Exception, KeyboardInterrupt) as e:
            return e, client


def store_row(key: str, classification: store.Classification, text: str) -> store.StoreRow:
    return store.StoreRow(
        key=key, classification=classification, prompt_version=prompt.PROMPT_VERSION, text=text
    )


def rows(count: int, start: int = 0) -> tuple[store.StoreRow, ...]:
    result = []
    for i in range(start, start + count):
        text = f"Pending entry {i}."
        result.append(
            store.StoreRow(
                key=prompt.key_for_text(text),
                classification=store.Classification(
                    category=prompt.Category.FIXED,
                    breaking=False,
                    breaking_reason="",
                    model=prompt.MODEL,
                    effort=prompt.EFFORT,
                    release="1.38.0",
                    classified_at=1_700_000_000.0,
                ),
                prompt_version=prompt.PROMPT_VERSION,
                text=text,
            )
        )
    return tuple(result)


def import_pending(path: Any) -> None:
    conn = store.connect(path)
    try:
        store.import_pending(conn, path)
    finally:
        conn.close()


def files(directory: Any) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def interrupt_main_thread() -> None:
    """Delivers a real SIGINT to the main thread (what Ctrl-C does), waking it from a
    blocking wait or the event loop's selector."""
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)


def one_batch_caps() -> budget.Caps:
    return budget.caps_for_run(40, (budget.REQUEST_INPUT_BASE + 20_000,))
