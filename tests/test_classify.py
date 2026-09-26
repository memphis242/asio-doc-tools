import json
import os
import signal
import sqlite3
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import anthropic  # imported at collection time: classify imports it lazily on first use

# (over a second), which would otherwise dwarf the small, deliberate delays below.
import httpx
import pytest

from asio_doc_tools import classify, diag
from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.history import Entry
from asio_doc_tools.versions import Version


def _usage(input_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def _text_response(items: list[dict], *, stop_reason: str = "end_turn", usage=None) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps({"items": items}))],
        usage=usage or _usage(),
        stop_details=None,
    )


def _refusal_response() -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="refusal",
        content=[],
        usage=_usage(),
        stop_details=SimpleNamespace(category="cyber", explanation="looked risky"),
    )


def _max_tokens_response(output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="max_tokens", content=[], usage=_usage(output_tokens=output_tokens), stop_details=None
    )


def _blocking(delay_s: float, response: SimpleNamespace):
    """A queueable callable that sleeps before returning `response`."""

    def make() -> SimpleNamespace:
        time.sleep(delay_s)
        return response

    return make


def _raising(exc: BaseException):
    def make() -> SimpleNamespace:
        raise exc

    return make


def _good_item(id_: str, *, category: str = "fixed", breaking: bool = False, reason: str = "") -> dict:
    return {"id": id_, "category": category, "breaking": breaking, "breaking_reason": reason}


def _good_items(count: int) -> list[dict]:
    return [_good_item(f"e{i}") for i in range(count)]


def _invalid_response() -> SimpleNamespace:
    return _text_response([_good_item("not-an-input-id")])


class FakeMessages:
    """Each queued item is either a response, or a zero-arg callable that produces one
    (or raises) - the latter lets a test block a call or raise from it. Calls are served
    in the order they arrive, guarded by a lock since several worker threads may call
    `create()` concurrently.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **kwargs) -> SimpleNamespace:
        with self._lock:
            self.calls.append(kwargs)
            if not self._responses:
                raise AssertionError("fake client received more requests than responses were queued")
            next_item = self._responses.pop(0)
        return next_item() if callable(next_item) else next_item


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.messages = FakeMessages(responses)

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls


class EchoMessages:
    """Answers every request validly, with the observed normal usage: about 2,100 + 50
    input tokens per entry and 32 output tokens per entry."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **kwargs) -> SimpleNamespace:
        with self._lock:
            self.calls.append(kwargs)
        ids = [item["id"] for item in json.loads(kwargs["messages"][0]["content"])["items"]]
        return _text_response(
            [_good_item(id_) for id_ in ids], usage=_usage(2_100 + 50 * len(ids), 32 * len(ids))
        )


def _entry(text: str) -> Entry:
    return Entry(html=text, text=text, children=())


def _item(text: str, version: str = "1.38.0") -> classify.ClassifyItem:
    return classify.ClassifyItem(release=Version.parse(version), entry=_entry(text))


def _items(count: int) -> list[classify.ClassifyItem]:
    return [_item(f"Entry number {i}.") for i in range(count)]


def _store_path(tmp_path):
    return tmp_path / "classifications.sqlite3"


def _store_row(key: str, classification: classify.Classification, text: str) -> classify._StoreRow:
    return classify._StoreRow(
        key=key, classification=classification, prompt_version=classify.PROMPT_VERSION, text=text
    )


_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

_STATUS_ERRORS = {
    400: anthropic.BadRequestError,
    401: anthropic.AuthenticationError,
    403: anthropic.PermissionDeniedError,
    404: anthropic.NotFoundError,
    429: anthropic.RateLimitError,
}


def _status_error(status: int, headers: dict[str, str] | None = None) -> anthropic.APIStatusError:
    response = httpx.Response(status, headers=headers or {}, request=_REQUEST)
    default_type = anthropic.InternalServerError if status >= 500 else anthropic.APIStatusError
    return _STATUS_ERRORS.get(status, default_type)(f"Error code: {status}", response=response, body=None)


def _rate_limited() -> anthropic.APIStatusError:
    return _status_error(429, {"retry-after-ms": "1"})


def _connection_error(cause: Exception) -> anthropic.APIConnectionError:
    error = anthropic.APIConnectionError(request=_REQUEST)
    error.__cause__ = cause
    return error


def _timeout_error(cause: Exception) -> anthropic.APITimeoutError:
    error = anthropic.APITimeoutError(request=_REQUEST)
    error.__cause__ = cause
    return error


_ONE_BATCH_CAPS = classify._caps_for_run(40, (classify._REQUEST_INPUT_BASE + 20_000,))


def _ctx(client, *, caps: classify._Caps | None = None, clock=time.monotonic) -> classify._RunContext:
    stop = classify._StopFlag()
    budget = classify._RunBudget(caps or _ONE_BATCH_CAPS, stop, clock=clock)
    return classify._RunContext(client=client, budget=budget)


@pytest.fixture
def fast_retries(monkeypatch) -> None:
    monkeypatch.setattr(classify, "_RETRY_BASE_DELAY_S", 0.001)


def _interrupt_main_thread() -> None:
    """Delivers a real SIGINT to the main thread (what Ctrl-C does), waking it from a
    blocking wait."""
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)


class _Exited(BaseException):
    """Raised by the fake exit function in place of ending the process (a BaseException,
    like SystemExit, so the run's own error handling lets it through)."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def _fake_exit(status: int):
    raise _Exited(status)


# ---------------------------------------------------------------------------
# request shape and the client
# ---------------------------------------------------------------------------


def test_request_shape_has_no_model_choice_or_fallback(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    client = FakeClient([_text_response([_good_item("e0")])])
    classify.classify([_item("Fixed a bug.")], client_factory=lambda: client, store_path=store_path)
    kwargs = client.calls[0]
    assert kwargs["model"] == classify.MODEL == "claude-sonnet-5"
    assert kwargs["output_config"]["effort"] == classify.EFFORT == "low"
    assert "betas" not in kwargs
    assert "extra_body" not in kwargs
    assert "cache_control" not in kwargs
    assert "cache_control" not in json.dumps(kwargs)


def test_max_tokens_scales_with_the_requests_entry_count(tmp_path) -> None:
    client = FakeClient([_text_response(_good_items(3))])
    classify.classify(_items(3), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert client.calls[0]["max_tokens"] == 2048 + 128 * 3
    assert classify._max_tokens_for(classify._BATCH_SIZE) == 7_168
    assert classify._max_tokens_for(classify._BATCH_SIZE) <= classify._SDK_NONSTREAMING_MAX_TOKENS


def test_default_client_disables_sdk_retries_and_sets_explicit_timeouts(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    client = classify.default_client_factory()
    assert client.max_retries == 0
    assert client.timeout.read == classify._REQUEST_TIMEOUT_S == 240.0
    assert client.timeout.connect == classify._CONNECT_TIMEOUT_S == 10.0


def test_missing_credentials_are_reported_before_any_spend(tmp_path, monkeypatch, capsys) -> None:
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either
    with pytest.raises(AsioDocsError, match="ANTHROPIC_API_KEY"):
        classify.classify([_item("Fixed a bug.")], store_path=_store_path(tmp_path))
    assert capsys.readouterr().err == ""  # no preview: nothing was about to be sent


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def test_store_key_is_stable_and_a_cache_hit_avoids_the_api(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Fixed a bug in ip::tcp.")
    key = classify.cache_key(item.entry)

    client = FakeClient([_text_response([_good_item("e0")])])
    result = classify.classify([item], client_factory=lambda: client, store_path=store_path)
    assert result[0].category == classify.Category.FIXED
    assert len(client.calls) == 1
    assert key in classify.stored_keys(store_path)

    # A second call for the exact same entry must not touch the API at all.
    client2 = FakeClient([])
    result2 = classify.classify([item], client_factory=lambda: client2, store_path=store_path)
    assert result2[0].category == classify.Category.FIXED
    assert client2.calls == []


def test_upsert_over_an_existing_row_keeps_one_row(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Added a new overload.")
    classify.classify(
        [item],
        client_factory=lambda: FakeClient([_text_response([_good_item("e0", category="fixed")])]),
        store_path=store_path,
    )
    key = classify.cache_key(item.entry)
    replacement = classify.Classification(
        category=classify.Category.ADDED,
        breaking=False,
        breaking_reason="",
        model=classify.MODEL,
        effort=classify.EFFORT,
        release="1.38.0",
        classified_at=time.time(),
    )
    conn = classify._connect(store_path)
    try:
        classify._upsert_classifications(
            conn, [_store_row(key, replacement, item.entry.full_text())], store_path
        )
        rows = conn.execute("SELECT category FROM classifications WHERE key = ?", (key,)).fetchall()
    finally:
        conn.close()
    assert rows == [("added",)]


def test_breaking_entry_carries_its_reason(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Removed the deprecated io_service typedef.")
    client = FakeClient(
        [_text_response([_good_item("e0", category="changed", breaking=True, reason="io_service is gone")])]
    )
    result = classify.classify([item], client_factory=lambda: client, store_path=store_path)
    assert result[0].breaking is True
    assert result[0].breaking_reason == "io_service is gone"


def test_corrupt_store_is_reported_not_silently_discarded(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    store_path.write_bytes(b"not a sqlite database")
    with pytest.raises(AsioDocsError, match="corrupt"):
        classify.classify([_item("x")], client_factory=lambda: FakeClient([]), store_path=store_path)


def test_schema_version_newer_than_understood_is_an_error(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    conn = sqlite3.connect(store_path)
    conn.execute("PRAGMA user_version = 999")
    conn.commit()
    conn.close()
    with pytest.raises(AsioDocsError, match="schema version"):
        classify.classify([_item("x")], client_factory=lambda: FakeClient([]), store_path=store_path)
    with pytest.raises(AsioDocsError, match="schema version"):
        classify.stored_keys(store_path)


def test_identical_text_across_items_is_classified_once(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    items = [_item("Fixed the same bug.", "1.38.0"), _item("Fixed the same bug.", "1.37.0")]
    client = FakeClient([_text_response([_good_item("e0")])])
    result = classify.classify(items, client_factory=lambda: client, store_path=store_path)
    assert len(client.calls) == 1
    assert len(result) == 2
    assert result[0].category == result[1].category == classify.Category.FIXED


def test_stored_row_is_queryable_by_its_own_text_column(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Fixed a race in the reactor.")
    client = FakeClient([_text_response([_good_item("e0")])])
    classify.classify([item], client_factory=lambda: client, store_path=store_path)

    conn = sqlite3.connect(store_path)
    try:
        row = conn.execute(
            "SELECT text, model, effort, prompt_version FROM classifications WHERE key = ?",
            (classify.cache_key(item.entry),),
        ).fetchone()
    finally:
        conn.close()
    assert row == (item.entry.full_text(), classify.MODEL, classify.EFFORT, classify.PROMPT_VERSION)


def test_database_error_at_read_time_is_reported(tmp_path) -> None:
    class _RaisingConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.DatabaseError("simulated corruption detected mid-read")

    with pytest.raises(AsioDocsError, match="corrupt"):
        classify._load_classifications(_RaisingConnection(), ["some-key"], tmp_path / "store.sqlite3")


def test_database_error_at_write_time_is_reported_with_the_store_path(tmp_path) -> None:
    class _RaisingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def executemany(self, *args, **kwargs):
            raise sqlite3.DatabaseError("simulated corruption detected mid-write")

    store_path = tmp_path / "store.sqlite3"
    classification = classify.Classification(
        category=classify.Category.FIXED,
        breaking=False,
        breaking_reason="",
        model=classify.MODEL,
        effort=classify.EFFORT,
        release="1.0.0",
        classified_at=0.0,
    )
    with pytest.raises(AsioDocsError, match="corrupt") as excinfo:
        classify._upsert_classifications(
            _RaisingConnection(), [_store_row("key", classification, "text")], store_path
        )
    assert str(store_path) in str(excinfo.value)


def test_stored_keys_on_a_missing_store_creates_nothing(tmp_path) -> None:
    store_path = tmp_path / "not-yet" / "classifications.sqlite3"
    assert classify.stored_keys(store_path) == frozenset()
    assert not store_path.parent.exists()


def test_stored_keys_on_an_empty_store_file_leaves_it_untouched(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    store_path.write_bytes(b"")
    assert classify.stored_keys(store_path) == frozenset()
    assert store_path.read_bytes() == b""


def test_stored_keys_on_a_corrupt_store_is_an_error_and_keeps_the_file(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    junk = b"not a sqlite database, but paid-for data may be in here"
    store_path.write_bytes(junk)
    with pytest.raises(AsioDocsError, match="corrupt"):
        classify.stored_keys(store_path)
    assert store_path.read_bytes() == junk


# ---------------------------------------------------------------------------
# replies: splitting, resending, refusals
# ---------------------------------------------------------------------------


def test_max_tokens_splits_the_batch_in_half(tmp_path) -> None:
    responses = [
        _max_tokens_response(),  # full batch of 4
        _text_response(_good_items(2)),  # first half
        _text_response(_good_items(2)),  # second half
    ]
    client = FakeClient(responses)
    result = classify.classify(_items(4), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(result) == 4
    assert all(c.category == classify.Category.FIXED for c in result)
    assert [call["max_tokens"] for call in client.calls] == [2048 + 128 * 4, 2048 + 128 * 2, 2048 + 128 * 2]


def test_single_item_batch_hitting_max_tokens_is_an_error(tmp_path) -> None:
    client = FakeClient([_max_tokens_response()])
    with pytest.raises(AsioDocsError, match="max_tokens"):
        classify.classify(
            [_item("A very long entry.")], client_factory=lambda: client, store_path=_store_path(tmp_path)
        )
    assert len(client.calls) == 1


def test_an_all_max_tokens_batch_splits_once_then_stops_the_run(tmp_path, monkeypatch) -> None:
    # Three full batches, one worker: the first batch hits max_tokens whole and again on
    # its first half, which stops the run: the second half and the other batches are
    # never sent, so the run makes exactly 2 requests.
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    batch = classify._BATCH_SIZE
    client = FakeClient(
        [
            _max_tokens_response(output_tokens=classify._max_tokens_for(batch)),
            _max_tokens_response(output_tokens=classify._max_tokens_for(batch // 2)),
        ]
    )
    with pytest.raises(AsioDocsError, match="max_tokens"):
        classify.classify(_items(3 * batch), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(client.calls) == 2
    assert classify.stored_keys(_store_path(tmp_path)) == frozenset()


def _worst_case_batch_responses() -> list:
    """The most requests one batch of 4 can make: every logical send retried twice, the
    whole batch hitting max_tokens, and both halves resent after an invalid reply."""
    return [
        _raising(_rate_limited()), _raising(_rate_limited()), _max_tokens_response(),
        _raising(_rate_limited()), _raising(_rate_limited()), _invalid_response(),
        _raising(_rate_limited()), _raising(_rate_limited()), _text_response(_good_items(2)),
        _raising(_rate_limited()), _raising(_rate_limited()), _invalid_response(),
        _raising(_rate_limited()), _raising(_rate_limited()), _text_response(_good_items(2)),
    ]  # fmt: skip


def test_one_batch_makes_at_most_15_attempts(tmp_path, monkeypatch, fast_retries) -> None:
    # With the run's request cap out of the way, the per-batch structure alone bounds a
    # batch to 5 logical sends of at most 3 attempts each.
    monkeypatch.setattr(classify, "_SPARE_REQUESTS_CAP", 100)
    client = FakeClient(_worst_case_batch_responses())
    result = classify.classify(_items(4), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(result) == 4
    assert len(client.calls) == 15


def test_the_run_request_cap_binds_before_the_per_batch_maximum(tmp_path, fast_retries) -> None:
    # The same worst case under the default caps: a one-batch run may make R = 2 * 1 + 4.
    client = FakeClient(_worst_case_batch_responses())
    with pytest.raises(AsioDocsError, match=r"hard cap of 6 API requests") as excinfo:
        classify.classify(_items(4), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(client.calls) == 6
    assert "Rerunning continues" in str(excinfo.value)


def test_a_single_batch_absorbs_a_split_and_a_resend_under_the_default_caps(tmp_path) -> None:
    # A 28-entry run (one batch) whose whole-batch reply hits max_tokens at full cost, and
    # whose first half then needs a resend: all within the default caps.
    responses = [
        _max_tokens_response(output_tokens=classify._max_tokens_for(28)),
        _invalid_response(),
        _text_response(_good_items(14), usage=_usage(900, 450)),
        _text_response(_good_items(14), usage=_usage(900, 450)),
    ]
    client = FakeClient(responses)
    result = classify.classify(_items(28), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(result) == 28
    assert len(client.calls) == 4


def test_invalid_reply_is_retried_once_then_succeeds(tmp_path) -> None:
    responses = [
        _text_response([_good_item("e0"), _good_item("e0")]),  # duplicate id: invalid
        _text_response([_good_item("e0", category="added")]),  # resend succeeds
    ]
    client = FakeClient(responses)
    result = classify.classify(
        [_item("Some entry.")], client_factory=lambda: client, store_path=_store_path(tmp_path)
    )
    assert result[0].category == classify.Category.ADDED
    assert len(client.calls) == 2


def test_invalid_reply_twice_is_an_error(tmp_path) -> None:
    client = FakeClient([_invalid_response(), _invalid_response()])
    with pytest.raises(AsioDocsError, match="invalid"):
        classify.classify(
            [_item("Some entry.")], client_factory=lambda: client, store_path=_store_path(tmp_path)
        )
    assert len(client.calls) == 2


def test_refusal_is_an_error(tmp_path) -> None:
    client = FakeClient([_refusal_response()])
    with pytest.raises(AsioDocsError, match="refused"):
        classify.classify(
            [_item("Some entry.")], client_factory=lambda: client, store_path=_store_path(tmp_path)
        )


def test_split_batch_stores_the_successful_half_when_the_other_half_fails(tmp_path) -> None:
    left = _item("Fixed the left entry.")
    right = _item("Fixed the right entry.")
    store_path = _store_path(tmp_path)
    responses = [
        _max_tokens_response(),  # the full batch of 2
        _text_response([_good_item("e0", category="fixed")]),  # first half succeeds
        _refusal_response(),  # second half is refused
    ]
    client = FakeClient(responses)

    with pytest.raises(AsioDocsError, match="refused"):
        classify.classify([left, right], client_factory=lambda: client, store_path=store_path)

    stored = classify.stored_keys(store_path)
    assert classify.cache_key(left.entry) in stored
    assert classify.cache_key(right.entry) not in stored


def test_a_failed_first_half_means_the_second_half_is_never_sent(tmp_path) -> None:
    client = FakeClient([_max_tokens_response(), _refusal_response()])
    with pytest.raises(AsioDocsError, match="refused"):
        classify.classify(_items(2), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(client.calls) == 2


def test_usage_accounting_includes_split_and_resent_responses() -> None:
    # One batch of 2: the whole batch hits max_tokens (split), the first half's reply is
    # invalid (resent), and both halves then succeed. Every response's usage counts.
    texts = {"left-key": "Fixed the left entry.", "right-key": "Fixed the right entry."}
    responses = [
        _max_tokens_response(),  # usage: 10 / 5
        _text_response([_good_item("e0"), _good_item("e0")], usage=_usage(20, 7)),  # invalid: duplicate id
        _text_response([_good_item("e0")], usage=_usage(11, 3)),  # resend succeeds
        _text_response([_good_item("e0")], usage=_usage(13, 4)),  # second half succeeds
    ]
    ctx = _ctx(FakeClient(responses))

    outcome = classify._classify_batch(ctx, ["left-key", "right-key"], texts)

    assert set(outcome.classified) == {"left-key", "right-key"}
    assert outcome.failure is None
    assert not ctx.stop.is_set()
    usage = ctx.budget.usage()
    assert usage.requests == 4
    assert usage.input_tokens == 10 + 20 + 11 + 13
    assert usage.output_tokens == 5 + 7 + 3 + 4
    assert usage.reserved_output_tokens == 0


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------


def test_retryable_errors_are_retried_up_to_3_attempts_each_counted(fast_retries) -> None:
    client = FakeClient([_raising(_rate_limited()) for _ in range(5)])
    ctx = _ctx(client)
    with pytest.raises(AsioDocsError, match="rate limit.*after 3 attempts"):
        classify._send(ctx, "{}", 1)
    assert len(client.calls) == 3
    assert ctx.budget.usage().requests == 3


def test_a_retry_that_succeeds_returns_its_response(fast_retries) -> None:
    client = FakeClient([_raising(_status_error(529)), _text_response([_good_item("e0")])])
    ctx = _ctx(client)
    response = classify._send(ctx, "{}", 1)
    assert response.stop_reason == "end_turn"
    assert len(client.calls) == 2
    assert ctx.budget.usage().requests == 2


@pytest.mark.parametrize(
    "error",
    [
        _status_error(400),
        _status_error(401),
        _status_error(403),
        _status_error(404),
        _status_error(500, {"x-should-retry": "false"}),
    ],
    ids=["bad-request", "authentication", "permission", "not-found", "server-says-no-retry"],
)
def test_non_retryable_errors_are_not_retried(error, fast_retries) -> None:
    client = FakeClient([_raising(error), _text_response([_good_item("e0")])])
    with pytest.raises(AsioDocsError):
        classify._send(_ctx(client), "{}", 1)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("error", "possibly_billed"),
    [
        (_status_error(429), False),
        (_status_error(500), False),
        (_status_error(529), False),
        (_connection_error(httpx.ConnectError("connection refused")), False),
        (_timeout_error(httpx.ConnectTimeout("connect timed out")), False),
        (_timeout_error(httpx.ReadTimeout("read timed out")), True),
        (_connection_error(httpx.RemoteProtocolError("peer closed the connection mid-response")), True),
    ],
    ids=["429", "500", "529", "connect-refused", "connect-timeout", "read-timeout", "dropped-mid-response"],
)
def test_a_failed_attempt_refunds_or_keeps_its_reservation(error, possibly_billed, fast_retries) -> None:
    client = FakeClient([_raising(error), _text_response([_good_item("e0")], usage=_usage(10, 5))])
    ctx = _ctx(client)
    classify._send(ctx, "{}", 1)
    usage = ctx.budget.usage()
    assert usage.requests == 2
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)
    assert (usage.in_flight_requests, usage.reserved_input_tokens, usage.reserved_output_tokens) == (0, 0, 0)
    assert usage.cut_off_requests == (1 if possibly_billed else 0)
    assert usage.cut_off_output_tokens == (classify._max_tokens_for(1) if possibly_billed else 0)
    assert usage.cut_off_input_tokens == (classify._input_bound("{}") if possibly_billed else 0)


def test_retry_delay_honors_retry_after_capped_at_30_s() -> None:
    assert classify._retry_delay_s(_status_error(429, {"retry-after-ms": "1500"}), 1) == 1.5
    assert classify._retry_delay_s(_status_error(429, {"retry-after": "7"}), 1) == 7.0
    assert classify._retry_delay_s(_status_error(429, {"retry-after": "600"}), 1) == 30.0
    assert classify._retry_delay_s(_status_error(429, {"retry-after": "soon"}), 1) == 2.0
    assert classify._retry_delay_s(_status_error(503), 2) == 4.0
    assert classify._retry_delay_s(_timeout_error(httpx.ReadTimeout("t")), 1) == 2.0


def test_a_stop_during_retry_backoff_ends_the_send_without_another_request() -> None:
    ctx_holder: list[classify._RunContext] = []

    def rate_limited_then_stopped():
        # The stop arrives 20 ms into the 30 s backoff that follows this failure.
        threading.Timer(0.02, lambda: ctx_holder[0].stop.trip(AsioDocsError("stopped by the test"))).start()
        raise _status_error(429, {"retry-after": "30"})

    client = FakeClient([rate_limited_then_stopped, _text_response([_good_item("e0")])])
    ctx_holder.append(_ctx(client))
    started = time.monotonic()
    with pytest.raises(classify._RunStopped):
        classify._send(ctx_holder[0], "{}", 1)
    assert time.monotonic() - started < 5.0
    assert len(client.calls) == 1


def test_retry_backoff_never_waits_past_the_time_limit() -> None:
    caps = classify._Caps(
        max_requests=10, max_output_tokens=100_000, max_input_tokens=10**7, time_limit_s=0.05
    )
    client = FakeClient(
        [_raising(_status_error(429, {"retry-after": "30"})), _text_response([_good_item("e0")])]
    )
    ctx = _ctx(client, caps=caps)
    started = time.monotonic()
    with pytest.raises(classify._RunStopped):
        classify._send(ctx, "{}", 1)
    assert time.monotonic() - started < 1.0
    assert len(client.calls) == 1
    assert "time limit" in str(ctx.stop.reason)


# ---------------------------------------------------------------------------
# the run budget
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_budget_refuses_attempts_after_the_time_limit() -> None:
    clock = _FakeClock()
    stop = classify._StopFlag()
    budget = classify._RunBudget(_ONE_BATCH_CAPS, stop, clock=clock)
    budget.settle_billed(budget.admit(100, 1_000), input_tokens=1, output_tokens=1)
    clock.now += classify._RUN_TIME_LIMIT_S
    with pytest.raises(classify._RunStopped):
        budget.admit(100, 1_000)
    assert stop.is_set()
    assert "time limit" in str(stop.reason)
    assert budget.usage().requests == 1


def test_budget_output_cap_counts_reservations_of_requests_in_flight() -> None:
    stop = classify._StopFlag()
    budget = classify._RunBudget(
        classify._Caps(max_requests=10, max_output_tokens=1_000, max_input_tokens=10**7, time_limit_s=60.0),
        stop,
    )
    first = budget.admit(600, 1_000)
    with pytest.raises(classify._RunStopped):
        budget.admit(600, 1_000)  # 600 reserved + 600 > 1,000, although nothing is billed yet
    assert "output tokens" in str(stop.reason)
    budget.settle_billed(first, input_tokens=10, output_tokens=40)
    assert budget.usage() == classify._Usage(
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


def test_budget_refuses_everything_once_the_stop_flag_is_set() -> None:
    stop = classify._StopFlag()
    budget = classify._RunBudget(_ONE_BATCH_CAPS, stop)
    stop.trip(AsioDocsError("first reason"))
    stop.trip(AsioDocsError("second reason"))
    with pytest.raises(classify._RunStopped):
        budget.admit(100, 1_000)
    assert str(stop.reason) == "first reason"
    assert budget.usage().requests == 0


def test_worst_case_of_the_full_history_is_under_the_ceiling() -> None:
    # 995 entries in 25 batches, every one as large as the largest real one (14,760 bytes).
    caps = classify._caps_for_run(995, (classify._REQUEST_INPUT_BASE + 14_760,) * 25)
    assert (caps.max_requests, caps.max_output_tokens) == (54, 206_208)
    assert caps.max_input_tokens == 54 * (5_095 + 14_760)
    assert caps.max_cost_usd() < classify._MAX_RUN_COST_USD == 5.0


def test_a_normal_full_history_run_stays_within_its_caps(tmp_path) -> None:
    messages = EchoMessages()
    result = classify.classify(
        _items(995),
        client_factory=lambda: SimpleNamespace(messages=messages),
        store_path=_store_path(tmp_path),
    )
    assert len(result) == 995
    assert len(messages.calls) == 25


def test_request_cap_refusal_ends_the_run_and_keeps_stored_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    monkeypatch.setattr(classify, "_REQUESTS_PER_BATCH_CAP", 1)
    monkeypatch.setattr(classify, "_SPARE_REQUESTS_CAP", 0)
    store_path = _store_path(tmp_path)
    items = _items(3)
    client = FakeClient([_invalid_response(), _text_response(_good_items(1)), _text_response(_good_items(1))])

    with pytest.raises(AsioDocsError, match=r"hard cap of 3 API requests") as excinfo:
        classify.classify(items, client_factory=lambda: client, store_path=store_path)

    assert len(client.calls) == 3
    assert classify.stored_keys(store_path) == {classify.cache_key(item.entry) for item in items[:2]}
    message = str(excinfo.value)
    assert "2 of 3 entries" in message
    assert "Rerunning continues" in message

    # Rerunning sends only what is missing.
    client2 = FakeClient([_text_response(_good_items(1))])
    classify.classify(items, client_factory=lambda: client2, store_path=store_path)
    assert len(client2.calls) == 1


def test_output_cap_refusal_ends_the_run(tmp_path, monkeypatch) -> None:
    # Caps leaving room for exactly one request's max_tokens at a time.
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    monkeypatch.setattr(classify, "_SPARE_FULL_SIZE_OUTPUTS_CAP", 0)
    monkeypatch.setattr(classify, "_OUTPUT_TOKENS_PER_ENTRY_CAP", 0)
    store_path = _store_path(tmp_path)
    client = FakeClient([_text_response(_good_items(1)), _text_response(_good_items(1))])
    with pytest.raises(AsioDocsError, match="output tokens"):
        classify.classify(_items(2), client_factory=lambda: client, store_path=store_path)
    assert len(client.calls) == 1
    assert len(classify.stored_keys(store_path)) == 1


# ---------------------------------------------------------------------------
# stopping
# ---------------------------------------------------------------------------


def _two_concurrent_calls(first_result, second_result, *, second_delay_s: float = 0.05, on_both_started=None):
    """Two queue items guaranteed to be inside create() at the same time: the first waits
    until the second has started, runs `on_both_started`, then produces `first_result`;
    the second produces `second_result` after `second_delay_s`."""
    second_started = threading.Event()

    def first():
        assert second_started.wait(timeout=5.0)
        if on_both_started is not None:
            on_both_started()
        return first_result() if callable(first_result) else first_result

    def second():
        second_started.set()
        time.sleep(second_delay_s)
        return second_result

    return [first, second]


def test_after_a_batch_error_only_in_flight_requests_finish_and_are_stored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 2)
    store_path = _store_path(tmp_path)
    client = FakeClient(
        _two_concurrent_calls(_refusal_response(), _text_response(_good_items(1)))
        + [_text_response(_good_items(1)) for _ in range(4)]
    )
    started = time.monotonic()
    with pytest.raises(AsioDocsError, match="refused") as excinfo:
        classify.classify(_items(6), client_factory=lambda: client, store_path=store_path)
    assert time.monotonic() - started < 3.0
    assert len(client.calls) == 2  # nothing was sent after the refusal
    assert len(classify.stored_keys(store_path)) == 1  # the in-flight request's result was kept
    assert "1 of 6 entries" in str(excinfo.value)


def test_a_worker_interrupt_does_not_wait_for_queued_batches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    store_path = _store_path(tmp_path)
    items = _items(4)
    block_delay = 0.3
    responses = [
        _text_response([_good_item("e0")]),  # batch 0: succeeds and is stored
        _raising(KeyboardInterrupt()),  # batch 1: interrupted
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 2: never sent
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 3: never sent
    ]
    client = FakeClient(responses)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        classify.classify(items, client_factory=lambda: client, store_path=store_path)
    assert time.monotonic() - started < block_delay
    assert len(client.calls) == 2
    assert classify.cache_key(items[0].entry) in classify.stored_keys(store_path)


def test_ctrl_c_keeps_in_flight_results_and_sends_nothing_more(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 2)
    store_path = _store_path(tmp_path)
    client = FakeClient(
        _two_concurrent_calls(
            _blocking(0.05, _text_response(_good_items(1))),
            _text_response(_good_items(1)),
            second_delay_s=0.1,
            on_both_started=_interrupt_main_thread,
        )
        + [_text_response(_good_items(1)) for _ in range(3)]
    )
    with pytest.raises(KeyboardInterrupt):
        classify.classify(_items(5), client_factory=lambda: client, store_path=store_path)
    assert len(client.calls) == 2
    assert len(classify.stored_keys(store_path)) == 2  # both in-flight results were kept
    err = capsys.readouterr().err
    assert "Ctrl-C again" in err
    assert "used 2 requests" in err


def _waiting(harvest, *, stop=None, watchdog_in_s=10.0, report_usage=lambda: None) -> classify._Waiting:
    if stop is None:
        stop = classify._StopFlag()
        stop.trip(AsioDocsError("interrupted"))
    return classify._Waiting(
        harvest=harvest,
        stop=stop,
        watchdog_at=time.monotonic() + watchdog_in_s,
        clock=time.monotonic,
        progress=lambda: "progress",
        report_usage=report_usage,
        exit_process=_fake_exit,
    )


def test_first_ctrl_c_cancels_queued_batches_and_harvests_in_flight_ones(capsys) -> None:
    done: Future = Future()
    done.set_result(classify._BatchOutcome({}))
    in_flight: Future = Future()
    assert in_flight.set_running_or_notify_cancel()
    queued: Future = Future()
    threading.Timer(0.02, lambda: in_flight.set_result(classify._BatchOutcome({}))).start()
    harvested: list[Future] = []

    error = _waiting(
        harvested.append, report_usage=lambda: pytest.fail("usage is reported by the caller on this path")
    ).stop_and_harvest([done, in_flight, queued], interrupted=True)

    assert error is None
    assert queued.cancelled()
    assert harvested == [done, in_flight]
    assert "Ctrl-C again" in capsys.readouterr().err


def test_a_failing_harvest_does_not_stop_the_others_and_the_first_error_is_kept() -> None:
    futures: list[Future] = []
    for _ in range(3):
        future: Future = Future()
        future.set_result(classify._BatchOutcome({}))
        futures.append(future)
    harvested: list[Future] = []

    def harvest(future: Future) -> None:
        harvested.append(future)
        if future is not futures[2]:
            raise ValueError(f"harvest {futures.index(future)} failed")

    error = _waiting(harvest).stop_and_harvest(futures, interrupted=False)

    assert harvested == futures
    assert str(error) == "harvest 0 failed"


def test_second_ctrl_c_abandons_in_flight_requests_and_exits_at_once(capsys) -> None:
    done: Future = Future()
    done.set_result(classify._BatchOutcome({}))
    in_flight: Future = Future()
    assert in_flight.set_running_or_notify_cancel()
    harvested: list[Future] = []
    reported: list[bool] = []
    threading.Timer(0.02, _interrupt_main_thread).start()
    try:
        with pytest.raises(_Exited) as excinfo:
            _waiting(harvested.append, report_usage=lambda: reported.append(True)).stop_and_harvest(
                [done, in_flight], interrupted=True
            )
    finally:
        in_flight.set_result(classify._BatchOutcome({}))
    assert excinfo.value.status == 130
    assert reported == [True]
    # What finished is stored (harvest is idempotent per future); what is in flight is abandoned.
    assert list(dict.fromkeys(harvested)) == [done]
    assert "abandoned 1 in-flight request" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# spend reporting
# ---------------------------------------------------------------------------


def test_preview_and_usage_are_printed_even_when_quiet(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(diag, "_quiet", True)
    client = FakeClient([_text_response([_good_item("e0")])])
    classify.classify(
        [_item("Fixed a bug.")], client_factory=lambda: client, store_path=_store_path(tmp_path)
    )
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "hard caps" in lines[0]
    assert "used 1 request" in lines[1]


def test_usage_is_printed_when_the_run_fails(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(diag, "_quiet", True)
    client = FakeClient([_refusal_response()])
    with pytest.raises(AsioDocsError):
        classify.classify(
            [_item("Some entry.")], client_factory=lambda: client, store_path=_store_path(tmp_path)
        )
    assert "used 1 request" in capsys.readouterr().err


def test_a_fully_stored_run_spends_and_prints_nothing_when_quiet(tmp_path, monkeypatch, capsys) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Fixed a bug.")
    classify.classify(
        [item], client_factory=lambda: FakeClient([_text_response([_good_item("e0")])]), store_path=store_path
    )
    capsys.readouterr()
    monkeypatch.setattr(diag, "_quiet", True)
    classify.classify(
        [item], client_factory=lambda: pytest.fail("no client is needed"), store_path=store_path
    )
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# unexpected worker failures
# ---------------------------------------------------------------------------


def test_every_paid_result_is_stored_before_the_first_worker_failure_is_raised(tmp_path, monkeypatch) -> None:
    # Three requests in flight at once; each finishes only after the main thread has
    # harvested the one before, so "first" is well defined: two raise, the last one is a
    # billed reply.
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 3)
    harvested = threading.Semaphore(0)
    real_harvest = classify._Harvester.__call__

    def counting_harvest(self, future):
        real_harvest(self, future)
        harvested.release()

    monkeypatch.setattr(classify._Harvester, "__call__", counting_harvest)
    all_started = threading.Barrier(3)
    turn = [threading.Event() for _ in range(3)]
    turn[0].set()

    def call(index: int, result):
        def run():
            all_started.wait(timeout=5.0)
            assert turn[index].wait(timeout=5.0)
            if index + 1 < len(turn):
                threading.Thread(
                    target=lambda: harvested.acquire(timeout=5.0) and turn[index + 1].set()
                ).start()
            if isinstance(result, BaseException):
                raise result
            return result

        return run

    results = [RuntimeError("first failure"), RuntimeError("second failure"), _text_response(_good_items(1))]
    store_path = _store_path(tmp_path)
    client = FakeClient([call(i, result) for i, result in enumerate(results)])
    with pytest.raises(RuntimeError, match="first failure"):
        classify.classify(_items(3), client_factory=lambda: client, store_path=store_path)
    assert len(client.calls) == 3
    assert len(classify.stored_keys(store_path)) == 1  # the billed reply that finished last


def test_a_split_keeps_its_first_half_when_the_second_half_raises(tmp_path) -> None:
    first, second = _item("Fixed the first entry."), _item("Fixed the second entry.")
    store_path = _store_path(tmp_path)
    client = FakeClient(
        [_max_tokens_response(), _text_response(_good_items(1)), _raising(RuntimeError("second half"))]
    )
    with pytest.raises(RuntimeError, match="second half"):
        classify.classify([first, second], client_factory=lambda: client, store_path=store_path)
    assert classify.stored_keys(store_path) == {classify.cache_key(first.entry)}


# ---------------------------------------------------------------------------
# a store that cannot be written
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to read-only files")
def test_a_read_only_store_is_refused_before_anything_is_sent(tmp_path, capsys) -> None:
    store_path = _store_path(tmp_path)
    classify._connect(store_path).close()
    store_path.chmod(0o444)
    try:
        client = FakeClient([])
        with pytest.raises(AsioDocsError, match="not writable") as excinfo:
            classify.classify(_items(8), client_factory=lambda: client, store_path=store_path)
    finally:
        store_path.chmod(0o644)
    assert str(store_path) in str(excinfo.value)
    assert client.calls == []
    assert capsys.readouterr().err == ""  # no spend preview either


def test_results_the_store_rejects_mid_run_are_saved_and_imported_by_the_next_run(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    store_path = _store_path(tmp_path)
    pending_path = classify._pending_path(store_path)
    items = _items(3)
    good_upsert = classify._UPSERT_SQL
    monkeypatch.setattr(
        classify, "_UPSERT_SQL", "INSERT INTO no_such_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    # Each request takes 50 ms, so the second is in flight when the first one's write fails.
    client = FakeClient([_blocking(0.05, _text_response(_good_items(1))) for _ in range(3)])
    with pytest.raises(AsioDocsError, match="could not write") as excinfo:
        classify.classify(items, client_factory=lambda: client, store_path=store_path)
    assert len(client.calls) == 2  # nothing was sent after the failed write
    assert str(pending_path) in str(excinfo.value)
    assert len(pending_path.read_text().splitlines()) == 2
    assert classify.stored_keys(store_path) == frozenset()

    # The next run imports both, and pays only for the entry that was never classified.
    monkeypatch.setattr(classify, "_UPSERT_SQL", good_upsert)
    client2 = FakeClient([_text_response(_good_items(1))])
    classify.classify(items, client_factory=lambda: client2, store_path=store_path)
    assert len(client2.calls) == 1
    assert len(classify.stored_keys(store_path)) == 3
    assert not pending_path.exists()


def test_results_that_cannot_be_saved_anywhere_are_printed(tmp_path, monkeypatch, capsys) -> None:
    store_path = _store_path(tmp_path)

    def disk_full(path, rows):
        raise OSError(28, "No space left on device", str(path))

    monkeypatch.setattr(classify, "_append_pending", disk_full)
    monkeypatch.setattr(
        classify, "_UPSERT_SQL", "INSERT INTO no_such_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    item = _item("Fixed a bug that must not be lost.")
    client = FakeClient([_text_response([_good_item("e0", category="added")])])
    with pytest.raises(AsioDocsError, match="printed above") as excinfo:
        classify.classify([item], client_factory=lambda: client, store_path=store_path)
    assert "could not be saved anywhere" in str(excinfo.value)
    printed = [line for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert len(printed) == 1
    row = classify._parse_pending_line(printed[0])  # the printed line is importable as is
    assert row.key == classify.cache_key(item.entry)
    assert row.classification.category == classify.Category.ADDED


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[]",
        '{"key": "k"}',
        "{broken",
    ],
    ids=["not-json", "not-an-object", "missing-fields", "truncated"],
)
def test_a_malformed_pending_file_is_rejected_and_kept(tmp_path, line) -> None:
    store_path = _store_path(tmp_path)
    pending_path = classify._pending_path(store_path)
    pending_path.write_text(line + "\n")
    client = FakeClient([])
    with pytest.raises(AsioDocsError, match="line 1") as excinfo:
        classify.classify(_items(1), client_factory=lambda: client, store_path=store_path)
    assert str(pending_path) in str(excinfo.value)
    assert pending_path.read_text() == line + "\n"
    assert client.calls == []


def _pending_row(**changes) -> str:
    text = "Fixed a bug."
    record = {
        "key": classify._row_key(classify.PROMPT_VERSION, classify.MODEL, classify.EFFORT, text),
        "category": "fixed",
        "breaking": False,
        "breaking_reason": "",
        "model": classify.MODEL,
        "effort": classify.EFFORT,
        "prompt_version": classify.PROMPT_VERSION,
        "release": "1.38.0",
        "text": text,
        "classified_at": 1_700_000_000.5,
    }
    return json.dumps({**record, **changes})


def test_a_valid_pending_line_round_trips() -> None:
    row = classify._parse_pending_line(_pending_row())
    assert classify._parse_pending_line(classify._pending_lines([row]).strip()) == row
    assert row.key == classify._cache_key("Fixed a bug.")


@pytest.mark.parametrize(
    ("changes", "why"),
    [
        ({"category": "improved"}, "improved"),
        ({"breaking": 0}, "breaking"),
        ({"classified_at": "yesterday"}, "classified_at"),
        ({"release": "one point oh"}, ""),
        ({"text": "Fixed a different bug."}, "does not match"),
        ({"extra": 1}, "unexpected"),
    ],
    ids=["category", "breaking", "classified_at", "release", "key-mismatch", "extra-field"],
)
def test_pending_lines_are_validated_strictly(changes, why) -> None:
    with pytest.raises(ValueError, match=why):
        classify._parse_pending_line(_pending_row(**changes))


# ---------------------------------------------------------------------------
# input bounds, the cost ceiling, and the preview
# ---------------------------------------------------------------------------


def test_over_long_entries_are_truncated_in_the_payload_but_stored_whole(tmp_path) -> None:
    text = "Fixed " + "x" * (classify._ENTRY_TEXT_CAP + 500)
    item = _item(text)
    client = FakeClient([_text_response(_good_items(1))])
    classify.classify([item], client_factory=lambda: client, store_path=_store_path(tmp_path))
    sent = json.loads(client.calls[0]["messages"][0]["content"])["items"][0]["text"]
    assert sent.startswith(text[: classify._ENTRY_TEXT_CAP])
    assert "truncated" in sent and len(sent) < len(text)
    conn = sqlite3.connect(_store_path(tmp_path))
    try:
        (stored_text,) = conn.execute("SELECT text FROM classifications").fetchone()
    finally:
        conn.close()
    assert stored_text == text
    assert classify.cache_key(item.entry) in classify.stored_keys(_store_path(tmp_path))


def test_entries_within_the_cap_are_sent_unchanged() -> None:
    text = "y" * classify._ENTRY_TEXT_CAP
    assert classify._payload_text(text) == text


def test_budget_input_cap_counts_reservations_of_requests_in_flight() -> None:
    stop = classify._StopFlag()
    caps = classify._Caps(max_requests=10, max_output_tokens=10**6, max_input_tokens=1_000, time_limit_s=60.0)
    budget = classify._RunBudget(caps, stop)
    first = budget.admit(10, 600)
    with pytest.raises(classify._RunStopped):
        budget.admit(10, 600)
    assert "input tokens" in str(stop.reason)
    budget.settle_billed(first, input_tokens=450, output_tokens=5)
    assert budget.usage().input_tokens == 450


def test_the_input_cap_ends_a_run_whose_requests_would_exceed_it(tmp_path, monkeypatch, fast_retries) -> None:
    # Timed-out attempts keep their input reservation (they may have been billed), so with
    # room for two requests' inputs, the third attempt is refused.
    monkeypatch.setattr(
        classify, "_run_input_cap", lambda bounds_sum, bounds_max, batches, requests: 2 * bounds_max
    )
    client = FakeClient([_raising(_timeout_error(httpx.ReadTimeout("read timed out"))) for _ in range(3)])
    with pytest.raises(AsioDocsError, match="input tokens"):
        classify.classify(_items(1), client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert len(client.calls) == 2


def test_the_request_input_bound_covers_the_actual_payload() -> None:
    ids, payload = classify._batch_payload(["k"], {"k": "Fixed a bug."})
    assert ids == ("e0",)
    assert classify._input_bound(payload) == 3_624 + 447 + 1_024 + len(payload.encode())


def test_a_run_whose_worst_case_is_over_the_ceiling_is_refused_before_anything_is_sent(
    tmp_path, capsys
) -> None:
    # 40 entries of 9,000 non-ASCII characters: each is sent JSON-escaped at 6 bytes a
    # character, so the batch's input bound alone is over 2 MB.
    items = [_item(f"{i} " + "\u00e9" * 9_000) for i in range(40)]
    client = FakeClient([])
    with pytest.raises(AsioDocsError, match=r"over the \$5\.00 limit") as excinfo:
        classify.classify(items, client_factory=lambda: client, store_path=_store_path(tmp_path))
    assert "smaller range" in str(excinfo.value)
    assert client.calls == []
    assert capsys.readouterr().err == ""


def test_the_preview_states_the_exact_provable_worst_case(tmp_path, capsys) -> None:
    items = _items(3)
    texts = {classify.cache_key(item.entry): item.entry.full_text() for item in items}
    _, payload = classify._batch_payload(list(texts), texts)
    caps = classify._caps_for_run(3, (classify._input_bound(payload),))
    client = FakeClient([_text_response(_good_items(3))])
    classify.classify(items, client_factory=lambda: client, store_path=_store_path(tmp_path))
    preview = capsys.readouterr().err.splitlines()[0]
    assert f"{caps.max_input_tokens:,} input tokens" in preview
    assert preview.endswith(f"at most {classify._at_most_usd(caps.max_cost_usd())}")
    assert caps.max_cost_usd() == (caps.max_input_tokens * 2 + caps.max_output_tokens * 10) / 1_000_000


# ---------------------------------------------------------------------------
# the wall-clock watchdog
# ---------------------------------------------------------------------------


def test_the_watchdog_abandons_a_request_that_never_ends(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 2)
    monkeypatch.setattr(classify, "_watchdog_s", lambda: 1.0)
    released = threading.Event()  # set by the exit, so the stuck worker thread can end

    def exit_after_releasing(status: int):
        released.set()
        raise _Exited(status)

    def dripping_reply():
        assert released.wait(timeout=10.0)
        return _text_response(_good_items(1))

    monkeypatch.setattr(classify, "_exit_process", exit_after_releasing)
    store_path = _store_path(tmp_path)
    client = FakeClient([_text_response(_good_items(1)), dripping_reply])
    started = time.monotonic()
    with pytest.raises(_Exited) as excinfo:
        classify.classify(_items(2), client_factory=lambda: client, store_path=store_path)
    assert excinfo.value.status == 1
    assert 1.0 <= time.monotonic() - started < 5.0
    assert len(classify.stored_keys(store_path)) == 1  # the reply that finished was stored
    err = capsys.readouterr().err
    assert "wall-clock limit" in err and "1 request still in flight" in err
    usage = next(line for line in err.splitlines() if "classification used" in line)
    assert "possibly billed" in usage


def test_the_usage_line_counts_abandoned_reservations_as_possibly_billed() -> None:
    usage = classify._Usage(
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
    line = classify._usage_line(usage)
    assert "20,000 input and 7,168 output tokens possibly billed" in line
    assert usage.cost_usd() == ((5_000 + 20_000) * 2 + (1_000 + 7_168) * 10) / 1_000_000
