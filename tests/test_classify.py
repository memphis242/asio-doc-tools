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


def _ctx(client, *, caps: classify._Caps | None = None, clock=time.monotonic) -> classify._RunContext:
    stop = classify._StopFlag()
    budget = classify._RunBudget(caps or classify._caps_for_run(40, 1), stop, clock=clock)
    return classify._RunContext(client=client, budget=budget)


@pytest.fixture
def fast_retries(monkeypatch) -> None:
    monkeypatch.setattr(classify, "_RETRY_BASE_DELAY_S", 0.001)


def _interrupt_main_thread() -> None:
    """Delivers a real SIGINT to the main thread (what Ctrl-C does), waking it from a
    blocking wait."""
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)


class _Exited(Exception):
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
        classify._upsert_classifications(conn, [(key, replacement, item.entry.full_text())], store_path)
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
        classify._upsert_classifications(_RaisingConnection(), [("key", classification, "text")], store_path)
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

    classified = classify._classify_batch(ctx, ["left-key", "right-key"], texts)

    assert set(classified) == {"left-key", "right-key"}
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
    assert usage.output_tokens == 5
    assert usage.reserved_output_tokens == 0
    assert usage.cut_off_requests == (1 if possibly_billed else 0)
    assert usage.cut_off_output_tokens == (classify._max_tokens_for(1) if possibly_billed else 0)


def test_retry_delay_honors_retry_after_capped_at_30_s() -> None:
    assert classify._retry_delay_s(_status_error(429, {"retry-after-ms": "1500"}), 1) == 1.5
    assert classify._retry_delay_s(_status_error(429, {"retry-after": "7"}), 1) == 7.0
    assert classify._retry_delay_s(_status_error(429, {"retry-after": "600"}), 1) == 30.0
    assert classify._retry_delay_s(_status_error(429, {"retry-after": "soon"}), 1) == 2.0
    assert classify._retry_delay_s(_status_error(503), 2) == 4.0
    assert classify._retry_delay_s(_timeout_error(httpx.ReadTimeout("t")), 1) == 2.0


def test_a_stop_during_retry_backoff_ends_the_send_without_another_request() -> None:
    client = FakeClient(
        [_raising(_status_error(429, {"retry-after": "30"})), _text_response([_good_item("e0")])]
    )
    ctx = _ctx(client)
    threading.Timer(0.02, lambda: ctx.stop.trip(AsioDocsError("stopped by the test"))).start()
    started = time.monotonic()
    with pytest.raises(classify._RunStopped):
        classify._send(ctx, "{}", 1)
    assert time.monotonic() - started < 1.0
    assert len(client.calls) == 1


def test_retry_backoff_never_waits_past_the_time_limit() -> None:
    caps = classify._Caps(max_requests=10, max_output_tokens=100_000, time_limit_s=0.05)
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
    budget = classify._RunBudget(classify._caps_for_run(40, 1), stop, clock=clock)
    budget.settle_billed(budget.admit(100), input_tokens=1, output_tokens=1)
    clock.now += classify._RUN_TIME_LIMIT_S
    with pytest.raises(classify._RunStopped):
        budget.admit(100)
    assert stop.is_set()
    assert "time limit" in str(stop.reason)
    assert budget.usage().requests == 1


def test_budget_output_cap_counts_reservations_of_requests_in_flight() -> None:
    stop = classify._StopFlag()
    budget = classify._RunBudget(
        classify._Caps(max_requests=10, max_output_tokens=1_000, time_limit_s=60.0), stop
    )
    first = budget.admit(600)
    with pytest.raises(classify._RunStopped):
        budget.admit(600)  # 600 reserved + 600 > 1,000, although nothing is billed yet
    assert "output tokens" in str(stop.reason)
    budget.settle_billed(first, input_tokens=10, output_tokens=40)
    assert budget.usage() == classify._Usage(
        requests=1,
        input_tokens=10,
        output_tokens=40,
        cut_off_requests=0,
        cut_off_output_tokens=0,
        reserved_output_tokens=0,
    )


def test_budget_refuses_everything_once_the_stop_flag_is_set() -> None:
    stop = classify._StopFlag()
    budget = classify._RunBudget(classify._caps_for_run(40, 1), stop)
    stop.trip(AsioDocsError("first reason"))
    stop.trip(AsioDocsError("second reason"))
    with pytest.raises(classify._RunStopped):
        budget.admit(100)
    assert str(stop.reason) == "first reason"
    assert budget.usage().requests == 0


def test_worst_case_cost_of_the_full_history_is_small() -> None:
    caps = classify._caps_for_run(995, 25)
    assert (caps.max_requests, caps.max_output_tokens) == (54, 206_208)
    assert classify._max_cost_usd(caps) < 3.0


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
    assert time.monotonic() - started < 1.0
    assert len(client.calls) == 2  # nothing was sent after the refusal
    assert len(classify.stored_keys(store_path)) == 1  # the in-flight request's result was kept
    assert "1 of 6 entries" in str(excinfo.value)


def test_a_store_write_failure_stops_the_run_and_reports_what_was_not_stored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    monkeypatch.setattr(
        classify, "_UPSERT_SQL", "INSERT INTO no_such_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    store_path = _store_path(tmp_path)
    # Each request takes 50 ms, so the second is in flight when the first one's write fails.
    client = FakeClient([_blocking(0.05, _text_response(_good_items(1))) for _ in range(3)])
    with pytest.raises(AsioDocsError, match="could not write") as excinfo:
        classify.classify(_items(3), client_factory=lambda: client, store_path=store_path)
    message = str(excinfo.value)
    assert str(store_path) in message
    assert "2 more entries could not be stored" in message  # the first and the in-flight second
    assert len(client.calls) == 2  # nothing was sent after the failed write
    assert classify.stored_keys(store_path) == frozenset()


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


def test_first_ctrl_c_cancels_queued_batches_and_harvests_in_flight_ones(capsys) -> None:
    done: Future = Future()
    done.set_result({"k0": "result"})
    in_flight: Future = Future()
    assert in_flight.set_running_or_notify_cancel()
    queued: Future = Future()
    threading.Timer(0.02, lambda: in_flight.set_result({"k1": "result"})).start()
    stop = classify._StopFlag()
    stop.trip(AsioDocsError("interrupted"))
    harvested: list[Future] = []

    classify._stop_and_harvest(
        [done, in_flight, queued],
        harvested.append,
        stop=stop,
        interrupted=True,
        report_usage=lambda: pytest.fail("usage is reported by the caller on this path"),
        exit_process=_fake_exit,
    )

    assert queued.cancelled()
    assert harvested == [done, in_flight]
    assert "Ctrl-C again" in capsys.readouterr().err


def test_second_ctrl_c_abandons_in_flight_requests_and_exits_at_once() -> None:
    in_flight: Future = Future()
    assert in_flight.set_running_or_notify_cancel()
    stop = classify._StopFlag()
    stop.trip(AsioDocsError("interrupted"))
    harvested: list[Future] = []
    reported: list[bool] = []
    threading.Timer(0.02, _interrupt_main_thread).start()
    try:
        with pytest.raises(_Exited) as excinfo:
            classify._stop_and_harvest(
                [in_flight],
                harvested.append,
                stop=stop,
                interrupted=True,
                report_usage=lambda: reported.append(True),
                exit_process=_fake_exit,
            )
    finally:
        in_flight.set_result({})
    assert excinfo.value.status == 130
    assert reported == [True]
    assert harvested == []


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
