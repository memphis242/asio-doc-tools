import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import anthropic  # noqa: F401 - imported at collection time, not timed: classify._send()
# imports it lazily on first use (over a second), which would otherwise dwarf the small,
# deliberate delays the wall-clock-timing tests below use to distinguish "waited on
# cancelled work" from "did not".
import pytest

from asio_doc_tools import classify
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


def _max_tokens_response() -> SimpleNamespace:
    return SimpleNamespace(stop_reason="max_tokens", content=[], usage=_usage(), stop_details=None)


def _blocking(delay_s: float, response: SimpleNamespace):
    """A queueable callable that sleeps before returning `response` - used to prove a
    batch was never cancelled by making it take a long time if it does run.
    """

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


class FakeMessages:
    """Each queued item is either a response, or a zero-arg callable that produces one
    (or raises) - the latter lets a test block a call or raise KeyboardInterrupt from it.
    Calls are served in the order they arrive, guarded by a lock since several worker
    threads may call `create()` concurrently.
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
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(responses)

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls


def _entry(text: str) -> Entry:
    return Entry(html=text, text=text, children=())


def _item(text: str, version: str = "1.38.0") -> classify.ClassifyItem:
    return classify.ClassifyItem(release=Version.parse(version), entry=_entry(text))


def _store_path(tmp_path):
    return tmp_path / "classifications.sqlite3"


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


def test_reclassify_upserts_over_the_existing_row(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Added a new overload.")
    classify.classify(
        [item],
        client_factory=lambda: FakeClient([_text_response([_good_item("e0", category="fixed")])]),
        store_path=store_path,
    )
    client = FakeClient([_text_response([_good_item("e0", category="added")])])
    result = classify.classify([item], client_factory=lambda: client, store_path=store_path, reclassify=True)
    assert len(client.calls) == 1
    assert result[0].category == classify.Category.ADDED

    # Still exactly one row for this key - an upsert, not a duplicate insert.
    conn = sqlite3.connect(store_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM classifications WHERE key = ?", (classify.cache_key(item.entry),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_breaking_entry_carries_its_reason(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Removed the deprecated io_service typedef.")
    client = FakeClient(
        [_text_response([_good_item("e0", category="changed", breaking=True, reason="io_service is gone")])]
    )
    result = classify.classify([item], client_factory=lambda: client, store_path=store_path)
    assert result[0].breaking is True
    assert result[0].breaking_reason == "io_service is gone"


def test_max_tokens_splits_the_batch_in_half(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    items = [_item(f"Entry number {i}.") for i in range(4)]
    responses = [
        _max_tokens_response(),  # full batch of 4
        _text_response([_good_item("e0"), _good_item("e1")]),  # left half
        _text_response([_good_item("e0"), _good_item("e1")]),  # right half
    ]
    client = FakeClient(responses)
    result = classify.classify(items, client_factory=lambda: client, store_path=store_path)
    assert len(result) == 4
    assert all(c.category == classify.Category.FIXED for c in result)
    assert len(client.calls) == 3  # the oversized batch, then its two halves


def test_single_item_batch_hitting_max_tokens_is_an_error(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("A very long entry.")
    client = FakeClient([_max_tokens_response()])
    with pytest.raises(AsioDocsError):
        classify.classify([item], client_factory=lambda: client, store_path=store_path)


def test_invalid_reply_is_retried_once_then_succeeds(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Some entry.")
    responses = [
        _text_response([_good_item("e0"), _good_item("e0")]),  # duplicate id: invalid
        _text_response([_good_item("e0", category="added")]),  # retry succeeds
    ]
    client = FakeClient(responses)
    result = classify.classify([item], client_factory=lambda: client, store_path=store_path)
    assert result[0].category == classify.Category.ADDED
    assert len(client.calls) == 2  # the invalid reply, then the retry


def test_invalid_reply_twice_is_an_error(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Some entry.")
    responses = [
        _text_response([{"id": "wrong-id", "category": "fixed", "breaking": False, "breaking_reason": ""}]),
        _text_response([{"id": "still-wrong", "category": "fixed", "breaking": False, "breaking_reason": ""}]),
    ]
    client = FakeClient(responses)
    with pytest.raises(AsioDocsError):
        classify.classify([item], client_factory=lambda: client, store_path=store_path)


def test_refusal_is_an_error(tmp_path) -> None:
    store_path = _store_path(tmp_path)
    item = _item("Some entry.")
    client = FakeClient([_refusal_response()])
    with pytest.raises(AsioDocsError, match="refused"):
        classify.classify([item], client_factory=lambda: client, store_path=store_path)


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


def test_batch_failure_cancels_queued_batches_and_stores_what_completed(tmp_path, monkeypatch) -> None:
    # One entry per batch, one worker: batch 0 fails; the single worker cannot possibly
    # have started batches 2-5 by the time batch 0's failure is discovered and queued
    # batches are cancelled, since it can only be working on (at most) batch 1 at that
    # point. Batches 2-5 are configured to block for a while if they ever *do* run, so a
    # regression back to "wait for every queued batch" shows up as a large elapsed time.
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    store_path = _store_path(tmp_path)
    items = [_item(f"Entry {i}.") for i in range(6)]

    block_delay = 0.3
    responses = [
        _refusal_response(),  # batch 0: fails
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 1: maybe in flight
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 2: must be cancelled
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 3: must be cancelled
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 4: must be cancelled
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 5: must be cancelled
    ]
    client = FakeClient(responses)

    started = time.monotonic()
    with pytest.raises(AsioDocsError, match="refused") as excinfo:
        classify.classify(items, client_factory=lambda: client, store_path=store_path)
    elapsed = time.monotonic() - started

    # The old bug waits for every queued batch to run to completion before the error can
    # surface (5 blocked batches * block_delay, serialized through one worker): well over
    # a second here. A correct fix returns promptly - at most one blocked batch might have
    # already started as a race with the cancellation, never all five.
    assert elapsed < 3 * block_delay, f"classify() waited on cancelled batches ({elapsed:.2f}s)"
    assert "cancelled" in str(excinfo.value)

    # At most the failing batch plus one racing-to-start batch were ever called.
    assert len(client.calls) <= 2
    # Whatever did complete successfully (batch 1, if it won the race) is stored; the
    # failing batch and every genuinely cancelled batch are not.
    stored = classify.stored_keys(store_path)
    successful_calls = len(client.calls) - 1  # minus the one that returned the refusal
    assert len(stored) == successful_calls


def test_split_batch_stores_the_successful_half_when_the_other_half_fails(tmp_path) -> None:
    # A single top-level batch of 2 forces a max_tokens split into two size-1 halves,
    # executed sequentially (no threading involved): left succeeds, right is refused.
    left = _item("Fixed the left entry.")
    right = _item("Fixed the right entry.")
    store_path = _store_path(tmp_path)
    responses = [
        _max_tokens_response(),  # the full batch of 2
        _text_response([_good_item("e0", category="fixed")]),  # left half succeeds
        _refusal_response(),  # right half is refused
    ]
    client = FakeClient(responses)

    with pytest.raises(AsioDocsError, match="refused"):
        classify.classify([left, right], client_factory=lambda: client, store_path=store_path)

    stored = classify.stored_keys(store_path)
    assert classify.cache_key(left.entry) in stored
    assert classify.cache_key(right.entry) not in stored


def test_keyboard_interrupt_does_not_wait_for_queued_batches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classify, "_BATCH_SIZE", 1)
    monkeypatch.setattr(classify, "_MAX_WORKERS", 1)
    store_path = _store_path(tmp_path)
    items = [_item(f"Entry {i}.") for i in range(4)]

    block_delay = 0.3
    responses = [
        _text_response([_good_item("e0")]),  # batch 0: succeeds and is stored
        _raising(KeyboardInterrupt()),  # batch 1: interrupted
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 2: must not be waited on
        _blocking(block_delay, _text_response([_good_item("e0")])),  # batch 3: must not be waited on
    ]
    client = FakeClient(responses)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        classify.classify(items, client_factory=lambda: client, store_path=store_path)
    elapsed = time.monotonic() - started

    assert elapsed < 2 * block_delay, f"classify() waited on queued batches after KeyboardInterrupt ({elapsed:.2f}s)"
    # Batch 0's result, upserted before the interrupt, survives it.
    assert classify.cache_key(items[0].entry) in classify.stored_keys(store_path)


def test_token_accounting_includes_split_and_retried_responses() -> None:
    # One top-level batch of 2: the full-batch call hits max_tokens (split in two), the
    # left half's first reply is invalid (retried), and both halves eventually succeed.
    # Every one of these four responses' usage must be counted, not just the last one.
    left_key = "left-key"
    right_key = "right-key"
    texts = {left_key: "Fixed the left entry.", right_key: "Fixed the right entry."}
    responses = [
        _max_tokens_response(),  # usage: 10/5 (default)
        _text_response([_good_item("e0"), _good_item("e0")], usage=_usage(20, 7)),  # left: invalid (dup id)
        _text_response([_good_item("e0")], usage=_usage(11, 3)),  # left retry: succeeds
        _text_response([_good_item("e0")], usage=_usage(13, 4)),  # right: succeeds
    ]
    client = FakeClient(responses)

    outcome = classify._classify_batch(client, [left_key, right_key], texts)

    assert outcome.error is None
    assert set(outcome.classified) == {left_key, right_key}
    assert outcome.input_tokens == 10 + 20 + 11 + 13
    assert outcome.output_tokens == 5 + 7 + 3 + 4


def test_database_error_at_read_time_is_reported(tmp_path) -> None:
    class _RaisingConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.DatabaseError("simulated corruption detected mid-read")

    with pytest.raises(AsioDocsError, match="corrupt"):
        classify._load_classifications(_RaisingConnection(), ["some-key"], tmp_path / "store.sqlite3")
