import json
import sqlite3
from types import SimpleNamespace

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


def _good_item(id_: str, *, category: str = "fixed", breaking: bool = False, reason: str = "") -> dict:
    return {"id": id_, "category": category, "breaking": breaking, "breaking_reason": reason}


class FakeMessages:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("fake client received more requests than responses were queued")
        return self._responses.pop(0)


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
