"""Behavior that must be the same whichever engine runs: every test here runs against the
asyncio engine (with an async fake client) and the threaded reference (with a sync one)."""

import errno
import os
import sqlite3
import threading
import time

import classify_support as cs
import pytest

from asio_doc_tools import cli, classify, diag, notesdiff
from asio_doc_tools.classify import asyncio_engine, budget, prompt, store
from asio_doc_tools.classify.threaded_reference import engine as threaded_engine
from asio_doc_tools.diag import AsioDocsError
from classify_support import Delay, Echo, Raise, WaitForCalls


@pytest.fixture(params=cs.ENGINES)
def engine(request) -> cs.Engine:
    return cs.Engine(request.param)


def _one_per_batch(monkeypatch, in_flight: int) -> None:
    monkeypatch.setattr(budget, "BATCH_SIZE", 1)
    monkeypatch.setattr(budget, "MAX_IN_FLIGHT", in_flight)


# ---------------------------------------------------------------------------
# requests, the store, and replies
# ---------------------------------------------------------------------------


def test_requests_have_the_fixed_shape_and_scaled_max_tokens(engine, tmp_path) -> None:
    client = engine.client([cs.good_response(3)])
    engine.classify(cs.items(3), client, cs.store_path(tmp_path))
    kwargs = client.calls[0]
    assert kwargs == prompt.request_kwargs(2048 + 128 * 3, kwargs["messages"][0]["content"])


def test_a_stored_result_is_never_asked_for_again(engine, tmp_path) -> None:
    path = cs.store_path(tmp_path)
    entry_item = cs.item("Fixed a bug in ip::tcp.")
    engine.classify([entry_item], engine.client([cs.good_response()]), path)
    assert classify.cache_key(entry_item.entry) in classify.stored_keys(path)
    result = classify.classify(
        [entry_item],
        engine=engine.name,
        client_factory=lambda: pytest.fail("no client is needed"),
        store_path=path,
    )
    assert result[0].category == classify.Category.FIXED


def test_a_breaking_entry_carries_its_reason_and_identical_texts_are_sent_once(engine, tmp_path) -> None:
    same = [cs.item("Removed io_service.", "1.38.0"), cs.item("Removed io_service.", "1.37.0")]
    client = engine.client(
        [
            cs.text_response(
                [cs.good_item("e0", category="changed", breaking=True, reason="io_service is gone")]
            )
        ]
    )
    result = engine.classify(same, client, cs.store_path(tmp_path))
    assert len(client.calls) == 1
    assert [r.breaking_reason for r in result] == ["io_service is gone"] * 2


def test_the_stored_row_holds_the_text_and_the_classification_identity(engine, tmp_path) -> None:
    path = cs.store_path(tmp_path)
    entry_item = cs.item("Fixed a race in the reactor.")
    engine.classify([entry_item], engine.client([cs.good_response()]), path)
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT text, model, effort, prompt_version FROM classifications").fetchone()
    finally:
        conn.close()
    assert row == (entry_item.entry.full_text(), classify.MODEL, classify.EFFORT, classify.PROMPT_VERSION)


def test_max_tokens_splits_the_batch_in_half(engine, tmp_path) -> None:
    client = engine.client([cs.max_tokens_response(), cs.good_response(2), cs.good_response(2)])
    result = engine.classify(cs.items(4), client, cs.store_path(tmp_path))
    assert len(result) == 4
    assert [call["max_tokens"] for call in client.calls] == [2048 + 128 * 4, 2048 + 128 * 2, 2048 + 128 * 2]


def test_a_single_entry_hitting_max_tokens_is_an_error(engine, tmp_path) -> None:
    result, client = engine.run(
        [cs.item("A very long entry.")], [cs.max_tokens_response()], cs.store_path(tmp_path)
    )
    assert isinstance(result, AsioDocsError) and "max_tokens" in str(result)
    assert len(client.calls) == 1


def test_an_all_max_tokens_batch_splits_once_then_stops_the_run(engine, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(budget, "MAX_IN_FLIGHT", 1)
    size = budget.BATCH_SIZE
    script = [
        cs.max_tokens_response(output_tokens=budget.max_tokens_for(size)),
        cs.max_tokens_response(output_tokens=budget.max_tokens_for(size // 2)),
    ]
    result, client = engine.run(cs.items(3 * size), script, cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError) and "max_tokens" in str(result)
    assert len(client.calls) == 2  # the second half and the other batches were never sent
    assert classify.stored_keys(cs.store_path(tmp_path)) == frozenset()


def test_the_run_request_cap_binds_before_the_per_batch_maximum(engine, tmp_path) -> None:
    # The worst case of one batch (15 attempts) under the default caps: R = 2 * 1 + 4.
    rl = cs.rate_limited
    script = [Raise(rl()), Raise(rl()), cs.max_tokens_response()] + [
        Raise(rl()),
        Raise(rl()),
        cs.invalid_response(),
        Raise(rl()),
        Raise(rl()),
        cs.good_response(2),
    ] * 2
    result, client = engine.run(cs.items(4), script, cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError) and "hard cap of 6 API requests" in str(result)
    assert "Rerunning continues" in str(result)
    assert len(client.calls) == 6


def test_a_single_batch_absorbs_a_split_and_a_resend_under_the_default_caps(engine, tmp_path) -> None:
    script = [
        cs.max_tokens_response(output_tokens=budget.max_tokens_for(28)),
        cs.invalid_response(),
        cs.good_response(14, usage_=cs.usage(900, 450)),
        cs.good_response(14, usage_=cs.usage(900, 450)),
    ]
    result, client = engine.run(cs.items(28), script, cs.store_path(tmp_path))
    assert len(result) == 28
    assert len(client.calls) == 4


def test_an_invalid_reply_is_resent_once(engine, tmp_path) -> None:
    ok, client = engine.run(
        [cs.item("Some entry.")],
        [cs.invalid_response(), cs.text_response([cs.good_item("e0", category="added")])],
        cs.store_path(tmp_path),
    )
    assert ok[0].category == classify.Category.ADDED and len(client.calls) == 2
    failed, client = engine.run(
        [cs.item("Other entry.")], [cs.invalid_response(), cs.invalid_response()], cs.store_path(tmp_path)
    )
    assert isinstance(failed, AsioDocsError) and "invalid" in str(failed) and len(client.calls) == 2


def test_a_refusal_is_an_error(engine, tmp_path) -> None:
    result, _ = engine.run([cs.item("Some entry.")], [cs.refusal_response()], cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError) and "refused" in str(result)


def test_a_split_stores_its_first_half_when_the_second_half_is_refused(engine, tmp_path) -> None:
    path = cs.store_path(tmp_path)
    first, second = cs.item("Fixed the first entry."), cs.item("Fixed the second entry.")
    result, _ = engine.run(
        [first, second], [cs.max_tokens_response(), cs.good_response(), cs.refusal_response()], path
    )
    assert isinstance(result, AsioDocsError) and "refused" in str(result)
    assert classify.stored_keys(path) == cs.keys_of([first])


def test_a_split_stores_its_first_half_before_the_second_halfs_exception_is_raised(engine, tmp_path) -> None:
    path = cs.store_path(tmp_path)
    first, second = cs.item("Fixed the first entry."), cs.item("Fixed the second entry.")
    result, _ = engine.run(
        [first, second],
        [cs.max_tokens_response(), cs.good_response(), Raise(RuntimeError("second half"))],
        path,
    )
    assert isinstance(result, RuntimeError) and str(result) == "second half"
    assert classify.stored_keys(path) == cs.keys_of([first])


def test_after_a_batch_error_only_the_requests_in_flight_finish_and_are_stored(
    engine, tmp_path, monkeypatch
) -> None:
    _one_per_batch(monkeypatch, 2)
    path = cs.store_path(tmp_path)
    script = [WaitForCalls(2, Delay(0.1, cs.good_response())), WaitForCalls(2, cs.refusal_response())] + [
        cs.good_response()
    ] * 4
    result, client = engine.run(cs.items(6), script, path)
    assert isinstance(result, AsioDocsError) and "refused" in str(result) and "1 of 6 entries" in str(result)
    assert len(client.calls) == 2  # nothing was sent after the refusal
    assert len(classify.stored_keys(path)) == 1  # the request in flight was kept


def test_a_normal_full_history_run_stays_within_its_caps(engine, tmp_path) -> None:
    result, client = engine.run(cs.items(995), [Echo()] * 25, cs.store_path(tmp_path))
    assert len(result) == 995
    assert len(client.calls) == 25


# ---------------------------------------------------------------------------
# caps, retries, and time
# ---------------------------------------------------------------------------


def test_the_request_cap_ends_the_run_and_a_rerun_pays_only_for_the_rest(
    engine, tmp_path, monkeypatch
) -> None:
    _one_per_batch(monkeypatch, 1)
    monkeypatch.setattr(budget, "REQUESTS_PER_BATCH_CAP", 1)
    monkeypatch.setattr(budget, "SPARE_REQUESTS_CAP", 0)
    path = cs.store_path(tmp_path)
    three = cs.items(3)
    result, client = engine.run(three, [cs.invalid_response(), cs.good_response(), cs.good_response()], path)
    assert isinstance(result, AsioDocsError) and "hard cap of 3 API requests" in str(result)
    assert "2 of 3 entries" in str(result)
    assert len(client.calls) == 3
    assert classify.stored_keys(path) == cs.keys_of(three[:2])
    rerun, client = engine.run(three, [cs.good_response()], path)
    assert len(rerun) == 3 and len(client.calls) == 1


def test_the_output_cap_ends_the_run(engine, tmp_path, monkeypatch) -> None:
    # Caps leaving room for exactly one request's max_tokens at a time.
    _one_per_batch(monkeypatch, 1)
    monkeypatch.setattr(budget, "SPARE_FULL_SIZE_OUTPUTS_CAP", 0)
    monkeypatch.setattr(budget, "OUTPUT_TOKENS_PER_ENTRY_CAP", 0)
    path = cs.store_path(tmp_path)
    result, client = engine.run(cs.items(2), [cs.good_response(), cs.good_response()], path)
    assert isinstance(result, AsioDocsError) and "output tokens" in str(result)
    assert len(client.calls) == 1 and len(classify.stored_keys(path)) == 1


def test_the_input_cap_ends_a_run_whose_requests_would_exceed_it(engine, tmp_path, monkeypatch) -> None:
    # Timed-out attempts keep their input reservation (they may have been billed), so with
    # room for two requests' inputs, the third attempt is refused.
    monkeypatch.setattr(
        budget, "run_input_cap", lambda bounds_sum, bounds_max, batches, requests: 2 * bounds_max
    )
    monkeypatch.setattr("asio_doc_tools.classify.retry._RETRY_BASE_DELAY_S", 0.001)
    timeout = cs.timeout_error(__import__("httpx").ReadTimeout("read timed out"))
    result, client = engine.run(cs.items(1), [Raise(timeout)] * 3, cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError) and "input tokens" in str(result)
    assert len(client.calls) == 2


def test_a_retry_wait_ends_as_soon_as_the_run_stops(engine, tmp_path, monkeypatch) -> None:
    # One batch waits out a 30 s retry-after; the other is refused 50 ms in, which stops
    # the run, so the wait ends at once and no retry is sent.
    _one_per_batch(monkeypatch, 2)
    script = [
        WaitForCalls(2, Raise(cs.status_error(429, {"retry-after": "30"}))),
        WaitForCalls(2, Delay(0.05, cs.refusal_response())),
    ]
    started = time.monotonic()
    result, client = engine.run(cs.items(2), script, cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError) and "refused" in str(result)
    assert time.monotonic() - started < 5.0
    assert len(client.calls) == 2


def test_a_retry_wait_never_runs_past_the_time_limit(engine, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(budget, "RUN_TIME_LIMIT_S", 0.2)
    started = time.monotonic()
    result, client = engine.run(
        cs.items(1), [Raise(cs.status_error(429, {"retry-after": "30"}))], cs.store_path(tmp_path)
    )
    assert isinstance(result, AsioDocsError) and "time limit" in str(result)
    assert time.monotonic() - started < 5.0
    assert len(client.calls) == 1


def test_usage_over_a_reservation_stops_the_run_and_keeps_the_result(engine, tmp_path, monkeypatch) -> None:
    _one_per_batch(monkeypatch, 1)
    path = cs.store_path(tmp_path)
    two = cs.items(2)
    over = cs.good_response(usage_=cs.usage(10**7, 5))
    result, client = engine.run(two, [over, cs.good_response()], path)
    assert isinstance(result, AsioDocsError)
    assert "billed 10,000,000 input tokens" in str(result) and "bound was" in str(result)
    assert len(client.calls) == 1  # the second batch was never sent
    assert classify.stored_keys(path) == cs.keys_of(two[:1])


def test_a_run_over_the_cost_ceiling_is_refused_before_anything_is_sent(engine, tmp_path, capsys) -> None:
    # 40 entries of 9,000 non-ASCII characters, sent JSON-escaped at 6 bytes a character.
    huge = [cs.item(f"{i} " + "é" * 9_000) for i in range(40)]
    result, client = engine.run(huge, [], cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError) and "over the $6.00 limit" in str(result)
    assert "smaller range" in str(result)
    assert client.calls == []
    assert capsys.readouterr().err == ""


def test_over_long_entries_are_truncated_in_the_payload_but_stored_whole(engine, tmp_path) -> None:
    import json

    path = cs.store_path(tmp_path)
    text = "Fixed " + "x" * (prompt.ENTRY_TEXT_CAP + 500)
    client = engine.client([cs.good_response()])
    engine.classify([cs.item(text)], client, path)
    sent = json.loads(client.calls[0]["messages"][0]["content"])["items"][0]["text"]
    assert "truncated" in sent and len(sent) < len(text)
    conn = sqlite3.connect(path)
    try:
        (stored_text,) = conn.execute("SELECT text FROM classifications").fetchone()
    finally:
        conn.close()
    assert stored_text == text


# ---------------------------------------------------------------------------
# the store failing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to read-only files")
def test_a_read_only_store_is_refused_before_anything_is_sent(engine, tmp_path, capsys) -> None:
    path = cs.store_path(tmp_path)
    store.connect(path).close()
    path.chmod(0o444)
    try:
        result, client = engine.run(cs.items(8), [], path)
    finally:
        path.chmod(0o644)
    assert isinstance(result, AsioDocsError) and "not writable" in str(result) and str(path) in str(result)
    assert client.calls == []
    assert capsys.readouterr().err == ""


def test_results_the_store_rejects_mid_run_are_saved_and_not_paid_for_again(
    engine, tmp_path, monkeypatch
) -> None:
    _one_per_batch(monkeypatch, 1)
    path = cs.store_path(tmp_path)
    three = cs.items(3)
    good_upsert = store._UPSERT_SQL
    monkeypatch.setattr(
        store, "_UPSERT_SQL", "INSERT INTO no_such_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    result, client = engine.run(three, [Delay(0.05, cs.good_response())] * 3, path)
    assert isinstance(result, AsioDocsError) and "could not write" in str(result)
    assert str(store.pending_dir(path)) in str(result)
    paid = len(client.calls)
    assert 1 <= paid <= 2  # the threaded reference's second request may be in flight already
    saved = sum(len(p.read_text().splitlines()) for p in store.pending_dir(path).glob("*.jsonl"))
    assert saved == paid
    assert classify.stored_keys(path) == frozenset()

    monkeypatch.setattr(store, "_UPSERT_SQL", good_upsert)
    rerun, client = engine.run(three, [cs.good_response()] * (3 - paid), path)
    assert len(rerun) == 3 and len(client.calls) == 3 - paid
    assert cs.files(store.pending_dir(path)) == []


@pytest.mark.parametrize("how", ["save-fails", "save-torn"])
def test_results_that_cannot_be_saved_anywhere_are_printed(
    engine, tmp_path, monkeypatch, capsys, how
) -> None:
    path = cs.store_path(tmp_path)
    monkeypatch.setattr(
        store, "_UPSERT_SQL", "INSERT INTO no_such_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    if how == "save-fails":

        def disk_full(store_path, rows):
            raise OSError(errno.ENOSPC, "No space left on device", str(store_path))

        monkeypatch.setattr(store, "save_pending", disk_full)
    else:

        def disk_full_midway(fd: int, data: bytes) -> None:
            os.write(fd, data[:10])
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(store, "_write_all", disk_full_midway)
    entry_item = cs.item("Fixed a bug that must not be lost.")
    result, _ = engine.run([entry_item], [cs.text_response([cs.good_item("e0", category="added")])], path)
    assert isinstance(result, AsioDocsError) and "printed above" in str(result)
    printed = [line for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert [store.parse_pending_line(line).key for line in printed] == [classify.cache_key(entry_item.entry)]
    assert cs.files(store.pending_dir(path)) == []


@pytest.mark.parametrize("line", ["not json", "[]", '{"key": "k"}', "{broken"])
def test_a_malformed_pending_file_is_rejected_and_kept(engine, tmp_path, line) -> None:
    path = cs.store_path(tmp_path)
    store.pending_dir(path).mkdir()
    pending = store.pending_dir(path) / store.unique_pending_name()
    pending.write_text(line + "\n")
    result, client = engine.run(cs.items(1), [], path)
    assert isinstance(result, AsioDocsError) and "line 1" in str(result) and str(pending) in str(result)
    assert pending.read_text() == line + "\n"
    assert client.calls == []


# ---------------------------------------------------------------------------
# what a run prints, and its client
# ---------------------------------------------------------------------------


def test_the_preview_states_the_exact_provable_worst_case(engine, tmp_path, capsys) -> None:
    three = cs.items(3)
    texts = {classify.cache_key(it.entry): it.entry.full_text() for it in three}
    caps = budget.caps_for_run(3, (budget.input_bound(prompt.batch_payload(list(texts), texts)[1]),))
    engine.classify(three, engine.client([cs.good_response(3)]), cs.store_path(tmp_path))
    preview = capsys.readouterr().err.splitlines()[0]
    assert f"{caps.max_input_tokens:,} input tokens" in preview
    assert preview.endswith(f"at most {budget.at_most_usd(caps.max_cost_usd())}")


def test_the_preview_and_usage_are_printed_even_when_quiet(engine, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(diag, "_quiet", True)
    engine.classify([cs.item("Fixed a bug.")], engine.client([cs.good_response()]), cs.store_path(tmp_path))
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2 and "hard caps" in lines[0] and "used 1 request" in lines[1]
    result, _ = engine.run([cs.item("Some entry.")], [cs.refusal_response()], cs.store_path(tmp_path))
    assert isinstance(result, AsioDocsError)
    assert "used 1 request" in capsys.readouterr().err


def test_a_fully_stored_run_spends_and_prints_nothing_when_quiet(
    engine, tmp_path, monkeypatch, capsys
) -> None:
    path = cs.store_path(tmp_path)
    entry_item = cs.item("Fixed a bug.")
    engine.classify([entry_item], engine.client([cs.good_response()]), path)
    capsys.readouterr()
    monkeypatch.setattr(diag, "_quiet", True)
    classify.classify(
        [entry_item],
        engine=engine.name,
        client_factory=lambda: pytest.fail("no client is needed"),
        store_path=path,
    )
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "factory", [asyncio_engine.default_client_factory, threaded_engine.default_client_factory]
)
def test_the_default_clients_disable_sdk_retries_and_set_explicit_timeouts(monkeypatch, factory) -> None:
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    client = factory()
    expected = (
        anthropic.AsyncAnthropic if factory is asyncio_engine.default_client_factory else anthropic.Anthropic
    )
    assert isinstance(client, expected)
    assert client.max_retries == 0
    assert (client.timeout.read, client.timeout.connect) == (240.0, 10.0)


def test_missing_credentials_are_reported_before_any_spend(engine, tmp_path, monkeypatch, capsys) -> None:
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either
    with pytest.raises(AsioDocsError, match="ANTHROPIC_API_KEY"):
        classify.classify([cs.item("Fixed a bug.")], engine=engine.name, store_path=cs.store_path(tmp_path))
    assert capsys.readouterr().err == ""  # no preview: nothing was about to be sent


def test_classify_must_run_on_the_main_thread(engine, tmp_path) -> None:
    raised: list[BaseException] = []

    def run() -> None:
        try:
            classify.classify(
                [cs.item("x")],
                engine=engine.name,
                client_factory=lambda: engine.client([]),
                store_path=cs.store_path(tmp_path),
            )
        except BaseException as e:
            raised.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert len(raised) == 1 and isinstance(raised[0], AssertionError)


@pytest.mark.parametrize(("raised", "status"), [(AsioDocsError("stopped"), 1), (KeyboardInterrupt(), 130)])
def test_how_a_run_ends_becomes_the_exit_status(monkeypatch, raised, status) -> None:
    def build_diff_result(*args, **kwargs):
        raise raised

    monkeypatch.setattr(notesdiff, "build_diff_result", build_diff_result)
    assert cli.main(["diff", "1.38.1", "1.38.2", "--engine", "threads"]) == status
    assert cli.main(["diff", "1.38.1", "1.38.2"]) == status
