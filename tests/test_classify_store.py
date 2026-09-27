import errno
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import classify_support as cs
import pytest

from asio_doc_tools import classify
from asio_doc_tools.classify import budget, prompt, store
from asio_doc_tools.diag import AsioDocsError


def test_upsert_over_an_existing_row_keeps_one_row(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    item = cs.item("Added a new overload.")
    classify.classify(
        [item],
        engine="threads",
        client_factory=lambda: cs.ScriptedClient([cs.text_response([cs.good_item("e0", category="fixed")])]),
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
    conn = store.connect(store_path)
    try:
        store.upsert_classifications(
            conn, [cs.store_row(key, replacement, item.entry.full_text())], store_path
        )
        rows = conn.execute("SELECT category FROM classifications WHERE key = ?", (key,)).fetchall()
    finally:
        conn.close()
    assert rows == [("added",)]


def test_corrupt_store_is_reported_not_silently_discarded(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    store_path.write_bytes(b"not a sqlite database")
    with pytest.raises(AsioDocsError, match="corrupt"):
        classify.classify(
            [cs.item("x")], client_factory=lambda: pytest.fail("no client is needed"), store_path=store_path
        )


def test_schema_version_newer_than_understood_is_an_error(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    conn = sqlite3.connect(store_path)
    conn.execute("PRAGMA user_version = 999")
    conn.commit()
    conn.close()
    with pytest.raises(AsioDocsError, match="schema version"):
        classify.classify(
            [cs.item("x")], client_factory=lambda: pytest.fail("no client is needed"), store_path=store_path
        )
    with pytest.raises(AsioDocsError, match="schema version"):
        classify.stored_keys(store_path)


def test_database_error_at_read_time_is_reported(tmp_path) -> None:
    class _RaisingConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.DatabaseError("simulated corruption detected mid-read")

    with pytest.raises(AsioDocsError, match="corrupt"):
        store.load_classifications(_RaisingConnection(), ["some-key"], tmp_path / "store.sqlite3")


def test_database_error_at_write_time_is_reported_with_the_store_path(tmp_path) -> None:
    class _RaisingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def execute(self, *args, **kwargs):
            return None  # the lock-wait pragma

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
        store.upsert_classifications(
            _RaisingConnection(), [cs.store_row("key", classification, "text")], store_path
        )
    assert str(store_path) in str(excinfo.value)


def test_stored_keys_on_a_missing_store_creates_nothing(tmp_path) -> None:
    store_path = tmp_path / "not-yet" / "classifications.sqlite3"
    assert classify.stored_keys(store_path) == frozenset()
    assert not store_path.parent.exists()


def test_stored_keys_on_an_empty_store_file_leaves_it_untouched(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    store_path.write_bytes(b"")
    assert classify.stored_keys(store_path) == frozenset()
    assert store_path.read_bytes() == b""


def test_stored_keys_on_a_corrupt_store_is_an_error_and_keeps_the_file(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    junk = b"not a sqlite database, but paid-for data may be in here"
    store_path.write_bytes(junk)
    with pytest.raises(AsioDocsError, match="corrupt"):
        classify.stored_keys(store_path)
    assert store_path.read_bytes() == junk


def _pending_row(**changes) -> str:
    text = "Fixed a bug."
    record = {
        "key": prompt.row_key(classify.PROMPT_VERSION, classify.MODEL, classify.EFFORT, text),
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
    row = store.parse_pending_line(_pending_row())
    assert store.parse_pending_line(store.pending_lines([row]).strip()) == row
    assert row.key == prompt.key_for_text("Fixed a bug.")


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
        store.parse_pending_line(_pending_row(**changes))


@pytest.mark.parametrize(
    "interruption",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk-full", "ctrl-c"],
)
def test_a_torn_pending_write_leaves_no_file_and_does_not_block_later_saves(
    tmp_path, monkeypatch, interruption
) -> None:
    store_path = cs.store_path(tmp_path)
    store.connect(store_path).close()
    real_write_all = store._write_all

    def torn_write(fd: int, data: bytes) -> None:
        real_write_all(fd, data[: len(data) // 2])  # half a line reaches the disk
        raise interruption

    monkeypatch.setattr(store, "_write_all", torn_write)
    with pytest.raises(type(interruption)):
        store.save_pending(store_path, cs.rows(2))
    assert cs.files(store.pending_dir(store_path)) == []

    monkeypatch.setattr(store, "_write_all", real_write_all)
    store.save_pending(store_path, cs.rows(2, start=2))
    cs.import_pending(store_path)
    assert classify.stored_keys(store_path) == {row.key for row in cs.rows(2, start=2)}
    assert cs.files(store.pending_dir(store_path)) == []


def test_an_orphaned_temporary_file_is_imported_without_its_torn_tail(tmp_path, capsys) -> None:
    store_path = cs.store_path(tmp_path)
    directory = store.pending_dir(store_path)
    directory.mkdir()
    rows = cs.rows(3)
    complete = store.pending_lines(rows[:2]).encode()
    torn = store.pending_lines(rows[2:]).encode()[:25]
    old_ns = time.time_ns() - 2 * store.PENDING_ORPHAN_AGE_NS
    (directory / f".tmp-{old_ns}-1-{'a' * 32}.jsonl").write_bytes(complete + torn)
    fresh = directory / f".tmp-{store.unique_pending_name()}"  # a save in progress elsewhere
    fresh.write_bytes(torn)
    cs.import_pending(store_path)
    assert classify.stored_keys(store_path) == {row.key for row in rows[:2]}
    assert cs.files(directory) == [fresh.name]
    assert "incomplete last line" in capsys.readouterr().err


def test_two_writers_and_a_concurrent_import_lose_nothing(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    store.connect(store_path).close()
    writers_done = threading.Event()
    errors: list[BaseException] = []

    def writer(start: int) -> None:
        try:
            for i in range(start, start + 60, 3):
                store.save_pending(store_path, cs.rows(3, start=i))
        except BaseException as e:
            errors.append(e)

    def importer() -> None:
        try:
            while not writers_done.is_set():
                cs.import_pending(store_path)
        except BaseException as e:
            errors.append(e)

    writers = [threading.Thread(target=writer, args=(start,)) for start in (0, 1_000)]
    importers = [threading.Thread(target=importer) for _ in range(2)]
    for thread in writers + importers:
        thread.start()
    for thread in writers:
        thread.join()
    writers_done.set()
    for thread in importers:
        thread.join()
    cs.import_pending(store_path)  # whatever the importers had not reached yet

    assert errors == []
    assert classify.stored_keys(store_path) == {row.key for row in cs.rows(60) + cs.rows(60, start=1_000)}
    assert cs.files(store.pending_dir(store_path)) == []


@pytest.mark.parametrize(
    "failure", [AsioDocsError("the store failed"), KeyboardInterrupt()], ids=["store-error", "ctrl-c"]
)
def test_an_import_that_fails_partway_leaves_the_rest_importable(tmp_path, monkeypatch, failure) -> None:
    store_path = cs.store_path(tmp_path)
    store.connect(store_path).close()
    first = store.save_pending(store_path, cs.rows(2))
    second = store.save_pending(store_path, cs.rows(2, start=2))
    real_upsert = store.upsert_classifications
    calls = []

    def failing_second_time(conn, rows, path, **kwargs):
        calls.append(rows)
        if len(calls) == 2:
            raise failure
        real_upsert(conn, rows, path, **kwargs)

    monkeypatch.setattr(store, "upsert_classifications", failing_second_time)
    with pytest.raises(type(failure)):
        cs.import_pending(store_path)
    assert classify.stored_keys(store_path) == {row.key for row in cs.rows(2)}
    assert not first.exists()
    assert second.exists()  # renamed back from its claimed name, as it was

    monkeypatch.setattr(store, "upsert_classifications", real_upsert)
    cs.import_pending(store_path)
    assert classify.stored_keys(store_path) == {row.key for row in cs.rows(4)}
    assert cs.files(store.pending_dir(store_path)) == []


def test_a_claim_left_by_a_dead_import_is_taken_over_but_a_live_one_is_not(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    directory = store.pending_dir(store_path)
    directory.mkdir()
    old_ns = time.time_ns() - 2 * store.PENDING_ORPHAN_AGE_NS
    dead = directory / f".claimed-{old_ns}-1-{'b' * 32}-{store.unique_pending_name()}"
    dead.write_text(store.pending_lines(cs.rows(1)))
    live = directory / f".claimed-{time.time_ns()}-1-{'c' * 32}-{store.unique_pending_name()}"
    live.write_text(store.pending_lines(cs.rows(1, start=1)))
    cs.import_pending(store_path)
    assert classify.stored_keys(store_path) == {cs.rows(1)[0].key}
    assert cs.files(directory) == [live.name]


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()  # exited and reaped: no process has this pid (until it is reused)
    return child.pid


def test_files_of_a_dead_process_are_taken_over_at_once_but_not_those_of_a_live_one(tmp_path) -> None:
    store_path = cs.store_path(tmp_path)
    directory = store.pending_dir(store_path)
    directory.mkdir()
    now_ns = time.time_ns()
    dead, live, own = _dead_pid(), os.getppid(), os.getpid()

    def named(prefix: str, pid: int, ns: int = now_ns) -> str:
        return f"{prefix}{ns}-{pid}-{os.urandom(16).hex()}"

    dead_claim = directory / f"{named('.claimed-', dead)}-{store.unique_pending_name()}"
    dead_claim.write_text(store.pending_lines(cs.rows(1)))
    dead_temporary = directory / f"{named('.tmp-', dead)}.jsonl"
    dead_temporary.write_text(store.pending_lines(cs.rows(1, start=1)))
    live_claim = directory / f"{named('.claimed-', live)}-{store.unique_pending_name()}"
    live_claim.write_text(store.pending_lines(cs.rows(1, start=2)))
    old_own_claim = directory / (
        f"{named('.claimed-', own, now_ns - 2 * store.PENDING_ORPHAN_AGE_NS)}-{store.unique_pending_name()}"
    )
    old_own_claim.write_text(store.pending_lines(cs.rows(1, start=3)))

    cs.import_pending(store_path)

    assert classify.stored_keys(store_path) == {cs.rows(1)[0].key, cs.rows(1, start=1)[0].key}
    assert cs.files(directory) == sorted([live_claim.name, old_own_claim.name])


def test_orphan_rules_for_unusual_pids() -> None:
    now_ns = time.time_ns()
    old_ns = now_ns - 2 * store.PENDING_ORPHAN_AGE_NS
    assert not store.orphaned(old_ns, os.getpid(), now_ns)  # never this process's own
    assert not store.orphaned(now_ns, 0, now_ns)  # pid 0 is never signalled
    assert store.orphaned(old_ns, 0, now_ns)  # but the age rule still applies
    assert not store.orphaned(now_ns, 9_999_999_999, now_ns)  # out of range: age rule
    assert store.orphaned(now_ns, _dead_pid(), now_ns)


def test_a_store_write_near_the_hard_limit_does_not_wait_for_the_lock(tmp_path) -> None:
    path = cs.store_path(tmp_path)
    conn = store.connect(path)
    other = sqlite3.connect(path, timeout=0.0, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    other.execute("PRAGMA user_version = 1")
    try:
        entry_item = cs.item("Fixed a bug recorded at the last moment.")
        key = classify.cache_key(entry_item.entry)
        stop = budget.StopFlag()
        recorder = store.Recorder(
            conn, path, {}, {key: entry_item.entry.full_text()}, {key: entry_item.release}, stop, lambda: 0.0
        )
        started = time.monotonic()
        recorder.record({key: prompt.RawResult(prompt.Category.FIXED, False, "")}, None)
        assert time.monotonic() - started < 1.0
    finally:
        other.rollback()
        other.close()
        conn.close()
    assert "locked" in str(stop.reason)
    assert len(list(store.pending_dir(path).glob("*.jsonl"))) == 1  # saved for the next run
