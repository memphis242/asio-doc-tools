"""The classification store (SQLite) and the pending directory beside it, which
holds paid-for results the store could not take; both engines store through `Recorder`.
"""

import contextlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .. import paths
from ..diag import AsioDocsError, note, warn
from ..versions import Version
from .budget import StopFlag, quantity
from .prompt import EFFORT, MODEL, PROMPT_VERSION, Category, RawResult, row_key


@dataclass(frozen=True, slots=True)
class Classification:
    category: Category
    breaking: bool
    breaking_reason: str
    model: str
    effort: str
    release: str  # the release version first classified under this text, for provenance
    classified_at: float  # unix epoch seconds (UTC)


def default_store_path() -> Path:
    return paths.data_dir() / "classifications.sqlite3"


# Bump alongside a schema change to `_CREATE_TABLE_SQL`. A store whose PRAGMA
# user_version is newer than this is refused rather than misread.
_SCHEMA_VERSION: Final = 1

_CREATE_TABLE_SQL: Final = """
CREATE TABLE IF NOT EXISTS classifications (
    key TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK (category IN ('fixed','added','changed','deprecated','other')),
    breaking INTEGER NOT NULL CHECK (breaking IN (0,1)),
    breaking_reason TEXT NOT NULL,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    release TEXT NOT NULL,
    text TEXT NOT NULL,
    classified_at REAL NOT NULL
) WITHOUT ROWID
"""

_UPSERT_SQL: Final = """
INSERT INTO classifications
    (key, category, breaking, breaking_reason, model, effort, prompt_version, release, text, classified_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(key) DO UPDATE SET
    category = excluded.category,
    breaking = excluded.breaking,
    breaking_reason = excluded.breaking_reason,
    model = excluded.model,
    effort = excluded.effort,
    prompt_version = excluded.prompt_version,
    release = excluded.release,
    text = excluded.text,
    classified_at = excluded.classified_at
"""


def _check_schema_version(version: int, path: Path) -> None:
    if version > _SCHEMA_VERSION:
        raise AsioDocsError(
            f"the classification store at {path} has schema version {version}, newer than this "
            f"version of asio-doc-tools understands (up to {_SCHEMA_VERSION}); upgrade "
            "asio-doc-tools before using it"
        )


def _corrupt_store_error(path: Path, error: sqlite3.DatabaseError) -> AsioDocsError:
    return AsioDocsError(
        f"the classification store at {path} is corrupt or unreadable ({error}); it holds "
        "paid-for API results, so fix or remove that file manually before continuing"
    )


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    if version == 0:
        with conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    else:
        _check_schema_version(version, path)


def connect(path: Path) -> sqlite3.Connection:
    """Opens (creating if needed) the classification store, never silently discarding it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        _ensure_schema(conn, path)
    except sqlite3.OperationalError as e:
        raise AsioDocsError(
            f"could not open the classification store at {path} (still locked after a 10s "
            f"timeout, or the directory is not writable): {e}"
        ) from e
    except sqlite3.DatabaseError as e:
        raise _corrupt_store_error(path, e) from e
    return conn


def _row_to_classification(row: tuple[Any, ...]) -> Classification:
    category, breaking, breaking_reason, model, effort, release, classified_at = row
    return Classification(
        category=Category(category),
        breaking=bool(breaking),
        breaking_reason=breaking_reason,
        model=model,
        effort=effort,
        release=release,
        classified_at=classified_at,
    )


# SQLite's default limit on bound parameters is 999; stay comfortably under it per query.
_QUERY_CHUNK_SIZE: Final = 500


def load_classifications(
    conn: sqlite3.Connection, keys: Sequence[str], path: Path
) -> dict[str, Classification]:
    result: dict[str, Classification] = {}
    unique_keys = list(dict.fromkeys(keys))
    try:
        for i in range(0, len(unique_keys), _QUERY_CHUNK_SIZE):
            chunk = unique_keys[i : i + _QUERY_CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                "SELECT key, category, breaking, breaking_reason, model, effort, release, classified_at "
                f"FROM classifications WHERE key IN ({placeholders})",
                chunk,
            ).fetchall()
            for key, *rest in rows:
                result[key] = _row_to_classification(tuple(rest))
    # OperationalError is a subclass of DatabaseError; catch it first for a more specific message.
    except sqlite3.OperationalError as e:
        raise AsioDocsError(f"could not read the classification store at {path}: {e}") from e
    except sqlite3.DatabaseError as e:
        raise _corrupt_store_error(path, e) from e
    return result


@dataclass(frozen=True, slots=True)
class StoreRow:
    """One stored classification, with everything its row holds."""

    key: str
    classification: Classification
    prompt_version: str
    text: str


# How long a store write waits for another process's lock before failing.
STORE_LOCK_WAIT_S: Final = 10.0


def upsert_classifications(
    conn: sqlite3.Connection,
    rows: Sequence[StoreRow],
    path: Path,
    *,
    lock_wait_s: float = STORE_LOCK_WAIT_S,
) -> None:
    """Writes `rows` in one transaction, waiting up to `lock_wait_s` for the store's lock:
    on any failure none of them are stored."""
    assert rows
    assert 0.0 <= lock_wait_s <= STORE_LOCK_WAIT_S
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(lock_wait_s * 1000)}")
        with conn:
            conn.executemany(
                _UPSERT_SQL,
                [
                    (
                        row.key,
                        row.classification.category.value,
                        int(row.classification.breaking),
                        row.classification.breaking_reason,
                        row.classification.model,
                        row.classification.effort,
                        row.prompt_version,
                        row.classification.release,
                        row.text,
                        row.classification.classified_at,
                    )
                    for row in rows
                ],
            )
    # OperationalError is a subclass of DatabaseError; catch it first for a more specific message.
    except sqlite3.OperationalError as e:
        raise AsioDocsError(
            f"could not write to the classification store at {path} (still locked after a 10s "
            f"timeout, the disk is full, or the file is not writable): {e}"
        ) from e
    except sqlite3.DatabaseError as e:
        raise _corrupt_store_error(path, e) from e


def stored_keys(store_path: Path | None = None) -> frozenset[str]:
    """Every key currently in the store, for read-only inspection (e.g. `releases`).

    Opens the store read-only and never creates it: no store yet means nothing is
    classified yet. Results waiting in the pending directory (see `import_pending`) are not
    counted until a classify() run imports them.
    """
    path = store_path if store_path is not None else default_store_path()
    if not path.exists():
        return frozenset()
    try:
        conn = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.OperationalError as e:
        raise AsioDocsError(f"could not open the classification store at {path}: {e}") from e
    try:
        (version,) = conn.execute("PRAGMA user_version").fetchone()
        if version == 0:
            return frozenset()  # an empty store: its table is created by the first classify()
        _check_schema_version(version, path)
        rows = conn.execute("SELECT key FROM classifications").fetchall()
    except sqlite3.OperationalError as e:
        raise AsioDocsError(f"could not read the classification store at {path}: {e}") from e
    except sqlite3.DatabaseError as e:
        raise _corrupt_store_error(path, e) from e
    finally:
        conn.close()
    return frozenset(row[0] for row in rows)


def check_store_writable(conn: sqlite3.Connection, path: Path) -> None:
    """Proves the store accepts writes before anything is paid for, with a write that is
    rolled back: BEGIN IMMEDIATE alone succeeds even on a read-only file."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            conn.execute(f"PRAGMA user_version = {int(version)}")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
    # OperationalError is a subclass of DatabaseError; catch it first for a more specific message.
    except sqlite3.OperationalError as e:
        raise AsioDocsError(
            f"the classification store at {path} is not writable ({e}), so nothing was sent: "
            "results that cannot be stored would be paid for again by every run; make the file "
            "and its directory writable (or wait for whatever holds its lock), then rerun"
        ) from e
    except sqlite3.DatabaseError as e:
        raise _corrupt_store_error(path, e) from e


# ---------------------------------------------------------------------------
# The pending directory
#
# When a batch's results cannot be written to the store in the middle of a run (the
# disk fills up, a lock outlasts its timeout), or when a run is abandoned (see
# _Waiting.abandon), they are saved to a pending directory next to the store, so they
# are not paid for again. Every classify() run imports that directory into the store
# before working out what is missing.
#
# Every save is its own file, never appended to: its lines go to a temporary file
# (".tmp-<name>"), which is fsynced, renamed to its final name ("<name>"), and the
# directory fsynced, so a final file is always complete. A save that fails midway (a
# full disk, an interrupt) removes its temporary file, and its results go to stderr as
# JSON lines instead. An import claims each file first by renaming it to a name unique
# to this process (".claimed-<time>-<pid>-<uuid>-<name>"), imports it in one
# transaction, and deletes it; a failed import renames it back. So two runs never import
# or delete one file twice, and a file whose import fails stays for a later run.
# Files left by a process that died (a temporary or claimed file whose pid no longer
# exists, or, when that cannot be told, over an hour old) are imported too, a temporary
# one without its torn last line, if any.
#
# Every line of a file is validated as strictly as a reply. An invalid line stops the run
# with an error naming the file and the line, and the file is kept for the user.
# ---------------------------------------------------------------------------

_PENDING_FIELDS: Final = frozenset(
    {
        "key",
        "category",
        "breaking",
        "breaking_reason",
        "model",
        "effort",
        "prompt_version",
        "release",
        "text",
        "classified_at",
    }
)
_PENDING_STRING_FIELDS: Final = (
    "key",
    "breaking_reason",
    "model",
    "effort",
    "prompt_version",
    "release",
    "text",
)
# <time in ns>-<pid>-<uuid>.jsonl; temporary and claimed files add a prefix to it.
_PENDING_NAME_RE: Final = re.compile(r"(?P<ns>[0-9]{1,20})-(?P<pid>[0-9]{1,10})-[0-9a-f]{32}\.jsonl")
_PENDING_TEMP_PREFIX: Final = ".tmp-"
_PENDING_CLAIMED_RE: Final = re.compile(
    r"\.claimed-(?P<ns>[0-9]{1,20})-(?P<pid>[0-9]{1,10})-[0-9a-f]{32}-(?P<inner>.+)"
)
# A temporary or claimed file this old belongs to a process that died, even when its pid
# is in use (by another process, since pids are reused): writing or importing one file
# takes well under a second.
PENDING_ORPHAN_AGE_NS: Final = 3_600 * 1_000_000_000


def pending_dir(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.stem}.pending")


def _pending_record(row: StoreRow) -> dict[str, Any]:
    record = {
        "key": row.key,
        "category": row.classification.category.value,
        "breaking": row.classification.breaking,
        "breaking_reason": row.classification.breaking_reason,
        "model": row.classification.model,
        "effort": row.classification.effort,
        "prompt_version": row.prompt_version,
        "release": row.classification.release,
        "text": row.text,
        "classified_at": row.classification.classified_at,
    }
    assert record.keys() == _PENDING_FIELDS
    return record


def pending_lines(rows: Sequence[StoreRow]) -> str:
    return "".join(json.dumps(_pending_record(row), ensure_ascii=False) + "\n" for row in rows)


def parse_pending_line(line: str) -> StoreRow:
    """One pending-file line as a store row, or ValueError saying what is wrong with it."""
    data = json.loads(line)  # json.JSONDecodeError is a ValueError
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    if data.keys() != _PENDING_FIELDS:
        wrong = sorted(set(data) ^ _PENDING_FIELDS)
        raise ValueError(f"missing or unexpected fields {wrong}")
    for field in _PENDING_STRING_FIELDS:
        if not isinstance(data[field], str):
            raise ValueError(f"{field!r} is not a string")
    if not data["text"].strip():
        raise ValueError("'text' is empty")
    if not isinstance(data["breaking"], bool):
        raise ValueError("'breaking' is not true or false")
    classified_at = data["classified_at"]
    if isinstance(classified_at, bool) or not isinstance(classified_at, int | float):
        raise ValueError("'classified_at' is not a number")
    if not math.isfinite(classified_at):
        raise ValueError("'classified_at' is not finite")
    category = Category(data["category"])  # ValueError for an unknown category
    Version.parse(data["release"])  # ValueError for a malformed version
    if data["key"] != row_key(data["prompt_version"], data["model"], data["effort"], data["text"]):
        raise ValueError("'key' does not match its text, prompt version, model, and effort")
    return StoreRow(
        key=data["key"],
        classification=Classification(
            category=category,
            breaking=data["breaking"],
            breaking_reason=data["breaking_reason"],
            model=data["model"],
            effort=data["effort"],
            release=data["release"],
            classified_at=float(classified_at),
        ),
        prompt_version=data["prompt_version"],
        text=data["text"],
    )


def _unique_suffix() -> str:
    """<time in ns>-<pid>-<32 random hex digits>: unique across processes and calls."""
    return f"{time.time_ns()}-{os.getpid()}-{os.urandom(16).hex()}"


def unique_pending_name() -> str:
    name = f"{_unique_suffix()}.jsonl"
    assert _PENDING_NAME_RE.fullmatch(name)
    return name


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_pending(store_path: Path, rows: Sequence[StoreRow]) -> Path:
    """Saves `rows` as a new, complete file in the pending directory, and returns its path.

    Raises OSError when that fails, having removed its temporary file; an interrupt
    midway removes it too, so no partial file is ever left under a final name.
    """
    assert rows
    directory = pending_dir(store_path)
    directory.mkdir(mode=0o700, exist_ok=True)
    name = unique_pending_name()
    temporary = directory / f"{_PENDING_TEMP_PREFIX}{name}"
    final = directory / name
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(fd, pending_lines(rows).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, final)
    except BaseException:
        # The rows are reported elsewhere by the caller; what was written of them is not
        # worth an error of its own, and a file left behind is imported once orphaned.
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    _fsync_dir(directory)
    return final


def _pending_rows(path: Path, raw: bytes, *, torn_tail_allowed: bool) -> tuple[StoreRow, ...]:
    """The rows of one pending file, or AsioDocsError naming the file and the bad line."""
    kept = (
        "it holds paid-for classifications a run could not store, so it is kept: fix or delete that "
        "line, then rerun"
    )
    if not raw.endswith(b"\n") and raw:
        if not torn_tail_allowed:
            raise AsioDocsError(f"{path} ends without a newline, so its last line is incomplete; {kept}")
        cut = raw.rfind(b"\n") + 1
        warn(
            f"{path} was being written when its process stopped; its incomplete last line "
            f"({len(raw) - cut} bytes) cannot be read and is left out"
        )
        raw = raw[:cut]
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as e:
        raise AsioDocsError(f"{path} is not valid UTF-8 ({e}); {kept}") from e
    rows: list[StoreRow] = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(parse_pending_line(line))
        except ValueError as e:
            raise AsioDocsError(
                f"line {number} of {path} is not a valid saved classification ({e}); {kept}"
            ) from e
    return tuple(rows)


def _pending_candidates(directory: Path, now_ns: int) -> tuple[tuple[str, str], ...]:
    """(name in the directory, its unclaimed name) of every file an import should take:
    every final file, and every temporary or claimed file orphaned by a dead process."""
    try:
        names = sorted(entry.name for entry in os.scandir(directory) if entry.is_file(follow_symlinks=False))
    except FileNotFoundError:
        return ()
    except OSError as e:
        raise AsioDocsError(
            f"could not list {directory} ({e}), which holds paid-for classifications a run could not "
            "store; make it readable, then rerun"
        ) from e
    candidates: list[tuple[str, str]] = []
    for name in names:
        if _PENDING_NAME_RE.fullmatch(name):
            candidates.append((name, name))
        elif name.startswith(_PENDING_TEMP_PREFIX):
            temporary = _PENDING_NAME_RE.fullmatch(name.removeprefix(_PENDING_TEMP_PREFIX))
            if temporary is not None and orphaned(int(temporary["ns"]), int(temporary["pid"]), now_ns):
                candidates.append((name, name))
        elif (claimed := _PENDING_CLAIMED_RE.fullmatch(name)) is not None:
            if orphaned(int(claimed["ns"]), int(claimed["pid"]), now_ns):
                candidates.append((name, claimed["inner"]))
    return tuple(candidates)


def orphaned(written_ns: int, pid: int, now_ns: int) -> bool:
    """Whether the process that named a temporary or claimed file `pid` at `written_ns` is
    gone, so an import may take the file over.

    Never for this process's own files. At once when no process has that pid; otherwise
    (the pid is alive, possibly reused, or cannot be checked) once the file is over an
    hour old.
    """
    if pid == os.getpid():
        return False
    # pid 0 would signal this process's group; os.kill(pid, 0) ends a process on Windows.
    if pid > 0 and os.name == "posix":
        try:
            os.kill(pid, 0)  # signal 0 checks that the process exists and sends nothing
        except ProcessLookupError:
            return True
        except (OSError, OverflowError):
            pass  # alive but not ours to signal (PermissionError), or an out-of-range pid
    return now_ns - written_ns > PENDING_ORPHAN_AGE_NS


def _import_pending_file(conn: sqlite3.Connection, store_path: Path, name: str, unclaimed: str) -> int | None:
    """Claims, imports, and deletes one pending file; returns how many rows it held, or None
    when another run claimed it first."""
    directory = pending_dir(store_path)
    claimed = directory / f".claimed-{_unique_suffix()}-{unclaimed}"
    original = directory / unclaimed
    try:
        os.rename(directory / name, claimed)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise AsioDocsError(
            f"could not claim {directory / name} to import it ({e}); fix its permissions, then rerun"
        ) from e
    try:
        raw = claimed.read_bytes()
        rows = _pending_rows(original, raw, torn_tail_allowed=unclaimed.startswith(_PENDING_TEMP_PREFIX))
        if rows:
            upsert_classifications(conn, rows, store_path)
    except BaseException as failure:
        try:
            os.rename(claimed, original)
        except OSError as e:
            failure.add_note(
                f"{claimed} could not be renamed back to {original} ({e}); a run in an hour takes it"
            )
        if isinstance(failure, OSError):
            raise AsioDocsError(
                f"could not read {original} ({failure}); make it readable, then rerun"
            ) from failure
        raise
    try:
        claimed.unlink()
    except OSError as e:
        warn(
            f"imported {original} into the store but could not delete it ({e}); a later run imports it again"
        )
    return len(rows)


def import_pending(conn: sqlite3.Connection, store_path: Path) -> None:
    """Moves the results earlier runs could not store from the pending directory into the store."""
    imported = 0
    for name, unclaimed in _pending_candidates(pending_dir(store_path), time.time_ns()):
        rows = _import_pending_file(conn, store_path, name, unclaimed)
        imported += rows if rows is not None else 0
    if imported:
        note(
            f"imported {quantity(imported, 'classification')} saved by an earlier run from "
            f"{pending_dir(store_path)}"
        )


# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------


class Recorder:
    """Where every batch's results go: into the store at once, one transaction per batch,
    so they stay stored whatever happens to the run afterwards; when the store fails, to
    the pending directory (see `import_pending`); and when that fails too, to stderr as
    JSON lines. Paid-for results are never dropped.

    A store failure sets the stop flag with that error as the reason. A batch's own
    unexpected exception is kept (the first one) for the engine to raise once every
    result is recorded. A store write waits for another process's lock no longer than
    `lock_wait_s()` allows, so recording never holds a run past its hard limit.

    Used only from the thread that created it, which owns the sqlite connection (sqlite
    connections are not shared across threads): the asyncio engine's loop thread, or the
    threaded reference's main thread.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        store_path: Path,
        store: dict[str, Classification],
        texts: Mapping[str, str],
        release_by_key: Mapping[str, Version],
        stop: StopFlag,
        lock_wait_s: Callable[[], float],
    ) -> None:
        self._owner: Final = threading.get_ident()
        self._conn: Final = conn
        self._store_path: Final = store_path
        self._store: Final = store
        self._texts: Final = texts
        self._release_by_key: Final = release_by_key
        self._stop: Final = stop
        self._lock_wait_s: Final = lock_wait_s
        self._stored_keys: set[str] = set()
        self._saved_to_pending = 0
        self._printed = 0
        self._first_failure: BaseException | None = None

    @property
    def first_failure(self) -> BaseException | None:
        return self._first_failure

    def record(self, classified: Mapping[str, RawResult], failure: BaseException | None) -> None:
        """Records one finished batch: what it `classified`, and the unexpected exception,
        if any, that ended it."""
        assert threading.get_ident() == self._owner
        if classified:
            self._store_rows(self._rows(classified))
        if failure is not None and self._first_failure is None:
            self._first_failure = failure

    def save_for_later(self, results: Sequence[Mapping[str, RawResult]]) -> None:
        """For a run being abandoned: saves `results` to the pending directory, without
        touching the store, whose lock could make it wait. The next run imports them."""
        assert threading.get_ident() == self._owner
        rows = tuple(row for classified in results for row in self._rows(classified))
        if rows:
            self._save_elsewhere(rows)

    def progress(self, total: int) -> str:
        parts = [
            f"{len(self._stored_keys)} of {quantity(total, 'entry', 'entries')} were classified and stored"
        ]
        if self._saved_to_pending:
            parts.append(
                f"{quantity(self._saved_to_pending, 'more was', 'more were')} saved to "
                f"{pending_dir(self._store_path)} and go into the store at the start of the next run"
            )
        if self._printed:
            parts.append(
                f"{quantity(self._printed, 'more', 'more')} could not be saved anywhere and were printed "
                "above as JSON lines"
            )
        return "; ".join(parts)

    def _rows(self, classified: Mapping[str, RawResult]) -> tuple[StoreRow, ...]:
        classified_at = time.time()
        return tuple(
            StoreRow(
                key=key,
                classification=Classification(
                    category=result.category,
                    breaking=result.breaking,
                    breaking_reason=result.breaking_reason,
                    model=MODEL,
                    effort=EFFORT,
                    release=str(self._release_by_key[key]),
                    classified_at=classified_at,
                ),
                prompt_version=PROMPT_VERSION,
                text=self._texts[key],
            )
            for key, result in classified.items()
        )

    def _store_rows(self, rows: Sequence[StoreRow]) -> None:
        try:
            upsert_classifications(self._conn, rows, self._store_path, lock_wait_s=self._lock_wait_s())
        except AsioDocsError as e:
            self._stop.trip(e)
            self._save_elsewhere(rows)
            return
        for row in rows:
            self._store[row.key] = row.classification
        self._stored_keys.update(row.key for row in rows)

    def _save_elsewhere(self, rows: Sequence[StoreRow]) -> None:
        directory = pending_dir(self._store_path)
        try:
            save_pending(self._store_path, rows)
        except OSError as e:
            warn(
                f"could not save {quantity(len(rows), 'paid-for classification')} to {directory} "
                f"either ({e}); here they are as JSON lines, as a file there would have held them:\n"
                + pending_lines(rows).rstrip("\n")
            )
            self._printed += len(rows)
            return
        self._saved_to_pending += len(rows)
