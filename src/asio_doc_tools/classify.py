"""LLM classification of revision-history entries, persisted locally in SQLite.

Every entry is classified by exactly one model (`MODEL`, `claude-sonnet-5`) at
exactly one effort level (`EFFORT`) - there is no per-call model choice and no
server-side fallback to a different model. Each top-level `history.Entry` is
classified once per (prompt version, model, effort, entry text) combination;
the result is cached in a local SQLite database keyed by a sha256 of that
combination, so repeated diffs over the same releases never re-ask the model.
Classification is a paid, non-deterministic operation, so a cache hit must
never be discarded and a corrupt store must never be silently dropped or
recreated.

What a run can spend is bounded on every code path: every HTTP attempt is first
admitted by the run's budget (hard caps on requests, output tokens, and time),
and the run prints what it expects to spend before its first request and what it
used on every exit. The "Run budget" section below has the arithmetic.
"""

import contextlib
import json
import math
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

from . import paths
from .diag import AsioDocsError, error, note, spend, warn
from .history import Entry
from .versions import Version

if TYPE_CHECKING:
    from concurrent.futures import Future

# Bump whenever the system prompt's meaning changes, so previously cached
# classifications (keyed on this value) are re-asked rather than reused stale.
PROMPT_VERSION: Final = "classify-2"

# Classification uses exactly this one model, at this one effort level, always - no
# per-call model choice and no server-side fallback to a different model.
MODEL: Final = "claude-sonnet-5"
EFFORT: Final = "low"
# A request's latency grows with its batch (about 0.9 s + 0.24 s per entry at low
# effort) and is unaffected by up to 8 requests in flight, so throughput scales with
# the worker count: the whole history (about 1,000 entries, 25 batches) takes 4
# rounds of requests with 8 workers, against 7 with 4, for the same batches and cost.
_BATCH_SIZE: Final = 40
_MAX_WORKERS: Final = 8


class Category(StrEnum):
    FIXED = "fixed"
    ADDED = "added"
    CHANGED = "changed"
    DEPRECATED = "deprecated"
    OTHER = "other"


SYSTEM_PROMPT: Final = f"""\
You classify entries from the revision history of Asio, a C++ library for \
asynchronous I/O and networking (the standalone, non-Boost distribution). \
The reader is a developer upgrading from one Asio version to another who \
needs to know what affects their code.

Each input item is one revision-history entry (its own text, plus any nested
sub-points, already flattened into the item's text). Assign it exactly one
category:

- fixed: corrects a defect - a bug, crash, leak, race, incorrect behavior,
  compile/link/build error, compiler warning, or documentation mistake. This
  includes an entry whose stated purpose is to fix, avoid, prevent, or work
  around such a defect, even when the entry's main verb is "changed" or
  "added" - classify by what the change accomplishes, not by its verb.
- added: a new capability - a new function, class, overload, member, trait,
  macro, configuration option, platform or compiler support, or example.
- changed: modifies existing behavior, implementation, performance, defaults,
  requirements, or interfaces, including removals, renames, and moves; also
  improvements, updates, reworks, or optimisations (including adding a
  previously-missing optimisation) that are not new capabilities and are not
  stated fixes for a defect.
- deprecated: marks an existing facility as deprecated (it still exists).
- other: only documentation, examples, tests, or release housekeeping, with
  no effect on the library's behavior or interface.

An entry whose sub-points mix several kinds gets the category of its main
thrust.

Independently, decide whether the entry is breaking: upgrading may force
users to change source code or build configuration, or silently changes
run-time behavior that correct programs may rely on. This requires positive
evidence in the entry's own text that code which previously compiled, linked,
and behaved correctly may now fail to compile or link, or behave differently:
removed or renamed public APIs, changed defaults, stricter constraints that
reject previously valid code, raised minimum compiler/C++ standard/platform
requirements, removed headers or macros, a changed signature/return
type/template parameter where the entry indicates calling code is affected,
or a documented ABI change. Do not infer breakage from internal
implementation mechanics alone - linkage (e.g. static vs. inline), inlining,
symbol visibility, or similar strategy changes - and do not infer breakage
from a fix that makes a declaration or behavior match what was already
documented or intended, unless the entry itself says previously working code
is affected. In particular, an entry that fixes an incorrect signature,
return type, or template parameter to match its documented or intended
contract is not breaking on that basis alone, because the prior form was
itself the defect: say breaking only if the entry states that code relying on
the old, incorrect form must change. Not breaking: purely additive changes,
deprecations (the
facility remains), moves that keep the old name available, performance-only
changes, and bug fixes that restore documented behavior (unless the entry
says code relying on the old, incorrect behavior must change). An entry is
breaking if any of its sub-points is breaking, even if its main thrust is
not.

When breaking is true, breaking_reason is one short sentence stating what
users must change or watch for. When breaking is false, breaking_reason is
the empty string.

Classify strictly from the given text; do not use outside knowledge of Asio's
actual history. Respond with exactly one result per input id, using only the
ids given."""

SCHEMA: Final = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string", "enum": [c.value for c in Category]},
                    "breaking": {"type": "boolean"},
                    "breaking_reason": {"type": "string"},
                },
                "required": ["id", "category", "breaking", "breaking_reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

ClientFactory = Callable[[], Any]


# The SDK is imported where it is used: importing it costs over a second, which
# every other command (and every man page build worker) would otherwise pay.


def default_client_factory() -> Any:
    """The client every classification request goes through. The SDK's own retries are
    off: each retry is made here instead, so the run budget admits and counts it. The
    timeouts are explained in the "Run budget" section."""
    import anthropic

    client = anthropic.Anthropic(
        max_retries=0, timeout=anthropic.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
    )
    # Without credentials the SDK fails only at the first request, with a bare TypeError.
    if not client.api_key and not client.auth_token and client.credentials is None:
        raise AsioDocsError(
            "no Anthropic API credentials found; set ANTHROPIC_API_KEY or run 'ant auth login'"
        )
    return client


@dataclass(frozen=True, slots=True)
class ClassifyItem:
    """One unit to classify: a top-level entry and the release it came from."""

    release: Version
    entry: Entry


@dataclass(frozen=True, slots=True)
class Classification:
    category: Category
    breaking: bool
    breaking_reason: str
    model: str
    effort: str
    release: str  # the release version first classified under this text, for provenance
    classified_at: float  # unix epoch seconds (UTC)


@dataclass(frozen=True, slots=True)
class _RawResult:
    category: Category
    breaking: bool
    breaking_reason: str


def default_store_path() -> Path:
    return paths.data_dir() / "classifications.sqlite3"


def cache_key(entry: Entry) -> str:
    return _cache_key(entry.full_text())


def _cache_key(text: str) -> str:
    normalized = text.strip()
    return sha256(f"{PROMPT_VERSION}\n{MODEL}\n{EFFORT}\n{normalized}".encode()).hexdigest()


def _request_id(error: Exception) -> str:
    return getattr(error, "request_id", None) or "no request id"


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


def _connect(path: Path) -> sqlite3.Connection:
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


def _load_classifications(
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
class _StoreRow:
    """One stored classification, with everything its row holds."""

    key: str
    classification: Classification
    prompt_version: str
    text: str


def _row_key(prompt_version: str, model: str, effort: str, text: str) -> str:
    """The store key of a row classified under `prompt_version`, `model`, and `effort`."""
    return sha256(f"{prompt_version}\n{model}\n{effort}\n{text.strip()}".encode()).hexdigest()


assert _row_key(PROMPT_VERSION, MODEL, EFFORT, " Fixed a bug.\n") == _cache_key("Fixed a bug.")


# How long a store write waits for another process's lock before failing.
_STORE_LOCK_WAIT_S: Final = 10.0


def _upsert_classifications(
    conn: sqlite3.Connection,
    rows: Sequence[_StoreRow],
    path: Path,
    *,
    lock_wait_s: float = _STORE_LOCK_WAIT_S,
) -> None:
    """Writes `rows` in one transaction, waiting up to `lock_wait_s` for the store's lock:
    on any failure none of them are stored."""
    assert rows
    assert 0.0 <= lock_wait_s <= _STORE_LOCK_WAIT_S
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
    classified yet. Results waiting in the pending directory (see `_import_pending`) are not
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


def _check_store_writable(conn: sqlite3.Connection, path: Path) -> None:
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
# Files left by a process that died (a temporary or claimed file over an hour old) are
# imported too, a temporary one without its torn last line, if any.
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
_PENDING_NAME_RE: Final = re.compile(r"(?P<ns>[0-9]{1,20})-[0-9]{1,10}-[0-9a-f]{32}\.jsonl")
_PENDING_TEMP_PREFIX: Final = ".tmp-"
_PENDING_CLAIMED_RE: Final = re.compile(
    r"\.claimed-(?P<ns>[0-9]{1,20})-[0-9]{1,10}-[0-9a-f]{32}-(?P<inner>.+)"
)
# A temporary or claimed file this old belongs to a process that died: writing or
# importing one file takes well under a second.
_PENDING_ORPHAN_AGE_NS: Final = 3_600 * 1_000_000_000


def _pending_dir(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.stem}.pending")


def _pending_record(row: _StoreRow) -> dict[str, Any]:
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


def _pending_lines(rows: Sequence[_StoreRow]) -> str:
    return "".join(json.dumps(_pending_record(row), ensure_ascii=False) + "\n" for row in rows)


def _parse_pending_line(line: str) -> _StoreRow:
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
    if data["key"] != _row_key(data["prompt_version"], data["model"], data["effort"], data["text"]):
        raise ValueError("'key' does not match its text, prompt version, model, and effort")
    return _StoreRow(
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


def _unique_pending_name() -> str:
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


def _save_pending(store_path: Path, rows: Sequence[_StoreRow]) -> Path:
    """Saves `rows` as a new, complete file in the pending directory, and returns its path.

    Raises OSError when that fails, having removed its temporary file; an interrupt
    midway removes it too, so no partial file is ever left under a final name.
    """
    assert rows
    directory = _pending_dir(store_path)
    directory.mkdir(mode=0o700, exist_ok=True)
    name = _unique_pending_name()
    temporary = directory / f"{_PENDING_TEMP_PREFIX}{name}"
    final = directory / name
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(fd, _pending_lines(rows).encode())
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


def _pending_rows(path: Path, raw: bytes, *, torn_tail_allowed: bool) -> tuple[_StoreRow, ...]:
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
    rows: list[_StoreRow] = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(_parse_pending_line(line))
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
            match = _PENDING_NAME_RE.fullmatch(name.removeprefix(_PENDING_TEMP_PREFIX))
            if match is not None and now_ns - int(match["ns"]) > _PENDING_ORPHAN_AGE_NS:
                candidates.append((name, name))
        elif (claimed := _PENDING_CLAIMED_RE.fullmatch(name)) is not None:
            if now_ns - int(claimed["ns"]) > _PENDING_ORPHAN_AGE_NS:
                candidates.append((name, claimed["inner"]))
    return tuple(candidates)


def _import_pending_file(conn: sqlite3.Connection, store_path: Path, name: str, unclaimed: str) -> int | None:
    """Claims, imports, and deletes one pending file; returns how many rows it held, or None
    when another run claimed it first."""
    directory = _pending_dir(store_path)
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
            _upsert_classifications(conn, rows, store_path)
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


def _import_pending(conn: sqlite3.Connection, store_path: Path) -> None:
    """Moves the results earlier runs could not store from the pending directory into the store."""
    imported = 0
    for name, unclaimed in _pending_candidates(_pending_dir(store_path), time.time_ns()):
        rows = _import_pending_file(conn, store_path, name, unclaimed)
        imported += rows if rows is not None else 0
    if imported:
        note(
            f"imported {_count(imported, 'classification')} saved by an earlier run from "
            f"{_pending_dir(store_path)}"
        )


@contextlib.contextmanager
def _sigint_ignored() -> Iterator[None]:
    """Ignores Ctrl-C (on the main thread, the only one a signal handler can be set from).

    Replacing the handler is what protects the main thread: blocking SIGINT on it with
    pthread_sigmask does not, since the kernel then delivers it to a worker thread and
    Python still raises KeyboardInterrupt on the main thread.
    """
    assert threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous if previous is not None else signal.SIG_DFL)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

# The longest entry in the whole revision history (as of 1.38.2) is 4,580 characters (in
# 1.11.0), so no real entry is truncated. Truncation keeps a pathological page within
# the input bound; the store key and the stored text stay the full text.
_LONGEST_OBSERVED_ENTRY_CHARS: Final = 4_580
_ENTRY_TEXT_CAP: Final = max(4_000, 2 * _LONGEST_OBSERVED_ENTRY_CHARS)
assert _ENTRY_TEXT_CAP == 9_160
assert _LONGEST_OBSERVED_ENTRY_CHARS < _ENTRY_TEXT_CAP <= 20_000


def _payload_text(text: str) -> str:
    """`text` as sent: longer than _ENTRY_TEXT_CAP characters, it is cut with a visible marker."""
    if len(text) <= _ENTRY_TEXT_CAP:
        return text
    return f"{text[:_ENTRY_TEXT_CAP]} [... {len(text) - _ENTRY_TEXT_CAP:,} more characters truncated]"


def _build_payload(ids: Sequence[str], batch_keys: Sequence[str], texts: Mapping[str, str]) -> str:
    items = [{"id": id_, "text": _payload_text(texts[key])} for id_, key in zip(ids, batch_keys, strict=True)]
    return json.dumps({"items": items}, indent=2)


def _batch_payload(batch_keys: Sequence[str], texts: Mapping[str, str]) -> tuple[tuple[str, ...], str]:
    """The ids and the payload of one request for `batch_keys`."""
    ids = tuple(f"e{i}" for i in range(len(batch_keys)))
    return ids, _build_payload(ids, batch_keys, texts)


# ---------------------------------------------------------------------------
# Run budget
#
# Every HTTP attempt a classify() run makes (a batch's first send, a split half, a
# resend after an invalid reply, and every retry of any of these) is first admitted by
# the run's _RunBudget, so the bounds below hold on every code path.
#
# Requests per batch. A request for n entries asks for max_tokens = 2,048 + 128 n
# (7,168 for a full batch of 40, 4,608 for a half of 20, 2,176 for one entry), at least
# 5x the normal output of about 32 tokens per entry, thinking included. A max_tokens
# stop therefore means runaway output, which smaller requests do not fix, so a batch is
# split at most once:
#
#   reply                                          logical sends for the batch
#   valid                                          1
#   invalid (bad JSON, missing or unknown ids)     2: the same request is resent once
#   max_tokens on the full batch                   1 + two halves of 1 or 2 each: at most 5
#   max_tokens on a half, a resend, or one entry   error
#   refusal, invalid twice, non-retryable error    error
#
# A logical send is at most 3 HTTP attempts: up to 2 retries, only after a 408, 409,
# 429, or 5xx status (529 included), a connection error, or a timeout, waiting 2 s then
# 4 s, or the server's retry-after, never more than 30 s. The SDK's own retries are off.
# So one batch makes at most 5 * 3 = 15 attempts, and any error stops the whole run.
#
# Input per request. An entry's text is sent whole up to 9,160 characters (twice the
# longest real entry) and cut with a visible marker beyond that. A request's input is
# bounded by the UTF-8 bytes of the system prompt (3,624), the JSON schema (447), and
# its payload, plus 1,024 for message framing and the hidden structured-output prompt:
# count_tokens measured 7 tokens for a one-token message and a constant 378 for the
# structured output, 385 in all, so 1,024 is over 2.5x that. This rests on a token
# never covering less than one byte of text, so tokens <= bytes; measured ratios were
# 0.32-0.48 tokens per byte for real payloads and 0.97 at most, for random punctuation.
#
# Caps per run, for N entries to classify in B = ceil(N / 40) batches whose planned
# requests have input bounds b_1..b_B:
#
#   requests       R = 2 B + 4
#   output tokens  T = (min(8, B) + 3) * 7,168 + 128 N
#   input tokens   I = (b_1 + ... + b_B) + (R - B) * max(b_i)
#   time           no attempt starts more than 300 s after the run started
#   wall clock     at 560 s the run stops waiting: it saves whatever finished to the
#                  pending directory, reports, and exits (status 1), abandoning any
#                  request still in flight
#
# Before sending, an attempt reserves its whole max_tokens and its input bound, and is
# refused unless, for output and for input alike, billed + possibly billed + reserved +
# its own reservation stays within the cap. A response turns its reservations into its
# actual usage; a failure that generated nothing (an HTTP error status, a connection that
# was never made) refunds them; a timeout or a connection dropped mid-request keeps them
# as possibly billed. So billed output <= T, billed input <= I, and requests <= R,
# whatever happens. Every request beyond the first per batch (a half, a resend, a retry)
# carries at most one batch's payload, which is why (R - B) * max(b_i) covers them. A
# response billed for more than its reservation (the API passing max_tokens, or text
# taking more tokens than bytes) stops the run, since the caps then prove nothing: only
# the requests already in flight finish.
#
# Why these numbers. Up to 8 requests are in flight, each reserving up to 7,168, so a
# run needs min(8, B) * 7,168 of headroom for reservations alone. On top of that, 128 per
# entry is 4x the normal 32, and 3 more full-size max_tokens cover a max_tokens stop (up
# to 7,168 billed) plus the halves and the resend that follow it. A normal run peaks at
# about 32 N billed plus min(8, B) * 7,168 reserved, which is 96 N + 21,504 under T, and
# its input is well under the sum of its batches' bounds. R gives every batch a second
# request plus 4 spare, so even a one-batch run absorbs a split (3 requests), a resend
# (1), and 2 retries.
#
# Worst case, at Sonnet 5's $2 / $10 per million input / output tokens. A run whose
# worst case, T * $10/M + I * $2/M, is over $5.00 is refused before anything is sent (a
# page with far more or far longer entries than Asio's history would be): diffing a
# smaller range first costs less, and its results carry over.
#
#                       requests  output tokens  input tokens  cost at most
#   one batch of 28     6         32,256         58,464        $0.44
#   full history, 995   54        206,208        904,519       $3.88
#
# (for 1.38.1 -> 1.38.2, whose payload is 4,649 bytes; and for all 995 entries, whose 25
# payloads total 201,349 bytes, the largest 14,760). A normal full-history run: 25
# requests, about 32K output and 115K input tokens, about $0.55.
#
# Time. A request times out after 240 s without receiving a byte (a non-streaming reply
# arrives whole once generated, and a full-size 7,168 tokens takes about 57 s at the
# observed 126 tokens/s: about 4x margin), or after 10 s trying to connect. Retry waits
# end at the time limit, so the last attempt starts by 300 s and, unless its reply
# arrives a byte at a time, ends by 550 s. The watchdog at 560 s bounds even that: no
# batch is harvested after it, a store write waits for another process's lock only until
# it, and abandoning the run writes nothing but one pending file (with Ctrl-C ignored)
# before exiting, so the process ends within moments of 560 s.
#
# Stopping. The first error, cap refusal, store-write failure, or Ctrl-C sets the run's
# stop flag, which every attempt checks first: from then on no request is sent, only the
# ones already in flight (at most 8) finish, and their results are stored.
# ---------------------------------------------------------------------------

_MAX_TOKENS_BASE: Final = 2048
_MAX_TOKENS_PER_ENTRY: Final = 128
# The SDK refuses a non-streaming request whose max_tokens could take over 10 minutes to
# generate at its assumed 128,000 tokens per hour (`_calculate_nonstreaming_timeout`):
# 600 s * 128,000 / 3,600 s = 21,333. It only checks that under its default timeout,
# which this client replaces, so the limit is kept here instead.
_SDK_NONSTREAMING_MAX_TOKENS: Final = 21_333

_SYSTEM_PROMPT_BYTES: Final = len(SYSTEM_PROMPT.encode())
_SCHEMA_BYTES: Final = len(json.dumps(SCHEMA).encode())
# count_tokens, for claude-sonnet-5 with this request's shape: a user message "x" alone
# counts 7 tokens, and output_config's json_schema format adds 378 whatever the payload
# (thinking and effort add none).
_MEASURED_MESSAGE_FRAMING_TOKENS: Final = 7
_MEASURED_STRUCTURED_OUTPUT_TOKENS: Final = 378
_INPUT_FRAMING_ALLOWANCE: Final = 1_024
_REQUEST_INPUT_BASE: Final = _SYSTEM_PROMPT_BYTES + _SCHEMA_BYTES + _INPUT_FRAMING_ALLOWANCE

_REQUESTS_PER_BATCH_CAP: Final = 2
_SPARE_REQUESTS_CAP: Final = 4
_SPARE_FULL_SIZE_OUTPUTS_CAP: Final = 3
_OUTPUT_TOKENS_PER_ENTRY_CAP: Final = 128
_RUN_TIME_LIMIT_S: Final = 300.0
_MAX_RUN_COST_USD: Final = 5.00

_REQUEST_TIMEOUT_S: Final = 240.0
_CONNECT_TIMEOUT_S: Final = 10.0
_WATCHDOG_GRACE_S: Final = 10.0
_MAX_ATTEMPTS: Final = 3
_RETRY_BASE_DELAY_S: Final = 2.0
_RETRY_MAX_DELAY_S: Final = 30.0
# Retried as well: every 5xx status (529, overloaded, included).
_RETRYABLE_STATUSES: Final = frozenset({408, 409, 429})

# Observed at low effort, for the estimate printed before a run: about 32 output tokens
# per entry, thinking included; and count_tokens found every request's input to be
# 1,521 tokens plus about one token per 2.6-2.7 bytes of payload.
_EXPECTED_OUTPUT_TOKENS_PER_ENTRY: Final = 32
_EXPECTED_INPUT_TOKENS_PER_REQUEST: Final = 1_521
_EXPECTED_PAYLOAD_BYTES_PER_TOKEN: Final = 2.6
# Claude Sonnet 5 list prices in US dollars per million tokens: what the printed costs,
# and the $5.00 ceiling on a run's worst case, are computed with.
_INPUT_USD_PER_MTOK: Final = 2.0
_OUTPUT_USD_PER_MTOK: Final = 10.0

# The full history as of 1.38.2, for the checks below: 995 entries in 25 batches, whose
# payloads total 201,349 bytes, the largest 14,760.
_FULL_HISTORY_PAYLOAD_BYTES: Final = 201_349
_FULL_HISTORY_LARGEST_PAYLOAD_BYTES: Final = 14_760


def _max_tokens_for(entry_count: int) -> int:
    """max_tokens for a request classifying `entry_count` entries."""
    assert entry_count >= 1
    max_tokens = _MAX_TOKENS_BASE + _MAX_TOKENS_PER_ENTRY * entry_count
    assert max_tokens <= _SDK_NONSTREAMING_MAX_TOKENS
    return max_tokens


def _input_bound(payload: str) -> int:
    """An upper bound on a request's input tokens: see "Input per request" above."""
    return _REQUEST_INPUT_BASE + len(payload.encode())


def _watchdog_s() -> float:
    """How long after its start a run stops waiting for anything."""
    return _RUN_TIME_LIMIT_S + _CONNECT_TIMEOUT_S + _REQUEST_TIMEOUT_S + _WATCHDOG_GRACE_S


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _INPUT_USD_PER_MTOK + output_tokens * _OUTPUT_USD_PER_MTOK) / 1_000_000


@dataclass(frozen=True, slots=True)
class _Caps:
    max_requests: int
    max_output_tokens: int
    max_input_tokens: int
    time_limit_s: float

    def max_cost_usd(self) -> float:
        """The most a run under these caps can be billed."""
        return _cost_usd(self.max_input_tokens, self.max_output_tokens)


def _run_input_cap(bounds_sum: int, bounds_max: int, batches: int, requests: int) -> int:
    assert requests >= batches >= 1
    assert bounds_max * batches >= bounds_sum >= bounds_max
    return bounds_sum + (requests - batches) * bounds_max


def _caps_for_run(entries: int, batch_input_bounds: Sequence[int]) -> _Caps:
    """The caps for `entries` entries in batches whose planned requests have these input bounds."""
    batches = len(batch_input_bounds)
    assert 1 <= batches <= entries
    assert all(bound > _REQUEST_INPUT_BASE for bound in batch_input_bounds)
    max_requests = _REQUESTS_PER_BATCH_CAP * batches + _SPARE_REQUESTS_CAP
    return _Caps(
        max_requests=max_requests,
        max_output_tokens=(min(_MAX_WORKERS, batches) + _SPARE_FULL_SIZE_OUTPUTS_CAP)
        * _max_tokens_for(_BATCH_SIZE)
        + _OUTPUT_TOKENS_PER_ENTRY_CAP * entries,
        max_input_tokens=_run_input_cap(
            sum(batch_input_bounds), max(batch_input_bounds), batches, max_requests
        ),
        time_limit_s=_RUN_TIME_LIMIT_S,
    )


def _check_cost_ceiling(caps: _Caps) -> None:
    worst = caps.max_cost_usd()
    if worst > _MAX_RUN_COST_USD:
        raise AsioDocsError(
            f"this run could cost up to ${worst:,.2f} (at most {caps.max_requests:,} requests, "
            f"{caps.max_output_tokens:,} output and {caps.max_input_tokens:,} input tokens), over "
            f"the ${_MAX_RUN_COST_USD:.2f} limit per run, so nothing was sent; diff a smaller range "
            "of versions first: its results are stored, so a wider diff afterwards pays only for "
            "what is still missing"
        )


# The arithmetic in the "Run budget" comment above, checked at import time.
assert _max_tokens_for(_BATCH_SIZE) == 7_168 <= _SDK_NONSTREAMING_MAX_TOKENS
assert _max_tokens_for(_BATCH_SIZE) >= 5 * _EXPECTED_OUTPUT_TOKENS_PER_ENTRY * _BATCH_SIZE
assert _OUTPUT_TOKENS_PER_ENTRY_CAP >= 4 * _EXPECTED_OUTPUT_TOKENS_PER_ENTRY
assert (_SYSTEM_PROMPT_BYTES, _SCHEMA_BYTES, _REQUEST_INPUT_BASE) == (3_624, 447, 5_095)
assert _INPUT_FRAMING_ALLOWANCE >= 2 * (_MEASURED_MESSAGE_FRAMING_TOKENS + _MEASURED_STRUCTURED_OUTPUT_TOKENS)
# A one-batch run absorbs a split, a resend, and a retry: requests 1 + 2 + 1 + 1, and for
# output, the whole batch billed at max_tokens plus both halves and a resend reserved at once.
assert _caps_for_run(_BATCH_SIZE, (_REQUEST_INPUT_BASE + 1,)).max_requests >= 5
assert (
    _max_tokens_for(_BATCH_SIZE) + 3 * _max_tokens_for(_BATCH_SIZE // 2)
    <= _caps_for_run(_BATCH_SIZE, (_REQUEST_INPUT_BASE + 1,)).max_output_tokens
)
# One batch of 1.38.1 -> 1.38.2 (28 entries, a 4,649-byte payload).
assert _caps_for_run(28, (_REQUEST_INPUT_BASE + 4_649,)) == _Caps(
    max_requests=6, max_output_tokens=32_256, max_input_tokens=58_464, time_limit_s=300.0
)
# The full history: its exact worst case, and even with every batch as large as its
# largest, under the ceiling.
_FULL_HISTORY_CAPS: Final = _caps_for_run(
    995,
    (_REQUEST_INPUT_BASE + _FULL_HISTORY_LARGEST_PAYLOAD_BYTES,) * 25,
)
assert (
    _run_input_cap(
        25 * _REQUEST_INPUT_BASE + _FULL_HISTORY_PAYLOAD_BYTES,
        _REQUEST_INPUT_BASE + _FULL_HISTORY_LARGEST_PAYLOAD_BYTES,
        25,
        _FULL_HISTORY_CAPS.max_requests,
    )
    == 904_519
)
assert (_FULL_HISTORY_CAPS.max_requests, _FULL_HISTORY_CAPS.max_output_tokens) == (54, 206_208)
assert _cost_usd(904_519, 206_208) < 3.88 <= _FULL_HISTORY_CAPS.max_cost_usd() < _MAX_RUN_COST_USD
# A request times out after at least 4x the time it takes to generate its max_tokens at
# the observed 126 tokens/s, and the watchdog comes after every attempt's own timeouts.
assert _REQUEST_TIMEOUT_S >= 4 * _max_tokens_for(_BATCH_SIZE) / 126
assert _watchdog_s() == 560.0 > _RUN_TIME_LIMIT_S + _CONNECT_TIMEOUT_S + _REQUEST_TIMEOUT_S
assert _MAX_ATTEMPTS == 3 and _RETRY_BASE_DELAY_S * 2 ** (_MAX_ATTEMPTS - 2) <= _RETRY_MAX_DELAY_S


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural if plural is not None else singular + 's')}"


def _about_usd(amount: float) -> str:
    assert amount >= 0
    if amount < 0.001:
        return "under $0.001"
    return f"about ${amount:.3f}" if amount < 0.1 else f"about ${amount:.2f}"


def _at_most_usd(amount: float) -> str:
    """`amount` rounded up to the cent, so the figure printed is never below the bound."""
    assert amount >= 0
    return f"${math.ceil(amount * 100) / 100:.2f}"


def _expected_input_tokens(payload_bytes: int) -> int:
    return _EXPECTED_INPUT_TOKENS_PER_REQUEST + round(payload_bytes / _EXPECTED_PAYLOAD_BYTES_PER_TOKEN)


class _RunStopped(Exception):
    """An attempt refused because the run is stopping. Internal: the reason the run
    stopped is the stop flag's, and this never reaches the user."""


class _StopFlag:
    """A run's stop flag and the reason it was first set (thread-safe)."""

    def __init__(self) -> None:
        self._event: Final = threading.Event()
        self._lock: Final = threading.Lock()
        self._reason: AsioDocsError | None = None

    def trip(self, reason: AsioDocsError) -> None:
        """Sets the flag; the first reason given is the one kept."""
        with self._lock:
            if self._reason is None:
                self._reason = reason
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> AsioDocsError | None:
        with self._lock:
            assert (self._reason is None) == (not self._event.is_set())
            return self._reason

    def wait(self, seconds: float) -> None:
        """Sleeps for `seconds`, or less if the flag is set meanwhile."""
        self._event.wait(max(0.0, seconds))


@dataclass(frozen=True, slots=True)
class _Reservation:
    """One admitted attempt, holding its max_tokens and input bound until it is settled."""

    serial: int
    max_tokens: int
    input_bound: int


@dataclass(frozen=True, slots=True)
class _Usage:
    requests: int  # HTTP attempts admitted, whatever their outcome
    input_tokens: int  # billed, as reported by responses
    output_tokens: int
    cut_off_requests: int  # attempts that failed after possibly being billed
    cut_off_input_tokens: int  # their reservations, counted as possibly billed
    cut_off_output_tokens: int
    in_flight_requests: int  # attempts admitted and not settled yet
    reserved_input_tokens: int  # their reservations
    reserved_output_tokens: int

    @property
    def possibly_billed_requests(self) -> int:
        return self.cut_off_requests + self.in_flight_requests

    @property
    def possibly_billed_input_tokens(self) -> int:
        return self.cut_off_input_tokens + self.reserved_input_tokens

    @property
    def possibly_billed_output_tokens(self) -> int:
        return self.cut_off_output_tokens + self.reserved_output_tokens

    def cost_usd(self) -> float:
        """What was billed, plus everything that may have been."""
        return _cost_usd(
            self.input_tokens + self.possibly_billed_input_tokens,
            self.output_tokens + self.possibly_billed_output_tokens,
        )


class _RunBudget:
    """A run's hard caps, consulted before every HTTP attempt (thread-safe).

    admit() either reserves an attempt's max_tokens and input bound or refuses the
    attempt; a refusal for a cap sets the stop flag with that cap as its reason. Every
    admitted attempt is then settled exactly once, by settle_billed() or settle_failed().
    """

    def __init__(self, caps: _Caps, stop: _StopFlag, clock: Callable[[], float] = time.monotonic) -> None:
        self._caps: Final = caps
        self._stop: Final = stop
        self._clock: Final = clock
        self._started_at: Final = clock()
        self._deadline: Final = self._started_at + caps.time_limit_s
        self._lock: Final = threading.Lock()
        self._open: set[int] = set()  # serials of admitted, unsettled attempts
        self._requests = 0
        self._billed_input = 0
        self._billed_output = 0
        self._cut_off_requests = 0
        self._cut_off_input = 0
        self._cut_off_output = 0
        self._reserved_input = 0
        self._reserved_output = 0

    @property
    def stop(self) -> _StopFlag:
        return self._stop

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    @property
    def started_at(self) -> float:
        return self._started_at

    def admit(self, max_tokens: int, input_bound: int) -> _Reservation:
        """Reserves one attempt, or raises _RunStopped: when the stop flag is already set,
        or when this attempt would pass a cap (which sets the stop flag)."""
        assert max_tokens > 0 and input_bound > 0
        with self._lock:
            if self._stop.is_set():
                raise _RunStopped
            refusal = self._refusal(max_tokens, input_bound)
            if refusal is not None:
                self._stop.trip(refusal)
                raise _RunStopped
            self._requests += 1
            self._reserved_output += max_tokens
            self._reserved_input += input_bound
            self._open.add(self._requests)
            return _Reservation(serial=self._requests, max_tokens=max_tokens, input_bound=input_bound)

    def settle_billed(self, reservation: _Reservation, *, input_tokens: int, output_tokens: int) -> None:
        """A response arrived: its reservations become the usage it reports.

        Usage over a reservation means the bound behind it (max_tokens for output, bytes >=
        tokens for input) does not hold, so the caps no longer prove anything: the stop
        flag is set, and only the requests already in flight finish. The response itself
        is paid for and kept.
        """
        with self._lock:
            self._release(reservation)
            self._billed_input += input_tokens
            self._billed_output += output_tokens
            for kind, billed, bound in (
                ("input", input_tokens, reservation.input_bound),
                ("output", output_tokens, reservation.max_tokens),
            ):
                if billed > bound:
                    self._stop.trip(
                        AsioDocsError(
                            f"the API billed {billed:,} {kind} tokens for a request whose computed upper "
                            f"bound was {bound:,}, so this run's hard caps no longer hold and it stopped; "
                            "please report this"
                        )
                    )

    def settle_failed(self, reservation: _Reservation, *, possibly_billed: bool) -> None:
        """No usable response: the reservations are refunded, or kept as possibly billed."""
        with self._lock:
            self._release(reservation)
            if possibly_billed:
                self._cut_off_requests += 1
                self._cut_off_input += reservation.input_bound
                self._cut_off_output += reservation.max_tokens

    def seconds_left(self) -> float:
        return max(0.0, self._deadline - self._clock())

    def usage(self) -> _Usage:
        with self._lock:
            return _Usage(
                requests=self._requests,
                input_tokens=self._billed_input,
                output_tokens=self._billed_output,
                cut_off_requests=self._cut_off_requests,
                cut_off_input_tokens=self._cut_off_input,
                cut_off_output_tokens=self._cut_off_output,
                in_flight_requests=len(self._open),
                reserved_input_tokens=self._reserved_input,
                reserved_output_tokens=self._reserved_output,
            )

    def _release(self, reservation: _Reservation) -> None:
        assert reservation.serial in self._open, "an attempt was settled twice"
        self._open.remove(reservation.serial)
        self._reserved_output -= reservation.max_tokens
        self._reserved_input -= reservation.input_bound
        assert self._reserved_output >= 0 and self._reserved_input >= 0

    def _refusal(self, max_tokens: int, input_bound: int) -> AsioDocsError | None:
        caps = self._caps
        committed_output = self._billed_output + self._cut_off_output + self._reserved_output
        committed_input = self._billed_input + self._cut_off_input + self._reserved_input
        if self._clock() >= self._deadline:
            return AsioDocsError(
                f"stopped at this run's time limit of {caps.time_limit_s:.0f} s ({self._used()})"
            )
        if self._requests >= caps.max_requests:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_requests} API requests ({self._used()})"
            )
        if committed_output + max_tokens > caps.max_output_tokens:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_output_tokens:,} output tokens: the next "
                f"request could generate up to {max_tokens:,} on top of {committed_output:,} already "
                f"billed or reserved by requests in flight ({self._used()})"
            )
        if committed_input + input_bound > caps.max_input_tokens:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_input_tokens:,} input tokens: the next "
                f"request could take up to {input_bound:,} on top of {committed_input:,} already billed "
                f"or reserved by requests in flight ({self._used()})"
            )
        return None

    def _used(self) -> str:
        cut_off = (
            f", up to {self._cut_off_input:,} input and {self._cut_off_output:,} output more possibly "
            f"billed by {_count(self._cut_off_requests, 'request')} cut off"
            if self._cut_off_requests
            else ""
        )
        return (
            f"{_count(self._requests, 'request')} used, {self._billed_input:,} input and "
            f"{self._billed_output:,} output tokens billed{cut_off}"
        )


def _preview_line(batch_payload_bytes: Sequence[int], entries: int, caps: _Caps) -> str:
    batches = len(batch_payload_bytes)
    expected_cost = _cost_usd(
        sum(_expected_input_tokens(size) for size in batch_payload_bytes),
        _EXPECTED_OUTPUT_TOKENS_PER_ENTRY * entries,
    )
    return (
        f"classifying {_count(entries, 'entry', 'entries')} in {_count(batches, 'batch', 'batches')} "
        f"with {MODEL} ({EFFORT} effort): expect {_count(batches, 'request')}, "
        f"{_about_usd(expected_cost)}; hard caps {_count(caps.max_requests, 'request')}, "
        f"{caps.max_output_tokens:,} output and {caps.max_input_tokens:,} input tokens, "
        f"{caps.time_limit_s:.0f} s ({_watchdog_s():.0f} s wall clock): at most "
        f"{_at_most_usd(caps.max_cost_usd())}"
    )


def _usage_line(usage: _Usage) -> str:
    possibly = (
        f", plus up to {usage.possibly_billed_input_tokens:,} input and "
        f"{usage.possibly_billed_output_tokens:,} output tokens possibly billed by "
        f"{_count(usage.possibly_billed_requests, 'request')} cut off or abandoned in flight"
        if usage.possibly_billed_requests
        else ""
    )
    return (
        f"classification used {_count(usage.requests, 'request')}: {usage.input_tokens:,} input and "
        f"{usage.output_tokens:,} output tokens{possibly}, {_about_usd(usage.cost_usd())}"
    )


# ---------------------------------------------------------------------------
# Sending one request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ApiFailure:
    """How to handle the exception the SDK raised for one attempt."""

    message: str
    retryable: bool
    possibly_billed: bool  # the attempt may have generated (and been billed for) output


def _cause_suffix(error: BaseException) -> str:
    return f" ({error.__cause__!r})" if error.__cause__ is not None else ""


def _status_is_retryable(error: Any) -> bool:
    # The server can say outright that retrying will not help.
    if error.response.headers.get("x-should-retry") == "false":
        return False
    return error.status_code in _RETRYABLE_STATUSES or error.status_code >= 500


def _status_message(error: Any) -> str:
    request_id = _request_id(error)
    match error.status_code:
        case 400:
            return f"Anthropic API rejected the request ({request_id}): {error}"
        case 401:
            return (
                f"Anthropic API authentication failed ({request_id}); set ANTHROPIC_API_KEY or run "
                "'ant auth login'"
            )
        case 403:
            return f"Anthropic API denied permission ({request_id}): {error}"
        case 404:
            return f"Anthropic API could not find model {MODEL!r} ({request_id})"
        case 429:
            return f"Anthropic API rate limit exceeded ({request_id}): {error}"
        case 529:
            return f"the Anthropic API is overloaded ({request_id}): {error}"
        case status:
            return f"Anthropic API error (status {status}, request {request_id}): {error}"


def _api_failure(error: Exception) -> _ApiFailure:
    import anthropic
    import httpx

    match error:
        case anthropic.APITimeoutError():
            # Without a connection (or a free one in the pool) nothing was sent.
            never_sent = isinstance(error.__cause__, (httpx.ConnectTimeout, httpx.PoolTimeout))
            return _ApiFailure(
                f"an Anthropic API request timed out{_cause_suffix(error)}",
                retryable=True,
                possibly_billed=not never_sent,
            )
        case anthropic.APIConnectionError():
            never_sent = isinstance(error.__cause__, httpx.ConnectError)
            return _ApiFailure(
                f"could not reach the Anthropic API{_cause_suffix(error)}",
                retryable=True,
                possibly_billed=not never_sent,
            )
        case anthropic.APIStatusError():
            # An error status means the API rejected or abandoned the request: nothing billed.
            return _ApiFailure(
                _status_message(error), retryable=_status_is_retryable(error), possibly_billed=False
            )
        case _:
            return _ApiFailure(
                f"unexpected Anthropic API error: {error!r}", retryable=False, possibly_billed=True
            )


def _retry_after_s(error: Exception) -> float | None:
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


def _retry_delay_s(error: Exception, failed_attempt: int) -> float:
    """How long to wait after failed attempt number `failed_attempt` (from 1) of a send."""
    assert 1 <= failed_attempt < _MAX_ATTEMPTS
    requested = _retry_after_s(error)
    delay = requested if requested is not None else _RETRY_BASE_DELAY_S * 2 ** (failed_attempt - 1)
    return min(delay, _RETRY_MAX_DELAY_S)


def _billed_input_tokens(usage: Any) -> int:
    # Cache reads and writes are billed input too, though requests here never ask for caching.
    cached = (getattr(usage, "cache_creation_input_tokens", None) or 0) + (
        getattr(usage, "cache_read_input_tokens", None) or 0
    )
    return int(usage.input_tokens) + int(cached)


@dataclass(frozen=True, slots=True)
class _RunContext:
    """What every worker of one classify() run shares."""

    client: Any
    budget: _RunBudget

    @property
    def stop(self) -> _StopFlag:
        return self.budget.stop


def _send(ctx: _RunContext, payload: str, entry_count: int) -> Any:
    """One logical request: up to _MAX_ATTEMPTS HTTP attempts, each admitted by the budget,
    which reserves its max_tokens and its input bound first.

    Raises _RunStopped when the budget refuses an attempt, and AsioDocsError when a
    failure is not retryable or outlasts the retries.
    """
    import anthropic

    max_tokens = _max_tokens_for(entry_count)
    input_bound = _input_bound(payload)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        reservation = ctx.budget.admit(max_tokens, input_bound)
        try:
            response = ctx.client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": payload}],
            )
        except anthropic.APIError as e:
            failure = _api_failure(e)
            ctx.budget.settle_failed(reservation, possibly_billed=failure.possibly_billed)
            if not failure.retryable or attempt == _MAX_ATTEMPTS:
                attempts = f" (after {attempt} attempts)" if attempt > 1 else ""
                raise AsioDocsError(f"{failure.message}{attempts}") from e
            # Wakes early when the run is stopping, and never sleeps past its time limit;
            # the next admit() then refuses.
            ctx.stop.wait(min(_retry_delay_s(e, attempt), ctx.budget.seconds_left()))
            continue
        except BaseException:
            # Not an API failure (an interrupt, or a bug): the request may have been billed.
            ctx.budget.settle_failed(reservation, possibly_billed=True)
            raise
        usage = getattr(response, "usage", None)
        if usage is None:
            ctx.budget.settle_failed(reservation, possibly_billed=True)
        else:
            ctx.budget.settle_billed(
                reservation, input_tokens=_billed_input_tokens(usage), output_tokens=usage.output_tokens
            )
        return response
    raise AssertionError("unreachable: the last attempt returns or raises")


def _check_refusal(response: Any) -> None:
    if response.stop_reason != "refusal":
        return
    details = getattr(response, "stop_details", None)
    parts = [p for p in (getattr(details, "category", None), getattr(details, "explanation", None)) if p]
    suffix = f" ({'; '.join(parts)})" if parts else ""
    raise AsioDocsError(f"the model refused to classify a batch of entries{suffix}")


def _extract_reply_items(response: Any) -> list[Any] | None:
    text_block = next((b for b in response.content if getattr(b, "type", None) == "text"), None)
    if text_block is None:
        return None
    try:
        data = json.loads(text_block.text)
        items = data["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return items if isinstance(items, list) else None


def _validate_reply(response: Any, ids: Sequence[str]) -> dict[str, _RawResult] | None:
    items = _extract_reply_items(response)
    if items is None:
        return None
    expected = set(ids)
    seen: dict[str, _RawResult] = {}
    for raw in items:
        if not isinstance(raw, dict):
            return None
        try:
            id_ = raw["id"]
            result = _RawResult(
                category=Category(raw["category"]),
                breaking=bool(raw["breaking"]),
                breaking_reason=str(raw["breaking_reason"]),
            )
        except (KeyError, ValueError, TypeError):
            return None
        if id_ not in expected or id_ in seen:
            return None
        seen[id_] = result
    return seen if seen.keys() == expected else None


def _diagnose_reply(response: Any, ids: Sequence[str]) -> str:
    items = _extract_reply_items(response)
    if items is None:
        return "the reply was not valid JSON matching the schema"
    expected = set(ids)
    seen_ids = [raw.get("id") for raw in items if isinstance(raw, dict)]
    missing = sorted(expected - set(seen_ids))
    unknown = sorted(set(seen_ids) - expected)
    duplicates = sorted({i for i in seen_ids if seen_ids.count(i) > 1} - set(missing) - set(unknown))
    parts = []
    if missing:
        parts.append(f"missing ids {missing}")
    if unknown:
        parts.append(f"unknown ids {unknown}")
    if duplicates:
        parts.append(f"duplicate ids {duplicates}")
    return "; ".join(parts) if parts else "one or more reply entries failed schema validation"


# ---------------------------------------------------------------------------
# Classifying batches
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BatchOutcome:
    """What a batch (or part of one) classified, and the unexpected exception, if any,
    that ended it. Expected failures are the stop flag's reason instead."""

    classified: Mapping[str, _RawResult]
    failure: BaseException | None = None


def _classify_batch(ctx: _RunContext, batch_keys: Sequence[str], texts: Mapping[str, str]) -> _BatchOutcome:
    """Worker entry point: classifies one batch, returning whatever it classified.

    Never raises. An expected failure (an API error, a refusal, an invalid reply twice, a
    budget refusal) sets the run's stop flag with the failure as the reason; anything else
    sets it too and comes back as the outcome's `failure`, for the main thread to raise
    once everything paid for is stored. Either way the outcome keeps what was classified
    before the failure (a split's first half).
    """
    assert batch_keys
    return _classify_part(ctx, batch_keys, texts, allow_split=True)


def _classify_part(
    ctx: _RunContext, keys: Sequence[str], texts: Mapping[str, str], *, allow_split: bool
) -> _BatchOutcome:
    try:
        return _classify_keys(ctx, keys, texts, allow_split=allow_split)
    except _RunStopped:
        return _BatchOutcome({})
    except AsioDocsError as e:
        ctx.stop.trip(e)
        return _BatchOutcome({})
    except BaseException as e:
        ctx.stop.trip(AsioDocsError(f"a classification worker failed unexpectedly: {e!r}"))
        return _BatchOutcome({}, failure=e)


def _classify_keys(
    ctx: _RunContext, keys: Sequence[str], texts: Mapping[str, str], *, allow_split: bool
) -> _BatchOutcome:
    """Classifies `keys` with one request, resent once after an invalid reply. When the
    reply hits max_tokens and `allow_split`, classifies the two halves instead, which do
    not split again (see the "Run budget" section)."""
    assert keys
    ids, payload = _batch_payload(keys, texts)
    max_tokens = _max_tokens_for(len(keys))

    response = _send(ctx, payload, len(keys))
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
        first = _classify_part(ctx, keys[:mid], texts, allow_split=False)
        second = _classify_part(ctx, keys[mid:], texts, allow_split=False)
        return _BatchOutcome(
            {**first.classified, **second.classified},
            failure=first.failure if first.failure is not None else second.failure,
        )

    _check_refusal(response)
    parsed = _validate_reply(response, ids)
    if parsed is None:
        resent = _send(ctx, payload, len(keys))
        if resent.stop_reason == "max_tokens":
            raise AsioDocsError(
                f"the model hit its max_tokens limit ({max_tokens:,} output tokens) resending a request "
                f"of {len(keys)} entries after an invalid reply"
            )
        _check_refusal(resent)
        parsed = _validate_reply(resent, ids)
        if parsed is None:
            raise AsioDocsError(
                f"the model returned an invalid classification reply twice for a request of "
                f"{len(keys)} entries ({_diagnose_reply(resent, ids)})"
            )
    return _BatchOutcome({key: parsed[id_] for key, id_ in zip(keys, ids, strict=True)})


class _Harvester:
    """Stores finished batches' results, on the thread that owns the sqlite connection
    (connections are not shared across threads), one transaction per batch, so every
    batch harvested stays stored whatever happens to the run afterwards.

    Never raises for a batch's own failure. A store-write failure sets the stop flag with
    that error as the reason, and the batch's results go to the pending directory instead
    (see `_import_pending`), or, if even that fails, to stderr as JSON lines. A worker's
    unexpected exception is kept (the first one) for the caller to raise once every batch
    is harvested. A store write waits for another process's lock no longer than
    `lock_wait_s()` allows, so harvesting never holds the run past its watchdog.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        store_path: Path,
        store: dict[str, Classification],
        texts: Mapping[str, str],
        release_by_key: Mapping[str, Version],
        stop: _StopFlag,
        lock_wait_s: Callable[[], float],
    ) -> None:
        self._conn: Final = conn
        self._store_path: Final = store_path
        self._lock_wait_s: Final = lock_wait_s
        self._store: Final = store
        self._texts: Final = texts
        self._release_by_key: Final = release_by_key
        self._stop: Final = stop
        self._harvested: "set[Future[_BatchOutcome]]" = set()
        self._stored_keys: set[str] = set()
        self._saved_to_pending = 0
        self._printed = 0
        self._first_failure: BaseException | None = None

    @property
    def first_failure(self) -> BaseException | None:
        return self._first_failure

    def __call__(self, future: "Future[_BatchOutcome]") -> None:
        assert future.done()
        if future in self._harvested:
            return
        if future.cancelled():
            self._harvested.add(future)
            return
        try:
            outcome = future.result()
        except BaseException as e:  # _classify_batch never raises; this keeps a bug in it visible
            outcome = _BatchOutcome({}, failure=e)
        if outcome.classified:
            self._store_rows(self._rows(outcome.classified))
        if outcome.failure is not None and self._first_failure is None:
            self._first_failure = outcome.failure
        # Marked only now: an interrupt during the store write rolls its transaction back
        # and leaves this future to be harvested again.
        self._harvested.add(future)

    def save_for_later(self, futures: Sequence["Future[_BatchOutcome]"]) -> None:
        """For a run being abandoned: saves the results of every finished batch not yet
        harvested to the pending directory, without touching the store, whose lock could
        make it wait. The next run imports them."""
        rows: list[_StoreRow] = []
        for future in futures:
            if not future.done() or future.cancelled() or future in self._harvested:
                continue
            try:
                outcome = future.result()
            except BaseException as e:  # _classify_batch never raises; this keeps a bug in it visible
                outcome = _BatchOutcome({}, failure=e)
            rows.extend(self._rows(outcome.classified))
            self._harvested.add(future)
        if rows:
            self._save_elsewhere(rows)

    def progress(self, total: int) -> str:
        parts = [
            f"{len(self._stored_keys)} of {_count(total, 'entry', 'entries')} were classified and stored"
        ]
        if self._saved_to_pending:
            parts.append(
                f"{_count(self._saved_to_pending, 'more was', 'more were')} saved to "
                f"{_pending_dir(self._store_path)} and go into the store at the start of the next run"
            )
        if self._printed:
            parts.append(
                f"{_count(self._printed, 'more', 'more')} could not be saved anywhere and were printed above "
                "as JSON lines"
            )
        return "; ".join(parts)

    def _rows(self, classified: Mapping[str, _RawResult]) -> tuple[_StoreRow, ...]:
        classified_at = time.time()
        return tuple(
            _StoreRow(
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

    def _store_rows(self, rows: Sequence[_StoreRow]) -> None:
        try:
            _upsert_classifications(self._conn, rows, self._store_path, lock_wait_s=self._lock_wait_s())
        except AsioDocsError as e:
            self._stop.trip(e)
            self._save_elsewhere(rows)
            return
        for row in rows:
            self._store[row.key] = row.classification
        self._stored_keys.update(row.key for row in rows)

    def _save_elsewhere(self, rows: Sequence[_StoreRow]) -> None:
        directory = _pending_dir(self._store_path)
        try:
            _save_pending(self._store_path, rows)
        except OSError as e:
            warn(
                f"could not save {_count(len(rows), 'paid-for classification')} to {directory} "
                f"either ({e}); here they are as JSON lines, as a file there would have held them:\n"
                + _pending_lines(rows).rstrip("\n")
            )
            self._printed += len(rows)
            return
        self._saved_to_pending += len(rows)


_Harvest = Callable[["Future[_BatchOutcome]"], None]


@dataclass(frozen=True, slots=True)
class _Waiting:
    """How the main thread waits for a run's batches, and gives up on them.

    Every wait ends at the watchdog (`watchdog_at`, on `clock`), and no batch is harvested
    after it: the run is abandoned instead. Whatever finished is saved for the next run,
    usage is reported, and the process exits.
    """

    harvest: _Harvest
    save_for_later: Callable[[Sequence["Future[_BatchOutcome]"]], None]
    stop: _StopFlag
    watchdog_at: float
    clock: Callable[[], float]
    progress: Callable[[], str]
    report_usage: Callable[[], None]
    exit_process: Callable[[int], NoReturn]

    def await_batches(self, futures: Sequence["Future[_BatchOutcome]"]) -> None:
        """Harvests each batch as it finishes. Once the stop flag is set, cancels the
        batches not yet started and waits only for the ones in flight."""
        from concurrent.futures import as_completed

        stopping = False
        try:
            for future in as_completed(futures, timeout=self._seconds_to_watchdog()):
                self._harvest_unless_watchdog(future, futures)
                if self.stop.is_set() and not stopping:
                    stopping = True
                    for queued in futures:
                        queued.cancel()  # a no-op on a batch already running or done
                    in_flight = sum(1 for f in futures if not f.done())
                    if in_flight:
                        note(
                            f"stopping early: waiting for {_count(in_flight, 'in-flight request')} so "
                            "their results are kept"
                        )
        except TimeoutError:
            self._watchdog_fired(futures)

    def stop_and_harvest(
        self, futures: Sequence["Future[_BatchOutcome]"], *, interrupted: bool
    ) -> Exception | None:
        """After a Ctrl-C (or an unexpected failure) has set the stop flag: cancels the
        batches not yet started, then waits for the ones in flight and harvests every
        finished batch, so their paid-for results are stored. Each in-flight batch ends
        within one attempt's timeout, since the stop flag refuses its further attempts.
        Harvests every batch even when one harvest fails, and returns the first such
        failure.

        A Ctrl-C while waiting abandons the requests in flight, and so does the watchdog.
        """
        from concurrent.futures import as_completed

        assert self.stop.is_set()
        first_error: Exception | None = None

        def harvest_one(future: "Future[_BatchOutcome]") -> None:
            nonlocal first_error
            try:
                self._harvest_unless_watchdog(future, futures)
            except Exception as e:
                if first_error is None:
                    first_error = e

        try:
            for future in futures:
                future.cancel()
            in_flight = [future for future in futures if not future.done()]
            if in_flight:
                again = "again " if interrupted else ""
                spend(
                    f"stopping: waiting for {_count(len(in_flight), 'in-flight request')} so their "
                    f"results are kept; press Ctrl-C {again}to abandon them"
                )
            for future in futures:
                if future.done() and not future.cancelled():
                    harvest_one(future)
            for future in as_completed(in_flight, timeout=self._seconds_to_watchdog()):
                harvest_one(future)
        except KeyboardInterrupt:
            self.abandon(futures, status=130, message=None)
        except TimeoutError:
            self._watchdog_fired(futures)
        return first_error

    def abandon(
        self, futures: Sequence["Future[_BatchOutcome]"], *, status: int, message: str | None
    ) -> NoReturn:
        """Stops waiting and exits the process at once with `status`: a normal exit would
        wait for the (not daemonic) worker threads.

        Before exiting it saves every finished batch not harvested yet to the pending
        directory (never to the store, whose lock could make it wait), prints `message` as
        an error followed by what was stored or, without one, a note that the requests in
        flight were abandoned, and reports usage, counting what is still in flight as
        possibly billed. Ctrl-C is ignored meanwhile, so it cannot cut the save short.
        """
        with _sigint_ignored():
            try:
                in_flight = sum(1 for future in futures if not future.done())
                try:
                    self.save_for_later(futures)
                except Exception as e:
                    warn(f"could not save the finished batches while abandoning the run: {e!r}")
                if message is not None:
                    error(f"{message}; {self.progress()}. {_RERUN_HINT}")
                else:
                    spend(
                        f"abandoned {_count(in_flight, 'in-flight request')}, which may still be billed; "
                        f"{self.progress()}"
                    )
                self.report_usage()
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                self.exit_process(status)

    def _harvest_unless_watchdog(
        self, future: "Future[_BatchOutcome]", futures: Sequence["Future[_BatchOutcome]"]
    ) -> None:
        # as_completed yields every future already finished before checking its timeout.
        if self._seconds_to_watchdog() <= 0:
            self._watchdog_fired(futures)
        self.harvest(future)

    def _seconds_to_watchdog(self) -> float:
        return max(0.0, self.watchdog_at - self.clock())

    def _watchdog_fired(self, futures: Sequence["Future[_BatchOutcome]"]) -> NoReturn:
        in_flight = sum(1 for future in futures if not future.done())
        self.abandon(
            futures,
            status=1,
            message=(
                f"the run reached its {_watchdog_s():.0f} s wall-clock limit and abandoned "
                f"{_count(in_flight, 'request')} still in flight, which may still be billed"
            ),
        )


_RERUN_HINT: Final = "Rerunning continues where this run stopped: stored entries are never sent again."

# The process exit used when a run is abandoned (see _Waiting.abandon); tests replace it.
_exit_process: Callable[[int], NoReturn] = os._exit


def _run_batches(
    conn: sqlite3.Connection,
    store_path: Path,
    pending: Sequence[str],
    texts: Mapping[str, str],
    release_by_key: Mapping[str, Version],
    store: dict[str, Classification],
    client_factory: ClientFactory,
) -> None:
    """Classifies the `pending` keys into `store` (and the store file) within one run budget."""
    # Imported only when there is something to send: the import costs every cached run
    # (and every other command) tens of milliseconds.
    from concurrent.futures import ThreadPoolExecutor

    assert pending
    batches = tuple(tuple(pending[i : i + _BATCH_SIZE]) for i in range(0, len(pending), _BATCH_SIZE))
    payload_bytes = tuple(len(_batch_payload(batch, texts)[1].encode()) for batch in batches)
    caps = _caps_for_run(len(pending), tuple(_REQUEST_INPUT_BASE + size for size in payload_bytes))
    # Nothing is sent, and nothing about sending is printed, unless the run's worst case is
    # affordable and its results can be stored.
    _check_cost_ceiling(caps)
    _check_store_writable(conn, store_path)
    client = client_factory()

    stop = _StopFlag()
    budget = _RunBudget(caps, stop)
    ctx = _RunContext(client=client, budget=budget)
    watchdog_at = budget.started_at + _watchdog_s()

    def store_lock_wait_s() -> float:
        return min(_STORE_LOCK_WAIT_S, max(0.0, watchdog_at - budget.clock()))

    harvest = _Harvester(conn, store_path, store, texts, release_by_key, stop, store_lock_wait_s)
    usage_reported = False

    def report_usage() -> None:
        nonlocal usage_reported
        if not usage_reported:
            usage_reported = True
            spend(_usage_line(budget.usage()))

    waiting = _Waiting(
        harvest=harvest,
        save_for_later=harvest.save_for_later,
        stop=stop,
        watchdog_at=watchdog_at,
        clock=budget.clock,
        progress=lambda: harvest.progress(len(pending)),
        report_usage=report_usage,
        exit_process=_exit_process,
    )
    spend(_preview_line(payload_bytes, len(pending), caps))
    try:
        # Workers take no batch until every batch is submitted: a Ctrl-C landing mid-submit
        # (possibly between a submit and its append) then cannot leave a request in flight
        # whose future `futures` is missing, since no request has been sent yet.
        all_submitted = threading.Event()
        pool = ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(batches)), initializer=all_submitted.wait)
        futures: "list[Future[_BatchOutcome]]" = []
        try:
            for batch in batches:
                futures.append(pool.submit(_classify_batch, ctx, batch, texts))
            all_submitted.set()
            waiting.await_batches(futures)
        # Not SystemExit and the like: those end the process, and draining would only delay it.
        except (KeyboardInterrupt, Exception) as e:
            interrupted = isinstance(e, KeyboardInterrupt)
            stop.trip(
                AsioDocsError("interrupted" if interrupted else f"stopped by an unexpected error: {e!r}")
            )
            all_submitted.set()  # the workers find the stop flag set and send nothing
            harvest_error = waiting.stop_and_harvest(futures, interrupted=interrupted)
            if interrupted:
                note(f"interrupted: {harvest.progress(len(pending))}. {_RERUN_HINT}")
            if harvest_error is not None:
                e.add_note(f"storing the batches in flight afterwards also failed: {harvest_error!r}")
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        assert budget.usage().in_flight_requests == 0, "every attempt is settled once its batch is done"
    finally:
        report_usage()

    # Everything paid for is stored (or saved) by now; a worker's unexpected failure is a
    # bug, raised with its traceback, ahead of the stop flag's reason.
    if harvest.first_failure is not None:
        raise harvest.first_failure
    reason = stop.reason
    if reason is not None:
        raise AsioDocsError(f"{reason}; {harvest.progress(len(pending))}. {_RERUN_HINT}") from reason


def classify(
    items: Sequence[ClassifyItem],
    *,
    client_factory: ClientFactory = default_client_factory,
    store_path: Path | None = None,
) -> tuple[Classification, ...]:
    """Classifies each item, returning one Classification per item in the same order.

    Results already in the store are reused; only entries not stored yet are sent to the
    model, in batches of up to 40 with several in flight at once, within one run budget
    (see the "Run budget" section). Each batch's results are stored as soon as it
    finishes, so a run that stops early keeps everything it paid for, and rerunning
    sends only what is still missing. Results an earlier run could not store (see
    `_import_pending`) go into the store first.

    A run that stops before classifying everything (an error, a hard cap, a store-write
    failure) raises AsioDocsError naming the reason and how many entries were stored. A
    first Ctrl-C waits for the requests in flight and stores their results, then
    re-raises KeyboardInterrupt; a second one exits the process at once (status 130), as
    does the wall-clock watchdog (status 1).
    """
    resolved_store_path = store_path if store_path is not None else default_store_path()
    conn = _connect(resolved_store_path)
    try:
        _import_pending(conn, resolved_store_path)
        keys = tuple(cache_key(item.entry) for item in items)
        unique_texts: dict[str, str] = {}
        release_by_key: dict[str, Version] = {}
        for key, item in zip(keys, items, strict=True):
            unique_texts.setdefault(key, item.entry.full_text())
            release_by_key.setdefault(key, item.release)

        store = _load_classifications(conn, tuple(unique_texts), resolved_store_path)
        pending = tuple(k for k in unique_texts if k not in store)
        if pending:
            _run_batches(
                conn, resolved_store_path, pending, unique_texts, release_by_key, store, client_factory
            )
        elif unique_texts:
            note(
                f"all {_count(len(unique_texts), 'entry', 'entries')} already classified in the store; "
                "no API requests needed"
            )
        return tuple(store[key] for key in keys)
    finally:
        conn.close()
