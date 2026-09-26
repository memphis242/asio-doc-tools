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

import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

from . import paths
from .diag import AsioDocsError, note, spend
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


def _upsert_classifications(
    conn: sqlite3.Connection, rows: Sequence[tuple[str, Classification, str]], path: Path
) -> None:
    """Writes `rows` in one transaction: on any failure none of them are stored."""
    assert rows
    try:
        with conn:
            conn.executemany(
                _UPSERT_SQL,
                [
                    (
                        key,
                        c.category.value,
                        int(c.breaking),
                        c.breaking_reason,
                        c.model,
                        c.effort,
                        PROMPT_VERSION,
                        c.release,
                        text,
                        c.classified_at,
                    )
                    for key, c, text in rows
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
    classified yet.
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


def _build_payload(ids: Sequence[str], batch_keys: Sequence[str], texts: Mapping[str, str]) -> str:
    items = [{"id": id_, "text": texts[key]} for id_, key in zip(ids, batch_keys, strict=True)]
    return json.dumps({"items": items}, indent=2)


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
# Caps per run, for N entries to classify in B = ceil(N / 40) batches:
#
#   requests       R = 2 B + 4
#   output tokens  T = (min(8, B) + 3) * 7,168 + 128 N
#   time           no attempt starts more than 300 s after the run started
#
# Before sending, an attempt reserves its whole max_tokens, and is refused unless billed
# + possibly billed + reserved + its own max_tokens <= T. A response turns its
# reservation into its actual usage.output_tokens; a failure that generated nothing (an
# HTTP error status, a connection that was never made) refunds it; a timeout or a
# connection dropped mid-request keeps it as possibly billed. The API never generates
# more than max_tokens, so billed output <= T, and requests <= R, whatever happens.
#
# Why these numbers. Up to 8 requests are in flight, each reserving up to 7,168, so a
# run needs min(8, B) * 7,168 of headroom for reservations alone. On top of that, 128 per
# entry is 4x the normal 32, and 3 more full-size max_tokens cover a max_tokens stop (up
# to 7,168 billed) plus the halves and the resend that follow it. A normal run peaks at
# about 32 N billed plus min(8, B) * 7,168 reserved, which is 96 N + 21,504 under T. R
# gives every batch a second request plus 4 spare, so even a one-batch run absorbs a
# split (3 requests), a resend (1), and 2 retries.
#
# Worst case, at Sonnet 5's $2 / $10 per million input / output tokens. Input is not
# capped directly: it is at most R requests of one batch each, estimated at 4,100
# tokens for a full batch (about 2,100 for the system prompt and schema, plus about 50
# per entry of average length).
#
#                       requests  output tokens  input tokens (est.)  cost at most
#   N entries           R         T              4,100 R              T * $10/M + 4,100 R * $2/M
#   one batch of 28     6         32,256         24,600               about $0.37
#   full history, 995   54        206,208        221,400              about $2.50
#
# A normal full-history run: 25 requests, about 32K output and 102K input tokens, $0.52.
#
# Time. A request times out after 240 s without receiving a byte (a non-streaming reply
# arrives whole once generated, and a full-size 7,168 tokens takes about 57 s at the
# observed 126 tokens/s: about 4x margin), or after 10 s trying to connect. Retry waits
# end at the time limit, so the last attempt starts by 300 s and ends by about 550 s
# even if the API hangs.
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

_REQUESTS_PER_BATCH_CAP: Final = 2
_SPARE_REQUESTS_CAP: Final = 4
_SPARE_FULL_SIZE_OUTPUTS_CAP: Final = 3
_OUTPUT_TOKENS_PER_ENTRY_CAP: Final = 128
_RUN_TIME_LIMIT_S: Final = 300.0

_REQUEST_TIMEOUT_S: Final = 240.0
_CONNECT_TIMEOUT_S: Final = 10.0
_MAX_ATTEMPTS: Final = 3
_RETRY_BASE_DELAY_S: Final = 2.0
_RETRY_MAX_DELAY_S: Final = 30.0
# Retried as well: every 5xx status (529, overloaded, included).
_RETRYABLE_STATUSES: Final = frozenset({408, 409, 429})

# Observed at low effort, for the estimates printed before and after a run: about 32
# output tokens per entry, thinking included; and input of 3,267 tokens for a request of
# 28 entries whose payload was 4,649 characters, which at about 4 characters per token
# leaves about 2,100 for the system prompt and schema. The whole history averages about
# 200 payload characters, so about 50 tokens, per entry.
_EXPECTED_OUTPUT_TOKENS_PER_ENTRY: Final = 32
_ESTIMATED_INPUT_TOKENS_PER_REQUEST: Final = 2_100
_ESTIMATED_INPUT_TOKENS_PER_ENTRY: Final = 50
# Claude Sonnet 5 list prices in US dollars per million tokens, for those estimates only:
# nothing is limited in dollars (the hard caps are in requests, tokens, and time).
_INPUT_USD_PER_MTOK: Final = 2.0
_OUTPUT_USD_PER_MTOK: Final = 10.0


def _max_tokens_for(entry_count: int) -> int:
    """max_tokens for a request classifying `entry_count` entries."""
    assert entry_count >= 1
    max_tokens = _MAX_TOKENS_BASE + _MAX_TOKENS_PER_ENTRY * entry_count
    assert max_tokens <= _SDK_NONSTREAMING_MAX_TOKENS
    return max_tokens


@dataclass(frozen=True, slots=True)
class _Caps:
    max_requests: int
    max_output_tokens: int
    time_limit_s: float


def _caps_for_run(entries: int, batches: int) -> _Caps:
    assert 1 <= batches <= entries
    return _Caps(
        max_requests=_REQUESTS_PER_BATCH_CAP * batches + _SPARE_REQUESTS_CAP,
        max_output_tokens=(min(_MAX_WORKERS, batches) + _SPARE_FULL_SIZE_OUTPUTS_CAP)
        * _max_tokens_for(_BATCH_SIZE)
        + _OUTPUT_TOKENS_PER_ENTRY_CAP * entries,
        time_limit_s=_RUN_TIME_LIMIT_S,
    )


# The arithmetic in the "Run budget" comment above, checked at import time.
assert _max_tokens_for(_BATCH_SIZE) == 7_168 <= _SDK_NONSTREAMING_MAX_TOKENS
assert _max_tokens_for(_BATCH_SIZE) >= 5 * _EXPECTED_OUTPUT_TOKENS_PER_ENTRY * _BATCH_SIZE
assert _OUTPUT_TOKENS_PER_ENTRY_CAP >= 4 * _EXPECTED_OUTPUT_TOKENS_PER_ENTRY
assert _caps_for_run(995, 25) == _Caps(max_requests=54, max_output_tokens=206_208, time_limit_s=300.0)
# A one-batch run absorbs a split, a resend, and a retry: requests 1 + 2 + 1 + 1, and for
# output, the whole batch billed at max_tokens plus both halves and a resend reserved at once.
assert _caps_for_run(_BATCH_SIZE, 1).max_requests >= 5
assert (
    _max_tokens_for(_BATCH_SIZE) + 3 * _max_tokens_for(_BATCH_SIZE // 2)
    <= _caps_for_run(_BATCH_SIZE, 1).max_output_tokens
)
# A request times out after at least 4x the time it takes to generate its max_tokens at
# the observed 126 tokens/s.
assert _REQUEST_TIMEOUT_S >= 4 * _max_tokens_for(_BATCH_SIZE) / 126
assert _MAX_ATTEMPTS == 3 and _RETRY_BASE_DELAY_S * 2 ** (_MAX_ATTEMPTS - 2) <= _RETRY_MAX_DELAY_S


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural if plural is not None else singular + 's')}"


def _about_usd(amount: float) -> str:
    assert amount >= 0
    if amount < 0.001:
        return "under $0.001"
    return f"about ${amount:.3f}" if amount < 0.1 else f"about ${amount:.2f}"


def _estimated_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _INPUT_USD_PER_MTOK + output_tokens * _OUTPUT_USD_PER_MTOK) / 1_000_000


def _estimated_input_tokens(entry_count: int) -> int:
    return _ESTIMATED_INPUT_TOKENS_PER_REQUEST + _ESTIMATED_INPUT_TOKENS_PER_ENTRY * entry_count


def _max_cost_usd(caps: _Caps) -> float:
    """The most a run under `caps` can cost: exact for output, estimated for input."""
    return _estimated_cost_usd(
        caps.max_requests * _estimated_input_tokens(_BATCH_SIZE), caps.max_output_tokens
    )


# The full history's worst case, from the "Run budget" comment above: about $2.50.
assert _max_cost_usd(_caps_for_run(995, 25)) < 3.0


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
    """One admitted attempt, holding its max_tokens until it is settled."""

    serial: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class _Usage:
    requests: int  # HTTP attempts admitted, whatever their outcome
    input_tokens: int
    output_tokens: int  # billed, as reported by responses
    cut_off_requests: int  # attempts that failed after possibly generating output
    cut_off_output_tokens: int  # their max_tokens, counted as possibly billed
    reserved_output_tokens: int  # max_tokens of attempts still in flight

    def estimated_cost_usd(self) -> float:
        return _estimated_cost_usd(self.input_tokens, self.output_tokens + self.cut_off_output_tokens)


class _RunBudget:
    """A run's hard caps, consulted before every HTTP attempt (thread-safe).

    admit() either reserves an attempt's max_tokens or refuses the attempt; a refusal
    for a cap sets the stop flag with that cap as its reason. Every admitted attempt is
    then settled exactly once, by settle_billed() or settle_failed().
    """

    def __init__(self, caps: _Caps, stop: _StopFlag, clock: Callable[[], float] = time.monotonic) -> None:
        self._caps: Final = caps
        self._stop: Final = stop
        self._clock: Final = clock
        self._deadline: Final = clock() + caps.time_limit_s
        self._lock: Final = threading.Lock()
        self._open: set[int] = set()  # serials of admitted, unsettled attempts
        self._requests = 0
        self._input_tokens = 0
        self._billed_output = 0
        self._cut_off_requests = 0
        self._cut_off_output = 0
        self._reserved_output = 0

    @property
    def caps(self) -> _Caps:
        return self._caps

    @property
    def stop(self) -> _StopFlag:
        return self._stop

    def admit(self, max_tokens: int) -> _Reservation:
        """Reserves one attempt, or raises _RunStopped: when the stop flag is already set,
        or when this attempt would pass a cap (which sets the stop flag)."""
        assert max_tokens > 0
        with self._lock:
            if self._stop.is_set():
                raise _RunStopped
            refusal = self._refusal(max_tokens)
            if refusal is not None:
                self._stop.trip(refusal)
                raise _RunStopped
            self._requests += 1
            self._reserved_output += max_tokens
            self._open.add(self._requests)
            return _Reservation(serial=self._requests, max_tokens=max_tokens)

    def settle_billed(self, reservation: _Reservation, *, input_tokens: int, output_tokens: int) -> None:
        """A response arrived: its reservation becomes the output it reports."""
        with self._lock:
            self._release(reservation)
            self._input_tokens += input_tokens
            self._billed_output += output_tokens

    def settle_failed(self, reservation: _Reservation, *, possibly_billed: bool) -> None:
        """No usable response: the reservation is refunded, or kept as possibly billed."""
        with self._lock:
            self._release(reservation)
            if possibly_billed:
                self._cut_off_requests += 1
                self._cut_off_output += reservation.max_tokens

    def seconds_left(self) -> float:
        return max(0.0, self._deadline - self._clock())

    def usage(self) -> _Usage:
        with self._lock:
            return _Usage(
                requests=self._requests,
                input_tokens=self._input_tokens,
                output_tokens=self._billed_output,
                cut_off_requests=self._cut_off_requests,
                cut_off_output_tokens=self._cut_off_output,
                reserved_output_tokens=self._reserved_output,
            )

    def _release(self, reservation: _Reservation) -> None:
        assert reservation.serial in self._open, "an attempt was settled twice"
        self._open.remove(reservation.serial)
        self._reserved_output -= reservation.max_tokens
        assert self._reserved_output >= 0

    def _refusal(self, max_tokens: int) -> AsioDocsError | None:
        caps = self._caps
        committed = self._billed_output + self._cut_off_output + self._reserved_output
        if self._clock() >= self._deadline:
            return AsioDocsError(
                f"stopped at this run's time limit of {caps.time_limit_s:.0f} s ({self._used()})"
            )
        if self._requests >= caps.max_requests:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_requests} API requests ({self._used()})"
            )
        if committed + max_tokens > caps.max_output_tokens:
            return AsioDocsError(
                f"stopped at this run's hard cap of {caps.max_output_tokens:,} output tokens: the next "
                f"request could generate up to {max_tokens:,} on top of {committed:,} already billed or "
                f"reserved by requests in flight ({self._used()})"
            )
        return None

    def _used(self) -> str:
        cut_off = (
            f", up to {self._cut_off_output:,} more possibly billed by "
            f"{_count(self._cut_off_requests, 'request')} cut off"
            if self._cut_off_requests
            else ""
        )
        return (
            f"{_count(self._requests, 'request')} used, {self._billed_output:,} output tokens billed{cut_off}"
        )


def _preview_line(batches: Sequence[Sequence[str]], caps: _Caps) -> str:
    entries = sum(len(batch) for batch in batches)
    expected_cost = _estimated_cost_usd(
        sum(_estimated_input_tokens(len(batch)) for batch in batches),
        _EXPECTED_OUTPUT_TOKENS_PER_ENTRY * entries,
    )
    return (
        f"classifying {_count(entries, 'entry', 'entries')} in {_count(len(batches), 'batch', 'batches')} "
        f"with {MODEL} ({EFFORT} effort): expect {_count(len(batches), 'request')}, "
        f"{_about_usd(expected_cost)}; hard caps {_count(caps.max_requests, 'request')}, "
        f"{caps.max_output_tokens:,} output tokens, {caps.time_limit_s:.0f} s: at most "
        f"{_about_usd(_max_cost_usd(caps))}"
    )


def _usage_line(usage: _Usage) -> str:
    cut_off = (
        f" (plus up to {usage.cut_off_output_tokens:,} output tokens for "
        f"{_count(usage.cut_off_requests, 'request')} cut off before reporting usage)"
        if usage.cut_off_requests
        else ""
    )
    return (
        f"classification used {_count(usage.requests, 'request')}: {usage.input_tokens:,} input and "
        f"{usage.output_tokens:,} output tokens{cut_off}, {_about_usd(usage.estimated_cost_usd())}"
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


@dataclass(frozen=True, slots=True)
class _RunContext:
    """What every worker of one classify() run shares."""

    client: Any
    budget: _RunBudget

    @property
    def stop(self) -> _StopFlag:
        return self.budget.stop


def _send(ctx: _RunContext, payload: str, entry_count: int) -> Any:
    """One logical request: up to _MAX_ATTEMPTS HTTP attempts, each admitted by the budget.

    Raises _RunStopped when the budget refuses an attempt, and AsioDocsError when a
    failure is not retryable or outlasts the retries.
    """
    import anthropic

    max_tokens = _max_tokens_for(entry_count)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        reservation = ctx.budget.admit(max_tokens)
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
                reservation, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
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


def _classify_batch(
    ctx: _RunContext, batch_keys: Sequence[str], texts: Mapping[str, str]
) -> dict[str, _RawResult]:
    """Worker entry point: classifies one batch, returning whatever it classified.

    An expected failure (an API error, a refusal, an invalid reply twice, a budget
    refusal) does not raise: it sets the run's stop flag with the failure as the reason,
    and the batch returns what it classified before it (a split's first half), so the
    caller still stores everything paid for. Anything else also sets the stop flag, so
    that no other worker sends another request, and then propagates.
    """
    assert batch_keys
    try:
        return _classify_part(ctx, batch_keys, texts, allow_split=True)
    except BaseException as e:
        ctx.stop.trip(AsioDocsError(f"a classification worker failed unexpectedly: {e!r}"))
        raise


def _classify_part(
    ctx: _RunContext, keys: Sequence[str], texts: Mapping[str, str], *, allow_split: bool
) -> dict[str, _RawResult]:
    try:
        return _classify_keys(ctx, keys, texts, allow_split=allow_split)
    except _RunStopped:
        return {}
    except AsioDocsError as e:
        ctx.stop.trip(e)
        return {}


def _classify_keys(
    ctx: _RunContext, keys: Sequence[str], texts: Mapping[str, str], *, allow_split: bool
) -> dict[str, _RawResult]:
    """Classifies `keys` with one request, resent once after an invalid reply. When the
    reply hits max_tokens and `allow_split`, classifies the two halves instead, which do
    not split again (see the "Run budget" section)."""
    assert keys
    ids = tuple(f"e{i}" for i in range(len(keys)))
    payload = _build_payload(ids, keys, texts)
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
        # After a failed first half the stop flag is set, so the second half sends nothing.
        first = _classify_part(ctx, keys[:mid], texts, allow_split=False)
        second = _classify_part(ctx, keys[mid:], texts, allow_split=False)
        return {**first, **second}

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
    return {key: parsed[id_] for key, id_ in zip(keys, ids, strict=True)}


class _Harvester:
    """Stores finished batches' results, on the thread that owns the sqlite connection
    (connections are not shared across threads), one transaction per batch, so every
    batch harvested stays stored whatever happens to the run afterwards.

    A store-write failure does not raise: it sets the stop flag with that error as the
    reason, and the results that could not be stored are counted for the final report
    (the next run sends those entries again).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        store_path: Path,
        store: dict[str, Classification],
        texts: Mapping[str, str],
        release_by_key: Mapping[str, Version],
        stop: _StopFlag,
    ) -> None:
        self._conn: Final = conn
        self._store_path: Final = store_path
        self._store: Final = store
        self._texts: Final = texts
        self._release_by_key: Final = release_by_key
        self._stop: Final = stop
        self._harvested: "set[Future[dict[str, _RawResult]]]" = set()
        self._stored_keys: set[str] = set()
        self._unstored = 0

    def __call__(self, future: "Future[dict[str, _RawResult]]") -> None:
        assert future.done()
        if future in self._harvested:
            return
        if future.cancelled():
            self._harvested.add(future)
            return
        try:
            classified = future.result()
        except BaseException:
            self._harvested.add(future)  # a worker's unexpected failure, raised once
            raise
        if classified:
            try:
                self._write(classified)
            except AsioDocsError as e:
                self._stop.trip(e)
                self._unstored += len(classified)
        # Marked only now: an interrupt during the write rolls its transaction back and
        # leaves this future to be harvested again.
        self._harvested.add(future)

    def progress(self, total: int) -> str:
        unstored = (
            f"; {_count(self._unstored, 'more entry', 'more entries')} could not be stored"
            if self._unstored
            else ""
        )
        stored = f"{len(self._stored_keys)} of {_count(total, 'entry', 'entries')}"
        return f"{stored} were classified and stored{unstored}"

    def _write(self, classified: Mapping[str, _RawResult]) -> None:
        classified_at = time.time()
        rows = tuple(
            (
                key,
                Classification(
                    category=result.category,
                    breaking=result.breaking,
                    breaking_reason=result.breaking_reason,
                    model=MODEL,
                    effort=EFFORT,
                    release=str(self._release_by_key[key]),
                    classified_at=classified_at,
                ),
                self._texts[key],
            )
            for key, result in classified.items()
        )
        _upsert_classifications(self._conn, rows, self._store_path)
        for key, classification, _text in rows:
            self._store[key] = classification
        self._stored_keys.update(classified)


_Harvest = Callable[["Future[dict[str, _RawResult]]"], None]


def _await_batches(
    futures: Sequence["Future[dict[str, _RawResult]]"], harvest: _Harvest, stop: _StopFlag
) -> None:
    """Harvests each batch as it finishes. Once the stop flag is set, cancels the batches
    not yet started and waits only for the ones in flight."""
    from concurrent.futures import as_completed

    stopping = False
    for future in as_completed(futures):
        harvest(future)
        if stop.is_set() and not stopping:
            stopping = True
            for queued in futures:
                queued.cancel()  # a no-op on a batch already running or done
            in_flight = sum(1 for f in futures if not f.done())
            if in_flight:
                note(
                    f"stopping early: waiting for {_count(in_flight, 'in-flight request')} so their "
                    "results are kept"
                )


def _stop_and_harvest(
    futures: Sequence["Future[dict[str, _RawResult]]"],
    harvest: _Harvest,
    *,
    stop: _StopFlag,
    interrupted: bool,
    report_usage: Callable[[], None],
    exit_process: Callable[[int], NoReturn],
) -> None:
    """After a Ctrl-C (or an unexpected failure) has set the stop flag: cancels the
    batches not yet started, then waits for the ones in flight and harvests every
    finished batch, so their paid-for results are stored. Each in-flight batch ends
    within one attempt's timeout, since the stop flag refuses its further attempts.

    A Ctrl-C while waiting abandons the requests in flight: usage is reported and the
    process exits at once with status 130, since a normal exit would wait for the (not
    daemonic) worker threads. What was harvested is already committed.
    """
    from concurrent.futures import as_completed

    assert stop.is_set()
    try:
        for future in futures:
            future.cancel()
        in_flight = [future for future in futures if not future.done()]
        if in_flight:
            again = "again " if interrupted else ""
            spend(
                f"stopping: waiting for {_count(len(in_flight), 'in-flight request')} so their results "
                f"are kept; press Ctrl-C {again}to abandon them"
            )
        for future in futures:
            if future.done() and not future.cancelled():
                harvest(future)
        for future in as_completed(in_flight):
            harvest(future)
    except KeyboardInterrupt:
        report_usage()
        exit_process(130)


_RERUN_HINT: Final = "Rerunning continues where this run stopped: stored entries are never sent again."


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
    caps = _caps_for_run(len(pending), len(batches))
    stop = _StopFlag()
    budget = _RunBudget(caps, stop)
    harvest = _Harvester(conn, store_path, store, texts, release_by_key, stop)
    usage_reported = False

    def report_usage() -> None:
        nonlocal usage_reported
        if not usage_reported:
            usage_reported = True
            spend(_usage_line(budget.usage()))

    ctx = _RunContext(client=client_factory(), budget=budget)
    spend(_preview_line(batches, caps))
    try:
        # Workers take no batch until every batch is submitted: a Ctrl-C landing mid-submit
        # (possibly between a submit and its append) then cannot leave a request in flight
        # whose future `futures` is missing, since no request has been sent yet.
        all_submitted = threading.Event()
        pool = ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(batches)), initializer=all_submitted.wait)
        futures: "list[Future[dict[str, _RawResult]]]" = []
        try:
            for batch in batches:
                futures.append(pool.submit(_classify_batch, ctx, batch, texts))
            all_submitted.set()
            _await_batches(futures, harvest, stop)
        except BaseException as e:
            interrupted = isinstance(e, KeyboardInterrupt)
            stop.trip(
                AsioDocsError("interrupted" if interrupted else f"stopped by an unexpected error: {e!r}")
            )
            all_submitted.set()  # the workers find the stop flag set and send nothing
            _stop_and_harvest(
                futures,
                harvest,
                stop=stop,
                interrupted=interrupted,
                report_usage=report_usage,
                exit_process=os._exit,
            )
            if interrupted:
                note(f"interrupted: {harvest.progress(len(pending))}. {_RERUN_HINT}")
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        assert budget.usage().reserved_output_tokens == 0, "every attempt is settled once its batch is done"
    finally:
        report_usage()

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
    sends only what is still missing.

    A run that stops before classifying everything (an error, a hard cap, a store-write
    failure) raises AsioDocsError naming the reason and how many entries were stored. A
    first Ctrl-C waits for the requests in flight and stores their results, then
    re-raises KeyboardInterrupt; a second one exits the process at once (status 130).
    """
    resolved_store_path = store_path if store_path is not None else default_store_path()
    conn = _connect(resolved_store_path)
    try:
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
