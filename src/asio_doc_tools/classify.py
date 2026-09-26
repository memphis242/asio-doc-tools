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
"""

import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import anthropic

from . import paths
from .diag import AsioDocsError, note
from .history import Entry
from .versions import Version

# Bump whenever the system prompt's meaning changes, so previously cached
# classifications (keyed on this value) are re-asked rather than reused stale.
PROMPT_VERSION: Final = "classify-2"

# Classification uses exactly this one model, at this one effort level, always - no
# per-call model choice and no server-side fallback to a different model.
MODEL: Final = "claude-sonnet-5"
EFFORT: Final = "low"
_MAX_TOKENS: Final = 16000
_BATCH_SIZE: Final = 40
_MAX_WORKERS: Final = 4


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


def default_client_factory() -> Any:
    return anthropic.Anthropic(max_retries=4)


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


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    if version == 0:
        with conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    elif version > _SCHEMA_VERSION:
        raise AsioDocsError(
            f"the classification store at {path} has schema version {version}, newer than this "
            f"version of asio-doc-tools understands (up to {_SCHEMA_VERSION}); upgrade "
            "asio-doc-tools before using it"
        )


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
        raise AsioDocsError(
            f"the classification store at {path} is corrupt or unreadable ({e}); it holds "
            "paid-for API results, so fix or remove that file manually before continuing"
        ) from e
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


def _load_classifications(conn: sqlite3.Connection, keys: Sequence[str]) -> dict[str, Classification]:
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
    except sqlite3.OperationalError as e:
        raise AsioDocsError(f"could not read the classification store: {e}") from e
    return result


def _upsert_classifications(conn: sqlite3.Connection, rows: Sequence[tuple[str, Classification, str]]) -> None:
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
    except sqlite3.OperationalError as e:
        raise AsioDocsError(f"could not write to the classification store (still locked?): {e}") from e


def stored_keys(store_path: Path | None = None) -> frozenset[str]:
    """Every key currently in the store, for read-only inspection (e.g. `releases`)."""
    conn = _connect(store_path if store_path is not None else default_store_path())
    try:
        try:
            rows = conn.execute("SELECT key FROM classifications").fetchall()
        except sqlite3.OperationalError as e:
            raise AsioDocsError(f"could not read the classification store: {e}") from e
        return frozenset(row[0] for row in rows)
    finally:
        conn.close()


def _build_payload(ids: Sequence[str], batch_keys: Sequence[str], texts: dict[str, str]) -> str:
    items = [{"id": id_, "text": texts[key]} for id_, key in zip(ids, batch_keys, strict=True)]
    return json.dumps({"items": items}, indent=2)


def _send(client: Any, payload: str) -> Any:
    try:
        return client.messages.create(
            model=MODEL,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
        )
    except anthropic.AuthenticationError as e:
        raise AsioDocsError(
            f"Anthropic API authentication failed ({_request_id(e)}); set ANTHROPIC_API_KEY "
            "or run 'ant auth login'."
        ) from e
    except anthropic.PermissionDeniedError as e:
        raise AsioDocsError(f"Anthropic API denied permission ({_request_id(e)}): {e}") from e
    except anthropic.NotFoundError as e:
        raise AsioDocsError(
            f"Anthropic API could not find model {MODEL!r} ({_request_id(e)})."
        ) from e
    except anthropic.RateLimitError as e:
        raise AsioDocsError(f"Anthropic API rate limit exceeded ({_request_id(e)}): {e}") from e
    except anthropic.BadRequestError as e:
        raise AsioDocsError(f"Anthropic API rejected the request ({_request_id(e)}): {e}") from e
    except anthropic.APIStatusError as e:
        raise AsioDocsError(
            f"Anthropic API error (status {e.status_code}, request {_request_id(e)}): {e}"
        ) from e
    except anthropic.APIConnectionError as e:
        raise AsioDocsError(f"could not reach the Anthropic API: {e}") from e


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


def _classify_batch(
    client: Any, batch_keys: Sequence[str], texts: dict[str, str]
) -> tuple[dict[str, _RawResult], int, int]:
    assert batch_keys
    ids = [f"e{i}" for i in range(len(batch_keys))]
    payload = _build_payload(ids, batch_keys, texts)

    response = _send(client, payload)
    if response.stop_reason == "max_tokens":
        if len(batch_keys) == 1:
            raise AsioDocsError(
                "classifying a single entry hit the model's max_tokens limit; entry text: "
                f"{texts[batch_keys[0]]!r}"
            )
        mid = len(batch_keys) // 2
        left, left_in, left_out = _classify_batch(client, batch_keys[:mid], texts)
        right, right_in, right_out = _classify_batch(client, batch_keys[mid:], texts)
        return {**left, **right}, left_in + right_in, left_out + right_out

    _check_refusal(response)
    parsed = _validate_reply(response, ids)
    if parsed is None:
        retry_response = _send(client, payload)
        if retry_response.stop_reason == "max_tokens":
            raise AsioDocsError(
                f"the model hit max_tokens retrying a batch of {len(batch_keys)} entries after "
                "an invalid reply"
            )
        _check_refusal(retry_response)
        parsed = _validate_reply(retry_response, ids)
        if parsed is None:
            raise AsioDocsError(
                f"the model returned an invalid classification reply twice for a batch of "
                f"{len(batch_keys)} entries ({_diagnose_reply(retry_response, ids)})"
            )
        response = retry_response

    results = {batch_keys[i]: parsed[ids[i]] for i in range(len(batch_keys))}
    usage = response.usage
    return results, usage.input_tokens, usage.output_tokens


def classify(
    items: Sequence[ClassifyItem],
    *,
    client_factory: ClientFactory = default_client_factory,
    store_path: Path | None = None,
    reclassify: bool = False,
) -> tuple[Classification, ...]:
    """Classifies each item, returning one Classification per item in the same order.

    Results already in the store are reused (unless `reclassify`); only misses are sent to
    the model, in batches of about 40, concurrently. sqlite3 connections are not shared
    across threads, so the worker threads only call the API; each batch's rows are upserted
    on this (the calling) thread as that batch's future completes, in its own transaction, so
    an interrupted run keeps every batch that finished.
    """
    conn = _connect(store_path if store_path is not None else default_store_path())
    try:
        keys = tuple(cache_key(item.entry) for item in items)
        unique_texts: dict[str, str] = {}
        release_by_key: dict[str, Version] = {}
        for key, item in zip(keys, items, strict=True):
            unique_texts.setdefault(key, item.entry.full_text())
            release_by_key.setdefault(key, item.release)

        store = {} if reclassify else _load_classifications(conn, tuple(unique_texts))
        pending_keys = list(unique_texts) if reclassify else [k for k in unique_texts if k not in store]

        if pending_keys:
            batches = [pending_keys[i : i + _BATCH_SIZE] for i in range(0, len(pending_keys), _BATCH_SIZE)]
            note(
                f"classifying {len(pending_keys)} unlabeled entries in {len(batches)} requests "
                f"with {MODEL}..."
            )

            client = client_factory()
            total_input = total_output = 0
            with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(batches))) as pool:
                futures = [pool.submit(_classify_batch, client, batch, unique_texts) for batch in batches]
                for future in as_completed(futures):
                    results, input_tokens, output_tokens = future.result()
                    total_input += input_tokens
                    total_output += output_tokens
                    classified_at = time.time()
                    rows: list[tuple[str, Classification, str]] = []
                    for key, result in results.items():
                        classification = Classification(
                            category=result.category,
                            breaking=result.breaking,
                            breaking_reason=result.breaking_reason,
                            model=MODEL,
                            effort=EFFORT,
                            release=str(release_by_key[key]),
                            classified_at=classified_at,
                        )
                        store[key] = classification
                        rows.append((key, classification, unique_texts[key]))
                    _upsert_classifications(conn, rows)

            note(f"classification used {total_input} input tokens and {total_output} output tokens")

        return tuple(store[key] for key in keys)
    finally:
        conn.close()
