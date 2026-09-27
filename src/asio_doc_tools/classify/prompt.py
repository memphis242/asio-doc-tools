"""What is sent to the model and how its replies are read.

The prompt, the reply schema, and the identity of a classification (PROMPT_VERSION,
MODEL, EFFORT) are fixed: stored results are keyed by them (see `cache_key`), so a
change to any of them makes every stored result a miss and re-classifies it.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Final

from ..diag import AsioDocsError
from ..history import Entry

# Bump whenever the system prompt's meaning changes, so previously cached
# classifications (keyed on this value) are re-asked rather than reused stale.
PROMPT_VERSION: Final = "classify-2"

# Classification uses exactly this one model, at this one effort level, always - no
# per-call model choice and no server-side fallback to a different model.
MODEL: Final = "claude-sonnet-5"
EFFORT: Final = "low"


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


@dataclass(frozen=True, slots=True)
class RawResult:
    """One entry's classification as the model replied it."""

    category: Category
    breaking: bool
    breaking_reason: str


def cache_key(entry: Entry) -> str:
    """The store key of `entry`'s classification under this prompt, model, and effort."""
    return key_for_text(entry.full_text())


def key_for_text(text: str) -> str:
    normalized = text.strip()
    return sha256(f"{PROMPT_VERSION}\n{MODEL}\n{EFFORT}\n{normalized}".encode()).hexdigest()


def row_key(prompt_version: str, model: str, effort: str, text: str) -> str:
    """The store key of a row classified under `prompt_version`, `model`, and `effort`."""
    return sha256(f"{prompt_version}\n{model}\n{effort}\n{text.strip()}".encode()).hexdigest()


assert row_key(PROMPT_VERSION, MODEL, EFFORT, " Fixed a bug.\n") == key_for_text("Fixed a bug.")
# The key of one fixed text as every run has stored it: a change here would orphan every
# stored (paid-for) classification.
assert key_for_text("Fixed a bug.") == "e1e6bf83860ec8706dfb88c18fa9645d1bdf44f1f5e456f0b26ad7330730addc"


# ---------------------------------------------------------------------------
# Requests and payloads
# ---------------------------------------------------------------------------

# The longest entry in the whole revision history (as of 1.38.2) is 4,580 characters (in
# 1.11.0), so no real entry is truncated. Truncation keeps a pathological page within
# the input bound; the store key and the stored text stay the full text.
_LONGEST_OBSERVED_ENTRY_CHARS: Final = 4_580
ENTRY_TEXT_CAP: Final = max(4_000, 2 * _LONGEST_OBSERVED_ENTRY_CHARS)
assert ENTRY_TEXT_CAP == 9_160
assert _LONGEST_OBSERVED_ENTRY_CHARS < ENTRY_TEXT_CAP <= 20_000


def payload_text(text: str) -> str:
    """`text` as sent: longer than ENTRY_TEXT_CAP characters, it is cut with a visible marker."""
    if len(text) <= ENTRY_TEXT_CAP:
        return text
    return f"{text[:ENTRY_TEXT_CAP]} [... {len(text) - ENTRY_TEXT_CAP:,} more characters truncated]"


def build_payload(ids: Sequence[str], batch_keys: Sequence[str], texts: Mapping[str, str]) -> str:
    items = [{"id": id_, "text": payload_text(texts[key])} for id_, key in zip(ids, batch_keys, strict=True)]
    return json.dumps({"items": items}, indent=2)


def batch_payload(batch_keys: Sequence[str], texts: Mapping[str, str]) -> tuple[tuple[str, ...], str]:
    """The ids and the payload of one request for `batch_keys`."""
    ids = tuple(f"e{i}" for i in range(len(batch_keys)))
    return ids, build_payload(ids, batch_keys, texts)


def request_kwargs(max_tokens: int, payload: str) -> dict[str, Any]:
    """The arguments of one `messages.create` call, the same for the sync and async clients."""
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": payload}],
    }


# ---------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------


def check_refusal(response: Any) -> None:
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


def validate_reply(response: Any, ids: Sequence[str]) -> dict[str, RawResult] | None:
    items = _extract_reply_items(response)
    if items is None:
        return None
    expected = set(ids)
    seen: dict[str, RawResult] = {}
    for raw in items:
        if not isinstance(raw, dict):
            return None
        try:
            id_ = raw["id"]
            result = RawResult(
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


def diagnose_reply(response: Any, ids: Sequence[str]) -> str:
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
