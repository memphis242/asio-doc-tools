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
admitted by the run's budget (hard caps on requests, output and input tokens, and
time), and the run prints what it expects to spend before its first request and what it
used on every exit. The "Run budget" comment in budget.py has the arithmetic.

Two engines run the requests, with the same results, budget, messages, and exit codes:

- `asyncio` (asyncio_engine.py, the default): tasks on one event loop and the async
  client. Cancelling a request in flight closes its connection at once, so the hard
  limit and a second Ctrl-C end the run cleanly.
- `threads` (threaded_reference/): a thread pool and the sync client, kept as a
  reference to compare with. A request in flight cannot be interrupted, so its hard
  limit and second Ctrl-C save what finished and exit the process.

The shared modules: prompt.py (what is sent and how replies are read), store.py (the
store, the pending directory, and recording results), budget.py (caps, reservations,
spend lines), retry.py (which failures are retried or billed), batch.py (one batch's
requests, split, and resend, as a generator both engines drive), and run.py (what every
run does around its engine).
"""

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from ..diag import note
from ..history import Entry
from ..versions import Version
from .budget import quantity
from .prompt import EFFORT, MODEL, PROMPT_VERSION, Category, cache_key
from .store import (
    Classification,
    connect,
    default_store_path,
    import_pending,
    load_classifications,
    stored_keys,
)

__all__ = [
    "DEFAULT_ENGINE",
    "EFFORT",
    "ENGINES",
    "MODEL",
    "PROMPT_VERSION",
    "Category",
    "Classification",
    "ClassifyItem",
    "ClientFactory",
    "cache_key",
    "classify",
    "default_store_path",
    "stored_keys",
]

ENGINES: Final = ("asyncio", "threads")
DEFAULT_ENGINE: Final = "asyncio"

# Makes a client: an `anthropic.AsyncAnthropic` for the asyncio engine, an
# `anthropic.Anthropic` for the threaded one (or fakes of them, in tests).
ClientFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class ClassifyItem:
    """One unit to classify: a top-level entry and the release it came from."""

    release: Version
    entry: Entry


class _RunBatches(Protocol):
    def __call__(
        self,
        conn: Any,
        store_path: Path,
        pending: Sequence[str],
        texts: dict[str, str],
        release_by_key: Mapping[str, Version],
        store: dict[str, Classification],
        client_factory: ClientFactory | None,
    ) -> None: ...


def _engine(name: str) -> _RunBatches:
    # Imported only when there is something to send: every other command, and every
    # cached run, skips the cost of importing asyncio or concurrent.futures.
    match name:
        case "asyncio":
            from . import asyncio_engine

            return asyncio_engine.run_batches
        case "threads":
            from .threaded_reference import engine

            return engine.run_batches
    raise AssertionError(f"unknown classification engine {name!r}; engines are {ENGINES}")


def classify(
    items: Sequence[ClassifyItem],
    *,
    engine: str = DEFAULT_ENGINE,
    client_factory: ClientFactory | None = None,
    store_path: Path | None = None,
) -> tuple[Classification, ...]:
    """Classifies each item, returning one Classification per item in the same order.

    Results already in the store are reused; only entries not stored yet are sent to the
    model, in batches of up to 40 with several in flight at once, within one run budget,
    by `engine` (one of ENGINES), whose default client `client_factory` replaces. Each
    batch's results are stored as soon as it finishes, so a run that stops early keeps
    everything it paid for, and rerunning sends only what is still missing. Results an
    earlier run could not store (see store.import_pending) go into the store first.

    A run that stops before classifying everything (an error, a hard cap, a store-write
    failure, the wall-clock limit) raises AsioDocsError naming the reason and how many
    entries were stored. A first Ctrl-C waits for the requests in flight and stores their
    results; a second one gives up on them; either raises KeyboardInterrupt (status 130),
    except that the threaded reference exits the process at the second one.

    Must be called from the main thread: a run handles Ctrl-C itself, and signal handlers
    can only be set from the main thread.
    """
    assert threading.current_thread() is threading.main_thread()
    assert engine in ENGINES
    resolved_store_path = store_path if store_path is not None else default_store_path()
    conn = connect(resolved_store_path)
    try:
        import_pending(conn, resolved_store_path)
        keys = tuple(cache_key(item.entry) for item in items)
        unique_texts: dict[str, str] = {}
        release_by_key: dict[str, Version] = {}
        for key, item in zip(keys, items, strict=True):
            unique_texts.setdefault(key, item.entry.full_text())
            release_by_key.setdefault(key, item.release)

        store = load_classifications(conn, tuple(unique_texts), resolved_store_path)
        pending = tuple(k for k in unique_texts if k not in store)
        if pending:
            _engine(engine)(
                conn, resolved_store_path, pending, unique_texts, release_by_key, store, client_factory
            )
        elif unique_texts:
            note(
                f"all {quantity(len(unique_texts), 'entry', 'entries')} already classified in the store; "
                "no API requests needed"
            )
        return tuple(store[key] for key in keys)
    finally:
        conn.close()
