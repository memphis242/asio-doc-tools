"""Where things are in a release's doc/ tree (paths relative to it)."""

from typing import Final

LANDING: Final = "index.html"
REFERENCE_INDEX: Final = "asio/reference.html"
REFERENCE_DIR: Final = "asio/reference/"
HISTORY: Final = "asio/history.html"
TUTORIAL_INDEX: Final = "asio/tutorial.html"
USING: Final = "asio/using.html"

# Left out on purpose: the Networking TS and proposed standard executors notes,
# and the book-style keyword index (man -k covers that).
EXCLUDED_FILES: Final = frozenset({"asio/net_ts.html", "asio/std_executors.html", "asio/index.html"})
EXCLUDED_DIRS: Final = ("asio/net_ts/", "asio/std_executors/")


def is_excluded(rel: str) -> bool:
    return rel in EXCLUDED_FILES or rel.startswith(EXCLUDED_DIRS)
