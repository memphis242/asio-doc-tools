"""User-facing diagnostics: the error type the CLI reports, plus stderr notes and warnings.

Every failure the user can act on is raised as AsioDocsError with a message that
says what went wrong and what to do about it; the CLI prints it and exits
non-zero. Progress notes, API spend reports, and warnings go to stderr so stdout
stays clean for the actual output (a diff, a page listing).
"""

import sys

_PROG = "asio-docs"
_quiet = False


class AsioDocsError(Exception):
    """A failure to report to the user as-is (no traceback)."""


def set_quiet(quiet: bool) -> None:
    global _quiet
    _quiet = quiet


def note(message: str) -> None:
    """Progress information; suppressed by --quiet."""
    if not _quiet:
        print(f"{_PROG}: {message}", file=sys.stderr, flush=True)


def spend(message: str) -> None:
    """Paid API usage: what a command is about to spend and what it spent. Never
    suppressed, not even by --quiet, since it is the only record of a paid run."""
    print(f"{_PROG}: {message}", file=sys.stderr, flush=True)


def warn(message: str) -> None:
    """Something went wrong but the command can still produce a useful result."""
    print(f"{_PROG}: warning: {message}", file=sys.stderr, flush=True)


def error(message: str) -> None:
    print(f"{_PROG}: error: {message}", file=sys.stderr, flush=True)
