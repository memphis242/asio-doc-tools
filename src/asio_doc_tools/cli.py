"""Command-line entry point: `asio-docs <command> ...`."""

import argparse
import sys
from collections.abc import Sequence

from . import __version__, diag, man_cli, notes_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asio-docs",
        description="Man pages and release-notes diffs for the standalone Asio C++ library.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress notes on stderr")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    man_cli.register(commands)
    notes_cli.register(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diag.set_quiet(args.quiet)
    try:
        return int(args.run(args))
    except diag.AsioDocsError as e:
        diag.error(str(e))
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
