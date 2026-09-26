"""`asio-docs diff|releases` - release-notes diffing between two Asio versions."""

import argparse
import json
from collections.abc import Callable

from . import classify, history, notesdiff

_RENDERERS: dict[str, Callable[[notesdiff.DiffResult, argparse.Namespace], str]] = {
    "text": lambda result, args: notesdiff.render_text(result, color=args.color),
    "markdown": lambda result, _args: notesdiff.render_markdown(result),
    "json": lambda result, _args: json.dumps(notesdiff.render_json(result), indent=2),
}


def _run_diff(args: argparse.Namespace) -> int:
    result = notesdiff.build_diff_result(args.from_version, args.to_version, refresh=args.refresh)
    print(_RENDERERS[args.format](result, args))
    return 0


def _run_releases(args: argparse.Namespace) -> int:
    releases = history.fetch_history(refresh=args.refresh)
    stored = classify.stored_keys()
    for release in releases:
        total = len(release.entries)
        classified = sum(1 for entry in release.entries if classify.cache_key(entry) in stored)
        print(f"{release.version}: {total} entries ({classified} classified)")
    return 0


def register(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    diff_parser = commands.add_parser(
        "diff",
        help="show what changed between two Asio releases",
        description=(
            f"Classification uses {classify.MODEL} at {classify.EFFORT} effort. Each entry is sent "
            "once and its result stored, so only unclassified entries cost anything. A run that "
            "sends requests first prints what it expects to spend and its hard caps, and "
            "afterwards what it used, even with --quiet."
        ),
    )
    diff_parser.add_argument("from_version", metavar="FROM", help="a version, or 'latest'")
    diff_parser.add_argument("to_version", metavar="TO", help="a version, or 'latest'")
    diff_parser.add_argument("--format", choices=tuple(_RENDERERS), default="text")
    diff_parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    diff_parser.add_argument("--refresh", action="store_true", help="re-fetch the revision history page")
    diff_parser.set_defaults(run=_run_diff)

    releases_parser = commands.add_parser("releases", help="list Asio releases and their notes entry counts")
    releases_parser.add_argument("--refresh", action="store_true", help="re-fetch the revision history page")
    releases_parser.set_defaults(run=_run_releases)
