"""`asio-docs man ...`: generating, installing and inspecting Asio man pages."""

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import paths
from .diag import note
from .manpage import ManPage
from .versions import Version, resolve

if TYPE_CHECKING:
    from .docsource import Source

_SOURCE_CHOICES: "tuple[Source, ...]" = ("auto", "tarball", "online")


def _resolve_dir(path: Path) -> Path:
    """Canonicalizes a directory argument once, at the CLI boundary.

    Everything downstream (the install manifest's keys, the manpath comparison,
    the files `build` writes) works from this same expanded, symlink-resolved,
    absolute form, so e.g. installing with a relative `--man-dir` and later
    uninstalling from a different working directory still finds the same record.
    """
    return path.expanduser().resolve()


def _add_version_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "version", nargs="?", default="latest", help="Asio version, e.g. 1.38.2 (default: latest)"
    )


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=_SOURCE_CHOICES, default="auto", help="where to get the docs from")
    parser.add_argument("--refresh", action="store_true", help="ignore any cached doc tree and fetch again")


def _build_pages(args: argparse.Namespace) -> tuple[Version, tuple[ManPage, ...]]:
    # Imported lazily: the man page generator is developed alongside this module,
    # and importing it only when a subcommand actually runs keeps the CLI (and its
    # tests) working while that package is still in progress.
    from .mangen import build_pages

    version = resolve(args.version, refresh=args.refresh)
    from .docsource import ensure_doc_tree

    doc_dir = ensure_doc_tree(version, source=args.source, refresh=args.refresh)
    note(f"generating man pages for asio-{version} from {doc_dir}...")
    return version, build_pages(doc_dir, version)


def _run_install(args: argparse.Namespace) -> int:
    from .maninstall import install

    version, pages = _build_pages(args)
    report = install(pages, args.man_dir, version=version, force=args.force)

    by_section: dict[str, int] = {}
    for page in pages:
        by_section[page.subdir] = by_section.get(page.subdir, 0) + 1
    breakdown = ", ".join(f"{count} in {section}" for section, count in sorted(by_section.items()))
    print(f"installed {report.written} page(s) for asio-{report.version} into {report.man_dir} ({breakdown}).")
    if report.removed:
        print(f"removed {report.removed} stale page(s) from a previous version.")
    print("try: man asio")
    return 0


def _run_build(args: argparse.Namespace) -> int:
    from .maninstall import write_tree

    out = _resolve_dir(args.out)
    version, pages = _build_pages(args)
    written = write_tree(pages, out, compress=not args.no_compress)
    print(f"wrote {len(written)} page(s) for asio-{version} into {out}.")
    print(f"try: MANPATH={out} man asio")
    return 0


def _run_uninstall(args: argparse.Namespace) -> int:
    from .maninstall import uninstall

    man_dir = _resolve_dir(args.man_dir) if args.man_dir is not None else None
    removed = uninstall(man_dir)
    print(f"removed {removed} page(s).")
    return 0


def _run_status(args: argparse.Namespace) -> int:
    from .maninstall import status

    del args
    installs = status()
    if not installs:
        print("no asio-doc-tools man pages are installed.")
        return 0
    for result in installs:
        when = time.ctime(result.installed_at)
        breakdown = ", ".join(f"{count} in {section}" for section, count in result.count_by_section)
        print(f"asio-{result.version} installed into {result.man_dir} on {when} ({breakdown}).")
        if result.incomplete:
            print(
                f"  the last install into {result.man_dir} did not finish; "
                f"rerun `asio-docs man install --man-dir {result.man_dir}`."
            )
    return 0


def register(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    man_parser = commands.add_parser("man", help="generate and install Asio man pages")
    man_commands = man_parser.add_subparsers(dest="man_command", required=True, metavar="SUBCOMMAND")

    install_parser = man_commands.add_parser("install", help="build and install man pages for a version")
    _add_version_arg(install_parser)
    install_parser.add_argument(
        "--man-dir", type=Path, default=None, help="man tree to install into (default: XDG data dir)"
    )
    _add_source_args(install_parser)
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite files not installed by this tool (they are then tracked as this tool's own)",
    )
    install_parser.set_defaults(run=_run_install_with_default_man_dir)

    build_parser = man_commands.add_parser("build", help="build man pages into a directory, without installing")
    _add_version_arg(build_parser)
    build_parser.add_argument("--out", type=Path, required=True, help="directory to write pages into")
    build_parser.add_argument("--no-compress", action="store_true", help="write plain-text pages, not gzipped")
    _add_source_args(build_parser)
    build_parser.set_defaults(run=_run_build)

    uninstall_parser = man_commands.add_parser("uninstall", help="remove installed man pages")
    uninstall_parser.add_argument(
        "--man-dir", type=Path, default=None, help="man tree to remove from (default: the recorded install)"
    )
    uninstall_parser.set_defaults(run=_run_uninstall)

    status_parser = man_commands.add_parser("status", help="show what is currently installed")
    status_parser.set_defaults(run=_run_status)


def _run_install_with_default_man_dir(args: argparse.Namespace) -> int:
    args.man_dir = _resolve_dir(args.man_dir if args.man_dir is not None else paths.default_man_dir())
    return _run_install(args)
