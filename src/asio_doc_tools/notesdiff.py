"""Diffing Asio's revision history between two releases: range selection, grouping
by classification, and rendering as text, markdown, or JSON.
"""

import os
import shutil
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from . import classify, history, versions
from .diag import AsioDocsError, warn
from .history import Entry, Release
from .versions import Version

_MAX_TEXT_WIDTH: Final = 100


class Group(StrEnum):
    BREAKING = "breaking"
    DEPRECATED = "deprecated"
    ADDED = "added"
    CHANGED = "changed"
    FIXED = "fixed"
    OTHER = "other"


# Display order: every breaking entry first (whatever its category), then the
# non-breaking categories.
_GROUP_ORDER: Final = (
    Group.BREAKING,
    Group.DEPRECATED,
    Group.ADDED,
    Group.CHANGED,
    Group.FIXED,
    Group.OTHER,
)

_CATEGORY_TO_GROUP: Final = {
    classify.Category.DEPRECATED: Group.DEPRECATED,
    classify.Category.ADDED: Group.ADDED,
    classify.Category.CHANGED: Group.CHANGED,
    classify.Category.FIXED: Group.FIXED,
    classify.Category.OTHER: Group.OTHER,
}

_GROUP_HEADING: Final = {
    Group.BREAKING: "Breaking",
    Group.DEPRECATED: "Deprecated",
    Group.ADDED: "Added",
    Group.CHANGED: "Changed",
    Group.FIXED: "Fixed",
    Group.OTHER: "Other",
}

# One palette, one meaning, reused across every rendered group.
_ANSI: Final = {
    Group.BREAKING: "\033[31m",
    Group.DEPRECATED: "\033[33m",
    Group.ADDED: "\033[32m",
    Group.CHANGED: "\033[36m",
    Group.FIXED: "\033[34m",
    Group.OTHER: "\033[2m",
}
_ANSI_RESET: Final = "\033[0m"


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One classified top-level entry, in the release it appeared in."""

    release: Version
    entry: Entry
    classification: classify.Classification

    @property
    def group(self) -> Group:
        if self.classification.breaking:
            return Group.BREAKING
        return _CATEGORY_TO_GROUP[self.classification.category]


@dataclass(frozen=True, slots=True)
class DiffResult:
    from_version: Version  # the older of the two versions given
    to_version: Version  # the newer of the two versions given
    releases: tuple[Version, ...]  # releases in (from_version, to_version], ascending
    groups: dict[Group, tuple[DiffEntry, ...]]  # every Group present, in display order

    @property
    def entry_count(self) -> int:
        return sum(len(entries) for entries in self.groups.values())


def _nearest_known_versions_message(version: Version, known: Sequence[Version]) -> str:
    sorted_known = sorted(known)
    lower = max((k for k in sorted_known if k < version), default=None)
    upper = min((k for k in sorted_known if k > version), default=None)
    neighbors = [str(v) for v in (lower, upper) if v is not None]
    hint = f"nearest known releases: {', '.join(neighbors)}" if neighbors else "no known releases at all"
    return f"'{version}' is not a known Asio release ({hint})"


def resolve_range(
    from_spec: str, to_spec: str, *, refresh: bool = False
) -> tuple[Version, Version, tuple[Release, ...]]:
    """The (older, newer, releases-in-range) for a diff, releases ascending.

    FROM/TO may be given in either order; if newer-first, a note is written to stderr and
    they are swapped. Equal versions, or a version absent from the revision history, is an
    AsioDocsError.
    """
    from_version = versions.resolve(from_spec, refresh=refresh)
    to_version = versions.resolve(to_spec, refresh=refresh)
    if from_version == to_version:
        raise AsioDocsError(f"'{from_version}' and '{to_version}' are the same version; nothing to diff")

    older, newer = (from_version, to_version) if from_version < to_version else (to_version, from_version)
    if (from_version, to_version) == (newer, older):
        warn(f"diffing {older} -> {newer} ({from_version} and {to_version} were given newest-first)")

    all_releases = history.fetch_history(refresh=refresh)
    known = tuple(r.version for r in all_releases)
    for version in (older, newer):
        if version not in known:
            raise AsioDocsError(_nearest_known_versions_message(version, known))

    selected = tuple(sorted((r for r in all_releases if older < r.version <= newer), key=lambda r: r.version))
    return older, newer, selected


def _entry_markdown_text(entry: Entry) -> str:
    """`entry`'s text as markdown: inline <code> spans become backticks, other markup is stripped."""
    return history.entry_text_with_code(entry, code=lambda text: f"`{text}`")


def build_diff_result(
    from_spec: str,
    to_spec: str,
    *,
    engine: str = classify.DEFAULT_ENGINE,
    client_factory: classify.ClientFactory | None = None,
    store_path: Path | None = None,
    refresh: bool = False,
) -> DiffResult:
    older, newer, selected = resolve_range(from_spec, to_spec, refresh=refresh)
    items = tuple(classify.ClassifyItem(release=release.version, entry=entry)
                  for release in selected for entry in release.entries)
    classifications = classify.classify(
        items, engine=engine, client_factory=client_factory, store_path=store_path
    )
    return DiffResult(
        from_version=older,
        to_version=newer,
        releases=tuple(r.version for r in selected),
        groups=group_entries(items, classifications),
    )


def group_entries(
    items: Sequence[classify.ClassifyItem], classifications: Sequence[classify.Classification]
) -> dict[Group, tuple[DiffEntry, ...]]:
    """Groups classified entries by `Group`, releases ascending, page order within a release."""
    assert len(items) == len(classifications)
    diff_entries = [
        DiffEntry(release=item.release, entry=item.entry, classification=c)
        for item, c in zip(items, classifications, strict=True)
    ]
    # A stable sort on release alone reorders across releases (ascending) while leaving
    # same-release entries in their original page order.
    ordered = sorted(diff_entries, key=lambda de: de.release)
    buckets: dict[Group, list[DiffEntry]] = {g: [] for g in _GROUP_ORDER}
    for de in ordered:
        buckets[de.group].append(de)
    return {g: tuple(buckets[g]) for g in _GROUP_ORDER}


# ---------------------------------------------------------------------------
# text rendering
# ---------------------------------------------------------------------------


def _use_color(color_mode: str) -> bool:
    assert color_mode in ("auto", "always", "never")
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _terminal_width() -> int:
    if not sys.stdout.isatty():
        return _MAX_TEXT_WIDTH
    return min(shutil.get_terminal_size(fallback=(_MAX_TEXT_WIDTH, 24)).columns, _MAX_TEXT_WIDTH)


def _wrap_bullet(text: str, *, depth: int, width: int) -> list[str]:
    indent = "  " * depth
    hanging = indent + "  "
    wrapped = textwrap.wrap(
        text, width=max(20, width - len(indent)), initial_indent=f"{indent}- ", subsequent_indent=hanging
    )
    return wrapped or [f"{indent}- "]


def _render_text_node(entry: Entry, group: Group, width: int, use_color: bool, depth: int) -> list[str]:
    lines = _wrap_bullet(entry.text, depth=depth, width=width)
    if use_color:
        lines = [f"{_ANSI[group]}{line}{_ANSI_RESET}" for line in lines]
    for child in entry.children:
        lines.extend(_render_text_node(child, group, width, use_color, depth + 1))
    return lines


def _render_text_entry(de: DiffEntry, width: int, use_color: bool) -> list[str]:
    group = de.group
    tag = f"[{de.classification.category.value}] " if group is Group.BREAKING else ""
    text = f"{tag}{de.entry.text} [{de.release}]"
    lines = _wrap_bullet(text, depth=0, width=width)
    if use_color:
        lines = [f"{_ANSI[group]}{line}{_ANSI_RESET}" for line in lines]
    if de.classification.breaking and de.classification.breaking_reason:
        why_lines = _wrap_bullet(f"why: {de.classification.breaking_reason}", depth=1, width=width)
        if use_color:
            why_lines = [f"{_ANSI[Group.BREAKING]}{line}{_ANSI_RESET}" for line in why_lines]
        lines.extend(why_lines)
    for child in de.entry.children:
        lines.extend(_render_text_node(child, group, width, use_color, 1))
    return lines


def render_text(result: DiffResult, *, color: str = "auto") -> str:
    use_color = _use_color(color)
    width = _terminal_width()
    lines = [f"Asio {result.from_version} -> {result.to_version}: {len(result.releases)} releases, "
             f"{result.entry_count} entries"]
    counts = ", ".join(
        f"{_GROUP_HEADING[g]} {len(entries)}" for g, entries in result.groups.items() if entries
    )
    if counts:
        lines.append(counts)
    for group, entries in result.groups.items():
        if not entries:
            continue
        lines.append("")
        heading = f"{_GROUP_HEADING[group].upper()} ({len(entries)})"
        lines.append(f"{_ANSI[group]}{heading}{_ANSI_RESET}" if use_color else heading)
        for de in entries:
            lines.extend(_render_text_entry(de, width, use_color))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown_node(entry: Entry, depth: int) -> list[str]:
    indent = "  " * depth
    lines = [f"{indent}- {_entry_markdown_text(entry)}"]
    for child in entry.children:
        lines.extend(_render_markdown_node(child, depth + 1))
    return lines


def _render_markdown_entry(de: DiffEntry) -> list[str]:
    tag = f"`{de.classification.category.value}` " if de.group is Group.BREAKING else ""
    lines = [f"- {tag}{_entry_markdown_text(de.entry)} *{de.release}*"]
    if de.classification.breaking and de.classification.breaking_reason:
        lines.append(f"  - why: {de.classification.breaking_reason}")
    for child in de.entry.children:
        lines.extend(_render_markdown_node(child, 1))
    return lines


def render_markdown(result: DiffResult) -> str:
    lines = [
        f"# Asio {result.from_version} -> {result.to_version}",
        "",
        f"{len(result.releases)} releases, {result.entry_count} entries.",
    ]
    for group, entries in result.groups.items():
        if not entries:
            continue
        lines.append("")
        lines.append(f"## {_GROUP_HEADING[group]} ({len(entries)})")
        lines.append("")
        for de in entries:
            lines.extend(_render_markdown_entry(de))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# json rendering
# ---------------------------------------------------------------------------


def _json_node(entry: Entry) -> dict[str, Any]:
    return {"text": entry.text, "children": [_json_node(c) for c in entry.children]}


def _json_entry(de: DiffEntry) -> dict[str, Any]:
    return {
        "release": str(de.release),
        "category": de.classification.category.value,
        "breaking": de.classification.breaking,
        "breaking_reason": de.classification.breaking_reason,
        "text": de.entry.text,
        "children": [_json_node(c) for c in de.entry.children],
    }


def render_json(result: DiffResult) -> dict[str, Any]:
    entries = [_json_entry(de) for entries in result.groups.values() for de in entries]
    return {
        "from": str(result.from_version),
        "to": str(result.to_version),
        "releases": [str(v) for v in result.releases],
        "entries": entries,
    }
