"""The Asio revision history (asio/history.html): parsing into per-release entries.

A release's section is everything between its "Asio X.Y.Z" heading and the next
one. Its entries are the top-level list items there (nested lists become
children); a standalone paragraph (e.g. 1.0.0's "First stable release of Asio.")
counts as an entry too.
"""

import copy
import re
from dataclasses import dataclass
from typing import Final

from bs4 import BeautifulSoup, Tag

from . import net, versions
from .diag import AsioDocsError
from .versions import Version

_RELEASE_HEADING_RE: Final = re.compile(r"^Asio (\d+\.\d+\.\d+)$")
_HEADING_TAGS: Final = ("h2", "h3", "h4", "h5")
_LIST_TAGS: Final = ("ul", "ol")


@dataclass(frozen=True, slots=True)
class Entry:
    html: str  # this entry's own markup, nested lists removed
    text: str  # whitespace-normalized plain text of `html`
    children: tuple["Entry", ...]

    def full_text(self, depth: int = 0) -> str:
        """Own text plus descendants, one per line, each level indented and dashed."""
        assert depth >= 0
        prefix = "  " * depth + "- " if depth else ""
        lines = [prefix + self.text]
        lines.extend(child.full_text(depth + 1) for child in self.children)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Release:
    version: Version
    anchor: str  # HTML anchor of the heading, e.g. asio.history.asio_1_38_2 ("" if none)
    body_html: str  # the section's markup, for rendering the release on its own
    entries: tuple[Entry, ...]


def normalize_text(tag: Tag) -> str:
    # get_text() without a separator keeps "ip::tcp" intact across the per-token
    # <span>s of code markup; whitespace in the source then collapses to single spaces.
    return " ".join(tag.get_text().split())


def _list_items(list_tag: Tag) -> list[Tag]:
    return [li for li in list_tag.find_all("li", recursive=False)]


def _entry_from_li(li: Tag) -> Entry:
    nested_lists = [t for t in li.find_all(_LIST_TAGS) if t.find_parent("li") is li]
    children = tuple(_entry_from_li(item) for nested in nested_lists for item in _list_items(nested))
    own = copy.copy(li)
    for nested in [t for t in own.find_all(_LIST_TAGS) if t.find_parent("li") is own]:
        # Drop the list and, if present, the <div class="itemizedlist"> wrapper around it.
        wrapper = nested.parent
        if isinstance(wrapper, Tag) and wrapper.name == "div" and wrapper is not own:
            wrapper.decompose()
        else:
            nested.decompose()
    return Entry(html="".join(str(c) for c in own.contents).strip(), text=normalize_text(own), children=children)


def _entries_from_block(block: Tag) -> list[Entry]:
    lists = [block] if block.name in _LIST_TAGS else block.find_all(_LIST_TAGS, recursive=False)
    if block.name == "div" and not lists:
        lists = [t for t in block.find_all(_LIST_TAGS) if t.find_parent(_LIST_TAGS) is None]
    if lists:
        return [_entry_from_li(li) for lst in lists for li in _list_items(lst)]
    text = normalize_text(block)
    return [Entry(html=str(block), text=text, children=())] if text else []


def parse_history(html: str | bytes) -> tuple[Release, ...]:
    """Releases in page order (newest first)."""
    soup = BeautifulSoup(html, "lxml")
    headings = [
        h for h in soup.find_all(_HEADING_TAGS) if _RELEASE_HEADING_RE.match(" ".join(h.get_text().split()))
    ]
    if not headings:
        raise AsioDocsError("no 'Asio X.Y.Z' release headings found in the revision history page")
    heading_tag = headings[0].name
    releases: list[Release] = []
    for heading in headings:
        match = _RELEASE_HEADING_RE.match(" ".join(heading.get_text().split()))
        assert match is not None  # filtered above
        body = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag):
                if sibling.name == heading_tag:
                    break
                body.append(sibling)
        anchors = [a.get("name", "") for a in heading.find_all("a") if "asio_" in a.get("name", "")]
        releases.append(
            Release(
                version=Version.parse(match.group(1)),
                anchor=str(anchors[0]) if anchors else "",
                body_html="\n".join(str(tag) for tag in body),
                entries=tuple(entry for tag in body for entry in _entries_from_block(tag)),
            )
        )
    seen: set[Version] = set()
    for release in releases:
        if release.version in seen:
            raise AsioDocsError(f"release {release.version} appears twice in the revision history page")
        seen.add(release.version)
    return tuple(releases)


def history_url(version: Version) -> str:
    return versions.doc_root_url(version) + "asio/history.html"


def fetch_history(version: Version | None = None, *, refresh: bool = False) -> tuple[Release, ...]:
    """The revision history as published with `version`'s docs (default: the latest release).

    Each release's history page lists every release up to and including itself.
    """
    target = version if version is not None else versions.discover_latest(refresh=refresh)
    return parse_history(net.fetch_cached(history_url(target), max_age_s=None, refresh=refresh))
