"""The Asio revision history (asio/history.html): parsing into per-release entries.

A release's section is everything between its "Asio X.Y.Z" heading and the next
one. Its entries are the top-level list items there (nested lists become
children); a standalone paragraph (e.g. 1.0.0's "First stable release of Asio.")
counts as an entry too.

The page is parsed with lxml. Markup kept from it (`Entry.html`,
`Release.body_html`) is written back out in one canonical form, the one Beautiful
Soup's `str()` also produces: attributes sorted by name, multi-valued attributes
(class, rel, ...) whitespace-normalized, `&`, `<` and `>` escaped in text and
attribute values, void elements self-closed (`<br/>`), and text that is all ASCII
whitespace collapsed to a single newline (or space) outside <pre>/<textarea>.
"""

import hashlib
import json
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lxml import etree

from . import net, paths, versions
from .diag import AsioDocsError, warn
from .versions import Version

_RELEASE_HEADING_RE: Final = re.compile(r"^Asio (\d+\.\d+\.\d+)$")
_HEADING_TAGS: Final = ("h2", "h3", "h4", "h5")
_LIST_TAGS: Final = frozenset({"ul", "ol"})

_VOID_TAGS: Final = frozenset({
    "area", "base", "basefont", "bgsound", "br", "col", "command", "embed", "frame", "hr", "image",
    "img", "input", "isindex", "keygen", "link", "menuitem", "meta", "nextid", "param", "source",
    "spacer", "track", "wbr",
})  # fmt: skip
_RAW_TEXT_TAGS: Final = frozenset({"script", "style"})  # their text is written unescaped
_WHITESPACE_PRESERVING_TAGS: Final = frozenset({"pre", "textarea"})
# Text anywhere inside these is not part of an element's text.
_NON_TEXT_TAGS: Final = frozenset({"rp", "rt", "script", "style", "template"})
# Attributes holding a whitespace-separated list of values: on any tag, and per tag.
_MULTI_VALUED_ATTRIBUTES: Final = frozenset({"accesskey", "class", "dropzone"})
_MULTI_VALUED_ATTRIBUTES_BY_TAG: Final[dict[str, frozenset[str]]] = {
    "a": frozenset({"rel", "rev"}),
    "area": frozenset({"rel"}),
    "form": frozenset({"accept-charset"}),
    "icon": frozenset({"sizes"}),
    "iframe": frozenset({"sandbox"}),
    "link": frozenset({"rel", "rev"}),
    "object": frozenset({"archive"}),
    "output": frozenset({"for"}),
    "td": frozenset({"headers"}),
    "th": frozenset({"headers"}),
}
_ASCII_WHITESPACE: Final = " \t\n\f\r"
_NO_NAMES: Final = frozenset[str]()
_NOTHING_SKIPPED: Final = frozenset[etree._Element]()
_PARSER: Final = etree.HTMLParser(encoding="utf-8")

# Bump whenever parse_history()'s output for a given page can change (parsing rules,
# the canonical markup, the Entry/Release fields), so a cached parse made by an older
# parser is never served.
_PARSE_CACHE_VERSION: Final = "history-parse-1"


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


# ---------------------------------------------------------------------------
# HTML trees: parsing, text, and canonical markup


def _parse_html(html: str | bytes) -> etree._Element | None:
    """The document's root element; None for a document without one (e.g. empty).

    Raises etree.XMLSyntaxError for input lxml cannot recover a document from.
    """
    return etree.fromstring(html.encode() if isinstance(html, str) else html, _PARSER)


def _is_element(node: etree._Element) -> bool:
    return isinstance(node.tag, str)  # comments and processing instructions have a factory as tag


def _has_ancestor_in(node: etree._Element, tags: Collection[str]) -> bool:
    return any(ancestor.tag in tags for ancestor in node.iterancestors())


def _append_text(
    node: etree._Element, pieces: list[str], in_non_text: bool, skip: Collection[etree._Element]
) -> None:
    if not _is_element(node):
        return  # a comment or processing instruction is not text (its tail is, handled by the caller)
    excluded = in_non_text or node.tag in _NON_TEXT_TAGS
    if node.text is not None and not excluded:
        pieces.append(node.text)
    for child in node:
        if child not in skip:
            _append_text(child, pieces, excluded, skip)
        if child.tail is not None and not excluded:
            pieces.append(child.tail)


def _normalize_text(node: etree._Element, skip: Collection[etree._Element] = _NOTHING_SKIPPED) -> str:
    """`node`'s text minus the subtrees in `skip`, whitespace collapsed to single spaces.

    Text pieces are joined without a separator, which keeps "ip::tcp" intact across the
    per-token <span>s of code markup. Comments and everything inside <script>, <style>,
    <template>, <rt> and <rp> are not text.
    """
    pieces: list[str] = []
    _append_text(node, pieces, _has_ancestor_in(node, _NON_TEXT_TAGS), skip)
    return " ".join("".join(pieces).split())


def _collapse(text: str) -> str:
    """`text`, or a single newline (or space) in place of text that is all ASCII whitespace."""
    if text.strip(_ASCII_WHITESPACE):
        return text
    return "\n" if "\n" in text else " "


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _quote_attribute(value: str) -> str:
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace('"', "&quot;") + '"'


def _content(text: str, preserve_whitespace: bool, raw: bool) -> str:
    collapsed = text if preserve_whitespace else _collapse(text)
    return collapsed if raw else _escape(collapsed)


def _append_markup(
    node: etree._Element, pieces: list[str], preserve_whitespace: bool, skip: Collection[etree._Element]
) -> None:
    """Appends `node`'s canonical markup (not its tail), minus the subtrees in `skip`."""
    tag = node.tag
    if tag is etree.Comment:
        pieces.append(f"<!--{_content(node.text or '', preserve_whitespace, raw=True)}-->")
        return
    if tag is etree.ProcessingInstruction:
        pieces.append(f"<?{node.target} {node.text or ''}>")
        return
    assert isinstance(tag, str), f"unexpected node {node!r} in an HTML tree"
    attributes = ""
    if node.attrib:
        multi_valued = _MULTI_VALUED_ATTRIBUTES_BY_TAG.get(tag, _NO_NAMES)
        for name, value in sorted(node.attrib.items()):
            if name in _MULTI_VALUED_ATTRIBUTES or name in multi_valued:
                value = " ".join(value.split())
            attributes += f" {name}={_quote_attribute(_escape(value))}"
    if tag in _VOID_TAGS and node.text is None and len(node) == 0:
        pieces.append(f"<{tag}{attributes}/>")
        return
    pieces.append(f"<{tag}{attributes}>")
    preserve_whitespace = preserve_whitespace or tag in _WHITESPACE_PRESERVING_TAGS
    raw = tag in _RAW_TEXT_TAGS
    if node.text is not None:
        pieces.append(_content(node.text, preserve_whitespace, raw))
    for child in node:
        if child not in skip:
            _append_markup(child, pieces, preserve_whitespace, skip)
        if child.tail is not None:
            pieces.append(_content(child.tail, preserve_whitespace, raw))
    pieces.append(f"</{tag}>")


def _markup(node: etree._Element) -> str:
    pieces: list[str] = []
    _append_markup(node, pieces, _has_ancestor_in(node, _WHITESPACE_PRESERVING_TAGS), _NOTHING_SKIPPED)
    return "".join(pieces)


def _inner_markup(node: etree._Element, skip: Collection[etree._Element]) -> str:
    """`node`'s contents minus the subtrees in `skip`, stripped. Its own direct text,
    comments and processing instructions are kept as their plain, unescaped text."""
    preserve_whitespace = node.tag in _WHITESPACE_PRESERVING_TAGS or _has_ancestor_in(
        node, _WHITESPACE_PRESERVING_TAGS
    )
    pieces: list[str] = []
    if node.text is not None:
        pieces.append(_content(node.text, preserve_whitespace, raw=True))
    for child in node:
        if child not in skip:
            if child.tag is etree.Comment:
                pieces.append(_content(child.text or "", preserve_whitespace, raw=True))
            elif child.tag is etree.ProcessingInstruction:
                pieces.append(f"{child.target} {child.text or ''}")
            else:
                _append_markup(child, pieces, preserve_whitespace, skip)
        if child.tail is not None:
            pieces.append(_content(child.tail, preserve_whitespace, raw=True))
    return "".join(pieces).strip()


# ---------------------------------------------------------------------------
# Releases and entries


def _list_items(list_node: etree._Element) -> list[etree._Element]:
    return [child for child in list_node if child.tag == "li"]


def _entry_from_li(li: etree._Element) -> Entry:
    nested_lists = [t for t in li.iterdescendants(*_LIST_TAGS) if next(t.iterancestors("li")) is li]
    children = tuple(_entry_from_li(item) for nested in nested_lists for item in _list_items(nested))
    # The entry's own markup and text leave out each nested list and, if present, the
    # <div class="itemizedlist"> wrapper around it.
    skip: set[etree._Element] = set()
    for nested in nested_lists:
        wrapper = nested.getparent()
        assert wrapper is not None  # `nested` is a descendant of `li`
        skip.add(wrapper if wrapper.tag == "div" else nested)
    return Entry(html=_inner_markup(li, skip), text=_normalize_text(li, skip), children=children)


def _entries_from_block(block: etree._Element) -> list[Entry]:
    lists = [block] if block.tag in _LIST_TAGS else [child for child in block if child.tag in _LIST_TAGS]
    if block.tag == "div" and not lists:
        lists = [t for t in block.iterdescendants(*_LIST_TAGS) if not _has_ancestor_in(t, _LIST_TAGS)]
    if lists:
        return [_entry_from_li(li) for lst in lists for li in _list_items(lst)]
    text = _normalize_text(block)
    return [Entry(html=_markup(block), text=text, children=())] if text else []


def parse_history(html: str | bytes) -> tuple[Release, ...]:
    """Releases in page order (newest first)."""
    try:
        root = _parse_html(html)
    except etree.XMLSyntaxError as e:
        raise AsioDocsError(f"could not parse the revision history page: {e}") from e
    headings: list[tuple[etree._Element, re.Match[str]]] = []
    if root is not None:
        for heading in root.iter(*_HEADING_TAGS):
            match = _RELEASE_HEADING_RE.match(_normalize_text(heading))
            if match is not None:
                headings.append((heading, match))
    if not headings:
        raise AsioDocsError("no 'Asio X.Y.Z' release headings found in the revision history page")
    heading_tag = headings[0][0].tag
    releases: list[Release] = []
    for heading, match in headings:
        body = []
        for sibling in heading.itersiblings():
            if _is_element(sibling):
                if sibling.tag == heading_tag:
                    break
                body.append(sibling)
        anchors = [a.get("name", "") for a in heading.iterdescendants("a") if "asio_" in a.get("name", "")]
        releases.append(
            Release(
                version=Version.parse(match.group(1)),
                anchor=anchors[0] if anchors else "",
                body_html="\n".join(_markup(tag) for tag in body),
                entries=tuple(entry for tag in body for entry in _entries_from_block(tag)),
            )
        )
    seen: set[Version] = set()
    for release in releases:
        if release.version in seen:
            raise AsioDocsError(f"release {release.version} appears twice in the revision history page")
        seen.add(release.version)
    return tuple(releases)


def entry_text_with_code(entry: Entry, *, code: Callable[[str], str]) -> str:
    """The whitespace-normalized text of `entry.html`, with each outermost <code> element
    replaced by `code` of that element's own normalized text (e.g. to wrap it in markdown
    backticks)."""
    root = _parse_html(entry.html)
    pieces: list[str] = []

    def visit(node: etree._Element, in_non_text: bool) -> None:
        if not _is_element(node):
            return
        if node.tag == "code":
            pieces.append(code(_normalize_text(node)))
            return
        excluded = in_non_text or node.tag in _NON_TEXT_TAGS
        if node.text is not None and not excluded:
            pieces.append(node.text)
        for child in node:
            visit(child, excluded)
            if child.tail is not None and not excluded:
                pieces.append(child.tail)

    if root is not None:
        visit(root, False)
    return " ".join("".join(pieces).split())


# ---------------------------------------------------------------------------
# Parse cache
#
# Parsing the full revision history takes a few hundred milliseconds, far more than
# the rest of a diff whose entries are already classified, so the parsed releases of
# a fetched page are cached as JSON, one file per page URL. The file's first line is
# a sha256 over the parser version, the page, and the JSON that follows it: a changed
# page, a changed parser, or a damaged file all fail that check, and the page is
# parsed again and the file rewritten.


def _parse_cache_file(url: str) -> Path:
    return paths.cache_dir() / "history" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"


def _parse_cache_digest(page: bytes, payload: bytes) -> bytes:
    digest = hashlib.sha256(_PARSE_CACHE_VERSION.encode() + b"\n")
    digest.update(hashlib.sha256(page).digest())
    digest.update(payload)
    return digest.hexdigest().encode()


def _entry_to_json(entry: Entry) -> list[Any]:
    return [entry.html, entry.text, [_entry_to_json(child) for child in entry.children]]


def _entry_from_json(data: Any) -> Entry:
    html, text, children = data
    if not (isinstance(html, str) and isinstance(text, str) and isinstance(children, list)):
        raise ValueError("malformed entry")
    return Entry(html=html, text=text, children=tuple(_entry_from_json(child) for child in children))


def _releases_from_json(data: Any) -> tuple[Release, ...]:
    """Raises ValueError, TypeError or KeyError for data not written by `_write_parse_cache`."""
    releases = []
    for version, anchor, body_html, entries in data:
        if not (isinstance(version, str) and isinstance(anchor, str) and isinstance(body_html, str)):
            raise ValueError("malformed release")
        releases.append(
            Release(
                version=Version.parse(version),
                anchor=anchor,
                body_html=body_html,
                entries=tuple(_entry_from_json(entry) for entry in entries),
            )
        )
    if not releases:
        raise ValueError("no releases")
    return tuple(releases)


def _read_parse_cache(path: Path, page: bytes) -> tuple[Release, ...] | None:
    """The cached parse of `page`, or None when there is no valid one."""
    try:
        stored = path.read_bytes()
    except OSError:
        return None  # missing or unreadable: parse again (writing it back reports a real problem)
    digest, _, payload = stored.partition(b"\n")
    if digest != _parse_cache_digest(page, payload):
        return None
    try:
        return _releases_from_json(json.loads(payload))
    except (ValueError, TypeError, KeyError):
        return None


def _write_parse_cache(path: Path, page: bytes, releases: tuple[Release, ...]) -> None:
    data = [[str(r.version), r.anchor, r.body_html, [_entry_to_json(e) for e in r.entries]] for r in releases]
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
    try:
        net.atomic_write(path, _parse_cache_digest(page, payload) + b"\n" + payload)
    except OSError as e:
        warn(f"could not cache the parsed revision history at {path} ({e}); the next run parses it again")


def _parse_history_cached(url: str, page: bytes) -> tuple[Release, ...]:
    path = _parse_cache_file(url)
    releases = _read_parse_cache(path, page)
    if releases is None:
        releases = parse_history(page)
        _write_parse_cache(path, page, releases)
    return releases


def history_url(version: Version) -> str:
    return versions.doc_root_url(version) + "asio/history.html"


def fetch_history(version: Version | None = None, *, refresh: bool = False) -> tuple[Release, ...]:
    """The revision history as published with `version`'s docs (default: the latest release).

    Each release's history page lists every release up to and including itself.
    """
    target = version if version is not None else versions.discover_latest(refresh=refresh)
    url = history_url(target)
    return _parse_history_cached(url, net.fetch_cached(url, max_age_s=None, refresh=refresh))
