"""DocBook/BoostBook HTML to the intermediate representation (ir.py).

Only the page's main DocBook container (div.section or div.chapter) is
converted; navigation chrome, the logo header, and the copyright footer sit
outside it and are never looked at. Index terms and bare anchors are dropped.
Links are resolved to doc-root-relative paths here; what they point at (a man
page or nothing) is decided later, once every page's name is known.
"""

import posixpath
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlsplit

import lxml.html
from lxml.html import HtmlElement

from ..diag import AsioDocsError
from ..links import external_url, is_bare_external_host
from . import ir

_PARSER: Final = lxml.html.HTMLParser(encoding="utf-8")
_HEADINGS: Final = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS: Final = frozenset({"p", "pre", "ul", "ol", "dl", "table", "div", "blockquote", "hr"}) | _HEADINGS
_ADMONITIONS: Final = frozenset({"note", "tip", "warning", "important", "caution"})
_EXTERNAL_SCHEMES: Final = frozenset({"http", "https", "ftp", "mailto"})
_SEPARATOR_DASH_RE: Final = re.compile(r"^[\s\u2014\u2013-]*$")


@dataclass(frozen=True, slots=True)
class ParsedDoc:
    rel: str  # doc-root-relative path, e.g. asio/reference/basic_stream_socket.html
    title: str  # the page's own title (DocBook title attribute), whitespace-normalized
    title_level: int  # HTML heading level of the title (2 for top-level chapters)
    up: str | None  # doc-root-relative path of the parent page (rel="up")
    next: str | None  # doc-root-relative path of the next page in reading order
    blocks: tuple[ir.Block, ...]  # the content below the title
    legal: tuple[ir.Block, ...]  # copyright and legal notice from the title page, if any


def _classes(element: HtmlElement) -> frozenset[str]:
    return frozenset(element.get("class", "").split())


def _normalize(text: str) -> str:
    return " ".join(text.split())


class _Converter:
    """Converts elements of one page; `base_rel` is where relative links start from."""

    def __init__(self, base_rel: str) -> None:
        self._base_rel: Final = base_rel

    # -- links -----------------------------------------------------------------

    def resolve_href(self, href: str) -> tuple[str, str] | str:
        """(doc-relative path, fragment) for links into the tree, or an absolute URL."""
        stripped = href.strip()
        if is_bare_external_host(stripped):
            # e.g. asio/history.html citing a standards paper as "www.open-std.org/..."
            # (sometimes with a leading "../") with no scheme: an external link, not
            # an in-tree path missing its "http://".
            return external_url(stripped)
        parts = urlsplit(stripped)
        if parts.scheme:
            return stripped
        if not parts.path:
            return self._base_rel, parts.fragment
        joined = posixpath.join(posixpath.dirname(self._base_rel), unquote(parts.path))
        return posixpath.normpath(joined), parts.fragment

    # -- blocks ----------------------------------------------------------------

    def flow(self, element: HtmlElement, *, skip: HtmlElement | None = None) -> list[ir.Block]:
        """The children of `element` (except `skip`) as blocks; loose inline content becomes paragraphs."""
        blocks: list[ir.Block] = []
        pending: list[ir.Inline] = []

        def flush() -> None:
            if pending:
                block = _paragraph(tuple(pending))
                if block is not None:
                    blocks.append(block)
                pending.clear()

        if element.text:
            pending.append(ir.Text(element.text))
        for child in element:
            if child is skip:
                pass
            elif isinstance(child.tag, str):
                if _is_block(child):
                    flush()
                    blocks.extend(self.block(child))
                else:
                    pending.extend(self.inline(child, in_pre=False))
            if child.tail:
                pending.append(ir.Text(child.tail))
        flush()
        return blocks

    def block(self, element: HtmlElement) -> list[ir.Block]:
        tag = element.tag
        match tag:
            case "p" | "li" | "dd" | "td" | "th":
                return self.flow(element)
            case "pre":
                return [ir.CodeBlock(tuple(self.inline_children(element, in_pre=True)))]
            case "ul":
                return [ir.ListBlock(tuple(ir.ListItem("", tuple(self.flow(li))) for li in _children(element, "li")))]
            case "ol":
                start_attribute = element.get("start", "").strip()
                start = int(start_attribute) if start_attribute.isdigit() else 1
                items = (
                    ir.ListItem(f"{start + i}.", tuple(self.flow(li))) for i, li in enumerate(_children(element, "li"))
                )
                return [ir.ListBlock(tuple(items))]
            case "dl":
                return [self._definition_list(element)]
            case "table":
                return [self._table(element, title=())]
            case "div":
                return self._div(element)
            case "blockquote":
                return [ir.Quote(tuple(self.flow(element)))]
            case "hr" | "br":
                return []
            case _ if tag in _HEADINGS:
                return [ir.Heading(int(tag[1]), tuple(self.inline_children(element, in_pre=False)))]
            case _:
                return self.flow(element)

    def _div(self, div: HtmlElement) -> list[ir.Block]:
        classes = _classes(div)
        if classes & _ADMONITIONS:
            return [self._admonition(div, sorted(classes & _ADMONITIONS)[0])]
        if "calloutlist" in classes:
            return [self._callouts(div)]
        if "footnotes" in classes:
            return [self._footnotes(div)]
        if "table" in classes:
            title_el = next((p for p in _children(div, "p") if "title" in _classes(p)), None)
            title = tuple(self.inline_children(title_el, in_pre=False)) if title_el is not None else ()
            tables = div.findall(".//table")
            if not tables:
                raise AsioDocsError(f"{self._base_rel}: a titled table without a <table> element")
            return [self._table(tables[0], title=title)]
        if classes & {"titlepage", "spirit-nav", "copyright-footer"}:
            return []
        if "mediaobject" in classes:
            images = div.findall(".//img")
            return [ir.Figure(img.get("alt", "").strip(), self._image_path(img)) for img in images]
        return self.flow(div)

    def _definition_list(self, dl: HtmlElement) -> ir.DefList:
        entries: list[ir.DefEntry] = []
        terms: list[tuple[ir.Inline, ...]] = []
        for child in dl:
            if child.tag == "dt":
                terms.append(tuple(self.inline_children(child, in_pre=False)))
            elif child.tag == "dd":
                term = _join_terms(terms)
                entries.append(ir.DefEntry(term, tuple(self.flow(child))))
                terms = []
        if terms:
            entries.append(ir.DefEntry(_join_terms(terms), ()))
        return ir.DefList(tuple(entries))

    def _table(self, table: HtmlElement, *, title: tuple[ir.Inline, ...]) -> ir.Table:
        header: tuple[tuple[ir.Block, ...], ...] = ()
        head_rows = table.findall("thead/tr")
        if head_rows:
            header = tuple(tuple(self.flow(cell)) for cell in _cells(head_rows[0]))
        body_rows = table.findall("tbody/tr") + table.findall("tr")
        if not header and body_rows and all(cell.tag == "th" for cell in _cells(body_rows[0])):
            header = tuple(tuple(self.flow(cell)) for cell in _cells(body_rows[0]))
            body_rows = body_rows[1:]
        rows = tuple(tuple(tuple(self.flow(cell)) for cell in _cells(row)) for row in body_rows)
        return ir.Table(title, header, rows)

    def _admonition(self, div: HtmlElement, kind: str) -> ir.Admonition:
        rows = div.findall(".//tr")
        heading = div.find(".//th")
        label = _normalize(heading.text_content()) if heading is not None else ""
        if not rows:
            return ir.Admonition(label or kind.capitalize(), tuple(self.flow(div)))
        cells = _cells(rows[-1])
        body = tuple(self.flow(cells[-1])) if cells else ()
        return ir.Admonition(label or kind.capitalize(), body)

    def _callouts(self, div: HtmlElement) -> ir.ListBlock:
        items: list[ir.ListItem] = []
        for row in div.findall(".//tr"):
            cells = _cells(row)
            if len(cells) != 2:
                raise AsioDocsError(f"{self._base_rel}: unexpected callout list layout ({len(cells)} cells in a row)")
            marker = cells[0].find(".//img")
            label = marker.get("alt", "").strip() if marker is not None else _normalize(cells[0].text_content())
            items.append(ir.ListItem(f"({label})", tuple(self.flow(cells[1]))))
        return ir.ListBlock(tuple(items))

    def _footnotes(self, div: HtmlElement) -> ir.Footnotes:
        items: list[ir.ListItem] = []
        for note in div.iter("div"):
            if "footnote" not in _classes(note):
                continue
            marker = next((a for a in note.iter("a") if "para" in _classes(a)), None)
            label = _normalize(marker.text_content()) if marker is not None else "*"
            items.append(ir.ListItem(label, tuple(self.flow(note))))
        return ir.Footnotes(tuple(items))

    def _image_path(self, img: HtmlElement) -> str:
        resolved = self.resolve_href(img.get("src", ""))
        return resolved if isinstance(resolved, str) else resolved[0]

    # -- inlines ---------------------------------------------------------------

    def inline_children(self, element: HtmlElement, *, in_pre: bool) -> Iterator[ir.Inline]:
        if element.text:
            yield ir.Text(element.text)
        for child in element:
            if isinstance(child.tag, str):
                yield from self.inline(child, in_pre=in_pre)
            if child.tail:
                yield ir.Text(child.tail)

    def inline(self, element: HtmlElement, *, in_pre: bool) -> list[ir.Inline]:
        classes = _classes(element)
        match element.tag:
            case "a":
                return self._anchor(element, classes, in_pre=in_pre)
            case "code" if not in_pre:
                return [ir.Span(ir.Style.CODE, tuple(self.inline_children(element, in_pre=in_pre)))]
            case "em" | "i":
                return [ir.Span(ir.Style.ITALIC, tuple(self.inline_children(element, in_pre=in_pre)))]
            case "strong" | "b":
                return [ir.Span(ir.Style.BOLD, tuple(self.inline_children(element, in_pre=in_pre)))]
            case "br":
                return [ir.Text("\n")] if in_pre else [ir.Break()]
            case "img":
                return [ir.Image(element.get("alt", "").strip(), self._image_path(element))]
            case "span" if "silver" in classes and _SEPARATOR_DASH_RE.match(element.text_content()):
                # The dash DocBook puts between the descriptions of overloads in member tables.
                return []
            case _:
                return list(self.inline_children(element, in_pre=in_pre))

    def _anchor(self, a: HtmlElement, classes: frozenset[str], *, in_pre: bool) -> list[ir.Inline]:
        if classes & {"indexterm", "para"}:
            return []
        if "co" in classes:
            marker = a.find(".//img")
            label = marker.get("alt", "").strip() if marker is not None else _normalize(a.text_content())
            return [ir.Callout(label)]
        if "footnote" in classes:
            return [ir.Text(_normalize(a.text_content()))]
        children = tuple(self.inline_children(a, in_pre=in_pre))
        href = a.get("href")
        if href is None:
            return list(children)
        target = self.resolve_href(href)
        if isinstance(target, str):
            if urlsplit(target).scheme.lower() in _EXTERNAL_SCHEMES:
                return [ir.ExternalLink(target, children)]
            return list(children)
        path, fragment = target
        return [ir.Link(path, fragment, children)]


def _is_block(element: HtmlElement) -> bool:
    return element.tag in _BLOCK_TAGS or (element.tag == "br" and "table-break" in _classes(element))


def _children(element: HtmlElement, tag: str) -> list[HtmlElement]:
    return [child for child in element if child.tag == tag]


def _cells(row: HtmlElement) -> list[HtmlElement]:
    return [cell for cell in row if cell.tag in ("td", "th")]


def _join_terms(terms: list[tuple[ir.Inline, ...]]) -> tuple[ir.Inline, ...]:
    joined: list[ir.Inline] = []
    for i, term in enumerate(terms):
        if i:
            joined.append(ir.Text(", "))
        joined.extend(term)
    return tuple(joined)


def _paragraph(inlines: tuple[ir.Inline, ...]) -> ir.Block | None:
    """A paragraph, a figure (a paragraph holding just an image), or None if blank."""
    meaningful = [n for n in inlines if not (isinstance(n, ir.Text) and not n.text.strip())]
    if not meaningful:
        return None
    if len(meaningful) == 1 and isinstance(meaningful[0], ir.Image):
        return ir.Figure(meaningful[0].alt, meaningful[0].path)
    if not ir.plain_text(inlines) and not any(isinstance(n, (ir.Image, ir.Callout)) for n in meaningful):
        return None
    return ir.Para(inlines)


def _main_container(root: HtmlElement) -> HtmlElement | None:
    for div in root.iter("div"):
        if _classes(div) & {"section", "chapter"}:
            return div
    return None


def _head_link(root: HtmlElement, rel: str, converter: _Converter) -> str | None:
    for link in root.iter("link"):
        if link.get("rel") == rel and link.get("href"):
            target = converter.resolve_href(link.get("href", ""))
            return None if isinstance(target, str) else target[0]
    return None


def parse_html(data: bytes, rel: str) -> ParsedDoc:
    """Convert one page's HTML; `rel` is its doc-root-relative path (used for links and errors)."""
    root = lxml.html.document_fromstring(data, parser=_PARSER)
    main = _main_container(root)
    if main is None:
        raise AsioDocsError(f"{rel}: no DocBook section or chapter found; is this an Asio documentation page?")
    converter = _Converter(rel)
    titlepage = next((c for c in main if c.tag == "div" and "titlepage" in _classes(c)), None)
    heading = None
    if titlepage is not None:
        heading = next((h for h in titlepage.iter() if h.tag in _HEADINGS), None)
    if heading is None:
        raise AsioDocsError(f"{rel}: the page has no title heading")
    title_link = next((a for a in heading.iter("a") if a.get("title")), None)
    title = _normalize(title_link.get("title", "") if title_link is not None else heading.text_content())
    if not title:
        raise AsioDocsError(f"{rel}: the page title is empty")

    legal: list[ir.Block] = []
    if titlepage is not None:
        for element in titlepage.iter("p", "div"):
            classes = _classes(element)
            if "copyright" in classes or "legalnotice" in classes:
                legal.extend(converter.flow(element))

    blocks = converter.flow(main, skip=titlepage)
    return ParsedDoc(
        rel=rel,
        title=title,
        title_level=int(heading.tag[1]),
        up=_head_link(root, "up", converter),
        next=_head_link(root, "next", converter),
        blocks=tuple(blocks),
        legal=tuple(legal),
    )


def parse_file(doc_dir: Path, rel: str) -> ParsedDoc:
    path = doc_dir / rel
    try:
        data = path.read_bytes()
    except OSError as e:
        raise AsioDocsError(f"cannot read {path}: {e.strerror or e}") from e
    return parse_html(data, rel)


def parse_fragment(html: str, base_rel: str) -> tuple[ir.Block, ...]:
    """Convert a fragment of a page (e.g. one release of the revision history)."""
    wrapper = lxml.html.fragment_fromstring(html, create_parent="div")
    return tuple(_Converter(base_rel).flow(wrapper))


# ---------------------------------------------------------------------------
# Metadata pass: what naming and cross-page decisions need, without the IR.


@dataclass(frozen=True, slots=True)
class DocMeta:
    rel: str
    title: str
    up: str | None
    next: str | None
    header: str  # from the Requirements section ("" if none)
    convenience_header: str  # "" if none (or "None")
    declaration: str  # whitespace-normalized text of the first code block ("" if none)


_HEADER_RE: Final = re.compile(r"^(Convenience header|Header):\s*(\S+)$")


def _requirements(main: HtmlElement) -> tuple[str, str]:
    header = convenience = ""
    in_requirements = False
    for child in main:
        if child.tag in _HEADINGS:
            in_requirements = _normalize(child.text_content()) == "Requirements"
        elif in_requirements and child.tag == "p":
            match = _HEADER_RE.match(_normalize(child.text_content()))
            if match is None:
                continue
            if match.group(1) == "Header":
                header = match.group(2)
            elif match.group(2) != "None":
                convenience = match.group(2)
    return header, convenience


def read_meta(doc_dir: Path, rel: str) -> DocMeta:
    path = doc_dir / rel
    try:
        data = path.read_bytes()
    except OSError as e:
        raise AsioDocsError(f"cannot read {path}: {e.strerror or e}") from e
    root = lxml.html.document_fromstring(data, parser=_PARSER)
    main = _main_container(root)
    if main is None:
        raise AsioDocsError(f"{rel}: no DocBook section or chapter found; is this an Asio documentation page?")
    converter = _Converter(rel)
    heading = next((h for h in main.iter() if h.tag in _HEADINGS), None)
    if heading is None:
        raise AsioDocsError(f"{rel}: the page has no title heading")
    title_link = next((a for a in heading.iter("a") if a.get("title")), None)
    title = _normalize(title_link.get("title", "") if title_link is not None else heading.text_content())
    header, convenience = _requirements(main)
    first_code = next((c for c in main if c.tag == "pre"), None)
    return DocMeta(
        rel=rel,
        title=title,
        up=_head_link(root, "up", converter),
        next=_head_link(root, "next", converter),
        header=header,
        convenience_header=convenience,
        declaration=_normalize(first_code.text_content()) if first_code is not None else "",
    )


# ---------------------------------------------------------------------------
# The reference index (asio/reference.html): a grid of categories.


@dataclass(frozen=True, slots=True)
class IndexGroup:
    title: str  # e.g. "Classes"
    entries: tuple[tuple[ir.Inline, ...], ...]


@dataclass(frozen=True, slots=True)
class IndexCategory:
    title: str  # e.g. "Core"
    groups: tuple[IndexGroup, ...]


def parse_reference_index(data: bytes, rel: str) -> tuple[IndexCategory, ...]:
    """The categories of the reference index, in page order.

    Each table has one header row of category titles spanning one or more
    columns; each column holds (h4 title, simple list) pairs.
    """
    root = lxml.html.document_fromstring(data, parser=_PARSER)
    main = _main_container(root)
    if main is None:
        raise AsioDocsError(f"{rel}: no DocBook section found")
    converter = _Converter(rel)
    categories: list[IndexCategory] = []
    for table in main.iter("table"):
        if "table" not in _classes(table):
            continue
        spans: list[tuple[str, int]] = []
        for th in table.findall("thead/tr/th"):
            spans.append((_normalize(th.text_content()), int(th.get("colspan", "1") or "1")))
        columns: list[list[IndexGroup]] = []
        for row in table.findall("tbody/tr"):
            for cell in _cells(row):
                groups: list[IndexGroup] = []
                title = ""
                for child in cell:
                    if child.tag in _HEADINGS:
                        title = _normalize(child.text_content())
                    elif child.tag == "table" and "simplelist" in _classes(child):
                        entries = tuple(
                            tuple(converter.inline_children(entry, in_pre=False)) for entry in child.iter("td")
                        )
                        groups.append(IndexGroup(title, entries))
                columns.append(groups)
        if sum(span for _, span in spans) != len(columns):
            raise AsioDocsError(
                f"{rel}: a reference index table has {len(columns)} columns but its headers span "
                f"{sum(span for _, span in spans)}; the page layout may have changed"
            )
        column = 0
        for title, span in spans:
            spanned = tuple(group for column_groups in columns[column : column + span] for group in column_groups)
            categories.append(IndexCategory(title, spanned))
            column += span
    if not categories:
        raise AsioDocsError(f"{rel}: no reference index tables found; the page layout may have changed")
    return tuple(categories)
