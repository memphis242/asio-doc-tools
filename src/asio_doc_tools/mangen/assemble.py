"""Page layouts: how the parsed documents of one man page become its roff source.

Every page is NAME, then (reference pages) SYNOPSIS, then DESCRIPTION, then the
document's own sections in order, then NOTES (footnotes), then SEE ALSO, which
ends with the URL of the page's online HTML version.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Final

from ..manpage import ManPage, Section
from ..versions import Version
from . import ir, roff
from .parse import ParsedDoc
from .render import PageRef, Renderer, Resolver, is_member_table

# Kept short: groff abbreviates long page names in the header to make room for it.
MANUAL_REFERENCE: Final = "Asio Reference"
MANUAL_TOPIC: Final = "Asio C++ Library"

_SENTENCE_END_RE: Final = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)[.!?](?=\s+[A-Z(\"'`]|\s*$)")
_LINK_ONLY_RESIDUE_RE: Final = re.compile(r"^[\s,.;:()]*(?:(?:and|or|see|also)[\s,.;:()]*)*$", re.IGNORECASE)
_REQUIREMENT_LINE_RE: Final = re.compile(r"^(?:Convenience header|Header):\s*\S+$")
_INHERITED_RE: Final = re.compile(r"^Inherited from \S+\.?$")
_NAV_TEXT_MAX_WORDS: Final = 4
_DECLARATION_SUMMARY_MAX: Final = 100


@dataclass(frozen=True, slots=True)
class DocSection:
    title: str
    level: int  # 1 = .SH, 2 = .SS
    blocks: tuple[ir.Block, ...]
    flush_code: bool = False
    keep_empty: bool = False  # emit the heading even without content (it has subsections)


@dataclass(frozen=True, slots=True)
class PageEnv:
    """What page layouts need from the rest of the build."""

    version: Version
    resolve: Resolver
    image_url: Callable[[str], str]
    page_url: Callable[[str], str]  # doc-relative path (with optional #fragment) -> absolute URL


@dataclass(frozen=True, slots=True)
class SourceSection:
    title: str
    level: int  # the HTML heading level
    blocks: tuple[ir.Block, ...]


@dataclass(frozen=True, slots=True)
class Split:
    """A document cut at its headings: the content before the first one, then each section."""

    pre: tuple[ir.Block, ...]
    sections: tuple[SourceSection, ...]
    footnotes: tuple[ir.Block, ...]


# ---------------------------------------------------------------------------
# Generic helpers


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    match = _SENTENCE_END_RE.search(text)
    return text[: match.end()] if match else text


def summary_from(blocks: Iterable[ir.Block]) -> str:
    """The first sentence of the first real paragraph ("Inherited from X." notes skipped)."""
    for block in blocks:
        if isinstance(block, (ir.CodeBlock, ir.Heading)):
            break
        if isinstance(block, ir.Para):
            text = ir.plain_text(block.inlines)
            if text and not _INHERITED_RE.match(text):
                return first_sentence(text)
    return ""


def split_sections(blocks: Iterable[ir.Block]) -> Split:
    pre: list[ir.Block] = []
    sections: list[tuple[str, int, list[ir.Block]]] = []
    footnotes: list[ir.Block] = []
    current = pre
    for block in blocks:
        match block:
            case ir.Heading(level=level, inlines=inlines):
                current = []
                sections.append((ir.plain_text(inlines), level, current))
            case ir.Footnotes():
                footnotes.append(block)
            case _:
                current.append(block)
    return Split(
        pre=tuple(pre),
        sections=tuple(SourceSection(title, level, tuple(body)) for title, level, body in sections),
        footnotes=tuple(footnotes),
    )


def links_in(blocks: Iterable[ir.Block]) -> Iterator[tuple[str, str]]:
    """Every (path, fragment) linked from `blocks`."""

    def from_inlines(nodes: Iterable[ir.Inline]) -> Iterator[tuple[str, str]]:
        for node in nodes:
            match node:
                case ir.Link(path=path, fragment=fragment, children=children):
                    yield path, fragment
                    yield from from_inlines(children)
                case ir.Span(children=children) | ir.ExternalLink(children=children) | ir.PageLink(children=children):
                    yield from from_inlines(children)

    for block in blocks:
        match block:
            case ir.Para(inlines=inlines) | ir.CodeBlock(inlines=inlines) | ir.Heading(inlines=inlines):
                yield from from_inlines(inlines)
            case ir.ListBlock(items=items) | ir.Footnotes(items=items):
                for item in items:
                    yield from links_in(item.blocks)
            case ir.DefList(entries=entries):
                for entry in entries:
                    yield from from_inlines(entry.term)
                    yield from links_in(entry.blocks)
            case ir.Admonition(blocks=inner) | ir.Quote(blocks=inner):
                yield from links_in(inner)
            case ir.Table(rows=rows, title=title):
                yield from from_inlines(title)
                for row in rows:
                    for cell in row:
                        yield from links_in(cell)


def _non_link_text(blocks: Iterable[ir.Block]) -> str:
    parts: list[str] = []

    def walk(nodes: Iterable[ir.Inline]) -> None:
        for node in nodes:
            match node:
                case ir.Text(text=text):
                    parts.append(text)
                case ir.Span(children=children):
                    walk(children)
                case ir.Link() | ir.PageLink() | ir.Callout() | ir.Image() | ir.Break():
                    pass
                case ir.ExternalLink(children=children):
                    walk(children)

    for block in blocks:
        if isinstance(block, ir.Para):
            walk(block.inlines)
        else:
            parts.append(ir.block_text((block,)))
    return " ".join(" ".join(parts).split())


def is_link_only(blocks: Iterable[ir.Block]) -> bool:
    return bool(_LINK_ONLY_RESIDUE_RE.match(_non_link_text(blocks)))


def is_requirements_only(blocks: Iterable[ir.Block]) -> bool:
    """A Requirements section holding only header lines (shown in SYNOPSIS instead)."""
    blocks = tuple(blocks)
    return bool(blocks) and all(
        isinstance(b, ir.Para) and _REQUIREMENT_LINE_RE.match(ir.plain_text(b.inlines)) for b in blocks
    )


def is_nav_paragraph(block: ir.Block, targets: frozenset[str]) -> bool:
    """A short paragraph around a single link to one of `targets` ("Return to the tutorial index")."""
    if not isinstance(block, ir.Para):
        return False
    links = [node for node in block.inlines if isinstance(node, ir.Link)]
    if len(links) != 1 or links[0].path not in targets:
        return False
    return len(_non_link_text((block,)).split()) <= _NAV_TEXT_MAX_WORDS


def strip_more_markers(block: ir.CodeBlock) -> ir.CodeBlock:
    """Drop the "more..." links (after a right guillemet) a member index page puts after each declaration."""

    def keep(node: ir.Inline) -> bool:
        return not (isinstance(node, ir.Span) and ir.plain_text(node.children).startswith("\u00bb"))

    return ir.CodeBlock(tuple(node for node in block.inlines if keep(node)))


# ---------------------------------------------------------------------------
# Page builder


class PageBuilder:
    """Collects sections for one page and produces its roff source."""

    def __init__(self, env: PageEnv, *, name: str, section: Section, summary: str, url: str) -> None:
        assert summary.strip(), f"{name}: empty summary"
        self._env: Final = env
        self._name: Final = name
        self._section: Final = section
        self._summary: Final = " ".join(summary.split())
        self._url: Final = url
        self.renderer: Final = Renderer(env.resolve, frozenset({name}), env.image_url)
        self._body: list[str] = []
        self._see_also_lead: list[ir.Block] = []
        self._notes: list[ir.Block] = []

    def add(self, section: DocSection) -> None:
        if not section.blocks and not section.keep_empty:
            return
        self.renderer.heading(section.title, level=section.level)
        self.renderer.blocks(section.blocks, flush_code=section.flush_code)
        self._body.extend(self.renderer.take())

    def flush(self) -> None:
        """Move what was rendered directly through `renderer` into the page body."""
        self._body.extend(self.renderer.take())

    def link(self, path: str, fragment: str = "") -> None:
        ref = self._env.resolve(path, fragment)
        if ref is not None:
            self.renderer.add_link(ref)

    def link_page(self, ref: PageRef) -> None:
        self.renderer.add_link(ref)

    def document_sections(self, split: Split, *, force_level: int | None = None) -> None:
        """Add a document's own sections (after its DESCRIPTION).

        The shallowest heading level becomes .SH and deeper ones .SS, unless
        `force_level` puts them all on one level. A "See Also" section feeds the
        page's SEE ALSO (its text too, if it is more than a list of links); a
        Requirements section that only names headers is left to SYNOPSIS.
        """
        top = min((section.level for section in split.sections), default=0)
        for section in split.sections:
            title = section.title
            if title.lower() == "see also":
                for path, fragment in links_in(section.blocks):
                    self.link(path, fragment)
                if not is_link_only(section.blocks):
                    self._see_also_lead.extend(section.blocks)
                continue
            if title.lower() == "requirements" and is_requirements_only(section.blocks):
                continue
            level = force_level if force_level is not None else (1 if section.level <= top else 2)
            self.add(DocSection(title, level, section.blocks))
        self._notes.extend(split.footnotes)

    def finish(self) -> ManPage:
        if self._notes:
            self.add(DocSection("Notes", 1, tuple(self._notes)))
        renderer = self.renderer
        renderer.heading("See also", level=1)
        renderer.blocks(self._see_also_lead)
        see_also = renderer.take()
        refs = renderer.see_also()
        if refs:
            if self._see_also_lead:
                see_also.append(".PP")
            for i, ref in enumerate(refs):
                comma = "," if i + 1 < len(refs) else ""
                see_also.append(f"\\fB{roff.breakable_name(ref.name)}\\fR({roff.escape(ref.section.value)}){comma}")
        if refs or self._see_also_lead:
            see_also.append(".PP")
        see_also.append(f"HTML version of this page: <{roff.url(self._url)}>")

        manual = MANUAL_REFERENCE if self._section is Section.REFERENCE else MANUAL_TOPIC
        head: list[str] = []
        if renderer.uses_tbl:
            head.append("'\\\" t")
        name_arg = '"' + roff.escape_name(self._name).replace('"', r"\(dq") + '"'
        head.append(
            f'.TH {name_arg} {roff.quote_arg(self._section.value)} "" '
            f"{roff.quote_arg(f'Asio {self._env.version}')} {roff.quote_arg(manual)}"
        )
        # Ragged right and no hyphenation suit code-heavy text; AD keeps groff's
        # tagged-paragraph macros from switching back to full justification.
        head.extend(
            (".nh", ".ad l", ".ds AD l", ".SH NAME", f"{roff.escape_name(self._name)} \\- {roff.escape(self._summary)}")
        )
        source = "\n".join([*head, *self._body, *see_also]) + "\n"
        return ManPage(name=self._name, section=self._section, source=source)


# ---------------------------------------------------------------------------
# Topic pages


def topic_page(
    env: PageEnv,
    doc: ParsedDoc,
    *,
    name: str,
    source_listings: tuple[ParsedDoc, ...] = (),
    nav_targets: frozenset[str] = frozenset(),
) -> ManPage:
    """An overview/tutorial/examples/build page; tutorial steps fold in their source listing.

    Paragraphs that only link to one of `nav_targets` (tutorial navigation) are dropped.
    """
    builder = PageBuilder(env, name=name, section=Section.TOPIC, summary=doc.title, url=env.page_url(doc.rel))
    blocks = [b for b in doc.blocks if not is_nav_paragraph(b, nav_targets)]
    split = split_sections(blocks)
    builder.add(DocSection("Description", 1, split.pre))
    builder.document_sections(split)
    for listing in source_listings:
        listing_blocks = tuple(b for b in listing.blocks if not is_nav_paragraph(b, nav_targets | {doc.rel}))
        builder.add(DocSection("Source listing", 1, listing_blocks))
    if doc.up is not None:
        builder.link(doc.up)
    return builder.finish()


# ---------------------------------------------------------------------------
# Reference pages


@dataclass(frozen=True, slots=True)
class ReferenceSource:
    """One documentation page contributing to a reference man page."""

    doc: ParsedDoc
    overloads: tuple[ParsedDoc, ...]  # full docs of each overload (empty if not an overload index)


def _move_member_table_tails(split: Split) -> tuple[Split, tuple[ir.Block, ...]]:
    """Blocks after the member tables of a class page belong to its description.

    Returns the split without them, and the blocks.
    """
    moved: list[ir.Block] = []
    sections: list[SourceSection] = []
    for section in split.sections:
        blocks = section.blocks
        if blocks and isinstance(blocks[0], ir.Table) and is_member_table(blocks[0]):
            end = 1 + max(i for i, block in enumerate(blocks) if isinstance(block, ir.Table))
            moved.extend(blocks[end:])
            sections.append(replace(section, blocks=blocks[:end]))
        else:
            sections.append(section)
    return replace(split, sections=tuple(sections)), tuple(moved)


def _declaration_index(pre: tuple[ir.Block, ...]) -> int | None:
    """Index of the declaration: the first code block, if only paragraphs precede it."""
    for i, block in enumerate(pre):
        if isinstance(block, ir.CodeBlock):
            return i
        if not isinstance(block, ir.Para):
            return None
    return None


def _overload_declarations(
    pre: tuple[ir.Block, ...], overload_rels: frozenset[str]
) -> tuple[list[ir.Block], list[ir.Block]]:
    """(declaration code blocks, remaining description) of a member index page.

    Declarations are the code blocks linking to the overload pages; the paragraph
    right before each one is that group's brief, repeated on the overload itself.
    """
    declarations: list[ir.Block] = []
    description: list[ir.Block] = []
    for block in pre:
        if isinstance(block, ir.CodeBlock) and any(path in overload_rels for path, _ in links_in((block,))):
            declarations.append(strip_more_markers(block))
            if description and isinstance(description[-1], ir.Para):
                description.pop()
        else:
            description.append(block)
    return declarations, description


def _reference_summary(pre: tuple[ir.Block, ...], *, title: str, is_entity: bool) -> str:
    """The brief; else a short declaration; else the title (requirements pages: the title)."""
    if not is_entity:
        return title
    brief = summary_from(pre)
    if brief:
        return brief
    index = _declaration_index(pre)
    if index is not None:
        block = pre[index]
        assert isinstance(block, ir.CodeBlock)  # _declaration_index() only finds code blocks
        declaration = ir.plain_text(strip_more_markers(block).inlines)
        if len(declaration) <= _DECLARATION_SUMMARY_MAX:
            return declaration
    return title


def reference_page(
    env: PageEnv,
    sources: tuple[ReferenceSource, ...],
    *,
    name: str,
    title: str,
    includes: tuple[str, ...],
    is_entity: bool,
) -> ManPage:
    """A class, function, or type page (`is_entity`), or a requirements page; overloads are merged."""
    assert sources
    primary = sources[0].doc
    merged = len(sources) > 1 or bool(sources[0].overloads)
    first_split = split_sections(primary.blocks)
    summary = _reference_summary(first_split.pre, title=title, is_entity=is_entity)
    builder = PageBuilder(env, name=name, section=Section.REFERENCE, summary=summary, url=env.page_url(primary.rel))

    synopsis: list[ir.Block] = []
    if includes:
        include_lines = "\n".join(includes)
        synopsis.append(ir.CodeBlock((ir.Span(ir.Style.BOLD, (ir.Text(include_lines),)),)))

    description: list[ir.Block] = []
    own_sections: list[Split] = []  # the sections of the non-overload pages
    overloads: list[ParsedDoc] = []
    for source in sources:
        split = split_sections(source.doc.blocks)
        if source.overloads:
            rels = frozenset(o.rel for o in source.overloads)
            declarations, rest = _overload_declarations(split.pre, rels)
            synopsis.extend(declarations)
            description.extend(rest)
            overloads.extend(source.overloads)
        else:
            index = _declaration_index(split.pre)
            if index is not None:
                synopsis.append(split.pre[index])
            if merged:
                # A page documenting one member next to that member's overload set
                # (e.g. the non-static and static overloads of `query`): one more overload.
                overloads.append(source.doc)
                continue
            split, moved = _move_member_table_tails(split)
            description.extend(block for i, block in enumerate(split.pre) if i != index)
            description.extend(moved)
        own_sections.append(split)

    builder.add(DocSection("Synopsis", 1, tuple(synopsis), flush_code=True))
    builder.add(DocSection("Description", 1, tuple(description)))
    for split in own_sections:
        builder.document_sections(split)
    for number, overload in enumerate(overloads, start=1):
        overload_split = split_sections(overload.blocks)
        builder.add(DocSection(f"Overload {number} of {len(overloads)}", 1, overload_split.pre, keep_empty=True))
        builder.document_sections(overload_split, force_level=2)
    if primary.up is not None:
        builder.link(primary.up)
    return builder.finish()
