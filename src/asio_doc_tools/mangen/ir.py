"""Intermediate representation between the DocBook HTML and roff.

The HTML is converted into this small document model first (parse.py), pages
are assembled and restructured on it (pages.py), and only then rendered to roff
(render.py). Everything is immutable. Text is kept raw (unescaped, whitespace as
in the source); the renderer collapses whitespace outside code blocks.
"""

from dataclasses import dataclass
from enum import Enum, auto


class Style(Enum):
    BOLD = auto()
    ITALIC = auto()
    CODE = auto()  # inline code: rendered bold, escaped for copy-paste


# ---------------------------------------------------------------------------
# Inline nodes


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class Span:
    style: Style
    children: tuple["Inline", ...]


@dataclass(frozen=True, slots=True)
class Link:
    """A link into the doc tree: `path` is relative to the doc root (normalized)."""

    path: str
    fragment: str
    children: tuple["Inline", ...]


@dataclass(frozen=True, slots=True)
class PageLink:
    """A link to a generated man page by name (for synthesized content)."""

    name: str
    section: str
    children: tuple["Inline", ...]


@dataclass(frozen=True, slots=True)
class ExternalLink:
    url: str
    children: tuple["Inline", ...]


@dataclass(frozen=True, slots=True)
class Break:
    """A forced line break (<br>)."""


@dataclass(frozen=True, slots=True)
class Callout:
    """A numbered callout marker inside a code listing."""

    label: str


@dataclass(frozen=True, slots=True)
class Image:
    """An image that is not a block of its own (rare; figures become Figure blocks)."""

    alt: str
    path: str


Inline = Text | Span | Link | PageLink | ExternalLink | Break | Callout | Image


# ---------------------------------------------------------------------------
# Block nodes


@dataclass(frozen=True, slots=True)
class Para:
    inlines: tuple[Inline, ...]


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """Preformatted text; Text nodes keep their newlines and indentation."""

    inlines: tuple[Inline, ...]


@dataclass(frozen=True, slots=True)
class Heading:
    level: int  # the HTML heading level (1-6)
    inlines: tuple[Inline, ...]


@dataclass(frozen=True, slots=True)
class ListItem:
    label: str  # "" for bullets; "1.", "(1)", "[5]" otherwise
    blocks: tuple["Block", ...]


@dataclass(frozen=True, slots=True)
class ListBlock:
    items: tuple[ListItem, ...]


@dataclass(frozen=True, slots=True)
class DefEntry:
    term: tuple[Inline, ...]
    blocks: tuple["Block", ...]


@dataclass(frozen=True, slots=True)
class DefList:
    entries: tuple[DefEntry, ...]


@dataclass(frozen=True, slots=True)
class Table:
    title: tuple[Inline, ...]  # empty when untitled
    header: tuple[tuple["Block", ...], ...]  # one entry per column; empty when headerless
    rows: tuple[tuple[tuple["Block", ...], ...], ...]


@dataclass(frozen=True, slots=True)
class Admonition:
    label: str  # e.g. "Note"
    blocks: tuple["Block", ...]


@dataclass(frozen=True, slots=True)
class Quote:
    blocks: tuple["Block", ...]


@dataclass(frozen=True, slots=True)
class Figure:
    alt: str
    path: str  # doc-root-relative image path


@dataclass(frozen=True, slots=True)
class Footnotes:
    items: tuple[ListItem, ...]


Block = Para | CodeBlock | Heading | ListBlock | DefList | Table | Admonition | Quote | Figure | Footnotes


def plain_text(inlines: tuple[Inline, ...]) -> str:
    """Whitespace-normalized text content (link text included, markers excluded)."""
    parts: list[str] = []

    def walk(nodes: tuple[Inline, ...]) -> None:
        for node in nodes:
            match node:
                case Text(text=text):
                    parts.append(text)
                case (
                    Span(children=children)
                    | Link(children=children)
                    | ExternalLink(children=children)
                    | PageLink(children=children)
                ):
                    walk(children)
                case Break():
                    parts.append(" ")
                case Callout() | Image():
                    pass

    walk(inlines)
    return " ".join("".join(parts).split())


def block_text(blocks: tuple[Block, ...]) -> str:
    """Whitespace-normalized text of paragraphs and code (for summaries and checks)."""
    parts: list[str] = []
    for block in blocks:
        match block:
            case Para(inlines=inlines) | CodeBlock(inlines=inlines) | Heading(inlines=inlines):
                parts.append(plain_text(inlines))
            case ListBlock(items=items) | Footnotes(items=items):
                parts.extend(block_text(item.blocks) for item in items)
            case DefList(entries=entries):
                for entry in entries:
                    parts.append(plain_text(entry.term))
                    parts.append(block_text(entry.blocks))
            case Admonition(blocks=inner) | Quote(blocks=inner):
                parts.append(block_text(inner))
            case Table(rows=rows):
                parts.extend(block_text(cell) for row in rows for cell in row)
            case Figure(alt=alt):
                parts.append(alt)
    return " ".join(" ".join(parts).split())
