"""The intermediate representation (ir.py) to roff using the man(7) macros.

Conventions (identical on every page):

- Inline code and strong text are bold, emphasis is italic. Code is escaped so
  it copies back out verbatim (ASCII minus, apostrophe, tilde, caret).
- Code blocks: .EX/.EE, indented by 4 (flush in SYNOPSIS).
- Lists: .IP with a bullet (or the item label); compact (.PD 0) when every item
  is one short paragraph. Definition lists and member tables: .TP 4 items.
- Other tables: tbl when they fit 80 columns (wrapping one wide column if need
  be), else one .TP item per row with the other columns as labeled paragraphs.
- Links into the docs render as their text; the target page is collected for
  SEE ALSO. When a link starts a list item or a table row, the target's man
  page name follows it after an arrow ("Timers -> asio.overview.timers"), since
  those lists are tables of contents.
- External links render as "text <url>".
- Paragraph text is emitted one paragraph per input line, so no line-start
  or end-of-sentence surprises arise from the source's line breaks.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

from ..manpage import Section
from . import ir, roff

TP_INDENT: Final = 4
CODE_INDENT: Final = 4
ARROW: Final = r"\(->"

# tbl is used for tables that fit an 80-column terminal, possibly by wrapping
# their one wide column (tables of 3+ columns only); anything else becomes .TP rows.
_TBL_MAX_WIDTH: Final = 70
_TBL_GAP: Final = 2
_TBL_NARROW: Final = 20  # the widest a column may be when another one wraps
_TBL_MIN_WRAP: Final = 20  # the narrowest a wrapped column may get
# A .TP tag wider than this puts its "-> page" part on the first line of the body.
_TAG_MAX_WIDTH: Final = 64

# List items up to this long (about a line) are listed without blank lines between them.
_COMPACT_MAX_TEXT: Final = 90
_WS_RE: Final = re.compile(r"[ \t\n\r\f\v]+")  # not \s: a no-break space must survive
_MEMBER_TABLE_HEADER: Final = ("Name", "Description")


@dataclass(frozen=True, slots=True, order=True)
class PageRef:
    name: str
    section: Section


Resolver = Callable[[str, str], PageRef | None]


@dataclass(frozen=True, slots=True)
class _Atom:
    text: str  # raw text, or roff when `roff` is set
    bold: bool = False
    italic: bool = False
    code: bool = False  # escape for verbatim copy-paste
    roff: bool = False  # already escaped (e.g. the arrow glyph)
    width: int = 0  # displayed width of a roff atom
    nav: bool = False  # part of a "-> page" reference


_BREAK: Final = _Atom("\n")


def _font(atom: _Atom) -> str:
    match (atom.bold, atom.italic):
        case (True, True):
            return r"\f(BI"
        case (True, False):
            return r"\fB"
        case (False, True):
            return r"\fI"
        case _:
            return r"\fR"


def is_member_table(table: ir.Table) -> bool:
    return tuple(ir.block_text(cell) for cell in table.header) == _MEMBER_TABLE_HEADER and all(
        len(row) == 2 for row in table.rows
    )


class Renderer:
    """Renders blocks of one man page; collects the pages it links to."""

    def __init__(self, resolve: Resolver, self_names: frozenset[str], image_url: Callable[[str], str]) -> None:
        self._resolve: Final = resolve
        self._self_names: Final = self_names
        self._image_url: Final = image_url
        self._links: set[PageRef] = set()
        self._nav_named: set[PageRef] = set()  # targets already shown by name inline
        self.uses_tbl = False
        self._out: list[str] = []
        self._fresh = True  # at the start of a section or item body: no paragraph break needed
        self._compact = False  # inside a .PD 0 region

    # -- results ---------------------------------------------------------------

    def see_also(self) -> tuple[PageRef, ...]:
        """Linked pages not already named inline, excluding this page, sorted by section and name."""
        refs = {ref for ref in self._links - self._nav_named if ref.name not in self._self_names}
        return tuple(sorted(refs, key=lambda ref: (ref.section.value, ref.name)))

    def add_link(self, ref: PageRef) -> None:
        self._links.add(ref)

    def take(self) -> list[str]:
        out, self._out = self._out, []
        return out

    # -- sections --------------------------------------------------------------

    def heading(self, title: str, *, level: int) -> None:
        assert level in (1, 2)
        macro = ".SH" if level == 1 else ".SS"
        self._out.append(f"{macro} {roff.quote_arg(title.upper() if level == 1 else title)}")
        self._fresh = True

    def blocks(self, blocks: Iterable[ir.Block], *, flush_code: bool = False) -> None:
        for block in blocks:
            self._block(block, flush_code=flush_code)

    # -- blocks ----------------------------------------------------------------

    def _paragraph_break(self) -> None:
        if not self._fresh:
            self._out.append(".PP")
        self._fresh = False

    def _block(self, block: ir.Block, *, flush_code: bool = False) -> None:
        match block:
            case ir.Para(inlines=inlines):
                lines = self.inline_lines(inlines)
                if lines:
                    self._paragraph_break()
                    self._out.extend(lines)
            case ir.CodeBlock(inlines=inlines):
                self._code(inlines, flush=flush_code)
            case ir.Heading(inlines=inlines):
                self._block(ir.Para((ir.Span(ir.Style.BOLD, inlines),)))
            case ir.ListBlock(items=items) | ir.Footnotes(items=items):
                self._list(items)
            case ir.DefList(entries=entries):
                for entry in entries:
                    self.tagged(entry.term, entry.blocks, nav=False)
                self._fresh = False
            case ir.Table():
                self._table(block)
            case ir.Admonition(label=label, blocks=inner):
                self._admonition(label, inner)
            case ir.Quote(blocks=inner):
                self._paragraph_break()
                self._out.append(".RS 4")
                self._fresh = True
                self.blocks(inner)
                self._out.append(".RE")
                self._fresh = False
            case ir.Figure(alt=alt, path=path):
                self._figure(alt, path)

    def _code(self, inlines: tuple[ir.Inline, ...], *, flush: bool) -> None:
        lines = self._code_lines(inlines)
        if not lines:
            return
        self._paragraph_break()
        if not flush:
            self._out.append(f".RS {CODE_INDENT}")
        self._out.append(".EX")
        self._out.extend(lines)
        self._out.append(".EE")
        if not flush:
            self._out.append(".RE")

    def _figure(self, alt: str, path: str) -> None:
        label = alt or path.rsplit("/", 1)[-1]
        self._paragraph_break()
        text = roff.escape(f"[figure: {label}]")
        self._out.append(roff.protect_line_start(f"\\fI{text}\\fR <{roff.url(self._image_url(path))}>"))

    def _admonition(self, label: str, inner: tuple[ir.Block, ...]) -> None:
        self._paragraph_break()
        self._out.append(f".RS {TP_INDENT}")
        self._fresh = True
        tag = ir.Span(ir.Style.BOLD, (ir.Text(f"{label}:"),))
        if inner and isinstance(inner[0], ir.Para):
            self._block(ir.Para((tag, ir.Text(" "), *inner[0].inlines)))
            self.blocks(inner[1:])
        else:
            self._block(ir.Para((tag,)))
            self.blocks(inner)
        self._out.append(".RE")
        self._fresh = False

    def _list(self, items: tuple[ir.ListItem, ...]) -> None:
        if not items:
            return
        compact = all(_is_compact_item(item) for item in items)
        enter_compact = compact and not self._compact
        for i, item in enumerate(items):
            if enter_compact and i == 1:
                self._out.append(".PD 0")
                self._compact = True
            if item.label:
                width = max(len(item.label) + 1, TP_INDENT)
                self._out.append(f".IP {roff.quote_arg(item.label)} {width}")
            else:
                self._out.append(r".IP \(bu 2")
            self._fresh = True
            self._item_body(item.blocks, nav=True)
        if enter_compact and len(items) > 1:
            self._out.append(".PD")
            self._compact = False
        self._fresh = False

    def tagged(self, tag: tuple[ir.Inline, ...], body: tuple[ir.Block, ...], *, nav: bool) -> None:
        """A .TP item; with `nav`, a link starting the tag is followed by its page name."""
        atoms: list[_Atom] = []
        self._flatten(tag, atoms, bold=False, italic=False, code=False, nav=[nav])
        atoms = [atom for atom in atoms if atom is not _BREAK]  # a tag is one line
        lead: list[str] = []
        if _visible_width(atoms) > _TAG_MAX_WIDTH and any(atom.nav for atom in atoms):
            lead = _emit_filled([atom for atom in atoms if atom.nav])
            atoms = [atom for atom in atoms if not atom.nav]
        self._out.append(f".TP {TP_INDENT}")
        self._out.append(" ".join(_emit_filled(atoms)) or r"\&")
        self._fresh = True
        self._item_body(body, nav=False, lead=lead)
        self._fresh = False

    def _item_body(self, blocks: tuple[ir.Block, ...], *, nav: bool, lead: list[str] | None = None) -> None:
        """The content of a list item or tagged paragraph, right after its macro."""
        if lead:
            self._out.extend(lead)
            if blocks and isinstance(blocks[0], ir.Para):
                self._out.append(".br")
        rest = blocks
        if blocks and isinstance(blocks[0], ir.Para):
            lines = self.inline_lines(blocks[0].inlines, nav=nav)
            if lines:
                self._out.extend(lines)
                self._fresh = False
            rest = blocks[1:]
        if rest:
            self._out.append(".RS")
            self.blocks(rest)
            self._out.append(".RE")
        self._fresh = False

    # -- tables ----------------------------------------------------------------

    def _table(self, table: ir.Table) -> None:
        if table.title:
            self._block(ir.Para((ir.Span(ir.Style.BOLD, table.title),)))
        if is_member_table(table):
            for name_cell, description in table.rows:
                self.tagged(_first_inlines(name_cell), description, nav=True)
            self._fresh = False
            return
        if self._tbl(table):
            return
        labels = tuple(ir.block_text(cell) for cell in table.header)
        for row in table.rows:
            if not row:
                continue
            body: list[ir.Block] = []
            for column, cell in enumerate(row[1:], start=1):
                if not cell:
                    continue
                label = labels[column] if column < len(labels) else ""
                body.extend(_labeled(label, cell))
            if len(row) == 1:
                self.blocks(row[0])
            else:
                self.tagged(_first_inlines(row[0]), tuple(body), nav=True)
        self._fresh = False

    def _tbl(self, table: ir.Table) -> bool:
        """Render as a tbl table if it fits (see _TBL_*); False to fall back to .TP rows."""
        rows = ((table.header,) if table.header else ()) + table.rows
        columns = max((len(row) for row in rows), default=0)
        if columns < 2 or any(len(row) != columns for row in rows):
            return False
        cells: list[list[str]] = []
        widths = [0] * columns
        for row in rows:
            rendered: list[str] = []
            for column, cell in enumerate(row):
                if len(cell) > 1:
                    return False
                first = cell[0] if cell else ir.Para(())
                if not isinstance(first, ir.Para):
                    return False
                inlines = first.inlines
                if any(isinstance(node, ir.Break) for node in inlines):
                    return False
                widths[column] = max(widths[column], len(ir.plain_text(inlines)))
                rendered.append(" ".join(self.inline_lines(inlines)))
            cells.append(rendered)
        gaps = _TBL_GAP * (columns - 1)
        wrapped: int | None = None
        wrap_width = 0
        if sum(widths) + gaps > _TBL_MAX_WIDTH:
            wrapped = max(range(columns), key=lambda column: widths[column])
            narrow = [width for column, width in enumerate(widths) if column != wrapped]
            wrap_width = _TBL_MAX_WIDTH - sum(narrow) - gaps
            if columns < 3 or max(narrow) > _TBL_NARROW or wrap_width < _TBL_MIN_WRAP:
                return False

        def spec(column: int, font: str) -> str:
            width = f"w({wrap_width}n)" if column == wrapped else ""
            gap = str(_TBL_GAP) if column + 1 < columns else ""
            return f"l{font}{width}{gap}"

        def cell_text(column: int, text: str) -> str:
            if column == wrapped and text:
                return f"T{{\n{roff.protect_line_start(text)}\nT}}"
            return "\\&" + text if text in ("_", "=", "") or text.startswith((".", "'")) else text

        self.uses_tbl = True
        self._paragraph_break()
        self._out.append(".TS")
        if table.header:
            self._out.append(" ".join(spec(column, "B") for column in range(columns)))
        self._out.append(" ".join(spec(column, "") for column in range(columns)) + ".")
        for i, rendered in enumerate(cells):
            self._out.append("\t".join(cell_text(column, text) for column, text in enumerate(rendered)))
            if i == 0 and table.header:
                self._out.append("_")
        self._out.append(".TE")
        self._fresh = False
        return True

    # -- inlines ---------------------------------------------------------------

    def inline_lines(self, inlines: tuple[ir.Inline, ...], *, nav: bool = False) -> list[str]:
        """Filled text as roff lines (one per forced break); [] if there is nothing to show.

        With `nav`, a link that starts the text is followed by the target's page name.
        """
        atoms: list[_Atom] = []
        self._flatten(inlines, atoms, bold=False, italic=False, code=False, nav=[nav])
        return _emit_filled(atoms)

    def _link_atoms(self, ref: PageRef | None, nav: list[bool], atoms: list[_Atom]) -> None:
        if ref is None:
            return
        self._links.add(ref)
        if nav[0] and ref.name not in self._self_names:
            atoms.append(_Atom(" ", nav=True))
            atoms.append(_Atom(ARROW, roff=True, width=1, nav=True))
            atoms.append(_Atom(" ", nav=True))
            atoms.append(_Atom(roff.breakable_name(ref.name), roff=True, width=len(ref.name), nav=True))
            self._nav_named.add(ref)
        nav[0] = False

    def _flatten(
        self, nodes: tuple[ir.Inline, ...], atoms: list[_Atom], *, bold: bool, italic: bool, code: bool, nav: list[bool]
    ) -> None:
        for node in nodes:
            match node:
                case ir.Text(text=text):
                    atoms.append(_Atom(text, bold, italic, code))
                    if text.strip():
                        nav[0] = False
                case ir.Span(style=style, children=children):
                    self._flatten(
                        children,
                        atoms,
                        bold=bold or style in (ir.Style.BOLD, ir.Style.CODE),
                        italic=italic or style is ir.Style.ITALIC,
                        code=code or style is ir.Style.CODE,
                        nav=nav,
                    )
                case ir.Link(path=path, fragment=fragment, children=children):
                    starts = nav[0]
                    self._flatten(children, atoms, bold=bold, italic=italic, code=code, nav=[False])
                    nav[0] = starts
                    self._link_atoms(self._resolve(path, fragment), nav, atoms)
                case ir.PageLink(name=name, section=section, children=children):
                    starts = nav[0]
                    self._flatten(children, atoms, bold=bold, italic=italic, code=code, nav=[False])
                    nav[0] = starts
                    self._link_atoms(PageRef(name, Section(section)), nav, atoms)
                case ir.ExternalLink(url=url, children=children):
                    text = ir.plain_text(children)
                    if text and text != url:
                        self._flatten(children, atoms, bold=bold, italic=italic, code=code, nav=[False])
                        atoms.append(_Atom(" "))
                    atoms.append(_Atom(f"<{roff.url(url)}>", roff=True))
                    nav[0] = False
                case ir.Break():
                    atoms.append(_BREAK)
                case ir.Callout(label=label):
                    atoms.append(_Atom(f"({label})", bold=True))
                case ir.Image(alt=alt, path=path):
                    label = alt or path.rsplit("/", 1)[-1]
                    atoms.append(_Atom(f"[figure: {label}]", italic=True))
                    nav[0] = False

    def _code_lines(self, inlines: tuple[ir.Inline, ...]) -> list[str]:
        atoms: list[_Atom] = []
        self._flatten(inlines, atoms, bold=False, italic=False, code=True, nav=[False])
        # Split into physical lines, keeping each atom's font.
        lines: list[list[_Atom]] = [[]]
        for atom in atoms:
            if atom is _BREAK:
                lines.append([])
                continue
            if atom.roff:
                lines[-1].append(atom)
                continue
            for i, piece in enumerate(atom.text.expandtabs(8).split("\n")):
                if i:
                    lines.append([])
                if piece:
                    lines[-1].append(_Atom(piece, atom.bold, atom.italic, code=True))
        rendered: list[str] = []
        for line in lines:
            text = _emit_code_line(line)
            if not text and (not rendered or not rendered[-1]):
                continue  # drop leading blank lines and collapse runs of them
            rendered.append(text)
        while rendered and not rendered[-1]:
            rendered.pop()
        return rendered


def _is_compact_item(item: ir.ListItem) -> bool:
    """One short paragraph (plus nested compact lists): table-of-contents style."""
    blocks = item.blocks
    if not blocks or not isinstance(blocks[0], ir.Para) or len(ir.plain_text(blocks[0].inlines)) > _COMPACT_MAX_TEXT:
        return False
    return all(isinstance(b, ir.ListBlock) and all(_is_compact_item(i) for i in b.items) for b in blocks[1:])


def _first_inlines(cell: tuple[ir.Block, ...]) -> tuple[ir.Inline, ...]:
    """The inline content of a table cell used as a tag (paragraphs joined by spaces)."""
    joined: list[ir.Inline] = []
    for block in cell:
        if isinstance(block, ir.Para):
            if joined:
                joined.append(ir.Text(" "))
            joined.extend(node for node in block.inlines if not isinstance(node, ir.Break))
    return tuple(joined)


def _labeled(label: str, cell: tuple[ir.Block, ...]) -> list[ir.Block]:
    if not label:
        return list(cell)
    tag = ir.Span(ir.Style.ITALIC, (ir.Text(f"{label}:"),))
    if cell and isinstance(cell[0], ir.Para):
        return [ir.Para((tag, ir.Text(" "), *cell[0].inlines)), *cell[1:]]
    return [ir.Para((tag,)), *cell]


def _visible_width(atoms: Iterable[_Atom]) -> int:
    return sum(atom.width if atom.roff else len(_WS_RE.sub(" ", atom.text)) for atom in atoms)


def _normalize_atoms(atoms: list[_Atom]) -> list[_Atom]:
    """Collapse whitespace across atoms; trim it at line starts/ends and around breaks."""
    result: list[_Atom] = []
    pending_space = False
    line_start = True
    for atom in atoms:
        if atom is _BREAK:
            if not line_start:
                result.append(_BREAK)
            pending_space = False
            line_start = True
            continue
        if atom.roff:
            if pending_space and not line_start:
                result.append(_Atom(" "))
            result.append(atom)
            pending_space = False
            line_start = False
            continue
        text = _WS_RE.sub(" ", atom.text)
        if not text:
            continue
        if text.startswith(" "):
            pending_space = True
            text = text[1:]
        trailing = text.endswith(" ")
        text = text.rstrip(" ")
        if not text:
            continue
        if pending_space and not line_start:
            result.append(_Atom(" "))
        result.append(_Atom(text, atom.bold, atom.italic, atom.code))
        line_start = False
        pending_space = trailing
    while result and result[-1] is _BREAK:
        result.pop()
    return result


def _emit(atoms: Iterable[_Atom]) -> str:
    """Roff text with font escapes; whitespace at a font boundary is set in roman."""
    items = list(atoms)
    out: list[str] = []
    font = r"\fR"
    for i, atom in enumerate(items):
        if atom.roff or atom.text.strip():
            wanted = _font(atom)
        else:
            following = next((a for a in items[i + 1 :] if a.roff or a.text.strip()), None)
            wanted = font if following is not None and _font(following) == font else r"\fR"
        if wanted != font:
            out.append(wanted)
            font = wanted
        out.append(atom.text if atom.roff else roff.escape_wrapping(atom.text, code=atom.code))
    if font != r"\fR":
        out.append(r"\fR")
    return "".join(out)


def _emit_filled(atoms: list[_Atom]) -> list[str]:
    lines: list[str] = []
    current: list[_Atom] = []
    for atom in [*_normalize_atoms(atoms), _BREAK]:
        if atom is _BREAK:
            text = _emit(current)
            if text:
                if lines:
                    lines.append(".br")
                lines.append(roff.protect_line_start(text))
            current = []
        else:
            current.append(atom)
    return lines


def _emit_code_line(atoms: list[_Atom]) -> str:
    # Trailing whitespace is invisible and only draws lint warnings.
    trimmed = list(atoms)
    while trimmed and not trimmed[-1].roff and not trimmed[-1].text.rstrip():
        trimmed.pop()
    if trimmed and not trimmed[-1].roff:
        last = trimmed[-1]
        trimmed[-1] = _Atom(last.text.rstrip(), last.bold, last.italic, code=True)
    return roff.protect_line_start(_emit(trimmed))
