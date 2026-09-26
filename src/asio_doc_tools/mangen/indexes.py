"""Pages that index other pages: the landing page, the reference index, the
revision history, and one page per release."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .. import history
from ..diag import AsioDocsError
from ..manpage import ManPage, Section
from . import ir, naming, parse
from .assemble import DocSection, PageBuilder, PageEnv, links_in, split_sections
from .docpaths import HISTORY, LANDING, REFERENCE_INDEX
from .render import PageRef

if TYPE_CHECKING:
    from .pages import Plan


def _text(text: str) -> ir.Text:
    return ir.Text(text)


def _code(text: str) -> ir.Span:
    return ir.Span(ir.Style.CODE, (ir.Text(text),))


def _para(*inlines: ir.Inline | str) -> ir.Para:
    return ir.Para(tuple(ir.Text(i) if isinstance(i, str) else i for i in inlines))


def _page_link(ref: PageRef, text: str) -> ir.PageLink:
    return ir.PageLink(ref.name, ref.section.value, (ir.Text(text),))


def _entries(label_and_text: Iterable[tuple[str, str]]) -> ir.DefList:
    return ir.DefList(tuple(ir.DefEntry((_code(label),), (_para(text),)) for label, text in label_and_text))


# ---------------------------------------------------------------------------
# Landing page


def _organization(topic_pages: int, reference_pages: int, releases: int) -> list[DocSection]:
    intro = (
        _para(
            "Every other page is named ",
            _code("asio."),
            ir.Span(ir.Style.ITALIC, (_text("name"),)),
            f". The {reference_pages} reference pages (classes, functions, types, and type requirements) "
            "are in section ",
            _code("3asio"),
            f"; the {topic_pages} overview, tutorial, examples, and build pages, and the release notes "
            f"of all {releases} releases, are in section ",
            _code("7"),
            ".",
        ),
    )
    topics = _entries(
        (
            (
                "asio.overview.core.strands",
                "An overview page, named after its place in the overview (asio/overview/core/strands.html).",
            ),
            (
                "asio.tutorial.timer1",
                "A tutorial step (tuttimer1.html), with the step's full source listing at the end.",
            ),
            ("asio.examples.cpp20", "The examples for one C++ standard (cpp20_examples.html)."),
            ("asio.build", "Using, building, and configuring Asio."),
            ("asio.reference", "The reference index: every reference page, by category."),
            ("asio.history", "All releases; the notes of each are on asio.<version>, e.g. asio.1.38.2."),
        )
    )
    references = _entries(
        (
            ("asio.io_context", "A class: its qualified C++ name with an asio. prefix and :: replaced by a dot."),
            ("asio.ip.tcp.socket", "Namespaces and nested types, e.g. ip::tcp::socket."),
            (
                "asio.basic_stream_socket.async_connect",
                "A member function. All overloads of a function share one page: a synopsis of every "
                "declaration, then one section per overload.",
            ),
            ("asio.basic_stream_socket.~basic_stream_socket", "A destructor keeps its tilde."),
            (
                "asio.ip.address.operator==",
                "Operators keep their symbol; quote such names in the shell: man 'asio.ip.address.operator<'.",
            ),
            ("asio.AcceptHandler", "A type requirements page keeps the name its documentation file has."),
            (
                "asio.hash<asio.ip.address>",
                "A specialization of a standard template (here std::hash) is named after its declaration.",
            ),
        )
    )
    rules = ir.DefList(tuple(ir.DefEntry((_code(what),), (_para(rule),)) for what, rule in naming.RULES_SUMMARY))
    finding = _entries(
        (
            (
                "man -k asio",
                "Searches the one-line summaries of all pages (same as apropos asio); run mandb first "
                "if pages were just installed.",
            ),
            ("man asio.ip.<TAB>", "Shell completion lists matching page names."),
            ("man 3asio asio.async_read", "Restricts the lookup to the reference section."),
        )
    )
    return [
        DocSection("How these pages are organized", 1, intro),
        DocSection("Topic pages", 2, (topics,)),
        DocSection("Reference pages", 2, (references,)),
        DocSection(
            "Special characters in reference page names",
            2,
            (
                _para(
                    "Reference names follow the C++ name as the documentation titles it. So that every "
                    "name is a single, valid file name that whatis and apropos read back intact:"
                ),
                rules,
            ),
        ),
        DocSection("Finding pages", 2, (finding,)),
    ]


def _toc(plan: "Plan", root: str) -> ir.ListBlock | None:
    """The topic pages below page `root` (by name), nested, in reading order."""
    order = plan.reading_order
    below = [unit for unit in plan.topics if unit.name.startswith(root + ".")]
    below.sort(key=lambda unit: (order.get(unit.rel, len(order)), unit.name))
    names = {unit.name for unit in below}

    def parent_of(name: str) -> str:
        head = name.rpartition(".")[0]
        while head and head != root and head not in names:
            head = head.rpartition(".")[0]
        return head if head in names else root

    children: dict[str, list[str]] = {}
    for unit in below:
        children.setdefault(parent_of(unit.name), []).append(unit.name)
    titles = {unit.name: plan.metas[unit.rel].title for unit in below}

    def build(parent: str) -> ir.ListBlock | None:
        items: list[ir.ListItem] = []
        for name in children.get(parent, ()):
            blocks: list[ir.Block] = [_para(_page_link(PageRef(name, Section.TOPIC), titles[name]))]
            nested = build(name)
            if nested is not None:
                blocks.append(nested)
            items.append(ir.ListItem("", tuple(blocks)))
        return ir.ListBlock(tuple(items)) if items else None

    return build(root)


def landing_page(env: PageEnv, plan: "Plan", doc: parse.ParsedDoc, releases: Sequence[history.Release]) -> ManPage:
    builder = PageBuilder(
        env,
        name=naming.PREFIX,
        section=Section.TOPIC,
        summary="Asio C++ library: introduction and guide to its manual pages",
        url=env.page_url(doc.rel),
    )
    intro = [block for block in doc.blocks if isinstance(block, ir.Para)]
    sections = [block for block in doc.blocks if isinstance(block, ir.DefList)]
    if not sections:
        raise AsioDocsError(f"{doc.rel}: the list of documentation sections was not found")
    builder.add(DocSection("Description", 1, tuple(intro)))
    for section in _organization(len(plan.topics), len(plan.references), len(releases)):
        builder.add(section)

    covered: set[str] = set()
    contents: list[ir.DefEntry] = []
    for entry in (e for deflist in sections for e in deflist.entries):
        target = next(links_in((ir.Para(entry.term),)), None)
        ref = env.resolve(*target) if target is not None else None
        if target is None or ref is None:
            continue  # a left-out section (Networking TS, keyword index, ...)
        blocks: list[ir.Block] = list(entry.blocks)
        toc = _toc(plan, ref.name)
        if toc is not None:
            blocks.append(toc)
        if target[0] == HISTORY and releases:
            first, last = naming.release_name(str(releases[0].version)), naming.release_name(str(releases[-1].version))
            blocks.append(_para("One page per release, from ", _code(first), " to ", _code(last), "."))
        covered.add(ref.name)
        covered.update(unit.name for unit in plan.topics if unit.name.startswith(ref.name + "."))
        contents.append(ir.DefEntry(entry.term, tuple(blocks)))
    others = [unit for unit in plan.topics if unit.name not in covered]
    if others:
        items = tuple(
            ir.ListItem("", (_para(_page_link(PageRef(u.name, Section.TOPIC), plan.metas[u.rel].title)),))
            for u in sorted(others, key=lambda u: u.name)
        )
        contents.append(ir.DefEntry((_text("Other pages"),), (ir.ListBlock(items),)))
    builder.renderer.heading("Contents", level=1)
    for entry in contents:
        builder.renderer.tagged(entry.term, entry.blocks, nav=True)
    builder.flush()
    if doc.legal:
        builder.add(DocSection("Copyright and license", 1, doc.legal))
    return builder.finish()


# ---------------------------------------------------------------------------
# Reference index


def reference_index_page(env: PageEnv, doc_dir: Path) -> ManPage:
    categories = parse.parse_reference_index((doc_dir / REFERENCE_INDEX).read_bytes(), REFERENCE_INDEX)
    builder = PageBuilder(
        env,
        name=naming.path_name("reference"),
        section=Section.TOPIC,
        summary="Asio reference index: every class, function, and type, by category",
        url=env.page_url(REFERENCE_INDEX),
    )
    builder.add(
        DocSection(
            "Description",
            1,
            (
                _para(
                    "Each entry is followed by the name of its man page (in section ",
                    _code("3asio"),
                    "), e.g. ",
                    _code("man asio.io_context"),
                    " or ",
                    _code("man 3asio asio.async_read"),
                    ". The members of a class are listed on the class's page.",
                ),
            ),
        )
    )
    for category in categories:
        builder.renderer.heading(category.title, level=1)
        for group in category.groups:
            if group.title:
                builder.renderer.heading(group.title, level=2)
            items = tuple(ir.ListItem("", (ir.Para(entry),)) for entry in group.entries)
            builder.renderer.blocks((ir.ListBlock(items),))
        builder.flush()
    builder.link(LANDING)
    return builder.finish()


# ---------------------------------------------------------------------------
# Revision history


def _entry_count(release: history.Release) -> str:
    count = len(release.entries)
    return f"{count} entr{'y' if count == 1 else 'ies'}"


def history_page(env: PageEnv, releases: Sequence[history.Release]) -> ManPage:
    assert releases  # parse_history() never returns an empty history
    builder = PageBuilder(
        env,
        name=naming.path_name("history"),
        section=Section.TOPIC,
        summary="Asio revision history: the release notes of every release",
        url=env.page_url(HISTORY),
    )
    builder.add(
        DocSection(
            "Description",
            1,
            (
                _para(
                    "The notes of each release are on their own page, named after the version: ",
                    _code(f"man {naming.release_name(str(releases[0].version))}"),
                    ". Releases are listed newest first, with the number of entries in their notes.",
                ),
            ),
        )
    )
    items = tuple(
        ir.ListItem(
            "",
            (
                _para(
                    _page_link(PageRef(naming.release_name(str(r.version)), Section.TOPIC), f"Asio {r.version}"),
                    f" ({_entry_count(r)})",
                ),
            ),
        )
        for r in releases
    )
    builder.add(DocSection("Releases", 1, (ir.ListBlock(items),)))
    builder.link(LANDING)
    return builder.finish()


def release_page(env: PageEnv, releases: Sequence[history.Release], index: int) -> ManPage:
    release = releases[index]
    name = naming.release_name(str(release.version))
    anchor = f"#{release.anchor}" if release.anchor else ""
    builder = PageBuilder(
        env,
        name=name,
        section=Section.TOPIC,
        summary=f"Asio {release.version} release notes",
        url=env.page_url(HISTORY) + anchor,
    )
    split = split_sections(parse.parse_fragment(release.body_html, HISTORY))
    if not split.pre and not split.sections:
        raise AsioDocsError(f"{HISTORY}: release {release.version} has no notes")
    builder.add(DocSection("Description", 1, split.pre))
    builder.document_sections(split)
    builder.link(HISTORY)
    for neighbor in (releases[i] for i in (index - 1, index + 1) if 0 <= i < len(releases)):
        builder.link_page(PageRef(naming.release_name(str(neighbor.version)), Section.TOPIC))
    return builder.finish()


def index_pages(env: PageEnv, plan: "Plan", doc_dir: Path, releases: Sequence[history.Release]) -> list[ManPage]:
    pages: list[ManPage] = []
    if REFERENCE_INDEX in plan.metas:
        pages.append(reference_index_page(env, doc_dir))
    if HISTORY in plan.metas:
        pages.append(history_page(env, releases))
        pages.extend(release_page(env, releases, i) for i in range(len(releases)))
    landing = parse.parse_file(doc_dir, LANDING)
    pages.append(landing_page(env, plan, landing, releases))
    return pages
