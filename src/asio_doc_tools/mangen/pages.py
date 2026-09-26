"""Which man pages a doc tree yields, what each is called, and building them all.

The build runs in three steps:

1. A metadata pass over every HTML file (titles, parent/next links, headers).
2. Planning: every file is assigned to exactly one man page (overload pages to
   their member's page, tutorial source listings to their step), names are
   derived, and any two pages claiming the same name are reported as an error.
3. Rendering: each page is parsed, assembled, and rendered to roff. The index
   pages (landing, reference index, history, releases) are built last, from
   the plan.

Steps 1 and 3 fan out over worker processes; results are sorted, so the output
does not depend on scheduling.
"""

import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urljoin

from .. import history, versions
from ..diag import AsioDocsError
from ..manpage import ManPage, Section
from ..versions import Version
from . import indexes, naming, parse
from .assemble import PageEnv, ReferenceSource, reference_page, topic_page
from .docpaths import HISTORY, LANDING, REFERENCE_DIR, REFERENCE_INDEX, TUTORIAL_INDEX, USING, is_excluded
from .render import PageRef

_TOPIC_RENAMES: Final = {USING: naming.path_name("build")}

_OVERLOAD_RE: Final = re.compile(r"^(?P<index>.+)/overload(?P<number>\d+)\.html$")
_TUTORIAL_LISTING_RE: Final = re.compile(r"^(?P<step>asio/tutorial/[^/]+)/src\.html$")
_STATIC_SUFFIX: Final = "__static"
_DECLARED_SPECIALIZATION_RE: Final = re.compile(r"(?:struct|class)\s+([A-Za-z_]\w*\s*<.*>)\s*$")

_PARALLEL_MIN_ITEMS: Final = 64
_CHUNK_SIZE: Final = 32


# ---------------------------------------------------------------------------
# Catalog: every doc path that maps to a man page


@dataclass(frozen=True, slots=True)
class Catalog:
    by_path: dict[str, PageRef]  # doc-relative path -> page (overloads and listings included)
    by_anchor: dict[str, PageRef]  # "path#fragment" -> page, for links into part of a page

    def resolve(self, path: str, fragment: str) -> PageRef | None:
        if fragment:
            ref = self.by_anchor.get(f"{path}#{fragment}")
            if ref is not None:
                return ref
        return self.by_path.get(path)


@dataclass(frozen=True, slots=True)
class Urls:
    root: str

    def page(self, rel: str) -> str:
        return urljoin(self.root, rel)

    def image(self, path: str) -> str:
        return urljoin(self.root, path)


# ---------------------------------------------------------------------------
# Plan


@dataclass(frozen=True, slots=True)
class TopicUnit:
    name: str
    rel: str
    listings: tuple[str, ...]  # tutorial source listings folded into the page
    nav_targets: tuple[str, ...]  # link-only paragraphs to these are navigation, dropped


@dataclass(frozen=True, slots=True)
class ReferenceUnit:
    name: str
    title: str  # the C++ name (or requirements title) used when a page has no brief
    rels: tuple[str, ...]  # member/entity pages merged into this man page
    overloads: tuple[tuple[str, ...], ...]  # per entry of `rels`: its overload pages, in order
    includes: tuple[str, ...]
    is_entity: bool  # named after a C++ entity (False: a requirements page named after its file)


Unit = TopicUnit | ReferenceUnit


@dataclass(frozen=True, slots=True)
class Plan:
    topics: tuple[TopicUnit, ...]
    references: tuple[ReferenceUnit, ...]
    catalog: Catalog
    metas: dict[str, parse.DocMeta]
    reading_order: dict[str, int]  # rel -> position when following "next" links from the landing page


def discover(doc_dir: Path) -> tuple[str, ...]:
    """All convertible HTML files, doc-root-relative, sorted."""
    if not (doc_dir / LANDING).is_file():
        raise AsioDocsError(
            f"{doc_dir} does not look like an Asio doc tree (no {LANDING}); "
            "point at the doc/ directory of an Asio release"
        )
    rels = sorted(p.relative_to(doc_dir).as_posix() for p in doc_dir.rglob("*.html") if p.is_file())
    return tuple(rel for rel in rels if not is_excluded(rel))


def topic_name(rel: str) -> str:
    """asio/overview/core/strands.html -> asio.overview.core.strands, with the tutorial's
    `tut` and the examples' `_examples` affixes dropped."""
    if rel in _TOPIC_RENAMES:
        return _TOPIC_RENAMES[rel]
    assert rel.endswith(".html"), rel
    parts = rel[: -len(".html")].split("/")
    if parts[0] == "asio" and len(parts) > 1:
        parts = parts[1:]
    match parts:
        case ["tutorial", step] if step.startswith("tut") and len(step) > 3:
            parts = ["tutorial", step[3:]]
        case ["examples", page] if page.endswith("_examples") and len(page) > len("_examples"):
            parts = ["examples", page[: -len("_examples")]]
    return naming.path_name(*parts)


def _reference_path_name(rel: str) -> str:
    """Fallback name from the file path: asio/reference/ip__tcp/socket.html -> asio.ip.tcp.socket."""
    stem = rel[len(REFERENCE_DIR) : -len(".html")]
    parts = [part for segment in stem.split("/") for part in segment.split("__") if part]
    return naming.path_name(*parts)


def _title_repairs(metas: dict[str, parse.DocMeta]) -> dict[str, str]:
    """Titles DocBook mangled for specializations of std templates, mapped to their declared name.

    E.g. the page titled "ip::address >" declares `struct hash< asio::ip::address >`.
    Only top-level pages whose title is not a C++ name are candidates.
    """
    repairs: dict[str, str] = {}
    for rel, meta in metas.items():
        if not rel.startswith(REFERENCE_DIR) or "/" in rel[len(REFERENCE_DIR) :]:
            continue
        if naming.reference_name(meta.title) is not None:
            continue
        match = _DECLARED_SPECIALIZATION_RE.search(meta.declaration)
        if match and naming.reference_name(match.group(1)) is not None:
            repairs[meta.title] = match.group(1)
    return repairs


def _reference_title(title: str, repairs: dict[str, str]) -> str:
    for broken, fixed in repairs.items():
        if title == broken:
            return fixed
        if title.startswith(broken + "::"):
            return fixed + title[len(broken) :]
    return title


def _include_lines(rel: str, metas: dict[str, parse.DocMeta]) -> tuple[str, ...]:
    """The #include for a reference page: its own Requirements, else its nearest parent's."""
    seen: set[str] = set()
    current: str | None = rel
    while current is not None and current.startswith(REFERENCE_DIR) and current not in seen:
        meta = metas.get(current)
        if meta is None:
            break
        if meta.header:
            line = f"#include <{meta.header}>"
            if meta.convenience_header and meta.convenience_header != meta.header:
                line += f"  // or <{meta.convenience_header}>"
            return (line,)
        seen.add(current)
        current = meta.up
    return ()


def _reading_order(metas: dict[str, parse.DocMeta]) -> dict[str, int]:
    order: dict[str, int] = {}
    current: str | None = LANDING
    while current is not None and current not in order:
        order[current] = len(order)
        meta = metas.get(current)
        current = meta.next if meta is not None else None
    return order


def _collision(name: str, first: str, second: str) -> AsioDocsError:
    return AsioDocsError(
        f"two documentation pages map to the same man page name {name!r}: {first} and {second}; "
        "the naming rules need extending for this Asio version"
    )


class _Names:
    """Claims page names; two sources claiming one name is an error."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def claim(self, name: str, source: str) -> None:
        owner = self._owners.get(name)
        if owner is not None:
            raise _collision(name, owner, source)
        self._owners[name] = source


def plan(metas: dict[str, parse.DocMeta], releases: Sequence[history.Release]) -> Plan:
    rels = sorted(metas)
    overloads: dict[str, list[tuple[int, str]]] = {}
    listings: dict[str, list[str]] = {}
    reference_rels: list[str] = []
    topic_rels: list[str] = []
    for rel in rels:
        if rel in (LANDING, REFERENCE_INDEX, HISTORY):
            continue
        if rel.startswith(REFERENCE_DIR):
            match = _OVERLOAD_RE.match(rel)
            if match:
                overloads.setdefault(match.group("index") + ".html", []).append((int(match.group("number")), rel))
            else:
                reference_rels.append(rel)
            continue
        match = _TUTORIAL_LISTING_RE.match(rel)
        if match and match.group("step") + ".html" in metas:
            listings.setdefault(match.group("step") + ".html", []).append(rel)
            continue
        topic_rels.append(rel)
    orphans = sorted(set(overloads) - set(reference_rels))
    if orphans:
        raise AsioDocsError(
            f"overload pages without their member index page: {', '.join(orphans[:5])}"
            + (" ..." if len(orphans) > 5 else "")
        )

    names = _Names()
    by_path: dict[str, PageRef] = {}
    by_anchor: dict[str, PageRef] = {}

    def assign(rel: str, ref: PageRef) -> None:
        assert rel not in by_path, rel
        by_path[rel] = ref

    for rel, name, section in (
        (LANDING, naming.PREFIX, Section.TOPIC),
        (REFERENCE_INDEX, topic_name(REFERENCE_INDEX), Section.TOPIC),
        (HISTORY, topic_name(HISTORY), Section.TOPIC),
    ):
        if rel in metas:
            names.claim(name, rel)
            assign(rel, PageRef(name, section))
    for release in releases:
        name = naming.release_name(str(release.version))
        names.claim(name, f"{HISTORY} (release {release.version})")
        if release.anchor:
            by_anchor[f"{HISTORY}#{release.anchor}"] = PageRef(name, Section.TOPIC)

    topics: list[TopicUnit] = []
    for rel in topic_rels:
        name = topic_name(rel)
        names.claim(name, rel)
        ref = PageRef(name, Section.TOPIC)
        assign(rel, ref)
        own_listings = tuple(listings.get(rel, ()))
        for listing in own_listings:
            assign(listing, ref)
        nav = (*own_listings, TUTORIAL_INDEX) if own_listings else ()
        topics.append(TopicUnit(name, rel, own_listings, nav))

    repairs = _title_repairs(metas)
    groups: dict[str, list[tuple[str, str]]] = {}
    not_entities: set[str] = set()
    for rel in reference_rels:
        title = _reference_title(metas[rel].title, repairs)
        name = naming.reference_name(title) or _reference_path_name(rel)
        groups.setdefault(name, []).append((rel, title))
        if naming.reference_name(title) is None:
            not_entities.add(name)
    references: list[ReferenceUnit] = []
    for name, members in sorted(groups.items()):
        # Static and non-static overloads of one member are documented on two
        # sibling pages (x.html, x__static.html); they are one member, so one page.
        members.sort(key=lambda member: (member[0].endswith(_STATIC_SUFFIX + ".html"), member[0]))
        stems = {member[0][: -len(".html")].removesuffix(_STATIC_SUFFIX) for member in members}
        if len(stems) != 1:
            raise _collision(name, members[0][0], members[1][0])
        names.claim(name, members[0][0])
        ref = PageRef(name, Section.REFERENCE)
        unit_overloads: list[tuple[str, ...]] = []
        includes: list[str] = []
        for rel, _ in members:
            own = tuple(overload for _, overload in sorted(overloads.get(rel, ())))
            unit_overloads.append(own)
            for page in (rel, *own):
                assign(page, ref)
                for line in _include_lines(page, metas):
                    if line not in includes:
                        includes.append(line)
        references.append(
            ReferenceUnit(
                name=name,
                title=members[0][1],
                rels=tuple(rel for rel, _ in members),
                overloads=tuple(unit_overloads),
                includes=tuple(includes),
                is_entity=name not in not_entities,
            )
        )

    return Plan(
        topics=tuple(topics),
        references=tuple(references),
        catalog=Catalog(by_path, by_anchor),
        metas=metas,
        reading_order=_reading_order(metas),
    )


# ---------------------------------------------------------------------------
# Parallel execution


@dataclass(frozen=True, slots=True)
class _WorkerState:
    doc_dir: Path
    env: PageEnv | None


_state: _WorkerState | None = None


def _init_worker(state: _WorkerState) -> None:
    global _state
    _state = state


def _read_meta(rel: str) -> parse.DocMeta:
    assert _state is not None
    return parse.read_meta(_state.doc_dir, rel)


def _build_unit(unit: Unit) -> ManPage:
    assert _state is not None and _state.env is not None
    doc_dir, env = _state.doc_dir, _state.env
    try:
        match unit:
            case TopicUnit():
                doc = parse.parse_file(doc_dir, unit.rel)
                listings = tuple(parse.parse_file(doc_dir, rel) for rel in unit.listings)
                return topic_page(
                    env, doc, name=unit.name, source_listings=listings, nav_targets=frozenset(unit.nav_targets)
                )
            case ReferenceUnit():
                sources = tuple(
                    ReferenceSource(
                        parse.parse_file(doc_dir, rel), tuple(parse.parse_file(doc_dir, o) for o in overloads)
                    )
                    for rel, overloads in zip(unit.rels, unit.overloads, strict=True)
                )
                return reference_page(
                    env, sources, name=unit.name, title=unit.title, includes=unit.includes, is_entity=unit.is_entity
                )
    except Exception as e:
        source = unit.rel if isinstance(unit, TopicUnit) else ", ".join(unit.rels)
        e.add_note(f"while generating man page {unit.name} from {source}")
        raise


def _run[T, R](fn: Callable[[T], R], items: Sequence[T], state: _WorkerState) -> list[R]:
    """Map `fn` over `items`, in worker processes when that pays off; results keep input order."""
    workers = os.process_cpu_count() or 1
    if len(items) < _PARALLEL_MIN_ITEMS or workers < 2:
        _init_worker(state)
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(state,)) as pool:
        return list(pool.map(fn, items, chunksize=_CHUNK_SIZE))


# ---------------------------------------------------------------------------
# Entry point


def build(doc_dir: Path, version: Version) -> tuple[ManPage, ...]:
    rels = discover(doc_dir)
    metas = {meta.rel: meta for meta in _run(_read_meta, rels, _WorkerState(doc_dir, None))}
    releases: tuple[history.Release, ...] = ()
    if HISTORY in metas:
        releases = history.parse_history((doc_dir / HISTORY).read_bytes())
    the_plan = plan(metas, releases)
    urls = Urls(versions.doc_root_url(version))
    env = PageEnv(version=version, resolve=the_plan.catalog.resolve, image_url=urls.image, page_url=urls.page)
    units: list[Unit] = [*the_plan.topics, *the_plan.references]
    pages = _run(_build_unit, units, _WorkerState(doc_dir, env))
    pages.extend(indexes.index_pages(env, the_plan, doc_dir, releases))
    by_name: dict[str, ManPage] = {}
    for page in pages:
        assert page.name not in by_name, f"duplicate page {page.name}"  # names were claimed in plan()
        by_name[page.name] = page
    return tuple(sorted(pages, key=lambda page: (page.section.value, page.name)))
