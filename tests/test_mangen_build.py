"""End-to-end: build man pages from a tiny doc tree of trimmed real Asio 1.38.2 pages."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.manpage import ManPage, Section
from asio_doc_tools.mangen import build_pages
from asio_doc_tools.versions import Version

FIXTURE = Path(__file__).parent / "fixtures" / "mangen" / "doc"
VERSION = Version(1, 38, 2)


@pytest.fixture(scope="module")
def pages() -> dict[str, ManPage]:
    return {page.name: page for page in build_pages(FIXTURE, VERSION)}


def _section(source: str, title: str) -> str:
    """The roff source of one .SH section."""
    match = re.search(rf'^\.SH "?{re.escape(title)}"?$(.*?)(?=^\.SH |\Z)', source, re.MULTILINE | re.DOTALL)
    assert match, f"no {title} section"
    return match.group(1)


def _prose(source: str) -> str:
    """The source without code blocks, with the escapes of names and code undone."""
    without_code = re.sub(r"^\.EX$.*?^\.EE$", "", source, flags=re.MULTILINE | re.DOTALL)
    for escape, text in (
        (r"\(ti", "~"),
        (r"\-", "-"),
        (r"\&", ""),
        (r"\:", ""),
        (r"\fB", ""),
        (r"\fR", ""),
        (r"\fI", ""),
    ):
        without_code = without_code.replace(escape, text)
    return without_code


def test_page_set(pages: dict[str, ManPage]) -> None:
    assert set(pages) == {
        "asio",
        "asio.overview",
        "asio.overview.core",
        "asio.overview.core.strands",
        "asio.tutorial",
        "asio.tutorial.timer1",
        "asio.reference",
        "asio.history",
        "asio.1.38.2",
        "asio.1.38.1",
        "asio.basic_stream_socket",
        "asio.basic_stream_socket.assign",
        "asio.basic_stream_socket.~basic_stream_socket",
        "asio.AcceptHandler",
    }
    assert pages["asio.basic_stream_socket.assign"].section is Section.REFERENCE
    assert pages["asio.AcceptHandler"].section is Section.REFERENCE
    assert pages["asio.overview.core.strands"].section is Section.TOPIC
    assert pages["asio.1.38.2"].section is Section.TOPIC


def test_every_page_is_well_formed(pages: dict[str, ManPage]) -> None:
    for name, page in pages.items():
        source = page.source
        assert source.isascii(), name
        assert "\u2014" not in source
        header = source.splitlines()[0]
        assert header.startswith(".TH"), name
        name_section = _section(source, "NAME").strip().splitlines()
        assert len(name_section) == 1 and " \\- " in name_section[0], name
        assert "HTML version of this page" in _section(source, "SEE ALSO"), name
        if name != "asio":  # only the landing page carries the copyright (code listings aside)
            assert "Copyright" not in _prose(source) and "Boost Software License" not in _prose(source), name


def test_landing_page(pages: dict[str, ManPage]) -> None:
    source = pages["asio"].source
    for name in (
        "asio.overview",
        "asio.overview.core.strands",
        "asio.tutorial.timer1",
        "asio.reference",
        "asio.history",
    ):
        assert name in _section(source, "CONTENTS")
    assert "Boost Software License" in source
    assert "Networking TS" not in _section(source, "CONTENTS")  # a left-out section


def test_overloads_are_merged(pages: dict[str, ManPage]) -> None:
    source = pages["asio.basic_stream_socket.assign"].source
    synopsis = _section(source, "SYNOPSIS")
    assert "#include <asio/basic_stream_socket.hpp>" in synopsis  # inherited from the class page
    assert synopsis.count("void assign(") == 2
    assert "more..." not in source and "\\(Fo" not in source
    first, second = source.index('.SH "OVERLOAD 1 OF 2"'), source.index('.SH "OVERLOAD 2 OF 2"')
    assert first < second
    assert "asio::error_code & ec" in source[second:]


def test_class_page_names_member_pages(pages: dict[str, ManPage]) -> None:
    source = pages["asio.basic_stream_socket"].source
    assert "asio.basic_stream_socket.assign" in _section(source, "MEMBER FUNCTIONS")
    assert "asio.basic_stream_socket.~basic_stream_socket" in _prose(source)
    description = _section(source, "DESCRIPTION")
    assert "provides asynchronous and blocking stream-oriented socket" in description  # moved above the tables
    assert "REQUIREMENTS" not in source  # shown as #include in SYNOPSIS instead


def test_requirements_page_is_summarized_by_its_title(pages: dict[str, ManPage]) -> None:
    assert "asio.AcceptHandler \\- Accept handler requirements" in pages["asio.AcceptHandler"].source


def test_tutorial_step_folds_in_its_source_listing(pages: dict[str, ManPage]) -> None:
    source = pages["asio.tutorial.timer1"].source
    listing = _section(source, "SOURCE LISTING")
    assert "timer.cpp" in listing and "int main()" in listing
    assert "full source listing" not in source and "Return to" not in source
    assert "Next:" in source  # the link to the next step is not navigation chrome
    assert "asio.tutorial" in _section(source, "SEE ALSO")


def test_history(pages: dict[str, ManPage]) -> None:
    listing = _section(pages["asio.history"].source, "RELEASES")
    assert listing.index("asio.1.38.2") < listing.index("asio.1.38.1")
    assert "entries" in listing
    assert "asio.1.38.1" in _section(pages["asio.1.38.2"].source, "SEE ALSO")


def test_see_also_is_sorted_and_excludes_the_page_itself(pages: dict[str, ManPage]) -> None:
    source = pages["asio.overview.core.strands"].source
    refs = re.findall(r"\\fB([^\\]+)\\fR\((\w+)\)", _section(source, "SEE ALSO"))
    assert refs == sorted(refs, key=lambda ref: (ref[1], ref[0]))
    assert ("asio.overview.core", "7") in refs
    assert all(name != "asio.overview.core.strands" for name, _ in refs)


def test_build_is_deterministic(pages: dict[str, ManPage]) -> None:
    again = {page.name: page for page in build_pages(FIXTURE, VERSION)}
    assert again == pages


def test_name_collision_is_an_error(tmp_path: Path) -> None:
    doc = tmp_path / "doc"
    shutil.copytree(FIXTURE, doc)
    # A second page titled like an existing one.
    original = doc / "asio/reference/basic_stream_socket/_basic_stream_socket.html"
    (doc / "asio/reference/copy.html").write_bytes(original.read_bytes())
    with pytest.raises(AsioDocsError, match=r"_basic_stream_socket\.html.*copy\.html|copy\.html.*_basic_stream_socket"):
        build_pages(doc, VERSION)


def test_not_a_doc_tree(tmp_path: Path) -> None:
    with pytest.raises(AsioDocsError, match="does not look like an Asio doc tree"):
        build_pages(tmp_path, VERSION)


def _write_tree(pages: dict[str, ManPage], root: Path) -> list[Path]:
    paths = []
    for page in pages.values():
        path = root / page.subdir / page.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.source)
        paths.append(path)
    return paths


@pytest.mark.skipif(shutil.which("groff") is None, reason="groff not installed")
def test_groff_renders_without_warnings(pages: dict[str, ManPage], tmp_path: Path) -> None:
    for path in _write_tree(pages, tmp_path):
        result = subprocess.run(
            ["groff", "-man", "-t", "-Tutf8", "-ww", "-z", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0 and not result.stderr, f"{path.name}: {result.stderr}"


@pytest.mark.skipif(shutil.which("lexgrog") is None, reason="lexgrog (man-db) not installed")
def test_whatis_parses_every_name(pages: dict[str, ManPage], tmp_path: Path) -> None:
    for path in _write_tree(pages, tmp_path):
        result = subprocess.run(["lexgrog", str(path)], capture_output=True, text=True, check=False)
        name = path.name.rsplit(".", 1)[0]
        assert result.stdout.startswith(f'{path}: "{name} - '), result.stdout + result.stderr
