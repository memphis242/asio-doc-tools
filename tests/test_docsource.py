import http.server
import io
import tarfile
import threading
from pathlib import Path

import pytest

from asio_doc_tools import net
from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.docsource import ensure_doc_tree
from asio_doc_tools.versions import Version

VERSION = Version(9, 9, 9)

_REQUIRED = ("index.html", "asio/reference.html", "asio/history.html", "asio/overview.html")


def _make_tarball(path: Path, version: Version, *, members: dict[str, bytes], include_traversal: bool = False) -> None:
    with tarfile.open(path, "w:bz2") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=f"asio-{version}/doc/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if include_traversal:
            evil = b"pwned"
            info = tarfile.TarInfo(name=f"asio-{version}/doc/../../evil.txt")
            info.size = len(evil)
            tar.addfile(info, io.BytesIO(evil))


def _minimal_doc_members() -> dict[str, bytes]:
    return {name: f"<html>{name}</html>".encode() for name in _REQUIRED}


@pytest.fixture(autouse=True)
def _xdg_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def test_tarball_extraction_and_path_traversal_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = tmp_path / "asio.tar.bz2"
    _make_tarball(tarball, VERSION, members=_minimal_doc_members(), include_traversal=True)
    body = tarball.read_bytes()

    def fake_fetch(url: str, *, validate=None, **kwargs):
        assert validate is not None and validate(body) is None
        return body

    monkeypatch.setattr(net, "fetch", fake_fetch)

    # tarfile's filter="data" refuses a member that would land outside the
    # extraction directory, surfaced here as a clear user-facing error rather
    # than a partially- or wrongly-extracted tree, and no cache tree is left
    # behind looking complete.
    with pytest.raises(AsioDocsError):
        ensure_doc_tree(VERSION, source="tarball")

    cache_dir = tmp_path / "cache" / "asio-doc-tools" / "docs"
    assert not (cache_dir / f"asio-{VERSION}" / ".complete").exists()
    assert not any(cache_dir.glob("**/evil.txt"))


def test_tarball_missing_required_files_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = tmp_path / "asio.tar.bz2"
    _make_tarball(tarball, VERSION, members={"index.html": b"<html></html>"})
    body = tarball.read_bytes()
    monkeypatch.setattr(net, "fetch", lambda url, *, validate=None, **kwargs: body)

    with pytest.raises(AsioDocsError):
        ensure_doc_tree(VERSION, source="tarball")


def test_tarball_404_on_both_folders_is_a_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(url: str, *, validate=None, **kwargs):
        raise net.FetchError(f"GET {url} failed: HTTP 404 Not Found")

    monkeypatch.setattr(net, "fetch", always_fail)

    with pytest.raises(AsioDocsError):
        ensure_doc_tree(VERSION, source="tarball")


def test_marker_makes_a_second_call_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = tmp_path / "asio.tar.bz2"
    _make_tarball(tarball, VERSION, members=_minimal_doc_members())
    body = tarball.read_bytes()
    calls = []

    def fake_fetch(url: str, *, validate=None, **kwargs):
        calls.append(url)
        return body

    monkeypatch.setattr(net, "fetch", fake_fetch)

    first = ensure_doc_tree(VERSION, source="tarball")
    second = ensure_doc_tree(VERSION, source="tarball")

    assert first == second
    assert len(calls) == 1  # the second call used the completion marker, no fetch


def test_refresh_forces_a_new_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = tmp_path / "asio.tar.bz2"
    _make_tarball(tarball, VERSION, members=_minimal_doc_members())
    body = tarball.read_bytes()
    calls = []
    monkeypatch.setattr(net, "fetch", lambda url, *, validate=None, **kwargs: (calls.append(url), body)[1])

    ensure_doc_tree(VERSION, source="tarball")
    ensure_doc_tree(VERSION, source="tarball", refresh=True)

    assert len(calls) == 2


class _DocTreeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _serve(directory: Path) -> tuple[http.server.HTTPServer, str, threading.Thread]:
    handler = lambda *args, **kwargs: _DocTreeHandler(*args, directory=str(directory), **kwargs)  # noqa: E731
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/", thread


@pytest.fixture
def doc_site(tmp_path: Path):
    site_root = tmp_path / "site"
    (site_root / "asio").mkdir(parents=True)
    (site_root / "index.html").write_text('<a href="asio/overview.html">overview</a>')
    (site_root / "asio" / "overview.html").write_text(
        '<a href="reference.html">reference</a> <a href="history.html#frag">history</a> '
        '<a href="net_ts.html">net_ts</a> <a href="index.html">skip me</a>'
    )
    (site_root / "asio" / "reference.html").write_text(
        '<a href="overview.html">back</a> <a href="reference/socket.html">socket</a>'
    )
    (site_root / "asio" / "reference").mkdir()
    (site_root / "asio" / "reference" / "socket.html").write_text('<a href="../only_via_parent.html">up</a>')
    (site_root / "asio" / "only_via_parent.html").write_text("<html>reachable only through ../</html>")
    (site_root / "asio" / "history.html").write_text(
        '<html>history <a href="www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4370.html">n4370</a></html>'
    )
    (site_root / "asio" / "net_ts.html").write_text("<html>excluded from man pages, still mirrored</html>")
    (site_root / "asio" / "index.html").write_text('<a href="only_via_index.html">x</a>')
    (site_root / "asio" / "only_via_index.html").write_text("<html>reachable only through the index</html>")

    server, url, thread = _serve(site_root)
    try:
        yield url
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_online_crawl_mirrors_every_reachable_page(doc_site: str) -> None:
    doc_dir = ensure_doc_tree(VERSION, source="online", _online_root=doc_site)

    assert (doc_dir / "index.html").is_file()
    assert (doc_dir / "asio" / "overview.html").is_file()
    assert (doc_dir / "asio" / "reference.html").is_file()
    assert (doc_dir / "asio" / "history.html").is_file()
    assert (doc_dir / "asio" / "net_ts.html").is_file()
    assert (doc_dir / "asio" / "only_via_index.html").is_file()
    assert (doc_dir / "asio" / "only_via_parent.html").is_file()
    # A bare "www.example.org/path" href (real Asio docs cite standards papers this
    # way, missing the scheme) must not be mistaken for a same-site relative link.
    assert not (doc_dir / "asio" / "www.open-std.org").exists()


def test_online_crawl_reports_failed_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    (site_root / "index.html").write_text('<a href="missing.html">gone</a>')
    server, url, thread = _serve(site_root)
    try:
        with pytest.raises(AsioDocsError):
            ensure_doc_tree(VERSION, source="online", _online_root=url)
    finally:
        server.shutdown()
        thread.join(timeout=5)
