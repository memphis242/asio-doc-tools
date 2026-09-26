"""Obtaining a release's HTML docs as a local directory tree.

Two ways to get the same tree that `versions.doc_root_url(version)` serves online:
a release tarball (fast, one download, preferred) or crawling the live site page
by page (fallback, many requests). Either way the result lands at
`paths.cache_dir()/docs/asio-X.Y.Z/doc/` with a `.complete` marker recording how
it was obtained. The marker is written only once the new tree is completely in
place, and a previous tree (on a `refresh`) is moved aside rather than deleted
until then, so a caller can never observe a half-populated tree, nor a marker
left pointing at one a crashed run only partly deleted.
"""

import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from . import net, paths
from .diag import AsioDocsError, note, warn
from .links import is_bare_external_host
from .versions import Version, doc_root_url

Source = Literal["auto", "tarball", "online"]

_SOURCEFORGE_FOLDERS: Final = ("Stable", "Development")
_TARBALL_MAGIC: Final = b"BZh"
_REQUIRED_DOC_FILES: Final = ("index.html", "asio/reference.html", "asio/history.html", "asio/overview.html")
_CRAWL_WORKERS: Final = 8
_CRAWL_PROGRESS_INTERVAL: Final = 200

# Fetches one URL's bytes; swappable in tests for a fake/local server.
Fetcher = Callable[[str], bytes]


def _tree_dir(version: Version) -> Path:
    return paths.cache_dir() / "docs" / f"asio-{version}"


def _marker(version: Version) -> Path:
    return _tree_dir(version) / ".complete"


def _doc_dir(version: Version) -> Path:
    return _tree_dir(version) / "doc"


def _check_required_files(doc_dir: Path, *, how: str) -> None:
    missing = [name for name in _REQUIRED_DOC_FILES if not (doc_dir / name).is_file()]
    if missing:
        raise AsioDocsError(
            f"the doc tree obtained via {how} is missing expected files: {', '.join(missing)}"
        )


def _install_tree(version: Version, populate: Callable[[Path], None], *, how: str) -> Path:
    """Populate a fresh temp dir under the cache, then swap it into place atomically.

    `populate` receives an empty directory to fill with the `doc/` tree's contents
    (i.e. `tmp/index.html`, `tmp/asio/...`). Nothing under `_tree_dir(version)` is
    touched until the populated tree is verified and ready to become permanent.

    The swap itself: the marker is removed first (so a tree without one is never
    mistaken for complete), the old tree (if any) is renamed aside rather than
    deleted, the new tree is renamed into the old tree's place, the marker is
    written, and only then is the old tree actually removed. If the process is
    interrupted anywhere in that sequence, what is left is either the untouched
    old tree with no marker (so the next call redoes the work) or the new tree
    with its marker (fully installed); there is no state in between that looks
    complete.
    """
    tree_dir = _tree_dir(version)
    tree_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=tree_dir.parent, prefix=f".asio-{version}-new-"))
    old_aside = tree_dir.parent / f".asio-{version}-old-{os.getpid()}"
    marker = _marker(version)
    previous_marker_text = marker.read_text() if marker.is_file() else None
    moved_old_aside = False
    try:
        doc_tmp = tmp / "doc"
        doc_tmp.mkdir()
        populate(doc_tmp)
        _check_required_files(doc_tmp, how=how)
        if tree_dir.exists():
            marker.unlink(missing_ok=True)
            tree_dir.rename(old_aside)
            moved_old_aside = True
        try:
            tmp.rename(tree_dir)
        except OSError:
            if moved_old_aside:
                old_aside.rename(tree_dir)
                moved_old_aside = False
                if previous_marker_text is not None:
                    marker.write_text(previous_marker_text)
            raise
        marker.write_text(f"{how}\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if moved_old_aside:
            shutil.rmtree(old_aside, ignore_errors=True)
    return _doc_dir(version)


def _tarball_url(version: Version, folder: str) -> str:
    return f"https://downloads.sourceforge.net/project/asio/asio/{version}%20%28{folder}%29/asio-{version}.tar.bz2"


def _validate_tarball(body: bytes) -> str | None:
    return None if body.startswith(_TARBALL_MAGIC) else "does not start with the bzip2 magic 'BZh'"


def _extract_doc_tree(archive_path: Path, version: Version, doc_tmp: Path) -> None:
    prefix = f"asio-{version}/doc/"
    try:
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            members = [m for m in archive.getmembers() if m.name.startswith(prefix) and m.name != prefix]
            if not members:
                raise AsioDocsError(f"tarball for asio-{version} has no members under '{prefix}'")
            for member in members:
                member.name = member.name[len(prefix) :]
            try:
                archive.extractall(doc_tmp, members=members, filter="data")
            except tarfile.FilterError as e:
                raise AsioDocsError(
                    f"tarball for asio-{version} contains a member outside its 'doc/' tree ({e}); "
                    "refusing to extract it."
                ) from e
    except (tarfile.ReadError, EOFError, OSError) as e:
        # A mirror that truncates the download, or answers with a corrupt file that
        # still happens to start with the bzip2 magic. Reported the same way as any
        # other tarball problem, so "auto" falls back to crawling instead of a raw
        # traceback from deep inside tarfile.
        raise AsioDocsError(f"tarball for asio-{version} at {archive_path} is corrupt or truncated: {e}") from e


def _fetch_tarball(version: Version) -> Path | None:
    """Downloads the release tarball to the cache and returns its path, or None if unavailable."""
    for folder in _SOURCEFORGE_FOLDERS:
        url = _tarball_url(version, folder)
        try:
            body = net.fetch(url, validate=_validate_tarball)
        except net.FetchError as e:
            note(f"tarball not available at {url} ({e})")
            continue
        cache_path = paths.cache_dir() / "tarballs" / f"asio-{version}.tar.bz2"
        net.atomic_write(cache_path, body)
        return cache_path
    return None


def _ensure_via_tarball(version: Version) -> Path:
    archive_path = _fetch_tarball(version)
    if archive_path is None:
        raise AsioDocsError(
            f"could not find a release tarball for asio-{version} in the Stable or Development "
            "SourceForge folders."
        )
    return _install_tree(version, lambda doc_tmp: _extract_doc_tree(archive_path, version, doc_tmp), how="tarball")


def _is_same_site_link(href: str) -> bool:
    return bool(href) and not href.startswith(("http://", "https://", "mailto:")) and not is_bare_external_host(href)


def _links_of(html: bytes) -> Iterable[str]:
    for anchor in BeautifulSoup(html, "lxml").find_all("a", href=True):
        href = anchor["href"].strip()
        if _is_same_site_link(href):
            yield href


def _is_safe_relative_html_path(relative: str) -> bool:
    """False for anything that could not be joined onto doc_tmp without escaping it.

    A URL with an unexpected extra slash (e.g. ".../doc//etc/x.html") can, after
    the root prefix is stripped, leave a path that starts with "/" or contains a
    "." or ".." segment; `Path` joins an absolute path by discarding the base
    entirely, so that would otherwise write outside the doc tree.
    """
    if not relative or relative.startswith("/"):
        return False
    return all(segment not in ("", ".", "..") for segment in relative.split("/"))


def _relative_path(root: str, url: str) -> str | None:
    """The path of `url` relative to `root`, or None if it falls outside the doc root."""
    if not url.startswith(root):
        return None
    relative = urlsplit(url[len(root) :]).path  # drop any leftover query string
    return relative if _is_safe_relative_html_path(relative) else None


def _crawl(root: str, doc_tmp: Path, fetcher: Fetcher) -> None:
    """Breadth-first crawl of every `.html` page reachable from `root`'s index.html.

    The crawl mirrors the whole reachable tree, including pages the man page
    generator leaves out: some pages are linked only from those (e.g. the keyword
    index), and deciding what becomes a man page is the generator's job.
    """
    pending = {"index.html"}
    done: set[str] = set()
    failed: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=_CRAWL_WORKERS) as pool:
        while pending:
            batch = sorted(pending - done)
            pending.clear()
            futures = {pool.submit(fetcher, urljoin(root, path)): path for path in batch}
            for future in as_completed(futures):
                path = futures[future]
                done.add(path)
                try:
                    body = future.result()
                except net.FetchError as e:
                    failed[path] = str(e)
                    continue
                destination = doc_tmp / path
                # _relative_path (the only source of `pending`, besides the trivially
                # safe "index.html" seed) already rejects anything that could escape
                # doc_tmp; this is the last-resort check on that invariant.
                assert destination.resolve().is_relative_to(doc_tmp.resolve()), path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                for href in _links_of(body):
                    absolute = urljoin(urljoin(root, path), href.split("#", 1)[0])
                    relative = _relative_path(root, absolute)
                    if relative is None or not relative.endswith(".html") or relative in done:
                        continue
                    pending.add(relative)
                if len(done) % _CRAWL_PROGRESS_INTERVAL == 0:
                    note(f"crawled {len(done)} pages ({len(pending)} queued)...")

    if failed:
        sample = ", ".join(list(failed)[:5])
        raise AsioDocsError(
            f"failed to fetch {len(failed)} page(s) while crawling the online docs, "
            f"including: {sample}. Retry, or use --source tarball once it becomes available."
        )
    note(f"crawled {len(done)} pages.")


def _ensure_via_online(version: Version, *, root: str | None = None) -> Path:
    crawl_root = root if root is not None else doc_root_url(version)

    def fetcher(url: str) -> bytes:
        return net.fetch(url)

    return _install_tree(version, lambda doc_tmp: _crawl(crawl_root, doc_tmp, fetcher), how="online")


def ensure_doc_tree(
    version: Version,
    *,
    source: Source = "auto",
    refresh: bool = False,
    _online_root: str | None = None,
) -> Path:
    """A local directory laid out exactly like the release's `doc/` tree.

    `_online_root` is for tests and manual checks that crawl a local server
    instead of think-async.com; it only takes effect when `source` is "online"
    or "auto" falls back to crawling.
    """
    if not refresh and _marker(version).is_file():
        return _doc_dir(version)

    match source:
        case "tarball":
            return _ensure_via_tarball(version)
        case "online":
            return _ensure_via_online(version, root=_online_root)
        case "auto":
            try:
                return _ensure_via_tarball(version)
            except AsioDocsError as e:
                warn(f"falling back to crawling the online docs for asio-{version}: {e}")
                return _ensure_via_online(version, root=_online_root)
