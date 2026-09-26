import gzip
import json
from pathlib import Path

import pytest

from asio_doc_tools import maninstall, paths
from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.manpage import ManPage, Section
from asio_doc_tools.versions import Version

V1 = Version(1, 38, 2)
V2 = Version(1, 39, 0)


def _page(name: str, section: Section = Section.REFERENCE) -> ManPage:
    return ManPage(name=name, section=section, source=f".TH {name} 3\nbody for {name}\n")


@pytest.fixture(autouse=True)
def _xdg_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


@pytest.fixture(autouse=True)
def _fake_mandb(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_which(name: str) -> str | None:
        return "/usr/bin/mandb" if name in ("mandb", "manpath") else None

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        calls.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(maninstall.shutil, "which", fake_which)
    monkeypatch.setattr(maninstall.subprocess, "run", fake_run)
    return calls


def test_write_tree_compresses_and_lays_out_by_section(tmp_path: Path) -> None:
    pages = (_page("asio.a", Section.REFERENCE), _page("asio.b", Section.TOPIC))
    written = maninstall.write_tree(pages, tmp_path)

    assert len(written) == 2
    ref_file = tmp_path / "man3" / "asio.a.3asio.gz"
    topic_file = tmp_path / "man7" / "asio.b.7.gz"
    assert ref_file.is_file() and topic_file.is_file()
    assert gzip.decompress(ref_file.read_bytes()).decode() == pages[0].source


def test_write_tree_uncompressed(tmp_path: Path) -> None:
    pages = (_page("asio.a"),)
    written = maninstall.write_tree(pages, tmp_path, compress=False)
    assert written[0].read_text() == pages[0].source


def test_write_tree_rejects_duplicate_filenames(tmp_path: Path) -> None:
    pages = (_page("asio.a"), _page("asio.a"))
    with pytest.raises(AssertionError):
        maninstall.write_tree(pages, tmp_path)


def test_install_writes_manifest_and_runs_mandb(tmp_path: Path, _fake_mandb) -> None:
    man_dir = tmp_path / "man"
    pages = (_page("asio.a"), _page("asio.b"))

    report = maninstall.install(pages, man_dir, version=V1)

    assert report.written == 2
    assert report.removed == 0
    assert report.index_updated
    assert (man_dir / "man3" / "asio.a.3asio.gz").is_file()
    assert ["mandb", "--user-db", "--quiet", str(man_dir)] in _fake_mandb

    manifest = json.loads((paths.data_dir() / "installed-man-pages.json").read_text())
    assert manifest["man_dir"] == str(man_dir)
    assert manifest["version"] == str(V1)
    assert set(manifest["paths"]) == {"man3/asio.a.3asio.gz", "man3/asio.b.3asio.gz"}


def test_install_removes_stale_pages_from_previous_version(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.old"),), man_dir, version=V1)
    assert (man_dir / "man3" / "asio.old.3asio.gz").is_file()

    report = maninstall.install((_page("asio.new"),), man_dir, version=V2)

    assert report.removed == 1
    assert not (man_dir / "man3" / "asio.old.3asio.gz").exists()
    assert (man_dir / "man3" / "asio.new.3asio.gz").is_file()


def test_install_refuses_to_overwrite_foreign_file_without_force(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    foreign = man_dir / "man3" / "asio.a.3asio.gz"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"not ours")

    with pytest.raises(AsioDocsError):
        maninstall.install((_page("asio.a"),), man_dir, version=V1)

    assert foreign.read_bytes() == b"not ours"


def test_install_force_overwrites_foreign_file(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    foreign = man_dir / "man3" / "asio.a.3asio.gz"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"not ours")

    maninstall.install((_page("asio.a"),), man_dir, version=V1, force=True)

    assert foreign.read_bytes() != b"not ours"


def test_install_reinstall_of_same_pages_is_not_a_conflict(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"),), man_dir, version=V1)
    # Reinstalling the same page set should not be treated as a foreign-file conflict.
    maninstall.install((_page("asio.a"),), man_dir, version=V1)


def test_install_different_man_dir_leaves_old_files_alone_and_warns(tmp_path: Path, capsys) -> None:
    old_dir = tmp_path / "old-man"
    new_dir = tmp_path / "new-man"
    maninstall.install((_page("asio.a"),), old_dir, version=V1)

    report = maninstall.install((_page("asio.b"),), new_dir, version=V1)

    assert (old_dir / "man3" / "asio.a.3asio.gz").is_file()  # untouched
    assert (new_dir / "man3" / "asio.b.3asio.gz").is_file()
    assert any("uninstall" in w for w in report.warnings)


def test_uninstall_removes_manifest_and_files(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"), _page("asio.b")), man_dir, version=V1)

    removed = maninstall.uninstall(None)

    assert removed == 2
    assert not (man_dir / "man3" / "asio.a.3asio.gz").exists()
    assert not (paths.data_dir() / "installed-man-pages.json").exists()


def test_uninstall_with_nothing_installed_is_a_user_error() -> None:
    with pytest.raises(AsioDocsError):
        maninstall.uninstall(None)


def test_status_reports_counts_by_section(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"), _page("asio.b"), _page("asio.c", Section.TOPIC)), man_dir, version=V1)

    result = maninstall.status()

    assert result.installed
    assert result.version == V1
    assert result.man_dir == man_dir
    assert dict(result.count_by_section) == {"man3": 2, "man7": 1}


def test_status_when_nothing_installed() -> None:
    result = maninstall.status()
    assert not result.installed


def test_mandb_missing_warns_but_install_still_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(maninstall.shutil, "which", lambda name: None)
    man_dir = tmp_path / "man"

    report = maninstall.install((_page("asio.a"),), man_dir, version=V1)

    assert not report.index_updated
    assert report.written == 1
    assert any("mandb" in w for w in report.warnings)
