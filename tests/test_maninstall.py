import gzip
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from asio_doc_tools import maninstall, paths
from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.manpage import ManPage, Section
from asio_doc_tools.versions import Version

# Captured before the autouse _fake_mandb fixture monkeypatches subprocess.run
# (the same module object maninstall imports), so spawning a real subprocess for
# the low-fd-limit test below still uses the real implementation.
_REAL_SUBPROCESS_RUN = subprocess.run

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


def _manifest(tmp_path: Path) -> dict:
    return json.loads((paths.data_dir() / "installed-man-pages.json").read_text())


# -- write_tree ----------------------------------------------------------------


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


def test_write_tree_pages_are_world_readable(tmp_path: Path) -> None:
    # mkstemp defaults to 0o600 (owner-only); installed pages must be readable by
    # anyone consulting `man`, not just the user who ran the install.
    written = maninstall.write_tree((_page("asio.a"),), tmp_path)
    mode = written[0].stat().st_mode & 0o777
    assert mode == 0o644


def test_write_tree_does_not_leak_file_descriptors(tmp_path: Path) -> None:
    pages = tuple(_page(f"asio.p{i}") for i in range(64))

    def open_fd_count() -> int:
        return len(os.listdir("/proc/self/fd"))

    before = open_fd_count()
    maninstall.write_tree(pages, tmp_path)
    after = open_fd_count()
    # A one-fd-per-page leak would show up as +64; allow a little slack for
    # anything unrelated the interpreter itself opens during the call.
    assert after <= before + 4


def test_man_build_survives_a_low_file_descriptor_limit(tmp_path: Path) -> None:
    # Reproduces the reported failure: with a leak, writing more pages than the
    # process's file descriptor limit raises EMFILE partway through. Run in a
    # subprocess so lowering RLIMIT_NOFILE never affects the test runner itself.
    script = textwrap.dedent(
        f"""
        import resource, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / "src")!r})
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        from asio_doc_tools.manpage import ManPage, Section
        from asio_doc_tools import maninstall
        pages = tuple(
            ManPage(name=f"asio.p{{i}}", section=Section.REFERENCE, source=f".TH asio.p{{i}} 3\\nbody\\n")
            for i in range(200)
        )
        maninstall.write_tree(pages, {str(tmp_path)!r})
        print("ok")
        """
    )
    result = _REAL_SUBPROCESS_RUN([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# -- install: manifest, conflicts, stale removal --------------------------------


def test_install_writes_manifest_and_runs_mandb(tmp_path: Path, _fake_mandb) -> None:
    man_dir = tmp_path / "man"
    pages = (_page("asio.a"), _page("asio.b"))

    report = maninstall.install(pages, man_dir, version=V1)

    assert report.written == 2
    assert report.removed == 0
    assert report.index_updated
    assert report.man_dir == man_dir.resolve()
    assert (man_dir / "man3" / "asio.a.3asio.gz").is_file()
    assert ["mandb", "--user-db", "--quiet", str(man_dir.resolve())] in _fake_mandb

    manifest = _manifest(tmp_path)
    key = str(man_dir.resolve())
    assert set(manifest.keys()) == {key}
    assert manifest[key]["version"] == str(V1)
    assert set(manifest[key]["paths"]) == {"man3/asio.a.3asio.gz", "man3/asio.b.3asio.gz"}


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


def test_install_force_overwrites_foreign_file_and_adopts_it(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    foreign = man_dir / "man3" / "asio.a.3asio.gz"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"not ours")

    maninstall.install((_page("asio.a"),), man_dir, version=V1, force=True)

    assert foreign.read_bytes() != b"not ours"
    manifest = _manifest(tmp_path)
    key = str(man_dir.resolve())
    assert "man3/asio.a.3asio.gz" in manifest[key]["paths"]

    # Adopted: a later plain install (no --force) no longer treats it as foreign.
    maninstall.install((_page("asio.a"),), man_dir, version=V1)


def test_install_reinstall_of_same_pages_is_not_a_conflict(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"),), man_dir, version=V1)
    maninstall.install((_page("asio.a"),), man_dir, version=V1)


def test_install_into_two_dirs_are_independent_records(tmp_path: Path) -> None:
    old_dir = tmp_path / "old-man"
    new_dir = tmp_path / "new-man"
    maninstall.install((_page("asio.a"),), old_dir, version=V1)

    report = maninstall.install((_page("asio.b"),), new_dir, version=V1)

    assert (old_dir / "man3" / "asio.a.3asio.gz").is_file()  # untouched
    assert (new_dir / "man3" / "asio.b.3asio.gz").is_file()
    assert all("uninstall" not in w for w in report.warnings)

    manifest = _manifest(tmp_path)
    assert set(manifest.keys()) == {str(old_dir.resolve()), str(new_dir.resolve())}


def test_install_relative_man_dir_resolves_to_the_same_record_from_another_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    maninstall.install((_page("asio.a"),), Path("man"), version=V1)
    absolute_man_dir = workdir / "man"
    assert (absolute_man_dir / "man3" / "asio.a.3asio.gz").is_file()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    # uninstall from a different cwd, addressing the same directory by its
    # absolute path: must find the record created above, not create/miss one.
    removed = maninstall.uninstall(absolute_man_dir)
    assert removed == 1
    assert not (absolute_man_dir / "man3" / "asio.a.3asio.gz").exists()


def test_install_symlinked_man_dir_resolves_to_the_real_path(tmp_path: Path) -> None:
    real_dir = tmp_path / "real-man"
    real_dir.mkdir()
    link = tmp_path / "link-man"
    link.symlink_to(real_dir)

    maninstall.install((_page("asio.a"),), link, version=V1)

    manifest = _manifest(tmp_path)
    assert set(manifest.keys()) == {str(real_dir.resolve())}
    assert (real_dir / "man3" / "asio.a.3asio.gz").is_file()


def test_install_interrupted_after_intent_record_then_reinstalled(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.old"),), man_dir, version=V1)

    # Simulate a crash right after the pre-write intent record was saved (the
    # union of old and new paths) but before the new pages were actually written
    # or the stale ones removed: hand-write that intermediate manifest state.
    key = str(man_dir.resolve())
    manifest = _manifest(tmp_path)
    manifest[key]["paths"] = sorted(set(manifest[key]["paths"]) | {"man3/asio.new.3asio.gz"})
    (paths.data_dir() / "installed-man-pages.json").write_text(json.dumps(manifest))

    # A later install must not be confused by the manifest listing a page that
    # was never actually written; it should install cleanly and clean up.
    report = maninstall.install((_page("asio.new"),), man_dir, version=V2)

    assert (man_dir / "man3" / "asio.new.3asio.gz").is_file()
    assert not (man_dir / "man3" / "asio.old.3asio.gz").exists()
    assert report.removed == 1  # asio.old was stale relative to the final page set


def test_install_saves_an_incomplete_intent_record_before_writing_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.old"),), man_dir, version=V1)

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("simulated crash mid-install")

    monkeypatch.setattr(maninstall, "write_tree", boom)

    with pytest.raises(RuntimeError):
        maninstall.install((_page("asio.new"),), man_dir, version=V2)

    # The intent record on disk must not claim the new version is installed: the
    # old version is still what is actually there, and `incomplete` says so.
    manifest = _manifest(tmp_path)
    key = str(man_dir.resolve())
    assert manifest[key]["incomplete"] is True
    assert manifest[key]["version"] == str(V1)

    results = maninstall.status()
    assert len(results) == 1
    assert results[0].incomplete
    assert results[0].version == V1


def test_status_is_not_incomplete_after_a_normal_install(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"),), man_dir, version=V1)

    results = maninstall.status()

    assert len(results) == 1
    assert not results[0].incomplete


# -- symlink safety in stale removal / uninstall ---------------------------------


def test_stale_removal_unlinks_a_symlinked_page_without_touching_its_target(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.old"),), man_dir, version=V1)

    # Replace the installed (stale-to-be) page with a symlink to some unrelated
    # file, simulating tampering between install and the next install's cleanup.
    stale_path = man_dir / "man3" / "asio.old.3asio.gz"
    target = tmp_path / "external-target.txt"
    target.write_text("do not touch")
    stale_path.unlink()
    stale_path.symlink_to(target)

    report = maninstall.install((_page("asio.new"),), man_dir, version=V2)

    assert report.removed == 1
    assert not stale_path.exists()
    assert not stale_path.is_symlink()  # the symlink itself is gone
    assert target.read_text() == "do not touch"  # unlink() never followed it


def test_uninstall_refuses_to_delete_through_a_symlinked_section_dir(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"),), man_dir, version=V1)

    # Replace man3 itself with a symlink to a directory that happens to hold a
    # file with the same name, simulating man_dir being tampered with after
    # install. Deleting "through" that symlink would remove a file that is not
    # actually inside man_dir at all.
    real_man3 = man_dir / "man3"
    outside = tmp_path / "outside-man3"
    shutil.copytree(real_man3, outside)
    shutil.rmtree(real_man3)
    real_man3.symlink_to(outside)

    with pytest.raises(AsioDocsError):
        maninstall.uninstall(man_dir)

    # Nothing was deleted through the symlink, and the record is still there:
    # this call refused outright rather than removing what it safely could.
    assert (outside / "asio.a.3asio.gz").is_file()
    manifest = _manifest(tmp_path)
    assert str(man_dir.resolve()) in manifest


# -- manifest validation ---------------------------------------------------------


def _write_raw_manifest(content: object) -> None:
    manifest_path = paths.data_dir() / "installed-man-pages.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(content))


def test_status_rejects_manifest_with_absolute_looking_path(tmp_path: Path) -> None:
    _write_raw_manifest({str(tmp_path / "man"): {"version": str(V1), "installed_at": 0.0, "paths": ["/etc/passwd"]}})
    with pytest.raises(AsioDocsError):
        maninstall.status()


def test_status_rejects_manifest_with_traversal_path(tmp_path: Path) -> None:
    _write_raw_manifest(
        {str(tmp_path / "man"): {"version": str(V1), "installed_at": 0.0, "paths": ["man3/../../../etc/passwd"]}}
    )
    with pytest.raises(AsioDocsError):
        maninstall.status()


def test_status_rejects_manifest_with_non_list_paths(tmp_path: Path) -> None:
    _write_raw_manifest(
        {str(tmp_path / "man"): {"version": str(V1), "installed_at": 0.0, "paths": "man3/asio.a.3asio.gz"}}
    )
    with pytest.raises(AsioDocsError):
        maninstall.status()


def test_status_rejects_manifest_with_unparsable_version_cleanly(tmp_path: Path) -> None:
    _write_raw_manifest({str(tmp_path / "man"): {"version": "not-a-version", "installed_at": 0.0, "paths": []}})
    # Must surface as a clean AsioDocsError, never an uncaught ValueError/traceback.
    with pytest.raises(AsioDocsError):
        maninstall.status()


def test_status_rejects_manifest_with_relative_key(tmp_path: Path) -> None:
    _write_raw_manifest({"man": {"version": str(V1), "installed_at": 0.0, "paths": []}})
    with pytest.raises(AsioDocsError):
        maninstall.status()


# -- uninstall -------------------------------------------------------------------


def test_uninstall_removes_manifest_record_and_files(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"), _page("asio.b")), man_dir, version=V1)

    removed = maninstall.uninstall(None)

    assert removed == 2
    assert not (man_dir / "man3" / "asio.a.3asio.gz").exists()
    assert _manifest(tmp_path) == {}


def test_uninstall_with_nothing_installed_is_a_user_error() -> None:
    with pytest.raises(AsioDocsError):
        maninstall.uninstall(None)


def test_uninstall_without_man_dir_and_multiple_records_is_a_user_error(tmp_path: Path) -> None:
    maninstall.install((_page("asio.a"),), tmp_path / "man-a", version=V1)
    maninstall.install((_page("asio.b"),), tmp_path / "man-b", version=V1)

    with pytest.raises(AsioDocsError):
        maninstall.uninstall(None)

    # Neither record was touched by the failed, ambiguous uninstall.
    manifest = _manifest(tmp_path)
    assert len(manifest) == 2


def test_uninstall_mismatched_man_dir_keeps_all_records(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"),), man_dir, version=V1)

    with pytest.raises(AsioDocsError):
        maninstall.uninstall(tmp_path / "somewhere-else")

    # The mismatched call must not delete anything: the real record is intact.
    assert (man_dir / "man3" / "asio.a.3asio.gz").is_file()
    manifest = _manifest(tmp_path)
    assert str(man_dir.resolve()) in manifest


# -- status ------------------------------------------------------------------


def test_status_reports_counts_by_section(tmp_path: Path) -> None:
    man_dir = tmp_path / "man"
    maninstall.install((_page("asio.a"), _page("asio.b"), _page("asio.c", Section.TOPIC)), man_dir, version=V1)

    results = maninstall.status()

    assert len(results) == 1
    result = results[0]
    assert result.version == V1
    assert result.man_dir == man_dir.resolve()
    assert dict(result.count_by_section) == {"man3": 2, "man7": 1}


def test_status_lists_every_record(tmp_path: Path) -> None:
    maninstall.install((_page("asio.a"),), tmp_path / "man-a", version=V1)
    maninstall.install((_page("asio.b"),), tmp_path / "man-b", version=V2)

    results = maninstall.status()

    assert {r.man_dir for r in results} == {(tmp_path / "man-a").resolve(), (tmp_path / "man-b").resolve()}


def test_status_when_nothing_installed() -> None:
    assert maninstall.status() == ()


def test_mandb_missing_warns_but_install_still_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(maninstall.shutil, "which", lambda name: None)
    man_dir = tmp_path / "man"

    report = maninstall.install((_page("asio.a"),), man_dir, version=V1)

    assert not report.index_updated
    assert report.written == 1
    assert any("mandb" in w for w in report.warnings)
