"""Writing generated man pages to disk and installing them into a man tree.

`write_tree` is the primitive both `install` and `asio-docs man build` use: it
just writes pages under a directory. `install` additionally tracks what it put
where, so a later install or uninstall knows which files are safe to remove and
which belong to someone else, and refreshes `mandb`'s user index so `man` and
`apropos` see the result.

The manifest at `paths.data_dir()/installed-man-pages.json` holds one record per
man directory a caller has installed into, keyed by that directory's canonical
(expanded and symlink-resolved) absolute path, so installing into two different
directories are simply two independent records rather than one replacing the
other. Callers are expected to pass an already-canonical `man_dir` (the CLI
resolves `--man-dir`/`--out` once, at the argument-parsing boundary); every
function here re-resolves it anyway, both so direct callers such as tests get
the same one-record-per-real-directory behavior and so a record is always
looked up and stored under the same key regardless of how the path was spelled.
"""

import gzip
import json
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import net, paths
from .diag import AsioDocsError, warn
from .manpage import ManPage
from .versions import Version

_MANIFEST_NAME: Final = "installed-man-pages.json"
_INSTALLED_PATH_RE: Final = re.compile(r"^man[0-9]+/asio[^/]*(?:\.gz)?$")
_INSTALLED_FILE_MODE: Final = 0o644


def _manifest_path() -> Path:
    return paths.data_dir() / _MANIFEST_NAME


def _canonical(man_dir: Path) -> Path:
    return man_dir.expanduser().resolve()


@dataclass(frozen=True, slots=True)
class InstallReport:
    man_dir: Path
    version: Version
    written: int
    removed: int
    index_updated: bool
    warnings: tuple[str, ...]


def write_tree(pages: Sequence[ManPage], root: Path, *, compress: bool = True) -> tuple[Path, ...]:
    """Writes each page under root/<subdir>/<filename>[.gz]; returns the written paths."""
    filenames = [page.filename for page in pages]
    assert len(filenames) == len(set(filenames)), f"duplicate man page filenames: {filenames}"
    written: list[Path] = []
    for page in pages:
        relative = Path(page.subdir) / (page.filename + (".gz" if compress else ""))
        destination = root / relative
        body = gzip.compress(page.source.encode(), mtime=0) if compress else page.source.encode()
        net.atomic_write(destination, body, mode=_INSTALLED_FILE_MODE)
        written.append(destination)
    return tuple(written)


def _relative_paths(pages: Sequence[ManPage], *, compress: bool) -> tuple[str, ...]:
    return tuple(
        str(Path(page.subdir) / (page.filename + (".gz" if compress else ""))) for page in pages
    )


@dataclass(frozen=True, slots=True)
class _Record:
    version: Version
    installed_at: float
    paths: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {"version": str(self.version), "installed_at": self.installed_at, "paths": list(self.paths)}


def _invalid(manifest_path: Path, reason: str) -> AsioDocsError:
    return AsioDocsError(f"{manifest_path} is not a valid install manifest ({reason}); remove it and reinstall.")


def _validate_paths(paths_field: object, *, manifest_path: Path, man_dir: str) -> tuple[str, ...]:
    if not isinstance(paths_field, list) or not all(isinstance(p, str) for p in paths_field):
        raise _invalid(manifest_path, f"'{man_dir}'.paths must be a list of strings")
    for relative in paths_field:
        if not _INSTALLED_PATH_RE.match(relative):
            raise _invalid(manifest_path, f"'{man_dir}' lists an unexpected installed path {relative!r}")
    return tuple(paths_field)


def _record_from_json(data: object, *, manifest_path: Path, man_dir: str) -> _Record:
    if not isinstance(data, dict):
        raise _invalid(manifest_path, f"the record for '{man_dir}' is not an object")
    try:
        version = Version.parse(str(data["version"]))
        installed_at = float(data["installed_at"])  # type: ignore[arg-type]
    except KeyError as e:
        raise _invalid(manifest_path, f"'{man_dir}' is missing field {e}") from e
    except (TypeError, ValueError) as e:
        raise _invalid(manifest_path, f"'{man_dir}' has an invalid version or timestamp ({e})") from e
    paths_field = _validate_paths(data.get("paths"), manifest_path=manifest_path, man_dir=man_dir)
    return _Record(version=version, installed_at=installed_at, paths=paths_field)


def _load_manifest() -> dict[str, _Record]:
    """Every recorded install, keyed by canonical man_dir. Empty if none is recorded."""
    manifest_path = _manifest_path()
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise _invalid(manifest_path, str(e)) from e
    if not isinstance(raw, dict):
        raise _invalid(manifest_path, "the top level must be an object of man_dir -> record")
    records: dict[str, _Record] = {}
    for man_dir, data in raw.items():
        if not isinstance(man_dir, str) or not Path(man_dir).is_absolute():
            raise _invalid(manifest_path, f"key {man_dir!r} is not an absolute man_dir path")
        records[man_dir] = _record_from_json(data, manifest_path=manifest_path, man_dir=man_dir)
    return records


def _save_manifest(records: dict[str, _Record]) -> None:
    manifest_path = _manifest_path()
    body = json.dumps({man_dir: record.to_json() for man_dir, record in records.items()}, indent=2).encode()
    net.atomic_write(manifest_path, body)


def _run_mandb(man_dir: Path) -> str | None:
    """Refreshes the user man index for man_dir; returns a warning message on failure, else None."""
    if shutil.which("mandb") is None:
        return "'mandb' was not found; run it manually to make new pages show up in 'man -k'/'apropos'."
    result = subprocess.run(
        ["mandb", "--user-db", "--quiet", str(man_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"'mandb --user-db --quiet {man_dir}' failed: {stderr or f'exit code {result.returncode}'}"
    return None


def _effective_manpath() -> tuple[Path, ...]:
    if shutil.which("manpath") is None:
        return ()
    result = subprocess.run(["manpath"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ()
    return tuple(Path(p).expanduser().resolve() for p in result.stdout.strip().split(":") if p)


def _remove_listed(man_dir: Path, relative_paths: Sequence[str]) -> int:
    removed = 0
    for relative in relative_paths:
        candidate = man_dir / relative
        # `relative` was already checked by _validate_paths (or, for a record built
        # in this call, is one of our own filenames); this is the last-resort check
        # on that invariant before anything is unlinked.
        assert candidate.resolve().is_relative_to(man_dir), candidate
        if candidate.is_file():
            candidate.unlink()
            removed += 1
    return removed


def install(
    pages: Sequence[ManPage], man_dir: Path, *, version: Version, force: bool = False
) -> InstallReport:
    man_dir = _canonical(man_dir)
    key = str(man_dir)
    records = _load_manifest()
    previous = records.get(key)
    previously_known = set(previous.paths) if previous is not None else set()
    warnings: list[str] = []

    new_relative = _relative_paths(pages, compress=True)
    stale = previously_known - set(new_relative)

    if not force:
        conflicts = [
            relative
            for relative in new_relative
            if relative not in previously_known and (man_dir / relative).exists()
        ]
        if conflicts:
            sample = ", ".join(conflicts[:5])
            more = f" (and {len(conflicts) - 5} more)" if len(conflicts) > 5 else ""
            raise AsioDocsError(
                f"{len(conflicts)} target file(s) already exist and were not installed by "
                f"asio-doc-tools: {sample}{more}. Use --force to overwrite them (they will then be "
                "tracked as this tool's own pages going forward)."
            )

    # Record intent before writing anything: the union of what was there before and
    # what is about to be written. If the process is interrupted after this point,
    # every file on disk that could plausibly need cleaning up is still listed by
    # some record, so a later install or uninstall never loses track of a page.
    intent_paths = tuple(sorted(previously_known | set(new_relative)))
    records[key] = _Record(version=version, installed_at=time.time(), paths=intent_paths)
    _save_manifest(records)

    written = write_tree(pages, man_dir, compress=True)
    assert len(written) == len(new_relative)

    removed = _remove_listed(man_dir, sorted(stale))

    records[key] = _Record(version=version, installed_at=time.time(), paths=tuple(sorted(new_relative)))
    _save_manifest(records)

    mandb_warning = _run_mandb(man_dir)
    if mandb_warning is not None:
        warnings.append(mandb_warning)
    index_updated = mandb_warning is None

    manpath_dirs = _effective_manpath()
    if manpath_dirs and man_dir not in manpath_dirs:
        warnings.append(
            f"{man_dir} is not on your manpath. Add it with "
            f'\'export MANPATH=":{man_dir}"\' (the leading colon keeps the system paths), '
            f"or add a line 'MANDATORY_MANPATH {man_dir}' to ~/.manpath."
        )

    for message in warnings:
        warn(message)

    return InstallReport(
        man_dir=man_dir,
        version=version,
        written=len(written),
        removed=removed,
        index_updated=index_updated,
        warnings=tuple(warnings),
    )


def uninstall(man_dir: Path | None) -> int:
    """Removes one recorded install's pages and its manifest record; returns the count removed."""
    records = _load_manifest()
    if man_dir is not None:
        man_dir = _canonical(man_dir)
        key = str(man_dir)
        if key not in records:
            known = ", ".join(sorted(records)) if records else "none"
            raise AsioDocsError(f"no asio-doc-tools man pages are recorded as installed in {man_dir} "
                                 f"(recorded install(s): {known}).")
    elif not records:
        raise AsioDocsError("no asio-doc-tools man pages are recorded as installed.")
    elif len(records) > 1:
        raise AsioDocsError(
            f"pages are installed in more than one man dir ({', '.join(sorted(records))}); "
            "pass --man-dir to say which one."
        )
    else:
        key = next(iter(records))
        man_dir = Path(key)

    removed = _remove_listed(man_dir, records[key].paths)
    del records[key]
    _save_manifest(records)

    mandb_warning = _run_mandb(man_dir)
    if mandb_warning is not None:
        warn(mandb_warning)

    return removed


@dataclass(frozen=True, slots=True)
class Status:
    man_dir: Path
    version: Version
    installed_at: float
    count_by_section: tuple[tuple[str, int], ...]


def status() -> tuple[Status, ...]:
    """Every recorded install, oldest man_dir spelling first."""
    records = _load_manifest()
    result: list[Status] = []
    for man_dir, record in sorted(records.items()):
        counts: dict[str, int] = {}
        for relative in record.paths:
            section = Path(relative).parent.name
            counts[section] = counts.get(section, 0) + 1
        result.append(
            Status(
                man_dir=Path(man_dir),
                version=record.version,
                installed_at=record.installed_at,
                count_by_section=tuple(sorted(counts.items())),
            )
        )
    return tuple(result)
