"""Writing generated man pages to disk and installing them into a man tree.

`write_tree` is the primitive both `install` and `asio-docs man build` use: it
just writes pages under a directory. `install` additionally tracks what it put
where (the manifest at `paths.data_dir()/installed-man-pages.json`), so a later
install or uninstall knows which files are safe to remove and which belong to
someone else, and refreshes `mandb`'s user index so `man`/`apropos` see the
result.
"""

import gzip
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import paths
from .diag import AsioDocsError, warn
from .manpage import ManPage
from .versions import Version

_MANIFEST_NAME: Final = "installed-man-pages.json"


def _manifest_path() -> Path:
    return paths.data_dir() / _MANIFEST_NAME


@dataclass(frozen=True, slots=True)
class InstallReport:
    man_dir: Path
    version: Version
    written: int
    removed: int
    index_updated: bool
    warnings: tuple[str, ...]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_tree(pages: Sequence[ManPage], root: Path, *, compress: bool = True) -> tuple[Path, ...]:
    """Writes each page under root/<subdir>/<filename>[.gz]; returns the written paths."""
    filenames = [page.filename for page in pages]
    assert len(filenames) == len(set(filenames)), f"duplicate man page filenames: {filenames}"
    written: list[Path] = []
    for page in pages:
        relative = Path(page.subdir) / (page.filename + (".gz" if compress else ""))
        destination = root / relative
        body = gzip.compress(page.source.encode(), mtime=0) if compress else page.source.encode()
        _atomic_write_bytes(destination, body)
        written.append(destination)
    return tuple(written)


def _relative_paths(pages: Sequence[ManPage], *, compress: bool) -> tuple[str, ...]:
    return tuple(
        str(Path(page.subdir) / (page.filename + (".gz" if compress else ""))) for page in pages
    )


@dataclass(frozen=True, slots=True)
class _Manifest:
    man_dir: str
    version: str
    installed_at: float
    paths: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "man_dir": self.man_dir,
            "version": self.version,
            "installed_at": self.installed_at,
            "paths": list(self.paths),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "_Manifest":
        return cls(
            man_dir=str(data["man_dir"]),
            version=str(data["version"]),
            installed_at=float(data["installed_at"]),  # type: ignore[arg-type]
            paths=tuple(str(p) for p in data["paths"]),  # type: ignore[union-attr]
        )


def _load_manifest() -> _Manifest | None:
    manifest_path = _manifest_path()
    if not manifest_path.is_file():
        return None
    try:
        return _Manifest.from_json(json.loads(manifest_path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise AsioDocsError(
            f"{manifest_path} is not a valid install manifest ({e}); remove it and reinstall."
        ) from e


def _save_manifest(manifest: _Manifest) -> None:
    manifest_path = _manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(manifest_path, json.dumps(manifest.to_json(), indent=2).encode())


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
    return tuple(Path(p) for p in result.stdout.strip().split(":") if p)


def install(
    pages: Sequence[ManPage], man_dir: Path, *, version: Version, force: bool = False
) -> InstallReport:
    previous = _load_manifest()
    warnings: list[str] = []

    new_relative = _relative_paths(pages, compress=True)
    if previous is not None and Path(previous.man_dir) == man_dir:
        stale = set(previous.paths) - set(new_relative)
        previously_known = set(previous.paths)
    else:
        stale = set()
        previously_known = set()
        if previous is not None:
            warnings.append(
                f"a previous install at {previous.man_dir} is unrelated to {man_dir}; "
                f"its files are left alone (run 'asio-docs man uninstall --man-dir {previous.man_dir}' "
                "to remove them)."
            )

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
                f"asio-doc-tools: {sample}{more}. Use --force to overwrite them."
            )

    written = write_tree(pages, man_dir, compress=True)
    assert len(written) == len(new_relative)

    removed = 0
    for relative in sorted(stale):
        stale_path = man_dir / relative
        if stale_path.is_file():
            stale_path.unlink()
            removed += 1

    _save_manifest(
        _Manifest(
            man_dir=str(man_dir),
            version=str(version),
            installed_at=time.time(),
            paths=tuple(sorted(new_relative)),
        )
    )

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
    """Removes installed pages and the manifest; returns the count removed."""
    previous = _load_manifest()
    if previous is None:
        raise AsioDocsError("no asio-doc-tools man pages are recorded as installed.")
    target_dir = man_dir if man_dir is not None else Path(previous.man_dir)

    removed = 0
    if target_dir == Path(previous.man_dir):
        for relative in previous.paths:
            candidate = target_dir / relative
            if candidate.is_file():
                candidate.unlink()
                removed += 1
    else:
        warn(
            f"the install manifest points at {previous.man_dir}, not {target_dir}; "
            "nothing was removed there. Run uninstall without --man-dir to remove the recorded install."
        )

    _manifest_path().unlink(missing_ok=True)

    mandb_warning = _run_mandb(target_dir)
    if mandb_warning is not None:
        warn(mandb_warning)

    return removed


@dataclass(frozen=True, slots=True)
class Status:
    installed: bool
    man_dir: Path | None
    version: Version | None
    installed_at: float | None
    count_by_section: tuple[tuple[str, int], ...]


def status() -> Status:
    manifest = _load_manifest()
    if manifest is None:
        return Status(installed=False, man_dir=None, version=None, installed_at=None, count_by_section=())
    counts: dict[str, int] = {}
    for relative in manifest.paths:
        section = Path(relative).parent.name
        counts[section] = counts.get(section, 0) + 1
    return Status(
        installed=True,
        man_dir=Path(manifest.man_dir),
        version=Version.parse(manifest.version),
        installed_at=manifest.installed_at,
        count_by_section=tuple(sorted(counts.items())),
    )
