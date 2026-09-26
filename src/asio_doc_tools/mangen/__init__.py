"""Man pages from Asio's HTML documentation.

build_pages() turns a release's doc/ tree into ManPage objects; installing them
(man3/, man7/, compression, mandb) is the installer's job. See pages.py for the
build steps, naming.py for page names, and render.py for the roff conventions.
"""

from pathlib import Path

from ..manpage import ManPage
from ..versions import Version
from . import pages


def build_pages(doc_dir: Path, version: Version) -> tuple[ManPage, ...]:
    """Pure and deterministic: reads the HTML under doc_dir (laid out like a release's doc/ dir), no network."""
    return pages.build(Path(doc_dir), version)
