"""Asio release versions and the think-async.com URLs derived from them."""

import re
from dataclasses import dataclass
from typing import Final, Self

from . import net
from .diag import AsioDocsError

SITE_URL: Final = "https://think-async.com/Asio/"
_LANDING_PAGE_TTL_S: Final = 12 * 3600.0

# Accepts 1.38.2, 1-38-2, v1.38.2, asio-1-38-2 (git tag), asio-1.38.2 (doc dir).
_VERSION_RE: Final = re.compile(r"^(?:asio[-_])?v?(\d+)[.\-_](\d+)[.\-_](\d+)$", re.IGNORECASE)
_DOC_LINK_RE: Final = re.compile(rb"asio-(\d+\.\d+\.\d+)/doc/")


@dataclass(frozen=True, order=True, slots=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> Self:
        """Raises ValueError for anything that is not a three-part version."""
        match = _VERSION_RE.match(text.strip())
        if match is None:
            raise ValueError(f"not an Asio version: {text!r} (expected e.g. 1.38.2)")
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def dashed(self) -> str:
        """The git tag spelling, e.g. 1-38-2 (tag asio-1-38-2)."""
        return f"{self.major}-{self.minor}-{self.patch}"


def doc_root_url(version: Version) -> str:
    """Root of a release's HTML docs; relative doc paths (asio/reference.html, ...) hang off it."""
    return f"{SITE_URL}asio-{version}/doc/"


def discover_latest(*, refresh: bool = False) -> Version:
    """The newest release whose docs the think-async.com landing page links to."""
    landing = net.fetch_cached(SITE_URL, max_age_s=_LANDING_PAGE_TTL_S, refresh=refresh)
    found = {Version.parse(m.decode()) for m in _DOC_LINK_RE.findall(landing)}
    if not found:
        raise AsioDocsError(
            f"could not find any 'asio-X.Y.Z/doc/' link on {SITE_URL}; the site layout may "
            "have changed. Pass an explicit version instead of 'latest'."
        )
    return max(found)


def resolve(spec: str, *, refresh: bool = False) -> Version:
    """Turn a user-supplied version (or the word 'latest') into a Version."""
    if spec.strip().lower() == "latest":
        return discover_latest(refresh=refresh)
    try:
        return Version.parse(spec)
    except ValueError as e:
        raise AsioDocsError(f"{e}; or use 'latest'") from None
