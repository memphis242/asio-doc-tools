"""HTTP GET with retries, plus an on-disk response cache.

Version-pinned Asio URLs (asio-1.38.2/doc/...) never change, so callers cache
them forever (max_age_s=None). Pages that move with new releases (the site
landing page) get a TTL. When a refresh fails but a stale copy exists, the
stale copy is used with a warning, so the tools keep working offline.
"""

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

from . import __version__, paths
from .diag import AsioDocsError, warn

USER_AGENT: Final = f"asio-doc-tools/{__version__}"
DEFAULT_TIMEOUT_S: Final = 60.0
DEFAULT_ATTEMPTS: Final = 4
_RETRYABLE_HTTP_STATUS: Final = frozenset({408, 425, 429, 500, 502, 503, 504})

# Returns None when the body is acceptable, else a short reason. A rejected body
# is retried like a transient failure (e.g. a download mirror answering 200 with
# an HTML error page instead of the archive).
Validator = Callable[[bytes], str | None]


class FetchError(AsioDocsError):
    """A URL could not be fetched (after retries, where retrying made sense)."""


def fetch(
    url: str,
    *,
    validate: Validator | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> bytes:
    # Imported here rather than at module level: they cost tens of milliseconds, and a
    # run served entirely from the cache never needs them.
    import http.client
    import urllib.error
    import urllib.request

    assert attempts >= 1
    last_failure = "no attempt made"
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body: bytes = response.read()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_HTTP_STATUS:
                raise FetchError(f"GET {url} failed: HTTP {e.code} {e.reason}") from e
            last_failure = f"HTTP {e.code} {e.reason}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            # http.client.HTTPException covers a connection that drops mid-response
            # (IncompleteRead and friends), which response.read() raises directly
            # rather than wrapping in a URLError.
            last_failure = str(getattr(e, "reason", e))
        else:
            rejection = validate(body) if validate is not None else None
            if rejection is None:
                return body
            last_failure = f"unexpected response body ({rejection})"
        if attempt < attempts:
            time.sleep(min(2.0**attempt, 30.0))
    raise FetchError(
        f"GET {url} failed after {attempts} attempts: {last_failure}. "
        "Check your network connection, or retry later if the server is having trouble."
    )


def _cache_file(url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()
    return paths.cache_dir() / "http" / digest[:2] / digest


def atomic_write(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """Write via a temp file + rename so readers never see a partial file.

    `mode` overrides the permissions `mkstemp` gives the temp file (0o600, so
    normally owner-only); pass e.g. 0o644 for output meant to be world-readable,
    such as an installed man page. `fd` is handed to `os.fdopen` immediately, so
    the `with` block's own close covers every exit path (including the chmod or
    the write raising) and no descriptor can survive past this call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            if mode is not None:
                os.fchmod(f.fileno(), mode)
            f.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def fetch_cached(
    url: str,
    *,
    max_age_s: float | None,
    refresh: bool = False,
    validate: Validator | None = None,
) -> bytes:
    """GET through the cache. max_age_s=None means a cached copy never expires."""
    body_path = _cache_file(url)
    cached = body_path.is_file()
    if cached and not refresh:
        age_s = time.time() - body_path.stat().st_mtime
        if max_age_s is None or age_s <= max_age_s:
            return body_path.read_bytes()
    try:
        body = fetch(url, validate=validate)
    except FetchError as e:
        if not cached:
            raise
        warn(f"{e} Using the cached copy from {time.ctime(body_path.stat().st_mtime)}.")
        return body_path.read_bytes()
    atomic_write(body_path, body)
    atomic_write(
        body_path.with_suffix(".meta.json"),
        json.dumps({"url": url, "fetched_at": time.time()}).encode(),
    )
    return body
