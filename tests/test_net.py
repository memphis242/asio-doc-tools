import http.client
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from asio_doc_tools import net
from asio_doc_tools.diag import AsioDocsError


@pytest.fixture(autouse=True)
def _xdg_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # The retry backoff sleeps up to 30s; tests only care about the retry logic.
    monkeypatch.setattr(net.time, "sleep", lambda seconds: None)


def _fake_urlopen_sequence(responses):
    """Returns a urlopen stand-in that yields each item of `responses` in turn.

    An item that is an exception instance is raised; otherwise it is returned as
    a context manager whose `.read()` produces that item (bytes), or raises it if
    it's an exception raised specifically from read() rather than urlopen().
    """
    iterator = iter(responses)

    class _Response:
        def __init__(self, outcome):
            self._outcome = outcome

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            if isinstance(self._outcome, BaseException):
                raise self._outcome
            return self._outcome

    def urlopen(request, timeout):  # noqa: ANN001
        del request, timeout
        outcome = next(iterator)
        if isinstance(outcome, BaseException) and not isinstance(outcome, http.client.HTTPException):
            raise outcome
        return _Response(outcome)

    return urlopen


def test_fetch_retries_on_incomplete_read_and_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = [http.client.IncompleteRead(b"partial"), b"the full body"]
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_sequence(sequence))

    body = net.fetch("http://example.invalid/thing", attempts=2)

    assert body == b"the full body"


def test_fetch_retries_on_incomplete_read_and_eventually_raises_fetcherror(monkeypatch: pytest.MonkeyPatch) -> None:
    always_incomplete = [http.client.IncompleteRead(b"partial") for _ in range(3)]
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_sequence(always_incomplete))

    with pytest.raises(net.FetchError):
        net.fetch("http://example.invalid/thing", attempts=3)


def test_fetch_retries_on_generic_http_exception_from_read(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = [http.client.BadStatusLine("garbage"), b"ok"]
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_sequence(sequence))

    assert net.fetch("http://example.invalid/thing", attempts=2) == b"ok"


def test_fetch_nonretryable_http_error_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def urlopen(request, timeout):  # noqa: ANN001
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(net.FetchError):
        net.fetch("http://example.invalid/thing", attempts=4)

    assert len(calls) == 1  # 404 is not retried


# -- atomic_write ----------------------------------------------------------------


def test_atomic_write_default_mode(tmp_path: Path) -> None:
    target = tmp_path / "out" / "file.bin"
    net.atomic_write(target, b"hello")
    assert target.read_bytes() == b"hello"


def test_atomic_write_applies_requested_mode(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    net.atomic_write(target, b"hello", mode=0o644)
    assert (target.stat().st_mode & 0o777) == 0o644


def test_atomic_write_does_not_leak_file_descriptors(tmp_path: Path) -> None:
    def open_fd_count() -> int:
        return len(os.listdir("/proc/self/fd"))

    before = open_fd_count()
    for i in range(64):
        net.atomic_write(tmp_path / f"file{i}.bin", b"x", mode=0o644)
    after = open_fd_count()
    assert after <= before + 4


def test_atomic_write_cleans_up_temp_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "file.bin"

    def failing_fchmod(fd, mode):  # noqa: ANN001
        raise OSError("simulated chmod failure")

    monkeypatch.setattr(net.os, "fchmod", failing_fchmod)

    with pytest.raises(OSError):
        net.atomic_write(target, b"hello", mode=0o644)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []  # no leftover temp file


def test_fetch_cached_uses_stale_copy_when_refetch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fetch_once_then_fail(url, *, validate=None):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return b"first body"
        raise net.FetchError("simulated network failure")

    monkeypatch.setattr(net, "fetch", fetch_once_then_fail)

    first = net.fetch_cached("http://example.invalid/x", max_age_s=0)
    assert first == b"first body"

    # A second call with a failing fetch should fall back to the cached copy
    # rather than raising, since one is on disk.
    second = net.fetch_cached("http://example.invalid/x", max_age_s=0, refresh=True)
    assert second == b"first body"
