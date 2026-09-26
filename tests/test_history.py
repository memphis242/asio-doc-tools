from collections.abc import Callable
from pathlib import Path

import pytest

from asio_doc_tools import history
from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.history import Entry, Release, entry_text_with_code, parse_history
from asio_doc_tools.versions import Version

# Mirrors the structure of asio/history.html: sibling <h4> headings, each followed
# by a div.itemizedlist (possibly with nested lists and code markup) or a <p>.
HISTORY_HTML = """
<div class="section">
<h2 class="title">Revision History</h2>
<h4><a name="asio.history.h0"></a><a name="asio.history.asio_1_38_2"></a>
  <a class="link" href="history.html#asio.history.asio_1_38_2">Asio
      1.38.2</a></h4>
<div class="itemizedlist"><ul class="itemizedlist">
  <li class="listitem">Fixed <code class="computeroutput"><span class="identifier">ip</span><span
      class="special">::</span><span class="identifier">tcp</span></code> handling.</li>
  <li class="listitem">Removed some previously deprecated facilities:
    <div class="itemizedlist"><ul class="itemizedlist">
      <li class="listitem">Removed <code>spawn()</code> overloads.</li>
      <li class="listitem">Removed <code>io_service</code>.</li>
    </ul></div>
  </li>
</ul></div>
<h4><a name="asio.history.asio_1_0_0"></a><a class="link" href="#">Asio 1.0.0</a></h4>
<p>First stable release of Asio.</p>
</div>
"""


def test_releases_in_page_order_with_versions_and_anchors() -> None:
    releases = parse_history(HISTORY_HTML)
    assert [r.version for r in releases] == [Version(1, 38, 2), Version(1, 0, 0)]
    assert releases[0].anchor == "asio.history.asio_1_38_2"


def test_code_markup_text_is_not_split_by_token_spans() -> None:
    first = parse_history(HISTORY_HTML)[0].entries[0]
    assert first.text == "Fixed ip::tcp handling."
    assert first.children == ()


def test_nested_lists_become_children_and_leave_parent_text() -> None:
    parent = parse_history(HISTORY_HTML)[0].entries[1]
    assert parent.text == "Removed some previously deprecated facilities:"
    assert [c.text for c in parent.children] == ["Removed spawn() overloads.", "Removed io_service."]
    assert parent.full_text().splitlines()[1] == "  - Removed spawn() overloads."


def test_paragraph_only_release_becomes_one_entry() -> None:
    assert [e.text for e in parse_history(HISTORY_HTML)[1].entries] == ["First stable release of Asio."]


def test_page_without_release_headings_is_a_user_error() -> None:
    with pytest.raises(AsioDocsError):
        parse_history("<html><body><h4>Something else</h4></body></html>")


# ---------------------------------------------------------------------------
# canonical markup


CANONICAL_PAGE = (
    '<h4>Asio 1.2.0</h4>\n<div class="itemizedlist"><ul>\n'
    '  <li>a &lt; b<br><a  href="x&amp;y" class=" link  ext " title=\'say "hi"\'>l&amp;m</a>\n'
    "   <pre>  kept\n  </pre>  <p>Para <ul><li>deep</li></ul> after</p>\n"
    '    <div class="itemizedlist"><ul><li>child</li></ul></div>  tail\n  </li>\n'
    "</ul></div>\n<h4>Asio 1.1.0</h4>"
)


def test_markup_is_kept_in_canonical_form() -> None:
    release = parse_history(CANONICAL_PAGE)[0]
    assert release.body_html == (
        '<div class="itemizedlist"><ul>\n<li>a &lt; b<br/><a class="link ext" href="x&amp;y" '
        "title='say \"hi\"'>l&amp;m</a>\n<pre>  kept\n  </pre> <p>Para </p><ul><li>deep</li></ul> after\n"
        '    <div class="itemizedlist"><ul><li>child</li></ul></div>  tail\n  </li>\n</ul></div>'
    )
    entry = release.entries[0]
    # Nested lists (and a wrapping div) leave the entry's own markup; the text around them stays.
    assert entry.html == (
        'a < b<br/><a class="link ext" href="x&amp;y" title=\'say "hi"\'>l&amp;m</a>\n'
        "<pre>  kept\n  </pre> <p>Para </p> after\n      tail"
    )
    assert entry.text == "a < bl&m kept Para after tail"
    assert [c.text for c in entry.children] == ["deep", "child"]


def test_entry_text_with_code_formats_outermost_code_spans_only() -> None:
    html = "Use <code>a<code>b</code></code> and <code> x  y </code><script>no</script>."
    entry = Entry(html=html, text="", children=())
    assert entry_text_with_code(entry, code=lambda text: f"`{text}`") == "Use `ab` and `x y`."


# ---------------------------------------------------------------------------
# parse cache


@pytest.fixture
def cached_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Serves fetch_history() the page in the returned dict's "page" slot, with a
    per-test cache directory."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    served = {"page": HISTORY_HTML.encode()}
    monkeypatch.setattr(history.net, "fetch_cached", lambda url, **kwargs: served["page"])
    return served


def _cache_file() -> Path:
    return history._parse_cache_file(history.history_url(Version(1, 38, 2)))


def _fail_parse(page: str | bytes) -> tuple[Release, ...]:
    raise AssertionError("parsed although a valid cached parse exists")


def test_parse_cache_serves_the_second_fetch(
    cached_page: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = history.fetch_history(Version(1, 38, 2))
    assert first == parse_history(HISTORY_HTML)
    assert _cache_file().is_file()
    monkeypatch.setattr(history, "parse_history", _fail_parse)
    assert history.fetch_history(Version(1, 38, 2)) == first


def test_parse_cache_is_not_served_for_a_changed_page(cached_page: dict[str, bytes]) -> None:
    history.fetch_history(Version(1, 38, 2))
    changed = HISTORY_HTML.replace("First stable release of Asio.", "Changed text.")
    cached_page["page"] = changed.encode()
    assert history.fetch_history(Version(1, 38, 2)) == parse_history(changed)


def test_parse_cache_is_not_served_for_a_changed_parser(
    cached_page: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    history.fetch_history(Version(1, 38, 2))
    monkeypatch.setattr(history, "_PARSE_CACHE_VERSION", "a-newer-parser")
    parsed: list[str | bytes] = []
    real_parse = history.parse_history
    monkeypatch.setattr(history, "parse_history", lambda page: (parsed.append(page), real_parse(page))[1])
    history.fetch_history(Version(1, 38, 2))
    assert len(parsed) == 1


@pytest.mark.parametrize(
    "damage",
    [
        lambda data: data[: len(data) // 2],  # truncated
        lambda data: data.replace(b"First stable", b"First stAble"),  # altered content, digest intact
        lambda data: b"not a cache file",
        lambda data: b"",
    ],
)
def test_damaged_parse_cache_is_a_miss_and_is_rewritten(
    cached_page: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, damage: Callable[[bytes], bytes]
) -> None:
    expected = history.fetch_history(Version(1, 38, 2))
    _cache_file().write_bytes(damage(_cache_file().read_bytes()))
    assert history.fetch_history(Version(1, 38, 2)) == expected
    monkeypatch.setattr(history, "parse_history", _fail_parse)
    assert history.fetch_history(Version(1, 38, 2)) == expected


def test_unwritable_parse_cache_warns_and_still_returns_the_history(
    cached_page: dict[str, bytes], capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = _cache_file().parent
    blocker.parent.mkdir(parents=True)
    blocker.write_text("a file where the cache directory should be")
    assert history.fetch_history(Version(1, 38, 2)) == parse_history(HISTORY_HTML)
    assert "could not cache the parsed revision history" in capsys.readouterr().err
