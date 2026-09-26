import pytest

from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.history import parse_history
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
