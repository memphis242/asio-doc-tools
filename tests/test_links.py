import pytest

from asio_doc_tools.links import external_url, is_bare_external_host


@pytest.mark.parametrize(
    "href",
    [
        "www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4370.html",
        # The real asio/history.html writes some of these with a leading "../"
        # left over from how the page was generated.
        "../www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/n4656.html",
        "../../www.open-std.org/path/to/paper.html",
        "./www.open-std.org/path/to/paper.html",
    ],
)
def test_recognizes_bare_external_hosts(href: str) -> None:
    assert is_bare_external_host(href)


@pytest.mark.parametrize(
    "href",
    [
        "overview.html",
        "asio/reference.html",
        "asio/reference/basic_stream_socket.html",
        "../overview.html",
        "../../asio/reference.html",
        "./sibling.html",
        "",
        "reference.html#fragment",
    ],
)
def test_does_not_flag_ordinary_in_tree_links(href: str) -> None:
    assert not is_bare_external_host(href)


def test_external_url_drops_the_navigation_prefix() -> None:
    assert external_url("www.open-std.org/jtc1/n4370.html") == "http://www.open-std.org/jtc1/n4370.html"
    assert external_url("../www.open-std.org/jtc1/n4370.html") == "http://www.open-std.org/jtc1/n4370.html"
    assert external_url("../../www.open-std.org/path.html") == "http://www.open-std.org/path.html"
