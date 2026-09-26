import pytest

from asio_doc_tools.mangen.naming import path_name, reference_name, split_qualified
from asio_doc_tools.mangen.pages import topic_name


@pytest.mark.parametrize(
    ("title", "name"),
    [
        ("basic_stream_socket::assign", "asio.basic_stream_socket.assign"),
        ("ip::tcp::socket", "asio.ip.tcp.socket"),
        ("async_read", "asio.async_read"),
        ("basic_stream_socket::rebind_executor", "asio.basic_stream_socket.rebind_executor"),
        ("basic_stream_socket::~basic_stream_socket", "asio.basic_stream_socket.~basic_stream_socket"),
        ("AcceptHandler", "asio.AcceptHandler"),
        # Operators keep their symbols.
        ("ip::address::operator==", "asio.ip.address.operator=="),
        ("ip::address::operator<", "asio.ip.address.operator<"),
        ("ip::address::operator<<", "asio.ip.address.operator<<"),
        ("any_completion_handler::operator()", "asio.any_completion_handler.operator()"),
        ("x::operator[]", "asio.x.operator[]"),
        ("x::operator->", "asio.x.operator->"),
        ("x::operator--", "asio.x.operator--"),
        ("experimental::awaitable_operators::operator&&", "asio.experimental.awaitable_operators.operator&&"),
        # Whitespace: dropped, or `_` between identifier characters.
        ('buffer_literals::operator"" _buf', 'asio.buffer_literals.operator""_buf'),
        ("x::operator bool", "asio.x.operator_bool"),
        ("x::operator co_await", "asio.x.operator_co_await"),
        # Template specializations: commas become +, :: inside the arguments becomes a dot.
        (
            "associated_allocator< reference_wrapper< T >, Allocator >::get",
            "asio.associated_allocator<reference_wrapper<T>+Allocator>.get",
        ),
        (
            "async_result< std::packaged_task< Result(Args...)>, Signature >",
            "asio.async_result<std.packaged_task<Result(Args...)>+Signature>",
        ),
        ("experimental::promise_value_type<>", "asio.experimental.promise_value_type<>"),
        ("  ip::tcp::\n   socket ", "asio.ip.tcp.socket"),
    ],
)
def test_reference_name(title: str, name: str) -> None:
    assert reference_name(title) == name


@pytest.mark.parametrize(
    "title",
    [
        "Accept handler requirements",
        "Requirements on asynchronous operations",
        "ip::address >",  # DocBook's mangled title of a std::hash specialization
        "basic_stream_socket::",
        "",
        "x::operator",
    ],
)
def test_non_names_are_rejected(title: str) -> None:
    assert reference_name(title) is None


def test_names_never_contain_characters_man_db_or_whatis_cannot_handle() -> None:
    for title in ("a< b, c >::d", "x::operator bool", 'y::operator"" _z', "t< std::a, std::b >"):
        name = reference_name(title)
        assert name is not None
        assert not set(name) & set(" \t\n/,\0"), name


def test_split_qualified_keeps_template_arguments_together() -> None:
    assert split_qualified("a< b::c, d >::e") == ("a< b::c, d >", "e")
    assert split_qualified("x::operator()") == ("x", "operator()")


def test_path_name_percent_encodes_unsafe_characters() -> None:
    assert path_name("overview", "a/b") == "asio.overview.a%2Fb"
    assert path_name("na\u00efve") == "asio.na%C3%AFve"


@pytest.mark.parametrize(
    ("rel", "name"),
    [
        ("asio/overview.html", "asio.overview"),
        ("asio/overview/core.html", "asio.overview.core"),
        ("asio/overview/core/strands.html", "asio.overview.core.strands"),
        ("asio/tutorial.html", "asio.tutorial"),
        ("asio/tutorial/tuttimer1.html", "asio.tutorial.timer1"),
        ("asio/tutorial/tutdaytime7.html", "asio.tutorial.daytime7"),
        ("asio/tutorial/boost_bind.html", "asio.tutorial.boost_bind"),
        ("asio/examples.html", "asio.examples"),
        ("asio/examples/cpp20_examples.html", "asio.examples.cpp20"),
        ("asio/using.html", "asio.build"),
        ("asio/history.html", "asio.history"),
        ("asio/reference.html", "asio.reference"),
    ],
)
def test_topic_name(rel: str, name: str) -> None:
    assert topic_name(rel) == name
