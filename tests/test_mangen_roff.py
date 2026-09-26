import shutil
import subprocess

import pytest

from asio_doc_tools.mangen import roff


def test_backslash_is_escaped() -> None:
    assert roff.escape(r"c:\windows") == r"c:\ewindows"
    assert roff.escape(r'"\n"', code=True) == r'"\en"'


def test_code_is_escaped_for_copy_paste() -> None:
    assert roff.escape("a->b - 'c' `d` ~e ^f", code=True) == r"a\->b \- \(aqc\(aq \(gad\(ga \(tie \(haf"


def test_prose_keeps_hyphens_but_not_lone_dashes() -> None:
    assert roff.escape("well-known") == "well-known"
    assert roff.escape("a - b") == r"a \- b"
    assert roff.escape("- leading") == r"\- leading"


def test_dashes_and_non_ascii() -> None:
    escaped = roff.escape("x \u2014 y\u2013z \u00a9 \u00a0 \u2019 \u03bb")
    assert escaped == r"x \- y\-z \(co \~ ' \[u03BB]"
    assert escaped.isascii()


def test_line_start_protection() -> None:
    assert roff.protect_line_start(".foo") == r"\&.foo"
    assert roff.protect_line_start("'foo") == r"\&'foo"
    assert roff.protect_line_start("foo.") == "foo."


def test_quote_arg() -> None:
    assert roff.quote_arg('say "hi"') == r'"say \(dqhi\(dq"'


def test_names_hide_hyphens_from_the_whatis_parser() -> None:
    escaped = roff.escape_name("asio.x.operator--")
    assert "--" not in escaped and r"\- " not in escaped
    assert escaped.replace(r"\-\&", "-") == "asio.x.operator--"


def test_break_points() -> None:
    assert r"https:/\:/" not in roff.url("https://example.com/a/b")
    assert roff.url("https://example.com/a/b").count(r"\:") == 2
    short = "asio.basic_stream_socket.assign"
    assert roff.breakable_name(short) == short
    assert r"\:" in roff.breakable_name("asio." + "x" * 70 + ".y")
    assert r"\:" not in roff.escape_wrapping("a/b/c")
    assert r"/\:" in roff.escape_wrapping("../" + "d/" * 40)


@pytest.mark.skipif(shutil.which("groff") is None, reason="groff not installed")
def test_escaped_code_renders_back_verbatim() -> None:
    code = r"""if (a->b != 'c' && s == "x\n") { ~T(); x ^= y; } // `q` -1 --i"""
    source = "\n".join(
        [
            ".TH T 7",
            ".SH NAME",
            r"t \- test",
            ".SH CODE",
            ".EX",
            roff.protect_line_start(roff.escape(code, code=True)),
            ".EE",
            "",
        ]
    )
    result = subprocess.run(
        ["groff", "-man", "-Tutf8", "-P-cbou", "-rLL=200n"], input=source, capture_output=True, text=True, check=True
    )
    assert code in [line.strip() for line in result.stdout.splitlines()]
