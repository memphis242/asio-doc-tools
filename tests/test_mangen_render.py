from asio_doc_tools.manpage import Section
from asio_doc_tools.mangen import ir
from asio_doc_tools.mangen.assemble import first_sentence, is_link_only, strip_more_markers
from asio_doc_tools.mangen.parse import parse_fragment
from asio_doc_tools.mangen.render import PageRef, Renderer

_PAGES = {
    "asio/reference/io_context.html": PageRef("asio.io_context", Section.REFERENCE),
    "asio/overview/core/strands.html": PageRef("asio.overview.core.strands", Section.TOPIC),
}


def _renderer(self_name: str = "asio.self") -> Renderer:
    return Renderer(lambda path, fragment: _PAGES.get(path), frozenset({self_name}), lambda path: "https://x/" + path)


def _render(html: str, renderer: Renderer | None = None) -> list[str]:
    renderer = renderer or _renderer()
    renderer.blocks(parse_fragment(html, "asio/reference/page.html"))
    return renderer.take()


def test_prose_whitespace_is_collapsed_and_lines_are_safe() -> None:
    lines = _render("<p>\n   Call\n      <code>run()</code>  then\n <em>wait</em>.\n</p><p>.hidden</p>")
    assert lines[0] == r"Call \fBrun()\fR then \fIwait\fR."
    assert all(not line.startswith(" ") for line in lines)
    assert r"\&.hidden" in lines


def test_code_blocks_keep_indentation_and_drop_trailing_blanks() -> None:
    lines = _render('<pre class="programlisting">\nint main()\n{\n  return 0;   \n}\n\n</pre>')
    body = lines[lines.index(".EX") + 1 : lines.index(".EE")]
    assert body == ["int main()", "{", "  return 0;", "}"]


def test_internal_links_become_see_also_entries() -> None:
    renderer = _renderer()
    _render(
        '<p>See <a href="io_context.html">io_context</a>, <a href="page.html#frag">this page</a>,'
        ' <a href="../net_ts.html">left out</a> and <a href="https://example.com">x</a>.</p>',
        renderer,
    )
    assert renderer.see_also() == (PageRef("asio.io_context", Section.REFERENCE),)


def test_self_links_are_not_listed() -> None:
    renderer = _renderer(self_name="asio.io_context")
    _render('<p><a href="io_context.html">io_context</a></p>', renderer)
    assert renderer.see_also() == ()


def test_list_items_starting_with_a_link_name_their_page() -> None:
    renderer = _renderer()
    lines = _render(
        '<ul><li><a href="../overview/core/strands.html">Strands</a></li>'
        '<li>Plain text mentioning <a href="io_context.html">io_context</a></li></ul>',
        renderer,
    )
    text = "\n".join(lines)
    assert "asio.overview.core.strands" in text
    assert "asio.io_context" not in text  # a link inside the text stays plain text
    assert [ref.name for ref in renderer.see_also()] == ["asio.io_context"]  # named inline: not repeated


def test_external_links_show_their_url() -> None:
    lines = _render('<p>See <a href="https://example.com/x">the site</a>.</p>')
    assert "<https://example.com/\\:x>" in lines[0]


def test_member_tables_name_each_members_page() -> None:
    html = (
        '<div class="informaltable"><table class="table"><thead><tr><th><p>Name</p></th>'
        "<th><p>Description</p></th></tr></thead><tbody><tr><td><p>"
        '<a class="link" href="io_context.html"><span class="bold"><strong>io_context</strong></span></a>'
        ' <span class="silver">[constructor]</span></p></td><td><p>First. <br> <span class="silver"> \u2014</span>'
        "<br> Second.</p></td></tr></tbody></table></div>"
    )
    text = "\n".join(_render(html))
    assert ".TP" in text and "asio.io_context" in text and "[constructor]" in text
    assert "\u2014" not in text and "First." in text and "Second." in text


def test_admonitions_get_a_label() -> None:
    html = (
        '<div class="note"><table border="0" summary="Note"><tr><td rowspan="2"><img alt="[Note]" src="note.png"></td>'
        "<th>Note</th></tr><tr><td><p>Experimental.</p></td></tr></table></div>"
    )
    text = "\n".join(_render(html))
    assert r"\fBNote:\fR Experimental." in text


def test_figures_link_to_the_image() -> None:
    text = "\n".join(_render('<p><span class="inlinemediaobject"><img src="../../proactor.png"></span></p>'))
    assert "[figure: proactor.png]" in text and "https://x/" in text


def test_strip_more_markers() -> None:
    block = ir.CodeBlock(
        (
            ir.Text("void f();\n  "),
            ir.Span(ir.Style.ITALIC, (ir.Text("\u00bb "), ir.Link("a/overload1.html", "", (ir.Text("more..."),)))),
            ir.Text("\n"),
        )
    )
    assert ir.plain_text(strip_more_markers(block).inlines) == "void f();"


def test_first_sentence() -> None:
    assert first_sentence("Start an asynchronous connect. It returns immediately.") == "Start an asynchronous connect."
    assert first_sentence("Use e.g. a strand. More.") == "Use e.g. a strand."
    assert first_sentence("No period at all") == "No period at all"


def test_link_only_detection() -> None:
    link_only = parse_fragment(
        '<p><a href="x.html">A</a>, <a href="y.html">B</a> and <a href="z.html">C</a>.</p>', "p.html"
    )
    with_text = parse_fragment('<p>Read <a href="x.html">A</a> before calling this.</p>', "p.html")
    assert is_link_only(link_only)
    assert not is_link_only(with_text)
