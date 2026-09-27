import inspect
import json
import time

import pytest
from classify_support import Echo, scripted_client

from asio_doc_tools import classify, cli, notesdiff
from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.history import Entry, Release
from asio_doc_tools.versions import Version

V = Version.parse


def _entry(text: str, children: tuple[Entry, ...] = ()) -> Entry:
    return Entry(html=f"<p>{text}</p>", text=text, children=children)


def _release(version: str, *texts: str) -> Release:
    return Release(
        version=V(version), anchor="", body_html="", entries=tuple(_entry(t) for t in texts)
    )


HISTORY = (
    _release("1.38.2", "Fixed a crash.", "Added a new overload."),
    _release("1.38.1", "Changed a default."),
    _release("1.38.0", "Removed io_service."),
    _release("1.30.2", "First entry of the older baseline."),
)


def _classification(
    category: classify.Category = classify.Category.FIXED, breaking: bool = False, reason: str = ""
) -> classify.Classification:
    return classify.Classification(
        category=category,
        breaking=breaking,
        breaking_reason=reason,
        model=classify.MODEL,
        effort=classify.EFFORT,
        release="1.38.2",
        classified_at=time.time(),
    )


def _fake_fetch_history(monkeypatch, releases=HISTORY) -> None:
    monkeypatch.setattr(notesdiff.history, "fetch_history", lambda refresh=False: releases)


# ---------------------------------------------------------------------------
# range selection
# ---------------------------------------------------------------------------


def test_range_selection_is_ascending_and_exclusive_of_the_older_endpoint(monkeypatch) -> None:
    _fake_fetch_history(monkeypatch)
    older, newer, selected = notesdiff.resolve_range("1.38.0", "1.38.2")
    assert (older, newer) == (V("1.38.0"), V("1.38.2"))
    assert [r.version for r in selected] == [V("1.38.1"), V("1.38.2")]


def test_range_selection_swaps_when_given_newest_first(monkeypatch, capsys) -> None:
    _fake_fetch_history(monkeypatch)
    older, newer, selected = notesdiff.resolve_range("1.38.2", "1.38.0")
    assert (older, newer) == (V("1.38.0"), V("1.38.2"))
    assert [r.version for r in selected] == [V("1.38.1"), V("1.38.2")]
    assert "newest-first" in capsys.readouterr().err


def test_equal_versions_is_an_error(monkeypatch) -> None:
    _fake_fetch_history(monkeypatch)
    with pytest.raises(AsioDocsError):
        notesdiff.resolve_range("1.38.2", "1.38.2")


def test_unknown_version_names_nearest_known_versions(monkeypatch) -> None:
    _fake_fetch_history(monkeypatch)
    with pytest.raises(AsioDocsError, match=r"1\.30\.2.*1\.38\.0|1\.38\.0.*1\.30\.2"):
        notesdiff.resolve_range("1.38.0", "1.35.0")


def test_reclassify_is_gone() -> None:
    # Stored results are keyed by prompt version, model, and effort, so there is nothing
    # left for a "classify again" option to do but pay again for identical requests.
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["diff", "1.38.0", "1.38.2", "--reclassify"])
    assert "reclassify" not in inspect.signature(notesdiff.build_diff_result).parameters
    assert "reclassify" not in inspect.signature(classify.classify).parameters


@pytest.mark.parametrize("engine", classify.ENGINES)
def test_build_diff_result_classifies_only_entries_not_yet_stored(monkeypatch, tmp_path, engine) -> None:
    _fake_fetch_history(monkeypatch)
    store_path = tmp_path / "classifications.sqlite3"
    client = scripted_client(engine, [Echo(), Echo()])
    first = notesdiff.build_diff_result(
        "1.38.1", "1.38.2", engine=engine, client_factory=lambda: client, store_path=store_path
    )
    assert first.entry_count == 2
    assert len(client.calls) == 1
    wider = notesdiff.build_diff_result(
        "1.38.0", "1.38.2", engine=engine, client_factory=lambda: client, store_path=store_path
    )
    assert wider.entry_count == 3
    assert len(client.calls) == 2
    assert len(json.loads(client.calls[1]["messages"][0]["content"])["items"]) == 1  # only the new entry


def test_the_engine_is_chosen_on_the_command_line_and_defaults_to_asyncio() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["diff", "1.38.0", "1.38.2"]).engine == "asyncio" == classify.DEFAULT_ENGINE
    assert parser.parse_args(["diff", "1.38.0", "1.38.2", "--engine", "threads"]).engine == "threads"
    with pytest.raises(SystemExit):
        parser.parse_args(["diff", "1.38.0", "1.38.2", "--engine", "fibers"])


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def test_grouping_puts_every_breaking_entry_first_regardless_of_category() -> None:
    items = [
        classify.ClassifyItem(release=V("1.0.0"), entry=_entry("a")),
        classify.ClassifyItem(release=V("1.0.0"), entry=_entry("b")),
    ]
    classifications = [
        _classification(category=classify.Category.ADDED, breaking=True, reason="watch out"),
        _classification(category=classify.Category.FIXED, breaking=False),
    ]
    groups = notesdiff.group_entries(items, classifications)
    assert list(groups) == list(notesdiff._GROUP_ORDER)
    assert [de.entry.text for de in groups[notesdiff.Group.BREAKING]] == ["a"]
    assert [de.entry.text for de in groups[notesdiff.Group.FIXED]] == ["b"]


def test_grouping_orders_releases_ascending_preserving_page_order_within_a_release() -> None:
    items = [
        classify.ClassifyItem(release=V("1.2.0"), entry=_entry("newer first")),
        classify.ClassifyItem(release=V("1.2.0"), entry=_entry("newer second")),
        classify.ClassifyItem(release=V("1.1.0"), entry=_entry("older first")),
    ]
    classifications = [_classification() for _ in items]
    groups = notesdiff.group_entries(items, classifications)
    fixed = groups[notesdiff.Group.FIXED]
    assert [de.entry.text for de in fixed] == ["older first", "newer first", "newer second"]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _sample_result() -> notesdiff.DiffResult:
    items = [
        classify.ClassifyItem(
            release=V("1.2.0"),
            entry=_entry("Removed a deprecated header.", (_entry("It used to live in include/old.hpp."),)),
        ),
        classify.ClassifyItem(release=V("1.1.0"), entry=_entry("Fixed a leak.")),
    ]
    classifications = [
        _classification(category=classify.Category.CHANGED, breaking=True, reason="the header is gone"),
        _classification(category=classify.Category.FIXED),
    ]
    groups = notesdiff.group_entries(items, classifications)
    return notesdiff.DiffResult(
        from_version=V("1.1.0"), to_version=V("1.2.0"), releases=(V("1.1.0"), V("1.2.0")), groups=groups
    )


def test_render_text_has_a_header_and_a_why_line_for_breaking_entries() -> None:
    text = notesdiff.render_text(_sample_result(), color="never")
    assert text.startswith("Asio 1.1.0 -> 1.2.0: 2 releases, 2 entries")
    assert "why: the header is gone" in text
    assert "\033[" not in text


def test_render_text_color_always_adds_ansi_codes() -> None:
    text = notesdiff.render_text(_sample_result(), color="always")
    assert "\033[" in text


def test_render_markdown_has_group_sections_and_backtick_code(monkeypatch) -> None:
    items = [classify.ClassifyItem(release=V("1.0.0"), entry=_entry("Renamed foo", ()))]
    item_with_code = classify.ClassifyItem(
        release=V("1.0.0"),
        entry=Entry(html='Added <code>foo::bar()</code>.', text="Added foo::bar().", children=()),
    )
    classifications = [_classification(category=classify.Category.CHANGED)]
    groups = notesdiff.group_entries(items, classifications)
    result = notesdiff.DiffResult(
        from_version=V("1.0.0"), to_version=V("1.1.0"), releases=(V("1.0.0"),), groups=groups
    )
    md = notesdiff.render_markdown(result)
    assert "## Changed (1)" in md

    md_code = notesdiff._entry_markdown_text(item_with_code.entry)
    assert md_code == "Added `foo::bar()`."


def test_render_json_structure_round_trips() -> None:
    result = _sample_result()
    data = notesdiff.render_json(result)
    assert set(data) == {"from", "to", "releases", "entries"}
    assert data["from"] == "1.1.0" and data["to"] == "1.2.0"
    assert data["releases"] == ["1.1.0", "1.2.0"]
    json.dumps(data)  # must be plain-JSON-serializable
    breaking_entry = next(e for e in data["entries"] if e["breaking"])
    assert breaking_entry["breaking_reason"] == "the header is gone"
    assert breaking_entry["children"] == [{"text": "It used to live in include/old.hpp.", "children": []}]
