import pytest

from asio_doc_tools.diag import AsioDocsError
from asio_doc_tools.versions import Version, resolve


@pytest.mark.parametrize("text", ["1.38.2", "1-38-2", "asio-1-38-2", "asio-1.38.2", "v1.38.2", " 1.38.2 "])
def test_parse_accepts_common_spellings(text: str) -> None:
    assert Version.parse(text) == Version(1, 38, 2)


@pytest.mark.parametrize("text", ["", "1.38", "1.38.2.1", "boost-1.89", "latest", "1.x.2"])
def test_parse_rejects_non_versions(text: str) -> None:
    with pytest.raises(ValueError):
        Version.parse(text)


def test_ordering_is_numeric_not_lexicographic() -> None:
    assert Version.parse("1.4.0") < Version.parse("1.10.0") < Version.parse("1.38.2")
    assert str(Version(1, 38, 2)) == "1.38.2" and Version(1, 38, 2).dashed == "1-38-2"


def test_resolve_reports_bad_input_as_user_error() -> None:
    with pytest.raises(AsioDocsError):
        resolve("one.two.three")
