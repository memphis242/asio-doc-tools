"""roff (man(7)) escaping: turning arbitrary text into safe input for groff and mandoc.

Generated pages are pure ASCII: every non-ASCII character becomes a named glyph
or a \\[uXXXX] escape, so no input-encoding detection (preconv) is involved.
Two flavors of escaping exist:

- prose: hyphens stay hyphens (groff may typeset them as such);
- code: characters whose typographic rendering would break copy-paste are
  forced to their ASCII glyphs (\\- for minus, \\(aq for the apostrophe, ...).
"""

import re
from typing import Final

# Characters that are special to roff or that groff may render typographically.
_COMMON: Final = {
    "\\": r"\e",
    "\u00a0": r"\~",  # no-break space: an adjustable, unbreakable space
    "\u00a9": r"\(co",
    "\u00ab": r"\(Fc",
    "\u00bb": r"\(Fo",
    "\u2013": r"\-",  # en dash
    "\u2014": r"\-",  # em dash (never rendered as an em dash glyph)
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2192": r"\(->",
}
_PROSE: Final = str.maketrans({**_COMMON, "^": r"\(ha", "~": r"\(ti", "`": r"\(ga"})
_CODE: Final = str.maketrans({**_COMMON, "-": r"\-", "'": r"\(aq", "`": r"\(ga", "^": r"\(ha", "~": r"\(ti"})

_NON_ASCII_RE: Final = re.compile(r"[^\x00-\x7f]")
_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# A hyphen standing alone between spaces is a dash, not a hyphen: keep it ASCII.
_LONE_HYPHEN_RE: Final = re.compile(r"(?<= )-(?= |$)|^-(?= )")


def _unicode_escape(match: re.Match[str]) -> str:
    return f"\\[u{ord(match.group()):04X}]"


def escape(text: str, *, code: bool = False) -> str:
    """Escape `text` for use inside a roff text line or a quoted macro argument.

    Does not protect the start of a line; see protect_line_start().
    """
    assert not _CONTROL_RE.search(text), f"control character in {text!r}"
    assert "\n" not in text, "escape() works on single lines"
    if not code:
        text = _LONE_HYPHEN_RE.sub("\u2013", text)  # en dash maps to \- below
    escaped = text.translate(_CODE if code else _PROSE)
    return _NON_ASCII_RE.sub(_unicode_escape, escaped)


def escape_name(name: str) -> str:
    """Escape a man page name (for NAME, SEE ALSO, cross references).

    Each hyphen becomes \\-\\& so that it renders as ASCII and lexgrog/mandb do not
    mistake `--` or `- ` inside a name (operator--, operator-) for the NAME
    section's name/description separator.
    """
    return escape(name, code=True).replace(r"\-", r"\-\&")


# Only tokens that cannot fit on an 80-column line anyway get break points;
# shorter ones move to the next line whole.
_LONG_TOKEN: Final = 60
_LONG_TOKEN_RE: Final = re.compile(rf"(\S{{{_LONG_TOKEN + 1},}})")
# Where a long token may wrap: after a path separator (but not inside "//"), a
# scope operator, a comma, or DocBook's double-underscore file name mangling.
_URL_BREAK_RE: Final = re.compile(r"(?<=[^/]/)(?!/)|(?<=__)(?!_)")
_TEXT_BREAK_RE: Final = re.compile(r"(?<=[^/]/)(?!/)|(?<=::)(?!:)|(?<=,)")
_NAME_BREAK_RE: Final = re.compile(r"(?<=[.<+])(?![.])")


def _join_breakable(token: str, breaks: re.Pattern[str], *, code: bool) -> str:
    return "\\:".join(escape(piece, code=code) for piece in breaks.split(token) if piece)


def escape_wrapping(text: str, *, code: bool = False) -> str:
    """escape(), plus zero-width break points (\\:) inside very long tokens (paths, qualified names)."""
    parts = _LONG_TOKEN_RE.split(text)  # odd indices are the long tokens
    return "".join(
        _join_breakable(part, _TEXT_BREAK_RE, code=code) if i % 2 else escape(part, code=code)
        for i, part in enumerate(parts)
    )


def url(text: str) -> str:
    """A URL, with break points after its slashes so long ones can wrap."""
    return _join_breakable(text, _URL_BREAK_RE, code=True)


def breakable_name(name: str) -> str:
    """A page name in running text; very long ones may wrap after a dot, `<`, or `+`.

    Not for the NAME section: lexgrog keeps break escapes literally.
    """
    if len(name) <= _LONG_TOKEN:
        return escape_name(name)
    return "\\:".join(escape_name(piece) for piece in _NAME_BREAK_RE.split(name) if piece)


def protect_line_start(line: str) -> str:
    """A text line must not start with a control character ('.' or "'")."""
    return "\\&" + line if line.startswith((".", "'")) else line


def quote_arg(text: str) -> str:
    """A double-quoted macro argument (e.g. for .SH, .TH)."""
    return '"' + escape(text).replace('"', r"\(dq") + '"'
