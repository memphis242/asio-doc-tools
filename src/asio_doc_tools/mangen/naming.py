"""Man page names: derived from doc paths (topics) and qualified C++ names (reference).

Reference names follow the C++ name as the docs title it, so a user can guess
them: `basic_stream_socket::assign` is `asio.basic_stream_socket.assign`. The
character rules, applied in order, keep every name a valid, shell-friendly-ish
file name that man-db's whatis parser reads back as exactly one name:

1. `::` becomes `.`.
2. Whitespace between two identifier characters becomes `_`
   (`operator bool` -> `operator_bool`); all other whitespace is dropped
   (`operator"" _buf` -> `operator""_buf`, `foo< T >` -> `foo<T>`).
3. `,` becomes `+` (a comma would split the name in whatis/apropos):
   `associated_allocator< reference_wrapper< T >, Allocator >` ->
   `associated_allocator<reference_wrapper<T>+Allocator>`.
4. `/`, `%`, `\\`, control and non-ASCII characters are percent-encoded (`%2F`).

Everything else (operator symbols, `~` of destructors, template brackets) is
kept as is. RULES_SUMMARY states the same for the landing page.
"""

import re
from typing import Final

PREFIX: Final = "asio"

RULES_SUMMARY: Final = (
    ("::", "becomes a dot: basic_stream_socket::assign is asio.basic_stream_socket.assign"),
    (
        "whitespace",
        "between two identifier characters becomes _ (operator bool is operator_bool); elsewhere it is "
        'dropped (operator"" _buf is operator""_buf, foo< T > is foo<T>)',
    ),
    (
        ",",
        "becomes + (associated_allocator< reference_wrapper< T >, Allocator > is "
        "asio.associated_allocator<reference_wrapper<T>+Allocator>)",
    ),
    ("/ % \\", "and control or non-ASCII characters are percent-encoded (/ is %2F)"),
)

_IDENT_RE: Final = re.compile(r"[A-Za-z_]\w*")
_OPERATOR_SYMBOLS: Final = (
    "()", "[]", "->*", "->", "<<=", ">>=", "<=>", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
    "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "+", "-", "*", "/", "%", "^", "&", "|", "~", "!", "=", "<", ">", ",",
)  # fmt: skip
_PERCENT_ENCODED: Final = frozenset("/%\\")


class _Scanner:
    """Splits a qualified C++ name into its `::`-separated components."""

    def __init__(self, text: str) -> None:
        self._text: Final = text
        self._pos = 0

    def _peek(self, token: str) -> bool:
        return self._text.startswith(token, self._pos)

    def _skip_spaces(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos] == " ":
            self._pos += 1

    def _template_args(self) -> bool:
        """Consume a balanced <...> (brackets and parentheses nest inside)."""
        assert self._text[self._pos] == "<"
        depth = 0
        while self._pos < len(self._text):
            char = self._text[self._pos]
            if char in "<(":
                depth += 1
            elif char in ">)":
                depth -= 1
                if depth < 0:
                    return False
                if depth == 0:
                    self._pos += 1
                    return True
            self._pos += 1
        return False

    def _operator(self) -> bool:
        """Consume the part after the `operator` keyword."""
        if self._peek('""'):
            self._pos += 2
            self._skip_spaces()
            match = _IDENT_RE.match(self._text, self._pos)
            if match is None:
                return False
            self._pos = match.end()
            return True
        for symbol in _OPERATOR_SYMBOLS:
            if self._peek(symbol):
                self._pos += len(symbol)
                return True
        if not self._peek(" "):
            return False
        self._skip_spaces()  # conversion operator or `operator co_await`
        match = _IDENT_RE.match(self._text, self._pos)
        if match is None:
            return False
        self._pos = match.end()
        if self._pos < len(self._text) and self._text[self._pos] == "<":
            return self._template_args()
        return True

    def _component(self) -> bool:
        if self._peek("~"):
            self._pos += 1
        match = _IDENT_RE.match(self._text, self._pos)
        if match is None:
            return False
        self._pos = match.end()
        if match.group() == "operator":
            return self._operator()
        self._skip_spaces()
        if self._pos < len(self._text) and self._text[self._pos] == "<":
            return self._template_args()
        return True

    def components(self) -> tuple[str, ...] | None:
        parts: list[str] = []
        while True:
            start = self._pos
            if not self._component():
                return None
            parts.append(self._text[start : self._pos].strip())
            self._skip_spaces()
            if self._pos == len(self._text):
                return tuple(parts)
            if not self._peek("::"):
                return None
            self._pos += 2
            self._skip_spaces()


def split_qualified(title: str) -> tuple[str, ...] | None:
    """The `::`-separated components of a C++ qualified name, or None if `title` is not one."""
    text = " ".join(title.split())
    return _Scanner(text).components() if text else None


def _sanitize(text: str) -> str:
    text = " ".join(text.split()).replace("::", ".")
    out: list[str] = []
    for i, char in enumerate(text):
        if char == " ":
            before, after = text[i - 1 : i], text[i + 1 : i + 2]
            if (before.isalnum() or before == "_") and (after.isalnum() or after == "_"):
                out.append("_")
        elif char == ",":
            out.append("+")
        elif char in _PERCENT_ENCODED or not char.isascii() or not char.isprintable():
            out.extend(f"%{byte:02X}" for byte in char.encode())
        else:
            out.append(char)
    return "".join(out)


def reference_name(qualified: str) -> str | None:
    """`asio.`-prefixed man page name for a qualified C++ name, or None if it is not one."""
    components = split_qualified(qualified)
    if components is None:
        return None
    return ".".join((PREFIX, *(_sanitize(c) for c in components)))


def path_name(*components: str) -> str:
    """A name built from doc path components (topic pages, requirement pages)."""
    assert components and all(components), components
    return ".".join((PREFIX, *(_sanitize(c) for c in components)))


def release_name(version: str) -> str:
    return f"{PREFIX}.{version}"
