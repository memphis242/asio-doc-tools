"""The unit exchanged between the man page generator and the installer."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Section(StrEnum):
    # Library reference, suffixed like OpenSSL's 3ossl: `man 3asio <name>` works,
    # and the pages live in man3/ next to other library docs.
    REFERENCE = "3asio"
    # Overviews, tutorial, build notes, examples index, release notes.
    TOPIC = "7"


_FORBIDDEN_NAME_CHARS: Final = frozenset("/\0\n\t ")


@dataclass(frozen=True, slots=True)
class ManPage:
    name: str  # e.g. asio.basic_stream_socket.async_connect, asio.overview.core.strands, asio.1.38.2
    section: Section
    source: str  # complete roff (man macros) document, starting with .TH

    def __post_init__(self) -> None:
        # Names are sanitized by the generator before a ManPage is built.
        assert self.name.startswith("asio") and not _FORBIDDEN_NAME_CHARS & set(self.name), self.name
        assert self.source.lstrip().startswith(('.TH', "'\\\"")), f"{self.name}: source must start with .TH"

    @property
    def filename(self) -> str:
        return f"{self.name}.{self.section}"

    @property
    def subdir(self) -> str:
        return f"man{self.section.value[0]}"
