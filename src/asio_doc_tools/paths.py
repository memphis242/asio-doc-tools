"""Filesystem locations, following the XDG base directory spec.

- cache_dir(): cheap to recreate (HTTP responses, release tarballs, extracted docs).
- data_dir(): expensive or impossible to recreate (LLM classifications, install manifest).
- default_man_dir(): user man tree; man-db puts it on the manpath automatically
  when ~/.local/bin is on PATH.
"""

import os
from pathlib import Path
from typing import Final

APP_NAME: Final = "asio-doc-tools"


def _xdg_base(env_var: str, default_relative_to_home: str) -> Path:
    # The spec says relative values are invalid and must be ignored.
    value = os.environ.get(env_var, "")
    if value and Path(value).is_absolute():
        return Path(value)
    return Path.home() / default_relative_to_home


def cache_dir() -> Path:
    return _xdg_base("XDG_CACHE_HOME", ".cache") / APP_NAME


def data_dir() -> Path:
    return _xdg_base("XDG_DATA_HOME", ".local/share") / APP_NAME


def default_man_dir() -> Path:
    return _xdg_base("XDG_DATA_HOME", ".local/share") / "man"
