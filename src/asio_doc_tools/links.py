"""Recognizing an href that names another host without its scheme.

A handful of pages in Asio's docs (e.g. asio/history.html, citing standards
papers) write an external link as a bare "www.example.org/path" href with no
"http://" in front of it, sometimes with a leading "../" navigation prefix
(e.g. "../www.open-std.org/jtc1/.../n4370.html") left over from how the page
was generated. Nothing that actually points into the doc tree looks like that:
no directory anywhere in the tree has a dot in its name, so once any leading
"." and ".." navigation segments are skipped, a segment that contains a dot
and is followed by more path can only be a host name missing its scheme, never
a real directory. Both the doc tree crawler (docsource.py, so it does not try
to fetch the host as one of its own pages) and the man page generator
(mangen/parse.py, so the link renders as an external one instead of silently
resolving to a bogus in-tree path) rely on this same rule.
"""

_NAVIGATION_SEGMENTS = (".", "..")


def _skip_navigation_prefix(href: str) -> list[str]:
    segments = href.split("/")
    start = 0
    while start < len(segments) and segments[start] in _NAVIGATION_SEGMENTS:
        start += 1
    return segments[start:]


def is_bare_external_host(href: str) -> bool:
    """True if `href` is a same-site-looking relative link that is really an
    external host with its scheme missing (see the module docstring)."""
    remaining = _skip_navigation_prefix(href)
    if len(remaining) < 2:
        return False  # nothing follows the navigation prefix, or it's the last segment
    host_candidate = remaining[0]
    return bool(host_candidate) and not host_candidate.startswith(".") and "." in host_candidate


def external_url(href: str) -> str:
    """The "http://" URL a bare external host href (see `is_bare_external_host`) names.

    Any leading "." / ".." navigation segments are dropped first: they are an
    artifact of how the source page embedded the link, not part of the host or
    path being linked to.
    """
    return "http://" + "/".join(_skip_navigation_prefix(href))
