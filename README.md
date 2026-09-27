# asio-doc-tools

Personal tooling for the standalone [Asio](https://think-async.com/Asio/) C++ library:

- **Man pages.** Turns a release's HTML documentation into local man pages, so
  `man asio.async_read` works like any system library.
- **Release-notes diff.** Summarizes what was fixed, added, changed,
  deprecated, and breaking between any two Asio releases.

## Install

Requires Python 3.12+ with `beautifulsoup4`, `lxml`, and `anthropic`, plus
man-db (`man`, `mandb`).

```sh
git clone git@github.com:memphis242/asio-doc-tools.git ~/projects/asio-doc-tools
ln -s ~/projects/asio-doc-tools/bin/asio-docs ~/.local/bin/asio-docs
asio-docs man install            # latest release; or e.g. `asio-docs man install 1.38.2`
```

`bin/asio-docs` runs the package from the checkout it lives in, so the symlink
always tracks the working tree.

## Man pages

| Command | Content |
|---|---|
| `man asio` | Landing page: overview, how the pages are organized, full table of contents |
| `man asio.build` | Using, building, and configuring Asio |
| `man asio.overview`, `man asio.overview.core.strands`, ... | The overview, one page per topic |
| `man asio.tutorial`, `man asio.tutorial.timer1`, ... | The tutorial, one page per step (with its full source listing) |
| `man asio.examples`, `man asio.examples.cpp20`, ... | The examples index |
| `man asio.reference` | Index of every reference page |
| `man asio.<entity>` | Reference: `asio.io_context`, `asio.ip.tcp.socket`, `asio.basic_stream_socket.async_connect`, ... (`::` becomes `.`; all overloads of a function share one page) |
| `man asio.history` | List of releases |
| `man asio.<version>` | One release's notes, e.g. `man asio.1.38.2` |

Reference pages are in section `3asio` (`man 3asio asio.async_read`), everything
else in section `7`. `man -k asio` / `apropos asio` search the one-line
summaries, and shell completion works on page names (`man asio.ip.<TAB>`).
The "Networking TS Compatibility" and "Proposed Standard Executors" sections
are intentionally left out.

```sh
asio-docs man install [VERSION] [--man-dir DIR] [--force]   # build + install (+ mandb)
asio-docs man build [VERSION] --out DIR [--no-compress]     # build only, e.g. to inspect
asio-docs man status                                        # what is installed
asio-docs man uninstall                                     # remove exactly what was installed
```

Docs come from the release tarball on SourceForge, falling back to crawling
think-async.com if the tarball is unavailable (`--source auto|tarball|online`).
Pages install into `~/.local/share/man` by default, which man-db already
searches when `~/.local/bin` is on `PATH`. A manifest records what was
installed, so upgrades remove stale pages and never touch files the tool did
not write.

## Release-notes diff

```sh
asio-docs diff 1.30.2 1.38.2                   # grouped: breaking, deprecated, added, changed, fixed, other
asio-docs diff 1.30.2 latest --format markdown # or --format json
asio-docs releases                             # all releases, entry counts, classification coverage
asio-docs diff 1.30.2 1.38.2 --engine threads  # the threaded reference engine (same results)
```

The diff covers every release after the older version up to and including the
newer one: what changes when upgrading. It reads the revision history directly
from think-async.com.

Upstream release notes are unlabeled free-form bullets, so each bullet is
classified by Claude (Claude Sonnet 5, low effort) through the Anthropic API.
Credentials resolve like any Anthropic SDK client: `ANTHROPIC_API_KEY`, or an
`ant auth login` profile. Every answer is stored permanently per bullet, so a
bullet is only ever sent once: repeated diffs are instant, free, and stable.
Stored answers are keyed by prompt version, model, and effort, so a change to
any of them re-classifies automatically.

Bullets go 40 to a request, with up to 32 requests in flight at once, so even
the full history takes one round of requests.

Spending is hard-capped per run. Before its first request, a run prints how many
requests it expects, the expected cost, and its hard caps on requests, output
and input tokens, and time, with the most it can possibly cost. It prints actual
usage when it finishes, even with `-q`. For the full history (about 1,000
bullets) the expected cost is about $0.55 and the worst case $5.09; a run whose
worst case would pass $6.00 is refused before anything is sent (diff a smaller
range first: its answers carry over). Nothing is sent either when the
classification store cannot be written. When a cap is hit, the run stops and
keeps everything classified so far. Rerunning sends only what is still missing.
The first Ctrl-C waits for in-flight requests and stores their answers; a second
gives up on them at once, and so does a run still going after about 9 minutes.

Requests run on one asyncio event loop by default. `--engine threads` runs them on a
thread pool instead, through a reference implementation kept to compare the two
(see [its README](src/asio_doc_tools/classify/threaded_reference/README.md)). Both
engines give the same results and store them in the same place, and they have the
same costs and limits. asyncio is the default for its cleaner cancellation and
deadlines, not for speed: at this scale the two are equally fast
([bench/RESULTS.md](bench/RESULTS.md),
[bench/THREADS-VS-EVENT-LOOP.md](bench/THREADS-VS-EVENT-LOOP.md)).

## Files

| Location | Content |
|---|---|
| `~/.cache/asio-doc-tools/` | HTTP cache, release tarballs, extracted doc trees, parsed revision history (safe to delete) |
| `~/.local/share/asio-doc-tools/classifications.sqlite3` | Stored classifications, one row per bullet with its text (paid for; keep it). Query it with `sqlite3` |
| `~/.local/share/asio-doc-tools/classifications.pending/` | Answers a run paid for but could not write to the store (a full disk, say, or a run stopped by its time limit), one file per save, one JSON object per line; the next run moves them into the store. Normally empty or absent |
| `~/.local/share/asio-doc-tools/installed-man-pages.json` | Install manifest |

XDG base directory variables (`XDG_CACHE_HOME`, `XDG_DATA_HOME`) are honored.

## Development

```sh
python3 -m pytest
```
