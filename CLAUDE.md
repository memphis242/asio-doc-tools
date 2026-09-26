# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

`asio-docs`: personal tooling for the standalone Asio C++ library.

- `asio-docs man ...` turns a release's HTML documentation into local man pages
  (`man asio`, `man asio.build`, `man asio.tutorial`, `man asio.<entity>`,
  `man asio.<version>` for release notes) installed under `~/.local/share/man`.
- `asio-docs diff FROM TO` summarizes what was fixed, added, changed, deprecated,
  and breaking between two releases, working directly from the online docs at
  think-async.com and classifying release-note entries with the Claude API.

## Git and workflow policy (overrides the global concurrent-session rules)

Only one development session works on this project at a time, so the global
worktree/branching policy for concurrent sessions does not apply here:

- Work directly in the root checkout (`~/projects/asio-doc-tools`) on `main`.
  No worktrees, no feature branches.
- Commit directly to `main`, and push `main` once a remote exists; completed
  work is committed (and pushed) unprompted.
