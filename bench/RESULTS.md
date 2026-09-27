# Results: threads vs asyncio for concurrent Claude API requests

Measured 2026-09-26 on a laptop: Intel Core i5-7200U (2 cores, 4 hardware
threads, `nproc` 4), 7.6 GiB RAM, Linux 7.1.5 (Fedora 43). CPython 3.14.6 (the
standard build, GIL enabled), anthropic 0.102.0 on httpx 0.28.1 / httpcore
1.0.9, uvloop 0.22.1. Other desktop work ran at the same time (see
[Noise](#noise-and-outliers)). Local cells ran 3 times each, and the tables show
the median with the range; the live scenarios ran once. Method and definitions
are in [README.md](README.md).

Live spend: **$0.5534** of the $1.00 cap (768 requests, all successful; the
worst-case plan was $0.9779).

## The short version

1. **At the classifier's scale, threads vs asyncio makes no measurable
   difference.** The SDK spends about 5 ms of client CPU per request at 8 in
   flight, against a model latency of about 3 s for 10 entries or about 10 s
   for 40. Live, the project's own request took the same time in both models
   (3.2-3.6 s median at C=4 and C=16).
2. **Throughput stops scaling where the client runs out of one core, and the
   limit is the SDK's HTTP stack, not threads vs the event loop.** Every
   Python-level step of every request runs on one core: under the GIL for
   threads, on the single loop thread for asyncio. Once that core is full,
   throughput ~= CPU share / CPU per request. On this laptop the SDK topped out
   between about 55 and 130 req/s depending on the cell. Threads were behind
   asyncio in the fixed-latency sweeps (at 50 ms: 56-63 vs 84-96 req/s) and
   about level with it in the jittered one. The CPU per
   request itself grows with concurrency, from ~4-5 ms at C<=16 to 10-20 ms at
   C>=64, mostly inside httpcore's connection pool. The same event loop driving
   raw sockets reached 3,000+ req/s at 0.2 ms of CPU per request.
3. **Threads cost more at high concurrency.** At C=256 they used 21 ms of CPU
   per request vs 7 ms for asyncio, with 257 OS threads vs 2, and 2x the memory
   growth. At C=1024 they needed 1025 OS threads, and peak RSS grew by 88 MB
   against 44 MB. At low concurrency (C<=16) the two were even, or threads were
   slightly *better*: mostly less CPU per request and a lower median overhead.
4. **Cancellation is the qualitative difference.** asyncio cancelled 64
   in-flight requests and closed every connection in about 20 ms. With threads,
   a request blocked in a socket read cannot be interrupted by any public API.
   The caller can get control back at once (`shutdown(wait=False)`, or Ctrl-C),
   but the process cannot exit until every in-flight request finishes (10 s
   here), and the server completes, and on the real API bills, all 64 of them.
   Closing the client from another thread does not wake the blocked reads.
   `shutdown(SHUT_RDWR)` on the sockets does, but only by reaching into
   httpcore internals. `os._exit()` does, by skipping all cleanup.
5. **No rate limit came near.** 0 of 768 live requests got a 429. The deepest
   dip in remaining headroom was 99.2% of the per-minute request limit
   (asyncio, 128 in flight), and token headroom never dropped below 99.9%.

## Where throughput stops scaling, and why

With a fixed 1 s simulated latency, the ideal throughput is C req/s (Little's
law). Both SDK modes stay within 96-99% of ideal up to C=16, drop to 67-75% at
C=64, and reach only 12-44% at C=256 and 1024. The raw-socket reference stays at
97-100% through C=256. So the mock server and the event loop mechanism are not
the limit; the SDK client is.

The 50 ms sweep isolates the client cost. From C=64 up, both SDK modes run one
core flat out (97-103% CPU), and their throughput is what that core allows:

| 50 ms latency, C>=64 | CPU per request | CPU in use | predicted: CPU in use / CPU per request | measured |
|---|---|---|---|---|
| threads | 16-18 ms | 100-103% | ~56-63 req/s | 56-63 req/s |
| asyncio | 10-12 ms | 97-98% | ~84-96 req/s | 84-96 req/s |
| asyncio on uvloop | 10-11 ms | 97-98% | ~87-95 req/s | 87-94 req/s |

Past that point, raising C only lengthens a queue inside the client. From C=64
to C=1024 the mock saw at most 17-36 requests in flight at once from the SDK
modes, while client overhead grew to seconds (p50 7.5 s for asyncio, 10.9 s for
threads at C=1024). The same thing happens with 1 s latency. There the SDK
modes never got more than 482 (threads) or 625 (asyncio) of their 1024
requests onto the wire together.

Why the CPU per request grows with concurrency: profiling the async client at
C=64 put about 70% of the loop's time in httpcore's
`_assign_requests_to_connections`. That runs on every request start and every
response close. It scans every pooled connection for every queued request, and
it polls idle sockets (`has_expired` -> `is_socket_readable`). The count was
about 8,600 `is_idle()` calls per request. Its keep-alive cleanup also counts
*all* pooled connections against `max_keepalive_connections` (100 by default),
so once more than 100 are open it closes every connection that goes idle. That
is why the 1 s sweep's TCP-conns column shows nearly one new connection per
request at C>=256 (957-2048 connections for 1024-2048 requests), where C<=64
reuses one connection per slot. The 50 ms sweep churns less because its client
never had many more than 100 connections open. The sync pool has the same
algorithm behind a lock, so both modes pay this. Threads additionally pay for
GIL hand-offs (system CPU is 8-13% of their time vs 4-6% for asyncio) and more
connection churn.

uvloop does not help. The cost is in Python code, not in the loop's I/O
machinery. The SDK's documented remedy for high concurrency is its aiohttp
transport (`DefaultAioHttpClient`). **That scenario was not run:** it needs the
`httpx_aiohttp` package (the `anthropic[aiohttp]` extra), which is not
installed here. `aiohttp` itself is.

### Request sizes

At C=16, the classifier-like payload (6 KiB prompt, 4 KiB reply) costs almost
the same as a tiny one (4.0 vs 3.8 ms of CPU per request with threads). A
256 KiB prompt with a 64 KiB reply doubles it to ~7.6-8 ms. At C=256 the large
payload is where threads hurt most: 35 ms of CPU per request and 28 req/s,
against 19 ms and 52 req/s for asyncio.

### A small-C surprise: threads had lower median overhead

At C=16 and 1 s latency, the median overhead was 12 ms with threads and 26 ms
with asyncio. The difference is almost all on the request path (5.9 vs
19.2 ms), and the CPU per request is similar. This is consistent with how the
two schedule a burst of simultaneous requests. The event loop advances every
request by one short step per iteration, round-robin, so they all finish their
send work late together. The GIL hands a thread up to 5 ms at a time, about one
request's whole send path, so the earliest requests go out first. The total
work is the same; only its order differs.

## Memory and threads

| 1 s latency | threads: OS threads | threads: RSS growth | asyncio: OS threads | asyncio: RSS growth |
|---|---|---|---|---|
| C=16 | 17 | +1 MB | 2 | +1 MB |
| C=256 | 257 | +28 MB | 2 | +14 MB |
| C=1024 | 1025 | +88 MB | 2 | +44 MB |

The asyncio process is not thread-free. The SDK runs its one-time platform
detection through `asyncio.to_thread`, which starts one default-executor thread,
and against a hostname (the live API) `getaddrinfo` runs on that executor too.
Thread stacks are mostly virtual (8 MiB reserved, a few pages touched), which
is why 1024 threads cost tens of MB of RSS rather than 8 GiB.

## Cancellation

C=64 requests in flight plus 64 queued, 10 s simulated latency, cancellation
requested 0.5 s in. Times are measured from the cancel request (see the table
below).

- **asyncio** (`task.cancel()`, `asyncio.timeout`, or Ctrl-C through
  `asyncio.run`): the caller is back in 20-30 ms. The mock sees every connection
  close within 10-20 ms, cutting all 64 requests mid-flight and sending no
  replies. The process exits about 0.5 s after the cancel, which is ordinary
  interpreter teardown (the socket-shutdown variant below has the same exit time).
- **threads, `shutdown(wait=True, cancel_futures=True)`**: queued work is
  dropped, but the call blocks until the 64 in-flight requests complete (9.7 s).
- **threads, `shutdown(wait=False, ...)`**: control returns at once, but the
  interpreter joins the pool's worker threads at exit, so the process lives
  until the in-flight requests complete (10.3 s). All 64 replies are still
  sent.
- **threads + `client.close()`**: no change (exit 10.05 s). On Linux, closing a
  file descriptor does not wake a thread already blocked in `recv()` on it.
- **threads, Ctrl-C (the classifier's shape)**: `KeyboardInterrupt` reaches the
  main thread at once and the handler returns without waiting, but the process
  still exits only after 10.6 s, and the server still sent all 64 replies.
- **threads + `socket.shutdown(SHUT_RDWR)`** on every pooled socket: the
  blocked reads return at once (connections closed in 30 ms, exit 0.5 s). This
  is the one real interrupt for a blocked read, but the SDK does not expose it;
  the benchmark reaches through httpx/httpcore private attributes to do it.
- **threads + `os._exit()`**: 20 ms. The kernel closes the sockets, and no
  cleanup runs (no `finally`, no atexit, no flush).

"Replies the server still sent" counts work that completed after the user asked
to stop; on the real API that is generated and billed. The mock stops working on
a request when its connection closes. Whether the real API stops generating a
non-streaming request when the client disconnects is not measured here.

## Live API

**Tiny requests** ("Reply with the single word OK.", thinking off): about 1.1 s
per request up to C=32, the same in both models. At C=128, asyncio kept the
latency near 2.0 s (p95 2.5 s, 48.5 req/s). Threads fell to 26 req/s with a p95
of 7.5 s. That is the same client-side ceiling as in the mock, reached sooner
because TLS adds CPU per request. And with more than 100 connections open,
httpcore closes each connection as it goes idle (the keep-alive cleanup
above), so later requests pay for a fresh TLS handshake.

**The classification workload** (the project's exact request shape, 10 entries
per request, 160 entries): median latency 3.2-3.6 s per request at both C=4 and
C=16 in both models. That matches the classifier's own estimate of 0.9 s +
0.24 s per entry. Going from 4 to 16 in flight did not slow individual
requests, and cut the wall time from ~14 s to ~4 s. Every reply ended with
`end_turn` and passed the classifier's own validation. Output ran 314 tokens
median (max 395), well under the benchmark's 768 cap. Cost: $0.126 per 160
entries (~$0.0008 per entry).

**Rate limits**: no 429 in 768 requests. The lowest remaining request budget
seen in any response header was 99.2% of the limit (asyncio, 128 in flight, 256
requests in 5 s). Input, output, and combined token budgets never dropped below
99.9%.

## What this suggests for the classifier

The classifier sends at most 25 batches of 40 entries, 8 in flight, each taking
about 10 s. Every request's client overhead is milliseconds, so:

- **Switching to asyncio would not make it faster.** It is latency-bound, and
  both models deliver ideal throughput far past 8 in flight.
- **The wall-time lever is how many batches are in flight, not the mechanism.**
  25 batches at 8 workers take 4 rounds (~40 s). All 25 at once would be one
  round (~11 s), and this data shows no cost to that. Live, per-request
  latency did not grow from 4 to 16 in flight. A round of 25 requests costs
  about 125-250 ms of client CPU against a ~10 s round. And the whole history
  is a fraction of a percent of the observed per-minute request budget and
  about 1% of the input-token budget.
- **Cancellation is where asyncio would change behavior.** With threads,
  in-flight requests cannot be aborted. On Ctrl-C they run to completion (and
  are billed) whether or not anyone waits. And the interpreter will not exit
  until they finish or time out. With the classifier's client at this branch's
  base (42e6468: `max_retries=4`, the SDK's default 600 s timeout), that could
  take minutes on a hung connection. `main` has since turned SDK retries off,
  set a 240 s read timeout, and added a 560 s watchdog that saves finished
  results and exits with `os._exit`. A two-stage Ctrl-C (the first waits for
  in-flight requests and stores what was paid for, the second calls
  `os._exit`), which `main` also has now, is the right shape for threads. With
  asyncio, the "wait for what is already paid
  for" policy would still apply to the first Ctrl-C, since cancelling does not
  un-bill a request. But the second stage could cancel and close cleanly, so
  `finally` blocks and the SQLite commit path run, instead of `os._exit`
  skipping them. A deadline (`asyncio.timeout`) would also become a one-liner.
- **At hundreds in flight, the SDK's default transport is the bottleneck in
  either model** (~55-130 req/s on this laptop); the aiohttp transport would be
  the thing to evaluate then.

## In Asio terms

- **threads mode** is a thread pool where each thread does a *synchronous*
  `asio::write` then `asio::read` on its own socket. A blocked `read()` sits in
  the kernel, and only data arriving, or `shutdown()` on that socket from
  another thread, wakes it. That is what the cancellation table shows.
- **asyncio mode** is one `io_context` run by one thread, with C++20 coroutines
  (`co_await asio::async_read(sock, buf, asio::use_awaitable)`). A pending read
  is a registration with the reactor (epoll), not a parked thread, which is why
  1024 in-flight requests cost 2 OS threads instead of 1025.
- **Cancellation**: `task.cancel()` raises `CancelledError` at the suspended
  `await`, just as `socket.cancel()`, `close()`, or a per-operation cancellation
  slot completes a pending Asio operation with `asio::error::operation_aborted`.
  `asyncio.timeout(...)` corresponds to racing the operation against a
  `steady_timer` (`awaitable_operators`' `||`, or a timer that emits on a
  `cancellation_signal`). Both work for the same reason: the reactor owns the
  wait, so it can abandon it.
- **Hidden threads**: asyncio hands blocking calls (the SDK's platform check,
  `getaddrinfo`) to a small thread pool. Asio's `ip::tcp::resolver::async_resolve`
  does the same on POSIX, running `getaddrinfo` on a private internal thread.
- **Where the analogy stops**: in CPython both models run all request-handling
  code on one core (the GIL, or the single loop thread), so the ceiling is CPU
  per request. In C++, several threads can call `io_context::run()` on one
  context (with strands for ordering) and use every core. There the choice is
  mostly about memory per connection and cancellation, not a global lock. The
  httpcore finding is language-independent: a connection pool whose
  bookkeeping is O(connections x waiters) per operation will dominate any
  event loop, in C++ as much as in Python.

## Noise and outliers

The laptop ran a desktop session and had about 3 GiB swapped out when checked. In
12 of the 225 load cells (5%), throughput fell below half of its group's
median. Those cells were clustered in time (six of them in consecutive
low-concurrency cells of one 50 ms repeat) and spread across modes (9 asyncio
or uvloop, 3 threads; two-thirds of the 50 ms sweep's SDK cells are
asyncio-based). In every one, the client's CPU use collapsed (4-43% vs
normal) and the mock server's own event-loop lag rose about 10x (p99 7-15 ms,
normal 1 ms) at the same moment. Both processes were stalled together, which
points to the machine rather than either client or TCP. The kernel's listen-drop
and SYN-retransmit counters did not move across the later stalls. These cells
are the low minimums in the ranges below; medians are unaffected. One
`APIConnectionError` occurred in 2048 requests (threads, C=1024, jittered).

Outside the stalls the mock server stayed out of the way. It used at most 13%
of a core, and both its loop lag p99 and its reply-timer lateness p99 were
about 1 ms in the median cell (under 4 ms in 90% of cells). The raw-socket reference reached
3,000-3,500 req/s against it. The raw client's C=1024 cells spend their first
0.5-1.8 s opening 1024 connections, which is why they show 79% of ideal at
1 s latency.

## Result tables

### Concurrency sweep, fixed 1 s simulated latency

| C | N | mode | req/s med (min-max) | % of ideal | client overhead p50 / p95 / p99 ms | CPU ms/req | CPU % | peak RSS MB (+growth) | threads | fds | TCP conns | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | threads | 1.0 (0.9-1.0) | 99 | 5.5 / 7.2 / 7.4 | 5.08 (4.75-5.40) | 0 | 69 (+0) | 2 | 4 | 1 | 0 |
| 1 | 16 | asyncio | 1.0 (1.0-1.0) | 99 | 6.4 / 14.8 / 16.8 | 6.69 (6.59-7.94) | 1 | 69 (+0) | 3 | 7 | 1 | 0 |
| 1 | 16 | raw-asyncio | 1.0 (1.0-1.0) | 100 | 1.0 / 1.4 / 1.5 | 0.56 (0.50-0.65) | 0 | 63 (+0) | 1 | 7 | 1 | 0 |
| 4 | 16 | threads | 3.9 (3.5-4.0) | 99 | 11.3 / 23.7 / 26.3 | 5.17 (4.89-5.22) | 2 | 69 (+0) | 5 | 7 | 4 | 0 |
| 4 | 16 | asyncio | 3.9 (2.9-3.9) | 98 | 11.6 / 15.8 / 16.3 | 5.56 (4.61-19.18) | 2 | 69 (+0) | 2 | 10 | 4 | 0 |
| 4 | 16 | raw-asyncio | 4.0 (3.8-4.0) | 100 | 1.1 / 1.7 / 1.9 | 0.50 (0.45-0.52) | 0 | 63 (+0) | 1 | 10 | 4 | 0 |
| 8 | 32 | threads | 7.8 (6.8-7.9) | 98 | 12.8 / 29.7 / 35.5 | 5.03 (4.40-5.64) | 4 | 69 (+1) | 9 | 11 | 8 | 0 |
| 8 | 32 | asyncio | 7.8 (6.2-7.8) | 97 | 17.5 / 34.0 / 34.5 | 5.38 (4.72-15.00) | 4 | 69 (+1) | 2 | 14 | 8 | 0 |
| 8 | 32 | raw-asyncio | 8.0 (7.9-8.0) | 100 | 1.7 / 2.6 / 2.8 | 0.35 (0.29-0.38) | 0 | 63 (+0) | 1 | 14 | 8 | 0 |
| 16 | 64 | threads | 15.6 (15.4-15.6) | 97 | 12.4 / 26.5 / 39.5 | 4.44 (4.12-5.39) | 7 | 70 (+1) | 17 | 19 | 16 | 0 |
| 16 | 64 | asyncio | 15.3 (15.2-15.3) | 96 | 26.4 / 61.5 / 74.6 | 5.06 (4.85-5.62) | 8 | 70 (+1) | 2 | 22 | 16 | 0 |
| 16 | 64 | raw-asyncio | 15.9 (15.9-15.9) | 100 | 1.6 / 2.4 / 2.5 | 0.29 (0.26-0.31) | 0 | 63 (+0) | 1 | 22 | 16 | 0 |
| 64 | 256 | threads | 43.0 (29.9-54.8) | 67 | 66.5 / 420.9 / 1049.7 | 7.67 (6.15-23.81) | 34 | 74 (+6) | 65 | 67 | 64 | 0 |
| 64 | 256 | asyncio | 48.2 (47.4-48.6) | 75 | 58.1 / 889.5 / 1068.0 | 8.98 (8.47-9.73) | 43 | 72 (+4) | 2 | 70 | 64 | 0 |
| 64 | 256 | raw-asyncio | 63.2 (63.1-63.4) | 99 | 3.2 / 8.7 / 10.2 | 0.30 (0.20-0.31) | 2 | 64 (+1) | 1 | 70 | 64 | 0 |
| 256 | 1024 | threads | 43.3 (41.7-78.5) | 17 | 5157.3 / 7922.5 / 8458.1 | 21.27 (11.83-22.21) | 93 | 96 (+28) | 257 | 259 | 957 | 0 |
| 256 | 1024 | asyncio | 113.7 (5.9-123.7) | 44 | 983.3 / 1243.9 / 1310.8 | 6.66 (5.58-19.23) | 69 | 83 (+14) | 2 | 262 | 1024 | 0 |
| 256 | 1024 | raw-asyncio | 247.2 (243.5-248.4) | 97 | 6.9 / 13.0 / 15.6 | 0.20 (0.18-0.28) | 5 | 65 (+2) | 1 | 262 | 256 | 0 |
| 1024 | 2048 | threads | 82.9 (79.1-166.7) | 12 | 8163.8 / 13778.5 / 14638.7 | 12.50 (6.46-13.31) | 105 | 156 (+88) | 1025 | 491 | 2038 | 0 |
| 1024 | 2048 | asyncio | 104.4 (75.2-106.3) | 15 | 2176.2 / 16938.8 / 17240.1 | 9.03 (8.94-9.69) | 94 | 113 (+44) | 2 | 644 | 1923 | 0 |
| 1024 | 2048 | raw-asyncio | 812.3 (517.0-838.8) | 79 | 19.1 / 56.4 / 66.0 | 0.27 (0.24-0.56) | 22 | 71 (+8) | 1 | 1030 | 1024 | 0 |

### Concurrency sweep, 1 s median latency, log-normal jitter (sigma 0.5)

| C | N | mode | req/s med (min-max) | % of ideal | client overhead p50 / p95 / p99 ms | CPU ms/req | CPU % | peak RSS MB (+growth) | threads | fds | TCP conns | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | threads | 1.0 (1.0-1.0) | - | 5.6 / 8.1 / 9.0 | 5.74 (5.31-6.50) | 1 | 69 (+0) | 2 | 4 | 1 | 0 |
| 1 | 16 | asyncio | 1.0 (1.0-1.0) | - | 7.1 / 10.2 / 12.5 | 7.02 (6.82-7.34) | 1 | 69 (+0) | 2 | 7 | 1 | 0 |
| 4 | 16 | threads | 3.9 (3.9-3.9) | - | 5.2 / 7.9 / 8.9 | 4.85 (4.84-5.19) | 2 | 69 (+0) | 5 | 7 | 4 | 0 |
| 4 | 16 | asyncio | 3.9 (3.9-3.9) | - | 6.5 / 11.9 / 12.0 | 5.98 (5.44-6.42) | 2 | 69 (+0) | 2 | 10 | 4 | 0 |
| 8 | 32 | threads | 5.7 (5.7-5.7) | - | 5.3 / 8.1 / 9.1 | 4.71 (4.28-5.09) | 3 | 69 (+1) | 9 | 11 | 8 | 0 |
| 8 | 32 | asyncio | 5.7 (5.7-5.7) | - | 6.5 / 20.1 / 20.3 | 5.91 (5.18-6.17) | 3 | 70 (+1) | 2 | 14 | 8 | 0 |
| 16 | 64 | threads | 7.3 (7.3-7.3) | - | 4.7 / 9.8 / 13.3 | 4.35 (4.21-4.45) | 3 | 70 (+1) | 17 | 19 | 16 | 0 |
| 16 | 64 | asyncio | 7.3 (7.3-7.3) | - | 6.2 / 36.0 / 39.9 | 5.49 (5.13-5.88) | 4 | 70 (+1) | 2 | 22 | 16 | 0 |
| 64 | 256 | threads | 38.0 (38.0-38.1) | - | 5.5 / 14.8 / 22.9 | 4.44 (4.44-5.38) | 17 | 74 (+6) | 65 | 67 | 64 | 0 |
| 64 | 256 | asyncio | 37.6 (37.6-37.6) | - | 7.6 / 147.7 / 150.9 | 5.81 (5.47-6.25) | 22 | 72 (+3) | 2 | 70 | 64 | 0 |
| 256 | 1024 | threads | 112.4 (40.9-113.4) | - | 140.0 / 196.5 / 220.2 | 5.81 (5.67-8.52) | 64 | 92 (+24) | 257 | 258 | 1023 | 0 |
| 256 | 1024 | asyncio | 104.8 (7.8-104.9) | - | 285.8 / 650.9 / 753.2 | 5.78 (5.77-12.39) | 61 | 81 (+12) | 3 | 262 | 1024 | 0 |
| 1024 | 2048 | threads | 130.8 (120.6-133.1) | - | 3219.1 / 4395.7 / 4514.7 | 6.41 (6.31-6.83) | 84 | 150 (+82) | 1025 | 311 | 2048 | 1 APIConnectionError |
| 1024 | 2048 | asyncio | 121.9 (58.8-125.7) | - | 1543.6 / 10577.0 / 11103.8 | 6.37 (6.13-7.33) | 77 | 105 (+36) | 2 | 380 | 2048 | 0 |

### Concurrency sweep, fixed 50 ms simulated latency

| C | N | mode | req/s med (min-max) | % of ideal | client overhead p50 / p95 / p99 ms | CPU ms/req | CPU % | peak RSS MB (+growth) | threads | fds | TCP conns | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | threads | 18.1 (7.5-18.5) | 90 | 4.7 / 6.1 / 6.5 | 4.77 (3.74-5.04) | 7 | 69 (+0) | 2 | 4 | 1 | 0 |
| 1 | 16 | asyncio | 18.1 (17.7-18.3) | 91 | 4.6 / 5.4 / 5.7 | 4.54 (4.03-6.08) | 8 | 69 (+0) | 2 | 7 | 1 | 0 |
| 1 | 16 | asyncio-uvloop | 18.1 (18.1-18.3) | 91 | 4.7 / 5.6 / 6.1 | 4.68 (3.99-4.69) | 8 | 71 (+0) | 3 | 15 | 1 | 0 |
| 1 | 16 | raw-asyncio | 19.7 (19.6-19.7) | 98 | 0.6 / 0.8 / 0.8 | 0.40 (0.36-0.45) | 1 | 63 (+0) | 1 | 7 | 1 | 0 |
| 4 | 16 | threads | 65.5 (63.8-68.1) | 82 | 6.5 / 12.6 / 14.1 | 3.73 (3.70-5.01) | 25 | 69 (+0) | 5 | 7 | 4 | 0 |
| 4 | 16 | asyncio | 64.3 (64.0-64.5) | 80 | 9.0 / 13.4 / 14.3 | 4.22 (4.12-4.45) | 27 | 69 (+0) | 2 | 10 | 4 | 0 |
| 4 | 16 | asyncio-uvloop | 45.5 (22.0-64.3) | 57 | 27.5 / 51.5 / 56.3 | 6.50 (4.09-12.32) | 26 | 72 (+0) | 3 | 18 | 4 | 0 |
| 4 | 16 | raw-asyncio | 77.0 (50.7-77.2) | 96 | 0.9 / 1.3 / 1.4 | 0.38 (0.37-0.49) | 3 | 63 (+0) | 1 | 10 | 4 | 0 |
| 8 | 32 | threads | 121.2 (81.6-122.2) | 76 | 8.5 / 21.3 / 23.2 | 3.75 (3.75-5.25) | 45 | 69 (+1) | 9 | 11 | 8 | 0 |
| 8 | 32 | asyncio | 111.1 (9.9-115.0) | 69 | 14.4 / 28.5 / 29.8 | 4.15 (4.07-7.54) | 46 | 70 (+1) | 2 | 14 | 8 | 0 |
| 8 | 32 | asyncio-uvloop | 93.8 (16.2-112.1) | 59 | 19.5 / 58.3 / 61.5 | 5.19 (4.09-7.43) | 46 | 72 (+1) | 3 | 22 | 8 | 0 |
| 8 | 32 | raw-asyncio | 151.7 (98.6-153.1) | 95 | 1.2 / 1.6 / 2.0 | 0.32 (0.26-0.37) | 4 | 63 (+0) | 1 | 14 | 8 | 0 |
| 16 | 64 | threads | 183.8 (28.6-214.8) | 57 | 19.2 / 74.7 / 77.8 | 4.78 (3.81-6.07) | 82 | 70 (+1) | 17 | 19 | 16 | 0 |
| 16 | 64 | asyncio | 166.7 (4.8-175.0) | 52 | 35.4 / 67.0 / 74.8 | 4.85 (4.58-14.12) | 80 | 70 (+1) | 2 | 22 | 16 | 0 |
| 16 | 64 | asyncio-uvloop | 177.0 (156.7-178.6) | 55 | 31.2 / 52.3 / 68.0 | 4.42 (4.33-5.34) | 78 | 72 (+1) | 3 | 30 | 16 | 0 |
| 16 | 64 | raw-asyncio | 298.7 (295.7-301.1) | 93 | 1.0 / 2.3 / 2.6 | 0.26 (0.25-0.27) | 8 | 64 (+0) | 1 | 22 | 16 | 0 |
| 64 | 256 | threads | 56.4 (55.2-56.9) | 4 | 750.4 / 2887.5 / 3822.0 | 17.61 (14.72-18.21) | 100 | 74 (+5) | 65 | 25 | 22 | 0 |
| 64 | 256 | asyncio | 84.2 (84.0-86.4) | 7 | 466.7 / 1834.8 / 2354.1 | 11.57 (11.29-11.60) | 97 | 72 (+3) | 2 | 43 | 37 | 0 |
| 64 | 256 | asyncio-uvloop | 94.2 (62.5-96.6) | 7 | 409.7 / 1640.2 / 2216.9 | 10.27 (10.06-14.75) | 97 | 74 (+3) | 3 | 49 | 35 | 0 |
| 64 | 256 | raw-asyncio | 1050.1 (928.9-1050.8) | 82 | 2.7 / 5.2 / 6.2 | 0.22 (0.19-0.28) | 23 | 64 (+1) | 1 | 70 | 64 | 0 |
| 256 | 1024 | threads | 56.7 (49.7-61.1) | 1 | 3472.9 / 9250.4 / 12610.7 | 18.35 (16.76-19.86) | 102 | 91 (+23) | 257 | 103 | 289 | 0 |
| 256 | 1024 | asyncio | 90.6 (81.0-103.0) | 2 | 1801.9 / 6965.9 / 10146.2 | 10.80 (9.49-12.12) | 98 | 78 (+9) | 2 | 46 | 49 | 0 |
| 256 | 1024 | asyncio-uvloop | 93.2 (84.8-95.2) | 2 | 1738.3 / 7089.3 / 9798.9 | 10.48 (10.30-11.54) | 98 | 80 (+9) | 3 | 54 | 55 | 0 |
| 256 | 1024 | raw-asyncio | 3089.2 (2947.1-3222.7) | 60 | 6.0 / 16.6 / 22.1 | 0.18 (0.17-0.21) | 57 | 66 (+2) | 1 | 262 | 256 | 0 |
| 1024 | 2048 | threads | 62.7 (53.9-72.5) | 0 | 10863.8 / 24907.8 / 26117.7 | 16.32 (14.22-19.12) | 103 | 147 (+79) | 1025 | 103 | 1126 | 0 |
| 1024 | 2048 | asyncio | 96.2 (22.2-103.0) | 1 | 7532.8 / 19168.4 / 20260.0 | 10.20 (9.51-13.83) | 98 | 95 (+27) | 2 | 47 | 58 | 0 |
| 1024 | 2048 | asyncio-uvloop | 86.9 (22.0-106.9) | 1 | 7986.2 / 21595.5 / 22630.1 | 11.29 (9.16-12.12) | 98 | 98 (+27) | 3 | 61 | 105 | 0 |
| 1024 | 2048 | raw-asyncio | 3488.0 (3481.6-3610.8) | 17 | 37.0 / 52.1 / 55.4 | 0.25 (0.23-0.25) | 86 | 72 (+8) | 1 | 1030 | 1024 | 0 |

### Client overhead split, fixed 1 s latency

| C | mode | request path p50 / p99 ms | response path p50 / p99 ms |
|---|---|---|---|
| 1 | threads | 2.9 / 5.2 | 2.7 / 3.8 |
| 1 | asyncio | 3.3 / 9.2 | 2.8 / 8.2 |
| 1 | raw-asyncio | 0.3 / 0.4 | 0.7 / 1.4 |
| 16 | threads | 5.9 / 24.9 | 4.8 / 21.1 |
| 16 | asyncio | 19.2 / 49.1 | 4.4 / 31.2 |
| 16 | raw-asyncio | 0.1 / 0.8 | 1.4 / 2.3 |
| 256 | threads | 4198.8 / 6790.2 | 490.6 / 1183.4 |
| 256 | asyncio | 611.4 / 1213.2 | 272.8 / 660.7 |
| 256 | raw-asyncio | 0.5 / 8.3 | 4.9 / 15.4 |
| 1024 | threads | 6576.3 / 11594.8 | 1796.2 / 3334.0 |
| 1024 | asyncio | 1676.1 / 17153.7 | 572.0 / 1484.5 |
| 1024 | raw-asyncio | 0.5 / 21.6 | 19.0 / 65.6 |

### Request and reply sizes, fixed 200 ms latency

| size | prompt / reply | C | mode | req/s med (min-max) | client overhead p50 / p99 ms | CPU ms/req | errors |
|---|---|---|---|---|---|---|---|
| tiny | 32 B / 2 B | 16 | threads | 71.8 (71.4-72.5) | 8.9 / 30.7 | 3.79 (3.76-3.85) | 0 |
| tiny | 32 B / 2 B | 16 | asyncio | 66.3 (66.0-66.7) | 20.8 / 59.8 | 4.53 (4.46-4.53) | 0 |
| tiny | 32 B / 2 B | 256 | threads | 43.7 (42.8-44.4) | 5328.0 / 6862.5 | 23.43 (23.13-24.10) | 0 |
| tiny | 32 B / 2 B | 256 | asyncio | 51.9 (46.9-96.8) | 3436.2 / 18593.3 | 18.91 (10.07-20.88) | 0 |
| classifier-like | 6 KiB / 4 KiB | 16 | threads | 71.7 (68.8-72.4) | 9.0 / 26.4 | 4.01 (3.94-4.46) | 0 |
| classifier-like | 6 KiB / 4 KiB | 16 | asyncio | 66.0 (65.9-66.0) | 22.6 / 62.6 | 4.62 (4.61-4.69) | 0 |
| classifier-like | 6 KiB / 4 KiB | 256 | threads | 42.7 (36.0-46.8) | 5415.5 / 11315.4 | 24.16 (21.98-28.38) | 0 |
| classifier-like | 6 KiB / 4 KiB | 256 | asyncio | 68.9 (65.6-97.6) | 1259.0 / 13304.0 | 14.24 (10.03-14.85) | 0 |
| large | 256 KiB / 64 KiB | 16 | threads | 64.5 (64.3-65.7) | 16.1 / 49.3 | 7.56 (7.44-7.63) | 0 |
| large | 256 KiB / 64 KiB | 16 | asyncio | 55.6 (54.6-56.2) | 50.0 / 119.3 | 7.97 (7.96-8.43) | 0 |
| large | 256 KiB / 64 KiB | 256 | threads | 28.4 (28.2-29.2) | 6219.9 / 30083.4 | 35.23 (35.10-35.38) | 0 |
| large | 256 KiB / 64 KiB | 256 | asyncio | 51.7 (38.6-53.8) | 3270.6 / 16137.7 | 18.87 (18.12-25.33) | 0 |

### Mock server health

| scenario | cells | worst loop lag p99 ms | worst loop lag max ms | worst reply-timer lateness p99 ms | peak server CPU % |
|---|---|---|---|---|---|
| sweep-1s | 63 | 16.00 | 164.0 | 28.40 | 11 |
| sweep-1s-jitter | 42 | 12.00 | 65.0 | 23.63 | 10 |
| sweep-50ms | 84 | 16.00 | 82.0 | 19.84 | 13 |
| sizes | 36 | 2.00 | 8.0 | 3.61 | 12 |

### Cancellation (C=64 in flight + 64 queued, 10 s latency, cancel at 0.5 s; times from the cancel request)

| method | caller regains control (s) | last API connection closed (s) | process exited (s) | replies the server still sent | requests cut mid-flight | exit status |
|---|---|---|---|---|---|---|
| threads-shutdown-wait | 9.71 (9.69-9.82) | 9.79 (9.77-9.91) | 10.27 (10.25-10.40) | 64 | 0 | 0 |
| threads-shutdown-nowait | 0.00 (0.00-0.00) | 9.82 (9.80-9.94) | 10.35 (10.25-10.35) | 64 | 0 | 0 |
| threads-close-client | 0.00 (0.00-0.00) | 9.50 (9.50-9.50) | 10.05 (10.01-10.08) | 64 | 0 | 0 |
| threads-socket-shutdown | 0.03 (0.03-0.03) | 0.03 (0.03-0.03) | 0.51 (0.49-0.54) | 0 | 64 | 0 |
| threads-sigint | 0.00 (0.00-0.00) | 10.23 (10.20-10.27) | 10.57 (10.52-10.59) | 64 | 0 | -2 |
| threads-os-exit | 0.00 (0.00-0.00) | 0.02 (0.01-0.02) | 0.02 (0.02-0.03) | 0 | 64 | 130 |
| asyncio-cancel | 0.02 (0.02-0.02) | 0.01 (0.01-0.01) | 0.50 (0.45-0.53) | 0 | 64 | 0 |
| asyncio-timeout | 0.02 (0.02-0.03) | 0.01 (0.01-0.02) | 0.48 (0.46-0.52) | 0 | 64 | 0 |
| asyncio-sigint | 0.03 (0.02-0.03) | 0.02 (0.01-0.02) | 0.48 (0.47-0.51) | 0 | 64 | 0 |

### Live API, tiny requests

| C | N | mode | wall s | req/s | latency p50 / p95 / max s | 429s | not ok | lowest remaining: requests / input tok / output tok | cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | threads | 17.45 | 0.9 | 1.07 / 1.26 / 1.26 | 0 | 0 | 100.0% / 100.0% / 100.0% | $0.0012 |
| 1 | 16 | asyncio | 17.05 | 0.9 | 1.04 / 1.19 / 1.21 | 0 | 0 | 100.0% / 100.0% / 100.0% | $0.0012 |
| 8 | 16 | threads | 2.63 | 6.1 | 1.09 / 1.28 / 1.61 | 0 | 0 | 100.0% / 100.0% / 100.0% | $0.0012 |
| 8 | 16 | asyncio | 2.34 | 6.8 | 1.14 / 1.22 / 1.26 | 0 | 0 | 100.0% / 100.0% / 100.0% | $0.0012 |
| 32 | 64 | threads | 2.72 | 23.5 | 1.16 / 1.47 / 1.56 | 0 | 0 | 99.9% / 100.0% / 100.0% | $0.0046 |
| 32 | 64 | asyncio | 2.65 | 24.1 | 1.19 / 1.49 / 1.56 | 0 | 0 | 99.8% / 100.0% / 100.0% | $0.0046 |
| 128 | 256 | threads | 9.81 | 26.1 | 1.57 / 7.51 / 7.90 | 0 | 0 | 99.7% / 100.0% / 100.0% | $0.0184 |
| 128 | 256 | asyncio | 5.28 | 48.5 | 2.00 / 2.50 / 3.00 | 0 | 0 | 99.2% / 100.0% / 100.0% | $0.0184 |

### Live API, classification workload

| C | mode | wall s | latency p50 / p95 / max s | input tokens | output tokens | output tok/request p50 / max | cost | stop reasons | valid replies | 429s |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | threads | 13.73 | 3.33 / 3.89 / 4.10 | 36,675 | 5,220 | 314 / 377 | $0.1255 | end_turn: 16 | 16/16 | 0 |
| 4 | asyncio | 15.18 | 3.21 / 4.25 / 5.09 | 36,675 | 5,209 | 314 / 387 | $0.1254 | end_turn: 16 | 16/16 | 0 |
| 16 | threads | 4.33 | 3.55 / 4.15 / 4.31 | 36,675 | 5,260 | 314 / 395 | $0.1260 | end_turn: 16 | 16/16 | 0 |
| 16 | asyncio | 3.85 | 3.27 / 3.77 / 3.83 | 36,675 | 5,239 | 314 / 378 | $0.1257 | end_turn: 16 | 16/16 | 0 |
