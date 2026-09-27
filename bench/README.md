# Threads vs asyncio for concurrent Claude API requests

A benchmark comparing two ways to keep many Claude API requests in flight from Python:

- **threads**: the sync `anthropic.Anthropic` client shared by a
  `concurrent.futures.ThreadPoolExecutor(max_workers=C)`. This is what
  the classifier's threaded engine does (8 workers).
- **asyncio**: `anthropic.AsyncAnthropic` shared by N tasks in an `asyncio.TaskGroup`,
  with at most C of them past an `asyncio.Semaphore(C)` at a time.

It measures how the two differ across concurrency levels, request sizes, and
cancellation, and whether the live API's rate limits show up at these loads.
The numbers for one machine, and what they mean, are in [RESULTS.md](RESULTS.md).
Why blocking threads kept pace with the event loop up to a point, and why the
event loop pulled ahead past it, is explained in
[THREADS-VS-EVENT-LOOP.md](THREADS-VS-EVENT-LOOP.md).

Nothing here touches the project's classification store: the live classification
scenario sends the project's real request shape and throws the answers away.

## Running it

Linux only (the harness reads `/proc`). Python 3.12+ with `anthropic`; `uvloop`
is optional (used by the mock server when present, and for the `asyncio-uvloop` mode).

```sh
python3 bench/harness.py                          # all local scenarios, 3 repeats (about an hour on a 4-thread laptop)
python3 bench/harness.py --only cancel            # one scenario; the flag repeats
python3 bench/harness.py --quick --repeats 1      # smoke run: C in {1, 16, 256}, one repeat
python3 bench/harness.py --dry-run                # the live plan and its worst-case cost; no API call
python3 bench/harness.py --live                   # the live scenarios; spends at most $1.00
python3 bench/harness.py --report bench/out       # re-render the tables from saved results
```

Results go to `--out` (default `bench/out/`, git-ignored): one JSON file per
cell with every request's timestamps, plus `tables.md` with the summary tables.
The live results keep only the rate-limit response headers, never request or
organization ids.

## Files

| File | Role |
|---|---|
| `harness.py` | Command line: picks scenarios, writes the tables |
| `local.py` | Local scenarios: starts the mock server, runs each cell in a fresh worker process, samples it from `/proc` |
| `client_worker.py` | One cell: the threads / asyncio / asyncio-uvloop / raw-asyncio clients, and the cancellation variants |
| `mock_server.py` | Mock `POST /v1/messages` server (asyncio `Protocol` + loop timers, uvloop when available) |
| `live.py` | Live API scenarios: plans, preflight, runs, rate-limit header capture |
| `budget.py` | The shared spending cap every live request attempt goes through |
| `report.py` | Markdown tables from the result files |
| `common.py` | Shared helpers: percentiles, `/proc` readers, header names |

## Scenarios

### Local: concurrency sweeps (`sweep-1s`, `sweep-1s-jitter`, `sweep-50ms`)

Each client mode at C in {1, 4, 8, 16, 64, 256, 1024}, with N = max(16, 4C)
requests capped at 2048, against a mock that answers after a simulated model
latency: a fixed 1 s, a 1 s median with log-normal jitter (sigma 0.5), and a
fixed 50 ms (short enough that client-side overhead dominates). Besides the
two SDK modes, two reference modes run in some sweeps: `asyncio-uvloop` (the
asyncio mode on uvloop's libuv event loop) and `raw-asyncio` (no SDK: C
keep-alive connections written directly with asyncio streams, the floor the
event loop itself reaches, which also shows the mock server is not the limit).

Per cell (one fresh process each, after one unmeasured warm-up request):

- **req/s** over the measured window, and **% of ideal**: the rate the
  simulated latency alone allows (N requests in ceil(N/C) rounds; the SDK's
  default pool caps C at 1000 connections).
- **client overhead** per request: the caller's latency minus the time the mock
  held the request (`x-mock-send - x-mock-recv`, both on the system-wide
  monotonic clock). This is everything that is not the model: building the
  request, waiting for a connection, the GIL or the event loop, parsing.
  The overhead-split table divides it into the request path (caller start to
  the mock having the full request) and the response path (mock writing the
  reply to the caller holding a parsed `Message`).
- **CPU ms/req** and **CPU %**: the process's user+sys CPU over the window
  (`getrusage`), per request and as a share of one core.
- **peak RSS** (and growth over the pre-load RSS), **threads**, and **open
  fds**, sampled from `/proc/<pid>` by the harness every 10 ms, from outside
  the measured process.
- **TCP conns**: how many connections carried the cell's requests (from the mock).
- **errors** by exception type (`max_retries=0`, so nothing is retried away).

The mock server reports its own event-loop lag and reply-timer lateness for
every cell; the server-health table shows the worst of each.

### Local: request sizes (`sizes`)

C in {16, 256} at 200 ms latency with three payload shapes: tiny (32 B prompt,
2 B reply), classifier-like (6 KiB prompt, 4 KiB reply), and large (256 KiB
prompt, 64 KiB reply): how serialization and parsing cost scales with size in each mode.

### Local: cancellation (`cancel`)

C = 64 requests in flight (plus 64 queued) against a 10 s latency; cancellation
is requested 0.5 s in. Measured from the cancel request: when the caller
regains control, when the last API connection is closed (seen by the mock),
and when the process has exited; plus how many replies the mock still sent (on
the real API, work that is done and billed anyway) and how many requests were
cut mid-flight. Variants:

| Method | What it does |
|---|---|
| `threads-shutdown-wait` | `pool.shutdown(wait=True, cancel_futures=True)` |
| `threads-shutdown-nowait` | `pool.shutdown(wait=False, cancel_futures=True)`, then return from `main` (the interpreter joins the pool's threads at exit) |
| `threads-close-client` | the above plus `client.close()` from the main thread |
| `threads-socket-shutdown` | the above plus `shutdown(SHUT_RDWR)` on every pooled socket (reaches into httpcore internals; a demonstration) |
| `threads-sigint` | the classifier's shape: the caller waits in `as_completed`, the harness sends SIGINT, the handler cancels queued work without waiting and re-raises |
| `threads-os-exit` | `os._exit()` (what a second Ctrl-C can fall back to) |
| `asyncio-cancel` | `task.cancel()` on every task, then gather them |
| `asyncio-timeout` | the `TaskGroup` inside `asyncio.timeout_at(...)` |
| `asyncio-sigint` | the harness sends SIGINT; `asyncio.run` cancels the main task, the `TaskGroup` cancels its children |

### Live (`--live`)

Both need `ANTHROPIC_API_KEY` (or an `ant auth login` profile).

- **live-tiny**: "Reply with the single word OK." to `claude-sonnet-5` with
  thinking disabled and `max_tokens` 8, threads vs asyncio at C in
  {1, 8, 32, 128}, N = max(16, 2C). Records latency and the rate-limit headers
  of every response (via `.with_raw_response`) and counts any 429.
- **live-classify**: the project's own request (its `SYSTEM_PROMPT`, `SCHEMA`,
  `MODEL`, `EFFORT`, adaptive thinking, json_schema output) over a fixed subset
  of the revision history (the first 160 entries of the 1.38.2 page, read from
  the project's HTTP cache), 10 entries per request (16 requests), threads vs
  asyncio at C in {4, 16}. Reports latency, tokens, cost, stop reasons, and
  whether each reply passes the classifier's own validation. `max_tokens` is 768
  here instead of the classifier's 16000, which is what keeps the worst case
  inside the budget; a reply that hits it shows up as a `max_tokens` stop.

Which mode runs first alternates by concurrency level, and runs are 3 s apart.

## Live spending safety

`budget.py` holds one `Budget` for the whole live run: a hard cap of $1.00 and a
request cap equal to the planned request count.

- Before every attempt it reserves the attempt's worst case: its estimated input
  tokens at $2/MTok plus its full `max_tokens` at $10/MTok (Claude Sonnet 5
  prices). A reservation that would take spent plus reserved money past the cap,
  or the attempt count past the request cap, is refused and the request is skipped.
- After the attempt the reservation settles to the response's actual usage; to
  zero for an unbilled 4xx (including 429); and to the full worst case when the
  outcome is unknown (timeout, connection error, 5xx).
- The SDK's own retries are off everywhere (`max_retries=0`), so no attempt
  bypasses the budget. Requests have explicit timeouts (60 s tiny, 120 s
  classify), each scenario has a deadline (300 s, 600 s) after which no attempt
  starts, and every loop is over a fixed, planned list.
- The input estimate (bytes / 2.5, plus fixed overheads) is the only estimated
  term. Before a scenario spends anything, the harness checks every distinct
  request's estimate against the free token-counting endpoint and skips the
  scenario if any estimate is too low.
- A scenario whose total worst case does not fit the remaining budget is
  skipped whole, never run partially. `--dry-run` prints each plan and whether
  the sum of worst cases fits under the cap.

The spend plan is printed before each live scenario and the actual usage after
each one and in total.

## Limits of the method

- The SDK's `DefaultAioHttpClient` (the `anthropic[aiohttp]` extra) is not
  benchmarked: it needs the `httpx_aiohttp` package, which is not installed here.
- The mock runs on the same machine, so the client and server share CPUs; the
  server's own CPU and loop lag are reported for every cell to show how much it took.
- Loopback has no network latency or TLS; the live scenarios cover both.
