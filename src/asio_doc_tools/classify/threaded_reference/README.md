# The threaded reference engine

This directory holds the thread-pool engine that ran every classification until
commit `ec2b209`. It is kept runnable as a reference, to read and run side by side with
the default asyncio engine (`../asyncio_engine.py`). It is not the default, and it
exists to be compared with, not to be preferred.

Run it with:

```sh
asio-docs diff 1.38.1 1.38.2 --engine threads
```

or from Python, `classify.classify(items, engine="threads")`. Both engines produce the
same results and store them in the same store. They also print the same messages,
enforce the same budget, and end with the same exit codes. A diff run with one engine
is served from the store by the other.

## What the two engines share

Everything except how requests are scheduled, waited on, cancelled, and stopped comes
from the shared modules one directory up:

| Module | What it is |
|---|---|
| `prompt.py` | the prompt, schema, payloads, reply validation, and store keys |
| `batch.py` | one batch's requests, retries, split, and resend, as a generator that yields effects (`Call`, `Sleep`) and does no I/O itself |
| `budget.py` | the hard caps and the reservations every attempt makes; the "Run budget" comment has the arithmetic |
| `retry.py` | which failures are retried, which are refunded, and how long to wait |
| `store.py` | the SQLite store, the pending directory, and `Recorder`, which stores every batch's results |
| `run.py` | what every run does around its engine: planning, checks, and messages |

So the part worth comparing is `engine.py` here against `../asyncio_engine.py`: each
drives the same `batch.classify_batch` generator, one with a blocking
`client.messages.create` on a worker thread, the other with `await` on the event loop.

## How it differs from the asyncio engine

| | threaded reference (`engine.py`) | asyncio engine (`../asyncio_engine.py`) |
|---|---|---|
| Concurrency | a `ThreadPoolExecutor` of `MAX_IN_FLIGHT` (32) threads, each blocking in the sync `Anthropic` client | one event loop; a task per batch, at most `MAX_IN_FLIGHT` holding a slot (a `Semaphore`) |
| Storing results | worker threads return futures; the main thread harvests each one and stores it, since sqlite connections belong to one thread | each task stores its own results when they settle, on the loop thread |
| Starting | workers wait on a barrier until every batch is submitted, so a Ctrl-C mid-submit cannot orphan a request | nothing to guard: the tasks start on the loop, which is also where the Ctrl-C handler runs |
| First Ctrl-C | `KeyboardInterrupt` on the main thread; queued futures are cancelled, in-flight ones waited for and stored | a `loop.add_signal_handler` callback; tasks waiting for a slot are cancelled, in-flight ones waited for and stored |
| Second Ctrl-C | a thread blocked in a socket read cannot be interrupted, so it saves every finished result to the pending directory (with Ctrl-C ignored meanwhile), reports usage, and calls `os._exit(130)`, skipping every `finally` | cancels every task: each request's connection closes within milliseconds, finished results are stored normally, and the run raises `KeyboardInterrupt` (130) with every `finally` run |
| Hard limit (560 s) | the main thread's waits time out; it abandons the run as above and calls `os._exit(1)` | a `loop.call_at` timer cancels every task; the run raises `AsioDocsError` (status 1) |
| Requests given up on | keep running until the process exits: the server completes them and bills them | closed at once; counted as possibly billed, since cancelling does not un-bill what was generated |
| Threads | 32 workers plus the main thread | the main thread, plus the SDK's brief platform-detection thread |

Performance is not the difference. At this workload both are limited by the model's
latency (seconds per request) and spend milliseconds of client CPU per request, so they
take the same time. The asyncio engine is the default for its cancellation and its
deadlines, which `asyncio.timeout` or `loop.call_at` express in one line, while a thread
blocked in `recv()` has no public way to be interrupted.

The measurements behind this are in [`bench/RESULTS.md`](../../../../bench/RESULTS.md),
especially its "Cancellation" section. The design comparison is in
[`bench/THREADS-VS-EVENT-LOOP.md`](../../../../bench/THREADS-VS-EVENT-LOOP.md).

## Reading `engine.py`

- `_drive` runs a batch's generator on a worker thread: a `Call` blocks in
  `client.messages.create`, a `Sleep` blocks on the stop flag's `threading.Event`.
- `_Harvester` hands each finished future's outcome to the shared `Recorder`, exactly
  once, on the main thread.
- `_Waiting` is how the main thread waits: `await_batches` for a normal run,
  `stop_and_harvest` after a first Ctrl-C or an unexpected error, and `abandon` for a
  second Ctrl-C or the hard limit.
- `_sigint_ignored` swaps the SIGINT handler while `abandon` saves. Blocking SIGINT with
  `pthread_sigmask` would not work: the kernel delivers the signal to a worker thread
  instead, and Python still raises `KeyboardInterrupt` on the main thread.
- `_exit_process` is `os._exit`, replaceable in tests.
