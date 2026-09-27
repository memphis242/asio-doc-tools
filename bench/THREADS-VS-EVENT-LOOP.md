# Why blocking threads kept up with an event loop (until they didn't)

The benchmark in [RESULTS.md](RESULTS.md) found something that cuts against
the usual advice. A pool of threads, each blocked in a socket read, delivered
the same throughput as a single-threaded asyncio event loop, up to a point.
Past that point the event loop pulled ahead. This note explains why, with the
reasoning shown, so it can be reread later.

The short answer: the waiting itself was never the cost. The cost was the
Python code each request runs. That code runs on one core in both models: under
the GIL for threads, and on the loop's one thread for asyncio. While that core
has slack, how the waiting is represented barely matters. Once the core is
full, throughput is just `1 / (CPU per request)`. From then on the model that
spends less CPU per request wins, and under contention that is the event loop.

Every claim below is tagged:

- **[Measured]**: from the benchmark, or from the four micro-experiments E1-E4
  in the [appendix](#appendix-the-micro-experiments).
- **[Inferred]**: follows from measurements plus reading the code, but was not
  isolated by an experiment.
- **[Not measured]**: reasoning or outside knowledge only.

Machine: Intel i5-7200U (2 cores, 4 hardware threads), CPython 3.14.6 with the
GIL, Linux 7.1.5.

## 1. The conventional wisdom, and how much of it applies

The advice "use an event loop, not a thread per connection" comes mostly from
the **C10K problem** (Dan Kegel, 1999): serving 10,000 concurrent connections
on the hardware of the day. A thread per connection was expensive there for
four reasons:

1. Each thread reserves a stack and touches some of it.
2. The kernel scheduler has thousands of threads to manage.
3. Every wakeup is a context switch, which costs a few microseconds.
4. Switching stacks pollutes the caches and the TLB.

An event loop replaces all of that with one thread and an `epoll` set, where a
waiting connection is a kernel registration plus a small user-space object.

Python adds its own folk wisdom: "the GIL makes threads useless." That is true
for running Python code in parallel. It is false for waiting.

How much of this applied to this benchmark:

| Classic cost | Applies here? |
|---|---|
| Memory per thread | Yes, and it was measured: ~21 KB RSS per parked thread vs ~3 KB per pending task (E1), and +88 MB vs +44 MB at 1024 in flight in the benchmark. |
| Scheduler load, context switches | Not at 8-64 threads. It starts to show at hundreds of *runnable* threads (Section 4.2). |
| Cache/TLB effects | Not measured. |
| "GIL makes threads useless" | Wrong for waiting (E1 shows a blocked thread does not hold the GIL). Right that Python work does not run in parallel, but it does not in asyncio either. |

## 2. What was measured

**[Measured]** From RESULTS.md. C is the number of requests in flight; the
simulated model latency L is 1 s or 50 ms; "ideal" is C/L.

| Regime | 1 s latency | 50 ms latency |
|---|---|---|
| Even | C <= 16: both 96-99% of ideal | C <= 8: both 69-91% of ideal, within ~10% of each other |
| Diverging | C = 64: threads 43, asyncio 48 req/s | C = 16: threads 184, asyncio 167 req/s |
| Event loop ahead | C = 256: threads 43, asyncio 114 req/s | C >= 64: threads 56-63, asyncio 84-96 req/s |

Live, the project's own classification request took 3.2-3.6 s at both 4 and 16
in flight in both models. At C <= 16, threads even had slightly lower median
client overhead (12 ms vs 26 ms at C = 16, 1 s latency).

See RESULTS.md, sections *Where throughput stops scaling* and *A small-C
surprise*, and the sweep tables.

## 3. Why they were comparable

### 3.1 Little's law and the CPU budget

Little's law says requests in flight = throughput x time per request:

```
X = C / (L + o)          throughput, with o = client overhead per request
U = X * c                the client's CPU demand, with c = CPU per request
```

Both models must fit `U` into one core, because all their Python work is
serialized (Section 3.4). If `U` is far below 1 core, `o` stays small and `X`
is set by `L` alone. Neither model can do better than the latency allows.

**The classifier's own numbers.** 8 in flight, L ~ 10.5 s for a 40-entry
batch, c ~ 5 ms (**[Measured]**: 5.0-5.4 ms at C = 8 in the 1 s sweep):

```
X = 8 / 10.5 s        ~ 0.76 requests/s
U = 0.76/s * 5 ms     ~ 3.8 ms of CPU per second  =  0.4% of one core
```

The one resource both models share is idle 99.6% of the time. The concurrency
mechanism cannot matter at that point.

### 3.2 A blocked thread costs almost nothing while it waits

**[Measured]** (E1): 1000 threads each blocked in `sock.recv()` held the
following costs:

- **CPU:** 0.5 ms of CPU per second in total, all 1000 together.
- **Main thread speed:** unchanged. A CPU-bound pure-Python loop on the main
  thread took 310.6 ms with 1000 blocked threads vs 306.5 ms with none. So no
  blocked thread holds the GIL.

**[Not measured, but well documented]**: CPython releases the GIL around the
`recv()` system call (`Py_BEGIN_ALLOW_THREADS` in `socketmodule.c`). The thread
then sleeps on the socket's kernel wait queue. When data arrives, the kernel
marks it runnable. That is the same wait-queue wakeup that makes `epoll_wait`
return in an event loop. In both models the kernel does the waiting. They
differ only in what gets woken: a specific thread, or the loop, which then
finds the right task.

### 3.3 Waking up: the kernel is cheaper than Python-level machinery

**[Measured]** (E2): one round trip of one byte with an echo process.
"CPU" is this process only.

| How the waiter waits | CPU per round trip | Wall per round trip |
|---|---|---|
| Thread blocked in `recv()` | 14 us | 22 us |
| asyncio `loop.sock_recv()` | 78 us | 79 us |
| asyncio streams (transport + protocol, as httpx uses) | 99 us | 105 us |
| asyncio streams on uvloop | 42 us | 52 us |

With a single waiter, the blocking thread is 3-7x cheaper per wakeup. A
blocking `recv()` returns straight into Python. An asyncio wakeup runs Python
bookkeeping every time: a selector event becomes a callback handle, the
transport's `data_received` runs, a `Future` gets its result, and the task's
next step is scheduled before the coroutine resumes. `loop.sock_recv()` also
re-registers the socket with `epoll` on every call.

**[Measured]** (E2): the same picture shows up for pure in-process handoffs.
Two tasks ping-ponging through `asyncio.Event` cost 19 us of CPU per round
trip. Two threads doing the same through `threading.Event` cost 46 us, because
every handoff goes through the GIL and a condition variable. So a *task switch*
is cheaper than a *thread handoff*, but an *I/O wakeup* is cheaper for a
blocked thread. I/O wakeups are what an HTTP client mostly does.

**[Measured]** (E4): the loop's per-event cost falls under load. With 64-512
concurrent workers doing 1 ms of Python work each, asyncio's total CPU per
request was 1.01-1.04 ms. That leaves at most about 40 us for the loop's
mechanism (the calibrated 1 ms of work is itself approximate), far below the
99 us of the single-waiter case.

**[Inferred]**: under load, one `epoll_wait` returns many ready sockets, so
the loop's fixed cost per iteration is shared among them.

### 3.4 Neither model runs Python in parallel

With the GIL, one thread executes Python bytecode at a time. With asyncio, one
thread runs the loop. Either way, request-handling Python code is serialized
onto one core. The laptop's 4 hardware threads help only with work done outside
Python: the kernel's network stack, and C code that releases the GIL.

**[Measured]**: the threads client did exceed 100% CPU (up to 126% in E4 and
105% in the benchmark). **[Inferred]**: that extra is kernel and GIL-handoff
work (system CPU was 8-13% of the threads' time vs 4-6% for asyncio). It is not
parallel Python, because throughput did not rise with it.

So the two models differ in one thing: how a waiting request is represented.
For threads it is a parked kernel thread with its own stack: cheap to wake,
costly in memory. For asyncio it is an `epoll` registration plus Python
`Task`, `Future` and callback objects: cheap in memory, but every wakeup runs
Python. Below the core's limit, neither cost shows in the throughput.

### 3.5 Why threads were even slightly better at low concurrency

**[Measured]**:

- Benchmark, C = 16 at 1 s latency: median client overhead 12 ms for threads
  vs 26 ms for asyncio. Almost all of the gap is on the request path (5.9 vs
  19.2 ms).
- E4, at M <= 8 workers: overhead p50 0.6-0.7 ms for threads vs 0.9-1.1 ms
  for asyncio, and CPU per request 1.25 ms vs 1.52 ms at M = 8 with 1 ms of
  work.

**Interpretation** (not isolated by an experiment). Two effects fit:

1. **Cheaper wakeups.** E2 shows each blocked-thread wakeup costs less CPU
   than an asyncio wakeup.
2. **Scheduling order.** A burst of C requests is released at once. The event
   loop advances each one by a short step per iteration, round-robin, so they
   all finish their send work late together. This resembles processor sharing.
   The GIL instead hands a thread up to 5 ms at a time, which covers about one
   request's whole send path, so the first requests go out first (closer to
   FIFO). The total work is the same, but FIFO gives a lower median.

## 4. The crossover

### 4.1 When the core fills up

With `c` the CPU per request, there are two limits.

If the requests are spread out in time:

```
X ~ min( C / L ,  1 / c )
```

If they start together in rounds, as they did in the benchmark:

```
X ~ C / (L + C*c)    so    efficiency = X / (C/L) ~ 1 / (1 + C*c/L)
```

Either way the knee is at

```
C* ~ L / c
```

Past `C*`, throughput is `1/c`, and CPU per request is the whole game.

**Checked against the data [Measured, compared with the formula]**:

| Case | c used | Predicted | Measured |
|---|---|---|---|
| 1 s, C = 16 | ~5 ms | efficiency 93% | 96-97% |
| 1 s, C = 64 | ~9 ms | efficiency 64% | 67-75% |
| 50 ms, C = 4 | ~4 ms | efficiency 76% | 80-82% |
| 50 ms, C = 16 | ~5 ms | efficiency 39% | 52-57% |
| 50 ms, C >= 64, asyncio | 10-12 ms | X = 84-96 req/s (at 97-98% CPU) | 84-96 req/s |
| 50 ms, C >= 64, threads | 16-18 ms | X = 56-63 req/s (at 100-103% CPU) | 56-63 req/s |
| E4, L = 50 ms, 1 ms work, M = 512 | 1.02 ms asyncio, 2.4 ms threads | 980 and 420 req/s at 100% CPU | 947 (97% CPU) and 501 (118% CPU) req/s |

The round-start formula over-predicts the loss at 50 ms. Later rounds stop
being synchronized, which spreads the load. Past the knee, `1/c` predicts
throughput almost exactly.

**Rule of thumb**: measure `c` as CPU time divided by requests, at the
concurrency you intend to run, because `c` grows with C (Section 4.3). The
knee is at `C* ~ L / c`. For the classifier, L ~ 10 s and c ~ 5-15 ms, so
C* ~ 700-2000, far above its 8 or 25 in flight.

### 4.2 Why threads spend more CPU per request once the core is full

**[Measured]** (E3): wake latency through the GIL. A thread woken by I/O
cannot run Python until it gets the GIL.

| Other threads running Python | Switch interval | Wake latency p50 |
|---|---|---|
| 0 | 5 ms (default) | 0.11 ms |
| 1 | 5 ms | 5.1 ms |
| 1 | 0.5 ms | 0.65 ms |
| 1 | 20 ms | 20.1 ms |
| 3 | 5 ms | backlog: 413 ms and growing |

With even one CPU-busy thread, the woken thread waits a full switch interval.
It waits on a condition variable with that timeout, then asks the holder to
drop the GIL. With three busy threads it waits several intervals per message.
A thread that does I/O pays this again after *every* system call, because each
one releases and re-acquires the GIL. Messages then arrive faster than it gets
the GIL, and a backlog builds. This is the GIL "convoy effect" (David Beazley,
2010).

**[Measured]** (E4): once saturated, threads' CPU per request rises with the
number of threads: 1.6 ms at M = 64 to 2.4 ms at M = 512, for 1 ms of real
work. asyncio stays at 1.01-1.04 ms. In the benchmark at 50 ms latency and
C >= 64, threads used 16-18 ms per request vs asyncio's 10-12 ms.

**[Inferred]** why it rises:

- **Every system call becomes a contended handoff.** A saturated client always
  has a runnable thread wanting the GIL, so each release and re-acquire costs
  a condition-variable signal, a futex wakeup, a context switch, and possibly
  a wait of up to the switch interval. An HTTP request makes several such
  calls: connect, send, receive headers, receive body.
- **Hundreds of runnable threads** make the scheduler rotate through threads
  that immediately block again on the GIL.
- **httpcore's sync pool sits behind a lock**, adding handoffs (code reading).
- **More connection churn**: at 50 ms and C >= 256, threads opened 289-1126
  TCP connections vs 49-58 for asyncio (measured; the link to contention is
  inferred).

**[Not measured]**: cache and TLB effects of switching between hundreds of
thread stacks.

The asymmetry to remember: under load, the event loop gets cheaper per event
(batching, Section 3.3), while threads get more expensive per event
(contention). That opposite slope is the crossover.

### 4.3 Why CPU per request grows with concurrency in both models

**[Measured]** (RESULTS.md): profiling at C = 64 put about 70% of the loop's
time in httpcore's `_assign_requests_to_connections`, with about 8,600
`is_idle()` calls per request.

**[Inferred from the code]**: that function runs on every request start and
every response close. It scans every pooled connection for every queued
request. Its keep-alive cleanup compares the count of *all* connections with
`max_keepalive_connections` (100), so above 100 connections it closes every
connection that goes idle. **[Measured]**: at 1 s latency and C >= 256, nearly
every request opened a new TCP connection. Both models use this pool
algorithm, so both see `c` climb from ~5 ms to 10-20 ms.

## 5. "The event loop wins" really means "less Python per request wins"

**[Measured]** (RESULTS.md): the same asyncio loop driving raw sockets, with
no SDK, reached 3,000-3,500 req/s at 0.18-0.25 ms of CPU per request. The SDK
modes spent 5-20 ms per request.

**[Measured]** (E4): with a fixed 1 ms of Python work per request, asyncio's
own mechanism added at most about 40 us of CPU per request.

The event loop mechanism is cheap. What limited both SDK clients was the Python
code per request. Choosing asyncio over threads moved the ceiling by
1.5-2.6x. Cutting the Python per request moved it by 30-55x: at 50 ms latency
the raw client did 3,100-3,500 req/s against the SDK's 56-96.

## 6. Where the conventional wisdom does hold

- **Memory and thread count [Measured]**: at 1024 in flight, 1025 OS threads
  vs 2, and +88 MB vs +44 MB of RSS. Per parked waiter it was 21 KB vs 3 KB
  (E1). Each thread also reserves a full stack of address space: 16 MiB here
  (`ulimit -s` is 16384), so 1024 threads reserve 16 GiB of virtual memory.
  That is harmless on 64-bit, but it is why the classic advice exists. Waking
  1000 waiters at once took 67 ms for threads vs 28 ms for tasks (E1).
- **Cancellation [Measured]**: a thread blocked in `recv()` cannot be
  interrupted. `client.close()` from another thread did not wake it; only
  `shutdown(SHUT_RDWR)` on the socket did. A pending asyncio read is only a
  registration, so `task.cancel()` removed 64 in-flight requests and closed
  their connections in about 20 ms. With threads, the process could not exit
  for 10 s.
- **Deadlines and structured concurrency [Not measured as such]**:
  `asyncio.timeout()` and `TaskGroup` give a deadline over a whole group of
  requests, with guaranteed cleanup, in a couple of lines. With threads, every
  blocking call needs its own timeout, and nothing can stop a call from
  outside.

This is why the project's classifier is making asyncio its default engine. The
reason is cancellation and deadlines, not speed. The threaded engine is kept as
a reference under `src/asio_doc_tools/classify/threaded_reference/`.

## 7. In Asio and C++ terms

**[Not measured]**: what changes in C++.

**No GIL, and microseconds per request.** Parsing a small response in C++
costs microseconds, not milliseconds, and threads really do run in parallel. So
the knee `C* ~ L / c` moves out by orders of magnitude, and the one-core
ceiling mostly disappears. What remains are the classic C10K costs: memory per
thread, context switches and their cache effects, scheduler load at thousands
of threads, and cancellation.

**Three designs:**

1. **A thread per connection, with blocking `asio::read`.** This is the
   benchmark's threads model. Asio forbids concurrent calls on one socket
   object, and on Linux a `close()` from another thread would not wake the
   read anyway (the Python `client.close()` test showed exactly that). A
   shutdown of the native handle is the only reliable wake.
2. **One `io_context`, run by one thread, with C++20 coroutines.** This is the
   asyncio model, and your backend's design. A pending `async_read` is a
   reactor registration, and `socket.cancel()`, `close()`, or a cancellation
   slot completes it at once with `asio::error::operation_aborted`. Handlers
   need no locks. The E3 lesson carries over directly: a handler that computes
   for 5 ms delays every other ready handler by about that long (asyncio's
   wake latency was 7.9 ms p50 with 5 ms callbacks).
3. **N threads running `io_context::run()`**, on one context with strands, or
   one context per thread. This keeps the event-driven wait but uses every
   core. It is what a C++ server does once one core is not enough, and it has
   no Python equivalent without free-threading or multiple processes.

**For your backend**: a single-threaded `io_context` with coroutines over Unix
domain sockets, serving a GUI and perhaps a debug client, runs microseconds of
work per message, so it sits nowhere near its knee. What will matter:

- **Keep every handler short.** Your periodic flush to disk is the obvious
  candidate. Regular-file I/O on Linux does not go through `epoll`, so a
  synchronous write on the loop thread stalls every client for its duration.
  `asio::post` to a `thread_pool`, or Asio's io_uring-backed file support,
  keeps it off the loop.
- **Use cancellation slots or timers for deadlines.** This is the property that
  made the classifier switch.

## 8. What would change the conclusion

**[Not measured]**. Each of these would move the knee or change which model
wins:

- **Free-threaded CPython (3.14t).** Without the GIL, threads could run Python
  on all cores, so the threads model's ceiling could rise past asyncio's. That
  holds only if httpcore's pool lock and the SDK do not serialize the work
  first. asyncio stays on one core unless you run several loops.
- **Several loops or processes.** For example, one loop per process with the
  work sharded between them. This multiplies the `1/c` ceiling by the number
  of cores for either model.
- **The aiohttp transport** (`anthropic[aiohttp]`). Its HTTP parser is in C and
  its pool is simpler, so `c` would be lower and the knee higher. Per Section
  5, that lever is bigger than the choice between threads and asyncio.
- **HTTP/2 multiplexing.** Many requests would share one connection, removing
  the pool scan and the reconnect churn, at the cost of Python-level frame
  handling. It needs httpx's optional `h2` package, which is not installed here.
- **A real network.** Live, TLS added CPU per request. Threads at 128 in flight
  fell to 26 req/s vs 48.5 for asyncio: the same knee, reached earlier.

## Appendix: the micro-experiments

Each experiment was a short standalone script, run from a scratch directory
outside the repository on 2026-09-26, on the same laptop as the benchmark. The
laptop was lightly loaded (load average ~0.6). Each process was fresh, and the
results are medians over 3-5 runs.

**E1: the cost of waiting.** For N in {0, 1, 100, 1000}, in a fresh process:

- *threads*: N threads, each blocked in `sock.recv(1)` on its own
  `socketpair()`;
- *asyncio*: N tasks, each awaiting `loop.sock_recv(sock, 1)` on its own
  `socketpair()`.

The script let all waiters settle for 1 s, then measured:

- RSS growth per waiter, minus the cost of the socketpairs alone;
- `Threads:` from `/proc/self/status`;
- process CPU (`getrusage`) over 1 s of idle waiting;
- the best of 5 timings of a 3,000,000-iteration pure-Python loop on the main
  thread;
- the time to release all N waiters by writing one byte to each.

At N = 1000:

| | Threads | asyncio |
|---|---|---|
| RSS per waiter | 21.2 KB | 3.2 KB |
| OS threads | 1001 | 1 |
| Idle CPU | 0.51 ms/s | 0.33 ms/s |
| Main-thread loop | 310.6 ms | 308.4 ms |
| Release all | 67 ms | 28 ms |

At N = 100: 22.2 KB vs 8.5 KB per waiter, and release 6.8 ms vs 3.0 ms. The
main-thread loop took 306.5 ms with no waiters.

**E2: the cost of one wakeup.** 20,000 round trips per run; CPU is
`getrusage` for this process divided by round trips. Two kinds of test:

- *Echo modes*: a child process from `fork()` echoes one byte over a
  `socketpair()`. It is identical across modes, and its own CPU is not
  counted.
- *Handoff modes*: two threads, or two tasks, pass control back and forth
  through `threading.Event` / `asyncio.Event`, or ping-pong over a
  `socketpair()` inside one process.

| Mode | CPU per round trip | Wall per round trip |
|---|---|---|
| Echo: thread blocked in `recv()` | 14 us | 22 us |
| Echo: `loop.sock_recv` | 78 us | 79 us |
| Echo: asyncio streams | 99 us | 105 us |
| Echo: `sock_recv` on uvloop | 64 us | 66 us |
| Echo: streams on uvloop | 42 us | 52 us |
| Handoff: `threading.Event` | 46 us | 37 us |
| Handoff: `asyncio.Event` | 19 us | 20 us |
| Handoff: two threads over a socketpair | 30 us | 23 us |
| Handoff: two tasks over a socketpair | 139 us | 141 us |

**E3: wake latency under load.** A forked child writes `time.monotonic()` into
a pipe every 10 ms, 300 times. `CLOCK_MONOTONIC` is system-wide, so the parent
can subtract it from its own reading. The parent's waiter records the time from
the write to the moment it runs Python again.

- *Threads*: the waiter is a thread blocked in `os.read()`, while K other
  threads run a pure-Python loop, under `sys.setswitchinterval(S)`.
- *asyncio*: the waiter is a `loop.add_reader()` callback, on a loop that runs
  CPU-bound callbacks of L ms each, back to back.

Threads' results are in Section 4.2. With 8 spinning threads the backlog
reached a 3.2 s median. The asyncio results:

| Callback length | Wake latency p50 | Wake latency p99 |
|---|---|---|
| None (idle loop) | 0.16 ms | 0.29 ms |
| 1 ms | 1.6 ms | 2.1 ms |
| 5 ms | 7.9 ms | 10.0 ms |
| 20 ms | backlog: 1.5 s | - |

With 20 ms callbacks, one message is handled per 20 ms loop iteration, but
messages arrive every 10 ms, so a backlog builds.

**E4: the mechanism alone, at scale.** A delay server, running as an asyncio
Protocol on uvloop in a separate process, answers each `x\n` with `y\n` after
D = 50 ms. M workers each loop over the same steps: do W ms of calibrated
pure-Python work, send, and wait for the reply.

- *threads*: M threads, each with its own blocking socket;
- *asyncio*: M tasks, each with its own asyncio stream.

Each cell ran N = clamp(10M, 200, 6000) requests; CPU is the client's
`getrusage`. The model predicts `min(M / (D + W), 1000 / W)` req/s. Medians of
3 runs:

| W | M | Threads req/s | asyncio req/s | Threads CPU/req | asyncio CPU/req | Predicted |
|---|---|---|---|---|---|---|
| 1 ms | 8 | 152 | 151 | 1.25 ms | 1.52 ms | 157 |
| 1 ms | 32 | 566 | 556 | 1.10 ms | 1.15 ms | 627 |
| 1 ms | 64 | 645 | 884 | 1.61 ms | 1.04 ms | 1000 |
| 1 ms | 128 | 650 | 939 | 1.70 ms | 1.01 ms | 1000 |
| 1 ms | 512 | 501 | 947 | 2.38 ms | 1.02 ms | 1000 |
| 0.2 ms | 64 | 1122 | 1134 | 0.46 ms | 0.38 ms | 1275 |
| 0.2 ms | 128 | 1850 | 2119 | 0.59 ms | 0.33 ms | 2550 |
| 0.2 ms | 512 | 1494 | 2750 | 0.84 ms | 0.34 ms | 5000 |

The two models are even until `M * W` approaches `D + W`, the knee of
Section 4.1. Past it, asyncio holds near `1 / c`, while threads fall away as M
grows. This is the benchmark's shape, reproduced without the SDK.
