"""Live Claude API scenarios: the same threads-vs-asyncio comparison against the real API.

Both scenarios run only with an explicit `--live`; `--dry-run` prints their
plans (request counts and worst-case cost) without constructing a client.
Every attempt goes through one shared `Budget` (hard cap $1.00), the SDK's own
retries are off (`max_retries=0`, so a 429 is observed and counted rather than
retried away), every request has an explicit timeout, and every scenario has a
deadline after which no new attempt starts.

  live-tiny       "Reply with the single word OK." at C in {1, 8, 32, 128},
                  N = max(16, 2C); thinking disabled, max_tokens 8
  live-classify   the project's classification request (SYSTEM_PROMPT, SCHEMA, MODEL, EFFORT,
                  adaptive thinking, json_schema output) on a fixed 160-entry subset of the
                  revision history, 10 entries per request (16 requests), at C in {4, 16};
                  max_tokens is 768 here (the classifier uses 16000) so the worst case is bounded

Responses are measured and thrown away: nothing is stored anywhere but the
benchmark's own result files, and only rate-limit headers are kept from them.
"""

import asyncio
import dataclasses
import inspect
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from budget import UNBILLED_STATUSES, Budget, BudgetRefused, Pricing, SONNET_5_PRICING
from common import SRC_DIR, BenchError, Summary, ratelimit_headers, write_json

MODEL: Final = "claude-sonnet-5"
CAP_USD: Final = 1.00

# Input-token estimate: an upper bound, checked against the free count_tokens endpoint
# before any live request is sent. Measured on this workload: 2.5-2.9 bytes per token for
# the history entries (JSON, C++ identifiers), about 4 for the English system prompt, and
# about 550 extra tokens when a json_schema output format is set.
_BYTES_PER_TOKEN_FLOOR: Final = 2.5
_MESSAGE_OVERHEAD_TOKENS: Final = 16
_STRUCTURED_OUTPUT_OVERHEAD_TOKENS: Final = 560

_TINY_PROMPT: Final = "Reply with the single word OK."
_TINY_MAX_TOKENS: Final = 8
_TINY_LEVELS: Final = (1, 8, 32, 128)
_CLASSIFY_MAX_TOKENS: Final = 768
_CLASSIFY_LEVELS: Final = (4, 16)
_CLASSIFY_ENTRIES: Final = 160
_CLASSIFY_BATCH: Final = 10
_HISTORY_VERSION: Final = "1.38.2"
_PAUSE_BETWEEN_RUNS_S: Final = 3.0


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    params: Mapping[str, Any]  # messages.create keyword arguments
    est_input_tokens: int
    batch_ids: tuple[str, ...] = ()  # classification ids, for validating the reply

    @property
    def max_tokens(self) -> int:
        return int(self.params["max_tokens"])

    def count_tokens_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if k != "max_tokens"}


@dataclass(frozen=True, slots=True)
class PlannedRun:
    mode: str  # "threads" or "asyncio"
    concurrency: int
    requests: tuple[PlannedRequest, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    runs: tuple[PlannedRun, ...]
    distinct_requests: tuple[PlannedRequest, ...]
    timeout_s: float
    deadline_s: float

    def request_count(self) -> int:
        return sum(len(run.requests) for run in self.runs)

    def est_input_tokens(self) -> int:
        return sum(r.est_input_tokens for run in self.runs for r in run.requests)

    def max_output_tokens(self) -> int:
        return sum(r.max_tokens for run in self.runs for r in run.requests)

    def worst_case_usd(self, pricing: Pricing) -> float:
        return sum(pricing.worst_case_usd(r.est_input_tokens, r.max_tokens) for run in self.runs for r in run.requests)


def estimate_input_tokens(params: Mapping[str, Any]) -> int:
    text = str(params.get("system", ""))
    for message in params["messages"]:
        content = message["content"]
        assert isinstance(content, str)
        text += content
    overhead = _MESSAGE_OVERHEAD_TOKENS
    output_format = params.get("output_config", {}).get("format")
    if output_format is not None:
        text += json.dumps(output_format.get("schema", {}))
        overhead += _STRUCTURED_OUTPUT_OVERHEAD_TOKENS
    return math.ceil(len(text.encode()) / _BYTES_PER_TOKEN_FLOOR) + overhead


def _alternating_runs(levels: Sequence[int], requests_for: Callable[[int], tuple[PlannedRequest, ...]]) -> tuple[PlannedRun, ...]:
    """Both modes at every level; which goes first alternates, so neither always gets the
    fresher rate-limit bucket."""
    runs = []
    for index, concurrency in enumerate(levels):
        modes = ("threads", "asyncio") if index % 2 == 0 else ("asyncio", "threads")
        runs.extend(PlannedRun(mode, concurrency, requests_for(concurrency)) for mode in modes)
    return tuple(runs)


def tiny_scenario() -> Scenario:
    params = {
        "model": MODEL,
        "max_tokens": _TINY_MAX_TOKENS,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": _TINY_PROMPT}],
    }
    request = PlannedRequest(params, estimate_input_tokens(params))
    runs = _alternating_runs(_TINY_LEVELS, lambda c: (request,) * max(16, 2 * c))
    return Scenario("live-tiny", runs, (request,), timeout_s=60.0, deadline_s=300.0)


def _project_modules() -> tuple[Any, Any, Any]:
    """The project's classify and history modules and Version, imported read-only from src/."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from asio_doc_tools import classify, history
    from asio_doc_tools.versions import Version

    return classify, history, Version


def classify_scenario() -> Scenario:
    classify, history, Version = _project_modules()

    # The history page for a pinned release comes from the project's HTTP cache (it never
    # expires there), so building this plan makes no network request once it is cached.
    releases = history.fetch_history(Version.parse(_HISTORY_VERSION))
    texts = [entry.full_text() for release in releases for entry in release.entries][:_CLASSIFY_ENTRIES]
    if len(texts) != _CLASSIFY_ENTRIES:
        raise BenchError(f"the revision history has only {len(texts)} entries; need {_CLASSIFY_ENTRIES}")
    batch_requests = []
    for start in range(0, _CLASSIFY_ENTRIES, _CLASSIFY_BATCH):
        batch = texts[start : start + _CLASSIFY_BATCH]
        ids = [f"e{i}" for i in range(len(batch))]
        payload = classify._build_payload(ids, ids, dict(zip(ids, batch, strict=True)))
        params = {
            "model": classify.MODEL,
            "max_tokens": _CLASSIFY_MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": classify.EFFORT, "format": {"type": "json_schema", "schema": classify.SCHEMA}},
            "system": classify.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": payload}],
        }
        batch_requests.append(PlannedRequest(params, estimate_input_tokens(params), tuple(ids)))
    requests = tuple(batch_requests)
    runs = _alternating_runs(_CLASSIFY_LEVELS, lambda _c: requests)
    return Scenario("live-classify", runs, requests, timeout_s=120.0, deadline_s=600.0)


def all_scenarios() -> tuple[Scenario, ...]:
    return (tiny_scenario(), classify_scenario())


# ---------------------------------------------------------------------------
# Plans


def print_plan(scenarios: Sequence[Scenario], pricing: Pricing) -> bool:
    """Prints each scenario's plan; True when the sum of worst cases fits under the cap."""
    total_worst = 0.0
    total_requests = 0
    print(f"Live plan (model {MODEL}, ${pricing.input_per_mtok:g}/MTok in, ${pricing.output_per_mtok:g}/MTok out, cap ${CAP_USD:.2f})")
    for scenario in scenarios:
        worst = scenario.worst_case_usd(pricing)
        total_worst += worst
        total_requests += scenario.request_count()
        runs = ", ".join(f"{r.mode} C={r.concurrency} N={len(r.requests)}" for r in scenario.runs)
        print(
            f"  {scenario.name}: {scenario.request_count()} requests; est. input {scenario.est_input_tokens():,} tokens, "
            f"max output {scenario.max_output_tokens():,} tokens; worst case ${worst:.4f}; "
            f"per-request timeout {scenario.timeout_s:g}s, scenario deadline {scenario.deadline_s:g}s"
        )
        print(f"    runs: {runs}")
    fits = total_worst <= CAP_USD
    print(
        f"  total: {total_requests} requests (the request cap); worst case ${total_worst:.4f} "
        f"{'fits under' if fits else 'EXCEEDS'} the ${CAP_USD:.2f} cap"
    )
    return fits


# ---------------------------------------------------------------------------
# One attempt


@dataclass(slots=True)
class LiveSample:
    outcome: str  # "ok", "http_<status>", an exception class name, or "skipped: <why>"
    t0: float = math.nan
    t1: float = math.nan
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    valid_reply: bool | None = None
    ratelimit: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "t0": self.t0,
            "t1": self.t1,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "stop_reason": self.stop_reason,
            "valid_reply": self.valid_reply,
            "ratelimit": self.ratelimit,
        }


def _reply_is_valid(message: Any, ids: tuple[str, ...]) -> bool | None:
    if not ids:
        return None
    classify, _, _ = _project_modules()
    return classify._validate_reply(message, list(ids)) is not None


def _admit(request: PlannedRequest, budget: Budget, deadline: float) -> tuple[Any, LiveSample | None]:
    if time.monotonic() >= deadline:
        return None, LiveSample("skipped: scenario deadline")
    try:
        return budget.reserve(request.est_input_tokens, request.max_tokens), None
    except BudgetRefused as e:
        return None, LiveSample(f"skipped: {e}")


def _settle_success(budget: Budget, reservation: Any, message: Any, raw: Any, t0: float, request: PlannedRequest) -> LiveSample:
    t1 = time.monotonic()
    cost = budget.settle_usage(reservation, message.usage)
    return LiveSample(
        "ok",
        t0,
        t1,
        message.usage.input_tokens,
        message.usage.output_tokens,
        cost,
        message.stop_reason,
        _reply_is_valid(message, request.batch_ids),
        ratelimit_headers(raw.headers),
    )


def _settle_status_error(budget: Budget, reservation: Any, error: Any, t0: float) -> LiveSample:
    if error.status_code in UNBILLED_STATUSES:
        budget.settle_unbilled(reservation)
    else:
        budget.settle_unknown(reservation)
    return LiveSample(f"http_{error.status_code}", t0, time.monotonic(), ratelimit=ratelimit_headers(error.response.headers))


def _attempt_sync(client: Any, request: PlannedRequest, budget: Budget, deadline: float) -> LiveSample:
    import anthropic

    reservation, skipped = _admit(request, budget, deadline)
    if skipped is not None:
        return skipped
    t0 = time.monotonic()
    try:
        raw = client.messages.with_raw_response.create(**request.params)
        message = raw.parse()
    except anthropic.APIStatusError as e:
        return _settle_status_error(budget, reservation, e, t0)
    except anthropic.APIConnectionError as e:  # includes APITimeoutError
        budget.settle_unknown(reservation)
        return LiveSample(type(e).__name__, t0, time.monotonic())
    except BaseException:
        budget.settle_unknown(reservation)
        raise
    return _settle_success(budget, reservation, message, raw, t0, request)


async def _attempt_async(client: Any, request: PlannedRequest, budget: Budget, deadline: float) -> LiveSample:
    import anthropic

    reservation, skipped = _admit(request, budget, deadline)
    if skipped is not None:
        return skipped
    t0 = time.monotonic()
    try:
        raw = await client.messages.with_raw_response.create(**request.params)
        message = raw.parse()
        if inspect.isawaitable(message):
            message = await message
    except anthropic.APIStatusError as e:
        return _settle_status_error(budget, reservation, e, t0)
    except anthropic.APIConnectionError as e:
        budget.settle_unknown(reservation)
        return LiveSample(type(e).__name__, t0, time.monotonic())
    except BaseException:
        budget.settle_unknown(reservation)
        raise
    return _settle_success(budget, reservation, message, raw, t0, request)


# ---------------------------------------------------------------------------
# Runs


def _run_threads(run: PlannedRun, budget: Budget, deadline: float, timeout_s: float) -> tuple[float, list[LiveSample]]:
    import anthropic

    client = anthropic.Anthropic(max_retries=0, timeout=timeout_s)
    try:
        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=run.concurrency) as pool:
            futures = [pool.submit(_attempt_sync, client, r, budget, deadline) for r in run.requests]
            samples = [f.result() for f in futures]
        return time.monotonic() - t_start, samples
    finally:
        client.close()


def _run_asyncio(run: PlannedRun, budget: Budget, deadline: float, timeout_s: float) -> tuple[float, list[LiveSample]]:
    import anthropic

    async def main() -> tuple[float, list[LiveSample]]:
        async with anthropic.AsyncAnthropic(max_retries=0, timeout=timeout_s) as client:
            gate = asyncio.Semaphore(run.concurrency)

            async def one(request: PlannedRequest) -> LiveSample:
                async with gate:
                    return await _attempt_async(client, request, budget, deadline)

            t_start = time.monotonic()
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(one(r)) for r in run.requests]
            return time.monotonic() - t_start, [t.result() for t in tasks]

    return asyncio.run(main())


def _preflight(scenario: Scenario) -> str | None:
    """Checks every distinct request's input estimate against count_tokens (free); None when all hold."""
    import anthropic

    # Token counting is not billed, so its (bounded) SDK retries cost nothing.
    client = anthropic.Anthropic(max_retries=2, timeout=30.0)
    try:
        worst_ratio = 0.0
        for request in scenario.distinct_requests:
            try:
                counted = client.messages.count_tokens(**request.count_tokens_params()).input_tokens
            except anthropic.APIError as e:
                return f"token counting failed ({type(e).__name__}: {e})"
            worst_ratio = max(worst_ratio, counted / request.est_input_tokens)
            if counted > request.est_input_tokens:
                return f"a request counts {counted} input tokens, above its estimate of {request.est_input_tokens}"
        print(f"  preflight: every input estimate holds (largest counted/estimate ratio {worst_ratio:.2f})")
        return None
    finally:
        client.close()


def _ratelimit_summary(samples: Sequence[LiveSample]) -> dict[str, Any]:
    """Per limit kind (requests, input-tokens, ...): the limit and the smallest remaining seen."""
    kinds: dict[str, dict[str, float]] = {}
    for sample in samples:
        for name, value in sample.ratelimit.items():
            if not name.startswith("anthropic-ratelimit-") or name.endswith("-reset"):
                continue
            kind, _, which = name.removeprefix("anthropic-ratelimit-").rpartition("-")
            try:
                number = float(value)
            except ValueError:
                continue
            entry = kinds.setdefault(kind, {"limit": 0.0, "min_remaining": math.inf})
            if which == "limit":
                entry["limit"] = max(entry["limit"], number)
            elif which == "remaining":
                entry["min_remaining"] = min(entry["min_remaining"], number)
    return {
        kind: {**entry, "min_headroom": entry["min_remaining"] / entry["limit"] if entry["limit"] else math.nan}
        for kind, entry in kinds.items()
    }


def _run_summary(run: PlannedRun, wall_s: float, samples: Sequence[LiveSample]) -> dict[str, Any]:
    ok = [s for s in samples if s.outcome == "ok"]
    latencies = [s.t1 - s.t0 for s in ok]
    return {
        "mode": run.mode,
        "concurrency": run.concurrency,
        "requests": len(run.requests),
        "wall_s": wall_s,
        "ok": len(ok),
        "outcomes": dict(Counter(s.outcome for s in samples)),
        "latency_s": Summary.of(latencies).to_json(),
        "input_tokens": sum(s.input_tokens for s in samples),
        "output_tokens": sum(s.output_tokens for s in samples),
        "output_tokens_per_request": Summary.of([float(s.output_tokens) for s in ok]).to_json(),
        "cost_usd": sum(s.cost_usd for s in samples),
        "stop_reasons": dict(Counter(s.stop_reason for s in ok)),
        "valid_replies": sum(1 for s in ok if s.valid_reply),
        "ratelimit": _ratelimit_summary(samples),
        "samples": [s.to_json() for s in samples],
    }


def run_live(out_dir: Path) -> None:
    scenarios = all_scenarios()
    pricing = SONNET_5_PRICING
    if not print_plan(scenarios, pricing):
        raise BenchError("the live plan's worst case exceeds the cap; not running it")
    budget = Budget(CAP_USD, sum(s.request_count() for s in scenarios), pricing)
    runners = {"threads": _run_threads, "asyncio": _run_asyncio}
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        worst = scenario.worst_case_usd(pricing)
        print(f"\n== {scenario.name}: {scenario.request_count()} requests, worst case ${worst:.4f}; "
              f"budget remaining ${budget.remaining_usd():.4f}, {budget.remaining_requests()} requests")
        if worst > budget.remaining_usd() or scenario.request_count() > budget.remaining_requests():
            print("  SKIPPED: its worst case does not fit the remaining budget")
            continue
        problem = _preflight(scenario)
        if problem is not None:
            print(f"  SKIPPED: {problem}")
            continue
        before = budget.snapshot()
        deadline = time.monotonic() + scenario.deadline_s
        run_summaries = []
        for index, run in enumerate(scenario.runs):
            if index:
                time.sleep(_PAUSE_BETWEEN_RUNS_S)
            wall_s, samples = runners[run.mode](run, budget, deadline, scenario.timeout_s)
            summary = _run_summary(run, wall_s, samples)
            run_summaries.append(summary)
            latency = summary["latency_s"]
            print(
                f"  {run.mode:>8} C={run.concurrency:<4} N={len(run.requests):<4} wall {wall_s:6.2f}s  "
                f"ok {summary['ok']:>4}  p50 {latency['p50']:.2f}s p95 {latency['p95']:.2f}s  "
                f"tokens {summary['input_tokens']}/{summary['output_tokens']}  ${summary['cost_usd']:.4f}  "
                f"outcomes {summary['outcomes']}",
                flush=True,
            )
        after = budget.snapshot()
        print(
            f"  {scenario.name} spent ${after.spent_usd - before.spent_usd:.4f} "
            f"(of which ${after.unknown_outcome_usd - before.unknown_outcome_usd:.4f} counted at worst case "
            f"for unknown outcomes); total ${after.spent_usd:.4f} of ${CAP_USD:.2f}"
        )
        write_json(
            out_dir / f"{scenario.name}.json",
            {
                "scenario": scenario.name,
                "model": MODEL,
                "worst_case_usd": worst,
                "spent_usd": after.spent_usd - before.spent_usd,
                "runs": run_summaries,
            },
        )

    final = budget.snapshot()
    print(
        f"\nlive total: {final.attempts} attempts, ${final.spent_usd:.4f} spent of ${final.cap_usd:.2f} "
        f"(${final.unknown_outcome_usd:.4f} at worst case for unknown outcomes), "
        f"{final.estimate_violations} estimate violations"
    )
    write_json(out_dir / "budget.json", dataclasses.asdict(final))
