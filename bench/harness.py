"""Benchmark: sync Anthropic client on a thread pool vs AsyncAnthropic on asyncio.

  python3 bench/harness.py                     all local scenarios (mock API; free)
  python3 bench/harness.py --only cancel       one local scenario (repeatable flag)
  python3 bench/harness.py --quick --repeats 1 a fast smoke run of the local scenarios
  python3 bench/harness.py --dry-run           print the live plan and its worst-case cost
  python3 bench/harness.py --live              run the live API scenarios (hard cap $1.00)
  python3 bench/harness.py --report DIR        re-render the tables from a results directory

Local scenarios: sweep-1s, sweep-1s-jitter, sweep-50ms, sizes, cancel. Results
(raw per-cell JSON) go under --out, and the Markdown tables to <out>/tables.md.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

from common import BENCH_DIR, BenchError

LOCAL_SCENARIOS: Final = ("sweep-1s", "sweep-1s-jitter", "sweep-50ms", "sizes", "cancel")


def _write_tables(out_dir: Path) -> None:
    import report

    tables = report.render(out_dir)
    (out_dir / "tables.md").write_text(tables)
    print(f"\n{tables}\nTables written to {out_dir / 'tables.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "out", help="results directory (default: bench/out)")
    parser.add_argument("--only", action="append", choices=LOCAL_SCENARIOS, help="run just this local scenario")
    parser.add_argument("--repeats", type=int, default=3, help="repeats per local cell (default 3)")
    parser.add_argument("--quick", action="store_true", help="local sweeps at C in {1, 16, 256} only")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="run the live API scenarios (spends money, capped at $1.00)")
    group.add_argument("--dry-run", action="store_true", help="print the live plan without calling the API")
    group.add_argument("--report", type=Path, metavar="DIR", help="only re-render the tables from DIR")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    try:
        if args.report is not None:
            _write_tables(args.report)
        elif args.dry_run:
            import live
            from budget import SONNET_5_PRICING

            fits = live.print_plan(live.all_scenarios(), SONNET_5_PRICING)
            return 0 if fits else 1
        elif args.live:
            import live

            live.run_live(args.out / "live")
            _write_tables(args.out)
        else:
            import local

            local.run_local(args.out / "local", args.only or LOCAL_SCENARIOS, args.repeats, args.quick)
            _write_tables(args.out)
    except BenchError as e:
        print(f"harness: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nharness: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
