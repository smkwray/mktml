#!/usr/bin/env python3
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from scanner import run_replay_reproducibility_check


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run deterministic replay reproducibility checks.',
    )
    parser.add_argument('--start', required=True, help='Replay start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='Replay end date (YYYY-MM-DD)')
    parser.add_argument(
        '--runs',
        type=int,
        default=2,
        help='Number of replay runs to compare (default: 2)',
    )
    parser.add_argument(
        '--replay-limit',
        type=int,
        default=None,
        help='Optional ticker limit for replay scan',
    )
    parser.add_argument(
        '--flush-rows',
        type=int,
        default=5000,
        help='Replay upsert flush batch size',
    )
    parser.add_argument(
        '--report',
        default=None,
        help='Optional explicit markdown report path under reports/.',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_replay_reproducibility_check(
        start_date=args.start,
        end_date=args.end,
        replay_runs=args.runs,
        max_tickers=args.replay_limit,
        flush_rows=args.flush_rows,
        report_path=args.report,
    )

    print(
        "Replay validation result: "
        f"status={result.get('status')} rows_compared={result.get('rows_compared')} "
        f"mismatches={result.get('mismatch_count')} report={result.get('report_path')}"
    )
    if result.get('status') != 'PASS':
        print("Replay validation failed.")
        return 1
    print("Replay validation passed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
