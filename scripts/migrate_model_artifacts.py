#!/usr/bin/env python3
"""Migrate legacy model pickles to stable ensemble artifacts."""
import argparse
import json
import os
import sys
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# Ensure JSON output is not polluted by import-time logs.
os.environ.setdefault('MARKET_QUIET_IMPORTS', '1')

from config import MODEL_PATH_5D, MODEL_PATH_10D, MODEL_PATH_30D  # noqa: E402
from ml_engine import migrate_model_pickle_to_stable  # noqa: E402


def _parse_paths(raw_paths: str) -> List[str]:
    if not raw_paths:
        return [MODEL_PATH_5D, MODEL_PATH_10D, MODEL_PATH_30D]
    paths: List[str] = []
    for token in str(raw_paths).split(','):
        token = token.strip()
        if token:
            paths.append(os.path.abspath(token))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Migrate model_5d/10d/30d pickles to stable wrapper + XGBoost .ubj artifacts.'
    )
    parser.add_argument(
        '--paths',
        default='',
        help='Optional comma-separated model pickle paths. Defaults to config horizon model paths.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Return non-zero when any artifact is missing or fails to load during migration.',
    )
    args = parser.parse_args()

    paths = _parse_paths(args.paths)
    results = [migrate_model_pickle_to_stable(p) for p in paths]
    summary = {
        'paths': paths,
        'results': results,
        'migrated_count': sum(1 for r in results if r.get('status') == 'migrated'),
        'already_stable_count': sum(1 for r in results if r.get('status') == 'already_stable'),
        'load_failed_count': sum(1 for r in results if r.get('status') == 'load_failed'),
        'missing_count': sum(1 for r in results if r.get('status') == 'missing'),
    }
    summary['passes'] = bool(summary['load_failed_count'] == 0 and summary['missing_count'] == 0)
    print(json.dumps(summary, indent=2))
    if args.strict and not bool(summary['passes']):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
