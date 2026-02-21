#!/usr/bin/env python3
"""Build deterministic hash manifest for model/calibration artifacts."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import MODEL_DIR, MODEL_MANIFEST_FILE  # noqa: E402
from check_model_runtime_skew import _artifact_relpath, _discover_artifacts, _required_artifact_paths, _sha256_file  # noqa: E402


def _safe_version(pkg: str) -> str:
    try:
        return version(pkg)
    except (PackageNotFoundError, Exception):
        return 'missing'


def _build_manifest(model_dir: str) -> Dict[str, Any]:
    discovered = _discover_artifacts(model_dir)
    required_paths: List[str] = _required_artifact_paths(discovered)
    files: Dict[str, str] = {}
    for abs_path in required_paths:
        rel = _artifact_relpath(abs_path, model_dir)
        files[rel] = _sha256_file(abs_path)

    return {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model_dir': os.path.abspath(model_dir),
        'runtime': {
            'python': sys.version.split()[0],
            'scikit_learn': _safe_version('scikit-learn'),
            'xgboost': _safe_version('xgboost'),
            'numpy': _safe_version('numpy'),
            'pandas': _safe_version('pandas'),
            'duckdb': _safe_version('duckdb'),
        },
        'discovered_artifacts': discovered,
        'files': files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Build models/manifest.json with SHA-256 hashes.')
    parser.add_argument('--model-dir', default=MODEL_DIR, help='Model directory to scan.')
    parser.add_argument(
        '--output',
        default=MODEL_MANIFEST_FILE,
        help='Manifest output path (default: config MODEL_MANIFEST_FILE).',
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    out_path = os.path.abspath(args.output)
    manifest = _build_manifest(model_dir=model_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print(json.dumps({
        'status': 'ok',
        'output': out_path,
        'tracked_files': len((manifest.get('files') or {}).keys()),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
