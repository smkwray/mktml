#!/usr/bin/env python3
"""Check for runtime skew when loading persisted model artifacts."""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Set

import joblib

# Ensure JSON output is not polluted when unpickling imports ml_engine.
os.environ.setdefault('MARKET_QUIET_IMPORTS', '1')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import CALIBRATION_ARTIFACT_DIR, MODEL_DIR, MODEL_MANIFEST_FILE, MODEL_THRESHOLDS_FILE  # noqa: E402


def _safe_version(pkg: str) -> Optional[str]:
    """Return installed package version or None if not available."""
    try:
        return version(pkg)
    except (PackageNotFoundError, Exception):
        return None


def _normalize_warning(warn: warnings.WarningMessage) -> Dict[str, str]:
    """Normalize a captured warning object for JSON serialization."""
    return {
        'category': warn.category.__name__,
        'message': str(warn.message).strip(),
    }


def _sha256_file(path: str) -> str:
    """Return SHA256 for a file or marker if unreadable/missing."""
    if not path or not os.path.exists(path):
        return 'missing'
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return 'unreadable'


def _contains_xgboost_object(root_obj: Any) -> bool:
    """Best-effort check for XGBoost classes nested in a loaded pickle."""
    stack: List[Any] = [root_obj]
    seen: Set[int] = set()

    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)

        try:
            module_name = getattr(cur.__class__, '__module__', '') or ''
            if module_name.startswith('xgboost'):
                return True
        except Exception:
            pass

        if isinstance(cur, dict):
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple, set)):
            stack.extend(list(cur))
            continue

        for attr in ('named_estimators_', 'named_estimators', 'estimators_', 'estimators', '_Booster', 'booster'):
            try:
                if hasattr(cur, attr):
                    stack.append(getattr(cur, attr))
            except Exception:
                continue
    return False


def _summarize_captured_warnings(captured: List[warnings.WarningMessage]) -> Dict[str, Any]:
    """Compress warnings emitted while loading one artifact."""
    normalized = [_normalize_warning(w) for w in captured]
    class_counts: Dict[str, int] = {}
    sig_counts: Dict[str, int] = {}
    for item in normalized:
        category = item.get('category', 'Unknown')
        message = item.get('message', '')
        class_counts[category] = class_counts.get(category, 0) + 1
        sig = f'{category}|{message}'
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    top_unique: List[Dict[str, Any]] = []
    for sig, count in sorted(sig_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        category, message = sig.split('|', 1)
        top_unique.append({
            'category': category,
            'message': message,
            'count': count,
        })
    return {
        'count': len(normalized),
        'classes': class_counts,
        'top_unique_messages': top_unique,
        # Internal aggregate pass still uses this field for global summary.
        'items': normalized,
    }


def _load_json_with_warnings(path: str) -> Dict[str, Any]:
    """Load JSON while capturing any emitted warnings."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        status = 'ok'
        error: Optional[str] = None
        try:
            with open(path, 'r') as f:
                json.load(f)
        except Exception as exc:
            status = 'error'
            error = f'{exc.__class__.__name__}: {exc}'
        return {
            'path': path,
            'kind': 'json',
            'status': status,
            'error': error,
            'warnings': _summarize_captured_warnings(captured),
        }


def _load_pickle_with_warnings(path: str) -> Dict[str, Any]:
    """Load a pickle artifact while capturing any emitted warnings."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        status = 'ok'
        error: Optional[str] = None
        payload: Any = None
        try:
            payload = joblib.load(path)
        except Exception as exc:
            status = 'error'
            error = f'{exc.__class__.__name__}: {exc}'
        contains_xgboost = False
        if status == 'ok':
            try:
                contains_xgboost = _contains_xgboost_object(payload)
            except Exception:
                contains_xgboost = False
        return {
            'path': path,
            'kind': 'pickle',
            'status': status,
            'error': error,
            'warnings': _summarize_captured_warnings(captured),
            'artifact_meta': {
                'contains_xgboost': bool(contains_xgboost),
                'sha256': _sha256_file(path),
            },
        }


def _discover_artifacts(model_dir: str) -> Dict[str, List[str]]:
    """Find model artifacts used for runtime skew checks."""
    model_dir = os.path.abspath(model_dir)

    pkl_files = sorted(set(glob.glob(os.path.join(model_dir, '**', '*.pkl'), recursive=True)))
    ubj_files = sorted(set(glob.glob(os.path.join(model_dir, '**', '*.ubj'), recursive=True)))
    xgb_json_files = sorted(set(
        p for p in glob.glob(os.path.join(model_dir, '**', '*.json'), recursive=True)
        if os.path.isfile(p)
        and os.path.basename(p).startswith(('xgb_', 'xgboost_', 'model_'))
        and 'calibration' not in p
        and os.path.basename(p) != 'model_thresholds.json'
    ))

    calibration_files = sorted(set(
        glob.glob(
            os.path.join(model_dir, '**', 'probability_calibration_*.json'),
            recursive=True,
        )
    ))
    if not calibration_files:
        calibration_files = sorted(
            [f for f in glob.glob(os.path.join(CALIBRATION_ARTIFACT_DIR, '*.json'))
             if os.path.isfile(f)]
        )

    thresholds_files = sorted(
        [f for f in glob.glob(os.path.join(model_dir, '**', 'model_thresholds.json'), recursive=True)
         if os.path.isfile(f)]
    )
    if MODEL_THRESHOLDS_FILE not in thresholds_files and os.path.isfile(MODEL_THRESHOLDS_FILE):
        thresholds_files.append(MODEL_THRESHOLDS_FILE)
        thresholds_files = sorted(set(thresholds_files))

    return {
        'pkl': pkl_files,
        'xgboost_stable_models': sorted(set(ubj_files + xgb_json_files)),
        'calibration_json': calibration_files,
        'thresholds_json': thresholds_files,
    }


def _artifact_relpath(path: str, model_dir: str) -> str:
    """Return normalized artifact path relative to model_dir."""
    rel = os.path.relpath(path, model_dir)
    return rel.replace('\\', '/')


def _required_artifact_paths(discovered: Dict[str, List[str]]) -> List[str]:
    """Flatten discovered artifact groups that should be tracked in manifest."""
    paths: List[str] = []
    for key in ('pkl', 'xgboost_stable_models', 'calibration_json', 'thresholds_json'):
        paths.extend(discovered.get(key, []) or [])
    return sorted(set(os.path.abspath(p) for p in paths if p))


def _load_manifest(path: str) -> Dict[str, Any]:
    """Load manifest JSON file if available."""
    if not path or not os.path.exists(path):
        return {'exists': False, 'path': os.path.abspath(path), 'error': 'missing'}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'exists': True, 'path': os.path.abspath(path), 'error': 'invalid_format', 'manifest': {}}
        return {'exists': True, 'path': os.path.abspath(path), 'error': None, 'manifest': data}
    except Exception as exc:
        return {'exists': True, 'path': os.path.abspath(path), 'error': f'{exc.__class__.__name__}: {exc}', 'manifest': {}}


def _validate_manifest(model_dir: str, discovered: Dict[str, List[str]]) -> Dict[str, Any]:
    """Validate manifest presence and file hash entries for discovered artifacts."""
    manifest_state = _load_manifest(os.path.join(model_dir, 'manifest.json'))
    violations: List[Dict[str, str]] = []

    if not manifest_state.get('exists'):
        violations.append({
            'code': 'manifest_missing',
            'path': str(manifest_state.get('path', '')),
            'detail': 'manifest.json not found',
        })
        return {
            'exists': False,
            'path': str(manifest_state.get('path', '')),
            'error': str(manifest_state.get('error', 'missing')),
            'tracked_count': 0,
            'required_count': len(_required_artifact_paths(discovered)),
            'violations': violations,
        }

    if manifest_state.get('error'):
        violations.append({
            'code': 'manifest_unreadable',
            'path': str(manifest_state.get('path', '')),
            'detail': str(manifest_state.get('error')),
        })
        return {
            'exists': True,
            'path': str(manifest_state.get('path', '')),
            'error': str(manifest_state.get('error')),
            'tracked_count': 0,
            'required_count': len(_required_artifact_paths(discovered)),
            'violations': violations,
        }

    manifest = manifest_state.get('manifest') or {}
    files_obj = manifest.get('files') if isinstance(manifest.get('files'), dict) else {}
    required_paths = _required_artifact_paths(discovered)
    for abs_path in required_paths:
        rel = _artifact_relpath(abs_path, model_dir)
        expected_hash = str(files_obj.get(rel, '') or '')
        if not expected_hash:
            violations.append({
                'code': 'manifest_entry_missing',
                'path': abs_path,
                'detail': f'missing files[{rel}] entry',
            })
            continue
        current_hash = _sha256_file(abs_path)
        if expected_hash != current_hash:
            violations.append({
                'code': 'manifest_hash_mismatch',
                'path': abs_path,
                'detail': f'manifest={expected_hash} runtime={current_hash}',
            })

    return {
        'exists': True,
        'path': str(manifest_state.get('path', '')),
        'error': None,
        'tracked_count': int(len(files_obj)),
        'required_count': int(len(required_paths)),
        'violations': violations,
    }


def _summarize_warnings(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact warning summary from artifact load checks."""
    flat: List[Dict[str, str]] = []
    for check in checks:
        warning_obj = check.get('warnings') or {}
        for warn in warning_obj.get('items', []):
            flat.append({
                'path': check.get('path', ''),
                'category': warn.get('category', ''),
                'message': warn.get('message', ''),
            })
    class_counts: Dict[str, int] = {}
    signature_counts: Dict[str, int] = {}
    signature_sample_path: Dict[str, str] = {}
    for item in flat:
        cls = item.get('category', 'Unknown')
        class_counts[cls] = class_counts.get(cls, 0) + 1
        sig = f"{cls}|{item.get('message', '')}"
        signature_counts[sig] = signature_counts.get(sig, 0) + 1
        signature_sample_path.setdefault(sig, item.get('path', ''))

    unique_messages: List[Dict[str, Any]] = []
    for sig, count in sorted(signature_counts.items(), key=lambda kv: kv[1], reverse=True):
        category, message = sig.split('|', 1)
        unique_messages.append({
            'category': category,
            'message': message,
            'count': count,
            'sample_path': signature_sample_path.get(sig, ''),
        })

    return {
        'count': len(flat),
        'classes': class_counts,
        'unique_count': len(unique_messages),
        'top_unique_messages': unique_messages[:20],
    }


def _compact_artifact_checks(checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop per-warning raw lists from artifact checks to keep JSON concise."""
    compact: List[Dict[str, Any]] = []
    for check in checks:
        entry = dict(check)
        warning_obj = entry.get('warnings') or {}
        entry['warnings'] = {
            'count': int(warning_obj.get('count', 0)),
            'classes': warning_obj.get('classes', {}),
            'top_unique_messages': warning_obj.get('top_unique_messages', []),
        }
        if 'artifact_meta' in entry and isinstance(entry.get('artifact_meta'), dict):
            entry['artifact_meta'] = dict(entry.get('artifact_meta') or {})
        compact.append(entry)
    return compact


def _extract_sklearn_version_mismatches(checks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract sklearn train/runtime version mismatch details from warnings."""
    mismatches: List[Dict[str, str]] = []
    pattern = re.compile(r"from version ([0-9A-Za-z._-]+) when using version ([0-9A-Za-z._-]+)")
    for check in checks:
        warning_obj = check.get('warnings') or {}
        for warn in warning_obj.get('items', []):
            category = str(warn.get('category', ''))
            if category != 'InconsistentVersionWarning':
                continue
            message = str(warn.get('message', ''))
            m = pattern.search(message)
            train_v = m.group(1) if m else 'unknown'
            runtime_v = m.group(2) if m else 'unknown'
            mismatches.append({
                'path': check.get('path', ''),
                'train_version': train_v,
                'runtime_version': runtime_v,
            })
    return mismatches


def _evaluate_policy(
    strict: bool,
    model_dir: str,
    checks: List[Dict[str, Any]],
    discovered: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Evaluate explicit artifact compatibility policy signals."""
    violations: List[Dict[str, str]] = []
    sklearn_mismatches = _extract_sklearn_version_mismatches(checks)

    for check in checks:
        path = str(check.get('path', ''))
        kind = str(check.get('kind', ''))
        status = str(check.get('status', 'ok'))
        if status != 'ok':
            violations.append({
                'code': 'artifact_load_error',
                'path': path,
                'detail': status,
            })

        artifact_meta = check.get('artifact_meta') if isinstance(check.get('artifact_meta'), dict) else {}
        if kind == 'pickle' and bool(artifact_meta.get('contains_xgboost')):
            violations.append({
                'code': 'xgboost_pickle_forbidden',
                'path': path,
                'detail': 'XGBoost object detected in pickle artifact',
            })

    for mismatch in sklearn_mismatches:
        violations.append({
            'code': 'sklearn_version_mismatch',
            'path': mismatch.get('path', ''),
            'detail': f"trained={mismatch.get('train_version')} runtime={mismatch.get('runtime_version')}",
        })

    manifest_state = _validate_manifest(model_dir=model_dir, discovered=discovered)
    violations.extend(list(manifest_state.get('violations', []) or []))

    # Deduplicate by (code, path, detail) to keep output compact.
    dedup_seen: Set[str] = set()
    dedup_violations: List[Dict[str, str]] = []
    for v in violations:
        key = f"{v.get('code')}|{v.get('path')}|{v.get('detail')}"
        if key in dedup_seen:
            continue
        dedup_seen.add(key)
        dedup_violations.append(v)

    policy = {
        'strict_mode': bool(strict),
        'passes': len(dedup_violations) == 0,
        'violation_count': len(dedup_violations),
        'violations': dedup_violations,
        'manifest': {
            'exists': bool(manifest_state.get('exists', False)),
            'path': manifest_state.get('path', ''),
            'error': manifest_state.get('error'),
            'tracked_count': int(manifest_state.get('tracked_count', 0)),
            'required_count': int(manifest_state.get('required_count', 0)),
        },
    }
    return policy


def _build_summary(
    model_dir: str,
    strict: bool,
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the JSON report used by CI and humans."""
    warning_summary = _summarize_warnings(checks)
    compact_checks = _compact_artifact_checks(checks)
    discovered = _discover_artifacts(model_dir)
    policy = _evaluate_policy(
        strict=strict,
        model_dir=model_dir,
        checks=checks,
        discovered=discovered,
    )
    return {
        'runtime': {
            'python': sys.version.split()[0],
            'scikit_learn': _safe_version('scikit-learn'),
            'xgboost': _safe_version('xgboost'),
        },
        'mode': 'strict' if strict else 'standard',
        'model_dir': model_dir,
        'manifest_path': os.path.abspath(MODEL_MANIFEST_FILE),
        'discovered_artifacts': discovered,
        'artifact_checks': compact_checks,
        'compatibility_warnings': warning_summary,
        'policy': policy,
        'strict_mode': strict,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check model artifact compatibility warnings under current runtime.',
    )
    parser.add_argument(
        '--model-dir',
        default=MODEL_DIR,
        help='Directory containing model artifacts.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Return non-zero if any compatibility warnings are found.',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Optional file path to write JSON summary (stdout still prints summary).',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_dir = os.path.abspath(args.model_dir)
    discovered = _discover_artifacts(model_dir)

    checks: List[Dict[str, Any]] = []
    for path in discovered['pkl']:
        checks.append(_load_pickle_with_warnings(path))
    for path in discovered['calibration_json']:
        checks.append(_load_json_with_warnings(path))
    for path in discovered['thresholds_json']:
        checks.append(_load_json_with_warnings(path))

    summary = _build_summary(model_dir=model_dir, strict=args.strict, checks=checks)
    summary_json = json.dumps(summary)
    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(summary_json)
            f.write('\n')
    print(summary_json)

    if args.strict and (
        summary['compatibility_warnings']['count'] > 0
        or not bool((summary.get('policy') or {}).get('passes', True))
    ):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
